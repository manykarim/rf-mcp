"""Point agents to the skills that matter for their project: `project_skills` on the first result, links on hints.

Registered after `ClientRoots`, so a server-mode call's `root` is already filled in. The project's packages are
listed in a worker thread while the tool runs, once per session and root (design D3/D4 of
library-aware-skill-retrieval).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import anyio
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult

from robotmcp.config import ServerConfig
from robotmcp.runner.environment import EnvironmentResolver
from robotmcp.skills.catalog import catalog_entries, match
from robotmcp.skills.provider import resolve_skill_roots, served_skills
from robotmcp.telemetry import record_skill_pointed

logger = logging.getLogger(__name__)

MAX_PROJECT_SKILLS = 3
_PROCESS = "process"  # sessions of the stateless (2026-07-28) protocol cannot be told apart: one key per server


def _session_key(context: MiddlewareContext) -> str:
    ctx = context.fastmcp_context
    if ctx is None:
        return _PROCESS
    try:
        modern = getattr(ctx, "_is_modern_protocol", None)
        if callable(modern) and modern():
            return _PROCESS
        return str(ctx.session_id or _PROCESS)
    except Exception:
        return _PROCESS


def resolve_skill_uri(uri: str, served: dict[str, Path]) -> bool:
    """True when `skill://<name>/<path>` names a served skill and, for a reference, an existing file."""
    if not uri.startswith("skill://"):
        return False
    name, _, path = uri[len("skill://"):].partition("/")
    folder = served.get(name)
    if folder is None:
        return False
    if path in ("", "SKILL.md"):
        return True
    target = (folder / path).resolve()
    return target.is_file() and folder.resolve() in target.parents


class SkillPointers(Middleware):
    """Add `project_skills` to the first `rf_*` result per session and root, and resolve hint skill links."""

    def __init__(self, config: ServerConfig, environments: EnvironmentResolver) -> None:
        self._config = config
        self._environments = environments
        self._seen: set[tuple[str, Path]] = set()

    def _project_skills(self, root: Path) -> list[dict[str, Any]]:
        try:
            inventory = self._environments.installed_distributions(root)
            if not inventory:
                return []
            served = served_skills(resolve_skill_roots(self._config, project_roots=[root]))
            matches = match(inventory, catalog_entries(self._config), served)
            return [m.to_dict() for m in matches[:MAX_PROJECT_SKILLS]]
        except Exception:
            logger.exception("Matching skills for %s failed", root)
            return []

    async def on_call_tool(self, context: MiddlewareContext, call_next) -> Any:
        name = getattr(context.message, "name", "") or ""
        args = getattr(context.message, "arguments", None) or {}
        if not name.startswith("rf_"):
            return await call_next(context)
        try:
            root = self._config.resolve_root(args.get("root"))
            mode = self._config.resolved_mode(args.get("target"))
        except Exception:
            return await call_next(context)

        key = (_session_key(context), root)
        first = self._config.skill_pointers and mode == "server" and key not in self._seen
        if first:
            self._seen.add(key)
        found: list[dict[str, Any]] = []

        async def collect() -> None:
            found.extend(await anyio.to_thread.run_sync(self._project_skills, root))

        if not first:
            result = await call_next(context)
        else:
            error: BaseException | None = None
            async with anyio.create_task_group() as tg:
                tg.start_soon(collect)
                try:  # caught here: an error raised in a task group's body would come out as an ExceptionGroup
                    result = await call_next(context)
                except BaseException as exc:
                    error = exc
                    tg.cancel_scope.cancel()
            if error is not None or getattr(result, "is_error", False):
                self._seen.discard(key)  # a failed call does not use up the first result
            if error is not None:
                raise error

        try:
            return self._rewrite(result, root, found)
        except Exception:
            logger.exception("Adding skill pointers to %s failed", name)
            return result

    def _rewrite(self, result: Any, root: Path, project_skills: list[dict[str, Any]]) -> Any:
        if not isinstance(result, ToolResult) or result.is_error or not isinstance(result.structured_content, dict):
            return result
        data = dict(result.structured_content)
        changed = False
        hints = data.get("hints")
        if isinstance(hints, list) and any(isinstance(h, dict) and "skills" in h for h in hints):
            served = (
                served_skills(resolve_skill_roots(self._config, project_roots=[root]))
                if self._config.skill_pointers else {}
            )
            data["hints"] = [self._resolve_hint(h, served) if isinstance(h, dict) else h for h in hints]
            changed = True
        if project_skills:
            data["project_skills"] = project_skills
            for entry in project_skills:
                record_skill_pointed(entry["skill"].split("/")[2], "project_skills")
            changed = True
        if not changed:
            return result
        return ToolResult(structured_content=data, meta=result.meta)

    @staticmethod
    def _resolve_hint(hint: dict[str, Any], served: dict[str, Path]) -> dict[str, Any]:
        hint = dict(hint)
        candidates = hint.pop("skills", None) or []
        for uri in candidates:
            if resolve_skill_uri(str(uri), served):
                hint["skill"] = uri
                record_skill_pointed(str(uri)[len("skill://"):].split("/", 1)[0], "hint")
                break
        return hint

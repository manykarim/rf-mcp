"""Register / unregister the rf-mcp MCP server into coding-agent config files.

Merge-in-place and never overwrite: an agent's other MCP servers are preserved.
Every write is recorded in the hash-tracked manifest so uninstall is reversible
and safe. ``--what`` other than ``mcp`` is accepted but currently finds no bundled
assets (extension seam for future skills/subagents/hooks).
"""
from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from robotmcp.onboarding import adapters as A
from robotmcp.onboarding import codecs
from robotmcp.onboarding.manifest import Manifest, value_hash

WHAT_ALL = ["mcp", "skills", "agents", "hooks"]
WHAT_IMPLEMENTED = {"mcp"}


def resolved_command() -> str:
    """Absolute path to the installed ``robotmcp`` executable if resolvable, so the
    agent launches the tool-installed binary regardless of PATH context."""
    exe = shutil.which("robotmcp")
    if exe:
        return exe
    # running as `python -m robotmcp.server` / dev: fall back to a plain name
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0.endswith("robotmcp") and Path(argv0).exists():
        return str(Path(argv0).resolve())
    return "robotmcp"


@dataclass
class Result:
    agent: str
    scope: str
    what: str
    status: str          # installed | updated | already-present | removed | kept-user-modified | skipped | planned | no-assets | absent
    path: Optional[str] = None
    detail: str = ""


def _targets(agents_spec: str, home: Optional[Path]) -> List[A.AgentAdapter]:
    return A.resolve_selection(agents_spec, home=home)


def install(*, agents: str = "detected", scope: str = "project",
            whats: Optional[List[str]] = None, dry_run: bool = False,
            force: bool = False, command: Optional[str] = None,
            env: Optional[Dict[str, str]] = None,
            manifest: Optional[Manifest] = None,
            home: Optional[Path] = None, cwd: Optional[Path] = None) -> List[Result]:
    whats = whats or ["mcp"]
    env = env or {}
    command = command or resolved_command()
    manifest = manifest or Manifest()
    results: List[Result] = []

    for what in whats:
        if what not in WHAT_IMPLEMENTED:
            results.append(Result("*", scope, what, "no-assets",
                                  detail="no bundled assets of this kind yet"))
            continue
        for ad in _targets(agents, home):
            if ad.status != "supported":
                results.append(Result(ad.id, scope, what, "planned",
                                      detail="adapter convention unconfirmed"))
                continue
            if not ad.supports_scope(scope):
                other = "user" if scope == "project" else "project"
                results.append(Result(ad.id, scope, what, "skipped",
                                      detail=f"{scope} scope unsupported; try --scope {other}"))
                continue
            path = ad.resolve_path(scope, cwd=cwd, home=home)
            data, existed = codecs.load(path, ad.fmt)
            container = codecs.ensure_container(data, ad.container)
            entry = ad.build_entry(command, [], env)
            present = A.SERVER_NAME in container
            if present and not force:
                results.append(Result(ad.id, scope, what, "already-present",
                                      path=str(path), detail="use --force to overwrite"))
                continue
            if dry_run:
                results.append(Result(ad.id, scope, what,
                                      "updated" if present else "installed",
                                      path=str(path), detail="dry-run"))
                continue
            container[A.SERVER_NAME] = entry
            codecs.dump(path, ad.fmt, data)
            manifest.record(agent=ad.id, scope=scope, what=what, path=str(path),
                            value=entry, created_file=not existed)
            results.append(Result(ad.id, scope, what,
                                  "updated" if present else "installed", path=str(path)))
    if not dry_run:
        manifest.save()
    return results


def uninstall(*, agents: str = "detected", scope: Optional[str] = None,
              whats: Optional[List[str]] = None, dry_run: bool = False,
              manifest: Optional[Manifest] = None,
              home: Optional[Path] = None, cwd: Optional[Path] = None) -> List[Result]:
    whats = whats or WHAT_ALL
    manifest = manifest or Manifest()
    results: List[Result] = []
    # Selection by id; 'all'/'detected' still resolve to concrete adapters.
    wanted_ids = [a.id for a in _targets(agents, home)] if agents not in ("", None) else A.SUPPORTED_IDS
    if agents == "all":
        wanted_ids = A.SUPPORTED_IDS

    for e in list(manifest.entries_for(wanted_ids, scope, whats)):
        ad = A.get(e["agent"])
        path = Path(e["path"])
        data, existed = codecs.load(path, ad.fmt) if ad else ({}, False)
        container = codecs.get_container(data, ad.container) if ad else None
        if not container or A.SERVER_NAME not in container:
            if not dry_run:
                manifest.drop(e)
            results.append(Result(e["agent"], e["scope"], e["what"], "absent", path=str(path)))
            continue
        current_hash = value_hash(container[A.SERVER_NAME])
        if current_hash != e["value_hash"]:
            results.append(Result(e["agent"], e["scope"], e["what"], "kept-user-modified",
                                  path=str(path), detail="entry changed since install; left intact"))
            continue
        if dry_run:
            results.append(Result(e["agent"], e["scope"], e["what"], "removed",
                                  path=str(path), detail="dry-run"))
            continue
        del container[A.SERVER_NAME]
        codecs.prune_empty(data, ad.container)
        if e.get("created_file") and not data:
            path.unlink(missing_ok=True)
        else:
            codecs.dump(path, ad.fmt, data)
        manifest.drop(e)
        results.append(Result(e["agent"], e["scope"], e["what"], "removed", path=str(path)))
    if not dry_run:
        manifest.save()
    return results


def list_agents(*, home: Optional[Path] = None, cwd: Optional[Path] = None,
                manifest: Optional[Manifest] = None) -> List[Dict[str, str]]:
    manifest = manifest or Manifest()
    installed = {(e["agent"], e["scope"]) for e in manifest.entries}
    rows: List[Dict[str, str]] = []
    for ad in A.REGISTRY:
        rows.append({
            "id": ad.id, "name": ad.name, "status": ad.status,
            "detected": "yes" if ad.detect(home=home) else "no",
            "registered": "yes" if any(a == ad.id for a, _ in installed) else "no",
            "format": ad.fmt,
        })
    return rows

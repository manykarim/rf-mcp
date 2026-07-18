"""Per-agent adapters describing how to register the rf-mcp MCP server into each
coding agent's own configuration file.

Config conventions were researched 2026-07; formats drift, so each adapter is a
small self-contained record and every adapter is covered by a round-trip test.
An adapter with ``status="planned"`` (e.g. ``pi``) is listed but never written,
because its convention is unconfirmed.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SERVER_NAME = "robotmcp"


@dataclass(frozen=True)
class AgentAdapter:
    id: str
    name: str
    fmt: str                     # json | jsonc | toml | yaml
    container: List[str]         # key path to the servers map
    style: str                   # entry shape: standard | kilo | opencode | goose
    project_path: Optional[str]  # relative to CWD, or None if scope unsupported
    user_path: Optional[str]     # relative to $HOME (env-expanded), or None
    detect_bins: List[str] = field(default_factory=list)
    detect_dirs: List[str] = field(default_factory=list)  # relative to $HOME
    status: str = "supported"    # supported | planned

    # -- scope / path resolution -------------------------------------------
    def supports_scope(self, scope: str) -> bool:
        return (self.project_path if scope == "project" else self.user_path) is not None

    def resolve_path(self, scope: str, *, cwd: Optional[Path] = None,
                     home: Optional[Path] = None) -> Optional[Path]:
        cwd = cwd or Path.cwd()
        home = home or Path.home()
        if scope == "project":
            return (cwd / self.project_path) if self.project_path else None
        raw = self.user_path
        if not raw:
            return None
        raw = os.path.expandvars(raw)  # e.g. $CODEX_HOME
        p = Path(raw)
        return p if p.is_absolute() else (home / raw)

    # -- detection ----------------------------------------------------------
    def detect(self, *, home: Optional[Path] = None) -> bool:
        if self.status != "supported":
            return False
        for b in self.detect_bins:
            if shutil.which(b):
                return True
        home = home or Path.home()
        return any((home / d).exists() for d in self.detect_dirs)

    # -- entry shape --------------------------------------------------------
    def build_entry(self, command: str, args: List[str], env: Dict[str, str]) -> Any:
        if self.style == "opencode":
            e: Dict[str, Any] = {"type": "local", "command": [command, *args], "enabled": True}
            if env:
                e["environment"] = dict(env)
            return e
        if self.style == "goose":
            e = {"name": SERVER_NAME, "cmd": command, "args": list(args),
                 "type": "stdio", "enabled": True, "timeout": 300}
            if env:
                e["envs"] = dict(env)
            return e
        # standard / kilo
        e = {"command": command}
        if args:
            e["args"] = list(args)
        if env:
            e["env"] = dict(env)
        if self.style == "kilo":
            e = {"type": "stdio", **e}
        return e


# ---------------------------------------------------------------------------
# Registry — one record per agent.
# ---------------------------------------------------------------------------
REGISTRY: List[AgentAdapter] = [
    AgentAdapter(
        id="claude-code", name="Claude Code", fmt="json", container=["mcpServers"],
        style="standard", project_path=".mcp.json", user_path=".claude.json",
        detect_bins=["claude"], detect_dirs=[".claude", ".claude.json"],
    ),
    AgentAdapter(
        id="codex", name="OpenAI Codex", fmt="toml", container=["mcp_servers"],
        style="standard", project_path=".codex/config.toml",
        user_path="$CODEX_HOME/config.toml" if os.environ.get("CODEX_HOME") else ".codex/config.toml",
        detect_bins=["codex"], detect_dirs=[".codex"],
    ),
    AgentAdapter(
        id="copilot", name="GitHub Copilot (VS Code)", fmt="json", container=["servers"],
        style="standard", project_path=".vscode/mcp.json", user_path=None,
        detect_bins=["code"], detect_dirs=[".vscode"],
    ),
    AgentAdapter(
        id="opencode", name="opencode", fmt="json", container=["mcp"],
        style="opencode", project_path="opencode.json",
        user_path=".config/opencode/opencode.json",
        detect_bins=["opencode"], detect_dirs=[".config/opencode"],
    ),
    AgentAdapter(
        id="gemini", name="Gemini CLI", fmt="json", container=["mcpServers"],
        style="standard", project_path=".gemini/settings.json",
        user_path=".gemini/settings.json",
        detect_bins=["gemini"], detect_dirs=[".gemini"],
    ),
    AgentAdapter(
        id="kilo", name="Kilo Code", fmt="jsonc", container=["mcp"],
        style="kilo", project_path=".kilo/kilo.jsonc",
        user_path=".config/kilo/kilo.jsonc",
        detect_bins=["kilo"], detect_dirs=[".config/kilo"],
    ),
    AgentAdapter(
        id="goose", name="goose", fmt="yaml", container=["extensions"],
        style="goose", project_path=None, user_path=".config/goose/config.yaml",
        detect_bins=["goose"], detect_dirs=[".config/goose"],
    ),
    AgentAdapter(
        id="cursor", name="Cursor", fmt="json", container=["mcpServers"],
        style="standard", project_path=".cursor/mcp.json", user_path=".cursor/mcp.json",
        detect_bins=["cursor"], detect_dirs=[".cursor"],
    ),
    # Unconfirmed MCP-config convention — listed but never written.
    AgentAdapter(
        id="pi", name="pi", fmt="json", container=["mcpServers"], style="standard",
        project_path=None, user_path=None, status="planned",
    ),
]

BY_ID: Dict[str, AgentAdapter] = {a.id: a for a in REGISTRY}
SUPPORTED_IDS = [a.id for a in REGISTRY if a.status == "supported"]


def get(agent_id: str) -> Optional[AgentAdapter]:
    return BY_ID.get(agent_id)


def resolve_selection(spec: str, *, home: Optional[Path] = None) -> List[AgentAdapter]:
    """Turn an --agents value (``all`` | ``detected`` | csv) into adapters.
    ``planned`` adapters are excluded from all/detected but selectable explicitly
    so callers can surface their status."""
    spec = (spec or "detected").strip()
    if spec == "all":
        return [a for a in REGISTRY if a.status == "supported"]
    if spec == "detected":
        return [a for a in REGISTRY if a.status == "supported" and a.detect(home=home)]
    out: List[AgentAdapter] = []
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        a = BY_ID.get(token)
        if a:
            out.append(a)
    return out

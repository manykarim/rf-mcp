"""`rf-mcp init`: register rf-mcp with coding agents, in the project (`--scope project`) or for the user (`--scope user`).

Entries carry no project path: the server finds the project root and environment per call. Project files are merged
(only the `rf-mcp` entry is added or replaced). User-level entries are written with each agent's own CLI
(`claude mcp add --scope user`, `codex mcp add`, `copilot mcp add`, `code --add-mcp`, `agy mcp add`), so the agent's
file format and locking stay its own business; OpenCode has no such command and its user file is merged.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import tomlkit

AgentName = Literal["claude", "codex", "agy", "copilot", "opencode", "all"]
Scope = Literal["project", "user"]
InstallKind = Literal["project", "global"]
AGENTS: tuple[str, ...] = ("claude", "codex", "agy", "copilot", "opencode")
SERVER_NAME = "rf-mcp"

DEFAULT_ENV = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "ROBOTCODE_FORCE_AI_AGENT": "1"}

# Measured with codex-cli 0.166 (2026-11-07): Codex loads MCP tools lazily and shows the model neither the server
# instructions nor the tool descriptions, so without the synced skills GPT-5.6-Luna made 0 rf-mcp calls in 14 runs;
# and under approval_policy="never" every tool not annotated read-only (rf_run, rf_repl, rf_debug) is rejected.
CODEX_NOTES = (
    "Codex shows the agent neither rf-mcp's instructions nor its tool descriptions: run",
    "  rf-mcp sync-skills --agent codex",
    "so the skills in .agents/skills route Robot Framework work to these tools.",
    'Unattended runs (codex exec, approval_policy = "never") reject rf_run, rf_repl and rf_debug unless approved.',
    "They run outside Codex's sandbox (as the project's own robot runs would); to allow them, uncomment:",
    'default_tools_approval_mode = "approve"',
)

Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(cmd), capture_output=True, text=True, encoding="utf-8", errors="replace",
                          stdin=subprocess.DEVNULL, timeout=120, check=False)


@dataclass
class Target:
    """One place an entry goes: a file to merge or an agent CLI call."""

    agent: str
    status: str = "preview"  # preview | written | manual | error
    path: Path | None = None
    content: str | None = None
    command: list[str] | None = None
    message: str | None = None
    extra_files: dict[Path, str] = field(default_factory=dict)

    def report(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status, "written": self.status == "written"}
        if self.path is not None:
            out["path"] = str(self.path)
        if self.content is not None:
            out["content"] = self.content
        if self.command is not None:
            out["command"] = self.command
        if self.message:
            out["message"] = self.message
        return out


# --- what to start ----------------------------------------------------------------------------


def detect_install_kind(ws_root: Path) -> InstallKind:
    """`project` when this rf-mcp runs from the project's `.venv`, else `global` (uv tool, pipx, another venv)."""
    try:
        return "project" if Path(sys.prefix).resolve() == (ws_root / ".venv").resolve() else "global"
    except OSError:
        return "global"


def _rf_mcp_by_path_or_name() -> str:
    """`rf-mcp` when it is on PATH, else the full path of the executable next to this interpreter."""
    if shutil.which(SERVER_NAME):
        return SERVER_NAME
    for name in (f"{SERVER_NAME}.exe", SERVER_NAME):
        candidate = Path(sys.executable).parent / name
        if candidate.is_file():
            return str(candidate)
    return SERVER_NAME


def server_command(ws_root: Path, scope: Scope, install_kind: InstallKind) -> tuple[str, list[str]]:
    """Return (command, args) of the stdio entry; no project path, no flags."""
    if scope == "user":
        return _rf_mcp_by_path_or_name(), []
    if install_kind == "project":
        if (ws_root / "pyproject.toml").is_file():
            return "uv", ["run", SERVER_NAME]
        return sys.executable, ["-m", "robotmcp"]
    return SERVER_NAME, []


# --- merging files ----------------------------------------------------------------------------


def _strip_jsonc(text: str) -> str:
    """Drop // and /* */ comments outside strings."""
    out: list[str] = []
    i, n, in_str = 0, len(text), False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 1
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
            out.append(ch)
        elif text.startswith("//", i):
            while i < n and text[i] not in "\r\n":
                i += 1
            continue
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """(data, problem). A file with comments is not rewritten, so its comments survive."""
    if not path.exists():
        return {}, None
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}, None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        try:
            json.loads(_strip_jsonc(text))
        except json.JSONDecodeError:
            return None, f"{path} does not parse: {exc}"
        return None, f"{path} has comments; rf-mcp does not rewrite it, so add the entry below by hand."
    if not isinstance(data, dict):
        return None, f"{path} does not hold a JSON object."
    return data, None


def _merge_json(target: Target, path: Path, map_key: str, entry: dict[str, Any]) -> None:
    data, problem = _load_json(path)
    target.path = path
    if data is None:
        target.status = "error" if "does not parse" in (problem or "") or "JSON object" in (problem or "") else "manual"
        target.message = problem
        target.content = json.dumps({map_key: {SERVER_NAME: entry}}, indent=2) + "\n"
        return
    servers = data.get(map_key)
    if not isinstance(servers, dict):
        servers = {}
    servers[SERVER_NAME] = entry
    data[map_key] = servers
    target.content = json.dumps(data, indent=2) + "\n"


def _merge_codex_toml(target: Target, path: Path, command: str, args: list[str], url: str | None) -> None:
    target.path = path
    try:
        doc = tomlkit.parse(path.read_text(encoding="utf-8")) if path.exists() else tomlkit.document()
    except Exception as exc:
        target.status, target.message = "error", f"{path} does not parse: {exc}"
        return
    servers = doc.get("mcp_servers")
    if servers is None:
        servers = tomlkit.table(is_super_table=True)
        doc["mcp_servers"] = servers
    table = tomlkit.table()
    if url:
        table["url"] = url
    else:
        table["command"] = command.replace("\\", "/")
        table["args"] = [a.replace("\\", "/") for a in args]
        for line in CODEX_NOTES:
            table.add(tomlkit.comment(line))
        env = tomlkit.table()
        for key, value in DEFAULT_ENV.items():
            env[key] = value
        table["env"] = env
    servers[SERVER_NAME] = table
    target.content = tomlkit.dumps(doc)


# --- per-agent entries --------------------------------------------------------------------------


def _stdio_entry(command: str, args: list[str], **extra: Any) -> dict[str, Any]:
    return {**extra, "command": command, "args": args, "env": dict(DEFAULT_ENV)}


def project_targets(agent: str, ws_root: Path, command: str, args: list[str], url: str | None) -> list[Target]:
    t = Target(agent)
    entry: dict[str, Any]
    if agent == "claude":
        entry = {"type": "http", "url": url} if url else _stdio_entry(command, args, type="stdio")
        _merge_json(t, ws_root / ".mcp.json", "mcpServers", entry)
    elif agent == "codex":
        _merge_codex_toml(t, ws_root / ".codex" / "config.toml", command, args, url)
    elif agent == "agy":
        entry = {"url": url} if url else _stdio_entry(command, args)
        _merge_json(t, ws_root / ".gemini" / "settings.json", "mcpServers", entry)
        plugin_dir = ws_root / ".agents" / "plugins" / SERVER_NAME
        t.extra_files = {
            plugin_dir / "plugin.json": json.dumps({"name": SERVER_NAME}, indent=2) + "\n",
            plugin_dir / "mcp_config.json": json.dumps({"mcpServers": {SERVER_NAME: entry}}, indent=2) + "\n",
        }
    elif agent == "copilot":
        entry = {"type": "http", "url": url} if url else _stdio_entry(command, args, type="stdio")
        _merge_json(t, ws_root / ".vscode" / "mcp.json", "servers", entry)
    elif agent == "opencode":
        entry = ({"type": "remote", "url": url} if url
                 else {"type": "local", "command": [command, *args], "environment": dict(DEFAULT_ENV)})
        _merge_json(t, ws_root / "opencode.json", "mcp", entry)
    return [t]


def _config_home() -> Path:
    raw = os.environ.get("XDG_CONFIG_HOME")
    return Path(raw) if raw else Path.home() / ".config"


def user_targets(agent: str, command: str, args: list[str], url: str | None) -> list[Target]:
    env_flags = [f"{k}={v}" for k, v in DEFAULT_ENV.items()]
    if agent == "claude":
        cmd = (["claude", "mcp", "add", "--scope", "user", "--transport", "http", SERVER_NAME, url] if url else
               # The name goes before -e: claude's -e takes several values and would swallow it.
               ["claude", "mcp", "add", "--scope", "user", SERVER_NAME, *[x for f in env_flags for x in ("-e", f)],
                "--", command, *args])
        return [Target("claude", command=cmd)]
    if agent == "codex":
        cmd = (["codex", "mcp", "add", SERVER_NAME, "--url", url] if url else
               ["codex", "mcp", "add", SERVER_NAME, *[x for f in env_flags for x in ("--env", f)], "--", command, *args])
        return [Target("codex", command=cmd)]
    if agent == "agy":
        cmd = (["agy", "mcp", "add", SERVER_NAME, url] if url else
               ["agy", "mcp", "add", *[x for f in env_flags for x in ("-e", f)], SERVER_NAME, command, *args])
        return [Target("agy", command=cmd)]
    if agent == "copilot":
        vscode: dict[str, Any] = {"name": SERVER_NAME, "type": "http", "url": url} if url else {
            "name": SERVER_NAME, "type": "stdio", "command": command, "args": args, "env": dict(DEFAULT_ENV)}
        cli = (["copilot", "mcp", "add", "--transport", "http", SERVER_NAME, url] if url else
               ["copilot", "mcp", "add", *[x for f in env_flags for x in ("--env", f)], SERVER_NAME, "--", command,
                *args])
        return [Target("copilot", command=["code", "--add-mcp", json.dumps(vscode)]), Target("copilot-cli", command=cli)]
    if agent == "opencode":
        t = Target("opencode")
        oc_entry: dict[str, Any] = ({"type": "remote", "url": url} if url
                 else {"type": "local", "command": [command, *args], "environment": dict(DEFAULT_ENV)})
        _merge_json(t, _config_home() / "opencode" / "opencode.json", "mcp", oc_entry)
        return [t]
    return []


# --- apply ------------------------------------------------------------------------------------


def _write_file(t: Target) -> None:
    assert t.path is not None and t.content is not None
    t.path.parent.mkdir(parents=True, exist_ok=True)
    t.path.write_text(t.content, encoding="utf-8")
    for extra, text in t.extra_files.items():
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_text(text, encoding="utf-8")
    t.status = "written"


def _run_cli(t: Target, run: Runner) -> None:
    assert t.command is not None
    exe = shutil.which(t.command[0])
    if exe is None:
        t.status = "manual"
        t.message = f"`{t.command[0]}` is not installed or not on PATH; run the command yourself once it is."
        return
    if t.agent == "claude":  # `claude mcp add` refuses an existing name; replace it
        run([exe, "mcp", "remove", "--scope", "user", SERVER_NAME])
    proc = run([exe, *t.command[1:]])
    if proc.returncode == 0:
        t.status = "written"
        t.message = (proc.stdout or "").strip()[-300:] or None
    else:
        t.status = "error"
        t.message = ((proc.stderr or "") + (proc.stdout or "")).strip()[-600:]


def generate_agent_mcp_configs(
    *,
    agent: AgentName = "all",
    mode: Literal["server", "client"] = "server",
    root: Path | None = None,
    url: str | None = None,
    write_files: bool = False,
    scope: Scope = "project",
    install_kind: InstallKind | None = None,
    run: Runner = _run,
) -> dict[str, Any]:
    """Build (and with `write_files` apply) the rf-mcp entry for each selected agent.

    `mode` is kept for compatibility: the entry carries no `--mode`, the server defaults to `auto`.
    """
    del mode
    ws_root = (root or Path.cwd()).expanduser().resolve()
    kind = install_kind or detect_install_kind(ws_root)
    command, args = server_command(ws_root, scope, kind)
    selected = list(AGENTS) if agent == "all" else [agent]

    report: dict[str, Any] = {}
    for name in selected:
        targets = (project_targets(name, ws_root, command, args, url) if scope == "project"
                   else user_targets(name, command, args, url))
        for t in targets:
            if write_files and t.status == "preview":
                if t.command is not None:
                    _run_cli(t, run)
                elif t.path is not None and t.content is not None:
                    _write_file(t)
            report[t.agent] = t.report()
    return report


def init_exit_code(report: dict[str, Any]) -> int:
    """0 unless an entry could not be written (unparsable file, file with comments, failing CLI)."""
    return 1 if any(item.get("status") in ("error", "manual") for item in report.values()) else 0

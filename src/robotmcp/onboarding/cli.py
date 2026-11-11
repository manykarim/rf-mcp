"""Onboarding, diagnostics (`doctor`), MCP config generation (`init`), and skill sync (`sync-skills`) CLI."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal, Sequence

from robotmcp.config import ServerConfig
from robotmcp.hints import load_custom_rules_with_errors
from robotmcp.runner.cli_runner import resolve_robotcode_executable
from robotmcp.runner.shell_escape import build_agent_env
from robotmcp.skills.provider import discover_skill_summaries, resolve_skill_roots

AgentName = Literal["claude", "codex", "agy", "copilot", "opencode", "all"]

AGENT_SKILL_TARGETS: dict[str, str] = {
    "claude": ".claude/skills",
    "codex": ".agents/skills",
    "agy": ".agents/skills",
    "copilot": ".github/skills",
    "opencode": ".agents/skills",
}


REQUIRED_COMMANDS = ("discover", "analyze", "doc", "robot", "results", "session", "repl", "robot-debug", "rebot", "profiles", "config")
REQUIRED_SESSION_COMMANDS = ("start", "send", "wait", "status", "result", "interrupt", "restart", "list", "stop")
INSTALL_HINT = 'uv add --dev --prerelease=allow "robotcode[all]>=2.9.0.dev115"'


def _help_commands(help_text: str) -> set[str]:
    """Return the subcommand names listed under `Commands:` (and `Aliases:`) of a click help text."""
    names: set[str] = set()
    in_section = False
    for line in help_text.splitlines():
        if line.strip() in ("Commands:", "Aliases:"):
            in_section = True
            continue
        if in_section:
            m = re.match(r"^  (\S+)\s{2,}", line)
            if m:
                names.add(m.group(1))
            elif line and not line.startswith(" "):
                in_section = False
    return names


def _run_help(rc_exec: list[str], args: list[str], root: Path) -> str:
    proc = subprocess.run(
        [*rc_exec, *args, "--help"],
        cwd=str(root),
        env=build_agent_env(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def _check_robotcode_status(root: Path) -> dict[str, Any]:
    """Inspect `robotcode` CLI version and the subcommands rf-mcp calls (`session` needs >=2.9.0.dev115)."""
    rc_exec = resolve_robotcode_executable()
    try:
        ver_proc = subprocess.run(
            [*rc_exec, "--version"],
            cwd=str(root),
            env=build_agent_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        return {
            "installed": False,
            "version": None,
            "meets_min_version": False,
            "commands": {},
            "session_commands": {},
            "install_hint": INSTALL_HINT,
        }

    version_str = (ver_proc.stdout or ver_proc.stderr or "").strip()
    top_level = _help_commands(_run_help(rc_exec, [], root))
    required_commands = {name: name in top_level for name in REQUIRED_COMMANDS}
    session_level = (
        _help_commands(_run_help(rc_exec, ["session"], root)) if required_commands["session"] else set()
    )
    session_commands = {name: name in session_level for name in REQUIRED_SESSION_COMMANDS}

    meets_min = False
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", version_str)
    if m:
        major, minor = int(m.group(1)), int(m.group(2))
        if (major, minor) >= (2, 9) and all(session_commands.values()):
            meets_min = True

    complete = meets_min and all(required_commands.values())
    return {
        "installed": ver_proc.returncode == 0,
        "version": version_str,
        "meets_min_version": meets_min,
        "commands": required_commands,
        "session_commands": session_commands,
        "install_hint": None if complete else INSTALL_HINT,
    }


def run_doctor(
    *,
    root: Path | None = None,
    skills_dirs: Sequence[Path] | None = None,
    as_json: bool = False,
) -> dict[str, Any]:
    """Run `rf-mcp doctor` diagnostics and print report."""
    ws_root = (root or Path.cwd()).expanduser().resolve()
    cfg = ServerConfig(
        workspace_root=ws_root,
        skills_dirs=list(skills_dirs or []),
    )
    rc_info = _check_robotcode_status(ws_root)
    skill_roots = resolve_skill_roots(cfg)
    skills = discover_skill_summaries(skill_roots)
    hints_path = cfg.resolve_hints_file()
    hint_errors = load_custom_rules_with_errors(hints_path)[1] if hints_path else []

    report: dict[str, Any] = {
        "ok": bool(
            rc_info["installed"]
            and rc_info["meets_min_version"]
            and all(rc_info["commands"].values())
        ),
        "python": sys.version.split()[0],
        "workspace_root": str(ws_root),
        "robotcode": rc_info,
        "skills": skills,
        "hints_file": str(hints_path) if hints_path else None,
        "hints_status": "not present" if not hints_path else ("errors" if hint_errors else "ok"),
        "hints_errors": hint_errors,
    }

    if as_json:
        print(json.dumps(report, indent=2))
    else:
        print("=== rf-mcp doctor ===")
        print(f"Workspace Root : {ws_root}")
        print(f"Python         : {report['python']}")
        print(
            f"RobotCode      : {rc_info['version'] or 'NOT FOUND'} "
            f"({'OK' if report['ok'] else 'NEEDS UPGRADE/EXTRAS'})"
        )
        missing = [k for k, v in rc_info["commands"].items() if not v]
        missing += [f"session {k}" for k, v in rc_info.get("session_commands", {}).items() if not v]
        if missing:
            print(f"Missing Cmds   : {', '.join(missing)}")
        if rc_info["install_hint"]:
            print(f"Install Hint   : {rc_info['install_hint']}")
        print(f"Active Skills  : {', '.join(s['name'] for s in skills) or 'none'}")
        print(f"Custom Hints   : {report['hints_file'] or 'none (using built-in rules)'} [{report['hints_status']}]")
        for err in hint_errors:
            print(f"  Hint Error   : {err}")
    return report


def project_server_command(ws_root: Path, mode: str) -> tuple[str, list[str]]:
    """Return (command, args) that start rf-mcp from the project's environment.

    A uv project (`pyproject.toml` in the root) runs `uv run --directory <root> rf-mcp`, so the
    server sees the project's robotcode and libraries. Otherwise the interpreter that runs this
    command is used. The rf-mcp source checkout is never referenced.
    """
    server_flags = ["--mode", mode, "--transport", "stdio", "-r", str(ws_root)]
    if (ws_root / "pyproject.toml").is_file():
        return "uv", ["run", "--directory", str(ws_root), "rf-mcp", *server_flags]
    return sys.executable, ["-m", "robotmcp", *server_flags]


# Measured with codex-cli 0.166 (2026-11-07): Codex loads MCP tools lazily and shows the model neither the server
# instructions nor the tool descriptions, so without the synced skills GPT-5.6-Luna made 0 rf-mcp calls in 14 runs;
# and under approval_policy="never" every tool not annotated read-only (rf_run, rf_repl, rf_debug) is rejected.
CODEX_NOTES = (
    "# Codex shows the agent neither rf-mcp's instructions nor its tool descriptions: run\n"
    "#   rf-mcp sync-skills --agent codex\n"
    "# so the skills in .agents/skills route Robot Framework work to these tools.\n"
    "# Unattended runs (codex exec, approval_policy = \"never\") reject rf_run, rf_repl and rf_debug unless approved.\n"
    "# They run outside Codex's sandbox (as the project's own robot runs would); to allow them, uncomment:\n"
    "# default_tools_approval_mode = \"approve\"\n"
)


def generate_agent_mcp_configs(
    *,
    agent: AgentName = "all",
    mode: Literal["server", "client"] = "server",
    root: Path | None = None,
    url: str | None = None,
    write_files: bool = False,
) -> dict[str, Any]:
    """Generate (and optionally write) MCP client configuration files for coding agents."""
    ws_root = (root or Path.cwd()).expanduser().resolve()
    targets = (
        ["claude", "codex", "agy", "copilot", "opencode"]
        if agent == "all"
        else [agent]
    )
    generated: dict[str, Any] = {}

    server_command, server_args = project_server_command(ws_root, mode)
    default_env = {
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "ROBOTCODE_FORCE_AI_AGENT": "1",
    }

    for target_agent in targets:
        if target_agent == "claude":
            rel_path = ".mcp.json"
            entry: dict[str, Any] = (
                {"type": "http", "url": url}
                if url
                else {"type": "stdio", "command": server_command, "args": server_args, "env": default_env}
            )
            content = json.dumps({"mcpServers": {"rf-mcp": entry}}, indent=2) + "\n"
        elif target_agent == "codex":
            rel_path = ".codex/config.toml"
            if url:
                content = f'[mcp_servers.rf-mcp]\nurl = "{url}"\n'
            else:
                args_toml = ", ".join(f'"{a.replace(chr(92), "/")}"' for a in server_args)
                content = (
                    f'[mcp_servers.rf-mcp]\ncommand = "{server_command.replace(chr(92), "/")}"\nargs = [{args_toml}]\n'
                    + CODEX_NOTES
                    + '\n[mcp_servers.rf-mcp.env]\nPYTHONUTF8 = "1"\nPYTHONIOENCODING = "utf-8"\nROBOTCODE_FORCE_AI_AGENT = "1"\n'
                )
        elif target_agent == "agy":
            rel_path = ".gemini/settings.json"
            entry_agy: dict[str, Any] = (
                {"url": url}
                if url
                else {"command": server_command, "args": server_args, "env": default_env}
            )
            content = json.dumps({"mcpServers": {"rf-mcp": entry_agy}}, indent=2) + "\n"
            if write_files:
                plugin_dir = ws_root / ".agents" / "plugins" / "rf-mcp"
                plugin_dir.mkdir(parents=True, exist_ok=True)
                (plugin_dir / "plugin.json").write_text(
                    json.dumps({"name": "rf-mcp"}, indent=2) + "\n",
                    encoding="utf-8",
                )
                (plugin_dir / "mcp_config.json").write_text(
                    content,
                    encoding="utf-8",
                )
        elif target_agent == "copilot":
            rel_path = ".vscode/mcp.json"
            entry_cp: dict[str, Any] = (
                {"type": "http", "url": url}
                if url
                else {"type": "stdio", "command": server_command, "args": server_args, "env": default_env}
            )
            content = json.dumps({"servers": {"rf-mcp": entry_cp}}, indent=2) + "\n"
        elif target_agent == "opencode":
            rel_path = "opencode.json"
            entry_oc: dict[str, Any] = (
                {"type": "remote", "url": url}
                if url
                else {"type": "local", "command": [server_command, *server_args], "environment": default_env}
            )
            content = json.dumps({"mcp": {"rf-mcp": entry_oc}}, indent=2) + "\n"
        else:
            continue

        file_path = ws_root / rel_path
        if write_files:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

        generated[target_agent] = {
            "path": str(file_path),
            "written": write_files,
            "content": content,
        }

    return generated


def sync_skills_to_directory(
    *,
    root: Path | None = None,
    agent: str = "codex",
    target_dir: str | Path | None = None,
    skills_dirs: Sequence[Path] | None = None,
    force: bool = False,
) -> dict[str, list[str]]:
    """Copy active skill directories into the target agent skill folder.

    An existing skill folder whose content differs is left alone (listed under `skipped`)
    unless `force` is set; identical folders are listed under `unchanged`.
    """
    ws_root = (root or Path.cwd()).expanduser().resolve()
    cfg = ServerConfig(
        workspace_root=ws_root,
        skills_dirs=list(skills_dirs or []),
    )
    roots = resolve_skill_roots(cfg)
    summaries = discover_skill_summaries(roots)

    rel_target = (
        Path(target_dir)
        if target_dir is not None
        else Path(AGENT_SKILL_TARGETS.get(agent, ".agents/skills"))
    )
    dest_root = (
        rel_target.expanduser().resolve()
        if rel_target.is_absolute()
        else (ws_root / rel_target).resolve()
    )
    dest_root.mkdir(parents=True, exist_ok=True)

    result: dict[str, list[str]] = {"synced": [], "unchanged": [], "skipped": []}
    for item in summaries:
        src_dir = Path(item["path"])
        dst_dir = dest_root / item["name"]
        if src_dir.resolve() == dst_dir.resolve() or (
            dst_dir.exists() and _tree_snapshot(src_dir) == _tree_snapshot(dst_dir)
        ):
            result["unchanged"].append(str(dst_dir))
            continue
        if dst_dir.exists():
            if not force:
                result["skipped"].append(str(dst_dir))
                continue
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        result["synced"].append(str(dst_dir))

    return result


def _tree_snapshot(directory: Path) -> dict[str, bytes]:
    """Map each file below `directory` (relative POSIX path) to its bytes."""
    return {
        f.relative_to(directory).as_posix(): f.read_bytes()
        for f in sorted(directory.rglob("*"))
        if f.is_file() and "__pycache__" not in f.parts
    }


def run_onboarding_cli(argv: Sequence[str]) -> int:
    """Dispatch `rf-mcp {doctor|init|sync-skills}` subcommands."""
    parser = argparse.ArgumentParser(prog="rf-mcp")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    doc_p = subparsers.add_parser(
        "doctor", help="Check robotcode >= 2.9.0.dev115, extras, skills, and hints."
    )
    doc_p.add_argument("-r", "--root", default=None, help="Workspace root.")
    doc_p.add_argument(
        "--skills-dir", action="append", dest="skills_dirs", default=None
    )
    doc_p.add_argument("--json", action="store_true", help="Output JSON report.")

    init_p = subparsers.add_parser(
        "init",
        help="Generate MCP configuration for Claude Code, Codex, Antigravity, Copilot, or OpenCode.",
    )
    init_p.add_argument(
        "--agent",
        choices=["claude", "codex", "agy", "copilot", "opencode", "all"],
        default="all",
    )
    init_p.add_argument("--mode", choices=["server", "client"], default="server")
    init_p.add_argument("--url", default=None, help="Remote HTTP MCP URL (Modes 2 & 3).")
    init_p.add_argument("-r", "--root", default=None, help="Workspace root.")
    init_p.add_argument(
        "--write", action="store_true", help="Write config files to workspace."
    )

    sync_p = subparsers.add_parser(
        "sync-skills",
        help="Synchronize active MCP skills into a local agent skill directory.",
    )
    sync_p.add_argument(
        "--agent",
        choices=["claude", "codex", "agy", "copilot", "opencode"],
        default="codex",
    )
    sync_p.add_argument(
        "--target", default=None, help="Target directory (default: .agents/skills)."
    )
    sync_p.add_argument("-r", "--root", default=None, help="Workspace root.")
    sync_p.add_argument(
        "--skills-dir", action="append", dest="skills_dirs", default=None
    )
    sync_p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing skill folders whose content differs.",
    )

    ns = parser.parse_args(list(argv))
    ws_root = Path(ns.root).expanduser().resolve() if ns.root else Path.cwd().resolve()

    if ns.subcommand == "doctor":
        s_dirs = [Path(p) for p in (ns.skills_dirs or [])]
        rep = run_doctor(root=ws_root, skills_dirs=s_dirs, as_json=ns.json)
        return 0 if rep["ok"] else 1

    if ns.subcommand == "init":
        configs = generate_agent_mcp_configs(
            agent=ns.agent,
            mode=ns.mode,
            root=ws_root,
            url=ns.url,
            write_files=ns.write,
        )
        print(json.dumps(configs, indent=2))
        return 0

    if ns.subcommand == "sync-skills":
        s_dirs = [Path(p) for p in (ns.skills_dirs or [])]
        outcome = sync_skills_to_directory(
            root=ws_root,
            agent=ns.agent,
            target_dir=ns.target,
            skills_dirs=s_dirs,
            force=ns.force,
        )
        print(json.dumps(outcome, indent=2))
        if outcome["skipped"]:
            print(
                "Skipped existing skill folders that differ from the source; "
                "rerun with --force to overwrite them.",
                file=sys.stderr,
            )
            return 1
        return 0

    return 1

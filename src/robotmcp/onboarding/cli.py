"""Onboarding, diagnostics (`doctor`), MCP config generation (`init`), and skill sync (`sync-skills`) CLI."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from robotmcp.config import ServerConfig
from robotmcp.onboarding.agent_config import generate_agent_mcp_configs, init_exit_code
from robotmcp.hints import load_custom_rules_with_errors
from robotmcp.runner.cli_runner import resolve_robotcode_executable
from robotmcp.runner.shell_escape import build_agent_env
from robotmcp.skills.provider import discover_skill_summaries, resolve_skill_roots

AGENT_SKILL_TARGETS: dict[str, str] = {
    "claude": ".claude/skills",
    "codex": ".agents/skills",
    "agy": ".agents/skills",
    "copilot": ".github/skills",
    "opencode": ".agents/skills",
}


REQUIRED_COMMANDS = ("discover", "analyze", "doc", "robot", "results", "session", "repl", "robot-debug", "rebot", "profiles", "config")
REQUIRED_SESSION_COMMANDS = ("start", "send", "wait", "status", "result", "interrupt", "restart", "list", "stop")
INSTALL_HINT = 'uv add --dev "robotcode[all]>=2.9.0.dev115"'


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
    init_p.add_argument("--mode", choices=["server", "client"], default="server", help=argparse.SUPPRESS)
    init_p.add_argument("--url", default=None, help="Remote HTTP MCP URL (Modes 2 & 3).")
    init_p.add_argument("-r", "--root", default=None, help="Workspace root.")
    init_p.add_argument(
        "--scope",
        choices=["project", "user"],
        default="project",
        help="project: write the project's agent files (commit them); user: register rf-mcp for every project of "
        "this user with each agent's own CLI (for a global install, `uv tool install rf-mcp`).",
    )
    init_p.add_argument(
        "--write", action="store_true", help="Apply the entries (default: print what would change)."
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
            scope=ns.scope,
        )
        print(json.dumps(configs, indent=2))
        return init_exit_code(configs)

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

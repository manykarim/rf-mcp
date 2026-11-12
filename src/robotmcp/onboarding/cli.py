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
from robotmcp.onboarding.agent_config import detect_install_kind, generate_agent_mcp_configs, init_exit_code
from robotmcp.hints import load_custom_rules_with_errors
from robotmcp.runner.cli_runner import resolve_robotcode_executable
from robotmcp.runner.environment import EnvironmentResolver
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


def _check_robotcode_status(root: Path, rc_exec: list[str] | None = None) -> dict[str, Any]:
    """Inspect `robotcode` CLI version and the subcommands rf-mcp calls (`session` needs >=2.9.0.dev115)."""
    rc_exec = rc_exec or resolve_robotcode_executable()
    try:
        ver_proc = subprocess.run(
            [*rc_exec, "--version"],
            cwd=str(root),
            env=build_agent_env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=300,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
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


_ROOT_SOURCES = {"pinned": "given with -r", "cwd": "working directory, no project marker above it"}


def _python_version(interpreter: Path | None) -> str | None:
    if interpreter is None:
        return None
    if Path(sys.executable) == interpreter:
        return sys.version.split()[0]
    try:
        proc = subprocess.run([str(interpreter), "-c", "import sys; print(sys.version.split()[0])"],
                              capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() or None


def _uv_info(resolver: EnvironmentResolver) -> dict[str, Any]:
    path = resolver.uv
    if path is None:
        return {"path": None, "version": None}
    try:
        proc = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30,
                              stdin=subprocess.DEVNULL, check=False)
        version = proc.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        version = None
    return {"path": path, "version": version}


def run_doctor(
    *,
    root: Path | None = None,
    skills_dirs: Sequence[Path] | None = None,
    as_json: bool = False,
    resolver: EnvironmentResolver | None = None,
) -> dict[str, Any]:
    """Run `rf-mcp doctor`: the root and environment a call would get, robotcode, skills and hints."""
    cfg = ServerConfig(
        workspace_root=root.expanduser().resolve() if root else None,
        skills_dirs=list(skills_dirs or []),
    )
    ws_root, root_source = cfg.startup_root()
    resolver = resolver or EnvironmentResolver()
    env = resolver.resolve(ws_root)
    rc_info = _check_robotcode_status(ws_root, list(env.command_prefix))
    skill_roots = resolve_skill_roots(cfg)
    skills = discover_skill_summaries(skill_roots)
    hints_path = cfg.resolve_hints_file()
    hint_errors = load_custom_rules_with_errors(hints_path)[1] if hints_path else []
    uv = _uv_info(resolver)

    report: dict[str, Any] = {
        "ok": bool(
            rc_info["installed"]
            and rc_info["meets_min_version"]
            and all(rc_info["commands"].values())
            and env.libraries_visible
        ),
        "python": sys.version.split()[0],
        "install_kind": detect_install_kind(ws_root),
        "workspace_root": str(ws_root),
        "root_source": root_source,
        "environment": {
            "source": env.source,
            "interpreter": str(env.interpreter) if env.interpreter else None,
            "python": _python_version(env.interpreter),
            "robotcode_version": env.robotcode_version,
            "layered": env.layered,
            "layer_seconds": round(resolver.last_layer_seconds, 1) if env.layered and resolver.last_layer_seconds else None,
            "command": list(env.command_prefix),
            "warning": env.warning,
            "notes": list(env.notes),
        },
        "uv": uv,
        "robotcode": rc_info,
        "skills": skills,
        "hints_file": str(hints_path) if hints_path else None,
        "hints_status": "not present" if not hints_path else ("errors" if hint_errors else "ok"),
        "hints_errors": hint_errors,
    }

    if as_json:
        print(json.dumps(report, indent=2))
    else:
        e = report["environment"]
        how = _ROOT_SOURCES.get(root_source, f"found by {root_source.removeprefix('marker:')}")
        where = f"{e['source']} {e['interpreter'] or ''}".strip()
        if e["python"]:
            where += f" (Python {e['python']})"
        robotcode = f"robotcode {e['robotcode_version'] or 'unknown'}"
        if e["layered"]:
            robotcode += " added by uv" + (f", ready in {e['layer_seconds']} s" if e["layer_seconds"] else "")
        print("=== rf-mcp doctor ===")
        print(f"Install        : {report['install_kind']} (rf-mcp {_rf_mcp_version()}, Python {report['python']})")
        print(f"Project Root   : {ws_root} ({how})")
        print(f"Environment    : {where}")
        print(f"RobotCode      : {robotcode} ({'OK' if report['ok'] else 'NEEDS ATTENTION'})")
        print(f"uv             : {uv['version'] or 'not installed'}" + (f" ({uv['path']})" if uv["path"] else ""))
        missing = [k for k, v in rc_info["commands"].items() if not v]
        missing += [f"session {k}" for k, v in rc_info.get("session_commands", {}).items() if not v]
        if missing:
            print(f"Missing Cmds   : {', '.join(missing)}")
        if rc_info["install_hint"] and env.libraries_visible:
            print(f"Install Hint   : {rc_info['install_hint']}")
        print(f"Active Skills  : {', '.join(s['name'] for s in skills) or 'none'}")
        print(f"Custom Hints   : {report['hints_file'] or 'none (using built-in rules)'} [{report['hints_status']}]")
        for err in hint_errors:
            print(f"  Hint Error   : {err}")
        if env.warning:
            print(f"Warning        : {env.warning}")
    return report


def _rf_mcp_version() -> str:
    from robotmcp import __version__

    return __version__


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
        # Without -r, doctor finds the root the way the server does (working directory walked up).
        rep = run_doctor(root=Path(ns.root) if ns.root else None, skills_dirs=s_dirs, as_json=ns.json)
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

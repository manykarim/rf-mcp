"""Argparse front-end for the onboarding/installer subcommands.

Dispatched from ``robotmcp``'s ``main()`` BEFORE the server arg parser, so bare
``robotmcp`` still launches the MCP server and these subcommands never start it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from robotmcp.onboarding import diagnostics, installer
from robotmcp.onboarding import adapters as A

SUBCOMMANDS = ("init", "install", "uninstall", "list", "doctor")
VERSION_FLAGS = ("--version", "-V")


def _parse_env(pairs: List[str]) -> Dict[str, str]:
    """Turn repeated ``KEY=VALUE`` --env flags into a dict (bad entries ignored)."""
    out: Dict[str, str] = {}
    for item in pairs or []:
        if "=" in item:
            k, v = item.split("=", 1)
            if k.strip():
                out[k.strip()] = v
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="robotmcp",
        description="Install rf-mcp into coding agents and prepare the environment.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("init", help="Prepare the install and print the MCP config.")
    pi.add_argument("--browsers", action="store_true",
                    help="Initialize the Playwright browser (downloads a browser).")

    pd = sub.add_parser("doctor", help="Report installation health (read-only).")
    pd.add_argument("-C", "--project-dir", default=None,
                    help="Report which of THIS project's RF libraries rf-mcp can see.")
    sub.add_parser("list", help="List supported agents and their status.")

    def add_install_flags(sp, is_install: bool):
        sp.add_argument("--agents", default="detected",
                        help="all | detected | comma-separated agent ids")
        sp.add_argument("--scope", choices=["project", "user"],
                        default="project" if is_install else None,
                        help="project (default) or user (global) config")
        sp.add_argument("--what", default="mcp" if is_install else "mcp,skills,agents,hooks",
                        help="comma list of mcp,skills,agents,hooks")
        sp.add_argument("-C", "--project-dir", default=None,
                        help="Project directory: where the config is written AND whose "
                             "environment is inspected for project-aware launch (default: cwd).")
        sp.add_argument("--dry-run", action="store_true", help="Show the plan; write nothing.")
        sp.add_argument("--yes", "--no-input", dest="no_input", action="store_true",
                        help="Non-interactive; do not prompt.")
        if is_install:
            sp.add_argument("--force", action="store_true",
                            help="Overwrite an existing robotmcp entry / write despite verify failure.")
            sp.add_argument("--into-project", action="store_true",
                            help="Opt-in: install rf-mcp INTO the detected project env (mutating).")
            sp.add_argument("--attach", nargs="?", const="auto", default=None,
                            help="Use the attach bridge (optionally host[:port]); the project runs "
                                 "its own RF process with the McpAttach library.")
            sp.add_argument("--command", dest="command_override", default=None,
                            help="Override the launch command entirely (advanced).")
            sp.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                            help="Extra environment for the server entry (repeatable).")
            sp.add_argument("--no-verify", action="store_true",
                            help="Skip launching the resolved command to verify it before writing.")

    add_install_flags(sub.add_parser("install", help="Register rf-mcp into agents."), True)
    add_install_flags(sub.add_parser("uninstall", help="Remove rf-mcp from agents."), False)
    return p


def _print_results(results) -> int:
    if not results:
        print("Nothing to do.")
        return 0
    width = max(len(r.agent) for r in results)
    rc = 0
    for r in results:
        line = f"  {r.agent:<{width}}  {r.scope or '-':<7}  {r.what:<7}  {r.status}"
        if r.detail:
            line += f"  ({r.detail})"
        if r.path and r.status in ("installed", "updated", "removed"):
            line += f"  -> {r.path}"
        print(line)
    return rc


def _cmd_list() -> int:
    rows = installer.list_agents()
    print(f"  {'AGENT':<22} {'STATUS':<10} {'DETECTED':<9} {'REGISTERED':<11} FORMAT")
    for r in rows:
        print(f"  {r['name']:<22} {r['status']:<10} {r['detected']:<9} "
              f"{r['registered']:<11} {r['format']}")
    planned = [r for r in rows if r["status"] != "supported"]
    if planned:
        print("\nplanned adapters (convention unconfirmed, never written): "
              + ", ".join(r["id"] for r in planned))
    return 0


def _interactive_agents(spec: str, no_input: bool) -> str:
    """When run on a TTY without an explicit agent list, confirm the detected set."""
    if no_input or spec not in ("detected", "") or not sys.stdin.isatty():
        return spec or "detected"
    detected = [a for a in A.REGISTRY if a.status == "supported" and a.detect()]
    if not detected:
        print("No coding agents detected. Pass --agents <id,...> or --agents all.")
        return "detected"
    print("Detected agents: " + ", ".join(a.id for a in detected))
    ans = input("Register rf-mcp into these? [Y/n] ").strip().lower()
    return "detected" if ans in ("", "y", "yes") else ""


def run(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in VERSION_FLAGS:
        return diagnostics.cmd_version()
    args = build_parser().parse_args(argv)

    if args.command == "init":
        return diagnostics.cmd_init(browsers=args.browsers)
    if args.command == "doctor":
        return diagnostics.cmd_doctor(project_dir=args.project_dir)
    if args.command == "list":
        return _cmd_list()

    whats = [w.strip() for w in args.what.split(",") if w.strip()]
    proj = Path(args.project_dir).expanduser() if args.project_dir else None
    if proj is not None:
        from robotmcp.onboarding import project_env as _pe
        if not proj.exists():
            print(f"WARNING: --project-dir {proj} does not exist.")
        elif not _pe.looks_like_project(proj):
            print(f"WARNING: --project-dir {proj} has no project markers "
                  f"(pyproject/.git/.venv/*.robot); config will be written there anyway.")
    if args.command == "install":
        agents = _interactive_agents(args.agents, args.no_input)
        results = installer.install(
            agents=agents, scope=args.scope, whats=whats, dry_run=args.dry_run,
            force=args.force, command=args.command_override, env=_parse_env(args.env),
            cwd=proj, project_dir=proj, into_project=args.into_project,
            attach=args.attach, no_verify=args.no_verify)
        return _print_results(results)
    if args.command == "uninstall":
        results = installer.uninstall(agents=args.agents, scope=args.scope,
                                      whats=whats, dry_run=args.dry_run, cwd=proj)
        return _print_results(results)
    return 2

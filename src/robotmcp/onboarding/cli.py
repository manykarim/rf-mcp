"""Argparse front-end for the onboarding/installer subcommands.

Dispatched from ``robotmcp``'s ``main()`` BEFORE the server arg parser, so bare
``robotmcp`` still launches the MCP server and these subcommands never start it.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from robotmcp.onboarding import diagnostics, installer
from robotmcp.onboarding import adapters as A

SUBCOMMANDS = ("init", "install", "uninstall", "list", "doctor")
VERSION_FLAGS = ("--version", "-V")


class UsageError(Exception):
    """A bad argument value. Reported like argparse's own errors (exit 2)."""


def _parse_env(pairs: List[str]) -> Dict[str, str]:
    """Turn repeated ``KEY=VALUE`` --env flags into a dict.

    Malformed entries used to be dropped silently, so ``--env FOO`` produced a
    partial environment reported as a successful install (change:
    installer-cli-safety sec 3).
    """
    out: Dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise UsageError(f"--env {item!r} is not KEY=VALUE (missing '=').")
        k, v = item.split("=", 1)
        if not k.strip():
            raise UsageError(f"--env {item!r} has an empty key.")
        k = k.strip()
        if k in out and out[k] != v:
            raise UsageError(
                f"--env {k} given twice with different values "
                f"({out[k]!r} then {v!r}); pass it once."
            )
        out[k] = v
    return out


def _validate_whats(whats: List[str]) -> List[str]:
    """Reject unknown --what kinds instead of reporting them as 'no-assets'."""
    from robotmcp.onboarding import installer as _I

    if not whats:
        raise UsageError(f"--what is empty. Valid kinds: {', '.join(_I.WHAT_ALL)}.")
    unknown = [w for w in whats if w not in _I.WHAT_ALL]
    if unknown:
        raise UsageError(
            f"unknown --what kind(s): {', '.join(unknown)}. "
            f"Valid kinds: {', '.join(_I.WHAT_ALL)}."
        )
    return whats


def _validate_attach(attach: Optional[str]) -> Optional[str]:
    """Validate ``host[:port]``; an invalid port silently became the hostname before."""
    if attach in (None, "auto"):
        return attach
    value = attach.strip()
    if not value:
        raise UsageError("--attach was given an empty value; use host[:port] or omit it.")
    host, _, port = value.partition(":")
    if not host:
        raise UsageError(f"--attach {attach!r} has an empty host.")
    if _ and port == "":
        raise UsageError(f"--attach {attach!r} has an empty port after ':'.")
    if port:
        if not port.isdigit():
            raise UsageError(f"--attach port {port!r} is not a number.")
        if not (1 <= int(port) <= 65535):
            raise UsageError(f"--attach port {port} is out of range (1-65535).")
    return value


class _SuggestingParser(argparse.ArgumentParser):
    """argparse, plus a did-you-mean hint for a near-miss subcommand."""

    def error(self, message: str) -> None:  # type: ignore[override]
        import difflib

        m = re.search(r"invalid choice: '([^']+)'", message)
        if m:
            close = difflib.get_close_matches(m.group(1), SUBCOMMANDS, n=1, cutoff=0.6)
            if close:
                message += f"  (did you mean '{close[0]}'?)"
        super().error(message)


def build_parser() -> argparse.ArgumentParser:
    p = _SuggestingParser(
        prog="robotmcp",
        description="Install rf-mcp into coding agents and prepare the environment.",
        epilog="Run `robotmcp <command> --help` for a command's options. With no "
               "arguments, robotmcp starts the MCP server on stdin/stdout.",
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="{" + ",".join(SUBCOMMANDS) + "}")

    pi = sub.add_parser("init", help="Prepare the install and print the MCP config.")
    pi.add_argument("--browsers", action="store_true",
                    help="Initialize the Playwright browser (downloads a browser).")

    pd = sub.add_parser("doctor", help="Report installation health (read-only).")
    pd.add_argument("-C", "--project-dir", default=None,
                    help="Report which of THIS project's RF libraries rf-mcp can see.")
    pd.add_argument("--strict", action="store_true",
                    help="Exit non-zero when a checked capability is missing "
                         "(for CI; the default stays 0 so doctor is a plain report).")
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
        if not is_install:
            sp.add_argument("--force", action="store_true",
                            help="Remove an entry even if it was modified by hand "
                                 "(otherwise it is kept and reported).")
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


# Statuses that mean "the requested work did not happen". `_print_results` turns
# any of these into a non-zero exit so scripted installs can detect failure
# (change: installer-cli-safety sec 2). Everything else is a benign outcome.
FAILED_STATUSES = frozenset({"unverified", "error", "no-assets", "planned"})


def _print_results(results) -> int:
    if not results:
        # Previously "Nothing to do." + exit 0, which is what a typo'd --agents
        # produced. Selection errors now raise before reaching here, so an empty
        # result set means the selection resolved to no targets.
        print("Nothing to do: no matching agents. See `robotmcp list`.")
        return 1
    width = max(len(r.agent) for r in results)
    rc = 0
    for r in results:
        line = f"  {r.agent:<{width}}  {r.scope or '-':<7}  {r.what:<7}  {r.status}"
        if r.detail:
            line += f"  ({r.detail})"
        if r.path and r.status in ("installed", "updated", "removed",
                                   "would-install", "would-update", "would-remove"):
            line += f"  -> {r.path}"
        print(line)
        if r.status in FAILED_STATUSES:
            rc = 1
    return rc


def _cmd_list() -> int:
    rows = installer.list_agents()
    # The ID column is what `--agents` accepts. Without it the flag's valid values
    # appeared nowhere in the UI, so `--agents claude` (for `claude-code`) was an
    # easy mistake (change: installer-cli-safety sec 3).
    print(f"  {'ID':<14} {'AGENT':<22} {'STATUS':<10} {'DETECTED':<9} {'REGISTERED':<11} FORMAT")
    for r in rows:
        print(f"  {r['id']:<14} {r['name']:<22} {r['status']:<10} {r['detected']:<9} "
              f"{r['registered']:<11} {r['format']}")
    planned = [r for r in rows if r["status"] != "supported"]
    if planned:
        print("\nplanned adapters (convention unconfirmed, never written): "
              + ", ".join(r["id"] for r in planned))
    return 0


DECLINED = "\0declined"          # sentinel: the user said no at the prompt
NO_AGENTS_DETECTED = "\0none"    # sentinel: nothing to confirm in the first place


def _interactive_agents(spec: str, no_input: bool) -> str:
    """When run on a TTY without an explicit agent list, confirm the detected set.

    Returns a sentinel rather than ``""`` for a declined prompt: an empty spec used
    to be coerced back to ``detected`` by ``resolve_selection``, so answering "n"
    installed into every detected agent (change: installer-cli-safety sec 3b).
    """
    # NOTE: no `spec or "detected"` here. That fallback was a second copy of the
    # same coercion bug: with --yes, `--agents ""` still became "detected" even
    # after resolve_selection was fixed (change: installer-cli-safety sec 3b).
    if no_input or spec not in ("detected", "") or not sys.stdin.isatty():
        return spec
    detected = [a for a in A.REGISTRY if a.status == "supported" and a.detect()]
    if not detected:
        print("No coding agents detected. Pass --agents <id,...> or --agents all.")
        return NO_AGENTS_DETECTED
    print("Detected agents: " + ", ".join(a.id for a in detected))
    try:
        ans = input("Register rf-mcp into these? [Y/n] ").strip().lower()
    except EOFError:
        return DECLINED
    return "detected" if ans in ("", "y", "yes") else DECLINED


def _resolve_project_dir(raw: Optional[str]) -> Optional[Path]:
    """Validate --project-dir for EVERY subcommand that accepts it.

    The existence check used to live after the `doctor` branch had already
    returned, so `doctor -C /nonexistent` reported a clean result and exited 0.
    An empty value silently meant the CWD (change: installer-cli-safety sec 6).
    """
    if raw is None:
        return None
    if not raw.strip():
        raise UsageError("--project-dir was given an empty value; omit it to use the "
                         "current directory.")
    proj = Path(raw).expanduser()
    if proj.exists() and not proj.is_dir():
        raise UsageError(f"--project-dir {proj} is not a directory.")
    return proj


def _warn_project_dir(proj: Optional[Path]) -> None:
    if proj is None:
        return
    from robotmcp.onboarding import project_env as _pe
    if not proj.exists():
        print(f"WARNING: --project-dir {proj} does not exist.")
    elif not _pe.looks_like_project(proj):
        print(f"WARNING: --project-dir {proj} has no project markers "
              f"(pyproject/.git/.venv/*.robot); config will be written there anyway.")


def run(argv: Optional[List[str]] = None) -> int:
    try:
        return _run(argv)
    except (UsageError, A.UnknownAgentError, A.EmptySelectionError) as exc:
        print(f"robotmcp: error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        # Ctrl-C used to surface a raw traceback. Nothing is written before the
        # per-agent loop commits, so an interrupt leaves no partial state.
        print("\nrobotmcp: cancelled by user; nothing was changed.", file=sys.stderr)
        return 130


def _run(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in VERSION_FLAGS:
        return diagnostics.cmd_version()
    args = build_parser().parse_args(argv)

    if args.command == "init":
        return diagnostics.cmd_init(browsers=args.browsers)
    if args.command == "doctor":
        proj = _resolve_project_dir(args.project_dir)
        _warn_project_dir(proj)
        return diagnostics.cmd_doctor(project_dir=str(proj) if proj else None,
                                      strict=args.strict)
    if args.command == "list":
        return _cmd_list()

    whats = _validate_whats([w.strip() for w in args.what.split(",") if w.strip()])
    if args.agents is not None and not args.agents.strip():
        # Explicitly empty (e.g. `--agents "$UNSET_VAR"`). The default is
        # "detected"; an empty value must never silently mean "everything".
        raise A.EmptySelectionError()
    proj = _resolve_project_dir(args.project_dir)
    _warn_project_dir(proj)
    if args.command == "install":
        attach = _validate_attach(args.attach)
        if attach and args.command_override:
            raise UsageError("--attach and --command are mutually exclusive; "
                             "--command overrides the launch entirely.")
        env_extra = _parse_env(args.env)
        agents = _interactive_agents(args.agents, args.no_input)
        if agents == DECLINED:
            print("Cancelled: nothing was installed. "
                  "Re-run with --agents <id,...> to choose explicitly.")
            return 1
        if agents == NO_AGENTS_DETECTED:
            return 1
        results = installer.install(
            agents=agents, scope=args.scope, whats=whats, dry_run=args.dry_run,
            force=args.force, command=args.command_override, env=env_extra,
            cwd=proj, project_dir=proj, into_project=args.into_project,
            attach=attach, no_verify=args.no_verify)
        return _print_results(results)
    if args.command == "uninstall":
        results = installer.uninstall(agents=args.agents, scope=args.scope,
                                      whats=whats, dry_run=args.dry_run, cwd=proj,
                                      project_dir=proj, force=args.force)
        return _print_results(results)
    return 2

"""Console-script entry point for ``robotmcp`` / ``rf-mcp``.

Onboarding subcommands (``init``/``install``/``uninstall``/``list``/``doctor``) and
``--version`` are handled here WITHOUT importing the (heavy) MCP server module, so
they stay fast and their output is not polluted by server import-time diagnostics.
Anything else launches the MCP server.
"""
from __future__ import annotations

import sys
from typing import List, Optional

# Kept as literals so this module imports nothing heavy on the subcommand path.
_SUBCOMMANDS = frozenset({"init", "install", "uninstall", "list", "doctor"})
_VERSION_FLAGS = frozenset({"--version", "-V"})
_HELP_FLAGS = frozenset({"-h", "--help", "help"})

# Flags that belong to the SERVER parser. Anything else that is not a known
# subcommand is a user error (typically a typo) and must be diagnosed by the
# onboarding parser rather than falling through to the server, which printed a
# usage line with no subcommands, took ~6x longer, and emitted unrelated
# library-availability warnings first (change: installer-cli-safety sec 5/sec 7).
_SERVER_FLAG_PREFIXES = ("--with-frontend", "--without-frontend", "--transport",
                         "--host", "--port", "--path", "--log-level", "--frontend")


def _is_server_invocation(argv: List[str]) -> bool:
    """True when argv is empty (plain stdio server) or starts with a server flag."""
    if not argv:
        return True
    first = argv[0]
    return any(first == f or first.startswith(f + "=") for f in _SERVER_FLAG_PREFIXES)


def main(argv: Optional[List[str]] = None) -> None:
    _argv = list(sys.argv[1:] if argv is None else argv)
    if _argv and (_argv[0] in _SUBCOMMANDS or _argv[0] in _VERSION_FLAGS
                  or _argv[0] in _HELP_FLAGS or not _is_server_invocation(_argv)):
        from robotmcp.onboarding import run

        # `help` / `-h` / `--help` -> the onboarding parser's help, which lists the
        # subcommands. Previously these fell through to the server parser, so
        # `init`, `install`, `uninstall`, `list` and `doctor` were undiscoverable.
        if _argv[0] in _HELP_FLAGS:
            raise SystemExit(run(["--help"]))
        raise SystemExit(run(_argv))

    if not _argv and sys.stdin.isatty():
        # A curious user typing `robotmcp` got one banner line and a process that
        # blocked forever on stdin, with no way to discover the subcommands.
        print("robotmcp: starting the MCP server on stdin/stdout (Ctrl-C to stop).\n"
              "          This is meant to be launched by a coding agent, not a terminal.\n"
              "          Run `robotmcp --help` for setup commands "
              "(init, install, list, doctor).\n", file=sys.stderr)

    from robotmcp.server import main as server_main

    server_main(argv)


if __name__ == "__main__":
    main()

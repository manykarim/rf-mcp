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


def main(argv: Optional[List[str]] = None) -> None:
    _argv = list(sys.argv[1:] if argv is None else argv)
    if _argv and (_argv[0] in _SUBCOMMANDS or _argv[0] in _VERSION_FLAGS):
        from robotmcp.onboarding import run

        raise SystemExit(run(_argv))
    from robotmcp.server import main as server_main

    server_main(argv)


if __name__ == "__main__":
    main()

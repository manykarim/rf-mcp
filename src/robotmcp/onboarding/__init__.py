"""rf-mcp onboarding: tool-install diagnostics (`init`/`doctor`/`--version`) and the
multi-agent installer (`install`/`uninstall`/`list`).

Dispatched from ``robotmcp.server:main`` before the server parser so bare
``robotmcp`` still launches the MCP server.
"""
from robotmcp.onboarding.cli import SUBCOMMANDS, VERSION_FLAGS, run

__all__ = ["run", "SUBCOMMANDS", "VERSION_FLAGS"]

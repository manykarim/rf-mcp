"""Robot Framework MCP Server - Natural Language Test Automation Bridge."""

from robotmcp.attach.mcp_attach import McpAttach  # noqa: F401

# Expose Keywords from McpAttach at the package level

__all__ = ["McpAttach"]

from importlib import metadata

try:
    __version__ = metadata.version("rf-mcp")
except metadata.PackageNotFoundError:
    pass

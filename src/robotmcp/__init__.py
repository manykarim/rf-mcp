"""Robot Framework MCP Server - Natural Language Test Automation Bridge."""

from robotmcp.attach.mcp_attach import McpAttach  # noqa: F401

# Expose Keywords from McpAttach at the package level

__all__ = ["McpAttach"]

__version__ = "0.1.0"

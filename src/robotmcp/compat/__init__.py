"""Compatibility layer for supporting multiple fastmcp versions."""

from robotmcp.compat.fastmcp_compat import (
    DISABLED_TOOL_KWARGS,
    FASTMCP_V3,
    FASTMCP_VERSION,
    finalize_disabled_tools,
    get_tool_fn,
)

__all__ = [
    "DISABLED_TOOL_KWARGS",
    "FASTMCP_V3",
    "FASTMCP_VERSION",
    "finalize_disabled_tools",
    "get_tool_fn",
]

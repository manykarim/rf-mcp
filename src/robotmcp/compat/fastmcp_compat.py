"""FastMCP version compatibility layer.

Supports both fastmcp 2.8+ (enabled= parameter) and 3.0+ (mark-based visibility).

Version history:
- fastmcp 2.8.0: introduced ``enabled`` parameter on @mcp.tool()
- fastmcp 3.0.0: removed ``enabled``, replaced with mcp.disable()/mcp.enable()

Usage in server.py::

    from robotmcp.compat.fastmcp_compat import (
        DISABLED_TOOL_KWARGS, finalize_disabled_tools, get_tool_fn,
    )

    # Pattern A (single-line):
    @mcp.tool(**DISABLED_TOOL_KWARGS)
    async def discover_keywords(...): ...

    # Pattern B (multi-line):
    @mcp.tool(
        name="list_library_plugins",
        description="...",
        **DISABLED_TOOL_KWARGS,
    )
    async def list_library_plugins(...): ...

    # After all @mcp.tool() decorators, before main():
    finalize_disabled_tools(mcp, DISABLED_TOOL_NAMES)
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Any, Callable, Set

logger = logging.getLogger(__name__)

# ── Version detection ─────────────────────────────────────────────

try:
    _version_str = importlib.metadata.version("fastmcp")
    _major = int(_version_str.split(".")[0])
except Exception:
    _version_str = "0.0.0"
    _major = 0

FASTMCP_VERSION: str = _version_str
"""Installed fastmcp version string, e.g. '2.13.3' or '3.0.2'."""

FASTMCP_V3: bool = _major >= 3
"""True when running on fastmcp 3.x+."""

# ── Disabled tool support ─────────────────────────────────────────

DISABLED_TOOL_KWARGS: dict[str, Any] = {} if FASTMCP_V3 else {"enabled": False}
"""Kwargs to pass to @mcp.tool() for hiding a tool from clients.

On v2.8+: ``{'enabled': False}`` — tool is registered but filtered from tools/list.
On v3.0+: ``{}`` (empty) — tool is registered normally; disabled later via
``finalize_disabled_tools()``.
"""


def finalize_disabled_tools(mcp_server: Any, names: Set[str]) -> None:
    """Disable tools after all @mcp.tool() decorators have been processed.

    On v2.8+: no-op (``enabled=False`` already handled at registration time).
    On v3.0+: calls ``mcp_server.disable(names=...)`` to hide the tools.

    Must be called once after all tool registrations and before ``mcp.run()``.

    Args:
        mcp_server: The FastMCP server instance.
        names: Set of tool names to disable.
    """
    if not FASTMCP_V3:
        return  # v2: enabled=False already applied at decorator time

    if not names:
        return

    try:
        mcp_server.disable(names=names)
        logger.info("Disabled %d tools via fastmcp v3 API", len(names))
    except AttributeError:
        logger.warning(
            "fastmcp v3 detected but mcp.disable() not available. "
            "%d tools may be visible that should be hidden.",
            len(names),
        )
    except Exception as e:
        logger.warning("Failed to disable %d tools: %s", len(names), e)


# ── execute_step.fn compatibility ─────────────────────────────────

def get_tool_fn(tool_obj: Any) -> Callable[..., Any]:
    """Get the underlying async function from a FastMCP tool object.

    On v2: ``@mcp.tool`` returns a ``FunctionTool`` with a ``.fn`` attribute.
    On v3: ``@mcp.tool`` returns the original function directly.

    Args:
        tool_obj: The decorated tool (FunctionTool on v2, plain function on v3).

    Returns:
        The underlying async callable.
    """
    return getattr(tool_obj, "fn", tool_obj)


# ── Tool manager compatibility ────────────────────────────────────

class ToolManagerCompat:
    """Version-aware wrapper around FastMCP's internal tool management.

    On v2: delegates to ``server._tool_manager`` (private API).
    On v3: uses public ``server.disable()``, ``server.enable()``, etc.

    This class is used by the ToolManagerAdapter (ADR-006 tool profiles)
    to abstract away the version-specific tool management API.
    """

    def __init__(self, server: Any) -> None:
        self._server = server

    async def get_tools(self) -> dict[str, Any]:
        """Return all registered tools as {name: Tool}."""
        if not FASTMCP_V3:
            return await self._server._tool_manager.get_tools()
        # v3: try public API first, fall back to private
        if hasattr(self._server, "get_tools"):
            tools = self._server.get_tools()
            if hasattr(tools, "__await__"):
                tools = await tools
            return tools
        if hasattr(self._server, "_tool_manager"):
            return await self._server._tool_manager.get_tools()
        logger.warning("Cannot get tools: no compatible API found")
        return {}

    async def has_tool(self, name: str) -> bool:
        """Check if a tool is registered (regardless of enabled state)."""
        if not FASTMCP_V3:
            return await self._server._tool_manager.has_tool(name)
        tools = await self.get_tools()
        if isinstance(tools, dict):
            return name in tools
        return any(getattr(t, "name", None) == name for t in tools)

    def remove_tool(self, name: str) -> None:
        """Remove a tool from the registry.

        On v2: physically deletes from _tools dict.
        On v3: tries _tool_manager first, falls back to disable().
        """
        if not FASTMCP_V3:
            self._server._tool_manager.remove_tool(name)
            return
        if hasattr(self._server, "_tool_manager"):
            try:
                self._server._tool_manager.remove_tool(name)
                return
            except Exception:
                pass
        try:
            self._server.disable(names={name})
        except Exception as e:
            logger.warning("Failed to remove/disable tool '%s': %s", name, e)

    def add_tool(self, tool: Any) -> None:
        """Add a tool to the registry.

        On v2: inserts into _tools dict.
        On v3: tries _tool_manager first, falls back to enable().
        """
        if not FASTMCP_V3:
            self._server._tool_manager.add_tool(tool)
            return
        if hasattr(self._server, "_tool_manager"):
            try:
                self._server._tool_manager.add_tool(tool)
                return
            except Exception:
                pass
        tool_name = getattr(tool, "name", None) or getattr(tool, "key", None)
        if tool_name:
            try:
                self._server.enable(names={tool_name})
            except Exception as e:
                logger.warning("Failed to add/enable tool '%s': %s", tool_name, e)


# ── Lifespan compatibility ────────────────────────────────────────

def set_server_lifespan(mcp_server: Any, lifespan_fn: Any) -> None:
    """Set the lifespan context manager on the FastMCP server.

    On v2: mutates ``mcp._mcp_server.lifespan``.
    On v3: tries v2 path first, then public API if available.

    Args:
        mcp_server: The FastMCP server instance.
        lifespan_fn: An async context manager factory for the lifespan.
    """
    if hasattr(mcp_server, "_mcp_server"):
        mcp_server._mcp_server.lifespan = lifespan_fn  # type: ignore[attr-defined]
    elif hasattr(mcp_server, "lifespan"):
        mcp_server.lifespan = lifespan_fn
    else:
        logger.warning(
            "Cannot set lifespan: no compatible attribute found on FastMCP server"
        )

"""FastMCP Tool Manager Adapter - Anti-Corruption Layer.

This adapter isolates the domain from FastMCP's async ToolManager API.
It translates between the domain's ToolDescriptor model and FastMCP's
Tool objects. This follows the same pattern as FastMCPInstructionAdapter
in domains/instruction/adapters/fastmcp_adapter.py.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable, Dict, FrozenSet, Optional

logger = logging.getLogger(__name__)


class ToolManagerAdapter:
    """Anti-Corruption Layer wrapping FastMCP's ToolManager.

    FastMCP's ToolManager provides:
    - remove_tool(name: str)       -> synchronous, removes by name
    - add_tool(tool: Tool)         -> async, registers a Tool object
    - has_tool(name: str) -> bool  -> async check
    - get_tools() -> list[Tool]    -> async, returns all registered tools

    FastMCP does NOT have enable/disable per-tool. Dynamic profiles
    are achieved via remove_tool + add_tool (validated experimentally).

    After any add/remove, FastMCP automatically sends
    notifications/tools/list_changed to connected MCP clients.

    Description swapping works by removing the tool and re-adding it
    with a modified description. The tool's handler function reference
    is preserved via the _original_tools registry.

    Attributes:
        _server: The FastMCP server instance.
        _event_publisher: Optional callback for infrastructure events.
        _original_tools: Cache of original Tool objects for restoration.
    """

    def __init__(
        self,
        fastmcp_server: Any,    # FastMCP instance (typed as Any to avoid import)
        event_publisher: Optional[Callable[[object], None]] = None,
    ) -> None:
        """Initialize with a reference to the FastMCP server instance.

        Args:
            fastmcp_server: The module-level FastMCP() instance from server.py.
            event_publisher: Optional callback for infrastructure events.
        """
        self._server = fastmcp_server
        self._event_publisher = event_publisher
        # Cache original Tool objects so they can be re-added after removal.
        # Key: tool_name, Value: the original Tool object from FastMCP.
        self._original_tools: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """Snapshot all currently registered tools for later restoration.

        Must be called once at server startup, after all @mcp.tool
        decorators have been processed.
        """
        tools = await self._server._tool_manager.get_tools()
        # FastMCP get_tools() returns dict[str, Tool] or list[Tool]
        if isinstance(tools, dict):
            for name, tool in tools.items():
                self._original_tools[name] = tool
        else:
            for tool in tools:
                self._original_tools[tool.name] = tool
        logger.info(
            f"ToolManagerAdapter initialized with {len(self._original_tools)} tools"
        )

    async def remove_tool(self, tool_name: str) -> None:
        """Remove a tool from the visible set.

        Args:
            tool_name: Name of the tool to remove.
        """
        if await self._server._tool_manager.has_tool(tool_name):
            self._server._tool_manager.remove_tool(tool_name)
            logger.debug(f"Removed tool: {tool_name}")
        else:
            logger.debug(f"Tool '{tool_name}' already not visible, skip remove")

    async def add_tool_with_description(
        self, tool_name: str, description: str, schema: Dict
    ) -> None:
        """Add a tool back to the visible set with a specific description.

        If description differs from the original, creates a modified
        Tool object. Otherwise re-adds the original.

        Args:
            tool_name: Name of the tool to add.
            description: The description text to use.
            schema: The JSON schema to use for parameters.
        """
        original = self._original_tools.get(tool_name)
        if original is None:
            logger.warning(f"No original tool found for '{tool_name}', cannot add")
            return

        if not await self._server._tool_manager.has_tool(tool_name):
            # Create a modified copy with the desired description
            modified_tool = self._clone_tool_with_description(
                original, description, schema
            )
            self._server._tool_manager.add_tool(modified_tool)
            logger.debug(f"Added tool: {tool_name} (mode-specific description)")

    async def swap_tool_description(
        self, tool_name: str, description: str, schema: Dict
    ) -> None:
        """Swap a visible tool's description without changing its handler.

        Implemented as remove + add_with_description (atomic from the
        MCP client's perspective since the client only sees the final
        tools/list_changed notification).

        Args:
            tool_name: Name of the tool to swap.
            description: The new description text.
            schema: The new JSON schema.
        """
        await self.remove_tool(tool_name)
        await self.add_tool_with_description(tool_name, description, schema)

    async def get_visible_tool_names(self) -> FrozenSet[str]:
        """Return the set of currently visible tool names.

        Returns:
            FrozenSet of tool name strings.
        """
        tools = await self._server._tool_manager.get_tools()
        if isinstance(tools, dict):
            return frozenset(tools.keys())
        return frozenset(t.name for t in tools)

    async def restore_all(self) -> None:
        """Restore all original tools (used when switching to full profile).

        Forcefully replaces any modified tools (e.g., swapped descriptions)
        to ensure original schemas are intact.
        """
        for name, tool in self._original_tools.items():
            if await self._server._tool_manager.has_tool(name):
                self._server._tool_manager.remove_tool(name)
            self._server._tool_manager.add_tool(tool)
        logger.info("All original tools restored")

    def _clone_tool_with_description(
        self, original: Any, description: str, schema: Dict
    ) -> Any:
        """Create a modified copy of a FastMCP Tool with new description.

        Implementation note: Uses copy.copy() for shallow clone, then
        overrides description and parameters attributes. The handler
        function reference is preserved by the shallow copy.

        Args:
            original: The original FastMCP Tool object.
            description: The new description text.
            schema: The new JSON schema for parameters.

        Returns:
            A cloned Tool with modified description and schema.
        """
        cloned = copy.copy(original)
        cloned.description = description
        if schema and schema != getattr(original, 'parameters', None):
            cloned.parameters = schema
        return cloned

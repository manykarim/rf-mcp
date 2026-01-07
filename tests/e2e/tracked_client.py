"""MCP Client wrapper that automatically tracks tool calls to MetricsCollector."""

from typing import Any, Dict, Optional

from fastmcp import Client

from tests.e2e.metrics_collector import MetricsCollector


class TrackedMCPClient:
    """Wraps FastMCP Client to automatically record tool calls to MetricsCollector.

    This wrapper intercepts all `call_tool` invocations and records them
    to the provided MetricsCollector, enabling automatic metrics collection
    without requiring manual recording in test code.

    Example:
        async with Client(mcp) as raw_client:
            tracked = TrackedMCPClient(raw_client, metrics_collector)
            metrics_collector.start_recording()

            # Tool calls are automatically recorded
            result = await tracked.call_tool("analyze_scenario", {"scenario": "test"})

            metrics_collector.stop_recording()
            assert len(metrics_collector.tool_calls) == 1  # Works!
    """

    def __init__(self, client: Client, metrics_collector: MetricsCollector):
        """Initialize the tracked client.

        Args:
            client: The underlying FastMCP Client instance
            metrics_collector: MetricsCollector to record tool calls to
        """
        self._client = client
        self._metrics = metrics_collector

    async def call_tool(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Call a tool and automatically record metrics.

        Args:
            name: Name of the MCP tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool result from the MCP server

        Raises:
            Exception: Re-raises any exception from the tool call after recording
        """
        arguments = arguments or {}
        try:
            result = await self._client.call_tool(name, arguments)
            # Record successful call
            self._metrics.record_tool_call(
                tool_name=name,
                arguments=arguments,
                success=True,
                result=result.data if hasattr(result, "data") else {"success": True},
            )
            return result
        except Exception as e:
            # Record failed call
            self._metrics.record_tool_call(
                tool_name=name,
                arguments=arguments,
                success=False,
                error=str(e),
            )
            raise

    async def list_tools(self):
        """List available tools from the MCP server."""
        return await self._client.list_tools()

    async def list_prompts(self):
        """List available prompts from the MCP server."""
        return await self._client.list_prompts()

    async def list_resources(self):
        """List available resources from the MCP server."""
        return await self._client.list_resources()

    async def get_prompt(self, name: str, arguments: Optional[Dict[str, Any]] = None):
        """Get a prompt from the MCP server."""
        return await self._client.get_prompt(name, arguments)

    async def read_resource(self, uri: str):
        """Read a resource from the MCP server."""
        return await self._client.read_resource(uri)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped client.

        This allows the TrackedMCPClient to be used as a drop-in replacement
        for the regular Client in most cases.
        """
        return getattr(self._client, name)

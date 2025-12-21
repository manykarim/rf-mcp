"""Test fixtures for E2E AI agent testing."""

import os
import pytest
import pytest_asyncio
from typing import Optional, Dict, Any
from fastmcp import Client
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.test import TestModel

import robotmcp.server
from tests.e2e.metrics_collector import MetricsCollector


class MCPClientWithTracking:
    """MCP client wrapper that tracks tool calls for metrics."""

    def __init__(self, client: Client, metrics_collector: MetricsCollector):
        """Initialize the tracking client.

        Args:
            client: FastMCP client instance
            metrics_collector: Metrics collector to record tool calls
        """
        self._client = client
        self._metrics = metrics_collector

    async def call_tool(
        self, tool_name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Call a tool and record metrics.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool call result
        """
        args = arguments or {}
        success = False
        result = None
        error = None

        try:
            result = await self._client.call_tool(tool_name, args)
            success = True
            self._metrics.record_tool_call(
                tool_name=tool_name,
                arguments=args,
                success=success,
                result=result.data if hasattr(result, "data") else result,
            )
        except Exception as e:
            error = str(e)
            self._metrics.record_tool_call(
                tool_name=tool_name,
                arguments=args,
                success=success,
                error=error,
            )
            raise

        return result

    async def list_tools(self):
        """List available tools."""
        return await self._client.list_tools()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass


@pytest_asyncio.fixture
async def mcp_client():
    """Create an MCP client for testing."""
    mcp = robotmcp.server.mcp
    async with Client(mcp) as client:
        yield client


@pytest_asyncio.fixture
async def metrics_collector():
    """Create a metrics collector."""
    return MetricsCollector()


@pytest_asyncio.fixture
async def mcp_client_with_tracking(mcp_client, metrics_collector):
    """Create an MCP client with tool call tracking.

    Args:
        mcp_client: Base MCP client
        metrics_collector: Metrics collector

    Yields:
        MCPClientWithTracking instance
    """
    tracking_client = MCPClientWithTracking(mcp_client, metrics_collector)
    yield tracking_client


@pytest_asyncio.fixture
async def pydantic_agent():
    """Create a Pydantic AI agent for testing.

    Uses TestModel by default unless USE_REAL_LLM environment variable is set.

    Yields:
        Pydantic AI Agent instance
    """
    use_real_llm = os.getenv("USE_REAL_LLM", "false").lower() == "true"

    if use_real_llm:
        # Use real OpenAI model
        model = OpenAIModel("gpt-4o-mini")
    else:
        # Use TestModel for fast, deterministic testing
        model = TestModel()

    agent = Agent(
        model=model,
        system_prompt="""You are a test automation assistant using Robot Framework MCP server.
You have access to MCP tools for creating and executing Robot Framework tests.

When given a test scenario:
1. Use analyze_scenario to understand the requirements
2. Use execute_step to build the test step by step
3. Use build_test_suite to generate the final test suite

Always use the appropriate MCP tools to accomplish testing tasks.""",
    )

    yield agent

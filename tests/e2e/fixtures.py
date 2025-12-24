"""Test fixtures for E2E AI agent testing."""

import os
import pytest
import pytest_asyncio
from typing import Optional, Dict, Any
from fastmcp import Client, FastMCP
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.test import TestModel

import robotmcp.server
from tests.e2e.metrics_collector import MetricsCollector


@pytest_asyncio.fixture
async def mcp_server() -> FastMCP:
    """Provide the FastMCP server instance for testing.

    Returns:
        The robotmcp FastMCP server instance
    """
    return robotmcp.server.mcp


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
async def pydantic_agent():
    """Create a Pydantic AI agent for testing.

    Uses TestModel by default unless USE_REAL_LLM environment variable is set.

    Yields:
        Pydantic AI Agent instance
    """
    use_real_llm = os.getenv("USE_REAL_LLM", "false").lower() == "true"

    if use_real_llm:
        # Use real OpenAI model
        model = OpenAIModel("gpt-5-mini")
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

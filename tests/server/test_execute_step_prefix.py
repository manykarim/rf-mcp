import os
import sys

import pytest
import pytest_asyncio

# Ensure src path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from fastmcp import Client
from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_execute_step_accepts_library_prefix(mcp_client):
    """execute_step should accept library_prefix, but it currently rejects it."""
    await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Log",
            "arguments": ["Hi"],
            "session_id": "prefix_test",
            "library_prefix": "BuiltIn",
        },
    )

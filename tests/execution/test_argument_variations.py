import os
import sys

import pytest
import pytest_asyncio

# Ensure src path and library path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tests', 'libraries'))

from fastmcp import Client
from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_positional_and_named_arguments(mcp_client):
    # Positional argument containing '=' should not be treated as named
    res = await mcp_client.call_tool(
        "execute_step",
        {"keyword": "Log", "arguments": ["a=b"], "session_id": "s1"},
    )
    assert res.data["success"]

    # Named arguments with '=' in value via custom library
    res = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "UseKwArgs",
            "arguments": ["alpha=a=b", "beta=2"],
            "session_id": "s2",
            "library_prefix": "VarArgLib",
        },
    )
    assert "alpha=a=b" in res.data["output"]


@pytest.mark.asyncio
async def test_varargs_and_kwargs_with_custom_library(mcp_client):
    # *args handling with custom library
    res = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Concat",
            "arguments": ["a", "b=c"],
            "session_id": "s3",
            "library_prefix": "VarArgLib",
        },
    )
    assert res.data["output"] == "a,b=c"

    # Combination of positional, *args and named with '=' in value
    res = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Mix",
            "arguments": ["base", "mid", "opt=1=2"],
            "session_id": "s3",
            "library_prefix": "VarArgLib",
        },
    )
    assert res.data["output"].startswith("basemid")

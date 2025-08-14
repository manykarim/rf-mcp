import os
import sys
import importlib
import pytest
import pytest_asyncio

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
sys.path.insert(0, os.path.join(BASE_DIR, 'tests', 'libraries'))

# Ensure our stub Browser library is used even if real package was imported
importlib.invalidate_caches()
sys.modules.pop('Browser', None)

from fastmcp import Client
from fastmcp.exceptions import ToolError
from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_new_browser_positional_argument(mcp_client):
    with pytest.raises(ToolError) as exc:
        await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "New Browser",
                "arguments": ["chromium", "headless=False"],
                "session_id": "browser_positional",
                "library_prefix": "Browser",
            },
        )
    assert "takes 1 positional argument" in str(exc.value)


@pytest.mark.asyncio
async def test_new_browser_named_arguments(mcp_client):
    res = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "New Browser",
            "arguments": ["browser=chromium", "headless=False"],
            "session_id": "browser_named",
            "library_prefix": "Browser",
        },
    )
    assert res.data["success"]
    assert "'browser': 'chromium'" in res.data["output"]
    assert "'headless': False" in res.data["output"]


@pytest.mark.asyncio
async def test_new_browser_no_arguments(mcp_client):
    res = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "New Browser",
            "session_id": "browser_none",
            "library_prefix": "Browser",
        },
    )
    assert res.data["success"]
    assert res.data["output"] == "{'browser': None, 'headless': True}"

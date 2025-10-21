"""Argument parsing tests for Browser.New Browser keyword.

Ensures identifier=value pairs are treated as named args even when LibDoc
signature is not available (decorated/dynamic keywords), and mixed positional
and named combinations are handled.
"""

import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import mcp

from tests.utils.dependency_matrix import requires_extras

pytestmark = [
    requires_extras("web"),
    pytest.mark.optional_dependency("web"),
    pytest.mark.optional_web,
]


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_browser_new_browser_named_args(mcp_client):
    res = await mcp_client.call_tool(
        "debug_parse_keyword_arguments",
        {
            "keyword_name": "New Browser",
            "library_name": "Browser",
            "arguments": [
                "browser=chromium",
                "headless=False",
                "channel=edge",
            ],
        },
    )
    assert res.data.get("success") is True
    named = res.data.get("parsed", {}).get("named", {})
    assert named.get("browser") == "chromium"
    assert named.get("headless") in ("False", False)
    assert named.get("channel") == "edge"


@pytest.mark.asyncio
async def test_browser_new_browser_mixed_args(mcp_client):
    res = await mcp_client.call_tool(
        "debug_parse_keyword_arguments",
        {
            "keyword_name": "New Browser",
            "library_name": "Browser",
            "arguments": [
                "chromium",
                "headless=False",
            ],
        },
    )
    assert res.data.get("success") is True
    parsed = res.data.get("parsed", {})
    pos = parsed.get("positional", [])
    named = parsed.get("named", {})
    assert pos and pos[0] == "chromium"
    assert "headless" in named


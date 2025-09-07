"""Verify boolean handling for Browser.Click With Options in non-context path.

Skips if Browser library is unavailable. Uses headless to avoid UI requirements.
"""

import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_browser_click_with_options_boolean_args(mcp_client):
    # Check Browser availability
    avail = await mcp_client.call_tool(
        "check_library_availability", {"libraries": ["Browser"]}
    )
    available = set(avail.data.get("available_libraries", []))
    if "Browser" not in available:
        pytest.skip("Browser library not available in this environment")

    session_id = "browser_click_with_options_bool"

    # Init Browser session and search order
    await mcp_client.call_tool(
        "initialize_context",
        {"session_id": session_id, "libraries": ["Browser"]},
    )
    await mcp_client.call_tool(
        "set_library_search_order",
        {
            "libraries": ["Browser", "BuiltIn", "Collections", "String"],
            "session_id": session_id,
        },
    )

    # New headless browser and new page with simple content
    nb = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Browser.New Browser",
            "arguments": ["browser=chromium", "headless=True"],
            "session_id": session_id,
            "raise_on_failure": True,
        },
    )
    assert nb.data.get("success") is True

    # Data URL with a button
    html = "<html><head><title>T</title></head><body><button id='selectgold'>Click</button></body></html>"
    url = f"data:text/html,{html}"

    npg = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Browser.New Page",
            "arguments": [url],
            "session_id": session_id,
            "raise_on_failure": True,
        },
    )
    assert npg.data.get("success") is True

    # 1) force=True (string literal) should be converted to boolean True
    r1 = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Browser.Click With Options",
            "arguments": ["#selectgold", "force=True"],
            "session_id": session_id,
            "raise_on_failure": True,
        },
    )
    assert r1.data.get("success") is True

    # 2) force=${True} should resolve to Python True via built-ins
    r2 = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Browser.Click With Options",
            "arguments": ["#selectgold", "force=${True}"],
            "session_id": session_id,
            "raise_on_failure": True,
        },
    )
    assert r2.data.get("success") is True


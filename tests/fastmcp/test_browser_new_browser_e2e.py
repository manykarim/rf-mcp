"""E2E test for Browser.New Browser execution with headless=True.

Skips if Browser library is not available in the environment.
"""

import pytest
import pytest_asyncio

from fastmcp import Client
from fastmcp.exceptions import ToolError
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
async def test_browser_new_browser_e2e_headless(mcp_client):
    # Practical probe for Browser readiness
    probe_session = "browser_new_probe"
    try:
        await mcp_client.call_tool(
            "initialize_context",
            {"session_id": probe_session, "libraries": ["Browser"]},
        )
        await mcp_client.call_tool(
            "set_library_search_order",
            {
                "libraries": ["Browser", "BuiltIn", "Collections", "String"],
                "session_id": probe_session,
            },
        )
        res_probe = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Browser.New Browser",
                "arguments": ["browser=chromium", "headless=True"],
                "session_id": probe_session,
                "raise_on_failure": True,
            },
        )
        assert res_probe.data.get("success") is True
        await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Browser.Close Browser",
                "arguments": [],
                "session_id": probe_session,
                "raise_on_failure": True,
            },
        )
    except ToolError as e:
        pytest.skip(f"Browser not ready: {e}")

    session_id = "browser_headless_e2e"

    # Initialize session and ensure Browser is loaded and prioritized
    await mcp_client.call_tool(
        "initialize_context",
        {"session_id": session_id, "libraries": ["Browser"]},
    )
    await mcp_client.call_tool(
        "set_library_search_order",
        {"libraries": ["Browser", "BuiltIn", "Collections", "String"], "session_id": session_id},
    )

    # Start a headless browser
    new_browser = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Browser.New Browser",
            "arguments": ["browser=chromium", "headless=True"],
            "session_id": session_id,
            "raise_on_failure": True,
        },
    )
    assert new_browser.data.get("success") is True

    # Open a blank page to verify basic operation
    new_page = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Browser.New Page",
            "arguments": ["about:blank"],
            "session_id": session_id,
            "raise_on_failure": True,
        },
    )
    assert new_page.data.get("success") is True

    # Close Browser
    close = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Browser.Close Browser",
            "arguments": [],
            "session_id": session_id,
            "raise_on_failure": True,
        },
    )
    assert close.data.get("success") is True


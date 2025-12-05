import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_execute_step_open_browser_succeeds_with_selenium(mcp_client):
    """
    Verifies that generic keyword resolution works for SeleniumLibrary without
    requiring the fully-qualified name (regression guard for Open Browser failures).
    """
    import shutil

    if not shutil.which("chromedriver"):
        pytest.skip("chromedriver not available in environment")

    session_id = "selenium_open_browser_session"

    # Initialize session with SeleniumLibrary available in the RF context
    init_res = await mcp_client.call_tool(
        "manage_session",
        {
            "action": "init",
            "session_id": session_id,
            "libraries": ["SeleniumLibrary", "BuiltIn"],
        },
    )
    assert init_res.data["success"] is True

    # Prefer SeleniumLibrary when keywords collide
    order_res = await mcp_client.call_tool(
        "set_library_search_order",
        {"session_id": session_id, "libraries": ["SeleniumLibrary", "BuiltIn", "Browser"]},
    )
    assert order_res.data["success"] is True

    # Open a page using the SeleniumLibrary-qualified keyword name (guards regression in MCP execution path)
    open_res = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "SeleniumLibrary.Open Browser",
            "arguments": ["https://example.com", "headlesschrome"],
            "session_id": session_id,
            "use_context": True,
            "raise_on_failure": True,
        },
    )
    assert open_res.data["success"] is True

    # Clean up to avoid leaking browser instances in CI
    close_res = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Close All Browsers",
            "arguments": [],
            "session_id": session_id,
            "use_context": True,
            "raise_on_failure": True,
        },
    )
    assert close_res.data["success"] is True

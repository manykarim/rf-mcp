"""Validate get_page_source works in Browser sessions using RF context.

Skips if Browser library is not available.
"""

import pytest
import pytest_asyncio

from fastmcp import Client
from fastmcp.exceptions import ToolError

from tests.utils.dependency_matrix import requires_extras

pytestmark = [
    requires_extras("web"),
    pytest.mark.optional_dependency("web"),
    pytest.mark.optional_web,
]

from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_get_page_source_browser_rf_context_filtered(mcp_client):
    # Practical probe: try initializing and opening a headless browser quickly
    probe_session = "rfctx_browser_probe"
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
                "use_context": True,
                "raise_on_failure": True,
            },
        )
        assert res_probe.data.get("success") is True
        # Close immediately
        await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Browser.Close Browser",
                "arguments": [],
                "session_id": probe_session,
                "use_context": True,
                "raise_on_failure": True,
            },
        )
    except ToolError as e:
        pytest.skip(f"Browser not ready: {e}")

    session_id = "rfctx_browser_pagesource"

    # Initialize a session preferring Browser
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

    # Open Browser in RF context (use_context=True)
    res_new_browser = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Browser.New Browser",
            "arguments": ["browser=chromium", "headless=True"],
            "session_id": session_id,
            "use_context": True,
            "raise_on_failure": True,
        },
    )
    assert res_new_browser.data.get("success") is True

    # Open a data URL page so no external network is required
    data_url = "data:text/html,<html><head><title>RF</title></head><body><h1>Hello</h1></body></html>"
    res_new_page = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Browser.New Page",
            "arguments": [data_url],
            "session_id": session_id,
            "use_context": True,
            "raise_on_failure": True,
        },
    )
    assert res_new_page.data.get("success") is True

    # Now request page source via the tool with filtering enabled
    ps = await mcp_client.call_tool(
        "get_page_source",
        {
            "session_id": session_id,
            "filtered": True,
            "filtering_level": "standard",
        },
    )

    # Validate we got filtered DOM from the web path
    assert ps.data.get("success") is True
    assert ps.data.get("filtering_applied") is True
    assert ps.data.get("page_source_length", 0) > 0
    assert ps.data.get("filtered_page_source_length", 0) > 0
    aria_info = ps.data.get("aria_snapshot") or {}
    assert isinstance(aria_info, dict)
    if aria_info.get("success"):
        assert aria_info.get("content")
    else:
        # when keyword unavailable we expect an informative error or skip flag
        assert aria_info.get("error") or aria_info.get("skipped")
    # Title should be picked up from the data URL
    assert ps.data.get("page_title") in ("RF", None)  # Some environments may not parse title
    # Ensure it did not route to mobile error
    assert "AppiumLibrary" not in str(ps.data)

    # Optional: retrieve full filtered source
    ps_full = await mcp_client.call_tool(
        "get_page_source",
        {
            "session_id": session_id,
            "filtered": True,
            "filtering_level": "standard",
            "full_source": True,
        },
    )
    assert ps_full.data.get("success") is True
    assert isinstance(ps_full.data.get("page_source"), str)
    assert len(ps_full.data.get("page_source")) == ps_full.data.get(
        "filtered_page_source_length"
    )
    aria_full = ps_full.data.get("aria_snapshot") or {}
    assert isinstance(aria_full, dict)
    if aria_full.get("success"):
        assert aria_full.get("content")
    else:
        assert aria_full.get("error") or aria_full.get("skipped")

    # Allow callers to skip reduced DOM collection explicitly
    ps_no_reduced = await mcp_client.call_tool(
        "get_page_source",
        {
            "session_id": session_id,
            "include_reduced_dom": False,
        },
    )
    assert ps_no_reduced.data.get("success") is True
    aria_skipped = ps_no_reduced.data.get("aria_snapshot") or {}
    assert aria_skipped.get("skipped") is True

    # Validate application state reflects the snapshot metadata when available
    app_state = await mcp_client.call_tool(
        "get_application_state",
        {
            "session_id": session_id,
            "state_type": "dom",
        },
    )
    assert app_state.data.get("success") is True
    dom_info = (app_state.data.get("dom") or {})
    if isinstance(dom_info, dict) and dom_info.get("aria_snapshot"):
        dom_snapshot = dom_info["aria_snapshot"]
        assert dom_snapshot.get("content")
        assert dom_snapshot.get("selector")

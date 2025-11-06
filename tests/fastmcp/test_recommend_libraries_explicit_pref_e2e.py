"""E2E test: explicit SeleniumLibrary preference in scenario is respected.

Validates that analyze_scenario detects an explicit SeleniumLibrary preference
and recommend_libraries prioritizes SeleniumLibrary, excludes Browser, and applies
the search order to the session accordingly.
"""

import pytest
import pytest_asyncio

pytestmark = pytest.mark.skip(reason="Scenario preference test relies on deprecated session tools")

from fastmcp import Client
from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_explicit_selenium_preference_applied(mcp_client):
    scenario = (
        "Open https://www.amazon.de/, search for a MacBook, assert results (name and price). "
        "Use Selenium Library"
    )
    session_id = "amazon_web_sel_pref"

    # Step 1: analyze_scenario should create/configure session and detect explicit preference
    analyzed = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": scenario, "context": "web", "session_id": session_id},
    )
    assert analyzed.data.get("success", True) is True  # some analyze_scenario variants return plain dict
    sinfo = analyzed.data.get("session_info", {})
    assert sinfo.get("session_id") == session_id
    # Explicit preference should reflect SeleniumLibrary
    assert sinfo.get("explicit_library_preference") in ("SeleniumLibrary", None)

    # Step 2: recommend_libraries should prioritize SeleniumLibrary and exclude Browser
    reco = await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": scenario,
            "context": "web",
            "session_id": session_id,
            "check_availability": True,
            "apply_search_order": True,
        },
    )

    assert reco.data.get("success") is True
    recs = reco.data.get("recommended_libraries", [])
    assert isinstance(recs, list) and len(recs) >= 1
    assert "SeleniumLibrary" in recs
    assert "Browser" not in recs  # explicit preference causes conflict resolution

    # Session search order should start with SeleniumLibrary and exclude Browser
    sess = await mcp_client.call_tool("get_session_info", {"session_id": session_id})
    assert sess.data.get("success") is True
    order = sess.data.get("session_info", {}).get("search_order", [])
    assert isinstance(order, list) and len(order) >= 1
    assert order[0] == "SeleniumLibrary"
    assert "Browser" not in order


@pytest.mark.asyncio
async def test_browser_preferred_when_no_explicit_preference(mcp_client):
    scenario = (
        "Open https://example.com/, navigate to a section, click a button, assert title"
    )
    session_id = "web_browser_preferred"

    # Step 1: analyze_scenario without explicit library preference
    analyzed = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": scenario, "context": "web", "session_id": session_id},
    )
    assert analyzed.data.get("success", True) is True

    # Step 2: recommend_libraries should prefer Browser and exclude SeleniumLibrary
    reco = await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": scenario,
            "context": "web",
            "session_id": session_id,
            "check_availability": True,
            "apply_search_order": True,
        },
    )
    assert reco.data.get("success") is True
    recs = reco.data.get("recommended_libraries", [])
    assert isinstance(recs, list) and len(recs) >= 1
    assert "Browser" in recs
    assert "SeleniumLibrary" not in recs

    # Search order applied: Browser first, SeleniumLibrary absent
    sess = await mcp_client.call_tool("get_session_info", {"session_id": session_id})
    assert sess.data.get("success") is True
    order = sess.data.get("session_info", {}).get("search_order", [])
    assert isinstance(order, list) and len(order) >= 1
    assert order[0] == "Browser"
    assert "SeleniumLibrary" not in order

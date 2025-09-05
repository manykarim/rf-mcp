"""Mobile scenario tests: recommender excludes web libs and routing uses AppiumLibrary.

These tests do not require a running Appium server. They focus on:
- recommend_libraries excluding SeleniumLibrary/Browser for mobile context
- execute_step routing on mobile sessions not invoking SeleniumLibrary (i.e., no 'No browser is open')
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
async def test_mobile_recommender_excludes_web_libs(mcp_client):
    scenario = (
        "Open the mobile application tests/appium/SauceLabs.apk, perform actions, assert UI; Appium at http://localhost:4723"
    )
    res = await mcp_client.call_tool(
        "recommend_libraries",
        {"scenario": scenario, "context": "mobile", "check_availability": False, "apply_search_order": False},
    )
    assert res.data.get("success") is True
    recs = res.data.get("recommended_libraries", [])
    assert "AppiumLibrary" in recs
    assert "SeleniumLibrary" not in recs
    assert "Browser" not in recs


@pytest.mark.asyncio
async def test_mobile_routing_does_not_use_selenium(mcp_client):
    session_id = "mobile_routing"
    # Initialize session with AppiumLibrary so it's loaded and in profile
    await mcp_client.call_tool(
        "initialize_context",
        {"session_id": session_id, "libraries": ["AppiumLibrary"]},
    )
    # Ensure AppiumLibrary is first for mobile session
    await mcp_client.call_tool(
        "set_library_search_order",
        {"libraries": ["AppiumLibrary", "BuiltIn", "Collections", "String"], "session_id": session_id},
    )

    # Attempt a generic input action using Appium-style locator
    res = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Input Text",
            "arguments": ["accessibility_id=test-Username", "standard_user"],
            "session_id": session_id,
            "raise_on_failure": False,
        },
    )

    data = res.data
    # If it fails due to lack of server/device, it must not fail with Selenium error
    if data.get("success") is False:
        err = (data.get("error") or "").lower()
        assert "selenium" not in err
        assert "no browser is open" not in err

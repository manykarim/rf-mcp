"""Integration tests validating library preference handling via the new tool surface."""

from __future__ import annotations

import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


def _extract_summary(state_result: dict) -> dict:
    summary = state_result["sections"].get("summary")
    assert summary and summary.get("success") is True
    info = summary.get("session_info", {})
    assert info
    return info


@pytest.mark.asyncio
async def test_analyze_scenario_selenium_preference(mcp_client):
    scenario = "Use SeleniumLibrary for classic web automation"
    result = await mcp_client.call_tool(
        "analyze_scenario", {"scenario": scenario, "context": "web"}
    )
    assert result.data["success"] is True

    analysis = result.data["analysis"]
    assert analysis["explicit_library_preference"] == "SeleniumLibrary"
    assert analysis["detected_session_type"] == "web_automation"

    session_info = result.data["session_info"]
    assert session_info["explicit_library_preference"] == "SeleniumLibrary"
    assert session_info["search_order"][0] == "SeleniumLibrary"
    assert "Browser" not in session_info["recommended_libraries"]


@pytest.mark.asyncio
async def test_analyze_scenario_browser_preference(mcp_client):
    scenario = "Use Browser Library for modern cross-browser automation"
    result = await mcp_client.call_tool(
        "analyze_scenario", {"scenario": scenario, "context": "web"}
    )
    assert result.data["success"] is True

    analysis = result.data["analysis"]
    assert analysis["explicit_library_preference"] == "Browser"

    session_info = result.data["session_info"]
    assert session_info["explicit_library_preference"] == "Browser"
    assert session_info["search_order"][0] == "Browser"
    assert "SeleniumLibrary" not in session_info["recommended_libraries"]


@pytest.mark.asyncio
async def test_analyze_scenario_no_preference(mcp_client):
    scenario = "Create an automation suite without naming any library"
    result = await mcp_client.call_tool(
        "analyze_scenario", {"scenario": scenario, "context": "web"}
    )
    assert result.data["success"] is True
    session_info = result.data["session_info"]
    assert session_info["explicit_library_preference"] is None
    assert session_info["auto_configured"] is True


@pytest.mark.asyncio
async def test_execute_step_respects_preference(mcp_client):
    scenario = "Use SeleniumLibrary for web automation"
    analyze = await mcp_client.call_tool(
        "analyze_scenario", {"scenario": scenario, "context": "web"}
    )
    session_id = analyze.data["session_id"]

    result = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Log",
            "arguments": ["Preference check"],
            "session_id": session_id,
        },
    )
    assert result.data["success"] is True

    state = await mcp_client.call_tool(
        "get_session_state",
        {"session_id": session_id, "sections": ["summary"]},
    )
    summary = _extract_summary(state.data)
    assert summary["explicit_library_preference"] == "SeleniumLibrary"
    assert summary["search_order"][0] == "SeleniumLibrary"


@pytest.mark.asyncio
async def test_validation_section_available(mcp_client):
    scenario = "Use SeleniumLibrary to validate login"
    analyze = await mcp_client.call_tool(
        "analyze_scenario", {"scenario": scenario, "context": "web"}
    )
    session_id = analyze.data["session_id"]

    await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Log",
            "arguments": ["validation"],
            "session_id": session_id,
        },
    )

    state = await mcp_client.call_tool(
        "get_session_state",
        {"session_id": session_id, "sections": ["summary", "validation"]},
    )
    summary = _extract_summary(state.data)
    assert summary["session_id"] == session_id
    validation = state.data["sections"].get("validation")
    assert validation and validation.get("success") is True


@pytest.mark.asyncio
async def test_multiple_sessions_isolated(mcp_client):
    selenium = await mcp_client.call_tool(
        "analyze_scenario", {"scenario": "Use SeleniumLibrary", "context": "web"}
    )
    browser = await mcp_client.call_tool(
        "analyze_scenario", {"scenario": "Use Browser Library", "context": "web"}
    )

    selenium_id = selenium.data["session_id"]
    browser_id = browser.data["session_id"]
    assert selenium_id != browser_id

    selenium_state = await mcp_client.call_tool(
        "get_session_state",
        {"session_id": selenium_id, "sections": ["summary"]},
    )
    browser_state = await mcp_client.call_tool(
        "get_session_state",
        {"session_id": browser_id, "sections": ["summary"]},
    )

    selenium_summary = _extract_summary(selenium_state.data)
    browser_summary = _extract_summary(browser_state.data)

    assert selenium_summary["explicit_library_preference"] == "SeleniumLibrary"
    assert browser_summary["explicit_library_preference"] == "Browser"
    assert "Browser" not in selenium_summary.get("search_order", [])
    assert "SeleniumLibrary" not in browser_summary.get("search_order", [])

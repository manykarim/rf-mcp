"""Comprehensive FastMCP E2E tests for the recommend_libraries tool.

Covers multiple scenario types (web, api, data), availability payload, and
optional session search order application.
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
async def test_recommend_libraries_web_scenario(mcp_client):
    scenario = (
        "Open a web page, fill a login form, click submit, assert title and logout"
    )
    res = await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": scenario,
            "context": "web",
            "max_recommendations": 5,
            "check_availability": True,
            "apply_search_order": False,
        },
    )

    assert res.data.get("success") is True
    recs = res.data.get("recommended_libraries", [])
    assert isinstance(recs, list) and len(recs) >= 1
    # Expect Browser or SeleniumLibrary for web
    assert any(lib in recs for lib in ("Browser", "SeleniumLibrary"))
    # Availability included
    assert isinstance(res.data.get("availability"), dict)
    # Should not include mobile, system, or database libs by default for web context
    assert "AppiumLibrary" not in recs
    assert "DatabaseLibrary" not in recs
    assert "SSHLibrary" not in recs
    assert "OperatingSystem" not in recs
    assert "Process" not in recs


@pytest.mark.asyncio
async def test_recommend_libraries_api_scenario(mcp_client):
    scenario = "Call a REST API, GET list of items and verify status codes"
    res = await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": scenario,
            "context": "api",
            "max_recommendations": 5,
            "check_availability": True,
            "apply_search_order": False,
        },
    )

    assert res.data.get("success") is True
    recs = res.data.get("recommended_libraries", [])
    assert isinstance(recs, list) and len(recs) >= 1
    # Expect RequestsLibrary for API scenarios
    assert "RequestsLibrary" in recs


@pytest.mark.asyncio
async def test_recommend_libraries_data_scenario(mcp_client):
    scenario = "Parse XML, filter elements, and operate on lists and strings"
    res = await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": scenario,
            "context": "data",
            "max_recommendations": 5,
            "check_availability": True,
            "apply_search_order": False,
        },
    )

    assert res.data.get("success") is True
    recs = res.data.get("recommended_libraries", [])
    assert isinstance(recs, list) and len(recs) >= 1
    # Expect data-focused libraries to be present
    assert any(lib in recs for lib in ("XML", "Collections", "String"))


@pytest.mark.asyncio
async def test_recommend_libraries_applies_search_order(mcp_client):
    session_id = "recommend_apply_order"
    scenario = "Web UI: navigate, click buttons, assert elements present"
    res = await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": scenario,
            "context": "web",
            "max_recommendations": 4,
            "session_id": session_id,
            "check_availability": True,
            "apply_search_order": True,
        },
    )

    assert res.data.get("success") is True
    recs = res.data.get("recommended_libraries", [])
    assert isinstance(recs, list) and len(recs) >= 1
    assert isinstance(res.data.get("session_setup"), dict)

    # Confirm session search order contains some recommended libraries
    sess = await mcp_client.call_tool("get_session_info", {"session_id": session_id})
    assert sess.data.get("success") is True
    order = sess.data.get("session_info", {}).get("search_order", [])
    assert any(lib in order for lib in recs)

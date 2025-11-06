"""Smoke tests validating optional dependency combinations."""

from __future__ import annotations

import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import mcp

from tests.utils.dependency_matrix import (
    requires_combination,
    requires_extras,
)


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
@pytest.mark.optional_dependency("web")
@pytest.mark.optional_web
@requires_extras("web")
async def test_web_recommendations_include_browser_and_selenium(mcp_client):
    scenario = "Open a dashboard in the browser and verify widgets"
    result = await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": scenario,
            "context": "web",
            "check_availability": True,
            "max_recommendations": 5,
        },
    )

    assert result.data.get("success") is True
    recs = result.data.get("recommended_libraries", [])
    assert any(lib in recs for lib in ("Browser", "SeleniumLibrary"))


@pytest.mark.asyncio
@pytest.mark.optional_dependency("api")
@pytest.mark.optional_api
@requires_extras("api")
async def test_api_session_keyword_available(mcp_client):
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "Exercise API workflow", "context": "api"},
    )
    session_id = analyze.data["session_info"]["session_id"]

    create_session = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Create Session",
            "arguments": ["api", "https://example.com"],
            "session_id": session_id,
            "raise_on_failure": True,
        },
        timeout=30,
    )
    assert create_session.data.get("success") is True


@pytest.mark.asyncio
@pytest.mark.optional_dependency("mobile")
@pytest.mark.optional_mobile
@requires_extras("mobile")
async def test_mobile_recommendations_prefer_appium(mcp_client):
    scenario = "Automate login on a native mobile app"
    result = await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": scenario,
            "context": "mobile",
            "check_availability": True,
            "max_recommendations": 5,
        },
    )

    assert result.data.get("success") is True
    recs = result.data.get("recommended_libraries", [])
    assert "AppiumLibrary" in recs
    assert "SeleniumLibrary" not in recs


@pytest.mark.asyncio
@pytest.mark.optional_dependency("database")
@pytest.mark.optional_database
@requires_extras("database")
async def test_database_library_initialization(mcp_client):
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "Validate data warehouse", "context": "database"},
    )
    session_id = analyze.data["session_info"]["session_id"]

    # Initialize DB session using SQLite (in-memory).
    connect = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "DatabaseLibrary.Connect To Database Using Custom Params",
            "arguments": [
                "sqlite3",
                "database=':memory:'",
            ],
            "session_id": session_id,
            "raise_on_failure": True,
        },
        timeout=30,
    )
    assert connect.data.get("success") is True


@pytest.mark.asyncio
@pytest.mark.optional_dependency("web+api")
@pytest.mark.optional_web_api
@requires_combination("web+api")
async def test_web_and_api_recommendations_respected(mcp_client):
    scenario = "Validate SPA uses REST API"
    result = await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": scenario,
            "context": "web",
            "check_availability": True,
            "max_recommendations": 6,
        },
    )

    assert result.data.get("success") is True
    recs = set(result.data.get("recommended_libraries", []))
    assert "Browser" in recs or "SeleniumLibrary" in recs
    assert "RequestsLibrary" in recs


@pytest.mark.asyncio
@pytest.mark.optional_dependency("web+mobile")
@pytest.mark.optional_web_mobile
@requires_combination("web+mobile")
async def test_web_and_mobile_profiles_stay_isolated(mcp_client):
    web_session = "combo_web_session"
    mobile_session = "combo_mobile_session"

    await mcp_client.call_tool(
        "manage_session",
        {"action": "init", "session_id": web_session, "libraries": ["Browser"]},
    )
    await mcp_client.call_tool(
        "manage_session",
        {"action": "init", "session_id": mobile_session, "libraries": ["AppiumLibrary"]},
    )

    web_state = await mcp_client.call_tool(
        "get_session_state",
        {"session_id": web_session, "sections": ["summary"]},
    )
    mobile_state = await mcp_client.call_tool(
        "get_session_state",
        {"session_id": mobile_session, "sections": ["summary"]},
    )

    web_imports = web_state.data["sections"].get("summary", {}).get("session_info", {}).get("imported_libraries", [])
    mobile_imports = mobile_state.data["sections"].get("summary", {}).get("session_info", {}).get("imported_libraries", [])

    assert "Browser" in web_imports
    assert "AppiumLibrary" in mobile_imports
    assert "AppiumLibrary" not in web_imports
    assert "Browser" not in mobile_imports


@pytest.mark.asyncio
@pytest.mark.optional_dependency("api+database")
@pytest.mark.optional_api_database
@requires_combination("api+database")
async def test_api_and_database_recommendations(mcp_client):
    scenario = "Sync API response into database"
    result = await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": scenario,
            "context": "api",
            "check_availability": True,
            "max_recommendations": 6,
        },
    )

    recs = set(result.data.get("recommended_libraries", []))
    assert "RequestsLibrary" in recs
    assert "DatabaseLibrary" in recs


@pytest.mark.asyncio
@pytest.mark.optional_dependency("all")
@pytest.mark.optional_all
@requires_combination("all")
async def test_all_combination_libraries_import(mcp_client):
    libraries_to_check = {
        "Browser": ["Browser"],
        "SeleniumLibrary": ["SeleniumLibrary"],
        "AppiumLibrary": ["AppiumLibrary"],
        "RequestsLibrary": ["RequestsLibrary"],
        "DatabaseLibrary": ["DatabaseLibrary"],
    }

    for library, libs in libraries_to_check.items():
        session_id = f"combo_{library.lower()}"
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": session_id,
                "libraries": libs,
            },
        )
        info = await mcp_client.call_tool(
            "get_session_state",
            {"session_id": session_id, "sections": ["summary"]},
        )
        imported = info.data["sections"].get("summary", {}).get("session_info", {}).get("imported_libraries", [])
        assert library in imported


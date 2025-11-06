"""End-to-end coverage for session lifecycle and variable resolution."""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.models.session_models import PlatformType
from robotmcp.server import execution_engine, mcp


def _unique_session_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_session_manager_lifecycle_via_mcp(mcp_client):
    session_id = _unique_session_id("session-lifecycle")
    manager = execution_engine.session_manager

    initial_ids = set(manager.get_all_session_ids())
    assert manager.get_session(session_id) is None

    init = await mcp_client.call_tool(
        "manage_session",
        {
            "action": "init",
            "session_id": session_id,
            "libraries": ["BuiltIn"],
            "variables": {"GREETING": "hello"},
        },
    )
    assert init.data["success"] is True
    assert "BuiltIn" in init.data["libraries_loaded"]

    session = manager.get_session(session_id)
    assert session is not None
    assert manager.get_session_count() == len(initial_ids) + 1

    manager.apply_state_updates(
        session,
        {
            "current_browser": {"type": "chromium"},
            "current_context": {"id": "ctx-1"},
            "current_page": {"url": "https://example.org", "id": "page-1"},
            "variables": {"DYNAMIC": "value"},
        },
    )
    info = manager.get_session_info(session_id)
    assert info is not None
    assert info["current_url"] == "https://example.org"
    assert info["browser_type"] == "chromium"
    assert info["variables_count"] >= 1

    assert manager.detect_platform_from_scenario("Test a REST API endpoint") == PlatformType.API
    assert manager.detect_platform_from_scenario("Launch Android mobile banking app") == PlatformType.MOBILE

    manager.initialize_mobile_session(
        session,
        scenario="Open Android app com.example.bank/.MainActivity on emulator",
    )
    assert session.is_mobile_session() is True
    assert session.mobile_config is not None
    assert session.mobile_config.platform_name == "Android"
    assert "AppiumLibrary" in session.imported_libraries

    parsed_config = manager.parse_mobile_requirements(
        "Install Android app com.demo.app/.MainActivity from /tmp/demo.apk on emulator"
    )
    assert parsed_config.platform_name == "Android"
    assert parsed_config.app_package == "com.demo.app"
    assert parsed_config.app_activity == ".MainActivity" or parsed_config.app_activity is None
    assert parsed_config.app_path.endswith("demo.apk")

    # Ensure cleanup removes inactive sessions
    stale_id = _unique_session_id("stale")
    stale_session = manager.create_session(stale_id)
    stale_session.last_activity = stale_session.last_activity - timedelta(
        seconds=manager.config.SESSION_CLEANUP_TIMEOUT + 10
    )
    cleaned = manager.cleanup_expired_sessions()
    assert cleaned >= 1
    assert manager.get_session(stale_id) is None

    assert manager.remove_session(session_id) is True
    assert manager.get_session(session_id) is None
    assert session_id not in manager.get_all_session_ids()


@pytest.mark.asyncio
async def test_variable_resolution_end_to_end(mcp_client):
    session_id = _unique_session_id("variables")
    manager = execution_engine.session_manager

    await mcp_client.call_tool(
        "manage_session",
        {
            "action": "init",
            "session_id": session_id,
            "libraries": ["BuiltIn", "Collections"],
            "variables": {"BASE_URL": "https://example.test"},
        },
    )

    set_vars = await mcp_client.call_tool(
        "manage_session",
        {
            "action": "set_variables",
            "session_id": session_id,
            "variables": ["ENDPOINT=/status", "TOKEN=xyz123"],
        },
    )
    assert set_vars.data["success"] is True
    assert "ENDPOINT" in set_vars.data["set"]

    dictionary_build = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Create Dictionary",
            "arguments": [
                "base=${BASE_URL}",
                "url=${BASE_URL}${ENDPOINT}",
                "token=${TOKEN}",
            ],
            "session_id": session_id,
            "assign_to": "API_INFO",
            "use_context": True,
        },
    )
    assert dictionary_build.data["success"] is True

    full_url = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "",
            "session_id": session_id,
            "mode": "evaluate",
            "expression": "$API_INFO['url']",
            "assign_to": "FULL_URL",
            "use_context": True,
        },
    )
    assert full_url.data["success"] is True

    multi_assign = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "",
            "session_id": session_id,
            "mode": "evaluate",
            "expression": "($API_INFO['base'], $API_INFO['token'])",
            "assign_to": ["BASE_COPY", "TOKEN_COPY"],
            "use_context": True,
        },
    )
    assert multi_assign.data["success"] is True

    token_lookup = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Get Variable Value",
            "arguments": ["${TOKEN_COPY}"],
            "session_id": session_id,
            "use_context": True,
        },
    )
    assert token_lookup.data["success"] is True
    assert token_lookup.data.get("output") == "xyz123"

    state = await mcp_client.call_tool(
        "get_session_state",
        {"session_id": session_id, "sections": ["variables"]},
    )
    variables_section = state.data["sections"]["variables"]
    assert variables_section["success"] is True
    variables_map = variables_section["variables"]
    assert variables_map["FULL_URL"].endswith("/status")
    assert variables_map["BASE_COPY"] == "https://example.test"

    assert manager.remove_session(session_id) is True

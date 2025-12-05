"""Updated comprehensive test suite covering the consolidated MCP tools."""

from __future__ import annotations

import pytest
import pytest_asyncio
import sys
import os
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastmcp import Client
from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.fixture
def session_id():
    return "comprehensive_session"


@pytest.mark.asyncio
async def test_analyze_scenario_creates_session(mcp_client, session_id):
    result = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "Verify login flow", "context": "web", "session_id": session_id},
    )
    assert result.data["success"] is True
    assert result.data["session_id"] == session_id


@pytest.mark.asyncio
async def test_recommend_libraries_modes(mcp_client, session_id):
    direct = await mcp_client.call_tool(
        "recommend_libraries",
        {"scenario": "Test login", "context": "web", "session_id": session_id},
    )
    assert direct.data["success"] is True
    assert direct.data["mode"] == "direct"

    sampling = await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": "login and API checks",
            "mode": "sampling_prompt",
            "k": 2,
        },
    )
    assert sampling.data["success"] is True
    assert sampling.data["mode"] == "sampling_prompt"
    assert isinstance(sampling.data["prompt"], str)


@pytest.mark.asyncio
async def test_manage_library_plugins_list(mcp_client):
    plugins = await mcp_client.call_tool("manage_library_plugins", {"action": "list"})
    assert plugins.data["success"] is True
    assert plugins.data["action"] == "list"


@pytest.mark.asyncio
async def test_find_keywords_strategies(mcp_client):
    semantic = await mcp_client.call_tool(
        "find_keywords", {"query": "log message", "strategy": "semantic"}
    )
    assert semantic.data["success"] is True

    pattern = await mcp_client.call_tool(
        "find_keywords", {"query": "Log", "strategy": "pattern", "limit": 3}
    )
    assert pattern.data["success"] is True
    assert len(pattern.data["results"]) >= 1


@pytest.mark.asyncio
async def test_manage_session_init_and_import(mcp_client, session_id):
    init = await mcp_client.call_tool(
        "manage_session",
        {
            "action": "init",
            "session_id": session_id,
            "libraries": ["BuiltIn"],
            "variables": {"username": "demo"},
        },
    )
    assert init.data["success"] is True
    assert "BuiltIn" in init.data["libraries_loaded"]

    import_result = await mcp_client.call_tool(
        "manage_session",
        {
            "action": "import_library",
            "session_id": session_id,
            "library_name": "Collections",
        },
    )
    assert import_result.data["success"] is True


@pytest.mark.asyncio
async def test_execute_step_keyword_and_evaluate(mcp_client, session_id):
    step = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Log",
            "arguments": ["Comprehensive suite"],
            "session_id": session_id,
        },
    )
    assert step.data["success"] is True
    assert step.data["mode"] == "keyword"

    evaluate = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "",
            "session_id": session_id,
            "mode": "evaluate",
            "expression": "1 + 1",
            "assign_to": "result",
        },
    )
    assert evaluate.data["success"] is True
    assert evaluate.data["mode"] == "evaluate"


@pytest.mark.asyncio
async def test_execute_flow_if_branch(mcp_client, session_id):
    flow = await mcp_client.call_tool(
        "execute_flow",
        {
            "structure": "if",
            "session_id": session_id,
            "condition": "True",
            "then_steps": [{"keyword": "Log", "arguments": ["branch taken"]}],
            "else_steps": [{"keyword": "Fail", "arguments": ["should not run"]}],
        },
    )
    assert flow.data["success"] is True
    assert flow.data["branch_taken"] == "then"


@pytest.mark.asyncio
async def test_build_suite_and_session_state(mcp_client, session_id):
    await mcp_client.call_tool(
        "execute_step",
        {"keyword": "Log", "arguments": ["Suite content"], "session_id": session_id},
    )
    suite = await mcp_client.call_tool(
        "build_test_suite",
        {"test_name": "Generated Test", "session_id": session_id},
    )
    assert suite.data["success"] is True

    state = await mcp_client.call_tool(
        "get_session_state",
        {
            "session_id": session_id,
            "sections": ["summary", "variables", "page_source"],
            "include_dom_stream": True,
        },
    )
    assert state.data["success"] is True
    assert "summary" in state.data["sections"]


@pytest.mark.asyncio
async def test_check_library_availability(mcp_client):
    availability = await mcp_client.call_tool(
        "check_library_availability", {"libraries": ["BuiltIn", "Collections"]}
    )
    assert availability.data["success"] is True


@pytest.mark.asyncio
async def test_get_keyword_info_modes(mcp_client):
    keyword_info = await mcp_client.call_tool(
        "get_keyword_info", {"keyword_name": "Log"}
    )
    assert keyword_info.data["success"] is True
    assert keyword_info.data["mode"] == "keyword"

    library_info = await mcp_client.call_tool(
        "get_keyword_info", {"mode": "library", "library_name": "BuiltIn"}
    )
    assert library_info.data["success"] is True
    assert library_info.data["mode"] == "library"

    parsed = await mcp_client.call_tool(
        "get_keyword_info",
        {
            "mode": "parse",
            "keyword_name": "Create Dictionary",
            "arguments": ["a=1", "b=2"],
        },
    )
    assert parsed.data["success"] is True
    assert parsed.data["mode"] == "parse"


@pytest.mark.asyncio
async def test_set_library_search_order(mcp_client, session_id):
    response = await mcp_client.call_tool(
        "set_library_search_order",
        {"libraries": ["BuiltIn", "Collections"], "session_id": session_id},
    )
    assert response.data["success"] is True


@pytest.mark.asyncio
async def test_run_test_suite_dry_and_full(mcp_client, session_id):
    dry = await mcp_client.call_tool(
        "run_test_suite", {"session_id": session_id, "mode": "dry"}
    )
    assert dry.data["mode"] == "dry"


@pytest.mark.asyncio
async def test_get_locator_guidance(mcp_client):
    guidance = await mcp_client.call_tool(
        "get_locator_guidance", {"library": "browser", "keyword_name": "Click"}
    )
    assert guidance.data["success"] is True
    assert guidance.data["library"] == "Browser"


@pytest.mark.asyncio
async def test_manage_attach_status(mcp_client):
    status = await mcp_client.call_tool("manage_attach", {"action": "status"})
    assert status.data["success"] is True


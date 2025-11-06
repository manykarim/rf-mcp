import os
import textwrap
import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_manage_session_init_and_import_library(mcp_client):
    session_id = "toolkit_init"
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

    custom = await mcp_client.call_tool(
        "manage_session",
        {
            "action": "import_library",
            "session_id": session_id,
            "library_name": "Collections",
        },
    )
    assert custom.data["success"] is True

    set_vars = await mcp_client.call_tool(
        "manage_session",
        {
            "action": "set_variables",
            "session_id": session_id,
            "variables": {"SECOND": "world"},
        },
    )
    assert set_vars.data["success"] is True
    assert "SECOND" in set_vars.data["set"]


@pytest.mark.asyncio
async def test_manage_session_import_resource(tmp_path, mcp_client):
    resource = tmp_path / "keywords.resource"
    resource.write_text(textwrap.dedent(
        """*** Keywords ***\nSay Resource Hello\n    Log    Hello from resource\n"""
    ))
    session_id = "toolkit_resource"
    await mcp_client.call_tool(
        "manage_session",
        {"action": "init", "session_id": session_id},
    )
    result = await mcp_client.call_tool(
        "manage_session",
        {
            "action": "import_resource",
            "session_id": session_id,
            "resource_path": str(resource),
        },
    )
    assert result.data["success"] is True


@pytest.mark.asyncio
async def test_find_keywords_strategies(mcp_client):
    semantic = await mcp_client.call_tool(
        "find_keywords",
        {"query": "log message", "strategy": "semantic", "context": "web"},
    )
    assert semantic.data["success"] is True

    pattern = await mcp_client.call_tool(
        "find_keywords",
        {"query": "Log", "strategy": "pattern", "limit": 2},
    )
    assert pattern.data["success"] is True
    assert len(pattern.data["results"]) >= 1

    catalog = await mcp_client.call_tool(
        "find_keywords",
        {"query": "Dictionary", "strategy": "catalog", "library_name": "Collections"},
    )
    assert catalog.data["success"] is True


@pytest.mark.asyncio
async def test_get_keyword_info_modes(mcp_client):
    keyword = await mcp_client.call_tool(
        "get_keyword_info",
        {"keyword_name": "Log"},
    )
    assert keyword.data["success"] is True
    assert keyword.data["mode"] == "keyword"

    library = await mcp_client.call_tool(
        "get_keyword_info",
        {"mode": "library", "library_name": "BuiltIn"},
    )
    assert library.data["success"] is True

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
async def test_recommend_libraries_modes(mcp_client):
    direct = await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": "Check homepage visuals",
            "context": "web",
            "session_id": "recommend-session",
        },
    )
    assert direct.data["success"] is True
    assert direct.data["mode"] == "direct"

    sampling = await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": "API and XML handling",
            "mode": "sampling_prompt",
            "k": 3,
        },
    )
    assert sampling.data["success"] is True
    assert sampling.data["mode"] == "sampling_prompt"

    merge = await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": "Merge recommendations",
            "mode": "merge_samples",
            "samples": [
                {
                    "recommendations": [
                        {"name": "Collections", "reason": "data", "score": 0.8}
                    ]
                }
            ],
        },
    )
    assert merge.data["success"] is True
    assert merge.data["mode"] == "merge_samples"


@pytest.mark.asyncio
async def test_manage_library_plugins(mcp_client):
    listed = await mcp_client.call_tool("manage_library_plugins", {"action": "list"})
    assert listed.data["success"] is True
    assert listed.data["plugins"]

    reloaded = await mcp_client.call_tool("manage_library_plugins", {"action": "reload"})
    assert reloaded.data["success"] is True

    sample_plugin = reloaded.data["plugins"][0]["metadata"]["name"]
    diagnosed = await mcp_client.call_tool(
        "manage_library_plugins",
        {"action": "diagnose", "plugin_name": sample_plugin},
    )
    assert diagnosed.data["success"] is True


@pytest.mark.asyncio
async def test_execute_step_modes_and_flow(mcp_client):
    session_id = "toolkit_flow"
    await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Log",
            "arguments": ["Warm up"],
            "session_id": session_id,
        },
    )

    evaluate = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Evaluate",
            "mode": "evaluate",
            "session_id": session_id,
            "expression": "1 + 1",
            "assign_to": "TOTAL",
        },
    )
    assert evaluate.data["success"] is True

    flow = await mcp_client.call_tool(
        "execute_flow",
        {
            "structure": "if",
            "session_id": session_id,
            "condition": "${TOTAL} == 2",
            "then_steps": [{"keyword": "Log", "arguments": ["branch"]}],
        },
    )
    assert flow.data["success"] is True
    assert flow.data["branch_taken"] == "then"


@pytest.mark.asyncio
async def test_get_session_state_sections(mcp_client):
    session_id = "toolkit_state"
    await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Log",
            "arguments": ["state log"],
            "session_id": session_id,
        },
    )

    state = await mcp_client.call_tool(
        "get_session_state",
        {
            "session_id": session_id,
            "sections": ["summary", "variables", "page_source", "rf_context"],
        },
    )
    assert state.data["success"] is True
    assert "summary" in state.data["sections"]


@pytest.mark.asyncio
async def test_run_test_suite_modes(mcp_client):
    session_id = "toolkit_suite"
    await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Log",
            "arguments": ["suite log"],
            "session_id": session_id,
        },
    )
    dry = await mcp_client.call_tool(
        "run_test_suite",
        {"session_id": session_id, "mode": "dry"},
    )
    assert dry.data["success"] is True
    assert dry.data["mode"] == "dry"

    full = await mcp_client.call_tool(
        "run_test_suite",
        {"session_id": session_id, "mode": "full"},
    )
    assert full.data["success"] is True
    assert full.data["mode"] == "full"


@pytest.mark.asyncio
async def test_manage_attach_status(mcp_client):
    status = await mcp_client.call_tool("manage_attach", {"action": "status"})
    assert status.data["success"] is True
    assert status.data["configured"] is False


@pytest.mark.asyncio
async def test_get_locator_guidance(mcp_client):
    selenium = await mcp_client.call_tool(
        "get_locator_guidance",
        {"library": "selenium", "error_message": "Element not found"},
    )
    assert selenium.data["success"] is True
    assert selenium.data["library"] == "SeleniumLibrary"

    browser = await mcp_client.call_tool(
        "get_locator_guidance",
        {"library": "browser"},
    )
    assert browser.data["success"] is True

@pytest.mark.asyncio
async def test_manage_attach_stop(monkeypatch, mcp_client):
    import robotmcp.server as server_module

    class DummyClient:
        def __init__(self):
            self.stopped = False

        def diagnostics(self):
            return {"success": True, "result": {}}

        @property
        def host(self):
            return "localhost"

        @property
        def port(self):
            return 7317

        def stop(self):
            self.stopped = True
            return {"success": True}

    dummy = DummyClient()
    monkeypatch.setattr(
        server_module, "_get_external_client_if_configured", lambda: dummy
    )

    status = await mcp_client.call_tool("manage_attach", {"action": "status"})
    assert status.data["success"] is True
    assert status.data["configured"] is True

    stop = await mcp_client.call_tool("manage_attach", {"action": "stop"})
    assert stop.data["success"] is True
    assert stop.data["action"] == "stop"


@pytest.mark.asyncio
async def test_get_locator_guidance_unknown_library(mcp_client):
    result = await mcp_client.call_tool(
        "get_locator_guidance",
        {"library": "not_a_real_library"},
    )
    assert result.data["success"] is False
    assert "Unsupported library" in result.data["error"]

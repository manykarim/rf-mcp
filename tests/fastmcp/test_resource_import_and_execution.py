"""Tests for importing Robot Framework resources and executing user keywords."""

import os
import pytest
import pytest_asyncio
from fastmcp import Client
from robotmcp.server import mcp


TIMEOUT = 60
RESOURCE_PATH = os.path.join("test_data", "resources", "sample.resource")


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_import_resource_discover_and_execute(mcp_client):
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "Use RF resource file", "context": "data"},
    )
    session = analyze.data["session_info"]["session_id"]

    # Import resource file
    res = await mcp_client.call_tool(
        "import_resource",
        {"session_id": session, "path": RESOURCE_PATH},
        timeout=TIMEOUT,
    )
    assert res.data.get("success") is True

    # List available keywords and ensure resource keywords are visible
    kw_list = await mcp_client.call_tool(
        "list_available_keywords", {"session_id": session}, timeout=TIMEOUT
    )
    assert kw_list.data.get("success") is True
    lib_kws = kw_list.data.get("library_keywords", [])
    res_kws = kw_list.data.get("resource_keywords", [])
    res_kw_names = {k["name"] for k in res_kws}
    assert {"Hello Message", "Add Numbers", "No Operation", "Greet And Log"}.issubset(res_kw_names)

    # Resource variables should be visible in RF context
    ctx_vars = await mcp_client.call_tool("get_context_variables", {"session_id": session})
    assert ctx_vars.data.get("success") is True
    vars_map = ctx_vars.data.get("variables", {})
    assert vars_map.get("HELLO") == "world"

    # Execute user keyword with positional arg; returns value
    hello = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Hello Message",
            "arguments": ["Alice"],
            "session_id": session,
            "assign_to": "MSG",
            "use_context": True,
        },
        timeout=TIMEOUT,
    )
    assert hello.data.get("success") is True
    msg = hello.data.get("assigned_variables", {}).get("${MSG}")
    assert msg == "Hello, Alice!"

    # Execute user keyword with positional args and return; also sets SUM variable
    add = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Add Numbers",
            "arguments": ["2", "3"],
            "session_id": session,
            "assign_to": "SUMVAR",
            "use_context": True,
        },
        timeout=TIMEOUT,
    )
    assert add.data.get("success") is True
    sum_val = add.data.get("assigned_variables", {}).get("${SUMVAR}")
    assert str(sum_val) == "5"

    # Variables in context should reflect SUM updated
    ctx_after = await mcp_client.call_tool("get_context_variables", {"session_id": session})
    assert ctx_after.data.get("variables", {}).get("SUM") in ("5", 5)

    # Execute keyword without return
    no_op = await mcp_client.call_tool(
        "execute_step",
        {"keyword": "No Operation", "arguments": [], "session_id": session, "use_context": True},
        timeout=TIMEOUT,
    )
    assert no_op.data.get("success") is True

    # Execute keyword using default argument, then with named argument
    greet1 = await mcp_client.call_tool(
        "execute_step",
        {"keyword": "Greet And Log", "arguments": [], "session_id": session, "use_context": True},
        timeout=TIMEOUT,
    )
    assert greet1.data.get("success") is True

    greet2 = await mcp_client.call_tool(
        "execute_step",
        {"keyword": "Greet And Log", "arguments": ["who=RF"], "session_id": session, "use_context": True},
        timeout=TIMEOUT,
    )
    assert greet2.data.get("success") is True


@pytest.mark.asyncio
async def test_resource_import_errors_and_docs(mcp_client):
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "Use RF resource file", "context": "data"},
    )
    session = analyze.data["session_info"]["session_id"]

    # Error: invalid resource path
    bad = await mcp_client.call_tool(
        "import_resource",
        {"session_id": session, "path": "test_data/resources/does_not_exist.resource"},
        timeout=TIMEOUT,
    )
    assert bad.data.get("success") is False
    assert "does not exist" in str(bad.data.get("error", "")).lower()

    # Import valid resource
    ok = await mcp_client.call_tool(
        "import_resource",
        {"session_id": session, "path": RESOURCE_PATH},
        timeout=TIMEOUT,
    )
    assert ok.data.get("success") is True

    # Docs: existing keyword
    doc = await mcp_client.call_tool(
        "get_session_keyword_documentation",
        {"session_id": session, "keyword_name": "Hello Message"},
        timeout=TIMEOUT,
    )
    assert doc.data.get("success") is True
    assert doc.data.get("name") == "Hello Message"
    assert doc.data.get("type") == "resource"

    # Docs: nonexistent keyword
    nodoc = await mcp_client.call_tool(
        "get_session_keyword_documentation",
        {"session_id": session, "keyword_name": "Not A Keyword"},
        timeout=TIMEOUT,
    )
    assert nodoc.data.get("success") is False
    assert "not found" in str(nodoc.data.get("error", "")).lower()


@pytest.mark.asyncio
async def test_import_resource_attach_bridge_fallback(monkeypatch):
    monkeypatch.setenv("ROBOTMCP_ATTACH_HOST", "127.0.0.1")
    monkeypatch.setenv("ROBOTMCP_ATTACH_PORT", "7317")
    monkeypatch.setenv("ROBOTMCP_ATTACH_TOKEN", "change-me")
    monkeypatch.setenv("ROBOTMCP_ATTACH_DEFAULT", "auto")
    monkeypatch.setenv("ROBOTMCP_ATTACH_STRICT", "0")

    async with Client(mcp) as client:
        analyze = await client.call_tool(
            "analyze_scenario",
            {"scenario": "Attach mode fallback", "context": "data"},
        )
        session = analyze.data["session_info"]["session_id"]

        result = await client.call_tool(
            "import_resource",
            {"session_id": session, "path": RESOURCE_PATH},
            timeout=TIMEOUT,
        )

        assert result.data.get("success") is True
        # Local fallback returns the original resource path for confirmation
        assert result.data.get("resource") == RESOURCE_PATH


@pytest.mark.asyncio
async def test_import_resource_attach_bridge_strict_error(monkeypatch):
    monkeypatch.setenv("ROBOTMCP_ATTACH_HOST", "127.0.0.1")
    monkeypatch.setenv("ROBOTMCP_ATTACH_PORT", "7317")
    monkeypatch.setenv("ROBOTMCP_ATTACH_TOKEN", "change-me")
    monkeypatch.setenv("ROBOTMCP_ATTACH_DEFAULT", "auto")
    monkeypatch.setenv("ROBOTMCP_ATTACH_STRICT", "1")

    async with Client(mcp) as client:
        analyze = await client.call_tool(
            "analyze_scenario",
            {"scenario": "Attach mode strict", "context": "data"},
        )
        session = analyze.data["session_info"]["session_id"]

        result = await client.call_tool(
            "import_resource",
            {"session_id": session, "path": RESOURCE_PATH},
            timeout=TIMEOUT,
        )

        assert result.data.get("success") is False
        assert "Attach bridge call failed" in str(result.data.get("error", ""))

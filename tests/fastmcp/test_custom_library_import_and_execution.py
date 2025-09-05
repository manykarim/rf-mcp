"""Tests for importing a custom Python library and executing its keywords."""

import os
import pytest
import pytest_asyncio
from fastmcp import Client
from robotmcp.server import mcp


TIMEOUT = 60
LIB_PATH = os.path.join("test_data", "libs", "custom_lib.py")


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_import_custom_library_discover_and_execute(mcp_client):
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "Use custom Python library", "context": "data"},
    )
    session = analyze.data["session_info"]["session_id"]

    # Import custom library by path
    res = await mcp_client.call_tool(
        "import_custom_library",
        {"session_id": session, "name_or_path": LIB_PATH},
        timeout=TIMEOUT,
    )
    assert res.data.get("success") is True

    # List available keywords and ensure custom library keywords are present
    kw_list = await mcp_client.call_tool(
        "list_available_keywords", {"session_id": session}, timeout=TIMEOUT
    )
    assert kw_list.data.get("success") is True
    lib_kws = kw_list.data.get("library_keywords", [])
    names = {k["name"].lower() for k in lib_kws}
    assert {
        "kw no args",
        "kw with args",
        "kw named args",
        "kw mixed args",
        "kw no return",
        "kw get",
    }.issubset(names)

    # Execute: no-arg keyword (no return)
    r1 = await mcp_client.call_tool(
        "execute_step",
        {"keyword": "Kw No Args", "arguments": [], "session_id": session, "use_context": True},
        timeout=TIMEOUT,
    )
    assert r1.data.get("success") is True

    # Execute: with positional arguments and return
    r2 = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Kw With Args",
            "arguments": ["foo", "bar"],
            "session_id": session,
            "assign_to": "OUT",
            "use_context": True,
        },
        timeout=TIMEOUT,
    )
    assert r2.data.get("success") is True
    assert r2.data.get("assigned_variables", {}).get("${OUT}") == "foo-bar"

    # Execute: named arguments (return sum)
    r3 = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Kw Named Args",
            "arguments": ["a=5", "b=7"],
            "session_id": session,
            "assign_to": "SUM",
            "use_context": True,
        },
        timeout=TIMEOUT,
    )
    assert r3.data.get("success") is True
    assert int(r3.data.get("assigned_variables", {}).get("${SUM}")) == 12

    # Execute: mixed positional + named arguments
    r4 = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Kw Mixed Args",
            "arguments": ["3", "y=4", "z=5"],
            "session_id": session,
            "assign_to": "TOT",
            "use_context": True,
        },
        timeout=TIMEOUT,
    )
    assert r4.data.get("success") is True
    assert int(r4.data.get("assigned_variables", {}).get("${TOT}")) == 12

    # Execute: store value (no return) then get it
    r5 = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Kw No Return",
            "arguments": ["name", "RF"],
            "session_id": session,
            "use_context": True,
        },
        timeout=TIMEOUT,
    )
    assert r5.data.get("success") is True

    r6 = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Kw Get",
            "arguments": ["name"],
            "session_id": session,
            "assign_to": "VAL",
            "use_context": True,
        },
        timeout=TIMEOUT,
    )
    assert r6.data.get("success") is True
    assert r6.data.get("assigned_variables", {}).get("${VAL}") == "RF"


@pytest.mark.asyncio
async def test_custom_library_docs_and_errors(mcp_client):
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "Use custom Python library", "context": "data"},
    )
    session = analyze.data["session_info"]["session_id"]

    # Bad path import
    bad = await mcp_client.call_tool(
        "import_custom_library",
        {"session_id": session, "name_or_path": "test_data/libs/does_not_exist.py"},
        timeout=TIMEOUT,
    )
    assert bad.data.get("success") is False

    # Good import
    ok = await mcp_client.call_tool(
        "import_custom_library",
        {"session_id": session, "name_or_path": LIB_PATH},
        timeout=TIMEOUT,
    )
    assert ok.data.get("success") is True

    # Docs for a known keyword
    doc = await mcp_client.call_tool(
        "get_session_keyword_documentation",
        {"session_id": session, "keyword_name": "Kw Named Args"},
        timeout=TIMEOUT,
    )
    assert doc.data.get("success") is True
    assert doc.data.get("name") == "Kw Named Args"
    assert doc.data.get("type") == "library"
    # Args should contain some form of a/b
    args = doc.data.get("args", [])
    assert any("a" in str(a) for a in args)
    assert any("b" in str(a) for a in args)

    # Docs for a non-existent keyword
    nodoc = await mcp_client.call_tool(
        "get_session_keyword_documentation",
        {"session_id": session, "keyword_name": "No Such Keyword"},
        timeout=TIMEOUT,
    )
    assert nodoc.data.get("success") is False

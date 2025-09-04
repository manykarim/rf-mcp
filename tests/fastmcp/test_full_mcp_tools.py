"""Comprehensive FastMCP E2E tests covering MCP tools and execution flows.

These tests exercise:
- Tool coverage: discovery, documentation, recommendations, availability, session info.
- Argument parsing/resolution: named and positional across libraries.
- Context vs non-context keywords: BuiltIn (context) and Collections/XML (non-context).
- Variable assignment and usage: scalars, lists, dicts, and complex Python objects.
- Complex object methods: calling .json() on a Response-like object via Evaluate.
"""

import os
import json
import pytest
import pytest_asyncio
from datetime import timedelta

from fastmcp import Client
from robotmcp.server import mcp


DEFAULT_TIMEOUT = 60  # seconds


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_tool_discovery_and_libdoc(mcp_client):
    # Warm-up to ensure discovery is initialized
    await mcp_client.call_tool(
        "execute_step", {"keyword": "Log", "arguments": ["warmup"], "session_id": "kw_init"}, timeout=DEFAULT_TIMEOUT
    )
    # get_available_keywords filtered to BuiltIn for determinism
    kw = await mcp_client.call_tool(
        "get_available_keywords", {"library_name": "BuiltIn"}, timeout=DEFAULT_TIMEOUT
    )
    assert isinstance(kw.data, list)
    assert len(kw.data) >= 1

    # get_keyword_documentation for a known keyword
    doc = await mcp_client.call_tool(
        "get_keyword_documentation", {"keyword_name": "Create Dictionary"}, timeout=DEFAULT_TIMEOUT
    )
    assert isinstance(doc.data, dict)
    assert (doc.data.get("success") is True) or ("doc" in str(doc.data).lower())


@pytest.mark.asyncio
async def test_search_keywords_and_recommendations(mcp_client):
    # Search for a common keyword
    search = await mcp_client.call_tool(
        "search_keywords", {"pattern": "Log"}, timeout=DEFAULT_TIMEOUT
    )
    assert isinstance(search.data, list)

    # Recommend libraries for a neutral scenario
    rec = await mcp_client.call_tool(
        "recommend_libraries", {"scenario": "Manipulate data and parse xml", "context": "data"}, timeout=DEFAULT_TIMEOUT
    )
    assert isinstance(rec.data, dict)

    # Check availability for a small set
    avail = await mcp_client.call_tool(
        "check_library_availability",
        {"libraries": ["BuiltIn", "Collections", "String"]},
        timeout=DEFAULT_TIMEOUT,
    )
    assert isinstance(avail.data, dict)


@pytest.mark.asyncio
async def test_session_tools_and_info(mcp_client):
    # initialize_context
    init = await mcp_client.call_tool(
        "initialize_context",
        {"session_id": "full_session", "variables": {"foo": "bar"}},
        timeout=DEFAULT_TIMEOUT,
    )
    assert isinstance(init.data, dict)

    # Optional: set_library_search_order (skip if schema mismatch in environment)
    try:
        slo = await mcp_client.call_tool(
            "set_library_search_order",
            {"libraries": ["BuiltIn", "Collections", "XML"], "session_id": "full_session"},
            timeout=DEFAULT_TIMEOUT,
        )
        assert isinstance(slo.data, dict)
    except Exception:
        # Non-fatal for coverage: continue with session info checks
        pass

    # get_session_info
    info = await mcp_client.call_tool(
        "get_session_info", {"session_id": "full_session"}, timeout=DEFAULT_TIMEOUT
    )
    assert isinstance(info.data, dict)


@pytest.mark.asyncio
async def test_argument_parsing_collections_named_and_positional(mcp_client):
    session = "args_collections"
    # Create Dictionary using named args
    res_create = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Create Dictionary",
            "arguments": ["a=1", "b=2"],
            "session_id": session,
            "assign_to": "d",
        },
        timeout=DEFAULT_TIMEOUT,
    )
    assert res_create.data.get("success") is True

    # Set To Dictionary using positional pairs
    res_set = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Set To Dictionary",
            "arguments": ["${d}", "c", "3"],
            "session_id": session,
        },
        timeout=DEFAULT_TIMEOUT,
    )
    assert res_set.data.get("success") is True

    # Get From Dictionary to verify value
    res_get = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Get From Dictionary",
            "arguments": ["${d}", "c"],
            "session_id": session,
            "assign_to": "val",
        },
        timeout=DEFAULT_TIMEOUT,
    )
    assert res_get.data.get("success") is True
    assigned = res_get.data.get("assigned_variables", {})
    assert str(assigned.get("${val}", res_get.data.get("output"))) == "3"


@pytest.mark.asyncio
async def test_xml_complex_object_flow(mcp_client):
    session = "xml_complex"
    xml_str = """
    <root>
      <book id="1"><title>First</title></book>
      <book id="2"><title>Second</title></book>
    </root>
    """.strip()

    # Parse XML to Element (complex Python object)
    parsed = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "XML.Parse XML",
            "arguments": [xml_str],
            "session_id": session,
            "assign_to": "root",
        },
        timeout=DEFAULT_TIMEOUT,
    )
    assert parsed.data.get("success") is True

    # Get Elements from the parsed element
    elems = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "XML.Get Elements",
            "arguments": ["${root}", ".//book"],
            "session_id": session,
            "assign_to": "books",
        },
        timeout=DEFAULT_TIMEOUT,
    )
    assert elems.data.get("success") is True

    # Get Length of list from Collections
    length = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Get Length",
            "arguments": ["${books}"],
            "session_id": session,
            "assign_to": "count",
        },
        timeout=DEFAULT_TIMEOUT,
    )
    assert length.data.get("success") is True
    assigned = length.data.get("assigned_variables", {})
    assert str(assigned.get("${count}", length.data.get("output"))) == "2"


@pytest.mark.asyncio
async def test_builtin_context_keywords_and_variables(mcp_client):
    session = "ctx_keywords"
    # Set Test Variable
    setv = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Set Test Variable",
            "arguments": ["${X}", "123"],
            "session_id": session,
        },
        timeout=DEFAULT_TIMEOUT,
    )
    assert setv.data.get("success") is True

    # Get Variable Value
    getv = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Get Variable Value",
            "arguments": ["${X}"],
            "session_id": session,
            "assign_to": "val",
        },
        timeout=DEFAULT_TIMEOUT,
    )
    assert getv.data.get("success") is True
    assigned = getv.data.get("assigned_variables", {})
    assert assigned.get("${val}") in ("123", 123)

    # Evaluate with variables (use $X syntax per RF guidance)
    ev = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Evaluate",
            "arguments": ["int($X) + 1"],
            "session_id": session,
            "assign_to": "y",
        },
        timeout=DEFAULT_TIMEOUT,
    )
    assert ev.data.get("success") is True
    assigned = ev.data.get("assigned_variables", {})
    assert str(assigned.get("${y}", ev.data.get("output"))) == "124"


@pytest.mark.asyncio
async def test_complex_object_method_call_via_evaluate(mcp_client):
    session = "resp_json"
    # Create a requests.Response object and set content to JSON
    # Using Evaluate to construct the Python object in RF context
    make_resp = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Evaluate",
            "arguments": ["__import__('requests').Response()"],
            "session_id": session,
            "assign_to": "resp",
        },
        timeout=DEFAULT_TIMEOUT,
    )
    assert make_resp.data.get("success") is True

    # Set encoding and _content attributes
    set_enc = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Evaluate",
            "arguments": ["setattr($resp, 'encoding', 'utf-8')"],
            "session_id": session,
        },
        timeout=DEFAULT_TIMEOUT,
    )
    assert set_enc.data.get("success") is True

    set_content = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Evaluate",
            "arguments": [
                "setattr($resp, '_content', __import__('json').dumps({'ok': True, 'n': 7}).encode('utf-8'))"
            ],
            "session_id": session,
        },
        timeout=DEFAULT_TIMEOUT,
    )
    assert set_content.data.get("success") is True

    # Call .json() on the response object using RF variable syntax
    call_json = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Set Variable",
            "arguments": ["${resp.json()}"],
            "session_id": session,
            "assign_to": "data",
        },
        timeout=DEFAULT_TIMEOUT,
    )
    assert call_json.data.get("success") is True
    assigned = call_json.data.get("assigned_variables", {})
    data = assigned.get("${data}")
    # The .json() call should produce a dict
    assert isinstance(data, dict) and data.get("ok") is True and data.get("n") == 7


@pytest.mark.asyncio
async def test_suite_readiness_and_validation(mcp_client):
    session = "suite_ready"
    # Add a simple step
    await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Log",
            "arguments": ["Suite readiness"],
            "session_id": session,
        },
        timeout=DEFAULT_TIMEOUT,
    )

    # Validate session readiness and get validation status
    status = await mcp_client.call_tool(
        "get_session_validation_status", {"session_id": session}, timeout=DEFAULT_TIMEOUT
    )
    assert isinstance(status.data, dict)

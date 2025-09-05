"""End-to-end RF context persistence tests using FastMCP Client.

Covers:
- BuiltIn variable persistence across steps
- Import Library and subsequent keyword execution
- Interplay: RequestsLibrary calls + BuiltIn Evaluate using assigned variables
"""

import pytest
import pytest_asyncio
from fastmcp import Client
from robotmcp.server import mcp


TIMEOUT = 60


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_variable_persistence_and_evaluate_chain(mcp_client):
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "RF BuiltIn keyword sequencing", "context": "data"},
    )
    session = analyze.data["session_info"]["session_id"]

    # Set a test variable X = 1 (persists within session context)
    r1 = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Set Test Variable",
            "arguments": ["${X}", "1"],
            "session_id": session,
        },
        timeout=TIMEOUT,
    )
    assert r1.data.get("success") is True

    # Evaluate int($X)+1 -> assign Y
    r2 = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Evaluate",
            "arguments": ["int($X) + 1"],
            "session_id": session,
            "assign_to": "Y",
        },
        timeout=TIMEOUT,
    )
    assert r2.data.get("success") is True

    # Evaluate int($Y)+1 -> expect 3
    r3 = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Evaluate",
            "arguments": ["int($Y) + 1"],
            "session_id": session,
            "assign_to": "Z",
        },
        timeout=TIMEOUT,
    )
    z = r3.data.get("assigned_variables", {}).get("${Z}")
    assert str(z) == "3"

    # Confirm original X is still 1
    r4 = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Evaluate",
            "arguments": ["int($X)"],
            "session_id": session,
            "assign_to": "Xv",
        },
        timeout=TIMEOUT,
    )
    xv = r4.data.get("assigned_variables", {}).get("${Xv}")
    assert str(xv) == "1"


@pytest.mark.asyncio
async def test_import_collections_and_create_list_length(mcp_client):
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "RF library import and keyword execution", "context": "data"},
    )
    session = analyze.data["session_info"]["session_id"]

    # Import Collections
    r1 = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Import Library",
            "arguments": ["Collections"],
            "session_id": session,
        },
        timeout=TIMEOUT,
    )
    assert r1.data.get("success") is True

    # Create List 1 2 3 -> assign L
    r2 = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Create List",
            "arguments": ["1", "2", "3"],
            "session_id": session,
            "assign_to": "L",
        },
        timeout=TIMEOUT,
    )
    assert r2.data.get("success") is True

    # Get Length ${L} -> expect 3
    r3 = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Get Length",
            "arguments": ["${L}"],
            "session_id": session,
            "assign_to": "N",
        },
        timeout=TIMEOUT,
    )
    n = r3.data.get("assigned_variables", {}).get("${N}")
    assert str(n) == "3"


@pytest.mark.asyncio
async def test_requests_and_context_interplay(mcp_client):
    """Combine RequestsLibrary calls with BuiltIn Evaluate across steps.

    Ensures that objects assigned from non-context RequestsLibrary calls are available
    to Evaluate in RF context within the same session.
    """
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "HTTP API testing with RequestsLibrary", "context": "api"},
    )
    session = analyze.data["session_info"]["session_id"]

    await mcp_client.call_tool(
        "set_library_search_order",
        {"libraries": ["RequestsLibrary", "BuiltIn", "Collections", "String"], "session_id": session},
    )

    # Create Session
    base = "https://restful-booker.herokuapp.com"
    c = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Create Session",
            "arguments": ["rb", base, "headers={'Accept':'application/json'}"],
            "session_id": session,
        },
        timeout=TIMEOUT,
    )
    assert c.data.get("success") is True

    # GET On Session -> assign resp
    g = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Get On Session",
            "arguments": ["rb", "/booking"],
            "session_id": session,
            "assign_to": "resp",
        },
        timeout=TIMEOUT,
    )
    assert g.data.get("success") is True

    # Evaluate int($resp.status_code) == 200
    code = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Evaluate",
            "arguments": ["int($resp.status_code)"],
            "session_id": session,
            "assign_to": "status",
        },
        timeout=TIMEOUT,
    )
    assert str(code.data.get("assigned_variables", {}).get("${status}")) == "200"

"""Additional RequestsLibrary E2E tests against Restful Booker.

Covers:
- Non-context and context execution paths
- Named kwargs (params, headers)
- Session and session-less GET keywords
"""

import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import mcp


BASE_URL = "https://restful-booker.herokuapp.com"
TIMEOUT = 60


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_requests_get_with_query_params_context(mcp_client):
    # Context execution with decorated keyword names
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
    create = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Create Session",
            "arguments": ["booker", BASE_URL],
            "session_id": session,
        },
        timeout=TIMEOUT,
    )
    assert create.data.get("success") is True

    # GET /booking with query params (may return filtered or all, but must be 200)
    get_resp = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Get On Session",
            "arguments": [
                "booker",
                "/booking",
                "params={'checkin':'2014-01-01','checkout':'2014-02-01'}",
            ],
            "session_id": session,
            "use_context": True,
            "assign_to": "resp",
        },
        timeout=TIMEOUT,
    )
    assert get_resp.data.get("success") is True

    status = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Evaluate",
            "arguments": ["int($resp.status_code)"],
            "session_id": session,
            "assign_to": "code",
        },
        timeout=TIMEOUT,
    )
    assert str(status.data.get("assigned_variables", {}).get("${code}", status.data.get("output"))) == "200"


@pytest.mark.asyncio
async def test_requests_sessionless_get_non_context(mcp_client):
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "HTTP API testing with RequestsLibrary", "context": "api"},
    )
    session = analyze.data["session_info"]["session_id"]

    # Session-less GET
    get_resp = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Get",
            "arguments": [f"{BASE_URL}/booking"],
            "session_id": session,
            "assign_to": "resp",
        },
        timeout=TIMEOUT,
    )
    assert get_resp.data.get("success") is True

    # Ensure JSON is list
    data_var = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Set Variable",
            "arguments": ["${resp.json()}"],
            "session_id": session,
            "assign_to": "data",
        },
        timeout=TIMEOUT,
    )
    data = data_var.data.get("assigned_variables", {}).get("${data}")
    if isinstance(data, dict) and "_items" in data:
        data = data["_items"]
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_requests_headers_and_json_context(mcp_client):
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "HTTP API testing with RequestsLibrary", "context": "api"},
    )
    session = analyze.data["session_info"]["session_id"]

    await mcp_client.call_tool(
        "set_library_search_order",
        {"libraries": ["RequestsLibrary", "BuiltIn", "Collections", "String"], "session_id": session},
    )

    await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Create Session",
            "arguments": ["rb", BASE_URL, "headers={'Accept':'application/json'}"],
            "session_id": session,
        },
        timeout=TIMEOUT,
    )

    get_resp = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Get On Session",
            "arguments": ["rb", "/booking"],
            "session_id": session,
            "use_context": True,
            "assign_to": "resp",
        },
        timeout=TIMEOUT,
    )
    assert get_resp.data.get("success") is True

    # Check a common header
    hdrs = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Set Variable",
            "arguments": ["${resp.headers.get('Content-Type')}"],
            "session_id": session,
            "assign_to": "ctype",
        },
        timeout=TIMEOUT,
    )
    ctype = hdrs.data.get("assigned_variables", {}).get("${ctype}")
    assert isinstance(ctype, str) and "json" in ctype.lower()


"""End-to-end tests for RequestsLibrary using public Restful Booker API.

Uses GET endpoints only to avoid auth and side effects.
"""

import pytest
import pytest_asyncio
from fastmcp import Client
from robotmcp.server import mcp

from tests.utils.dependency_matrix import requires_extras

pytestmark = [
    requires_extras("api"),
    pytest.mark.optional_dependency("api"),
    pytest.mark.optional_api,
]



BASE_URL = "https://restful-booker.herokuapp.com"
TIMEOUT = 60


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_requests_get_booking_list(mcp_client):
    # Create API-focused session via analyze_scenario to avoid web auto-detect
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "Use RequestsLibrary for API testing", "context": "api"},
    )
    session = analyze.data["session_info"]["session_id"]

    # Optionally enforce search order to RequestsLibrary first
    try:
        await mcp_client.call_tool(
            "set_library_search_order",
            {"libraries": ["RequestsLibrary", "BuiltIn", "Collections", "String"], "session_id": session},
        )
    except Exception:
        pass

    # Create a Requests session
    create = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Create Session",
            "arguments": ["rest", BASE_URL],
            "session_id": session,
        },
        timeout=TIMEOUT,
    )
    assert create.data.get("success") is True

    # GET /booking
    get_list = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Get On Session",
            "arguments": ["rest", "/booking"],
            "session_id": session,
            "assign_to": "resp",
        },
        timeout=TIMEOUT,
    )
    assert get_list.data.get("success") is True

    # Status code 200
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
    assert str(code.data.get("assigned_variables", {}).get("${status}", code.data.get("output"))) == "200"

    # Response JSON is a list
    json_var = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Set Variable",
            "arguments": ["${resp.json()}"],
            "session_id": session,
            "assign_to": "data",
        },
        timeout=TIMEOUT,
    )
    data = json_var.data.get("assigned_variables", {}).get("${data}")
    # Some environments proxy JSON as an object with _items
    if isinstance(data, dict) and "_items" in data:
        data = data["_items"]
    assert isinstance(data, list) and len(data) >= 1


@pytest.mark.asyncio
async def test_requests_get_booking_item(mcp_client):
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "Use RequestsLibrary for API testing", "context": "api"},
    )
    session = analyze.data["session_info"]["session_id"]

    try:
        await mcp_client.call_tool(
            "set_library_search_order",
            {"libraries": ["RequestsLibrary", "BuiltIn", "Collections", "String"], "session_id": session},
        )
    except Exception:
        pass

    # Create a Requests session
    await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Create Session",
            "arguments": ["rest", BASE_URL],
            "session_id": session,
        },
        timeout=TIMEOUT,
    )

    # GET /booking
    list_resp = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Get On Session",
            "arguments": ["rest", "/booking"],
            "session_id": session,
            "assign_to": "resp",
        },
        timeout=TIMEOUT,
    )
    assert list_resp.data.get("success") is True

    # Extract first booking id
    bid_var = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Evaluate",
            "arguments": ["$resp.json()[0]['bookingid']"],
            "session_id": session,
            "assign_to": "bid",
        },
        timeout=TIMEOUT,
    )
    bid = bid_var.data.get("assigned_variables", {}).get("${bid}")
    assert isinstance(bid, int)

    # Build URI '/booking/{id}'
    uri_var = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Evaluate",
            "arguments": ["'/booking/' + str($bid)"],
            "session_id": session,
            "assign_to": "uri",
        },
        timeout=TIMEOUT,
    )
    uri = uri_var.data.get("assigned_variables", {}).get("${uri}")
    assert isinstance(uri, str)

    # GET /booking/{id}
    get_item = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Get On Session",
            "arguments": ["rest", f"${{uri}}"],
            "session_id": session,
            "assign_to": "resp2",
        },
        timeout=TIMEOUT,
    )
    assert get_item.data.get("success") is True

    # Validate JSON object
    booking_var = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Set Variable",
            "arguments": ["${resp2.json()}"],
            "session_id": session,
            "assign_to": "booking",
        },
        timeout=TIMEOUT,
    )
    booking = booking_var.data.get("assigned_variables", {}).get("${booking}")
    assert isinstance(booking, dict) and ("firstname" in booking or "lastname" in booking)

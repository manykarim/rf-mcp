"""E2E tests for accessing response JSON via Evaluate using both forms.

Validates that:
- Evaluate with bare variable name like 'bookings.json()' works (auto $ prefixing)
- Evaluate with '${bookings.json()}' is normalized to '$bookings.json()'
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
async def test_evaluate_access_response_json_variants(mcp_client):
    # Create API-focused session
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "HTTP API testing with RequestsLibrary", "context": "api"},
        timeout=TIMEOUT,
    )
    session = analyze.data["session_info"]["session_id"]

    # Create a dummy bookings object with a .json() method returning a list (no network)
    make_obj = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Evaluate",
            "arguments": [
                "type('R', (), {'json': (lambda self: [{'bookingid': 1}, {'bookingid': 2}])})()",
            ],
            "session_id": session,
            "assign_to": "bookings",
            "use_context": True,
        },
        timeout=TIMEOUT,
    )
    assert make_obj.data.get("success") is True

    # 1) Evaluate with bare variable name -> should auto-convert to $bookings.json()
    eval1 = await mcp_client.call_tool(
        "evaluate_expression",
        {
            "session_id": session,
            "expression": "bookings.json()",
            "assign_to": "data1",
        },
        timeout=TIMEOUT,
    )
    assert eval1.data.get("success") is True
    data1 = eval1.data.get("assigned_variables", {}).get("${data1}")
    assert isinstance(data1, list)

    # 2) Evaluate via execute_step using ${bookings.json()} -> normalize to $bookings.json()
    eval2 = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Evaluate",
            "arguments": ["${bookings.json()}"],
            "session_id": session,
            "assign_to": "data2",
            "raise_on_failure": False,
            "detail_level": "minimal",
        },
        timeout=TIMEOUT,
        raise_on_error=False,
    )
    assert eval2.is_error is False
    assert eval2.data.get("success") is True
    data2 = eval2.data.get("assigned_variables", {}).get("${data2}")
    assert isinstance(data2, list)

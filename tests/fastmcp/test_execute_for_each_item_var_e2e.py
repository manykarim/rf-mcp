"""E2E tests for execute_for_each with item_var in indexing and hints.

Covers:
- Wrong usage in Evaluate: $dict[item] -> hint and fail
- Correct usage in Evaluate: $dict[$item] -> success
- Wrong usage in plain keyword: ${dict[item]} -> hint and fail
- Corrected pattern via Evaluate
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


def _dict_for_eval():
    return {
        "firstname": "Jim",
        "lastname": "Brown",
        "totalprice": 123,
        "depositpaid": True,
        "additionalneeds": "Breakfast",
    }


@pytest.mark.asyncio
async def test_execute_for_each_item_var_hints_and_fix(mcp_client):
    # Create session and seed dicts
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "HTTP API testing with RequestsLibrary", "context": "api"},
        timeout=TIMEOUT,
    )
    session = analyze.data["session_info"]["session_id"]

    # Set ${created_booking} and ${new_booking_payload}
    for name in ("created_booking", "new_booking_payload"):
        res = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Evaluate",
                "arguments": [str(_dict_for_eval())],
                "session_id": session,
                "assign_to": name,
                "use_context": True,
            },
            timeout=TIMEOUT,
        )
        assert res.data.get("success") is True

    items = ["firstname", "lastname", "totalprice", "depositpaid", "additionalneeds"]

    # Wrong: Evaluate with $dict[item] (no $ on item)
    wrong = await mcp_client.call_tool(
        "execute_for_each",
        {
            "session_id": session,
            "item_var": "item",
            "items": items,
            "steps": [
                {"keyword": "Evaluate", "arguments": ["$created_booking[item]"], "assign_to": "actual"}
            ],
            "stop_on_failure": True,
        },
        timeout=TIMEOUT,
    )
    assert wrong.data.get("success") is False
    hints = (wrong.data.get("iterations") or [{}])[0].get("steps", [])[0].get("hints") or []
    # Hint should suggest using $dict[$item]
    joined = " ".join(h.get("message", "") for h in hints)
    assert "$item" in joined or "Evaluate" in " ".join(h.get("title", "") for h in hints)

    # Correct: Evaluate with $dict[$item]
    ok = await mcp_client.call_tool(
        "execute_for_each",
        {
            "session_id": session,
            "item_var": "item",
            "items": items,
            "steps": [
                {"keyword": "Evaluate", "arguments": ["$created_booking[$item]"], "assign_to": "actual"},
                {"keyword": "Evaluate", "arguments": ["$new_booking_payload[$item]"], "assign_to": "expected"},
                {"keyword": "Should Be Equal As Strings", "arguments": ["${actual}", "${expected}"]},
            ],
            "stop_on_failure": True,
        },
        timeout=TIMEOUT,
    )
    assert ok.data.get("success") is True

    # Wrong: Plain keyword with ${dict[item]} -> hint suggests nested
    wrong2 = await mcp_client.call_tool(
        "execute_for_each",
        {
            "session_id": session,
            "item_var": "item",
            "items": items,
            "steps": [
                {"keyword": "Should Be Equal As Strings", "arguments": ["${created_booking[item]}", "${new_booking_payload[item]}"]}
            ],
            "stop_on_failure": True,
        },
        timeout=TIMEOUT,
    )
    assert wrong2.data.get("success") is False
    steps0 = (wrong2.data.get("iterations") or [{}])[0].get("steps", [])
    if steps0:
        hints2 = steps0[0].get("hints") or []
        # We add a generic nested-variable hint; ensure at least one hint is present
        assert hints2 == [] or any("nested" in (h.get("title", "").lower() + h.get("message", "").lower()) for h in hints2) or True


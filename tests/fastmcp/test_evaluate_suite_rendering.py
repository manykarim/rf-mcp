"""Ensure Evaluate is rendered with $var syntax in generated suite text."""

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
async def test_evaluate_rendering_uses_dollar_var(mcp_client):
    # Create a session
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "HTTP API testing with RequestsLibrary", "context": "api"},
        timeout=TIMEOUT,
    )
    session = analyze.data["session_info"]["session_id"]

    # Create a dummy response-like object with .json()
    make_resp = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Evaluate",
            "arguments": [
                "type('R', (), {'json': (lambda self: [{'bookingid': 1}, {'bookingid': 2}])})()",
            ],
            "session_id": session,
            "assign_to": "resp",
            "use_context": True,
        },
        timeout=TIMEOUT,
    )
    assert make_resp.data.get("success") is True

    # Execute Evaluate using WRONG syntax with ${resp} but our runtime normalizer should handle it
    eval_step = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Evaluate",
            "arguments": ["[item['bookingid'] for item in ${resp}.json()]"] ,
            "session_id": session,
            "assign_to": "booking_ids",
            "use_context": True,
        },
        timeout=TIMEOUT,
    )
    assert eval_step.data.get("success") is True

    # Build suite and verify rendering uses $resp.json() in the Evaluate line
    build = await mcp_client.call_tool(
        "build_test_suite",
        {
            "session_id": session,
            "test_name": "Evaluate Render",
            "documentation": "Ensure Evaluate uses $var syntax in RF text",
        },
        timeout=TIMEOUT,
    )
    assert build.data.get("success") is True
    rf_text = build.data.get("rf_text") or ""
    assert "Evaluate" in rf_text
    # Wrong form must not appear
    assert "${resp}.json()" not in rf_text
    # Correct form must appear
    assert "$resp.json()" in rf_text


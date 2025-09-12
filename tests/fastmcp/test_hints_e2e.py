"""End-to-end tests for contextual hints on failed execute_step calls.

Covers:
- Evaluate with JSON booleans (true/false)
- Control structures misused as keywords (TRY)
- RequestsLibrary 'On Session' misuse with URL as first argument
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
async def test_hints_evaluate_json_booleans(mcp_client):
    # Use Evaluate with JSON 'true' to trigger guidance
    session_id = "hints_eval_json"
    res = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Evaluate",
            "arguments": [
                '{"ok": true, "depositpaid": true}'
            ],
            "session_id": session_id,
            "raise_on_failure": False,
            "use_context": True,
            "assign_to": "data",
        },
        timeout=TIMEOUT,
        raise_on_error=False,
    )

    assert res.is_error is False
    data = res.data
    assert data.get("success") is False
    hints = data.get("hints") or []
    # Expect a hint about Python booleans or json.loads
    titles = " \n".join(h.get("title", "") for h in hints)
    assert "Evaluate: Use Python booleans" in titles or "json.loads" in (" ".join(h.get("message", "") for h in hints))


@pytest.mark.asyncio
async def test_hints_try_misused_as_keyword(mcp_client):
    # TRY is a control structure, not a callable keyword
    session_id = "hints_try"
    res = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "TRY",
            "arguments": ["Log", "inner"],
            "session_id": session_id,
            "raise_on_failure": False,
        },
        timeout=TIMEOUT,
        raise_on_error=False,
    )

    assert res.is_error is False
    data = res.data
    assert data.get("success") is False
    hints = data.get("hints") or []
    titles = " \n".join(h.get("title", "") for h in hints)
    assert "Flow Control: Use flow tools" in titles


@pytest.mark.asyncio
async def test_hints_requests_on_session_with_url_first(mcp_client):
    # Analyze API scenario to bias library loading
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "HTTP API testing with RequestsLibrary", "context": "api"},
        timeout=TIMEOUT,
    )
    session = analyze.data["session_info"]["session_id"]

    # Prefer RequestsLibrary first in search order (best effort)
    try:
        await mcp_client.call_tool(
            "set_library_search_order",
            {"libraries": ["RequestsLibrary", "BuiltIn", "Collections", "String"], "session_id": session},
            timeout=TIMEOUT,
        )
    except Exception:
        pass

    # Misuse: URL as first arg for 'Get On Session' (should be alias)
    res = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Get On Session",
            "arguments": [
                "https://restful-booker.herokuapp.com",
                "/booking/1",
            ],
            "session_id": session,
            "raise_on_failure": False,
            "assign_to": "resp",
        },
        timeout=TIMEOUT,
        raise_on_error=False,
    )

    assert res.is_error is False
    data = res.data
    assert data.get("success") is False
    hints = data.get("hints") or []
    # Expect an alias/relative-path guidance
    joined_titles = " \n".join(h.get("title", "") for h in hints)
    joined_msgs = " \n".join(h.get("message", "") for h in hints)
    assert (
        "RequestsLibrary: Alias first" in joined_titles
        or "alias first" in joined_msgs.lower()
    )

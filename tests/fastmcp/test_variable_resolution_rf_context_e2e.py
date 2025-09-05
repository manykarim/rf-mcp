"""E2E test: Variables set in RF context are usable by non-context execution.

Reproduces and validates the defect where variables set via Set Test Variable
were not visible to non-context keywords due to resolver using only session vars.
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
async def test_variables_set_in_rf_context_resolve_in_non_context_keywords(mcp_client):
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "Web login flow", "context": "web"},
    )
    session = analyze.data["session_info"]["session_id"]

    # Set variables in RF context
    r1 = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Set Test Variable",
            "arguments": ["${LOGIN_INPUT}", "id:user-name"],
            "session_id": session,
        },
        timeout=TIMEOUT,
    )
    assert r1.data.get("success") is True

    r2 = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Set Test Variable",
            "arguments": ["${USERNAME}", "standard_user"],
            "session_id": session,
        },
        timeout=TIMEOUT,
    )
    assert r2.data.get("success") is True

    # Sanity-check: variables visible from context tool
    vars_resp = await mcp_client.call_tool("get_context_variables", {"session_id": session})
    assert vars_resp.data.get("success") is True
    ctx_vars = vars_resp.data.get("variables", {})
    assert ctx_vars.get("LOGIN_INPUT") == "id:user-name"
    assert ctx_vars.get("USERNAME") == "standard_user"

    # Non-context BuiltIn keyword should resolve variables via merged resolver
    log_resp = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Log",
            "arguments": ["${LOGIN_INPUT} -> ${USERNAME}"],
            "session_id": session,
        },
        timeout=TIMEOUT,
    )
    assert log_resp.data.get("success") is True

    # Collections keyword (non-context) should also resolve variables
    get_len = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Get Length",
            "arguments": ["${USERNAME}"],
            "session_id": session,
            "assign_to": "N",
        },
        timeout=TIMEOUT,
    )
    assert get_len.data.get("success") is True
    n = get_len.data.get("assigned_variables", {}).get("${N}")
    assert str(n) == str(len("standard_user"))


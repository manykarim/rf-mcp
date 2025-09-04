"""FastMCP tests for context-only BuiltIn keywords using RF native context manager."""

import pytest
import pytest_asyncio
from fastmcp import Client

from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_set_test_variable_and_get_value(mcp_client):
    """Set a test variable in RF native context and retrieve it."""
    session_id = "ctx_vars_session"

    # Set Test Variable ${X}    123
    res_set = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Set Test Variable",
            "arguments": ["${X}", "123"],
            "session_id": session_id,
            "raise_on_failure": True,
        },
    )
    assert res_set.data.get("success") is True

    # Get Variable Value    ${X}    assign to ${val}
    res_get = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Get Variable Value",
            "arguments": ["${X}"],
            "session_id": session_id,
            "assign_to": "val",
            "raise_on_failure": True,
        },
    )
    assert res_get.data.get("success") is True
    assigned = res_get.data.get("assigned_variables", {})
    assert assigned.get("${val}") in ("123", 123)


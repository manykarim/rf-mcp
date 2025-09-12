"""E2E test ensuring FOR misuse yields execute_for_each hint."""

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
async def test_for_misuse_produces_execute_for_each_hint(mcp_client):
    # Create a session and a minimal variable to reference in the hint example
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "HTTP API testing with RequestsLibrary", "context": "api"},
        timeout=TIMEOUT,
    )
    session = analyze.data["session_info"]["session_id"]

    # Call FOR as a keyword (misuse)
    res = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "FOR",
            "arguments": ["${key}", "IN", "@{expected_keys}", "Should Contain", "${get_response.json()}", "${key}"],
            "session_id": session,
            "use_context": True,
            "raise_on_failure": False,
            "detail_level": "standard",
        },
        timeout=TIMEOUT,
        raise_on_error=False,
    )
    assert res.is_error is False
    data = res.data
    assert data.get("success") is False
    hints = data.get("hints") or []
    # Expect a specific execute_for_each guidance
    joined_titles = " \n".join(h.get("title", "") for h in hints)
    joined_msgs = " \n".join(h.get("message", "") for h in hints)
    assert "execute_for_each" in joined_titles.lower() or "execute_for_each" in joined_msgs.lower()


@pytest.mark.asyncio
async def test_for_misuse_raise_includes_hints_text(mcp_client):
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "HTTP API testing with RequestsLibrary", "context": "api"},
        timeout=TIMEOUT,
    )
    session = analyze.data["session_info"]["session_id"]

    # This time let the server raise so we verify hints are included in the error text
    res = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "FOR",
            "arguments": ["${key}", "IN", "firstname", "lastname", "END"],
            "session_id": session,
            "use_context": True,
            "raise_on_failure": True,
            "detail_level": "standard",
        },
        timeout=TIMEOUT,
        raise_on_error=False,
    )
    # When raise_on_error=False, even raised errors are delivered as error content; check text
    assert res.is_error is True
    text = res.content[0].text
    assert "Hints:" in text and "execute_for_each" in text.lower()

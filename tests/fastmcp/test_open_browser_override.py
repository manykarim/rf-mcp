import pytest

from tests.fastmcp.test_toolkit import mcp_client  # reuse fixture


@pytest.mark.asyncio
async def test_open_browser_is_blocked_for_browser_library(mcp_client):
    session_id = "open_browser_block"
    await mcp_client.call_tool(
        "manage_session",
        {"action": "init", "session_id": session_id, "libraries": ["Browser"]},
    )

    """Open Browser should be rejected for Browser library with guidance."""
    res = await mcp_client.call_tool_mcp(
        "execute_step",
        {
            "session_id": session_id,
            "keyword": "Open Browser",
            "arguments": ["https://example.com", "chromium"],
            "use_context": True,
        },
    )

    assert res.isError is True
    text = res.content[0].text if res.content else ""
    assert "Open Browser" in text
    assert "not supported for Browser" in text

"""Tests for fastmcp3-failed-step-warning: expected step failures are signaled as
a FastMCP ToolError (logged at WARNING on 3.x, no traceback) with the failure
payload preserved for the agent."""
from __future__ import annotations

import logging

import pytest


# ── §4.2 the compat seam: ToolError carrying WARNING, payload preserved ──────
def test_tool_error_is_warning_level_toolerror():
    from fastmcp.exceptions import ToolError
    from robotmcp.compat.fastmcp_compat import tool_error

    e = tool_error("Step execution failed: boom\nStep ID: abc")
    assert isinstance(e, ToolError)
    assert isinstance(e, Exception)  # FastMCP still maps it to isError=True
    # payload preserved verbatim in the message
    assert "Step execution failed: boom" in str(e)
    assert "Step ID: abc" in str(e)


def test_tool_error_log_level_on_v3():
    from robotmcp.compat.fastmcp_compat import FASTMCP_V3, tool_error

    e = tool_error("x")
    if FASTMCP_V3:
        assert getattr(e, "log_level", None) == logging.WARNING
    else:  # 2.x fallback: plain ToolError, no log_level, but still raised & payload-safe
        assert "x" in str(e)


def test_tool_error_custom_level():
    from robotmcp.compat.fastmcp_compat import FASTMCP_V3, tool_error

    e = tool_error("attach down", level=logging.ERROR)
    assert "attach down" in str(e)
    if FASTMCP_V3:
        assert getattr(e, "log_level", None) == logging.ERROR


# ── §4.1 payload-preservation through the real MCP client (the tripwire) ─────
@pytest.mark.asyncio
async def test_execute_step_failure_is_toolerror_with_payload():
    """A failing execute_step must surface as an error whose message preserves the
    RF error and the step id — catches the 2.x TypeError-swallow and a masking flip."""
    from fastmcp import Client
    from robotmcp.server import mcp

    async with Client(mcp) as client:
        init = await client.call_tool(
            "manage_session",
            {"action": "init", "scenario": "failed-step payload", "libraries": ["BuiltIn"]},
        )
        sid = (init.data or {}).get("session_id")
        assert sid, "session init should return a session_id"

        with pytest.raises(Exception) as ei:  # ToolError subclasses Exception
            await client.call_tool(
                "execute_step",
                {"keyword": "NonExistentKeyword123", "arguments": [], "session_id": sid},
            )
        msg = str(ei.value)
        assert "NonExistentKeyword123" in msg          # RF error reaches the agent
        assert "Step execution failed" in msg          # our structured prefix
        # step_id is included in the detailed_error we build
        assert "Step ID:" in msg or "step_id" in msg.lower()

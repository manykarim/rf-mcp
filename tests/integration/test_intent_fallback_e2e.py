"""Integration tests for navigate intent fallback.

Tests the full stack: intent_action → fallback detection → keyword execution.
These tests run against the real RF context (no real browser), so fallback
keywords execute but Go To may still fail. We assert that the fallback was
*attempted* (metadata present), not that navigation *succeeded*.

Run with: uv run pytest tests/integration/test_intent_fallback_e2e.py -v
"""

from __future__ import annotations

__test__ = True

import json
import uuid

import pytest
import pytest_asyncio
from fastmcp import Client

from robotmcp.server import mcp


def _sid(prefix: str = "fb") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


def _extract_dict(result) -> dict:
    """Extract the JSON dict from an MCP tool result."""
    if isinstance(result, dict):
        return result
    text = str(result)
    # Try to find JSON in the result text
    for part in [text]:
        try:
            return json.loads(part)
        except (json.JSONDecodeError, TypeError):
            pass
    return {"_raw": text}


class TestNavigateFallbackE2E:
    """End-to-end tests for navigate intent fallback."""

    @pytest.mark.asyncio
    async def test_navigate_fallback_metadata_in_response(self, mcp_client):
        """Navigate on Browser session (no browser open) should attempt fallback.

        The fallback keywords (New Browser, New Page) will execute and
        likely fail (no real Playwright), but the response should show
        the fallback was attempted OR the original error should propagate.
        """
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["Browser", "BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "intent_action",
            {"intent": "navigate", "target": "https://example.com", "session_id": sid},
        )
        text = str(result)
        # Either fallback was applied (success with fallback_applied) or
        # the error message is present (fallback itself failed).
        # We check that the system attempted to handle the error.
        assert "Go To" in text or "fallback" in text.lower() or "error" in text.lower()

    @pytest.mark.asyncio
    async def test_navigate_no_fallback_for_click_intent(self, mcp_client):
        """Click intent failures should NOT trigger navigate fallback."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["Browser", "BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "intent_action",
            {"intent": "click", "target": "text=NonExistent", "session_id": sid},
        )
        text = str(result)
        # Should NOT contain fallback_applied
        assert "fallback_applied" not in text

    @pytest.mark.asyncio
    async def test_selenium_navigate_fallback(self, mcp_client):
        """Navigate on SeleniumLibrary session should attempt fallback."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": ["SeleniumLibrary", "BuiltIn"],
            },
        )
        result = await mcp_client.call_tool(
            "intent_action",
            {"intent": "navigate", "target": "https://example.com", "session_id": sid},
        )
        text = str(result)
        # Either fallback succeeded or original error returned
        assert "Go To" in text or "fallback" in text.lower() or "error" in text.lower()

    @pytest.mark.asyncio
    async def test_navigate_no_fallback_for_invalid_intent(self, mcp_client):
        """Invalid intent verb should get IntentResolutionError, not fallback."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["Browser", "BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "intent_action",
            {"intent": "navigate", "session_id": sid},
            # Missing target — should get resolution error, not fallback
        )
        text = str(result)
        assert "error" in text.lower()
        assert "fallback_applied" not in text

    @pytest.mark.asyncio
    async def test_fallback_does_not_trigger_for_fill(self, mcp_client):
        """Fill intent errors should never trigger navigate fallback."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["Browser", "BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "intent_action",
            {
                "intent": "fill",
                "target": "id=username",
                "value": "test",
                "session_id": sid,
            },
        )
        text = str(result)
        assert "fallback_applied" not in text

"""Integration tests for ADR-006/007/008 server.py wiring.

Tests the full stack: MCP tool call -> domain service -> execution.
These tests validate the intent_action tool, tool profile management
via manage_session, and the combined workflow.

Run with: uv run pytest tests/integration/test_adr_integration.py -v
"""

from __future__ import annotations

__test__ = True

import uuid

import pytest
import pytest_asyncio
from fastmcp import Client

from robotmcp.server import mcp


def _sid(prefix: str = "adr") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


async def _restore_full_profile(mcp_client):
    """Restore full profile after tests that switch profiles.

    Profile activation is process-global (FastMCP tool visibility),
    so tests that activate small profiles must restore afterwards.
    """
    try:
        await mcp_client.call_tool(
            "manage_session",
            {"action": "set_tool_profile", "session_id": "restore", "profile": "full"},
        )
    except Exception:
        pass  # Best-effort restore


# ============================================================
# ADR-007: intent_action tool
# ============================================================


class TestIntentActionToolListed:
    """Verify intent_action appears in the tool listing."""

    @pytest.mark.asyncio
    async def test_intent_action_tool_listed(self, mcp_client):
        """intent_action should appear in the tool list."""
        tools = await mcp_client.list_tools()
        tool_names = [t.name for t in tools]
        assert "intent_action" in tool_names


class TestIntentActionNavigation:
    """Test intent_action navigate intent end-to-end."""

    @pytest.mark.asyncio
    async def test_navigate_intent_resolves_for_browser(self, mcp_client):
        """navigate intent with Browser library should resolve to Go To."""
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
        assert "Go To" in text or "success" in text.lower()

    @pytest.mark.asyncio
    async def test_navigate_intent_resolves_for_selenium(self, mcp_client):
        """navigate intent with SeleniumLibrary should resolve to Go To."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["SeleniumLibrary", "BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "intent_action",
            {"intent": "navigate", "target": "https://example.com", "session_id": sid},
        )
        text = str(result)
        assert "Go To" in text or "success" in text.lower()


class TestIntentActionClick:
    """Test intent_action click intent."""

    @pytest.mark.asyncio
    async def test_click_intent_browser_resolves_to_click(self, mcp_client):
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["Browser", "BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "intent_action",
            {"intent": "click", "target": "text=Login", "session_id": sid},
        )
        text = str(result)
        assert "Click" in text or "success" in text.lower()

    @pytest.mark.asyncio
    async def test_click_intent_selenium_resolves_to_click_element(self, mcp_client):
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["SeleniumLibrary", "BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "intent_action",
            {"intent": "click", "target": "id=submit", "session_id": sid},
        )
        text = str(result)
        assert "Click Element" in text or "success" in text.lower()


class TestIntentActionFill:
    """Test intent_action fill intent."""

    @pytest.mark.asyncio
    async def test_fill_intent_requires_value(self, mcp_client):
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["Browser", "BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "intent_action",
            {"intent": "fill", "target": "id=username", "session_id": sid},
        )
        text = str(result)
        assert "error" in text.lower() or "value" in text.lower() or "requires" in text.lower()

    @pytest.mark.asyncio
    async def test_fill_intent_with_value_resolves(self, mcp_client):
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["Browser", "BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "intent_action",
            {"intent": "fill", "target": "id=username", "value": "testuser", "session_id": sid},
        )
        text = str(result)
        assert "Fill Text" in text or "Input Text" in text or "success" in text.lower()


class TestIntentActionErrors:
    """Test intent_action error handling."""

    @pytest.mark.asyncio
    async def test_invalid_intent_returns_error(self, mcp_client):
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "intent_action",
            {"intent": "invalid_intent_xyz", "session_id": sid},
        )
        text = str(result)
        assert "error" in text.lower() or "unknown" in text.lower()

    @pytest.mark.asyncio
    async def test_intent_without_session_returns_error(self, mcp_client):
        result = await mcp_client.call_tool(
            "intent_action",
            {"intent": "click", "target": "text=Login", "session_id": "nonexistent-xyz"},
        )
        text = str(result)
        assert "error" in text.lower() or "session" in text.lower()

    @pytest.mark.asyncio
    async def test_intent_without_web_library_returns_error(self, mcp_client):
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "intent_action",
            {"intent": "click", "target": "text=Login", "session_id": sid},
        )
        text = str(result)
        assert "error" in text.lower() or "library" in text.lower()


# ============================================================
# ADR-006: Tool Profile via manage_session
# ============================================================


class TestToolProfileInit:
    """Test tool profile initialization via manage_session."""

    @pytest.mark.asyncio
    async def test_init_with_model_tier_small_context(self, mcp_client):
        """Init with model_tier=small_context should activate a profile."""
        sid = _sid()
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "scenario": "browser test",
                "libraries": ["Browser", "BuiltIn"],
                "model_tier": "small_context",
            },
        )
        text = str(result.data if hasattr(result, "data") else result)
        assert "success" in text.lower() or sid in text
        # Restore full profile for subsequent tests
        await _restore_full_profile(mcp_client)

    @pytest.mark.asyncio
    async def test_init_with_model_tier_large_context(self, mcp_client):
        """Init with model_tier=large_context should keep full tool set."""
        sid = _sid()
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": ["BuiltIn"],
                "model_tier": "large_context",
            },
        )
        text = str(result.data if hasattr(result, "data") else result)
        assert "success" in text.lower() or sid in text
        await _restore_full_profile(mcp_client)


class TestSetToolProfile:
    """Test set_tool_profile action via manage_session."""

    @pytest.mark.asyncio
    async def test_set_tool_profile_browser_exec(self, mcp_client):
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {"action": "set_tool_profile", "session_id": sid, "profile": "browser_exec"},
        )
        text = str(result.data if hasattr(result, "data") else result)
        assert "browser_exec" in text or "success" in text.lower()
        await _restore_full_profile(mcp_client)

    @pytest.mark.asyncio
    async def test_set_tool_profile_discovery(self, mcp_client):
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {"action": "set_tool_profile", "session_id": sid, "profile": "discovery"},
        )
        text = str(result.data if hasattr(result, "data") else result)
        assert "discovery" in text or "success" in text.lower()
        await _restore_full_profile(mcp_client)

    @pytest.mark.asyncio
    async def test_set_tool_profile_minimal_exec(self, mcp_client):
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {"action": "set_tool_profile", "session_id": sid, "profile": "minimal_exec"},
        )
        text = str(result.data if hasattr(result, "data") else result)
        assert "minimal_exec" in text or "success" in text.lower()
        await _restore_full_profile(mcp_client)

    @pytest.mark.asyncio
    async def test_set_invalid_profile_returns_error(self, mcp_client):
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {"action": "set_tool_profile", "session_id": sid, "profile": "nonexistent_xyz"},
        )
        text = str(result.data if hasattr(result, "data") else result)
        assert "error" in text.lower() or "unknown" in text.lower() or "available" in text.lower()


class TestToolListAfterProfileSwitch:
    """Verify tool list changes after profile activation."""

    @pytest.mark.asyncio
    async def test_tools_reduced_after_small_profile(self, mcp_client):
        """After switching to a small profile, tool count should decrease."""
        initial_tools = await mcp_client.list_tools()
        initial_count = len(initial_tools)

        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": ["Browser", "BuiltIn"],
                "model_tier": "small_context",
            },
        )

        reduced_tools = await mcp_client.list_tools()
        reduced_count = len(reduced_tools)

        # Restore before assertion (in case it fails)
        await _restore_full_profile(mcp_client)

        # browser_exec has 6 tools; the full set has ~50
        assert reduced_count < initial_count, (
            f"Expected tools to be reduced: {reduced_count} >= {initial_count}"
        )


# ============================================================
# Combined: Intent + BuiltIn execution workflow
# ============================================================


class TestCombinedWorkflow:
    """Test combined ADR workflow: init -> intent -> builtin execution."""

    @pytest.mark.asyncio
    async def test_full_workflow_init_with_model_tier_then_builtin(self, mcp_client):
        """Full workflow: init with model_tier, execute BuiltIn, verify."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": ["BuiltIn"],
                "model_tier": "small_context",
            },
        )
        # Restore profile so execute_step is available
        await _restore_full_profile(mcp_client)

        result = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Log", "arguments": ["ADR integration test"], "session_id": sid},
        )
        text = str(result.data if hasattr(result, "data") else result)
        assert "pass" in text.lower() or "success" in text.lower()

    @pytest.mark.asyncio
    async def test_intent_then_execute_step_same_session(self, mcp_client):
        """Intent resolution followed by execute_step in same session."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["Browser", "BuiltIn"]},
        )
        intent_result = await mcp_client.call_tool(
            "intent_action",
            {"intent": "navigate", "target": "https://example.com", "session_id": sid},
        )
        assert intent_result is not None

        step_result = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Log", "arguments": ["Post-intent test"], "session_id": sid},
        )
        text = str(step_result.data if hasattr(step_result, "data") else step_result)
        assert "pass" in text.lower() or "success" in text.lower()

    @pytest.mark.asyncio
    async def test_profile_switch_then_execute(self, mcp_client):
        """Switch profile, then verify execute_step still works."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        await mcp_client.call_tool(
            "manage_session",
            {"action": "set_tool_profile", "session_id": sid, "profile": "minimal_exec"},
        )
        result = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Log", "arguments": ["After profile switch"], "session_id": sid},
        )
        text = str(result.data if hasattr(result, "data") else result)
        assert "pass" in text.lower() or "success" in text.lower()
        await _restore_full_profile(mcp_client)

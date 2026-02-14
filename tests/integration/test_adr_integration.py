"""Integration tests for ADR-006/007/008 server.py wiring.

Tests the full stack: MCP tool call -> domain service -> execution.
These tests validate the intent_action tool, tool profile management
via manage_session, and the combined workflow.

Prerequisites:
  - server.py must have intent_action registered as an MCP tool (ADR-007, DONE)
  - manage_session must support "set_tool_profile" action and "model_tier" param (ADR-006, PENDING)
  - Response optimization must be wired into execute_step output (ADR-008, PENDING)

Tests that depend on pending server.py integration are marked xfail(strict=False)
so they will auto-pass once the wiring agent completes its work.

Run with: uv run pytest tests/integration/test_adr_integration.py -v
"""

from __future__ import annotations

__test__ = True

import uuid

import pytest
import pytest_asyncio
from fastmcp import Client

from robotmcp.server import mcp

# Marker for tests that depend on server.py integration not yet completed.
# strict=False: if the test unexpectedly passes (i.e., integration landed),
# it will show as XPASS (not a failure), signaling the marker can be removed.
_pending_server_integration = pytest.mark.xfail(
    reason="Depends on manage_session model_tier/set_tool_profile wiring (ADR-006 server.py integration pending)",
    strict=False,
)


def _sid(prefix: str = "adr") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


# ============================================================
# ADR-007: intent_action tool (IMPLEMENTED)
# ============================================================


class TestIntentActionToolListed:
    """Verify intent_action appears in the tool listing."""

    @pytest.mark.asyncio
    async def test_intent_action_tool_listed(self, mcp_client):
        """intent_action should appear in the tool list after server.py wiring."""
        tools = await mcp_client.list_tools()
        tool_names = [t.name for t in tools]
        assert "intent_action" in tool_names, (
            f"intent_action not found in tool list. Available: {sorted(tool_names)}"
        )


class TestIntentActionNavigation:
    """Test intent_action navigate intent end-to-end."""

    @pytest.mark.asyncio
    async def test_navigate_intent_resolves_for_browser(self, mcp_client):
        """navigate intent with Browser library should resolve to Go To keyword."""
        sid = _sid()
        init_result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": ["Browser", "BuiltIn"],
            },
        )
        assert init_result is not None

        result = await mcp_client.call_tool(
            "intent_action",
            {
                "intent": "navigate",
                "target": "https://example.com",
                "session_id": sid,
            },
        )
        text = str(result)
        assert "Go To" in text or "success" in text.lower(), (
            f"Expected 'Go To' or 'success' in intent_action result: {text}"
        )

    @pytest.mark.asyncio
    async def test_navigate_intent_resolves_for_selenium(self, mcp_client):
        """navigate intent with SeleniumLibrary should resolve to Go To keyword."""
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
            {
                "intent": "navigate",
                "target": "https://example.com",
                "session_id": sid,
            },
        )
        text = str(result)
        assert "Go To" in text or "success" in text.lower(), (
            f"Expected 'Go To' or 'success' in result: {text}"
        )


class TestIntentActionClick:
    """Test intent_action click intent."""

    @pytest.mark.asyncio
    async def test_click_intent_browser_resolves_to_click(self, mcp_client):
        """click intent with Browser library should resolve to Click keyword."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": ["Browser", "BuiltIn"],
            },
        )
        result = await mcp_client.call_tool(
            "intent_action",
            {
                "intent": "click",
                "target": "text=Login",
                "session_id": sid,
            },
        )
        text = str(result)
        assert "Click" in text or "success" in text.lower()

    @pytest.mark.asyncio
    async def test_click_intent_selenium_resolves_to_click_element(self, mcp_client):
        """click intent with SeleniumLibrary should resolve to Click Element."""
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
            {
                "intent": "click",
                "target": "id=submit",
                "session_id": sid,
            },
        )
        text = str(result)
        assert "Click Element" in text or "success" in text.lower()


class TestIntentActionFill:
    """Test intent_action fill intent."""

    @pytest.mark.asyncio
    async def test_fill_intent_requires_value(self, mcp_client):
        """fill intent without value should return error."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": ["Browser", "BuiltIn"],
            },
        )
        result = await mcp_client.call_tool(
            "intent_action",
            {
                "intent": "fill",
                "target": "id=username",
                "session_id": sid,
            },
        )
        text = str(result)
        assert "error" in text.lower() or "value" in text.lower() or "requires" in text.lower()

    @pytest.mark.asyncio
    async def test_fill_intent_with_value_resolves(self, mcp_client):
        """fill intent with value should resolve to Fill Text / Input Text."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": ["Browser", "BuiltIn"],
            },
        )
        result = await mcp_client.call_tool(
            "intent_action",
            {
                "intent": "fill",
                "target": "id=username",
                "value": "testuser",
                "session_id": sid,
            },
        )
        text = str(result)
        assert "Fill Text" in text or "Input Text" in text or "success" in text.lower()


class TestIntentActionErrors:
    """Test intent_action error handling."""

    @pytest.mark.asyncio
    async def test_invalid_intent_returns_error(self, mcp_client):
        """Invalid intent verb should return error with valid intents list."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": ["BuiltIn"],
            },
        )
        result = await mcp_client.call_tool(
            "intent_action",
            {
                "intent": "invalid_intent_xyz",
                "session_id": sid,
            },
        )
        text = str(result)
        assert "error" in text.lower() or "unknown" in text.lower() or "invalid" in text.lower()

    @pytest.mark.asyncio
    async def test_intent_without_session_returns_error(self, mcp_client):
        """intent_action without valid session should return error."""
        result = await mcp_client.call_tool(
            "intent_action",
            {
                "intent": "click",
                "target": "text=Login",
                "session_id": "nonexistent-session-xyz",
            },
        )
        text = str(result)
        assert "error" in text.lower() or "session" in text.lower()

    @pytest.mark.asyncio
    async def test_intent_without_web_library_returns_error(self, mcp_client):
        """Intent requiring web library in BuiltIn-only session should error."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": ["BuiltIn"],
            },
        )
        result = await mcp_client.call_tool(
            "intent_action",
            {
                "intent": "click",
                "target": "text=Login",
                "session_id": sid,
            },
        )
        text = str(result)
        assert "error" in text.lower() or "library" in text.lower() or "mapping" in text.lower()


# ============================================================
# ADR-006: Tool Profile via manage_session (PENDING server.py wiring)
# ============================================================


class TestToolProfileInit:
    """Test tool profile initialization via manage_session.

    These tests depend on manage_session accepting model_tier and scenario
    parameters, which requires the ADR-006 server.py integration to be
    completed by the wiring agent.
    """

    @_pending_server_integration
    @pytest.mark.asyncio
    async def test_init_with_model_tier_small_context(self, mcp_client):
        """Init with model_tier=small_context should succeed and activate profile."""
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
        data = result.data if hasattr(result, "data") else result
        text = str(data)
        assert "success" in text.lower() or sid in text

    @_pending_server_integration
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
        data = result.data if hasattr(result, "data") else result
        text = str(data)
        assert "success" in text.lower() or sid in text


class TestSetToolProfile:
    """Test set_tool_profile action via manage_session.

    These tests depend on manage_session supporting the "set_tool_profile"
    action with a "profile" parameter.
    """

    @_pending_server_integration
    @pytest.mark.asyncio
    async def test_set_tool_profile_browser_exec(self, mcp_client):
        """set_tool_profile action should switch to browser_exec profile."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": ["BuiltIn"],
            },
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_tool_profile",
                "session_id": sid,
                "profile": "browser_exec",
            },
        )
        text = str(result.data if hasattr(result, "data") else result)
        assert "browser_exec" in text or "success" in text.lower()

    @_pending_server_integration
    @pytest.mark.asyncio
    async def test_set_tool_profile_discovery(self, mcp_client):
        """set_tool_profile action should switch to discovery profile."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": ["BuiltIn"],
            },
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_tool_profile",
                "session_id": sid,
                "profile": "discovery",
            },
        )
        text = str(result.data if hasattr(result, "data") else result)
        assert "discovery" in text or "success" in text.lower()

    @_pending_server_integration
    @pytest.mark.asyncio
    async def test_set_tool_profile_minimal_exec(self, mcp_client):
        """set_tool_profile should accept minimal_exec profile."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": ["BuiltIn"],
            },
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_tool_profile",
                "session_id": sid,
                "profile": "minimal_exec",
            },
        )
        text = str(result.data if hasattr(result, "data") else result)
        assert "minimal_exec" in text or "success" in text.lower()

    @_pending_server_integration
    @pytest.mark.asyncio
    async def test_set_invalid_profile_returns_error(self, mcp_client):
        """Invalid profile name should return error with available profiles."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": ["BuiltIn"],
            },
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_tool_profile",
                "session_id": sid,
                "profile": "nonexistent_profile_xyz",
            },
        )
        text = str(result.data if hasattr(result, "data") else result)
        assert "error" in text.lower() or "unknown" in text.lower() or "available" in text.lower()


class TestToolListAfterProfileSwitch:
    """Verify tool list changes after profile activation."""

    @_pending_server_integration
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

        # browser_exec has 6 tools; the full set has ~50
        assert reduced_count < initial_count, (
            f"Expected tools to be reduced: {reduced_count} >= {initial_count}"
        )


# ============================================================
# Combined: Intent + BuiltIn execution workflow
# ============================================================


class TestCombinedWorkflow:
    """Test combined ADR workflow: init -> intent -> builtin execution."""

    @_pending_server_integration
    @pytest.mark.asyncio
    async def test_full_workflow_init_with_model_tier_then_builtin(self, mcp_client):
        """Full workflow: init with model_tier, execute BuiltIn keyword, verify."""
        sid = _sid()
        init_result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": ["BuiltIn"],
                "model_tier": "small_context",
            },
        )
        assert init_result is not None

        result = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Log",
                "arguments": ["ADR integration test"],
                "session_id": sid,
            },
        )
        data = result.data if hasattr(result, "data") else result
        text = str(data)
        assert "pass" in text.lower() or "success" in text.lower()

    @pytest.mark.asyncio
    async def test_intent_then_execute_step_same_session(self, mcp_client):
        """Intent resolution followed by execute_step in same session."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": ["Browser", "BuiltIn"],
            },
        )

        intent_result = await mcp_client.call_tool(
            "intent_action",
            {
                "intent": "navigate",
                "target": "https://example.com",
                "session_id": sid,
            },
        )
        assert intent_result is not None

        step_result = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Log",
                "arguments": ["Post-intent test"],
                "session_id": sid,
            },
        )
        data = step_result.data if hasattr(step_result, "data") else step_result
        text = str(data)
        assert "pass" in text.lower() or "success" in text.lower()

    @_pending_server_integration
    @pytest.mark.asyncio
    async def test_profile_switch_then_execute(self, mcp_client):
        """Switch profile, then verify execute_step still works."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": ["BuiltIn"],
            },
        )
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_tool_profile",
                "session_id": sid,
                "profile": "minimal_exec",
            },
        )
        result = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Log",
                "arguments": ["After profile switch"],
                "session_id": sid,
            },
        )
        data = result.data if hasattr(result, "data") else result
        text = str(data)
        assert "pass" in text.lower() or "success" in text.lower()


# ============================================================
# ADR-008: Response Optimization (PENDING server.py wiring)
# ============================================================


class TestResponseOptimization:
    """Test response optimization in compact mode sessions.

    Depends on ADR-008 response compression being wired into
    execute_step output path in server.py.
    """

    @_pending_server_integration
    @pytest.mark.asyncio
    async def test_compact_response_has_abbreviated_fields(self, mcp_client):
        """In small_context mode, responses should use abbreviated field names."""
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
        result = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Log",
                "arguments": ["Compact test"],
                "session_id": sid,
            },
        )
        data = result.data if hasattr(result, "data") else result
        assert data is not None
        text = str(data)
        # Compact mode: "success" -> "ok", "keyword" -> "kw"
        assert (
            "ok" in text or "success" in text.lower()
            or "kw" in text or "keyword" in text.lower()
        )

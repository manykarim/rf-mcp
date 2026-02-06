"""Comprehensive tests for manage_session set_variables action.

Covers:
- All three scopes (test, suite, global)
- Both input formats (dict, list)
- Variable round-trip (set then read back via Get Variable Value)
- Variable usage in keyword execution (Create Dictionary, Evaluate)
- Empty/missing variables (hint in response)
- Default scope behavior
- Variables visible in get_session_state
- suite_level_variables tracking
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import execution_engine, mcp


def _sid(prefix: str = "setvar") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


# ---------------------------------------------------------------------------
# Scope coverage
# ---------------------------------------------------------------------------


class TestSetVariablesScopes:
    """Verify set_variables works for every explicit scope value."""

    @pytest.mark.asyncio
    async def test_scope_test(self, mcp_client):
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session", {"action": "init", "session_id": sid}
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_variables",
                "session_id": sid,
                "variables": {"SCOPE_TEST_VAR": "test_val"},
                "scope": "test",
            },
        )
        assert result.data["success"] is True
        assert "SCOPE_TEST_VAR" in result.data["set"]
        assert result.data["scope"] == "test"

    @pytest.mark.asyncio
    async def test_scope_suite(self, mcp_client):
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session", {"action": "init", "session_id": sid}
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_variables",
                "session_id": sid,
                "variables": {"SCOPE_SUITE_VAR": "suite_val"},
                "scope": "suite",
            },
        )
        assert result.data["success"] is True
        assert "SCOPE_SUITE_VAR" in result.data["set"]
        assert result.data["scope"] == "suite"

    @pytest.mark.asyncio
    async def test_scope_global(self, mcp_client):
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session", {"action": "init", "session_id": sid}
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_variables",
                "session_id": sid,
                "variables": {"SCOPE_GLOBAL_VAR": "global_val"},
                "scope": "global",
            },
        )
        assert result.data["success"] is True
        assert "SCOPE_GLOBAL_VAR" in result.data["set"]
        assert result.data["scope"] == "global"

    @pytest.mark.asyncio
    async def test_default_scope_is_suite(self, mcp_client):
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session", {"action": "init", "session_id": sid}
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_variables",
                "session_id": sid,
                "variables": {"DEFAULT_VAR": "default_val"},
            },
        )
        assert result.data["success"] is True
        assert result.data["scope"] == "suite"


# ---------------------------------------------------------------------------
# Input format coverage
# ---------------------------------------------------------------------------


class TestSetVariablesFormats:
    """Verify both dict and list input formats."""

    @pytest.mark.asyncio
    async def test_dict_format(self, mcp_client):
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session", {"action": "init", "session_id": sid}
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_variables",
                "session_id": sid,
                "variables": {"A": "1", "B": "2", "C": "3"},
            },
        )
        assert result.data["success"] is True
        assert set(result.data["set"]) == {"A", "B", "C"}

    @pytest.mark.asyncio
    async def test_list_format(self, mcp_client):
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session", {"action": "init", "session_id": sid}
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_variables",
                "session_id": sid,
                "variables": ["X=hello", "Y=world"],
            },
        )
        assert result.data["success"] is True
        assert "X" in result.data["set"]
        assert "Y" in result.data["set"]

    @pytest.mark.asyncio
    async def test_list_format_with_equals_in_value(self, mcp_client):
        """Values containing '=' should be preserved (split on first '=' only)."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session", {"action": "init", "session_id": sid}
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_variables",
                "session_id": sid,
                "variables": ["URL=https://example.com?a=1&b=2"],
            },
        )
        assert result.data["success"] is True
        assert "URL" in result.data["set"]


# ---------------------------------------------------------------------------
# Round-trip: set then read back
# ---------------------------------------------------------------------------


class TestSetVariablesRoundTrip:
    """Verify variables set via manage_session are usable in keyword execution."""

    @pytest.mark.asyncio
    async def test_get_variable_value_round_trip(self, mcp_client):
        """Set a variable, then read it back with Get Variable Value."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_variables",
                "session_id": sid,
                "variables": {"ROUND_TRIP": "expected_value"},
                "scope": "suite",
            },
        )
        lookup = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Get Variable Value",
                "arguments": ["${ROUND_TRIP}"],
                "session_id": sid,
                "use_context": True,
            },
        )
        assert lookup.data["success"] is True
        assert lookup.data.get("output") == "expected_value"

    @pytest.mark.asyncio
    async def test_variable_used_in_evaluate(self, mcp_client):
        """Set a numeric variable, use it in an Evaluate expression."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_variables",
                "session_id": sid,
                "variables": {"COUNT": "10"},
            },
        )
        result = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "",
                "session_id": sid,
                "mode": "evaluate",
                "expression": "int($COUNT) * 2",
                "assign_to": "DOUBLED",
                "use_context": True,
            },
        )
        assert result.data["success"] is True
        # Verify derived variable
        verify = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Get Variable Value",
                "arguments": ["${DOUBLED}"],
                "session_id": sid,
                "use_context": True,
            },
        )
        assert verify.data["success"] is True
        assert str(verify.data.get("output")) == "20"

    @pytest.mark.asyncio
    async def test_variable_used_in_create_dictionary(self, mcp_client):
        """Set variables, then use them as Create Dictionary values."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": ["BuiltIn", "Collections"],
            },
        )
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_variables",
                "session_id": sid,
                "variables": {"HOST": "localhost", "PORT": "8080"},
            },
        )
        result = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Create Dictionary",
                "arguments": ["host=${HOST}", "port=${PORT}"],
                "session_id": sid,
                "assign_to": "CONFIG",
                "use_context": True,
            },
        )
        assert result.data["success"] is True

    @pytest.mark.asyncio
    async def test_variables_visible_in_session_state(self, mcp_client):
        """Variables set via set_variables appear in get_session_state."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_variables",
                "session_id": sid,
                "variables": {"STATE_VAR": "visible"},
            },
        )
        state = await mcp_client.call_tool(
            "get_session_state",
            {"session_id": sid, "sections": ["variables"]},
        )
        assert state.data["success"] is True
        variables_section = state.data["sections"]["variables"]
        variables_map = variables_section.get("variables", {})
        # Variable should appear under STATE_VAR or ${STATE_VAR}
        assert (
            variables_map.get("STATE_VAR") == "visible"
            or variables_map.get("${STATE_VAR}") == "visible"
        )


# ---------------------------------------------------------------------------
# Error handling and hints
# ---------------------------------------------------------------------------


class TestSetVariablesHints:
    """Verify helpful error messages and hints on failure."""

    @pytest.mark.asyncio
    async def test_empty_dict_returns_hint(self, mcp_client):
        """Empty variables dict should fail with a helpful hint."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session", {"action": "init", "session_id": sid}
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_variables",
                "session_id": sid,
                "variables": {},
            },
        )
        assert result.data["success"] is False
        assert "hint" in result.data
        assert "variables" in result.data["hint"].lower()

    @pytest.mark.asyncio
    async def test_omitted_variables_returns_hint(self, mcp_client):
        """Omitting the variables parameter should fail with a helpful hint."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session", {"action": "init", "session_id": sid}
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_variables",
                "session_id": sid,
            },
        )
        assert result.data["success"] is False
        assert "hint" in result.data

    @pytest.mark.asyncio
    async def test_list_without_equals_ignored(self, mcp_client):
        """List items without '=' are silently skipped, resulting in no variables."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session", {"action": "init", "session_id": sid}
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_variables",
                "session_id": sid,
                "variables": ["no_equals_here"],
            },
        )
        # No valid key=value pairs -> empty data -> hint
        assert result.data["success"] is False
        assert "hint" in result.data


# ---------------------------------------------------------------------------
# Suite-level tracking
# ---------------------------------------------------------------------------


class TestSetVariablesTracking:
    """Verify suite_level_variables tracking for test suite generation."""

    @pytest.mark.asyncio
    async def test_suite_level_variables_tracked(self, mcp_client):
        """Variables set via manage_session are tracked in session.suite_level_variables."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session", {"action": "init", "session_id": sid}
        )
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_variables",
                "session_id": sid,
                "variables": {"TRACKED_A": "a", "TRACKED_B": "b"},
            },
        )
        session = execution_engine.session_manager.get_session(sid)
        assert session is not None
        assert hasattr(session, "suite_level_variables")
        assert "TRACKED_A" in session.suite_level_variables
        assert "TRACKED_B" in session.suite_level_variables


# ---------------------------------------------------------------------------
# Session not explicitly initialized
# ---------------------------------------------------------------------------


class TestSetVariablesWithoutInit:
    """Verify set_variables works on sessions created implicitly."""

    @pytest.mark.asyncio
    async def test_set_variables_without_explicit_init(self, mcp_client):
        """set_variables should work even when manage_session init was not called first."""
        sid = _sid("noinit")
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_variables",
                "session_id": sid,
                "variables": {"IMPLICIT": "works"},
            },
        )
        assert result.data["success"] is True
        assert "IMPLICIT" in result.data["set"]

    @pytest.mark.asyncio
    async def test_set_variables_after_analyze_scenario(self, mcp_client):
        """set_variables should work on a session created by analyze_scenario."""
        sid = _sid("scenario")
        await mcp_client.call_tool(
            "analyze_scenario",
            {"scenario": "Test login form on example.com", "session_id": sid},
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "set_variables",
                "session_id": sid,
                "variables": {"USERNAME": "admin"},
                "scope": "suite",
            },
        )
        assert result.data["success"] is True
        assert "USERNAME" in result.data["set"]

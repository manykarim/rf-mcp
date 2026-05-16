"""Tests for session-scoped tool profile activation (P1).

Verifies that:
- Activating a restricted profile in session A does NOT mutate the FastMCP registry.
- Session B with no binding still sees the full profile.
- The SessionToolProfileRegistry correctly tracks per-session bindings.
- bind_session_profile works without side effects to other sessions.

Run with: uv run pytest tests/unit/test_tool_profile_session_scoped.py -v
"""

__test__ = True

import pytest

from robotmcp.domains.tool_profile.aggregates import ProfilePresets
from robotmcp.domains.tool_profile.entities import ToolDescriptor
from robotmcp.domains.tool_profile.services import (
    SessionToolProfileRegistry,
    ToolProfileManager,
)
from robotmcp.domains.tool_profile.value_objects import ToolTag


# =============================================================================
# Fixtures
# =============================================================================


class MockToolManagerPort:
    """Records calls — used to assert that profile binding does NOT call remove_tool."""

    def __init__(self):
        self.visible = frozenset(ProfilePresets.ALL_TOOLS)
        self.removed = []
        self.added = []

    async def remove_tool(self, name):
        self.removed.append(name)
        self.visible = self.visible - {name}

    async def add_tool_with_description(self, name, desc, schema):
        self.added.append(name)
        self.visible = self.visible | {name}

    async def get_visible_tool_names(self):
        return self.visible

    async def swap_tool_description(self, name, desc, schema):
        pass


@pytest.fixture
def descriptors():
    return {
        t: ToolDescriptor(
            tool_name=t,
            tags=frozenset({ToolTag.CORE}),
            description_full="x",
            description_compact="x",
            description_minimal="x",
            schema_full={},
            token_estimate_full=10,
            token_estimate_compact=5,
            token_estimate_minimal=3,
        )
        for t in ProfilePresets.ALL_TOOLS
    }


@pytest.fixture
def mock_port():
    return MockToolManagerPort()


@pytest.fixture
def manager(mock_port, descriptors):
    return ToolProfileManager(mock_port, descriptors)


# =============================================================================
# SessionToolProfileRegistry unit tests
# =============================================================================


class TestSessionToolProfileRegistry:
    """Unit tests for the standalone SessionToolProfileRegistry."""

    @pytest.fixture
    def registry(self):
        profiles = {
            "full": ProfilePresets.full(),
            "browser_exec": ProfilePresets.browser_exec(),
            "api_exec": ProfilePresets.api_exec(),
        }
        return SessionToolProfileRegistry(profiles)

    def test_default_profile_is_full_when_no_binding(self, registry):
        profile = registry.get_profile("session-a")
        assert profile.name == "full"

    def test_bind_changes_profile_for_that_session(self, registry):
        registry.bind("session-a", "browser_exec")
        assert registry.get_profile("session-a").name == "browser_exec"

    def test_binding_session_a_does_not_affect_session_b(self, registry):
        registry.bind("session-a", "browser_exec")
        assert registry.get_profile("session-b").name == "full"

    def test_is_tool_allowed_respects_binding(self, registry):
        registry.bind("session-a", "browser_exec")
        # build_test_suite is now in browser_exec (P3 fix)
        assert registry.is_tool_allowed("session-a", "build_test_suite") is True
        # run_test_suite is NOT in browser_exec
        assert registry.is_tool_allowed("session-a", "run_test_suite") is False

    def test_is_tool_allowed_default_session_allows_all(self, registry):
        # No binding = full profile = all tools allowed
        assert registry.is_tool_allowed("session-no-binding", "build_test_suite") is True
        assert registry.is_tool_allowed("session-no-binding", "run_test_suite") is True

    def test_unbind_reverts_to_default(self, registry):
        registry.bind("session-a", "browser_exec")
        registry.unbind("session-a")
        assert registry.get_profile("session-a").name == "full"

    def test_bind_unknown_profile_raises_key_error(self, registry):
        with pytest.raises(KeyError, match="Unknown profile"):
            registry.bind("session-a", "nonexistent_profile")

    def test_list_bindings_snapshot(self, registry):
        registry.bind("s1", "browser_exec")
        registry.bind("s2", "api_exec")
        bindings = registry.list_bindings()
        assert bindings == {"s1": "browser_exec", "s2": "api_exec"}

    def test_get_profile_name_returns_correct_name(self, registry):
        registry.bind("session-a", "api_exec")
        assert registry.get_profile_name("session-a") == "api_exec"
        assert registry.get_profile_name("session-b") == "full"


# =============================================================================
# ToolProfileManager.bind_session_profile — no global mutation
# =============================================================================


class TestBindSessionProfileNoGlobalMutation:
    """bind_session_profile must not call remove_tool on the FastMCP port."""

    def test_bind_session_profile_does_not_call_remove_tool(self, manager, mock_port):
        profile = manager.bind_session_profile("session-a", "browser_exec")
        assert profile.name == "browser_exec"
        assert mock_port.removed == [], (
            "bind_session_profile must not remove tools from FastMCP registry"
        )
        assert mock_port.added == [], (
            "bind_session_profile must not add tools to FastMCP registry"
        )

    def test_bind_session_profile_does_not_affect_other_session(self, manager):
        manager.bind_session_profile("session-a", "browser_exec")
        assert manager.is_tool_allowed_for_session("session-b", "run_test_suite") is True

    def test_bind_session_profile_restricts_calling_session(self, manager):
        manager.bind_session_profile("session-a", "api_exec")
        # run_test_suite is not in api_exec
        assert manager.is_tool_allowed_for_session("session-a", "run_test_suite") is False

    def test_two_sessions_independent_profiles(self, manager, mock_port):
        manager.bind_session_profile("session-a", "browser_exec")
        manager.bind_session_profile("session-b", "api_exec")
        # No FastMCP mutation
        assert mock_port.removed == []
        # session-a cannot access execute_flow (not in browser_exec)
        assert manager.is_tool_allowed_for_session("session-a", "execute_flow") is False
        # session-b cannot access get_locator_guidance (not in api_exec)
        assert manager.is_tool_allowed_for_session("session-b", "get_locator_guidance") is False
        # session-c (unbound) can access everything
        assert manager.is_tool_allowed_for_session("session-c", "execute_flow") is True
        assert manager.is_tool_allowed_for_session("session-c", "run_test_suite") is True

    def test_bind_unknown_profile_raises_key_error(self, manager):
        with pytest.raises(KeyError, match="Unknown profile"):
            manager.bind_session_profile("session-a", "bogus_profile")

    def test_empty_session_id_uses_full_profile_by_default(self, manager):
        assert manager.is_tool_allowed_for_session("", "build_test_suite") is True
        assert manager.is_tool_allowed_for_session("", "run_test_suite") is True


# =============================================================================
# profile_gated_error — friendly error structure (P13)
# =============================================================================


class TestProfileGatedError:
    """profile_gated_error returns a structured dict with hint."""

    def test_gated_error_has_required_fields(self, manager):
        manager.bind_session_profile("session-a", "browser_exec")
        err = manager.profile_gated_error("session-a", "run_test_suite")
        assert err["success"] is False
        assert err["error"] == "profile_disabled"
        assert "run_test_suite" in err["tool"]
        assert "browser_exec" in err["profile"]
        assert "manage_session" in err["hint"]
        assert "set_tool_profile" in err["hint"]
        assert "full" in err["hint"]

    def test_gated_error_names_blocked_tool(self, manager):
        manager.bind_session_profile("session-a", "minimal_exec")
        err = manager.profile_gated_error("session-a", "execute_flow")
        assert err["tool"] == "execute_flow"

    def test_no_error_when_tool_is_allowed(self, manager):
        manager.bind_session_profile("session-a", "browser_exec")
        assert manager.is_tool_allowed_for_session("session-a", "build_test_suite") is True


# =============================================================================
# get_session_registry accessor
# =============================================================================


class TestGetSessionRegistry:
    def test_get_session_registry_returns_registry(self, manager):
        reg = manager.get_session_registry()
        assert isinstance(reg, SessionToolProfileRegistry)

    def test_registry_shared_between_accessor_calls(self, manager):
        reg1 = manager.get_session_registry()
        reg2 = manager.get_session_registry()
        assert reg1 is reg2

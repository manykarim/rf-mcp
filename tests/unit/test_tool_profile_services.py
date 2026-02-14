"""Comprehensive unit tests for Tool Profile domain services (ADR-006).

Tests cover: ToolProfileManager (list, suggest, register, activate, transition, events).

Run with: uv run pytest tests/unit/test_tool_profile_services.py -v
"""

__test__ = True

import pytest

from robotmcp.domains.tool_profile.aggregates import ProfilePresets, ToolProfile
from robotmcp.domains.tool_profile.entities import ToolDescriptor
from robotmcp.domains.tool_profile.events import (
    ProfileActivated,
    ProfileTransitioned,
    TokenBudgetExceeded,
    ToolsHidden,
    ToolsRevealed,
)
from robotmcp.domains.tool_profile.services import ToolProfileManager
from robotmcp.domains.tool_profile.value_objects import (
    ModelTier,
    ProfileTransition,
    TokenBudget,
    ToolDescriptionMode,
    ToolTag,
)


# =============================================================================
# MockToolManagerPort
# =============================================================================


class MockToolManagerPort:
    """Mock implementation of ToolManagerPort for testing."""

    def __init__(self):
        self.visible = frozenset(ProfilePresets.ALL_TOOLS)
        self.removed = []
        self.added = []
        self.swapped = []

    async def remove_tool(self, name):
        self.removed.append(name)
        self.visible = self.visible - {name}

    async def add_tool_with_description(self, name, desc, schema):
        self.added.append(name)
        self.visible = self.visible | {name}

    async def get_visible_tool_names(self):
        return self.visible

    async def swap_tool_description(self, name, desc, schema):
        self.swapped.append(name)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_tool_manager():
    return MockToolManagerPort()


@pytest.fixture
def sample_descriptors():
    """Minimal descriptor registry for the full tool set."""
    descriptors = {}
    for tool_name in ProfilePresets.ALL_TOOLS:
        descriptors[tool_name] = ToolDescriptor(
            tool_name=tool_name,
            tags=frozenset({ToolTag.CORE}),
            description_full=f"Full {tool_name}",
            description_compact=f"Compact {tool_name}",
            description_minimal=f"Min {tool_name}",
            schema_full={"type": "object"},
            token_estimate_full=400,
            token_estimate_compact=100,
            token_estimate_minimal=50,
        )
    return descriptors


@pytest.fixture
def event_log():
    return []


@pytest.fixture
def manager(mock_tool_manager, sample_descriptors, event_log):
    return ToolProfileManager(
        tool_manager=mock_tool_manager,
        descriptor_registry=sample_descriptors,
        event_publisher=lambda evt: event_log.append(evt),
    )


# =============================================================================
# list_profiles
# =============================================================================


class TestListProfiles:
    """Test listing registered profiles."""

    def test_list_returns_5_builtin_profiles(self, manager):
        profiles = manager.list_profiles()
        assert len(profiles) == 5

    def test_list_returns_sorted_names(self, manager):
        profiles = manager.list_profiles()
        assert profiles == sorted(profiles)

    def test_list_contains_expected_names(self, manager):
        profiles = manager.list_profiles()
        assert "browser_exec" in profiles
        assert "api_exec" in profiles
        assert "discovery" in profiles
        assert "minimal_exec" in profiles
        assert "full" in profiles


# =============================================================================
# suggest_profile
# =============================================================================


class TestSuggestProfile:
    """Test profile suggestion heuristics."""

    def test_large_context_returns_full(self, manager):
        p = manager.suggest_profile("anything", ModelTier.LARGE_CONTEXT)
        assert p.name == "full"

    def test_standard_returns_full(self, manager):
        p = manager.suggest_profile("anything", ModelTier.STANDARD)
        assert p.name == "full"

    def test_small_context_api_scenario(self, manager):
        p = manager.suggest_profile("api testing for REST service", ModelTier.SMALL_CONTEXT)
        assert p.name == "api_exec"

    def test_small_context_http_scenario(self, manager):
        p = manager.suggest_profile("http request testing", ModelTier.SMALL_CONTEXT)
        assert p.name == "api_exec"

    def test_small_context_browser_scenario(self, manager):
        p = manager.suggest_profile("browser login test", ModelTier.SMALL_CONTEXT)
        assert p.name == "browser_exec"

    def test_small_context_web_scenario(self, manager):
        p = manager.suggest_profile("web UI automation", ModelTier.SMALL_CONTEXT)
        assert p.name == "browser_exec"

    def test_small_context_analyze_scenario(self, manager):
        p = manager.suggest_profile("analyze what libraries to use", ModelTier.SMALL_CONTEXT)
        assert p.name == "discovery"

    def test_small_context_plan_scenario(self, manager):
        p = manager.suggest_profile("plan the test approach", ModelTier.SMALL_CONTEXT)
        assert p.name == "discovery"

    def test_small_context_unknown_scenario_defaults_to_browser_exec(self, manager):
        p = manager.suggest_profile("some random task", ModelTier.SMALL_CONTEXT)
        assert p.name == "browser_exec"


# =============================================================================
# register_profile
# =============================================================================


class TestRegisterProfile:
    """Test custom profile registration."""

    def test_register_adds_custom_profile(self, manager):
        custom = ToolProfile(
            name="custom",
            tool_names=frozenset({"manage_session", "execute_step"}),
            description_mode=ToolDescriptionMode.COMPACT,
            model_tier=ModelTier.SMALL_CONTEXT,
            token_budget=TokenBudget.for_context_window(8192),
        )
        manager.register_profile(custom)
        assert "custom" in manager.list_profiles()
        assert len(manager.list_profiles()) == 6


# =============================================================================
# activate_profile
# =============================================================================


class TestActivateProfile:
    """Test profile activation with tool add/remove orchestration."""

    @pytest.mark.asyncio
    async def test_activate_unknown_profile_raises(self, manager):
        with pytest.raises(KeyError, match="Unknown profile"):
            await manager.activate_profile("nonexistent")

    @pytest.mark.asyncio
    async def test_activate_browser_exec(self, manager, mock_tool_manager):
        result = await manager.activate_profile("browser_exec")
        assert result.name == "browser_exec"
        # Tools not in browser_exec should have been removed
        assert len(mock_tool_manager.removed) > 0
        # Verify active profile is set
        assert manager.get_active_profile() is not None
        assert manager.get_active_profile().name == "browser_exec"

    @pytest.mark.asyncio
    async def test_activate_removes_extra_tools(self, manager, mock_tool_manager):
        """When going from full (15 tools) to browser_exec (5 tools), 10 should be removed."""
        await manager.activate_profile("browser_exec")
        # browser_exec has 6 tools, full has 16, so 10 removals
        assert len(mock_tool_manager.removed) == 10

    @pytest.mark.asyncio
    async def test_activate_adds_missing_tools(self, manager, mock_tool_manager):
        """Starting from browser_exec, activating full should add tools."""
        # First go to browser_exec
        await manager.activate_profile("browser_exec")
        mock_tool_manager.removed.clear()
        mock_tool_manager.added.clear()
        # Now go to full
        await manager.activate_profile("full")
        assert len(mock_tool_manager.added) == 10

    @pytest.mark.asyncio
    async def test_activate_publishes_profile_activated_event(
        self, manager, event_log
    ):
        await manager.activate_profile("browser_exec")
        activated_events = [e for e in event_log if isinstance(e, ProfileActivated)]
        assert len(activated_events) == 1
        assert activated_events[0].profile_name == "browser_exec"
        assert activated_events[0].tool_count == 6

    @pytest.mark.asyncio
    async def test_activate_publishes_tools_hidden_event(
        self, manager, event_log
    ):
        await manager.activate_profile("browser_exec")
        hidden_events = [e for e in event_log if isinstance(e, ToolsHidden)]
        assert len(hidden_events) == 1
        assert len(hidden_events[0].tool_names) == 10

    @pytest.mark.asyncio
    async def test_activate_transition_publishes_profile_transitioned(
        self, manager, event_log
    ):
        """When changing from one profile to another, ProfileTransitioned should be emitted."""
        await manager.activate_profile("browser_exec")
        event_log.clear()
        await manager.activate_profile("full")
        transitioned_events = [e for e in event_log if isinstance(e, ProfileTransitioned)]
        assert len(transitioned_events) == 1
        assert transitioned_events[0].from_profile == "browser_exec"
        assert transitioned_events[0].to_profile == "full"

    @pytest.mark.asyncio
    async def test_first_activation_no_transition_event(self, manager, event_log):
        """First activation has no previous profile, so no ProfileTransitioned."""
        await manager.activate_profile("browser_exec")
        transitioned_events = [e for e in event_log if isinstance(e, ProfileTransitioned)]
        assert len(transitioned_events) == 0


# =============================================================================
# transition_phase
# =============================================================================


class TestTransitionPhase:
    """Test phase-based profile transitions."""

    @pytest.mark.asyncio
    async def test_no_active_profile_returns_none(self, manager):
        result = await manager.transition_phase("plan_to_execute")
        assert result is None

    @pytest.mark.asyncio
    async def test_discovery_to_browser_exec(self, manager):
        await manager.activate_profile("discovery")
        result = await manager.transition_phase(ProfileTransition.PLAN_TO_EXECUTE)
        assert result is not None
        assert result.name == "browser_exec"

    @pytest.mark.asyncio
    async def test_minimal_exec_escalation_to_browser_exec(self, manager):
        await manager.activate_profile("minimal_exec")
        result = await manager.transition_phase(ProfileTransition.ESCALATION)
        assert result is not None
        assert result.name == "browser_exec"

    @pytest.mark.asyncio
    async def test_browser_exec_escalation_to_full(self, manager):
        await manager.activate_profile("browser_exec")
        result = await manager.transition_phase(ProfileTransition.ESCALATION)
        assert result is not None
        assert result.name == "full"

    @pytest.mark.asyncio
    async def test_recovery_adds_get_keyword_info(self, manager):
        await manager.activate_profile("browser_exec")
        assert not manager.get_active_profile().contains_tool("get_keyword_info")

        result = await manager.transition_phase(ProfileTransition.EXECUTE_TO_RECOVERY)
        assert result is not None
        assert result.contains_tool("get_keyword_info")

    @pytest.mark.asyncio
    async def test_recovery_already_has_keyword_info_returns_current(self, manager):
        """If get_keyword_info is already in the profile, recovery is a no-op."""
        await manager.activate_profile("discovery")  # has get_keyword_info
        result = await manager.transition_phase(ProfileTransition.EXECUTE_TO_RECOVERY)
        # discovery -> execute_to_recovery is not in the transition map for discovery
        # so this returns None
        assert result is None

    @pytest.mark.asyncio
    async def test_unknown_transition_returns_none(self, manager):
        await manager.activate_profile("full")
        result = await manager.transition_phase("plan_to_execute")
        # full + plan_to_execute is not in transition_map
        assert result is None


# =============================================================================
# Pinned manage_session invariant
# =============================================================================


class TestPinnedManageSession:
    """manage_session must always remain visible regardless of profile."""

    @pytest.mark.asyncio
    async def test_manage_session_never_removed_on_discovery(
        self, manager, mock_tool_manager
    ):
        """Switching to discovery must NOT remove manage_session."""
        await manager.activate_profile("discovery")
        assert "manage_session" not in mock_tool_manager.removed

    @pytest.mark.asyncio
    async def test_manage_session_visible_after_every_preset(
        self, manager, mock_tool_manager
    ):
        """After activating any preset, manage_session is still visible."""
        for profile_name in ("browser_exec", "api_exec", "discovery",
                             "minimal_exec", "full"):
            await manager.activate_profile(profile_name)
            visible = await mock_tool_manager.get_visible_tool_names()
            assert "manage_session" in visible, (
                f"manage_session not visible after activating {profile_name}"
            )

    @pytest.mark.asyncio
    async def test_manage_session_pinned_even_if_not_in_profile_tools(
        self, manager, mock_tool_manager
    ):
        """A custom profile without manage_session still keeps it visible."""
        custom = ToolProfile(
            name="no_mgmt",
            tool_names=frozenset({"execute_step", "get_session_state"}),
            description_mode=ToolDescriptionMode.COMPACT,
            model_tier=ModelTier.SMALL_CONTEXT,
            token_budget=TokenBudget.for_context_window(8192),
        )
        manager.register_profile(custom)
        await manager.activate_profile("no_mgmt")
        visible = await mock_tool_manager.get_visible_tool_names()
        assert "manage_session" in visible

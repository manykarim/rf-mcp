"""Comprehensive unit tests for Tool Profile aggregates (ADR-006).

Tests cover: ToolProfile aggregate root, ProfilePresets factory methods.

Run with: uv run pytest tests/unit/test_tool_profile_aggregate.py -v
"""

__test__ = True

import pytest

from robotmcp.domains.tool_profile.aggregates import ProfilePresets, ToolProfile
from robotmcp.domains.tool_profile.entities import ToolDescriptor
from robotmcp.domains.tool_profile.value_objects import (
    ModelTier,
    TokenBudget,
    ToolDescriptionMode,
    ToolTag,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_profile():
    """Minimal valid profile for testing."""
    return ToolProfile(
        name="test_profile",
        tool_names=frozenset({"manage_session", "execute_step"}),
        description_mode=ToolDescriptionMode.COMPACT,
        model_tier=ModelTier.SMALL_CONTEXT,
        token_budget=TokenBudget.for_context_window(8192),
    )


@pytest.fixture
def sample_descriptors():
    """Sample ToolDescriptor registry for budget testing."""
    return {
        "manage_session": ToolDescriptor(
            tool_name="manage_session",
            tags=frozenset({ToolTag.CORE}),
            description_full="Full description of manage_session",
            description_compact="Session mgmt",
            description_minimal="Session",
            schema_full={"type": "object"},
            token_estimate_full=500,
            token_estimate_compact=150,
            token_estimate_minimal=80,
        ),
        "execute_step": ToolDescriptor(
            tool_name="execute_step",
            tags=frozenset({ToolTag.CORE, ToolTag.EXECUTION}),
            description_full="Full description of execute_step",
            description_compact="Run keyword",
            description_minimal="Run",
            schema_full={"type": "object"},
            token_estimate_full=600,
            token_estimate_compact=200,
            token_estimate_minimal=100,
        ),
        "get_session_state": ToolDescriptor(
            tool_name="get_session_state",
            tags=frozenset({ToolTag.CORE}),
            description_full="Full description of get_session_state",
            description_compact="Get state",
            description_minimal="State",
            schema_full={"type": "object"},
            token_estimate_full=400,
            token_estimate_compact=120,
            token_estimate_minimal=60,
        ),
    }


# =============================================================================
# ToolProfile Aggregate
# =============================================================================


class TestToolProfileConstruction:
    """Test ToolProfile construction and invariants."""

    def test_valid_construction(self, basic_profile):
        assert basic_profile.name == "test_profile"
        assert len(basic_profile.tool_names) == 2

    def test_empty_tool_names_raises(self):
        with pytest.raises(ValueError, match="at least one tool"):
            ToolProfile(
                name="empty",
                tool_names=frozenset(),
                description_mode=ToolDescriptionMode.COMPACT,
                model_tier=ModelTier.STANDARD,
            )

    def test_small_context_without_budget_raises(self):
        with pytest.raises(ValueError, match="SMALL_CONTEXT profiles require"):
            ToolProfile(
                name="no_budget",
                tool_names=frozenset({"manage_session"}),
                description_mode=ToolDescriptionMode.COMPACT,
                model_tier=ModelTier.SMALL_CONTEXT,
                token_budget=None,
            )

    def test_large_context_without_budget_is_valid(self):
        profile = ToolProfile(
            name="large",
            tool_names=frozenset({"manage_session"}),
            description_mode=ToolDescriptionMode.FULL,
            model_tier=ModelTier.LARGE_CONTEXT,
            token_budget=None,
        )
        assert profile.token_budget is None

    def test_standard_without_budget_is_valid(self):
        profile = ToolProfile(
            name="standard",
            tool_names=frozenset({"manage_session"}),
            description_mode=ToolDescriptionMode.FULL,
            model_tier=ModelTier.STANDARD,
            token_budget=None,
        )
        assert profile.token_budget is None


class TestToolProfileProperties:
    """Test ToolProfile property methods."""

    def test_tool_count(self, basic_profile):
        assert basic_profile.tool_count == 2

    def test_contains_tool_true(self, basic_profile):
        assert basic_profile.contains_tool("manage_session") is True

    def test_contains_tool_false(self, basic_profile):
        assert basic_profile.contains_tool("unknown_tool") is False


class TestToolProfileModification:
    """Test immutable modification methods."""

    def test_with_additional_tool(self, basic_profile):
        new_profile = basic_profile.with_additional_tool("get_session_state")
        assert new_profile.tool_count == 3
        assert new_profile.contains_tool("get_session_state")
        # Original unchanged
        assert basic_profile.tool_count == 2

    def test_with_additional_tool_preserves_name(self, basic_profile):
        new_profile = basic_profile.with_additional_tool("get_session_state")
        assert new_profile.name == basic_profile.name

    def test_with_additional_tool_idempotent(self, basic_profile):
        """Adding an already-present tool is a no-op in tool count."""
        new_profile = basic_profile.with_additional_tool("manage_session")
        assert new_profile.tool_count == 2

    def test_without_tool(self, basic_profile):
        new_profile = basic_profile.without_tool("manage_session")
        assert new_profile.tool_count == 1
        assert not new_profile.contains_tool("manage_session")
        assert new_profile.contains_tool("execute_step")
        # Original unchanged
        assert basic_profile.tool_count == 2

    def test_without_last_tool_raises(self):
        profile = ToolProfile(
            name="single",
            tool_names=frozenset({"manage_session"}),
            description_mode=ToolDescriptionMode.FULL,
            model_tier=ModelTier.LARGE_CONTEXT,
        )
        with pytest.raises(ValueError, match="Cannot remove last tool"):
            profile.without_tool("manage_session")

    def test_without_tool_not_in_profile(self, basic_profile):
        """Removing a tool not in the profile still returns a valid profile."""
        new_profile = basic_profile.without_tool("nonexistent")
        assert new_profile.tool_count == 2  # No change


class TestToolProfileBudget:
    """Test token budget estimation and validation."""

    def test_estimate_token_cost_compact(self, basic_profile, sample_descriptors):
        cost = basic_profile.estimate_token_cost(sample_descriptors)
        # manage_session compact=150, execute_step compact=200
        assert cost == 350

    def test_estimate_token_cost_with_missing_descriptor(self, basic_profile):
        """Missing descriptors are skipped (cost=0 contribution)."""
        cost = basic_profile.estimate_token_cost({})
        assert cost == 0

    def test_validate_budget_within(self, sample_descriptors):
        """Profile within budget returns no errors."""
        profile = ToolProfile(
            name="within",
            tool_names=frozenset({"manage_session", "execute_step"}),
            description_mode=ToolDescriptionMode.COMPACT,
            model_tier=ModelTier.SMALL_CONTEXT,
            token_budget=TokenBudget(
                context_window=8192,
                instruction_budget=200,
                tool_budget=1000,  # 350 < 1000
                response_budget=2000,
                reasoning_reserve=4000,
            ),
        )
        errors = profile.validate_budget(sample_descriptors)
        assert errors == []

    def test_validate_budget_exceeded(self, sample_descriptors):
        """Profile over budget returns error messages."""
        profile = ToolProfile(
            name="over",
            tool_names=frozenset({"manage_session", "execute_step"}),
            description_mode=ToolDescriptionMode.COMPACT,
            model_tier=ModelTier.SMALL_CONTEXT,
            token_budget=TokenBudget(
                context_window=8192,
                instruction_budget=200,
                tool_budget=100,  # 350 > 100
                response_budget=2000,
                reasoning_reserve=4000,
            ),
        )
        errors = profile.validate_budget(sample_descriptors)
        assert len(errors) == 1
        assert "exceeds budget" in errors[0]

    def test_validate_budget_no_budget_set(self, sample_descriptors):
        """No budget means no validation errors."""
        profile = ToolProfile(
            name="no_budget",
            tool_names=frozenset({"manage_session"}),
            description_mode=ToolDescriptionMode.FULL,
            model_tier=ModelTier.LARGE_CONTEXT,
            token_budget=None,
        )
        errors = profile.validate_budget(sample_descriptors)
        assert errors == []


# =============================================================================
# ProfilePresets
# =============================================================================


class TestProfilePresets:
    """Test built-in profile configurations."""

    def test_browser_exec_has_6_tools(self):
        p = ProfilePresets.browser_exec()
        assert p.tool_count == 6

    def test_browser_exec_tools(self):
        p = ProfilePresets.browser_exec()
        expected = frozenset({
            "manage_session", "execute_step",
            "get_session_state", "get_locator_guidance",
            "find_keywords", "intent_action",
        })
        assert p.tool_names == expected

    def test_browser_exec_mode_and_tier(self):
        p = ProfilePresets.browser_exec()
        assert p.description_mode == ToolDescriptionMode.COMPACT
        assert p.model_tier == ModelTier.SMALL_CONTEXT
        assert p.token_budget is not None

    def test_api_exec_has_5_tools(self):
        p = ProfilePresets.api_exec()
        assert p.tool_count == 5

    def test_api_exec_tools(self):
        p = ProfilePresets.api_exec()
        expected = frozenset({
            "manage_session", "execute_step",
            "get_session_state", "find_keywords",
            "intent_action",
        })
        assert p.tool_names == expected

    def test_api_exec_mode(self):
        p = ProfilePresets.api_exec()
        assert p.description_mode == ToolDescriptionMode.COMPACT

    def test_discovery_has_6_tools(self):
        p = ProfilePresets.discovery()
        assert p.tool_count == 6

    def test_discovery_tools(self):
        p = ProfilePresets.discovery()
        expected = frozenset({
            "manage_session", "analyze_scenario",
            "recommend_libraries", "check_library_availability",
            "find_keywords", "get_keyword_info",
        })
        assert p.tool_names == expected

    def test_discovery_mode(self):
        p = ProfilePresets.discovery()
        assert p.description_mode == ToolDescriptionMode.COMPACT
        assert p.model_tier == ModelTier.SMALL_CONTEXT

    def test_minimal_exec_has_3_tools(self):
        p = ProfilePresets.minimal_exec()
        assert p.tool_count == 3

    def test_minimal_exec_tools(self):
        p = ProfilePresets.minimal_exec()
        expected = frozenset({
            "manage_session", "execute_step", "get_session_state",
        })
        assert p.tool_names == expected

    def test_minimal_exec_mode(self):
        p = ProfilePresets.minimal_exec()
        assert p.description_mode == ToolDescriptionMode.MINIMAL

    def test_full_has_all_16_tools(self):
        p = ProfilePresets.full()
        assert p.tool_count == 16

    def test_full_tools_equal_all_tools(self):
        p = ProfilePresets.full()
        assert p.tool_names == ProfilePresets.ALL_TOOLS

    def test_full_mode(self):
        p = ProfilePresets.full()
        assert p.description_mode == ToolDescriptionMode.FULL
        assert p.model_tier == ModelTier.LARGE_CONTEXT
        assert p.token_budget is None

    def test_all_tools_has_16_entries(self):
        assert len(ProfilePresets.ALL_TOOLS) == 16

    def test_all_presets_have_unique_names(self):
        names = {
            ProfilePresets.browser_exec().name,
            ProfilePresets.api_exec().name,
            ProfilePresets.discovery().name,
            ProfilePresets.minimal_exec().name,
            ProfilePresets.full().name,
        }
        assert len(names) == 5

    def test_browser_exec_is_frozen(self):
        p = ProfilePresets.browser_exec()
        with pytest.raises(AttributeError):
            p.name = "changed"

"""Comprehensive unit tests for Tool Profile domain value objects (ADR-006).

Tests cover: ModelTier, ToolDescriptionMode, ToolTag, TokenBudget, ProfileTransition.

Run with: uv run pytest tests/unit/test_tool_profile_value_objects.py -v
"""

__test__ = True

import pytest

from robotmcp.domains.tool_profile.value_objects import (
    ModelTier,
    ProfileTransition,
    TokenBudget,
    ToolDescriptionMode,
    ToolTag,
)


# =============================================================================
# ModelTier
# =============================================================================


class TestModelTier:
    """Test ModelTier enum and from_context_window classification."""

    def test_small_context_at_4k(self):
        assert ModelTier.from_context_window(4096) == ModelTier.SMALL_CONTEXT

    def test_small_context_at_8k(self):
        assert ModelTier.from_context_window(8192) == ModelTier.SMALL_CONTEXT

    def test_small_context_at_16k(self):
        assert ModelTier.from_context_window(16384) == ModelTier.SMALL_CONTEXT

    def test_small_context_boundary_exact(self):
        """16384 is the upper boundary for SMALL_CONTEXT (inclusive)."""
        assert ModelTier.from_context_window(16384) == ModelTier.SMALL_CONTEXT

    def test_standard_just_above_small(self):
        """16385 should be STANDARD."""
        assert ModelTier.from_context_window(16385) == ModelTier.STANDARD

    def test_standard_at_32k(self):
        assert ModelTier.from_context_window(32768) == ModelTier.STANDARD

    def test_standard_at_64k(self):
        assert ModelTier.from_context_window(65536) == ModelTier.STANDARD

    def test_standard_boundary_exact(self):
        """65536 is the upper boundary for STANDARD (inclusive)."""
        assert ModelTier.from_context_window(65536) == ModelTier.STANDARD

    def test_large_context_just_above_standard(self):
        """65537 should be LARGE_CONTEXT."""
        assert ModelTier.from_context_window(65537) == ModelTier.LARGE_CONTEXT

    def test_large_context_at_128k(self):
        assert ModelTier.from_context_window(131072) == ModelTier.LARGE_CONTEXT

    def test_large_context_at_200k(self):
        assert ModelTier.from_context_window(200000) == ModelTier.LARGE_CONTEXT

    def test_very_small_context_1_token(self):
        assert ModelTier.from_context_window(1) == ModelTier.SMALL_CONTEXT

    def test_enum_values(self):
        assert ModelTier.SMALL_CONTEXT.value == "small_context"
        assert ModelTier.STANDARD.value == "standard"
        assert ModelTier.LARGE_CONTEXT.value == "large_context"

    def test_enum_has_exactly_3_members(self):
        assert len(ModelTier) == 3


# =============================================================================
# ToolDescriptionMode
# =============================================================================


class TestToolDescriptionMode:
    """Test ToolDescriptionMode enum and abbreviates_fields property."""

    def test_full_value(self):
        assert ToolDescriptionMode.FULL.value == "full"

    def test_compact_value(self):
        assert ToolDescriptionMode.COMPACT.value == "compact"

    def test_minimal_value(self):
        assert ToolDescriptionMode.MINIMAL.value == "minimal"

    def test_full_does_not_abbreviate(self):
        assert ToolDescriptionMode.FULL.abbreviates_fields is False

    def test_compact_abbreviates(self):
        assert ToolDescriptionMode.COMPACT.abbreviates_fields is True

    def test_minimal_abbreviates(self):
        assert ToolDescriptionMode.MINIMAL.abbreviates_fields is True

    def test_enum_has_exactly_3_members(self):
        assert len(ToolDescriptionMode) == 3


# =============================================================================
# ToolTag
# =============================================================================


class TestToolTag:
    """Test ToolTag enum completeness."""

    def test_core_value(self):
        assert ToolTag.CORE.value == "core"

    def test_discovery_value(self):
        assert ToolTag.DISCOVERY.value == "discovery"

    def test_execution_value(self):
        assert ToolTag.EXECUTION.value == "execution"

    def test_advanced_value(self):
        assert ToolTag.ADVANCED.value == "advanced"

    def test_reporting_value(self):
        assert ToolTag.REPORTING.value == "reporting"

    def test_recovery_value(self):
        assert ToolTag.RECOVERY.value == "recovery"

    def test_enum_has_exactly_6_members(self):
        assert len(ToolTag) == 6

    def test_all_values_unique(self):
        values = [t.value for t in ToolTag]
        assert len(values) == len(set(values))


# =============================================================================
# TokenBudget
# =============================================================================


class TestTokenBudget:
    """Test TokenBudget value object."""

    def test_for_small_context_window(self):
        budget = TokenBudget.for_context_window(8192)
        assert budget.context_window == 8192
        # SMALL_ALLOCATION: instruction=3%, tools=20%, responses=30%, reasoning=47%
        assert budget.instruction_budget == int(8192 * 0.03)
        assert budget.tool_budget == int(8192 * 0.20)
        assert budget.response_budget == int(8192 * 0.30)
        assert budget.reasoning_reserve == int(8192 * 0.47)

    def test_for_standard_context_window(self):
        budget = TokenBudget.for_context_window(32768)
        assert budget.context_window == 32768
        # STANDARD_ALLOCATION: instruction=4%, tools=15%, responses=35%, reasoning=46%
        assert budget.instruction_budget == int(32768 * 0.04)
        assert budget.tool_budget == int(32768 * 0.15)
        assert budget.response_budget == int(32768 * 0.35)
        assert budget.reasoning_reserve == int(32768 * 0.46)

    def test_for_large_context_window(self):
        budget = TokenBudget.for_context_window(131072)
        # Large context uses STANDARD_ALLOCATION
        assert budget.context_window == 131072
        assert budget.instruction_budget == int(131072 * 0.04)
        assert budget.tool_budget == int(131072 * 0.15)

    def test_small_allocation_boundary(self):
        """16384 (SMALL_CONTEXT) should use SMALL_ALLOCATION."""
        budget = TokenBudget.for_context_window(16384)
        assert budget.instruction_budget == int(16384 * 0.03)
        assert budget.tool_budget == int(16384 * 0.20)

    def test_standard_allocation_boundary(self):
        """16385 (STANDARD) should use STANDARD_ALLOCATION."""
        budget = TokenBudget.for_context_window(16385)
        assert budget.instruction_budget == int(16385 * 0.04)
        assert budget.tool_budget == int(16385 * 0.15)

    def test_post_init_rejects_zero_context_window(self):
        with pytest.raises(ValueError, match="Context window must be positive"):
            TokenBudget(
                context_window=0,
                instruction_budget=0,
                tool_budget=0,
                response_budget=0,
                reasoning_reserve=0,
            )

    def test_post_init_rejects_negative_context_window(self):
        with pytest.raises(ValueError, match="Context window must be positive"):
            TokenBudget(
                context_window=-100,
                instruction_budget=0,
                tool_budget=0,
                response_budget=0,
                reasoning_reserve=0,
            )

    def test_post_init_rejects_over_allocation(self):
        with pytest.raises(ValueError, match="Budget allocation.*exceeds"):
            TokenBudget(
                context_window=100,
                instruction_budget=50,
                tool_budget=50,
                response_budget=50,
                reasoning_reserve=50,
            )

    def test_post_init_allows_under_allocation(self):
        """Under-allocation is allowed (some tokens go unused)."""
        budget = TokenBudget(
            context_window=1000,
            instruction_budget=100,
            tool_budget=100,
            response_budget=100,
            reasoning_reserve=100,
        )
        assert budget.context_window == 1000

    def test_post_init_allows_exact_allocation(self):
        budget = TokenBudget(
            context_window=1000,
            instruction_budget=100,
            tool_budget=200,
            response_budget=300,
            reasoning_reserve=400,
        )
        assert budget.context_window == 1000

    def test_utilization_ratio(self):
        budget = TokenBudget(
            context_window=1000,
            instruction_budget=100,
            tool_budget=200,
            response_budget=200,
            reasoning_reserve=400,
        )
        # non-reasoning = 100+200+200 = 500; ratio = 500/1000 = 0.5
        assert budget.utilization_ratio == 0.5

    def test_utilization_ratio_for_context_window(self):
        budget = TokenBudget.for_context_window(8192)
        # SMALL: 3%+20%+30% = 53%
        expected = (budget.instruction_budget + budget.tool_budget + budget.response_budget) / 8192
        assert abs(budget.utilization_ratio - expected) < 0.001

    def test_fits_tool_cost_within_budget(self):
        budget = TokenBudget.for_context_window(8192)
        assert budget.fits_tool_cost(100) is True

    def test_fits_tool_cost_at_exact_budget(self):
        budget = TokenBudget.for_context_window(8192)
        assert budget.fits_tool_cost(budget.tool_budget) is True

    def test_fits_tool_cost_exceeds_budget(self):
        budget = TokenBudget.for_context_window(8192)
        assert budget.fits_tool_cost(budget.tool_budget + 1) is False

    def test_is_frozen(self):
        budget = TokenBudget.for_context_window(8192)
        with pytest.raises(AttributeError):
            budget.context_window = 9999


# =============================================================================
# ProfileTransition
# =============================================================================


class TestProfileTransition:
    """Test ProfileTransition value object."""

    def test_valid_plan_to_execute(self):
        t = ProfileTransition(
            from_profile="discovery",
            to_profile="browser_exec",
            trigger=ProfileTransition.PLAN_TO_EXECUTE,
        )
        assert t.from_profile == "discovery"
        assert t.to_profile == "browser_exec"
        assert t.trigger == "plan_to_execute"

    def test_valid_execute_to_recovery(self):
        t = ProfileTransition(
            from_profile="browser_exec",
            to_profile="browser_exec_recovery",
            trigger=ProfileTransition.EXECUTE_TO_RECOVERY,
        )
        assert t.trigger == "execute_to_recovery"

    def test_valid_escalation_same_profile(self):
        """Escalation is the only valid trigger for same-profile transitions."""
        t = ProfileTransition(
            from_profile="browser_exec",
            to_profile="browser_exec",
            trigger=ProfileTransition.ESCALATION,
        )
        assert t.trigger == "escalation"

    def test_valid_de_escalation(self):
        t = ProfileTransition(
            from_profile="full",
            to_profile="browser_exec",
            trigger=ProfileTransition.DE_ESCALATION,
        )
        assert t.trigger == "de_escalation"

    def test_valid_user_request(self):
        t = ProfileTransition(
            from_profile="discovery",
            to_profile="full",
            trigger=ProfileTransition.USER_REQUEST,
        )
        assert t.trigger == "user_request"

    def test_valid_session_init(self):
        t = ProfileTransition(
            from_profile="none",
            to_profile="browser_exec",
            trigger=ProfileTransition.SESSION_INIT,
        )
        assert t.trigger == "session_init"

    def test_valid_model_tier_change(self):
        t = ProfileTransition(
            from_profile="full",
            to_profile="browser_exec",
            trigger=ProfileTransition.MODEL_TIER_CHANGE,
        )
        assert t.trigger == "model_tier_change"

    def test_invalid_trigger_raises(self):
        with pytest.raises(ValueError, match="Invalid transition trigger"):
            ProfileTransition(
                from_profile="a",
                to_profile="b",
                trigger="invalid_trigger",
            )

    def test_same_profile_non_escalation_raises(self):
        with pytest.raises(ValueError, match="same profile is only valid for escalation"):
            ProfileTransition(
                from_profile="browser_exec",
                to_profile="browser_exec",
                trigger=ProfileTransition.USER_REQUEST,
            )

    def test_same_profile_plan_to_execute_raises(self):
        with pytest.raises(ValueError, match="same profile is only valid for escalation"):
            ProfileTransition(
                from_profile="full",
                to_profile="full",
                trigger=ProfileTransition.PLAN_TO_EXECUTE,
            )

    def test_valid_triggers_frozenset_has_7_members(self):
        assert len(ProfileTransition.VALID_TRIGGERS) == 7

    def test_is_frozen(self):
        t = ProfileTransition(
            from_profile="a",
            to_profile="b",
            trigger=ProfileTransition.USER_REQUEST,
        )
        with pytest.raises(AttributeError):
            t.trigger = "other"

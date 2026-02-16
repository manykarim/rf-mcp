"""Tool Profile Domain Value Objects.

This module contains immutable value objects for the Tool Profile bounded context.
Value objects are identified by their attributes rather than by identity.

All value objects are frozen dataclasses following the pattern established in
domains/shared/kernel.py and domains/instruction/value_objects.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, FrozenSet

# Re-export ModelTier from shared kernel (canonical definition)
from ..shared.kernel import ModelTier

__all__ = [
    "ModelTier",
    "ToolDescriptionMode",
    "ToolTag",
    "TokenBudget",
    "ProfileTransition",
]


class ToolDescriptionMode(Enum):
    """Controls how tool descriptions are formatted for the LLM.

    FULL:    Original description text + complete JSON schema.
             (~7,057 tokens for all 15 tools)
    COMPACT: Shortened description, essential params only.
             (~1,350 tokens for a 5-tool profile)
    MINIMAL: Single-sentence description, flat schema.
             (~900 tokens for a 3-tool profile)
    """
    FULL = "full"
    COMPACT = "compact"
    MINIMAL = "minimal"

    @property
    def abbreviates_fields(self) -> bool:
        """Whether response field names should be abbreviated."""
        return self in (ToolDescriptionMode.COMPACT, ToolDescriptionMode.MINIMAL)


class ToolTag(Enum):
    """Semantic tags for tool classification.

    Used for profile construction and filtering. A tool may have
    multiple tags. Tags align with workflow phases.
    """
    CORE = "core"                # Always needed: manage_session, execute_step, get_session_state
    DISCOVERY = "discovery"      # Planning phase: analyze_scenario, recommend_libraries, find_keywords
    EXECUTION = "execution"      # Execution phase: execute_step, execute_flow, get_locator_guidance
    ADVANCED = "advanced"        # Power-user: execute_flow, build_test_suite, run_test_suite
    REPORTING = "reporting"      # Suite generation: build_test_suite, run_test_suite
    RECOVERY = "recovery"        # Error recovery: get_session_state, find_keywords, get_keyword_info


@dataclass(frozen=True)
class TokenBudget:
    """Token budget allocation for a profile.

    Allocates a total context window into slices. The tool_budget
    represents the maximum tokens that tool descriptions + schemas
    may consume.

    Based on validated measurements:
    - Tool descriptions: 7,057 tokens (15 tools, full mode)
    - Server instructions: ~454 tokens
    - Instruction template: 62-573 tokens
    - Typical 6-call response: ~3,573 tokens

    Attributes:
        context_window: Total context window in tokens.
        instruction_budget: Tokens allocated for instructions.
        tool_budget: Tokens allocated for tool descriptions + schemas.
        response_budget: Tokens allocated for accumulated responses.
        reasoning_reserve: Tokens reserved for LLM reasoning.
    """
    context_window: int
    instruction_budget: int
    tool_budget: int
    response_budget: int
    reasoning_reserve: int

    # Allocation percentages by tier
    SMALL_ALLOCATION: ClassVar[dict] = {
        "instruction": 0.03,    # ~3% for instructions
        "tools": 0.20,          # ~20% for tool descriptions
        "responses": 0.30,      # ~30% for accumulated responses
        "reasoning": 0.47,      # ~47% reserved for LLM reasoning
    }
    STANDARD_ALLOCATION: ClassVar[dict] = {
        "instruction": 0.04,
        "tools": 0.15,
        "responses": 0.35,
        "reasoning": 0.46,
    }

    def __post_init__(self) -> None:
        """Validate budget allocation on creation."""
        if self.context_window <= 0:
            raise ValueError("Context window must be positive")
        total_allocated = (
            self.instruction_budget + self.tool_budget
            + self.response_budget + self.reasoning_reserve
        )
        if total_allocated > self.context_window:
            raise ValueError(
                f"Budget allocation ({total_allocated}) exceeds "
                f"context window ({self.context_window})"
            )

    @classmethod
    def for_context_window(cls, window_size: int) -> "TokenBudget":
        """Create a budget using default allocations for the given window.

        Args:
            window_size: Total context window in tokens.

        Returns:
            A TokenBudget with tier-appropriate allocations.
        """
        tier = ModelTier.from_context_window(window_size)
        alloc = (
            cls.SMALL_ALLOCATION
            if tier == ModelTier.SMALL_CONTEXT
            else cls.STANDARD_ALLOCATION
        )
        return cls(
            context_window=window_size,
            instruction_budget=int(window_size * alloc["instruction"]),
            tool_budget=int(window_size * alloc["tools"]),
            response_budget=int(window_size * alloc["responses"]),
            reasoning_reserve=int(window_size * alloc["reasoning"]),
        )

    @property
    def utilization_ratio(self) -> float:
        """Fraction of context window consumed by non-reasoning budget."""
        non_reasoning = (
            self.instruction_budget + self.tool_budget + self.response_budget
        )
        return non_reasoning / self.context_window

    def fits_tool_cost(self, token_cost: int) -> bool:
        """Check whether a given tool cost fits within the tool budget."""
        return token_cost <= self.tool_budget


@dataclass(frozen=True)
class ProfileTransition:
    """Represents a valid transition between profiles.

    Transitions are triggered by workflow phase changes, explicit
    user requests, or automatic escalation/de-escalation.

    Examples of valid transitions:
    - discovery -> browser_exec (after planning completes)
    - browser_exec -> browser_exec + get_keyword_info (escalation)
    - full -> browser_exec (user requests small-model profile)

    Attributes:
        from_profile: Name of the profile being transitioned from.
        to_profile: Name of the profile being transitioned to.
        trigger: The trigger that caused the transition.
    """

    from_profile: str
    to_profile: str
    trigger: str

    # Valid transition triggers
    PLAN_TO_EXECUTE: ClassVar[str] = "plan_to_execute"
    EXECUTE_TO_RECOVERY: ClassVar[str] = "execute_to_recovery"
    ESCALATION: ClassVar[str] = "escalation"
    DE_ESCALATION: ClassVar[str] = "de_escalation"
    USER_REQUEST: ClassVar[str] = "user_request"
    SESSION_INIT: ClassVar[str] = "session_init"
    MODEL_TIER_CHANGE: ClassVar[str] = "model_tier_change"

    VALID_TRIGGERS: ClassVar[FrozenSet[str]] = frozenset({
        PLAN_TO_EXECUTE, EXECUTE_TO_RECOVERY,
        ESCALATION, DE_ESCALATION,
        USER_REQUEST, SESSION_INIT, MODEL_TIER_CHANGE,
    })

    def __post_init__(self) -> None:
        """Validate transition on creation."""
        if self.trigger not in self.VALID_TRIGGERS:
            raise ValueError(
                f"Invalid transition trigger: '{self.trigger}'. "
                f"Must be one of: {', '.join(sorted(self.VALID_TRIGGERS))}"
            )
        if self.from_profile == self.to_profile and self.trigger != self.ESCALATION:
            raise ValueError(
                "Transition to the same profile is only valid for escalation"
            )

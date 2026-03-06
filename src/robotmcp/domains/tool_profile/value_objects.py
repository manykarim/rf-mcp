"""Tool Profile Domain Value Objects.

This module contains immutable value objects for the Tool Profile bounded context.
Value objects are identified by their attributes rather than by identity.

All value objects are frozen dataclasses following the pattern established in
domains/shared/kernel.py and domains/instruction/value_objects.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, FrozenSet, Tuple

# Re-export ModelTier from shared kernel (canonical definition)
from ..shared.kernel import ModelTier

__all__ = [
    "ModelTier",
    "ToolDescriptionMode",
    "ToolTag",
    "TokenBudget",
    "ProfileTransition",
    "SchemaMode",
    "SlimToolSchema",
    "AutoProfileSelection",
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


class SchemaMode(Enum):
    """Controls JSON Schema structural complexity (ADR-016).

    FULL: Complete schema with all properties, descriptions, unions
    STANDARD: All properties kept, property descriptions removed
    MINIMAL: Only required properties, no descriptions, unions flattened
    """
    FULL = "full"
    STANDARD = "standard"
    MINIMAL = "minimal"

    @property
    def strips_optional_fields(self) -> bool:
        """Whether optional (non-required) properties should be removed."""
        return self == SchemaMode.MINIMAL

    @property
    def strips_property_descriptions(self) -> bool:
        """Whether per-property descriptions should be removed."""
        return self in (SchemaMode.STANDARD, SchemaMode.MINIMAL)

    @property
    def flattens_unions(self) -> bool:
        """Whether anyOf/oneOf unions should be flattened to first variant."""
        return self == SchemaMode.MINIMAL


@dataclass(frozen=True)
class SlimToolSchema:
    """Structurally reduced JSON Schema for small models (ADR-016).

    Produces a simplified schema from a full JSON Schema by removing
    optional properties, stripping descriptions, and flattening unions
    based on the specified SchemaMode.

    Attributes:
        tool_name: The tool this schema belongs to.
        properties: Reduced property definitions.
        required: Tuple of required property names.
        schema_mode: The mode used to produce this schema.
        token_estimate: Estimated token cost of the reduced schema.
    """
    tool_name: str
    properties: Dict[str, Any]
    required: Tuple[str, ...]
    schema_mode: SchemaMode
    token_estimate: int

    def __post_init__(self) -> None:
        """Validate that schema has at least one property."""
        if not self.properties:
            raise ValueError("Schema must have at least one property")

    @classmethod
    def from_full_schema(
        cls,
        tool_name: str,
        full_schema: Dict[str, Any],
        mode: SchemaMode = SchemaMode.MINIMAL,
    ) -> "SlimToolSchema":
        """Create a slim schema from a full JSON Schema.

        Applies transformations based on the schema mode:
        - FULL: No changes (identity transform).
        - STANDARD: Remove per-property descriptions.
        - MINIMAL: Remove optional properties, descriptions, flatten unions.

        If removing optional properties would leave zero properties, the
        first property from the original schema is preserved.

        Args:
            tool_name: The tool name.
            full_schema: The full JSON Schema dict.
            mode: The schema simplification mode.

        Returns:
            A SlimToolSchema with the reduced properties.
        """
        props = dict(full_schema.get("properties", {}))
        required = list(full_schema.get("required", []))

        # Strip optional (non-required) properties in MINIMAL mode
        if mode.strips_optional_fields:
            props = {k: v for k, v in props.items() if k in required}

        # Strip per-property descriptions in STANDARD and MINIMAL modes
        if mode.strips_property_descriptions:
            props = {
                k: (
                    {kk: vv for kk, vv in v.items() if kk != "description"}
                    if isinstance(v, dict)
                    else v
                )
                for k, v in props.items()
            }

        # Flatten anyOf/oneOf unions in MINIMAL mode
        if mode.flattens_unions:
            for k, v in props.items():
                if isinstance(v, dict):
                    if "anyOf" in v:
                        props[k] = v["anyOf"][0] if v["anyOf"] else {"type": "string"}
                    elif "oneOf" in v:
                        props[k] = v["oneOf"][0] if v["oneOf"] else {"type": "string"}

        # Safety net: if all properties were removed, keep the first original one
        if not props and full_schema.get("properties"):
            first_key = next(iter(full_schema["properties"]))
            props = {first_key: full_schema["properties"][first_key]}
            required = [first_key]

        # Estimate token cost from serialized schema size (4 chars ~ 1 token)
        token_est = len(json.dumps(
            {"type": "object", "properties": props, "required": required}
        )) // 4

        return cls(
            tool_name=tool_name,
            properties=props,
            required=tuple(required),
            schema_mode=mode,
            token_estimate=token_est,
        )

    def to_schema(self) -> Dict[str, Any]:
        """Convert back to a JSON Schema dict.

        Returns:
            A dict with type, properties, and required keys.
        """
        return {
            "type": "object",
            "properties": self.properties,
            "required": list(self.required),
        }


@dataclass(frozen=True)
class AutoProfileSelection:
    """Result of automatic profile selection based on model tier and task domain (ADR-016).

    Encapsulates the recommendation produced by the auto-selection
    heuristic, including the selected profile, schema mode,
    instruction template, and confidence level.

    Attributes:
        model_tier: The detected model capability tier.
        task_domain: The task domain (e.g. "browser", "api", "desktop").
        recommended_profile: Name of the recommended profile.
        recommended_schema_mode: Schema mode for the profile.
        recommended_instruction_template: Instruction template name.
        confidence: Confidence in the recommendation (0.0 to 1.0).
        rationale: Human-readable explanation.
    """
    model_tier: ModelTier
    task_domain: str
    recommended_profile: str
    recommended_schema_mode: SchemaMode
    recommended_instruction_template: str
    confidence: float
    rationale: str

    def __post_init__(self) -> None:
        """Validate confidence is within [0.0, 1.0]."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be 0.0-1.0, got {self.confidence}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary.

        Returns:
            Dict with all fields as primitive types.
        """
        return {
            "tier": self.model_tier.value,
            "domain": self.task_domain,
            "profile": self.recommended_profile,
            "schema_mode": self.recommended_schema_mode.value,
            "template": self.recommended_instruction_template,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }

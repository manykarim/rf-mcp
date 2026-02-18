"""Tool Profile Domain Aggregates.

This module contains the aggregate roots for the Tool Profile bounded context.
Aggregates are clusters of domain objects that are treated as a unit
for data changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional

from .value_objects import (
    ModelTier,
    ProfileTransition,
    TokenBudget,
    ToolDescriptionMode,
    ToolTag,
)
from .entities import ToolDescriptor


@dataclass(frozen=True)
class ToolProfile:
    """Aggregate root: a named set of tools with description and schema policies.

    Invariants:
    - tool_names must be non-empty (at least 1 tool).
    - If model_tier is SMALL_CONTEXT, token_budget must be set and
      the sum of compact tool descriptions must not exceed it.
    - Every tool_name must have a corresponding ToolDescriptor in
      the registry (validated at activation time, not construction).

    A ToolProfile is frozen/immutable. Transitions produce a NEW profile.

    Attributes:
        name: Unique name for this profile.
        tool_names: Set of MCP tool names included in this profile.
        description_mode: How tool descriptions are formatted.
        model_tier: The target LLM context tier.
        token_budget: Optional token budget constraint.
        tags: Semantic tags describing this profile's purpose.
        description: Human-readable description of this profile.
    """

    name: str
    tool_names: FrozenSet[str]
    description_mode: ToolDescriptionMode
    model_tier: ModelTier
    token_budget: Optional[TokenBudget] = None
    tags: FrozenSet[ToolTag] = field(default_factory=frozenset)
    description: str = ""

    def __post_init__(self) -> None:
        """Validate invariants on creation."""
        if not self.tool_names:
            raise ValueError("ToolProfile must contain at least one tool")
        if self.model_tier == ModelTier.SMALL_CONTEXT and self.token_budget is None:
            raise ValueError(
                "SMALL_CONTEXT profiles require an explicit token_budget"
            )

    @property
    def tool_count(self) -> int:
        """Return the number of tools in this profile."""
        return len(self.tool_names)

    def contains_tool(self, tool_name: str) -> bool:
        """Check if a tool is in this profile.

        Args:
            tool_name: The tool name to check.

        Returns:
            True if the tool is in the profile.
        """
        return tool_name in self.tool_names

    def with_additional_tool(self, tool_name: str) -> "ToolProfile":
        """Return a new profile with one tool added (escalation).

        Args:
            tool_name: The tool name to add.

        Returns:
            A new ToolProfile with the additional tool.
        """
        return ToolProfile(
            name=self.name,
            tool_names=self.tool_names | {tool_name},
            description_mode=self.description_mode,
            model_tier=self.model_tier,
            token_budget=self.token_budget,
            tags=self.tags,
            description=self.description,
        )

    def without_tool(self, tool_name: str) -> "ToolProfile":
        """Return a new profile with one tool removed.

        Args:
            tool_name: The tool name to remove.

        Returns:
            A new ToolProfile without the specified tool.

        Raises:
            ValueError: If removing the tool would leave the profile empty.
        """
        remaining = self.tool_names - {tool_name}
        if not remaining:
            raise ValueError("Cannot remove last tool from profile")
        return ToolProfile(
            name=self.name,
            tool_names=frozenset(remaining),
            description_mode=self.description_mode,
            model_tier=self.model_tier,
            token_budget=self.token_budget,
            tags=self.tags,
            description=self.description,
        )

    def estimate_token_cost(
        self, descriptors: Dict[str, ToolDescriptor]
    ) -> int:
        """Estimate total token cost given a descriptor registry.

        Args:
            descriptors: Mapping of tool names to their descriptors.

        Returns:
            Total estimated token cost for all tools in this profile.
        """
        total = 0
        for name in self.tool_names:
            desc = descriptors.get(name)
            if desc is None:
                continue
            total += desc.token_estimate_for_mode(self.description_mode)
        return total

    def validate_budget(
        self, descriptors: Dict[str, ToolDescriptor]
    ) -> List[str]:
        """Check whether profile fits within its token budget.

        Args:
            descriptors: Mapping of tool names to their descriptors.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []
        if self.token_budget is None:
            return errors
        cost = self.estimate_token_cost(descriptors)
        if cost > self.token_budget.tool_budget:
            errors.append(
                f"Profile '{self.name}' tool cost ({cost} tokens) "
                f"exceeds budget ({self.token_budget.tool_budget} tokens)"
            )
        return errors


class ProfilePresets:
    """Factory for built-in profile configurations.

    Token estimates are based on compact description mode for small
    profiles and full description mode for the full profile.
    """

    # --- 16 enabled tools in the current server ---
    ALL_TOOLS: FrozenSet[str] = frozenset({
        "manage_session", "execute_step", "execute_flow",
        "get_session_state", "find_keywords", "get_keyword_info",
        "recommend_libraries", "analyze_scenario",
        "check_library_availability", "build_test_suite",
        "run_test_suite", "get_locator_guidance",
        "set_library_search_order", "manage_library_plugins",
        "manage_attach", "intent_action",
    })

    @classmethod
    def browser_exec(cls) -> ToolProfile:
        """6-tool profile for browser execution on small-context models.

        Estimated ~1,600 tokens (compact descriptions).
        """
        return ToolProfile(
            name="browser_exec",
            tool_names=frozenset({
                "manage_session", "execute_step",
                "get_session_state", "get_locator_guidance",
                "find_keywords", "intent_action",
            }),
            description_mode=ToolDescriptionMode.COMPACT,
            model_tier=ModelTier.SMALL_CONTEXT,
            token_budget=TokenBudget.for_context_window(8192),
            tags=frozenset({ToolTag.CORE, ToolTag.EXECUTION}),
            description="Browser test execution for small-context LLMs",
        )

    @classmethod
    def api_exec(cls) -> ToolProfile:
        """5-tool profile for API testing on small-context models.

        Estimated ~1,350 tokens (compact descriptions).
        """
        return ToolProfile(
            name="api_exec",
            tool_names=frozenset({
                "manage_session", "execute_step",
                "get_session_state", "find_keywords",
                "intent_action",
            }),
            description_mode=ToolDescriptionMode.COMPACT,
            model_tier=ModelTier.SMALL_CONTEXT,
            token_budget=TokenBudget.for_context_window(8192),
            tags=frozenset({ToolTag.CORE, ToolTag.EXECUTION}),
            description="API test execution for small-context LLMs",
        )

    @classmethod
    def discovery(cls) -> ToolProfile:
        """5-tool profile for planning/discovery phase.

        Estimated ~1,200 tokens (compact descriptions).
        """
        return ToolProfile(
            name="discovery",
            tool_names=frozenset({
                "manage_session", "analyze_scenario",
                "recommend_libraries", "check_library_availability",
                "find_keywords", "get_keyword_info",
            }),
            description_mode=ToolDescriptionMode.COMPACT,
            model_tier=ModelTier.SMALL_CONTEXT,
            token_budget=TokenBudget.for_context_window(8192),
            tags=frozenset({ToolTag.DISCOVERY}),
            description="Discovery and planning phase for small-context LLMs",
        )

    @classmethod
    def minimal_exec(cls) -> ToolProfile:
        """3-tool absolute minimum for execution.

        Estimated ~900 tokens (compact descriptions).
        """
        return ToolProfile(
            name="minimal_exec",
            tool_names=frozenset({
                "manage_session", "execute_step", "get_session_state",
            }),
            description_mode=ToolDescriptionMode.MINIMAL,
            model_tier=ModelTier.SMALL_CONTEXT,
            token_budget=TokenBudget.for_context_window(4096),
            tags=frozenset({ToolTag.CORE, ToolTag.EXECUTION}),
            description="Absolute minimum tools for test execution",
        )

    @classmethod
    def desktop_exec(cls) -> ToolProfile:
        """5-tool profile for desktop automation on small-context models.

        Estimated ~1,400 tokens (compact descriptions).
        """
        return ToolProfile(
            name="desktop_exec",
            tool_names=frozenset({
                "manage_session", "execute_step",
                "get_session_state", "get_locator_guidance",
                "find_keywords",
            }),
            description_mode=ToolDescriptionMode.COMPACT,
            model_tier=ModelTier.SMALL_CONTEXT,
            token_budget=TokenBudget.for_context_window(8192),
            tags=frozenset({ToolTag.CORE, ToolTag.EXECUTION}),
            description="Desktop automation for small-context LLMs",
        )

    @classmethod
    def full(cls) -> ToolProfile:
        """All 15 tools with full descriptions for large-context models."""
        return ToolProfile(
            name="full",
            tool_names=cls.ALL_TOOLS,
            description_mode=ToolDescriptionMode.FULL,
            model_tier=ModelTier.LARGE_CONTEXT,
            token_budget=None,
            tags=frozenset({
                ToolTag.CORE, ToolTag.DISCOVERY, ToolTag.EXECUTION,
                ToolTag.ADVANCED, ToolTag.REPORTING,
            }),
            description="Full tool set for large-context LLMs (32K+)",
        )

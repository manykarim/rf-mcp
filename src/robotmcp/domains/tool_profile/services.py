"""Tool Profile Domain Services.

This module contains domain services for the Tool Profile bounded context.
Domain services contain business logic that doesn't naturally fit
within an entity or value object.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, FrozenSet, List, Optional, Protocol

from .aggregates import ToolProfile, ProfilePresets
from .entities import ToolDescriptor
from .events import (
    ProfileActivated,
    ProfileTransitioned,
    TokenBudgetExceeded,
    ToolsHidden,
    ToolsRevealed,
)
from .value_objects import (
    AutoProfileSelection,
    ModelTier,
    ProfileTransition,
    SchemaMode,
    SlimToolSchema,
    ToolDescriptionMode,
)

logger = logging.getLogger(__name__)

_DEFAULT_PROFILE_NAME = "full"


class SessionToolProfileRegistry:
    """Per-session tool profile bindings — no global FastMCP side effects.

    Maintains a mapping of session_id -> ToolProfile name. Sessions without
    an explicit binding inherit the default profile (full). This replaces the
    global-mutation approach in ToolManagerAdapter: the FastMCP registry is
    never mutated after startup; instead each tool call consults this registry
    to decide whether to proceed or return a profile-gated error.

    Thread safety: operated on the asyncio event loop; no locking needed.
    """

    def __init__(self, profile_registry: Dict[str, ToolProfile]) -> None:
        self._profile_registry = profile_registry
        # session_id -> profile_name
        self._session_bindings: Dict[str, str] = {}

    def bind(self, session_id: str, profile_name: str) -> None:
        """Bind a session to a named profile.

        Args:
            session_id: The session identifier.
            profile_name: Must be a key in the profile registry.

        Raises:
            KeyError: If profile_name is not in the registry.
        """
        if profile_name not in self._profile_registry:
            raise KeyError(
                f"Unknown profile: '{profile_name}'. "
                f"Available: {', '.join(sorted(self._profile_registry))}"
            )
        self._session_bindings[session_id] = profile_name
        logger.debug("Session '%s' bound to profile '%s'", session_id, profile_name)

    def get_profile(self, session_id: str) -> ToolProfile:
        """Return the active profile for a session.

        Falls back to the default profile when no binding exists.

        Args:
            session_id: The session identifier (may be empty string).

        Returns:
            The bound ToolProfile, or the default profile.
        """
        profile_name = self._session_bindings.get(session_id, _DEFAULT_PROFILE_NAME)
        return self._profile_registry[profile_name]

    def get_profile_name(self, session_id: str) -> str:
        """Return the active profile name for a session."""
        return self._session_bindings.get(session_id, _DEFAULT_PROFILE_NAME)

    def is_tool_allowed(self, session_id: str, tool_name: str) -> bool:
        """Check whether a tool is visible in a session's active profile.

        The default profile (full) allows all tools, so this returns True
        for any session that has not explicitly bound a restricted profile.

        Args:
            session_id: The session identifier.
            tool_name: The MCP tool name to check.

        Returns:
            True if the tool is in the session's profile.
        """
        profile = self.get_profile(session_id)
        return profile.contains_tool(tool_name)

    def unbind(self, session_id: str) -> None:
        """Remove a session binding, reverting to the default profile."""
        self._session_bindings.pop(session_id, None)

    def list_bindings(self) -> Dict[str, str]:
        """Return a snapshot of all current session->profile bindings."""
        return dict(self._session_bindings)


class ToolManagerPort(Protocol):
    """Port (interface) to the infrastructure layer that manages actual tools.

    Implemented by ToolManagerAdapter (see Anti-Corruption Layer section).
    This protocol defines the async methods needed by the domain service
    to orchestrate tool visibility changes.
    """

    async def remove_tool(self, tool_name: str) -> None:
        """Remove a tool from the visible set."""
        ...

    async def add_tool_with_description(
        self, tool_name: str, description: str, schema: Dict
    ) -> None:
        """Add a tool to the visible set with specific description and schema."""
        ...

    async def get_visible_tool_names(self) -> frozenset[str]:
        """Return the set of currently visible tool names."""
        ...

    async def swap_tool_description(
        self, tool_name: str, description: str, schema: Dict
    ) -> None:
        """Swap a visible tool's description without changing its handler."""
        ...


class ToolProfileManager:
    """Domain service coordinating profile activation and transitions.

    This is the primary entry point for all profile operations. It:
    1. Maintains the active profile per session (or globally).
    2. Validates budget constraints before activation.
    3. Orchestrates add/remove calls on the ToolManagerPort.
    4. Publishes domain events for learning and observability.
    5. Supports profile suggestion based on scenario + model tier.

    Thread safety: This service is designed to run on the asyncio
    event loop. Profile activation is serialized (one at a time)
    to avoid race conditions with the underlying ToolManager.

    Attributes:
        _tool_manager: The infrastructure port for tool manipulation.
        _descriptors: Registry of tool descriptors for all known tools.
        _event_publisher: Optional callback for domain events.
        _active_profile: The currently active profile, or None.
        _profile_registry: Registry of named profiles.
    """

    def __init__(
        self,
        tool_manager: ToolManagerPort,
        descriptor_registry: Dict[str, ToolDescriptor],
        event_publisher: Optional[Callable[[object], None]] = None,
    ) -> None:
        """Initialize the profile manager.

        Args:
            tool_manager: Infrastructure port for tool manipulation.
            descriptor_registry: Registry of tool descriptors.
            event_publisher: Optional callback for domain events.
        """
        self._tool_manager = tool_manager
        self._descriptors = descriptor_registry
        self._event_publisher = event_publisher
        self._active_profile: Optional[ToolProfile] = None
        self._profile_registry: Dict[str, ToolProfile] = {
            "browser_exec": ProfilePresets.browser_exec(),
            "api_exec": ProfilePresets.api_exec(),
            "discovery": ProfilePresets.discovery(),
            "minimal_exec": ProfilePresets.minimal_exec(),
            "full": ProfilePresets.full(),
            "desktop_exec": ProfilePresets.desktop_exec(),
            "slim_exec": ProfilePresets.slim_exec(),
        }
        # Per-session bindings — no global FastMCP side effects (P1)
        self._session_registry = SessionToolProfileRegistry(self._profile_registry)

    async def activate_profile(
        self,
        profile_name: str,
        session_id: Optional[str] = None,
        trigger: str = "user_request",
    ) -> ToolProfile:
        """Activate a named profile, changing the visible tool set.

        Steps:
        1. Look up the profile in the registry.
        2. Validate token budget constraints.
        3. Compute tools to add and remove relative to current state.
        4. Call ToolManagerPort to apply changes.
        5. Publish ProfileActivated event.

        FastMCP automatically sends notifications/tools/list_changed
        to the MCP client after tools are added/removed.

        Args:
            profile_name: Name of the profile to activate.
            session_id: Optional session context.
            trigger: What triggered the activation.

        Returns:
            The activated ToolProfile.

        Raises:
            KeyError: If profile_name is not registered.
            ValueError: If profile exceeds budget and cannot be adjusted.
        """
        profile = self._profile_registry.get(profile_name)
        if profile is None:
            raise KeyError(
                f"Unknown profile: '{profile_name}'. "
                f"Available: {', '.join(sorted(self._profile_registry))}"
            )

        # Validate budget
        budget_errors = profile.validate_budget(self._descriptors)
        if budget_errors:
            self._publish_event(TokenBudgetExceeded(
                profile_name=profile_name,
                budget_limit=profile.token_budget.tool_budget
                    if profile.token_budget else 0,
                actual_cost=profile.estimate_token_cost(self._descriptors),
                overage=profile.estimate_token_cost(self._descriptors)
                    - (profile.token_budget.tool_budget
                       if profile.token_budget else 0),
                suggested_action="switch_to_compact",
            ))
            # Attempt fallback to more compact description mode
            if profile.description_mode != ToolDescriptionMode.MINIMAL:
                profile = ToolProfile(
                    name=profile.name,
                    tool_names=profile.tool_names,
                    description_mode=ToolDescriptionMode.MINIMAL,
                    model_tier=profile.model_tier,
                    token_budget=profile.token_budget,
                    tags=profile.tags,
                    description=profile.description,
                )
                fallback_errors = profile.validate_budget(self._descriptors)
                if fallback_errors:
                    raise ValueError(
                        f"Profile '{profile_name}' exceeds budget even "
                        f"in MINIMAL mode: {fallback_errors}"
                    )

        # Compute diff — manage_session is pinned (never removed)
        current_tools = await self._tool_manager.get_visible_tool_names()
        target_tools = profile.tool_names | {"manage_session"}
        to_remove = current_tools - target_tools
        to_add = target_tools - current_tools
        to_update = current_tools & target_tools  # may need description swap

        # Apply removals
        for name in to_remove:
            await self._tool_manager.remove_tool(name)

        # Apply additions with correct description mode
        for name in to_add:
            desc = self._descriptors.get(name)
            if desc is None:
                logger.warning(f"No descriptor for tool '{name}', skipping add")
                continue
            await self._tool_manager.add_tool_with_description(
                name,
                desc.description_for_mode(profile.description_mode),
                desc.schema_for_mode(profile.description_mode),
            )

        # Swap descriptions for tools that remain visible but need mode change
        if (
            self._active_profile
            and self._active_profile.description_mode != profile.description_mode
        ):
            for name in to_update:
                desc = self._descriptors.get(name)
                if desc is None:
                    continue
                await self._tool_manager.swap_tool_description(
                    name,
                    desc.description_for_mode(profile.description_mode),
                    desc.schema_for_mode(profile.description_mode),
                )

        # Track previous for transition event
        previous = self._active_profile
        self._active_profile = profile

        # Publish events
        if to_remove:
            self._publish_event(ToolsHidden(
                tool_names=frozenset(to_remove),
                reason="profile_switch",
                profile_name=profile.name,
            ))
        if to_add:
            self._publish_event(ToolsRevealed(
                tool_names=frozenset(to_add),
                reason="profile_switch",
                profile_name=profile.name,
            ))

        self._publish_event(ProfileActivated(
            profile_name=profile.name,
            model_tier=profile.model_tier.value,
            description_mode=profile.description_mode.value,
            tool_names=profile.tool_names,
            tool_count=profile.tool_count,
            estimated_tokens=profile.estimate_token_cost(self._descriptors),
            trigger=trigger,
            session_id=session_id,
        ))

        if previous is not None and previous.name != profile.name:
            self._publish_event(ProfileTransitioned(
                from_profile=previous.name,
                to_profile=profile.name,
                trigger=trigger,
                tools_added=frozenset(to_add),
                tools_removed=frozenset(to_remove),
                session_id=session_id,
            ))

        return profile

    def get_active_profile(self) -> Optional[ToolProfile]:
        """Return the currently active profile, or None if unset."""
        return self._active_profile

    def suggest_profile(
        self, scenario: str, model_tier: ModelTier
    ) -> ToolProfile:
        """Suggest a profile based on scenario keywords and model tier.

        Heuristic logic:
        - LARGE_CONTEXT -> full (always)
        - SMALL_CONTEXT + "api" -> api_exec
        - SMALL_CONTEXT + "browser"/"web"/"ui" -> browser_exec
        - SMALL_CONTEXT + "plan"/"analyze"/"discover" -> discovery
        - STANDARD -> full (most standard models handle 15 tools)

        Args:
            scenario: Natural language scenario description.
            model_tier: The LLM's context tier.

        Returns:
            The suggested ToolProfile.
        """
        if model_tier == ModelTier.LARGE_CONTEXT:
            return self._profile_registry["full"]

        if model_tier == ModelTier.STANDARD:
            return self._profile_registry["full"]

        # SMALL_CONTEXT heuristics
        scenario_lower = scenario.lower()

        if any(kw in scenario_lower for kw in ("api", "rest", "http", "request")):
            return self._profile_registry["api_exec"]

        if any(kw in scenario_lower for kw in (
            "plan", "analyze", "discover", "what libraries"
        )):
            return self._profile_registry["discovery"]

        if any(kw in scenario_lower for kw in (
            "browser", "web", "ui", "click", "navigate", "page"
        )):
            return self._profile_registry["browser_exec"]

        # Default for small context: browser_exec (most common use case)
        return self._profile_registry["browser_exec"]

    async def transition_phase(
        self,
        trigger: str,
        session_id: Optional[str] = None,
    ) -> Optional[ToolProfile]:
        """Transition from the current profile based on a trigger.

        Valid transitions:
        - discovery --[plan_to_execute]--> browser_exec or api_exec
        - browser_exec --[execute_to_recovery]--> browser_exec + get_keyword_info
        - minimal_exec --[escalation]--> browser_exec
        - any --[user_request]--> any registered profile

        Args:
            trigger: The transition trigger string.
            session_id: Optional session context.

        Returns:
            The new ToolProfile, or None if no transition applies.
        """
        if self._active_profile is None:
            return None

        transition_map = {
            ("discovery", ProfileTransition.PLAN_TO_EXECUTE): "browser_exec",
            ("browser_exec", ProfileTransition.EXECUTE_TO_RECOVERY): "browser_exec",
            ("api_exec", ProfileTransition.EXECUTE_TO_RECOVERY): "api_exec",
            ("minimal_exec", ProfileTransition.ESCALATION): "browser_exec",
            ("browser_exec", ProfileTransition.ESCALATION): "full",
            ("api_exec", ProfileTransition.ESCALATION): "full",
        }

        current_name = self._active_profile.name
        target_name = transition_map.get((current_name, trigger))

        if target_name is None:
            logger.debug(
                f"No transition defined for ({current_name}, {trigger})"
            )
            return None

        # For recovery trigger: add get_keyword_info to current profile
        # rather than full profile switch
        if trigger == ProfileTransition.EXECUTE_TO_RECOVERY:
            if not self._active_profile.contains_tool("get_keyword_info"):
                escalated = self._active_profile.with_additional_tool(
                    "get_keyword_info"
                )
                # Register the temporary escalated profile
                self._profile_registry[f"{current_name}_recovery"] = escalated
                return await self.activate_profile(
                    f"{current_name}_recovery",
                    session_id=session_id,
                    trigger=trigger,
                )
            return self._active_profile

        return await self.activate_profile(
            target_name, session_id=session_id, trigger=trigger
        )

    def register_profile(self, profile: ToolProfile) -> None:
        """Register a custom profile in the registry.

        Args:
            profile: The profile to register.
        """
        self._profile_registry[profile.name] = profile

    def list_profiles(self) -> List[str]:
        """List all registered profile names.

        Returns:
            Sorted list of profile names.
        """
        return sorted(self._profile_registry.keys())

    def bind_session_profile(
        self,
        session_id: str,
        profile_name: str,
        trigger: str = "user_request",
    ) -> ToolProfile:
        """Bind a session to a named profile without mutating the global tool registry.

        This is the P1 replacement for the global-mutation path: instead of
        calling remove_tool/add_tool on FastMCP, we record the session's
        preference in SessionToolProfileRegistry. Each tool call then consults
        the registry to decide whether the caller's session is allowed.

        Args:
            session_id: The session that is changing its profile.
            profile_name: Profile to activate for this session.
            trigger: What triggered the binding (for event publishing).

        Returns:
            The bound ToolProfile.

        Raises:
            KeyError: If profile_name is not registered.
        """
        profile = self._profile_registry.get(profile_name)
        if profile is None:
            raise KeyError(
                f"Unknown profile: '{profile_name}'. "
                f"Available: {', '.join(sorted(self._profile_registry))}"
            )
        self._session_registry.bind(session_id, profile_name)
        self._publish_event(ProfileActivated(
            profile_name=profile.name,
            model_tier=profile.model_tier.value,
            description_mode=profile.description_mode.value,
            tool_names=profile.tool_names,
            tool_count=profile.tool_count,
            estimated_tokens=0,
            trigger=trigger,
            session_id=session_id,
        ))
        return profile

    def get_session_registry(self) -> SessionToolProfileRegistry:
        """Return the per-session profile registry (for P1 tool gating)."""
        return self._session_registry

    def is_tool_allowed_for_session(self, session_id: str, tool_name: str) -> bool:
        """Check whether tool_name is in the active profile for session_id.

        Sessions with no explicit binding use the default (full) profile,
        which includes all tools.

        Args:
            session_id: The session identifier (may be empty).
            tool_name: The MCP tool name to check.

        Returns:
            True if the tool is permitted by the session's active profile.
        """
        return self._session_registry.is_tool_allowed(session_id, tool_name)

    def profile_gated_error(self, session_id: str, tool_name: str) -> Dict[str, str]:
        """Return a structured error dict for a profile-gated tool call.

        Args:
            session_id: The session whose profile blocked the call.
            tool_name: The tool that was blocked.

        Returns:
            Dict with success=False and a hint on how to unlock the tool.
        """
        active = self._session_registry.get_profile_name(session_id)
        return {
            "success": False,
            "error": "profile_disabled",
            "tool": tool_name,
            "profile": active,
            "hint": (
                f"Tool '{tool_name}' is not included in the '{active}' profile. "
                f"Call manage_session(action='set_tool_profile', "
                f"profile='full') to enable it, or switch to a profile that "
                f"includes '{tool_name}'."
            ),
        }

    def auto_select_profile(
        self,
        model_tier: ModelTier,
        task_domain: str = "browser",
    ) -> AutoProfileSelection:
        """Automatically select a profile based on model tier and task domain (ADR-016).

        Decision matrix:
        - SMALL_7B -> slim_exec for all domains (most constrained)
        - SMALL_CONTEXT -> domain-specific (browser_exec, api_exec, desktop_exec)
        - MEDIUM_13B -> domain-specific with STANDARD schema mode
        - STANDARD / LARGE_CONTEXT / HOSTED -> full profile

        Args:
            model_tier: The detected model capability tier.
            task_domain: The task domain ("browser", "api", "desktop", etc.).

        Returns:
            AutoProfileSelection with recommendation details.
        """
        domain = task_domain.lower().strip()

        if model_tier == ModelTier.SMALL_7B:
            return AutoProfileSelection(
                model_tier=model_tier,
                task_domain=domain,
                recommended_profile="slim_exec",
                recommended_schema_mode=SchemaMode.MINIMAL,
                recommended_instruction_template="slim_7b",
                confidence=0.95,
                rationale=(
                    "7B models have limited context and tool-calling ability; "
                    "slim_exec with MINIMAL schema maximizes usable context."
                ),
            )

        if model_tier == ModelTier.SMALL_CONTEXT:
            profile_map = {
                "api": "api_exec",
                "desktop": "desktop_exec",
            }
            profile = profile_map.get(domain, "browser_exec")
            return AutoProfileSelection(
                model_tier=model_tier,
                task_domain=domain,
                recommended_profile=profile,
                recommended_schema_mode=SchemaMode.STANDARD,
                recommended_instruction_template="small_context",
                confidence=0.85,
                rationale=(
                    f"Small-context model matched to {profile} profile "
                    f"with STANDARD schema for {domain} domain."
                ),
            )

        if model_tier == ModelTier.MEDIUM_13B:
            profile_map = {
                "api": "api_exec",
                "desktop": "desktop_exec",
            }
            profile = profile_map.get(domain, "browser_exec")
            return AutoProfileSelection(
                model_tier=model_tier,
                task_domain=domain,
                recommended_profile=profile,
                recommended_schema_mode=SchemaMode.STANDARD,
                recommended_instruction_template="medium_13b",
                confidence=0.80,
                rationale=(
                    f"13B model has moderate capacity; {profile} profile "
                    f"with STANDARD schema balances capability and context."
                ),
            )

        # STANDARD, LARGE_CONTEXT, HOSTED -> full
        return AutoProfileSelection(
            model_tier=model_tier,
            task_domain=domain,
            recommended_profile="full",
            recommended_schema_mode=SchemaMode.FULL,
            recommended_instruction_template="full",
            confidence=0.90,
            rationale=(
                f"Model tier {model_tier.value} has sufficient context "
                f"for full profile with complete schemas."
            ),
        )

    def generate_slim_schema(
        self, tool_name: str
    ) -> Optional[SlimToolSchema]:
        """Generate a slim schema for a tool using its full schema (ADR-016).

        Looks up the tool's full schema from the descriptor registry
        and produces a MINIMAL-mode SlimToolSchema.

        Args:
            tool_name: The tool to generate a slim schema for.

        Returns:
            A SlimToolSchema, or None if the tool is not in the registry
            or has no schema.
        """
        descriptor = self._descriptors.get(tool_name)
        if descriptor is None:
            logger.warning(f"No descriptor for tool '{tool_name}', cannot generate slim schema")
            return None

        full_schema = descriptor.schema_full
        if not full_schema or not full_schema.get("properties"):
            logger.debug(f"Tool '{tool_name}' has no properties in schema, skipping slim generation")
            return None

        return SlimToolSchema.from_full_schema(
            tool_name=tool_name,
            full_schema=full_schema,
            mode=SchemaMode.MINIMAL,
        )

    def _publish_event(self, event: object) -> None:
        """Publish a domain event.

        Args:
            event: The event to publish.
        """
        if self._event_publisher:
            try:
                self._event_publisher(event)
            except Exception as e:
                logger.error(f"Failed to publish event: {e}")

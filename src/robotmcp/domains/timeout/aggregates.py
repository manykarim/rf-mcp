"""Timeout Domain Aggregates.

This module contains the aggregate roots for the Timeout bounded context.
Aggregates are clusters of domain objects that are treated as a unit
for data changes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .value_objects import PolicyId, Milliseconds, DefaultTimeouts
from .entities import ActionType, TimeoutCategory


@dataclass
class TimeoutPolicy:
    """Aggregate root for timeout configuration.

    TimeoutPolicy manages all timeout settings for a session, implementing
    the dual timeout strategy (5s action / 60s navigation) with support
    for custom overrides per action type.

    The policy enforces bounds validation to prevent timeouts that are
    too short (causing false failures) or too long (wasting time on
    actual failures).

    Attributes:
        policy_id: Unique identifier for this policy.
        session_id: The session this policy belongs to.
        action_timeout: Default timeout for element actions.
        navigation_timeout: Default timeout for navigation operations.
        assertion_retry: Default timeout for assertion retries.
        read_timeout: Default timeout for read operations.
        custom_overrides: Per-action type timeout overrides.

    Examples:
        >>> policy = TimeoutPolicy.create_default("session_123")
        >>> policy.get_timeout_for(ActionType.CLICK)
        Milliseconds(value=5000)
        >>> policy.get_timeout_for(ActionType.NAVIGATE)
        Milliseconds(value=60000)
        >>> custom = policy.with_override(ActionType.CLICK, Milliseconds(10000))
        >>> custom.get_timeout_for(ActionType.CLICK)
        Milliseconds(value=10000)
    """

    policy_id: PolicyId
    session_id: str
    action_timeout: Milliseconds = field(default_factory=lambda: DefaultTimeouts.ACTION)
    navigation_timeout: Milliseconds = field(default_factory=lambda: DefaultTimeouts.NAVIGATION)
    assertion_retry: Milliseconds = field(default_factory=lambda: DefaultTimeouts.ASSERTION)
    read_timeout: Milliseconds = field(default_factory=lambda: DefaultTimeouts.READ)
    custom_overrides: Dict[ActionType, Milliseconds] = field(default_factory=dict)

    @classmethod
    def create_default(cls, session_id: str) -> "TimeoutPolicy":
        """Create a TimeoutPolicy with default values.

        Args:
            session_id: The session ID this policy is for.

        Returns:
            A new TimeoutPolicy with Playwright MCP default timeouts.
        """
        return cls(
            policy_id=PolicyId.for_session(session_id),
            session_id=session_id,
            action_timeout=DefaultTimeouts.ACTION,
            navigation_timeout=DefaultTimeouts.NAVIGATION,
            assertion_retry=DefaultTimeouts.ASSERTION,
            read_timeout=DefaultTimeouts.READ,
            custom_overrides={},
        )

    def get_timeout_for(self, action: ActionType) -> Milliseconds:
        """Get the effective timeout for an action type.

        This method checks for custom overrides first, then falls back
        to the category-based default timeout.

        Args:
            action: The action type to get timeout for.

        Returns:
            The Milliseconds timeout value to use.
        """
        # Check for custom override first
        if action in self.custom_overrides:
            return self.custom_overrides[action]

        # Fall back to category-based default
        category = TimeoutCategory.categorize(action)
        return self._get_timeout_for_category(category)

    def _get_timeout_for_category(self, category: str) -> Milliseconds:
        """Get the default timeout for a category.

        Args:
            category: One of "navigation", "action", "read", or "assertion".

        Returns:
            The default timeout for the category.
        """
        if category == "navigation":
            return self.navigation_timeout
        elif category == "action":
            return self.action_timeout
        elif category == "assertion":
            return self.assertion_retry
        else:  # "read"
            return self.read_timeout

    def with_override(self, action: ActionType, timeout: Milliseconds) -> "TimeoutPolicy":
        """Create a new policy with an override for a specific action.

        This method returns a new TimeoutPolicy instance, preserving
        immutability patterns for easier reasoning about state changes.

        Args:
            action: The action type to override.
            timeout: The timeout value to use for this action.

        Returns:
            A new TimeoutPolicy with the override applied.
        """
        # Validate and clamp the timeout
        category = TimeoutCategory.categorize(action)
        clamped_timeout = self._clamp_to_bounds(timeout, category)

        # Create new overrides dict
        new_overrides = dict(self.custom_overrides)
        new_overrides[action] = clamped_timeout

        return TimeoutPolicy(
            policy_id=self.policy_id,
            session_id=self.session_id,
            action_timeout=self.action_timeout,
            navigation_timeout=self.navigation_timeout,
            assertion_retry=self.assertion_retry,
            read_timeout=self.read_timeout,
            custom_overrides=new_overrides,
        )

    def with_action_timeout(self, timeout: Milliseconds) -> "TimeoutPolicy":
        """Create a new policy with a different action timeout.

        Args:
            timeout: The new action timeout value.

        Returns:
            A new TimeoutPolicy with the updated action timeout.
        """
        clamped = self._clamp_to_bounds(timeout, "action")
        return TimeoutPolicy(
            policy_id=self.policy_id,
            session_id=self.session_id,
            action_timeout=clamped,
            navigation_timeout=self.navigation_timeout,
            assertion_retry=self.assertion_retry,
            read_timeout=self.read_timeout,
            custom_overrides=dict(self.custom_overrides),
        )

    def with_navigation_timeout(self, timeout: Milliseconds) -> "TimeoutPolicy":
        """Create a new policy with a different navigation timeout.

        Args:
            timeout: The new navigation timeout value.

        Returns:
            A new TimeoutPolicy with the updated navigation timeout.
        """
        clamped = self._clamp_to_bounds(timeout, "navigation")
        return TimeoutPolicy(
            policy_id=self.policy_id,
            session_id=self.session_id,
            action_timeout=self.action_timeout,
            navigation_timeout=clamped,
            assertion_retry=self.assertion_retry,
            read_timeout=self.read_timeout,
            custom_overrides=dict(self.custom_overrides),
        )

    def clear_override(self, action: ActionType) -> "TimeoutPolicy":
        """Create a new policy without an override for a specific action.

        Args:
            action: The action type to remove the override for.

        Returns:
            A new TimeoutPolicy without the override.
        """
        new_overrides = dict(self.custom_overrides)
        new_overrides.pop(action, None)

        return TimeoutPolicy(
            policy_id=self.policy_id,
            session_id=self.session_id,
            action_timeout=self.action_timeout,
            navigation_timeout=self.navigation_timeout,
            assertion_retry=self.assertion_retry,
            read_timeout=self.read_timeout,
            custom_overrides=new_overrides,
        )

    def clear_all_overrides(self) -> "TimeoutPolicy":
        """Create a new policy with all overrides cleared.

        Returns:
            A new TimeoutPolicy with no custom overrides.
        """
        return TimeoutPolicy(
            policy_id=self.policy_id,
            session_id=self.session_id,
            action_timeout=self.action_timeout,
            navigation_timeout=self.navigation_timeout,
            assertion_retry=self.assertion_retry,
            read_timeout=self.read_timeout,
            custom_overrides={},
        )

    def validate(self) -> List[str]:
        """Validate the timeout policy configuration.

        Checks that all timeout values are within acceptable bounds
        and returns a list of any validation errors.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        # Validate action timeout
        min_action, max_action = DefaultTimeouts.get_bounds_for_category("action")
        if self.action_timeout < min_action:
            errors.append(
                f"Action timeout {self.action_timeout} is below minimum {min_action}"
            )
        if self.action_timeout > max_action:
            errors.append(
                f"Action timeout {self.action_timeout} exceeds maximum {max_action}"
            )

        # Validate navigation timeout
        min_nav, max_nav = DefaultTimeouts.get_bounds_for_category("navigation")
        if self.navigation_timeout < min_nav:
            errors.append(
                f"Navigation timeout {self.navigation_timeout} is below minimum {min_nav}"
            )
        if self.navigation_timeout > max_nav:
            errors.append(
                f"Navigation timeout {self.navigation_timeout} exceeds maximum {max_nav}"
            )

        # Validate assertion retry timeout
        if self.assertion_retry < min_action:
            errors.append(
                f"Assertion retry timeout {self.assertion_retry} is below minimum {min_action}"
            )

        # Validate read timeout
        if self.read_timeout < min_action:
            errors.append(
                f"Read timeout {self.read_timeout} is below minimum {min_action}"
            )

        # Validate custom overrides
        for action, timeout in self.custom_overrides.items():
            category = TimeoutCategory.categorize(action)
            min_bound, max_bound = DefaultTimeouts.get_bounds_for_category(category)

            if timeout < min_bound:
                errors.append(
                    f"Override for {action.value} ({timeout}) is below minimum {min_bound}"
                )
            if timeout > max_bound:
                errors.append(
                    f"Override for {action.value} ({timeout}) exceeds maximum {max_bound}"
                )

        return errors

    def _clamp_to_bounds(self, timeout: Milliseconds, category: str) -> Milliseconds:
        """Clamp a timeout value to the bounds for its category.

        Args:
            timeout: The timeout value to clamp.
            category: The category to get bounds from.

        Returns:
            The clamped timeout value.
        """
        min_bound, max_bound = DefaultTimeouts.get_bounds_for_category(category)

        if timeout < min_bound:
            return min_bound
        if timeout > max_bound:
            return max_bound
        return timeout

    def has_override(self, action: ActionType) -> bool:
        """Check if there's a custom override for an action.

        Args:
            action: The action type to check.

        Returns:
            True if there's a custom override, False otherwise.
        """
        return action in self.custom_overrides

    def get_override(self, action: ActionType) -> Optional[Milliseconds]:
        """Get the custom override for an action, if any.

        Args:
            action: The action type to check.

        Returns:
            The override timeout, or None if no override exists.
        """
        return self.custom_overrides.get(action)

    def to_dict(self) -> Dict:
        """Convert the policy to a dictionary representation.

        Returns:
            Dictionary with all policy settings.
        """
        return {
            "policy_id": str(self.policy_id),
            "session_id": self.session_id,
            "action_timeout_ms": self.action_timeout.value,
            "navigation_timeout_ms": self.navigation_timeout.value,
            "assertion_retry_ms": self.assertion_retry.value,
            "read_timeout_ms": self.read_timeout.value,
            "custom_overrides": {
                action.value: timeout.value
                for action, timeout in self.custom_overrides.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "TimeoutPolicy":
        """Create a TimeoutPolicy from a dictionary.

        Args:
            data: Dictionary with policy settings.

        Returns:
            A new TimeoutPolicy instance.
        """
        custom_overrides = {}
        if "custom_overrides" in data:
            for action_str, timeout_ms in data["custom_overrides"].items():
                action = ActionType(action_str)
                custom_overrides[action] = Milliseconds(timeout_ms)

        return cls(
            policy_id=PolicyId(data.get("policy_id", "")),
            session_id=data.get("session_id", ""),
            action_timeout=Milliseconds(data.get("action_timeout_ms", DefaultTimeouts.ACTION.value)),
            navigation_timeout=Milliseconds(data.get("navigation_timeout_ms", DefaultTimeouts.NAVIGATION.value)),
            assertion_retry=Milliseconds(data.get("assertion_retry_ms", DefaultTimeouts.ASSERTION.value)),
            read_timeout=Milliseconds(data.get("read_timeout_ms", DefaultTimeouts.READ.value)),
            custom_overrides=custom_overrides,
        )

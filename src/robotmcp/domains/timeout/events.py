"""Timeout Domain Events.

This module contains domain events for the Timeout bounded context.
Domain events represent something that happened in the domain that
domain experts care about.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

from .value_objects import PolicyId, Milliseconds
from .entities import ActionType


@dataclass
class TimeoutPolicyCreated:
    """Event emitted when a new timeout policy is created.

    This event is published when a new TimeoutPolicy aggregate is
    instantiated for a session, allowing other bounded contexts
    to react to policy creation.

    Attributes:
        policy_id: The unique identifier for the created policy.
        session_id: The session the policy belongs to.
        action_timeout: The configured action timeout.
        navigation_timeout: The configured navigation timeout.
        timestamp: When the policy was created.
    """
    policy_id: PolicyId
    session_id: str
    action_timeout: Milliseconds
    navigation_timeout: Milliseconds
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "event_type": "TimeoutPolicyCreated",
            "policy_id": str(self.policy_id),
            "session_id": self.session_id,
            "action_timeout_ms": self.action_timeout.value,
            "navigation_timeout_ms": self.navigation_timeout.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TimeoutExceeded:
    """Event emitted when an operation exceeds its timeout.

    This event provides detailed information about timeout failures,
    enabling monitoring, debugging, and potential automatic retry
    with adjusted timeouts.

    Attributes:
        execution_id: The ID of the execution that timed out.
        action_type: The type of action that timed out.
        timeout_used: The timeout value that was exceeded.
        elapsed: The actual elapsed time before timeout.
        category: The timeout category for the action.
        error_message: Optional detailed error message.
        timestamp: When the timeout occurred.
    """
    execution_id: str
    action_type: ActionType
    timeout_used: Milliseconds
    elapsed: Milliseconds
    category: Literal["navigation", "action", "read", "assertion"]
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def exceeded_by(self) -> Milliseconds:
        """Calculate how much the timeout was exceeded by.

        Returns:
            The amount of time exceeded (may be 0 if elapsed <= timeout).
        """
        if self.elapsed.value > self.timeout_used.value:
            return Milliseconds(self.elapsed.value - self.timeout_used.value)
        return Milliseconds(0)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "event_type": "TimeoutExceeded",
            "execution_id": self.execution_id,
            "action_type": self.action_type.value,
            "timeout_used_ms": self.timeout_used.value,
            "elapsed_ms": self.elapsed.value,
            "exceeded_by_ms": self.exceeded_by.value,
            "category": self.category,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TimeoutOverrideApplied:
    """Event emitted when a custom timeout override is used.

    This event tracks when the system uses a custom timeout instead
    of the category default, which is useful for debugging timeout
    configurations and understanding policy behavior.

    Attributes:
        policy_id: The policy that contains the override.
        action_type: The action type that was overridden.
        default_timeout: What the default timeout would have been.
        override_timeout: The actual timeout used.
        timestamp: When the override was applied.
    """
    policy_id: PolicyId
    action_type: ActionType
    default_timeout: Milliseconds
    override_timeout: Milliseconds
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def difference(self) -> Milliseconds:
        """Calculate the difference between override and default.

        Returns:
            Positive if override is longer, negative if shorter.
        """
        diff = self.override_timeout.value - self.default_timeout.value
        return Milliseconds(abs(diff))

    @property
    def is_longer_than_default(self) -> bool:
        """Check if the override is longer than the default.

        Returns:
            True if override > default, False otherwise.
        """
        return self.override_timeout.value > self.default_timeout.value

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "event_type": "TimeoutOverrideApplied",
            "policy_id": str(self.policy_id),
            "action_type": self.action_type.value,
            "default_timeout_ms": self.default_timeout.value,
            "override_timeout_ms": self.override_timeout.value,
            "difference_ms": self.difference.value,
            "is_longer": self.is_longer_than_default,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TimeoutPolicyUpdated:
    """Event emitted when a timeout policy is modified.

    This event tracks changes to timeout policies, enabling audit
    trails and reactive updates in dependent components.

    Attributes:
        policy_id: The policy that was updated.
        session_id: The session the policy belongs to.
        changed_field: Which field was changed.
        old_value: The previous timeout value.
        new_value: The new timeout value.
        timestamp: When the update occurred.
    """
    policy_id: PolicyId
    session_id: str
    changed_field: str
    old_value: Milliseconds
    new_value: Milliseconds
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "event_type": "TimeoutPolicyUpdated",
            "policy_id": str(self.policy_id),
            "session_id": self.session_id,
            "changed_field": self.changed_field,
            "old_value_ms": self.old_value.value,
            "new_value_ms": self.new_value.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TimeoutWarning:
    """Event emitted when a timeout is approaching but not yet exceeded.

    This event can be used to implement early warning systems that
    alert when operations are taking longer than expected.

    Attributes:
        execution_id: The ID of the execution in progress.
        action_type: The type of action being executed.
        timeout_configured: The configured timeout value.
        elapsed_so_far: How much time has elapsed.
        warning_threshold: The threshold that triggered the warning (e.g., 0.75).
        timestamp: When the warning was raised.
    """
    execution_id: str
    action_type: ActionType
    timeout_configured: Milliseconds
    elapsed_so_far: Milliseconds
    warning_threshold: float = 0.75  # Warning at 75% of timeout
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def percentage_used(self) -> float:
        """Calculate what percentage of the timeout has been used.

        Returns:
            Percentage as a float (e.g., 0.75 for 75%).
        """
        if self.timeout_configured.value == 0:
            return 1.0
        return self.elapsed_so_far.value / self.timeout_configured.value

    @property
    def time_remaining(self) -> Milliseconds:
        """Calculate how much time remains before timeout.

        Returns:
            Remaining time in milliseconds.
        """
        remaining = self.timeout_configured.value - self.elapsed_so_far.value
        return Milliseconds(max(0, remaining))

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "event_type": "TimeoutWarning",
            "execution_id": self.execution_id,
            "action_type": self.action_type.value,
            "timeout_configured_ms": self.timeout_configured.value,
            "elapsed_so_far_ms": self.elapsed_so_far.value,
            "percentage_used": self.percentage_used,
            "time_remaining_ms": self.time_remaining.value,
            "warning_threshold": self.warning_threshold,
            "timestamp": self.timestamp.isoformat(),
        }

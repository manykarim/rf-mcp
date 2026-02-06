"""Timeout Domain Services.

This module contains domain services for the Timeout bounded context.
Domain services contain business logic that doesn't naturally fit
within an entity or value object.
"""

from typing import Dict, List, Optional, Callable
from datetime import datetime
import logging

from .value_objects import PolicyId, Milliseconds, DefaultTimeouts
from .entities import ActionType, TimeoutCategory
from .aggregates import TimeoutPolicy
from .events import (
    TimeoutPolicyCreated,
    TimeoutExceeded,
    TimeoutOverrideApplied,
    TimeoutPolicyUpdated,
    TimeoutWarning,
)

logger = logging.getLogger(__name__)


class TimeoutService:
    """Service for managing timeout policies.

    This service provides high-level operations for creating, querying,
    and managing timeout policies. It acts as the primary interface
    for timeout-related operations in the system.

    The service follows the dual timeout strategy:
    - 5 seconds for element actions (clicks, typing, etc.)
    - 60 seconds for navigation operations (page loads, reloads)

    This strategy balances responsiveness (fast failure detection for
    element actions) with reliability (allowing time for navigation).

    Examples:
        >>> service = TimeoutService()
        >>> policy = service.create_default_policy("session_123")
        >>> timeout = service.get_timeout_for_action(policy, ActionType.CLICK)
        >>> print(timeout)
        5000ms
    """

    def __init__(
        self,
        event_publisher: Optional[Callable[[object], None]] = None,
    ) -> None:
        """Initialize the timeout service.

        Args:
            event_publisher: Optional callback for publishing domain events.
        """
        self._policies: Dict[str, TimeoutPolicy] = {}
        self._event_publisher = event_publisher

    def create_default_policy(self, session_id: str) -> TimeoutPolicy:
        """Create a default timeout policy for a session.

        Creates a new TimeoutPolicy with Playwright MCP default values:
        - Action timeout: 5 seconds
        - Navigation timeout: 60 seconds
        - Assertion retry: 10 seconds
        - Read timeout: 2 seconds

        Args:
            session_id: The session to create the policy for.

        Returns:
            A new TimeoutPolicy with default configuration.
        """
        policy = TimeoutPolicy.create_default(session_id)
        self._policies[session_id] = policy

        # Publish creation event
        self._publish_event(TimeoutPolicyCreated(
            policy_id=policy.policy_id,
            session_id=session_id,
            action_timeout=policy.action_timeout,
            navigation_timeout=policy.navigation_timeout,
        ))

        logger.debug(
            f"Created default timeout policy for session {session_id}: "
            f"action={policy.action_timeout}, navigation={policy.navigation_timeout}"
        )

        return policy

    def get_policy(self, session_id: str) -> Optional[TimeoutPolicy]:
        """Get the timeout policy for a session.

        Args:
            session_id: The session to get the policy for.

        Returns:
            The TimeoutPolicy if it exists, None otherwise.
        """
        return self._policies.get(session_id)

    def get_or_create_policy(self, session_id: str) -> TimeoutPolicy:
        """Get or create a timeout policy for a session.

        Args:
            session_id: The session to get/create the policy for.

        Returns:
            The existing or newly created TimeoutPolicy.
        """
        policy = self.get_policy(session_id)
        if policy is None:
            policy = self.create_default_policy(session_id)
        return policy

    def get_timeout_for_action(
        self,
        policy: TimeoutPolicy,
        action: ActionType,
    ) -> Milliseconds:
        """Get the effective timeout for an action.

        This method determines the appropriate timeout for an action,
        checking for custom overrides and applying category defaults.
        It also publishes events when overrides are used.

        Args:
            policy: The timeout policy to use.
            action: The action type to get timeout for.

        Returns:
            The effective timeout in milliseconds.
        """
        timeout = policy.get_timeout_for(action)

        # If there's an override, publish an event
        if policy.has_override(action):
            category = TimeoutCategory.categorize(action)
            default_timeout = DefaultTimeouts.get_default_for_category(category)

            self._publish_event(TimeoutOverrideApplied(
                policy_id=policy.policy_id,
                action_type=action,
                default_timeout=default_timeout,
                override_timeout=timeout,
            ))

        return timeout

    def validate_timeout_bounds(
        self,
        timeout: Milliseconds,
        category: str,
    ) -> Milliseconds:
        """Validate and clamp a timeout to category bounds.

        This method ensures timeout values are within acceptable ranges,
        clamping to minimum or maximum bounds as needed.

        Args:
            timeout: The timeout value to validate.
            category: The category for bounds checking ("navigation", "action", "read").

        Returns:
            The validated/clamped timeout value.
        """
        min_bound, max_bound = DefaultTimeouts.get_bounds_for_category(category)

        if timeout < min_bound:
            logger.warning(
                f"Timeout {timeout} below minimum for {category}, clamping to {min_bound}"
            )
            return min_bound

        if timeout > max_bound:
            logger.warning(
                f"Timeout {timeout} above maximum for {category}, clamping to {max_bound}"
            )
            return max_bound

        return timeout

    def update_action_timeout(
        self,
        session_id: str,
        timeout: Milliseconds,
    ) -> TimeoutPolicy:
        """Update the action timeout for a session.

        Args:
            session_id: The session to update.
            timeout: The new action timeout value.

        Returns:
            The updated TimeoutPolicy.

        Raises:
            KeyError: If no policy exists for the session.
        """
        policy = self.get_policy(session_id)
        if policy is None:
            raise KeyError(f"No policy exists for session {session_id}")

        old_timeout = policy.action_timeout
        new_policy = policy.with_action_timeout(timeout)
        self._policies[session_id] = new_policy

        self._publish_event(TimeoutPolicyUpdated(
            policy_id=new_policy.policy_id,
            session_id=session_id,
            changed_field="action_timeout",
            old_value=old_timeout,
            new_value=new_policy.action_timeout,
        ))

        return new_policy

    def update_navigation_timeout(
        self,
        session_id: str,
        timeout: Milliseconds,
    ) -> TimeoutPolicy:
        """Update the navigation timeout for a session.

        Args:
            session_id: The session to update.
            timeout: The new navigation timeout value.

        Returns:
            The updated TimeoutPolicy.

        Raises:
            KeyError: If no policy exists for the session.
        """
        policy = self.get_policy(session_id)
        if policy is None:
            raise KeyError(f"No policy exists for session {session_id}")

        old_timeout = policy.navigation_timeout
        new_policy = policy.with_navigation_timeout(timeout)
        self._policies[session_id] = new_policy

        self._publish_event(TimeoutPolicyUpdated(
            policy_id=new_policy.policy_id,
            session_id=session_id,
            changed_field="navigation_timeout",
            old_value=old_timeout,
            new_value=new_policy.navigation_timeout,
        ))

        return new_policy

    def set_override(
        self,
        session_id: str,
        action: ActionType,
        timeout: Milliseconds,
    ) -> TimeoutPolicy:
        """Set a custom timeout override for an action.

        Args:
            session_id: The session to update.
            action: The action type to override.
            timeout: The timeout value to use.

        Returns:
            The updated TimeoutPolicy.

        Raises:
            KeyError: If no policy exists for the session.
        """
        policy = self.get_policy(session_id)
        if policy is None:
            raise KeyError(f"No policy exists for session {session_id}")

        new_policy = policy.with_override(action, timeout)
        self._policies[session_id] = new_policy

        return new_policy

    def clear_override(
        self,
        session_id: str,
        action: ActionType,
    ) -> TimeoutPolicy:
        """Clear a custom timeout override for an action.

        Args:
            session_id: The session to update.
            action: The action type to clear override for.

        Returns:
            The updated TimeoutPolicy.

        Raises:
            KeyError: If no policy exists for the session.
        """
        policy = self.get_policy(session_id)
        if policy is None:
            raise KeyError(f"No policy exists for session {session_id}")

        new_policy = policy.clear_override(action)
        self._policies[session_id] = new_policy

        return new_policy

    def record_timeout_exceeded(
        self,
        execution_id: str,
        action: ActionType,
        timeout_used: Milliseconds,
        elapsed: Milliseconds,
        error_message: Optional[str] = None,
    ) -> TimeoutExceeded:
        """Record that a timeout was exceeded.

        This method creates and publishes a TimeoutExceeded event,
        which can be used for monitoring and debugging.

        Args:
            execution_id: The execution that timed out.
            action: The action that timed out.
            timeout_used: The timeout that was exceeded.
            elapsed: The actual elapsed time.
            error_message: Optional error details.

        Returns:
            The TimeoutExceeded event that was published.
        """
        category = TimeoutCategory.categorize(action)

        event = TimeoutExceeded(
            execution_id=execution_id,
            action_type=action,
            timeout_used=timeout_used,
            elapsed=elapsed,
            category=category,
            error_message=error_message,
        )

        self._publish_event(event)

        logger.warning(
            f"Timeout exceeded for {action.value}: "
            f"configured={timeout_used}, elapsed={elapsed}, "
            f"exceeded_by={event.exceeded_by}"
        )

        return event

    def check_timeout_warning(
        self,
        execution_id: str,
        action: ActionType,
        timeout_configured: Milliseconds,
        elapsed: Milliseconds,
        warning_threshold: float = 0.75,
    ) -> Optional[TimeoutWarning]:
        """Check if elapsed time warrants a timeout warning.

        This method checks if the elapsed time has exceeded a threshold
        percentage of the configured timeout and publishes a warning
        event if so.

        Args:
            execution_id: The execution to check.
            action: The action being executed.
            timeout_configured: The configured timeout.
            elapsed: The elapsed time so far.
            warning_threshold: Percentage at which to warn (default 0.75).

        Returns:
            TimeoutWarning event if threshold exceeded, None otherwise.
        """
        if timeout_configured.value == 0:
            return None

        percentage = elapsed.value / timeout_configured.value

        if percentage >= warning_threshold:
            event = TimeoutWarning(
                execution_id=execution_id,
                action_type=action,
                timeout_configured=timeout_configured,
                elapsed_so_far=elapsed,
                warning_threshold=warning_threshold,
            )

            self._publish_event(event)

            logger.info(
                f"Timeout warning for {action.value}: "
                f"{percentage:.0%} of timeout used, "
                f"{event.time_remaining} remaining"
            )

            return event

        return None

    def delete_policy(self, session_id: str) -> bool:
        """Delete the timeout policy for a session.

        Args:
            session_id: The session to delete the policy for.

        Returns:
            True if a policy was deleted, False if none existed.
        """
        if session_id in self._policies:
            del self._policies[session_id]
            logger.debug(f"Deleted timeout policy for session {session_id}")
            return True
        return False

    def get_all_policies(self) -> Dict[str, TimeoutPolicy]:
        """Get all active timeout policies.

        Returns:
            Dictionary mapping session IDs to their policies.
        """
        return dict(self._policies)

    def get_recommended_timeout(
        self,
        action: ActionType,
        context: Optional[Dict] = None,
    ) -> Milliseconds:
        """Get a recommended timeout for an action based on context.

        This method can be extended to provide intelligent timeout
        recommendations based on historical data, network conditions,
        or other contextual factors.

        Args:
            action: The action to get a recommendation for.
            context: Optional context for intelligent recommendations.

        Returns:
            Recommended timeout value.
        """
        category = TimeoutCategory.categorize(action)
        base_timeout = DefaultTimeouts.get_default_for_category(category)

        # Future: Apply context-based adjustments
        # For now, just return the default
        return base_timeout

    def _publish_event(self, event: object) -> None:
        """Publish a domain event.

        Args:
            event: The event to publish.
        """
        if self._event_publisher:
            try:
                self._event_publisher(event)
            except Exception as e:
                logger.error(f"Error publishing event {type(event).__name__}: {e}")


class TimeoutContextManager:
    """Context manager for tracking operation timeouts.

    This class provides a convenient way to track execution time
    and automatically handle timeout events.

    Examples:
        >>> service = TimeoutService()
        >>> async with TimeoutContextManager(
        ...     service=service,
        ...     execution_id="exec_1",
        ...     action=ActionType.CLICK,
        ...     timeout=Milliseconds(5000),
        ... ) as ctx:
        ...     await perform_click()
        ...     ctx.check_warning()
    """

    def __init__(
        self,
        service: TimeoutService,
        execution_id: str,
        action: ActionType,
        timeout: Milliseconds,
        warning_threshold: float = 0.75,
    ) -> None:
        """Initialize the context manager.

        Args:
            service: The TimeoutService for event publishing.
            execution_id: The execution being tracked.
            action: The action being performed.
            timeout: The configured timeout.
            warning_threshold: When to warn (as percentage of timeout).
        """
        self.service = service
        self.execution_id = execution_id
        self.action = action
        self.timeout = timeout
        self.warning_threshold = warning_threshold
        self._start_time: Optional[datetime] = None
        self._warned = False

    def __enter__(self) -> "TimeoutContextManager":
        """Enter the context, recording start time."""
        self._start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context, handling any timeout exceptions."""
        if exc_type is not None:
            # If there was a timeout-related exception, record it
            elapsed = self.elapsed
            if elapsed > self.timeout:
                self.service.record_timeout_exceeded(
                    execution_id=self.execution_id,
                    action=self.action,
                    timeout_used=self.timeout,
                    elapsed=elapsed,
                    error_message=str(exc_val) if exc_val else None,
                )

    @property
    def elapsed(self) -> Milliseconds:
        """Get the elapsed time since start.

        Returns:
            Elapsed time in milliseconds.
        """
        if self._start_time is None:
            return Milliseconds(0)

        delta = datetime.now() - self._start_time
        return Milliseconds(int(delta.total_seconds() * 1000))

    def check_warning(self) -> Optional[TimeoutWarning]:
        """Check if a timeout warning should be issued.

        Returns:
            TimeoutWarning if threshold exceeded (and not already warned).
        """
        if self._warned:
            return None

        warning = self.service.check_timeout_warning(
            execution_id=self.execution_id,
            action=self.action,
            timeout_configured=self.timeout,
            elapsed=self.elapsed,
            warning_threshold=self.warning_threshold,
        )

        if warning:
            self._warned = True

        return warning

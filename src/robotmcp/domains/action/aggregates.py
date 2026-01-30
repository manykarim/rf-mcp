"""Action Context Aggregates.

The ActionExecution aggregate is the central entity managing action
execution lifecycle, including pre-validation, execution, and response building.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, Protocol, TYPE_CHECKING

from robotmcp.domains.action.value_objects import (
    ActionParameters,
    ExecutionId,
    FilteredResponse,
    PreValidationResult,
    ResponseConfig,
)
from robotmcp.domains.action.events import (
    ActionCompleted,
    ActionFailed,
    ActionRetrying,
    ActionStarted,
    PreValidationCompleted,
    TimeoutExceeded,
)
from robotmcp.domains.action.services import (
    BrowserAdapter,
    ElementRegistry,
    PreValidator,
)
from robotmcp.domains.action.response_builder import ResponseBuilder
from robotmcp.domains.shared.kernel import ElementRef
from robotmcp.domains.timeout.value_objects import Milliseconds

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class PageSnapshot(Protocol):
    """Protocol for page snapshot interface.

    Defines the interface for snapshots from the Snapshot Context.
    """

    def to_yaml(self) -> str:
        """Convert snapshot to YAML format."""
        ...

    def estimate_tokens(self) -> int:
        """Estimate token count for this snapshot."""
        ...


class EventPublisher(Protocol):
    """Protocol for event publishing.

    Defines the interface for publishing domain events.
    """

    def publish(self, event: Any) -> None:
        """Publish a domain event."""
        ...


@dataclass
class ActionExecution:
    """Aggregate root for action execution.

    Manages the complete lifecycle of an action execution including:
    - Pre-validation of element actionability
    - Action execution with timeout management
    - Response building with token optimization

    This aggregate coordinates between the Action Context's internal
    services and external contexts (Element Registry, Snapshot, Timeout).

    Attributes:
        execution_id: Unique identifier for this execution
        session_id: The session this action belongs to
        action_type: Type of action being executed (e.g., "click", "fill")
        target_ref: Element reference for element actions
        parameters: Action parameters and options
        pre_validation: Result of pre-validation checks
        result: The action execution result
        response_config: Configuration for response filtering
        started_at: When execution started
        completed_at: When execution completed
        error: Error message if execution failed
        events: Domain events emitted during execution
    """
    execution_id: ExecutionId
    session_id: str
    action_type: str
    target_ref: Optional[ElementRef]
    parameters: ActionParameters
    response_config: ResponseConfig = field(default_factory=ResponseConfig.standard)

    # Execution state
    pre_validation: Optional[PreValidationResult] = None
    result: Optional[Any] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    # Domain events
    _events: List[Any] = field(default_factory=list, repr=False)

    @classmethod
    def create(
        cls,
        session_id: str,
        action_type: str,
        target_ref: Optional[ElementRef] = None,
        parameters: Optional[ActionParameters] = None,
        response_config: Optional[ResponseConfig] = None,
    ) -> "ActionExecution":
        """Factory method to create a new ActionExecution.

        Args:
            session_id: The session ID
            action_type: Type of action to execute
            target_ref: Optional element reference
            parameters: Optional action parameters
            response_config: Optional response configuration

        Returns:
            A new ActionExecution instance
        """
        return cls(
            execution_id=ExecutionId.generate(),
            session_id=session_id,
            action_type=action_type,
            target_ref=target_ref,
            parameters=parameters or ActionParameters(),
            response_config=response_config or ResponseConfig.standard(),
        )

    @property
    def is_started(self) -> bool:
        """Check if execution has started."""
        return self.started_at is not None

    @property
    def is_completed(self) -> bool:
        """Check if execution has completed."""
        return self.completed_at is not None

    @property
    def is_successful(self) -> bool:
        """Check if execution completed successfully."""
        return self.is_completed and self.error is None

    @property
    def duration_ms(self) -> float:
        """Get execution duration in milliseconds."""
        if self.started_at is None:
            return 0.0
        end = self.completed_at or datetime.now(timezone.utc)
        return (end - self.started_at).total_seconds() * 1000

    @property
    def events(self) -> List[Any]:
        """Get domain events emitted during execution."""
        return list(self._events)

    def clear_events(self) -> List[Any]:
        """Clear and return all pending events."""
        events = list(self._events)
        self._events.clear()
        return events

    def _emit_event(self, event: Any) -> None:
        """Emit a domain event.

        Args:
            event: The event to emit
        """
        self._events.append(event)
        logger.debug(f"Event emitted: {event.event_type}")

    def start(self) -> None:
        """Mark execution as started.

        Emits ActionStarted event.
        """
        self.started_at = datetime.now(timezone.utc)
        self._emit_event(ActionStarted(
            execution_id=self.execution_id,
            session_id=self.session_id,
            action_type=self.action_type,
            target_ref=self.target_ref,
        ))

    def validate_preconditions(
        self,
        validator: PreValidator,
        registry: ElementRegistry,
        browser_adapter: BrowserAdapter,
    ) -> PreValidationResult:
        """Run pre-validation checks.

        Validates that the target element (if any) is in a suitable
        state for the action. Emits PreValidationCompleted event.

        Args:
            validator: The pre-validator service
            registry: Element registry for ref lookup
            browser_adapter: Browser adapter for state checks

        Returns:
            PreValidationResult with check outcomes
        """
        start_time = time.perf_counter()

        if self.target_ref is None:
            # No element to validate (e.g., navigation action)
            self.pre_validation = PreValidationResult.skipped()
        else:
            self.pre_validation = validator.run_all_checks(
                action_type=self.action_type,
                ref=self.target_ref,
                registry=registry,
                browser_adapter=browser_adapter,
            )

        duration_ms = (time.perf_counter() - start_time) * 1000

        self._emit_event(PreValidationCompleted(
            execution_id=self.execution_id,
            result=self.pre_validation,
            duration_ms=duration_ms,
        ))

        return self.pre_validation

    def execute(
        self,
        action_handler: Callable[[str, Optional[str], ActionParameters], Any],
        locator: Optional[str],
        timeout: Milliseconds,
    ) -> Any:
        """Execute the action.

        Args:
            action_handler: Function that executes the browser action
            locator: The element locator string (if applicable)
            timeout: Timeout for the action

        Returns:
            The action result

        Raises:
            Exception: If action execution fails
        """
        if not self.is_started:
            self.start()

        # Check pre-validation passed
        if self.pre_validation and not self.pre_validation.passed:
            error_msg = self.pre_validation.failure_reason or "Pre-validation failed"
            self._fail("pre_validation", error_msg)
            raise ValueError(error_msg)

        try:
            # Execute the action
            start_time = time.perf_counter()

            # Add timeout to parameters if not overridden
            params = self.parameters
            if not params.has_timeout_override:
                params = params.with_timeout(timeout.value)

            self.result = action_handler(self.action_type, locator, params)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Check for timeout
            if elapsed_ms > timeout.value:
                self._emit_event(TimeoutExceeded(
                    execution_id=self.execution_id,
                    action_type=self.action_type,
                    timeout_used_ms=timeout.value,
                    elapsed_ms=elapsed_ms,
                    category=self._get_timeout_category(),
                ))

            return self.result

        except Exception as e:
            self._fail("execution", str(e))
            raise

    def _get_timeout_category(self) -> str:
        """Determine the timeout category for this action.

        Returns:
            "navigation", "action", or "read"
        """
        action_lower = self.action_type.lower().replace(" ", "_")

        navigation_actions = {
            "navigate", "go_to", "reload", "go_back", "go_forward"
        }
        read_actions = {
            "get_text", "get_attribute", "get_value", "get_url",
            "get_title", "get_page_source", "get_aria_snapshot"
        }

        if action_lower in navigation_actions:
            return "navigation"
        elif action_lower in read_actions:
            return "read"
        else:
            return "action"

    def complete(self, result: Any = None) -> None:
        """Mark execution as completed successfully.

        Args:
            result: Optional result to store
        """
        if result is not None:
            self.result = result
        self.completed_at = datetime.now(timezone.utc)

    def _fail(self, stage: str, error: str) -> None:
        """Mark execution as failed.

        Args:
            stage: Which stage failed
            error: Error message
        """
        self.error = error
        self.completed_at = datetime.now(timezone.utc)

        self._emit_event(ActionFailed(
            execution_id=self.execution_id,
            error=error,
            failed_at_stage=stage,  # type: ignore
            duration_ms=self.duration_ms,
            recoverable=self._is_recoverable_error(error),
        ))

    def _is_recoverable_error(self, error: str) -> bool:
        """Check if an error might be recoverable with retry.

        Args:
            error: The error message

        Returns:
            True if error might be recoverable
        """
        recoverable_patterns = [
            "timeout",
            "not found",
            "not visible",
            "not stable",
            "stale",
            "detached",
        ]
        error_lower = error.lower()
        return any(pattern in error_lower for pattern in recoverable_patterns)

    def build_response(
        self,
        snapshot: Optional[PageSnapshot] = None,
    ) -> FilteredResponse:
        """Build the token-optimized response.

        Args:
            snapshot: Optional page snapshot to include

        Returns:
            FilteredResponse with optimized content

        Raises:
            ValueError: If execution not completed
        """
        if not self.is_completed:
            raise ValueError("Cannot build response before execution completes")

        try:
            builder = ResponseBuilder(self.response_config)

            # Get snapshot YAML if provided
            snapshot_yaml = None
            if snapshot and self.response_config.include_snapshot:
                snapshot_yaml = snapshot.to_yaml()

            response = builder.build(
                action=self.action_type,
                action_result=self.result,
                snapshot=snapshot_yaml,
                error=self.error,
                ref=self.target_ref.value if self.target_ref else None,
            )

            self._emit_event(ActionCompleted(
                execution_id=self.execution_id,
                success=self.is_successful,
                duration_ms=self.duration_ms,
                response_tokens=response.token_estimate,
                result_summary=self._summarize_result(),
            ))

            return response

        except Exception as e:
            self._fail("response", str(e))
            raise

    def _summarize_result(self) -> Optional[str]:
        """Create a brief summary of the result.

        Returns:
            Short summary string or None
        """
        if self.result is None:
            return None

        if isinstance(self.result, str):
            if len(self.result) <= 50:
                return self.result
            return self.result[:47] + "..."

        if isinstance(self.result, bool):
            return str(self.result)

        if isinstance(self.result, (int, float)):
            return str(self.result)

        if isinstance(self.result, dict):
            return f"dict({len(self.result)} keys)"

        if isinstance(self.result, list):
            return f"list({len(self.result)} items)"

        return type(self.result).__name__


@dataclass
class RetryableActionExecution:
    """Wrapper that adds retry capability to ActionExecution.

    Provides automatic retry logic for transient failures, with
    configurable retry count and delay.
    """
    execution: ActionExecution
    max_retries: int = 3
    retry_delay_ms: float = 100.0
    _attempt: int = field(default=0, repr=False)

    def execute_with_retry(
        self,
        action_handler: Callable[[str, Optional[str], ActionParameters], Any],
        locator: Optional[str],
        timeout: Milliseconds,
    ) -> Any:
        """Execute the action with automatic retry on failure.

        Args:
            action_handler: Function that executes the browser action
            locator: The element locator string (if applicable)
            timeout: Timeout for each attempt

        Returns:
            The action result

        Raises:
            Exception: If all retry attempts fail
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            self._attempt = attempt

            try:
                return self.execution.execute(
                    action_handler=action_handler,
                    locator=locator,
                    timeout=timeout,
                )
            except Exception as e:
                last_error = e

                if attempt < self.max_retries and self._should_retry(e):
                    # Emit retry event
                    self.execution._emit_event(ActionRetrying(
                        execution_id=self.execution.execution_id,
                        attempt_number=attempt + 1,
                        max_attempts=self.max_retries,
                        previous_error=str(e),
                        delay_ms=self.retry_delay_ms,
                    ))

                    # Wait before retry
                    import time as time_module
                    time_module.sleep(self.retry_delay_ms / 1000.0)

                    # Reset error state for retry
                    self.execution.error = None
                    self.execution.completed_at = None
                else:
                    raise

        # Should not reach here, but raise last error if we do
        if last_error:
            raise last_error
        raise RuntimeError("Unexpected retry loop exit")

    def _should_retry(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry.

        Args:
            error: The exception that occurred

        Returns:
            True if the error is retryable
        """
        error_str = str(error).lower()
        retryable_patterns = [
            "timeout",
            "not found",
            "not visible",
            "stale",
            "detached",
            "element is not attached",
        ]
        return any(pattern in error_str for pattern in retryable_patterns)

    @property
    def current_attempt(self) -> int:
        """Get the current attempt number."""
        return self._attempt

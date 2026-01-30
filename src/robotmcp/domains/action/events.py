"""Action Context Domain Events.

Domain events emitted during action execution lifecycle. These events
enable monitoring, debugging, and cross-context communication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Optional

from robotmcp.domains.action.value_objects import ExecutionId, PreValidationResult
from robotmcp.domains.shared.kernel import ElementRef


@dataclass(frozen=True)
class ActionStarted:
    """Emitted when action execution begins.

    This event marks the start of an action execution, enabling
    tracking of action duration and correlation of related events.

    Attributes:
        execution_id: Unique identifier for this execution
        session_id: The session in which the action is executing
        action_type: The type of action being executed
        target_ref: Element reference if this is an element action
        timestamp: When the action started
    """
    execution_id: ExecutionId
    session_id: str
    action_type: str
    target_ref: Optional[ElementRef] = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def event_type(self) -> str:
        """Get the event type identifier."""
        return "action.started"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "execution_id": self.execution_id.value,
            "session_id": self.session_id,
            "action_type": self.action_type,
            "target_ref": self.target_ref.value if self.target_ref else None,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class PreValidationCompleted:
    """Emitted after pre-validation checks complete.

    This event provides visibility into the pre-validation phase,
    including which checks were performed and their outcomes.

    Attributes:
        execution_id: The action execution being validated
        result: The validation result with check details
        duration_ms: Time taken for validation in milliseconds
        timestamp: When validation completed
    """
    execution_id: ExecutionId
    result: PreValidationResult
    duration_ms: float
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def event_type(self) -> str:
        """Get the event type identifier."""
        return "action.pre_validation_completed"

    @property
    def passed(self) -> bool:
        """Check if validation passed."""
        return self.result.passed

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "execution_id": self.execution_id.value,
            "passed": self.result.passed,
            "checks_performed": list(self.result.checks_performed),
            "failed_checks": list(self.result.failed_checks),
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class ActionCompleted:
    """Emitted when action execution completes successfully.

    This event marks the successful completion of an action,
    providing timing and response size metrics.

    Attributes:
        execution_id: The completed action execution
        success: Whether the action succeeded
        duration_ms: Total action duration in milliseconds
        response_tokens: Estimated token count in response
        result_summary: Brief summary of the result
        timestamp: When the action completed
    """
    execution_id: ExecutionId
    success: bool
    duration_ms: float
    response_tokens: int
    result_summary: Optional[str] = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def event_type(self) -> str:
        """Get the event type identifier."""
        return "action.completed"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "execution_id": self.execution_id.value,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "response_tokens": self.response_tokens,
            "result_summary": self.result_summary,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class ActionFailed:
    """Emitted when action execution fails.

    This event provides details about action failures, including
    which stage failed and the error message.

    Attributes:
        execution_id: The failed action execution
        error: Error message describing the failure
        failed_at_stage: Which stage of execution failed
        duration_ms: Time elapsed before failure
        recoverable: Whether the error might be recoverable
        timestamp: When the failure occurred
    """
    execution_id: ExecutionId
    error: str
    failed_at_stage: Literal["pre_validation", "execution", "response"]
    duration_ms: float = 0.0
    recoverable: bool = False
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def event_type(self) -> str:
        """Get the event type identifier."""
        return "action.failed"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "execution_id": self.execution_id.value,
            "error": self.error,
            "failed_at_stage": self.failed_at_stage,
            "duration_ms": self.duration_ms,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class ActionRetrying:
    """Emitted when an action is being retried.

    This event provides visibility into retry attempts, enabling
    monitoring of flaky actions.

    Attributes:
        execution_id: The action execution being retried
        attempt_number: Current retry attempt (1-based)
        max_attempts: Maximum retry attempts configured
        previous_error: Error from the previous attempt
        delay_ms: Delay before this retry in milliseconds
        timestamp: When the retry was initiated
    """
    execution_id: ExecutionId
    attempt_number: int
    max_attempts: int
    previous_error: str
    delay_ms: float = 0.0
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def event_type(self) -> str:
        """Get the event type identifier."""
        return "action.retrying"

    @property
    def is_last_attempt(self) -> bool:
        """Check if this is the final retry attempt."""
        return self.attempt_number >= self.max_attempts

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "execution_id": self.execution_id.value,
            "attempt_number": self.attempt_number,
            "max_attempts": self.max_attempts,
            "previous_error": self.previous_error,
            "delay_ms": self.delay_ms,
            "is_last_attempt": self.is_last_attempt,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class TimeoutExceeded:
    """Emitted when an action times out.

    This event provides details about timeout occurrences, enabling
    analysis of timeout patterns and tuning of timeout values.

    Attributes:
        execution_id: The timed-out action execution
        action_type: Type of action that timed out
        timeout_used_ms: The timeout value that was exceeded
        elapsed_ms: Actual time elapsed before timeout
        category: Timeout category (navigation, action, read)
        timestamp: When the timeout occurred
    """
    execution_id: ExecutionId
    action_type: str
    timeout_used_ms: int
    elapsed_ms: float
    category: Literal["navigation", "action", "read"]
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def event_type(self) -> str:
        """Get the event type identifier."""
        return "action.timeout_exceeded"

    @property
    def overage_ms(self) -> float:
        """Calculate how much the timeout was exceeded by."""
        return max(0.0, self.elapsed_ms - self.timeout_used_ms)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "execution_id": self.execution_id.value,
            "action_type": self.action_type,
            "timeout_used_ms": self.timeout_used_ms,
            "elapsed_ms": self.elapsed_ms,
            "overage_ms": self.overage_ms,
            "category": self.category,
            "timestamp": self.timestamp.isoformat(),
        }

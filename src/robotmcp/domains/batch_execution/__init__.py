"""Batch Execution Bounded Context (ADR-011).

Multi-step keyword execution with inter-step references, timeout
management, tiered recovery, and resume-from-failure capability.
Reduces MCP round-trips from N to 1 for N-step test sequences.
"""
from .value_objects import (
    BatchId,
    BatchStatus,
    BatchTimeout,
    OnFailurePolicy,
    RecoveryAttemptLimit,
    StepReference,
    StepStatus,
    StepTimeout,
)
from .entities import (
    BatchStep,
    FailureDetail,
    RecoveryAttempt,
    StepResult,
)
from .aggregates import (
    BatchExecution,
    BatchState,
)
from .events import (
    BatchCompleted,
    BatchFailed,
    BatchResumed,
    BatchStarted,
    BatchTimedOut,
    StepExecuted,
    StepFailed,
    StepRecovered,
)

__all__ = [
    # Value objects
    "BatchId",
    "BatchStatus",
    "BatchTimeout",
    "OnFailurePolicy",
    "RecoveryAttemptLimit",
    "StepReference",
    "StepStatus",
    "StepTimeout",
    # Entities
    "BatchStep",
    "FailureDetail",
    "RecoveryAttempt",
    "StepResult",
    # Aggregates
    "BatchExecution",
    "BatchState",
    # Events
    "BatchCompleted",
    "BatchFailed",
    "BatchResumed",
    "BatchStarted",
    "BatchTimedOut",
    "StepExecuted",
    "StepFailed",
    "StepRecovered",
]

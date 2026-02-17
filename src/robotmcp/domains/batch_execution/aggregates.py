"""Batch Execution Domain Aggregate Roots.

Two aggregate roots:

- **BatchExecution**: The primary aggregate for executing a batch of
  keyword steps with inter-step references, timeout tracking, and
  result recording.

- **BatchState**: A snapshot of a failed BatchExecution that enables
  resume_batch (Phase 2). Created from a failed BatchExecution and
  stores enough context to continue from the failure point.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

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
from .entities import BatchStep, FailureDetail, RecoveryAttempt, StepResult


@dataclass
class BatchExecution:
    """Aggregate root for batch keyword execution.

    Owns all steps, results, and failure state. Enforces invariants:
    - At least one step is required
    - Forward references in step args are rejected
    - Timeout is tracked monotonically

    Lifecycle:
        1. ``create()`` — factory from MCP tool input
        2. ``start_clock()`` — begin timeout tracking
        3. For each step: ``resolve_args()`` then one of:
           ``record_success()``, ``record_recovery()``, ``record_failure()``
        4. ``finalize()`` — compute overall status

    Concurrency:
        Not thread-safe. The execution lock in KeywordExecutor ensures
        single-threaded access to the aggregate during batch execution.
    """
    __test__ = False  # Suppress pytest collection

    batch_id: BatchId
    session_id: str
    steps: List[BatchStep]
    on_failure: OnFailurePolicy = OnFailurePolicy.RECOVER
    max_recovery_attempts: RecoveryAttemptLimit = field(
        default_factory=RecoveryAttemptLimit.default
    )
    timeout: BatchTimeout = field(default_factory=BatchTimeout.default)
    results: List[StepResult] = field(default_factory=list)
    results_map: Dict[int, Any] = field(default_factory=dict)
    failure: Optional[FailureDetail] = None
    status: Optional[BatchStatus] = None
    _start_time: Optional[float] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("BatchExecution requires at least 1 step")

    @classmethod
    def create(
        cls,
        session_id: str,
        steps_data: List[Dict[str, Any]],
        on_failure: str = "recover",
        max_recovery_attempts: int = 2,
        timeout_ms: int = 120000,
    ) -> BatchExecution:
        """Factory from MCP tool input.

        Parses raw step dicts into BatchStep entities and validates
        all value objects.

        Args:
            session_id: The MCP session to execute within
            steps_data: List of step dicts with ``keyword``, ``args``,
                optional ``label`` and ``timeout``
            on_failure: Failure policy name (``stop``, ``retry``, ``recover``)
            max_recovery_attempts: Max recovery attempts per step (1-10)
            timeout_ms: Total batch timeout in milliseconds (1000-600000)

        Returns:
            A fully constructed BatchExecution ready for execution

        Raises:
            ValueError: If steps_data is empty or contains invalid data
        """
        if not steps_data:
            raise ValueError("At least one step is required")

        steps: List[BatchStep] = []
        for i, s in enumerate(steps_data):
            step_timeout = StepTimeout(s["timeout"]) if s.get("timeout") else None
            steps.append(BatchStep(
                index=i,
                keyword=s["keyword"],
                args=list(s.get("args", [])),
                label=s.get("label"),
                timeout=step_timeout,
                assign_to=s.get("assign_to"),
            ))

        policy = OnFailurePolicy(on_failure)
        limit = RecoveryAttemptLimit(max_recovery_attempts)
        timeout = BatchTimeout(timeout_ms)

        return cls(
            batch_id=BatchId.generate(),
            session_id=session_id,
            steps=steps,
            on_failure=policy,
            max_recovery_attempts=limit,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Clock management
    # ------------------------------------------------------------------

    def start_clock(self) -> None:
        """Begin timeout tracking using monotonic clock."""
        self._start_time = time.monotonic()

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds elapsed since start_clock() was called."""
        if self._start_time is None:
            return 0.0
        return (time.monotonic() - self._start_time) * 1000

    def is_timed_out(self) -> bool:
        """Check if the batch has exceeded its timeout budget."""
        return self.elapsed_ms >= self.timeout.value_ms

    # ------------------------------------------------------------------
    # Argument resolution
    # ------------------------------------------------------------------

    def resolve_args(self, step: BatchStep) -> List[str]:
        """Resolve ``${STEP_N}`` references in step args using results_map.

        Supports both 0-based and 1-based step indexing.  LLMs naturally use
        1-based references (``${STEP_3}`` = "result of the 3rd step" = index 2).
        When a reference would be a forward/self-reference under 0-based
        indexing but valid as 1-based, it is automatically interpreted as
        1-based (N-1).

        Args:
            step: The step whose args should be resolved

        Returns:
            List of resolved argument strings

        Raises:
            ValueError: If a forward reference or missing result is detected
        """
        resolved: List[str] = []
        for arg in step.args:
            refs = StepReference.find_all(str(arg))
            result = str(arg)
            for ref in refs:
                idx = ref.index
                if idx >= step.index and idx > 0 and (idx - 1) < step.index:
                    # 1-based interpretation: ${STEP_3} means index 2
                    idx = idx - 1
                if idx >= step.index:
                    raise ValueError(
                        f"Forward reference {ref.raw} in step {step.index}"
                    )
                if idx not in self.results_map:
                    raise ValueError(
                        f"Reference {ref.raw} not available "
                        f"(step {idx} not completed)"
                    )
                result = result.replace(
                    ref.raw, str(self.results_map[idx])
                )
            resolved.append(result)
        return resolved

    # ------------------------------------------------------------------
    # Result recording
    # ------------------------------------------------------------------

    def record_success(
        self,
        step: BatchStep,
        resolved_args: List[str],
        return_value: Any,
        time_ms: int,
    ) -> None:
        """Record a step that succeeded on first attempt.

        Args:
            step: The step that was executed
            resolved_args: Arguments after reference resolution
            return_value: Value returned by the keyword
            time_ms: Execution time in milliseconds
        """
        self.results.append(StepResult(
            index=step.index,
            keyword=step.keyword,
            args_resolved=resolved_args,
            status=StepStatus.PASS,
            return_value=return_value,
            time_ms=time_ms,
            label=step.label,
        ))
        self.results_map[step.index] = return_value

    def record_recovery(
        self,
        step: BatchStep,
        resolved_args: List[str],
        return_value: Any,
        time_ms: int,
        original_error: str,
        attempts: List[RecoveryAttempt],
    ) -> None:
        """Record a step that failed initially but was recovered.

        Args:
            step: The step that was executed
            resolved_args: Arguments after reference resolution
            return_value: Value returned after recovery
            time_ms: Total execution time including recovery
            original_error: The initial error message
            attempts: Log of recovery attempts made
        """
        self.results.append(StepResult(
            index=step.index,
            keyword=step.keyword,
            args_resolved=resolved_args,
            status=StepStatus.RECOVERED,
            return_value=return_value,
            time_ms=time_ms,
            error=original_error,
            label=step.label,
        ))
        self.results_map[step.index] = return_value

    def record_failure(
        self,
        step: BatchStep,
        resolved_args: List[str],
        error: str,
        time_ms: int,
        failure_detail: FailureDetail,
    ) -> None:
        """Record a step that failed without recovery.

        Sets the batch status to FAIL and captures the failure detail.

        Args:
            step: The step that failed
            resolved_args: Arguments after reference resolution
            error: The error message
            time_ms: Execution time in milliseconds
            failure_detail: Diagnostic context for the failure
        """
        self.results.append(StepResult(
            index=step.index,
            keyword=step.keyword,
            args_resolved=resolved_args,
            status=StepStatus.FAIL,
            error=error,
            time_ms=time_ms,
            label=step.label,
        ))
        self.failure = failure_detail
        self.status = BatchStatus.FAIL

    def record_timeout(self) -> None:
        """Mark the batch as timed out."""
        self.status = BatchStatus.TIMEOUT

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Compute the final batch status from individual step results.

        Must be called after all steps have been executed (or after
        failure/timeout). Idempotent: does nothing if status is already set.

        Status rules:
        - FAIL or TIMEOUT: already set by record_failure/record_timeout
        - RECOVERED: at least one step has RECOVERED status
        - PASS: all steps passed on first attempt
        """
        if self.status is not None:
            return  # Already set (FAIL or TIMEOUT)
        if any(r.status == StepStatus.RECOVERED for r in self.results):
            self.status = BatchStatus.RECOVERED
        else:
            self.status = BatchStatus.PASS

    # ------------------------------------------------------------------
    # Response serialization
    # ------------------------------------------------------------------

    def to_response_dict(self) -> Dict[str, Any]:
        """Serialize the batch execution result for MCP tool output.

        Returns:
            Dict suitable for JSON serialization in MCP responses
        """
        steps_executed = len(self.results)
        steps_total = len(self.steps)
        summary = self._build_summary(steps_executed, steps_total)

        d: Dict[str, Any] = {
            "status": self.status.value if self.status else "UNKNOWN",
            "summary": summary,
            "total_time_ms": int(self.elapsed_ms),
            "steps_executed": steps_executed,
            "steps_total": steps_total,
            "steps": [r.to_dict() for r in self.results],
        }
        if self.failure:
            d["failure"] = self.failure.to_dict()
            d["batch_id"] = self.batch_id.value  # For resume_batch
        return d

    def _build_summary(self, executed: int, total: int) -> str:
        """Build a human-readable summary string."""
        if self.status == BatchStatus.PASS:
            return f"{executed}/{total} passed"
        elif self.status == BatchStatus.RECOVERED:
            recovered_count = sum(
                1 for r in self.results
                if r.status == StepStatus.RECOVERED
            )
            return f"{executed}/{total} passed, {recovered_count} recovered"
        elif self.status == BatchStatus.FAIL:
            failed_idx = self.failure.step_index if self.failure else -1
            return f"{executed}/{total} executed, step {failed_idx} failed"
        elif self.status == BatchStatus.TIMEOUT:
            return (
                f"Timeout after {executed}/{total} steps "
                f"({int(self.elapsed_ms)}ms)"
            )
        return f"{executed}/{total} executed"


@dataclass
class BatchState:
    """Snapshot of a failed BatchExecution for resume_batch (Phase 2).

    Created from a failed BatchExecution, this captures enough context
    to continue execution from the failure point. Includes all prior
    results and the results_map so that ``${STEP_N}`` references from
    earlier steps remain available.

    Attributes:
        batch_id: The original batch identifier
        session_id: The MCP session this batch ran in
        original_steps: All steps from the original batch
        results: Results from steps that completed before failure
        results_map: Return values indexed by step index
        failed_at_index: Index of the step that failed
        on_failure: The failure policy from the original batch
        max_recovery_attempts: Recovery limit from the original batch
        timeout_ms: Original timeout budget in milliseconds
        elapsed_ms: Time already consumed before failure
        created_at: When this snapshot was created (for TTL expiry)
    """
    __test__ = False  # Suppress pytest collection

    batch_id: BatchId
    session_id: str
    original_steps: List[BatchStep]
    results: List[StepResult]
    results_map: Dict[int, Any]
    failed_at_index: int
    on_failure: OnFailurePolicy
    max_recovery_attempts: RecoveryAttemptLimit
    timeout_ms: int
    elapsed_ms: float
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_execution(cls, batch: BatchExecution) -> BatchState:
        """Create a BatchState snapshot from a failed BatchExecution.

        Args:
            batch: The failed batch execution to snapshot

        Returns:
            A BatchState capturing the execution state at failure

        Raises:
            ValueError: If the batch has not failed (no failure detail)
        """
        if batch.failure is None:
            raise ValueError(
                "Cannot create BatchState from non-failed execution"
            )
        return cls(
            batch_id=batch.batch_id,
            session_id=batch.session_id,
            original_steps=list(batch.steps),
            results=list(batch.results),
            results_map=dict(batch.results_map),
            failed_at_index=batch.failure.step_index,
            on_failure=batch.on_failure,
            max_recovery_attempts=batch.max_recovery_attempts,
            timeout_ms=batch.timeout.value_ms,
            elapsed_ms=batch.elapsed_ms,
        )

    @property
    def remaining_timeout_ms(self) -> float:
        """Time budget remaining after subtracting elapsed time."""
        return max(0, self.timeout_ms - self.elapsed_ms)

    @property
    def remaining_steps(self) -> List[BatchStep]:
        """Steps that were not yet executed (after the failure point)."""
        return [
            s for s in self.original_steps
            if s.index > self.failed_at_index
        ]

    @property
    def failed_step(self) -> Optional[BatchStep]:
        """The step that caused the failure, or None if not found."""
        for s in self.original_steps:
            if s.index == self.failed_at_index:
                return s
        return None

    def is_expired(self, ttl_seconds: float = 300.0) -> bool:
        """Check if this snapshot has exceeded its time-to-live.

        Args:
            ttl_seconds: Maximum age in seconds (default: 5 minutes)

        Returns:
            True if the snapshot is too old to resume
        """
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > ttl_seconds

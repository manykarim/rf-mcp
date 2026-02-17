"""Batch Execution Domain Services.

Contains the BatchRunner (sequential execution engine),
StepVariableResolver, BatchStateManager, and Protocol definitions.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from .value_objects import (
    BatchStatus, StepStatus, OnFailurePolicy, StepReference,
)
from .entities import BatchStep, StepResult, RecoveryAttempt, FailureDetail
from .aggregates import BatchExecution, BatchState

logger = logging.getLogger(__name__)


# ── Protocol Definitions ──────────────────────────────────────────────

@runtime_checkable
class KeywordExecutorProtocol(Protocol):
    """Protocol for keyword execution (anti-corruption layer)."""
    async def execute_keyword(
        self, session_id: str, keyword: str, args: List[str],
        timeout: Optional[str] = None,
        assign_to: Optional[str] = None,
    ) -> Dict[str, Any]: ...


@runtime_checkable
class RecoveryServiceProtocol(Protocol):
    """Protocol for recovery (anti-corruption layer to recovery domain)."""
    async def attempt_recovery(
        self, session_id: str, keyword: str, args: List[str],
        error_message: str, attempt_number: int,
    ) -> Optional[RecoveryAttempt]: ...


@runtime_checkable
class EvidenceCollectorProtocol(Protocol):
    """Protocol for evidence collection on failure."""
    async def collect_evidence(self, session_id: str) -> Dict[str, Any]: ...


EventPublisher = Optional[Callable[..., None]]


# ── StepVariableResolver ──────────────────────────────────────────────

class StepVariableResolver:
    """Stateless service for ${STEP_N} variable substitution."""

    def resolve(self, args: List[str], results_map: Dict[int, Any],
                step_index: int) -> List[str]:
        """Resolve ${STEP_N} references in args.

        Supports both 0-based and 1-based step indexing.
        Raises ValueError for forward references or missing results.
        """
        resolved = []
        for arg in args:
            refs = StepReference.find_all(str(arg))
            result = str(arg)
            for ref in refs:
                idx = ref.index
                if idx >= step_index and idx > 0 and (idx - 1) < step_index:
                    idx = idx - 1
                if idx >= step_index:
                    raise ValueError(
                        f"Forward reference {ref.raw} in step {step_index}"
                    )
                if idx not in results_map:
                    raise ValueError(
                        f"Reference {ref.raw} not available "
                        f"(step {idx} not completed)"
                    )
                result = result.replace(ref.raw, str(results_map[idx]))
            resolved.append(result)
        return resolved

    def validate_references(self, steps: List[BatchStep]) -> List[str]:
        """Validate all step references at batch creation time.

        Supports both 0-based and 1-based step indexing.
        Returns list of error messages (empty = valid).
        """
        errors: List[str] = []
        for step in steps:
            for arg in step.args:
                refs = StepReference.find_all(str(arg))
                for ref in refs:
                    idx = ref.index
                    if idx >= step.index and idx > 0 and (idx - 1) < step.index:
                        idx = idx - 1
                    if idx >= step.index:
                        errors.append(
                            f"Step {step.index}: forward reference {ref.raw}"
                        )
        return errors


# ── BatchRunner ───────────────────────────────────────────────────────

@dataclass
class BatchRunner:
    """Sequential execution engine for batch keyword execution.

    Drives the execution loop, delegates to Protocol-based dependencies.
    """
    keyword_executor: KeywordExecutorProtocol
    recovery_service: Optional[RecoveryServiceProtocol] = None
    evidence_collector: Optional[EvidenceCollectorProtocol] = None
    variable_resolver: StepVariableResolver = field(
        default_factory=StepVariableResolver
    )

    async def execute(self, batch: BatchExecution) -> BatchExecution:
        """Execute all steps in the batch sequentially.

        Returns the same BatchExecution with results populated.
        """
        batch.start_clock()

        for step in batch.steps:
            # 1. Check global timeout
            if batch.is_timed_out():
                batch.record_timeout()
                break

            # 2. Resolve ${STEP_N} variables
            try:
                resolved_args = batch.resolve_args(step)
            except ValueError as e:
                # Variable resolution error — abort immediately
                failure = FailureDetail(
                    step_index=step.index,
                    error=f"Variable resolution error: {e}",
                )
                batch.record_failure(
                    step, list(step.args), str(e), 0, failure
                )
                break

            # 3. Execute keyword
            step_start = time.monotonic()
            try:
                result = await self.keyword_executor.execute_keyword(
                    batch.session_id, step.keyword, resolved_args,
                    timeout=step.timeout.rf_format if step.timeout else None,
                    assign_to=step.assign_to,
                )
                time_ms = int((time.monotonic() - step_start) * 1000)

                if result.get("success", False):
                    return_value = result.get("result") or result.get("return_value")
                    batch.record_success(step, resolved_args, return_value, time_ms)
                else:
                    # Keyword executed but failed (ExecutionStatus)
                    error_msg = result.get("error", "Unknown error")
                    time_ms = int((time.monotonic() - step_start) * 1000)
                    await self._handle_failure(
                        batch, step, resolved_args, error_msg,
                        time_ms, step_start,
                    )
                    if batch.status == BatchStatus.FAIL:
                        break

            except Exception as e:
                time_ms = int((time.monotonic() - step_start) * 1000)
                await self._handle_failure(
                    batch, step, resolved_args, str(e),
                    time_ms, step_start,
                )
                if batch.status == BatchStatus.FAIL:
                    break

        # Finalize
        batch.finalize()
        return batch

    async def _handle_failure(
        self, batch: BatchExecution, step: BatchStep,
        resolved_args: List[str], error_msg: str,
        total_time_ms: int, step_start: float,
    ) -> None:
        """Handle step failure with recovery based on on_failure policy."""
        if batch.on_failure == OnFailurePolicy.STOP:
            evidence = await self._collect_evidence(batch.session_id)
            failure = FailureDetail(
                step_index=step.index, error=error_msg,
                screenshot_base64=evidence.get("screenshot_base64"),
                page_source_snippet=evidence.get("page_source_snippet"),
                current_url=evidence.get("current_url"),
                page_title=evidence.get("page_title"),
            )
            batch.record_failure(step, resolved_args, error_msg, total_time_ms, failure)
            return

        # RETRY or RECOVER — attempt recovery loop
        recovery_log: List[RecoveryAttempt] = []
        recovered = False
        last_error = error_msg

        for attempt in range(1, batch.max_recovery_attempts.value + 1):
            # Check global timeout before recovery
            if batch.is_timed_out():
                batch.record_timeout()
                return

            # Attempt recovery action
            recovery_attempt = None
            if batch.on_failure == OnFailurePolicy.RECOVER and self.recovery_service:
                try:
                    recovery_attempt = await self.recovery_service.attempt_recovery(
                        batch.session_id, step.keyword, resolved_args,
                        last_error, attempt,
                    )
                    if recovery_attempt:
                        recovery_log.append(recovery_attempt)
                except Exception as e:
                    logger.warning("Recovery attempt %d failed: %s", attempt, e)

            # Retry the original keyword
            if batch.is_timed_out():
                batch.record_timeout()
                return

            try:
                result = await self.keyword_executor.execute_keyword(
                    batch.session_id, step.keyword, resolved_args,
                    timeout=step.timeout.rf_format if step.timeout else None,
                    assign_to=step.assign_to,
                )
                time_ms = int((time.monotonic() - step_start) * 1000)

                if result.get("success", False):
                    return_value = result.get("result") or result.get("return_value")
                    batch.record_recovery(
                        step, resolved_args, return_value, time_ms,
                        error_msg, recovery_log,
                    )
                    recovered = True
                    break
                else:
                    last_error = result.get("error", "Unknown error")
            except Exception as e:
                last_error = str(e)

        if not recovered and batch.status is None:
            # All recovery attempts exhausted
            time_ms = int((time.monotonic() - step_start) * 1000)
            evidence = await self._collect_evidence(batch.session_id)
            failure = FailureDetail(
                step_index=step.index, error=last_error,
                screenshot_base64=evidence.get("screenshot_base64"),
                page_source_snippet=evidence.get("page_source_snippet"),
                current_url=evidence.get("current_url"),
                page_title=evidence.get("page_title"),
                recovery_log=recovery_log,
            )
            batch.record_failure(step, resolved_args, last_error, time_ms, failure)

    async def _collect_evidence(self, session_id: str) -> Dict[str, Any]:
        """Collect evidence on failure."""
        if self.evidence_collector:
            try:
                return await self.evidence_collector.collect_evidence(session_id)
            except Exception as e:
                logger.warning("Evidence collection failed: %s", e)
        return {}


# ── BatchStateManager ─────────────────────────────────────────────────

@dataclass
class BatchStateManager:
    """In-memory TTL-based state storage for resume_batch (Phase 2)."""
    _states: Dict[str, BatchState] = field(default_factory=dict)
    ttl_seconds: float = 300.0  # 5 minutes
    max_states: int = 10

    def store(self, state: BatchState) -> None:
        """Store a batch state, evicting oldest if at capacity."""
        self._cleanup_expired()
        if len(self._states) >= self.max_states:
            self._evict_oldest()
        self._states[state.batch_id.value] = state

    def get(self, batch_id: str) -> Optional[BatchState]:
        """Retrieve a batch state, returning None if expired or missing."""
        state = self._states.get(batch_id)
        if state is None:
            return None
        if state.is_expired(self.ttl_seconds):
            del self._states[batch_id]
            return None
        return state

    def remove(self, batch_id: str) -> bool:
        """Remove a batch state. Returns True if found."""
        return self._states.pop(batch_id, None) is not None

    @property
    def count(self) -> int:
        return len(self._states)

    def _cleanup_expired(self) -> None:
        expired = [
            bid for bid, state in self._states.items()
            if state.is_expired(self.ttl_seconds)
        ]
        for bid in expired:
            del self._states[bid]

    def _evict_oldest(self) -> None:
        if not self._states:
            return
        oldest_id = min(
            self._states, key=lambda bid: self._states[bid].created_at
        )
        del self._states[oldest_id]

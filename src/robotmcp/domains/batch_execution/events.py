"""Batch Execution Domain Events.

Events emitted during batch execution for observability,
analytics, and learning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class BatchStarted:
    """Emitted when batch execution begins."""
    batch_id: str
    session_id: str
    step_count: int
    on_failure: str
    timeout_ms: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "batch_started",
            "batch_id": self.batch_id,
            "session_id": self.session_id,
            "step_count": self.step_count,
            "on_failure": self.on_failure,
            "timeout_ms": self.timeout_ms,
        }


@dataclass(frozen=True)
class StepExecuted:
    """Emitted when a step completes successfully (PASS)."""
    batch_id: str
    step_index: int
    keyword: str
    time_ms: int
    return_value: Any = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "event_type": "step_executed",
            "batch_id": self.batch_id,
            "step_index": self.step_index,
            "keyword": self.keyword,
            "time_ms": self.time_ms,
        }
        if self.return_value is not None:
            d["return_value"] = self.return_value
        return d


@dataclass(frozen=True)
class StepFailed:
    """Emitted when a step raises an exception."""
    batch_id: str
    step_index: int
    keyword: str
    error: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "step_failed",
            "batch_id": self.batch_id,
            "step_index": self.step_index,
            "keyword": self.keyword,
            "error": self.error,
        }


@dataclass(frozen=True)
class StepRecovered:
    """Emitted when a step succeeds after recovery."""
    batch_id: str
    step_index: int
    keyword: str
    strategy: str
    attempt_number: int
    time_ms: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "step_recovered",
            "batch_id": self.batch_id,
            "step_index": self.step_index,
            "keyword": self.keyword,
            "strategy": self.strategy,
            "attempt_number": self.attempt_number,
            "time_ms": self.time_ms,
        }


@dataclass(frozen=True)
class BatchCompleted:
    """Emitted when batch finishes PASS or RECOVERED."""
    batch_id: str
    status: str
    steps_executed: int
    steps_total: int
    total_time_ms: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "batch_completed",
            "batch_id": self.batch_id,
            "status": self.status,
            "steps_executed": self.steps_executed,
            "steps_total": self.steps_total,
            "total_time_ms": self.total_time_ms,
        }


@dataclass(frozen=True)
class BatchFailed:
    """Emitted when batch terminates on failure."""
    batch_id: str
    failed_step_index: int
    error: str
    steps_completed: int
    steps_total: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "batch_failed",
            "batch_id": self.batch_id,
            "failed_step_index": self.failed_step_index,
            "error": self.error,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
        }


@dataclass(frozen=True)
class BatchTimedOut:
    """Emitted when global timeout is exceeded."""
    batch_id: str
    steps_completed: int
    steps_total: int
    elapsed_ms: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "batch_timed_out",
            "batch_id": self.batch_id,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass(frozen=True)
class BatchResumed:
    """Emitted when resume_batch is called (Phase 2)."""
    batch_id: str
    session_id: str
    resumed_from_index: int
    fix_steps_count: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "batch_resumed",
            "batch_id": self.batch_id,
            "session_id": self.session_id,
            "resumed_from_index": self.resumed_from_index,
            "fix_steps_count": self.fix_steps_count,
        }

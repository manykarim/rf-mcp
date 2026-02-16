"""Recovery Domain Events.

Events emitted during recovery lifecycle for observability,
analytics, and learning. All events are frozen dataclasses
following the same pattern as intent domain events.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ErrorClassified:
    """Emitted when an error message is classified.

    Consumers:
    - Learning system (track error pattern frequency)
    - Analytics (error classification distribution)
    """
    error_message: str
    classification: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "error_classified",
            "error_message": self.error_message,
            "classification": self.classification,
        }


@dataclass(frozen=True)
class RecoveryStrategySelected:
    """Emitted when a recovery strategy is chosen.

    Consumers:
    - Learning system (track strategy selection patterns)
    - Analytics (strategy usage frequency per error type)
    """
    plan_id: str
    classification: str
    strategy: str
    tier: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "recovery_strategy_selected",
            "plan_id": self.plan_id,
            "classification": self.classification,
            "strategy": self.strategy,
            "tier": self.tier,
        }


@dataclass(frozen=True)
class RecoveryAttempted:
    """Emitted when recovery actions are executed.

    Consumers:
    - Analytics (recovery attempt frequency)
    - Monitoring (real-time recovery tracking)
    """
    plan_id: str
    strategy: str
    tier: int
    actions_count: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "recovery_attempted",
            "plan_id": self.plan_id,
            "strategy": self.strategy,
            "tier": self.tier,
            "actions_count": self.actions_count,
        }


@dataclass(frozen=True)
class RecoverySucceeded:
    """Emitted when retry succeeds after recovery.

    Consumers:
    - Learning system (reinforce successful strategies)
    - Analytics (recovery success rate by strategy)
    """
    plan_id: str
    strategy: str
    tier: int
    total_time_ms: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "recovery_succeeded",
            "plan_id": self.plan_id,
            "strategy": self.strategy,
            "tier": self.tier,
            "total_time_ms": self.total_time_ms,
        }


@dataclass(frozen=True)
class RecoveryFailed:
    """Emitted when all recovery attempts are exhausted.

    Consumers:
    - Learning system (identify failing patterns)
    - Analytics (recovery failure rate by classification)
    """
    plan_id: str
    classification: str
    strategies_tried: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "recovery_failed",
            "plan_id": self.plan_id,
            "classification": self.classification,
            "strategies_tried": list(self.strategies_tried),
        }


@dataclass(frozen=True)
class EvidenceCollected:
    """Emitted when page state is captured as evidence.

    Consumers:
    - Diagnostics (attach evidence to failure reports)
    - Learning (correlate evidence with recovery outcomes)
    """
    plan_id: str
    has_screenshot: bool
    has_page_source: bool
    current_url: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "event_type": "evidence_collected",
            "plan_id": self.plan_id,
            "has_screenshot": self.has_screenshot,
            "has_page_source": self.has_page_source,
        }
        if self.current_url:
            d["current_url"] = self.current_url
        return d

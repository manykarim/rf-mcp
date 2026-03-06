"""Token Accounting Domain Events (ADR-017)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass(frozen=True)
class TokenBudgetExceeded:
    """Raised when token usage exceeds the configured budget."""

    tool_name: str
    token_count: int
    budget: int
    overage: int
    timestamp: datetime = field(default_factory=datetime.now)

    __test__ = False  # suppress pytest collection warning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "TokenBudgetExceeded",
            "tool": self.tool_name,
            "count": self.token_count,
            "budget": self.budget,
            "overage": self.overage,
            "ts": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class TokenRegressionDetected:
    """Raised when a profile's token count regresses beyond threshold."""

    profile: str
    baseline_tokens: int
    current_tokens: int
    delta: int
    threshold: int
    backend_used: str
    timestamp: datetime = field(default_factory=datetime.now)

    __test__ = False  # suppress pytest collection warning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "TokenRegressionDetected",
            "profile": self.profile,
            "baseline": self.baseline_tokens,
            "current": self.current_tokens,
            "delta": self.delta,
            "threshold": self.threshold,
            "backend": self.backend_used,
            "ts": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class TokenMeasurementRecorded:
    """Raised when a new token measurement is recorded."""

    tool_name: str
    measurement_type: str
    count: int
    backend: str
    timestamp: datetime = field(default_factory=datetime.now)

    __test__ = False  # suppress pytest collection warning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "TokenMeasurementRecorded",
            "tool": self.tool_name,
            "type": self.measurement_type,
            "count": self.count,
            "backend": self.backend,
            "ts": self.timestamp.isoformat(),
        }

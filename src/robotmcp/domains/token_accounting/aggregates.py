"""Token Accounting Domain Aggregates (ADR-017)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from .entities import TokenMeasurement
from .value_objects import TokenizerBackend


@dataclass
class TokenAccountant:
    """Aggregate root for token accounting."""

    backend: TokenizerBackend
    measurements: Deque[TokenMeasurement] = field(
        default_factory=lambda: deque(maxlen=1000)
    )

    MAX_MEASUREMENTS = 1000

    __test__ = False  # suppress pytest collection warning

    @classmethod
    def create(
        cls, backend: Optional[TokenizerBackend] = None
    ) -> "TokenAccountant":
        """Factory with optional env-based backend detection."""
        if backend is None:
            backend = TokenizerBackend.from_env()
        return cls(backend=backend)

    def record(self, measurement: TokenMeasurement) -> None:
        """Record a measurement (ring buffer, oldest evicted at maxlen)."""
        self.measurements.append(measurement)

    def get_measurements(
        self, tool_name: Optional[str] = None
    ) -> List[TokenMeasurement]:
        """Retrieve measurements, optionally filtered by tool."""
        if tool_name:
            return [
                m for m in self.measurements if m.tool_name == tool_name
            ]
        return list(self.measurements)

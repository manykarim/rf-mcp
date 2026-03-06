"""Token Accounting Domain Entities (ADR-017)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from .value_objects import TokenizerBackend


@dataclass
class TokenMeasurement:
    """A single token measurement record."""

    id: str
    tool_name: str
    profile: str
    backend: TokenizerBackend
    measurement_type: str  # "schema", "description", "response", "total"
    count: int
    timestamp: datetime = field(default_factory=datetime.now)

    VALID_TYPES = frozenset({"schema", "description", "response", "total"})

    __test__ = False  # suppress pytest collection warning

    def __post_init__(self):
        if self.measurement_type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid measurement_type: {self.measurement_type}"
            )
        if self.count < 0:
            raise ValueError(f"count cannot be negative: {self.count}")

    @classmethod
    def create(
        cls,
        tool_name: str,
        profile: str,
        backend: TokenizerBackend,
        measurement_type: str,
        count: int,
    ) -> "TokenMeasurement":
        """Factory method for creating measurements."""
        return cls(
            id=f"tm_{uuid.uuid4().hex[:8]}",
            tool_name=tool_name,
            profile=profile,
            backend=backend,
            measurement_type=measurement_type,
            count=count,
        )

    def to_dict(self):
        return {
            "id": self.id,
            "tool": self.tool_name,
            "profile": self.profile,
            "backend": self.backend.value,
            "type": self.measurement_type,
            "count": self.count,
            "ts": self.timestamp.isoformat(),
        }

"""Delta State Retrieval Events (ADR-018).

Domain events emitted during delta state operations.
All events are frozen dataclasses with to_dict() for serialization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass(frozen=True)
class StateVersionCreated:
    """Emitted when a new state version is stored."""
    session_id: str
    version_number: int
    content_hash: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "StateVersionCreated",
            "session": self.session_id,
            "version": self.version_number,
            "hash": self.content_hash[:8],
            "ts": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class DeltaComputed:
    """Emitted when a delta between two versions is computed."""
    session_id: str
    from_version: int
    to_version: int
    changed_count: int
    unchanged_count: int
    saved_tokens: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "DeltaComputed",
            "session": self.session_id,
            "from": self.from_version,
            "to": self.to_version,
            "changed": self.changed_count,
            "unchanged": self.unchanged_count,
            "saved_tokens": self.saved_tokens,
            "ts": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class StateVersionExpired:
    """Emitted when a state version is evicted or expires."""
    session_id: str
    version_number: int
    reason: str  # "ttl" or "lru"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "StateVersionExpired",
            "session": self.session_id,
            "version": self.version_number,
            "reason": self.reason,
            "ts": self.timestamp.isoformat(),
        }

"""Memory Domain — Entities.

Mutable types with identity, following ADR-001 conventions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .value_objects import MemoryEntry, MemoryType


# ---------------------------------------------------------------------------
# MemoryRecord
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    """A persisted memory with identity and access tracking.

    Identity: record_id (UUID)
    Lifecycle: created → accessed (many times) → pruned

    Invariants:
    - record_id is immutable after creation
    - access_count >= 0
    - accessed_at >= created_at
    """

    __test__ = False  # Suppress pytest collection

    record_id: str
    entry: MemoryEntry
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    session_id: Optional[str] = None

    @classmethod
    def create(
        cls,
        entry: MemoryEntry,
        session_id: Optional[str] = None,
    ) -> MemoryRecord:
        now = datetime.now()
        return cls(
            record_id=str(uuid.uuid4()),
            entry=entry,
            created_at=now,
            accessed_at=now,
            access_count=0,
            session_id=session_id,
        )

    def record_access(self) -> None:
        self.access_count += 1
        self.accessed_at = datetime.now()

    @property
    def age_days(self) -> float:
        return (datetime.now() - self.created_at).total_seconds() / 86400.0

    @property
    def days_since_access(self) -> float:
        return (datetime.now() - self.accessed_at).total_seconds() / 86400.0

    def is_stale(self, max_age_days: float) -> bool:
        return self.days_since_access > max_age_days

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "record_id": self.record_id,
            "content": self.entry.content,
            "memory_type": self.entry.memory_type.value,
            "metadata": dict(self.entry.metadata) if self.entry.metadata else {},
            "tags": list(self.entry.tags),
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
        }
        if self.session_id:
            result["session_id"] = self.session_id
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryRecord:
        entry = MemoryEntry(
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            metadata=data.get("metadata", {}),
            tags=tuple(data.get("tags", ())),
        )
        return cls(
            record_id=data["record_id"],
            entry=entry,
            created_at=datetime.fromisoformat(data["created_at"]),
            accessed_at=datetime.fromisoformat(data["accessed_at"]),
            access_count=data.get("access_count", 0),
            session_id=data.get("session_id"),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryRecord):
            return NotImplemented
        return self.record_id == other.record_id

    def __hash__(self) -> int:
        return hash(self.record_id)


# ---------------------------------------------------------------------------
# MemoryCollection
# ---------------------------------------------------------------------------

@dataclass
class MemoryCollection:
    """Tracks metadata about a vector collection.

    Identity: collection_id (derived from memory_type)
    """

    __test__ = False  # Suppress pytest collection

    collection_id: str
    memory_type: MemoryType
    dimension: int = 256
    record_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_updated_at: Optional[datetime] = None

    @classmethod
    def for_type(
        cls, memory_type: MemoryType, dimension: int = 256
    ) -> MemoryCollection:
        return cls(
            collection_id=memory_type.collection_name,
            memory_type=memory_type,
            dimension=dimension,
        )

    def increment_count(self, n: int = 1) -> None:
        self.record_count += n
        self.last_updated_at = datetime.now()

    def decrement_count(self, n: int = 1) -> None:
        self.record_count = max(0, self.record_count - n)
        self.last_updated_at = datetime.now()

    @property
    def is_empty(self) -> bool:
        return self.record_count == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "collection_id": self.collection_id,
            "memory_type": self.memory_type.value,
            "dimension": self.dimension,
            "record_count": self.record_count,
            "created_at": self.created_at.isoformat(),
            "last_updated_at": (
                self.last_updated_at.isoformat() if self.last_updated_at else None
            ),
        }

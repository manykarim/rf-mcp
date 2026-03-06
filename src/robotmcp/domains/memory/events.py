"""Memory Domain — Domain Events.

All frozen dataclasses with timestamp and to_dict(), following ADR-001 conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class MemoryStored:
    """Raised when a new memory record is persisted."""

    record_id: str
    memory_type: str
    content_preview: str
    collection_id: str
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "MemoryStored",
            "record_id": self.record_id,
            "memory_type": self.memory_type,
            "content_preview": self.content_preview[:100],
            "collection_id": self.collection_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class MemoryRecalled:
    """Raised when a memory query is executed."""

    query_text: str
    memory_type: Optional[str]
    result_count: int
    top_similarity: float
    query_time_ms: float
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_empty_result(self) -> bool:
        return self.result_count == 0

    @property
    def is_slow_query(self) -> bool:
        return self.query_time_ms > 500

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "MemoryRecalled",
            "query_text": self.query_text[:200],
            "memory_type": self.memory_type,
            "result_count": self.result_count,
            "top_similarity": round(self.top_similarity, 4),
            "query_time_ms": round(self.query_time_ms, 2),
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class MemoryDecayed:
    """Raised when time decay is applied to a recall result."""

    record_id: str
    original_similarity: float
    decayed_similarity: float
    age_days: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def decay_ratio(self) -> float:
        if self.original_similarity == 0:
            return 0.0
        return self.decayed_similarity / self.original_similarity

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "MemoryDecayed",
            "record_id": self.record_id,
            "original_similarity": round(self.original_similarity, 4),
            "decayed_similarity": round(self.decayed_similarity, 4),
            "age_days": round(self.age_days, 1),
            "decay_ratio": round(self.decay_ratio, 4),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class CollectionCreated:
    """Raised when a new vector collection is created."""

    collection_id: str
    memory_type: str
    dimension: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "CollectionCreated",
            "collection_id": self.collection_id,
            "memory_type": self.memory_type,
            "dimension": self.dimension,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class MemoryPruned:
    """Raised when old records are pruned from a collection."""

    collection_id: str
    memory_type: str
    records_removed: int
    max_age_days: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "MemoryPruned",
            "collection_id": self.collection_id,
            "memory_type": self.memory_type,
            "records_removed": self.records_removed,
            "max_age_days": self.max_age_days,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class LocatorStored:
    """Raised when a locator outcome is stored in memory."""

    record_id: str
    locator: str
    keyword: str
    library: str
    outcome: str
    page_url: str = ""
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "LocatorStored",
            "record_id": self.record_id,
            "locator": self.locator,
            "keyword": self.keyword,
            "library": self.library,
            "outcome": self.outcome,
            "page_url": self.page_url,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
        }

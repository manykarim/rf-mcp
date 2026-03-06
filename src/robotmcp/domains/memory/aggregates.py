"""Memory Domain — Aggregates.

Aggregate roots managing consistency boundaries, following ADR-001 conventions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .entities import MemoryCollection, MemoryRecord
from .events import CollectionCreated, MemoryPruned, MemoryStored
from .value_objects import (
    MemoryEntry,
    MemoryQuery,
    MemoryType,
    StorageConfig,
    TimeDecayFactor,
)


# ---------------------------------------------------------------------------
# MemoryStore (Aggregate Root)
# ---------------------------------------------------------------------------

@dataclass
class MemoryStore:
    """Aggregate root managing all collections and coordinating operations.

    Invariants:
    - Each MemoryType has at most one MemoryCollection
    - All collections use the same embedding dimension
    - Pending events are accumulated and published atomically
    """

    __test__ = False  # Suppress pytest collection

    store_id: str
    config: StorageConfig
    collections: Dict[str, MemoryCollection] = field(default_factory=dict)
    time_decay: TimeDecayFactor = field(default_factory=TimeDecayFactor)
    _pending_events: List[Any] = field(default_factory=list)

    @classmethod
    def create(cls, config: Optional[StorageConfig] = None) -> MemoryStore:
        if config is None:
            config = StorageConfig.default()
        return cls(
            store_id=str(uuid.uuid4()),
            config=config,
            time_decay=TimeDecayFactor(half_life_days=config.time_decay_half_life),
        )

    # -- Collection management -----------------------------------------------

    def ensure_collection(self, memory_type: MemoryType) -> MemoryCollection:
        cid = memory_type.collection_name
        if cid not in self.collections:
            coll = MemoryCollection.for_type(memory_type, self.config.dimension)
            self.collections[cid] = coll
            self._pending_events.append(
                CollectionCreated(
                    collection_id=cid,
                    memory_type=memory_type.value,
                    dimension=self.config.dimension,
                )
            )
        return self.collections[cid]

    def get_collection(self, memory_type: MemoryType) -> Optional[MemoryCollection]:
        return self.collections.get(memory_type.collection_name)

    def has_collection(self, memory_type: MemoryType) -> bool:
        return memory_type.collection_name in self.collections

    # -- Record coordination -------------------------------------------------

    def prepare_store(
        self,
        entry: MemoryEntry,
        session_id: Optional[str] = None,
    ) -> MemoryRecord:
        coll = self.ensure_collection(entry.memory_type)
        record = MemoryRecord.create(entry=entry, session_id=session_id)
        coll.increment_count()
        self._pending_events.append(
            MemoryStored(
                record_id=record.record_id,
                memory_type=entry.memory_type.value,
                content_preview=entry.content_preview,
                collection_id=coll.collection_id,
                session_id=session_id,
            )
        )
        return record

    def prepare_recall(self, query: MemoryQuery) -> MemoryQuery:
        if query.memory_type is not None:
            self.ensure_collection(query.memory_type)
        return query

    # -- Pruning -------------------------------------------------------------

    def mark_for_pruning(
        self,
        memory_type: MemoryType,
        max_age_days: Optional[float] = None,
    ) -> None:
        age = max_age_days or self.config.prune_age_days
        coll = self.get_collection(memory_type)
        if coll is not None:
            self._pending_events.append(
                MemoryPruned(
                    collection_id=coll.collection_id,
                    memory_type=memory_type.value,
                    records_removed=0,  # Actual count set by repository
                    max_age_days=age,
                )
            )

    # -- Events --------------------------------------------------------------

    def collect_events(self) -> List[Any]:
        events = list(self._pending_events)
        self._pending_events.clear()
        return events

    # -- Stats ---------------------------------------------------------------

    @property
    def total_record_count(self) -> int:
        return sum(c.record_count for c in self.collections.values())

    @property
    def collection_count(self) -> int:
        return len(self.collections)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "store_id": self.store_id,
            "enabled": self.config.enabled,
            "embedding_model": self.config.embedding_model,
            "dimension": self.config.dimension,
            "total_records": self.total_record_count,
            "collections": {
                k: v.to_dict() for k, v in self.collections.items()
            },
        }

    def validate(self) -> List[str]:
        errors: List[str] = []
        for coll in self.collections.values():
            if coll.dimension != self.config.dimension:
                errors.append(
                    f"Collection {coll.collection_id} dimension "
                    f"{coll.dimension} != config {self.config.dimension}"
                )
        return errors


# ---------------------------------------------------------------------------
# EmbeddingBackend (Aggregate Root)
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingBackend:
    """Manages embedding model lifecycle and backend detection.

    Detects available backends in priority order:
    1. model2vec (Tier 1, ~30 MB, fastest)
    2. fastembed (Tier 2, ~200 MB, higher quality)
    3. sentence-transformers (Tier 3, ~2 GB, best quality)
    """

    backend_name: Optional[str] = None
    model_name: Optional[str] = None
    dimension: Optional[int] = None
    is_available: bool = False
    _pending_events: List[Any] = field(default_factory=list)

    @classmethod
    def detect(cls, preferred_model: str = "potion-base-8M") -> EmbeddingBackend:
        """Probe available backends and return configured instance."""
        # Try model2vec first (lightest)
        try:
            from model2vec import StaticModel  # noqa: F401

            model_map = {
                "potion-base-8M": ("minishlab/potion-base-8M", 256),
            }
            if preferred_model in model_map:
                name, dim = model_map[preferred_model]
                return cls(
                    backend_name="model2vec",
                    model_name=name,
                    dimension=dim,
                    is_available=True,
                )
        except ImportError:
            pass

        # Try fastembed
        try:
            from fastembed import TextEmbedding  # noqa: F401

            return cls(
                backend_name="fastembed",
                model_name="BAAI/bge-small-en-v1.5",
                dimension=384,
                is_available=True,
            )
        except ImportError:
            pass

        # Try sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401

            return cls(
                backend_name="sentence-transformers",
                model_name="all-MiniLM-L6-v2",
                dimension=384,
                is_available=True,
            )
        except ImportError:
            pass

        return cls(is_available=False)

    def collect_events(self) -> List[Any]:
        events = list(self._pending_events)
        self._pending_events.clear()
        return events

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "is_available": self.is_available,
        }

"""Memory Domain — Repository (Port + In-Memory Test Double).

Protocol-based port for memory persistence, following ADR-001 conventions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from .entities import MemoryRecord
from .value_objects import EmbeddingVector, SimilarityScore


@runtime_checkable
class MemoryRepository(Protocol):
    """Port for memory persistence and vector search."""

    async def store(self, collection_name: str, record: MemoryRecord) -> None: ...

    async def get_by_id(
        self, collection_name: str, record_id: str
    ) -> Optional[MemoryRecord]: ...

    async def search(
        self,
        collection_names: List[str],
        query_embedding: EmbeddingVector,
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[Tuple[MemoryRecord, SimilarityScore]]: ...

    async def delete(self, collection_name: str, record_id: str) -> bool: ...

    async def delete_by_age(
        self, collection_name: str, max_age_days: float
    ) -> int: ...

    async def ensure_collection(
        self, collection_name: str, dimension: int
    ) -> None: ...

    async def collection_stats(
        self, collection_name: str
    ) -> Optional[Dict[str, Any]]: ...

    async def count(self, collection_name: Optional[str] = None) -> int: ...


class InMemoryMemoryRepository:
    """In-memory implementation for testing. Brute-force cosine search."""

    def __init__(self) -> None:
        self._records: Dict[str, Dict[str, MemoryRecord]] = {}  # coll -> {id -> record}
        self._dimensions: Dict[str, int] = {}

    async def store(self, collection_name: str, record: MemoryRecord) -> None:
        if collection_name not in self._records:
            self._records[collection_name] = {}
        self._records[collection_name][record.record_id] = record

    async def get_by_id(
        self, collection_name: str, record_id: str
    ) -> Optional[MemoryRecord]:
        coll = self._records.get(collection_name, {})
        return coll.get(record_id)

    async def search(
        self,
        collection_names: List[str],
        query_embedding: EmbeddingVector,
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> List[Tuple[MemoryRecord, SimilarityScore]]:
        results: List[Tuple[MemoryRecord, SimilarityScore]] = []
        for coll_name in collection_names:
            for record in self._records.get(coll_name, {}).values():
                if record.entry.embedding is None:
                    continue
                sim = self._cosine_similarity(
                    query_embedding.values, record.entry.embedding.values
                )
                if sim >= min_similarity:
                    results.append(
                        (record, SimilarityScore.cosine(sim))
                    )
        results.sort(key=lambda x: x[1].value, reverse=True)
        return results[:top_k]

    async def delete(self, collection_name: str, record_id: str) -> bool:
        coll = self._records.get(collection_name, {})
        if record_id in coll:
            del coll[record_id]
            return True
        return False

    async def delete_by_age(
        self, collection_name: str, max_age_days: float
    ) -> int:
        coll = self._records.get(collection_name, {})
        to_delete = [
            rid for rid, rec in coll.items() if rec.is_stale(max_age_days)
        ]
        for rid in to_delete:
            del coll[rid]
        return len(to_delete)

    async def ensure_collection(
        self, collection_name: str, dimension: int
    ) -> None:
        if collection_name not in self._records:
            self._records[collection_name] = {}
        self._dimensions[collection_name] = dimension

    async def collection_stats(
        self, collection_name: str
    ) -> Optional[Dict[str, Any]]:
        if collection_name not in self._records:
            return None
        return {
            "collection_name": collection_name,
            "record_count": len(self._records[collection_name]),
            "dimension": self._dimensions.get(collection_name, 0),
        }

    async def count(self, collection_name: Optional[str] = None) -> int:
        if collection_name:
            return len(self._records.get(collection_name, {}))
        return sum(len(recs) for recs in self._records.values())

    @staticmethod
    def _cosine_similarity(
        a: Tuple[float, ...], b: Tuple[float, ...]
    ) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / (norm_a * norm_b)))

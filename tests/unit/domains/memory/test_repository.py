"""Tests for memory domain repository.

Covers: MemoryRepository protocol, InMemoryMemoryRepository.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from robotmcp.domains.memory.entities import MemoryRecord
from robotmcp.domains.memory.repository import (
    InMemoryMemoryRepository,
    MemoryRepository,
)
from robotmcp.domains.memory.value_objects import (
    EmbeddingVector,
    MemoryEntry,
    MemoryType,
    SimilarityScore,
)

_async_mark = pytest.mark.asyncio(loop_scope="module")


# =========================================================================
# Helpers
# =========================================================================


def _make_entry(
    content: str = "test content",
    memory_type: MemoryType | None = None,
    embedding: EmbeddingVector | None = None,
    tags: tuple[str, ...] = (),
) -> MemoryEntry:
    mt = memory_type or MemoryType.keywords()
    return MemoryEntry(
        content=content,
        memory_type=mt,
        embedding=embedding,
        tags=tags,
    )


def _make_embedding(values: list[float], model: str = "test") -> EmbeddingVector:
    return EmbeddingVector.from_list(values, model)


def _make_record(
    content: str = "test content",
    memory_type: MemoryType | None = None,
    embedding: EmbeddingVector | None = None,
    session_id: str | None = None,
) -> MemoryRecord:
    entry = _make_entry(content=content, memory_type=memory_type, embedding=embedding)
    return MemoryRecord.create(entry, session_id=session_id)


# =========================================================================
# Protocol conformance
# =========================================================================


class TestMemoryRepositoryProtocol:
    def test_in_memory_implements_protocol(self):
        repo = InMemoryMemoryRepository()
        assert isinstance(repo, MemoryRepository)


# =========================================================================
# InMemoryMemoryRepository — store + get_by_id
# =========================================================================


@_async_mark
class TestStoreAndGetById:
    async def test_store_and_retrieve(self):
        repo = InMemoryMemoryRepository()
        record = _make_record()
        await repo.store("coll_a", record)

        retrieved = await repo.get_by_id("coll_a", record.record_id)
        assert retrieved is not None
        assert retrieved.record_id == record.record_id
        assert retrieved.entry.content == "test content"

    async def test_get_by_id_nonexistent_collection(self):
        repo = InMemoryMemoryRepository()
        result = await repo.get_by_id("no_such_coll", "no-id")
        assert result is None

    async def test_get_by_id_nonexistent_record(self):
        repo = InMemoryMemoryRepository()
        record = _make_record()
        await repo.store("coll_a", record)
        result = await repo.get_by_id("coll_a", "wrong-id")
        assert result is None

    async def test_store_multiple_records_same_collection(self):
        repo = InMemoryMemoryRepository()
        r1 = _make_record(content="first")
        r2 = _make_record(content="second")
        await repo.store("coll_a", r1)
        await repo.store("coll_a", r2)

        assert (await repo.get_by_id("coll_a", r1.record_id)) is not None
        assert (await repo.get_by_id("coll_a", r2.record_id)) is not None

    async def test_store_same_id_different_collections(self):
        repo = InMemoryMemoryRepository()
        r = _make_record(content="content")
        await repo.store("coll_a", r)
        await repo.store("coll_b", r)

        assert (await repo.get_by_id("coll_a", r.record_id)) is not None
        assert (await repo.get_by_id("coll_b", r.record_id)) is not None

    async def test_store_overwrites_same_id(self):
        repo = InMemoryMemoryRepository()
        entry1 = _make_entry(content="original")
        r1 = MemoryRecord.create(entry1)

        entry2 = _make_entry(content="updated")
        r2 = MemoryRecord(
            record_id=r1.record_id,
            entry=entry2,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
        )
        await repo.store("coll_a", r1)
        await repo.store("coll_a", r2)

        result = await repo.get_by_id("coll_a", r1.record_id)
        assert result.entry.content == "updated"

    async def test_auto_creates_collection_on_store(self):
        repo = InMemoryMemoryRepository()
        record = _make_record()
        await repo.store("new_coll", record)
        count = await repo.count("new_coll")
        assert count == 1


# =========================================================================
# InMemoryMemoryRepository — search
# =========================================================================


@_async_mark
class TestSearch:
    async def test_search_basic_cosine(self):
        repo = InMemoryMemoryRepository()

        # Store records with embeddings
        e1 = _make_embedding([1.0, 0.0, 0.0])
        r1 = _make_record(content="x-axis", embedding=e1)
        await repo.store("coll", r1)

        e2 = _make_embedding([0.0, 1.0, 0.0])
        r2 = _make_record(content="y-axis", embedding=e2)
        await repo.store("coll", r2)

        # Query close to x-axis
        query_emb = _make_embedding([0.9, 0.1, 0.0])
        results = await repo.search(["coll"], query_emb)

        assert len(results) == 2
        # x-axis record should be first (higher similarity)
        assert results[0][0].entry.content == "x-axis"
        assert results[0][1].value > results[1][1].value

    async def test_search_min_similarity_filter(self):
        repo = InMemoryMemoryRepository()

        e1 = _make_embedding([1.0, 0.0, 0.0])
        r1 = _make_record(content="x-axis", embedding=e1)
        await repo.store("coll", r1)

        e2 = _make_embedding([0.0, 1.0, 0.0])
        r2 = _make_record(content="y-axis", embedding=e2)
        await repo.store("coll", r2)

        # Query exactly along x-axis with high min_similarity
        query_emb = _make_embedding([1.0, 0.0, 0.0])
        results = await repo.search(["coll"], query_emb, min_similarity=0.99)

        # Only x-axis record should match (similarity ~1.0)
        assert len(results) == 1
        assert results[0][0].entry.content == "x-axis"

    async def test_search_top_k_limit(self):
        repo = InMemoryMemoryRepository()

        for i in range(10):
            e = _make_embedding([float(i), 1.0, 0.0])
            r = _make_record(content=f"rec-{i}", embedding=e)
            await repo.store("coll", r)

        query_emb = _make_embedding([5.0, 1.0, 0.0])
        results = await repo.search(["coll"], query_emb, top_k=3)

        assert len(results) == 3

    async def test_search_skips_records_without_embedding(self):
        repo = InMemoryMemoryRepository()

        # Record without embedding
        r1 = _make_record(content="no embedding")
        await repo.store("coll", r1)

        # Record with embedding
        e2 = _make_embedding([1.0, 0.0, 0.0])
        r2 = _make_record(content="has embedding", embedding=e2)
        await repo.store("coll", r2)

        query_emb = _make_embedding([1.0, 0.0, 0.0])
        results = await repo.search(["coll"], query_emb)

        assert len(results) == 1
        assert results[0][0].entry.content == "has embedding"

    async def test_search_empty_collection(self):
        repo = InMemoryMemoryRepository()
        query_emb = _make_embedding([1.0, 0.0, 0.0])
        results = await repo.search(["nonexistent"], query_emb)
        assert results == []

    async def test_search_returns_similarity_scores(self):
        repo = InMemoryMemoryRepository()

        e1 = _make_embedding([1.0, 0.0, 0.0])
        r1 = _make_record(content="target", embedding=e1)
        await repo.store("coll", r1)

        query_emb = _make_embedding([1.0, 0.0, 0.0])
        results = await repo.search(["coll"], query_emb)

        assert len(results) == 1
        record, score = results[0]
        assert isinstance(score, SimilarityScore)
        assert score.value == pytest.approx(1.0)
        assert score.distance_metric == "cosine"

    async def test_search_sorted_by_similarity_descending(self):
        repo = InMemoryMemoryRepository()

        embeddings = [
            ([1.0, 0.0, 0.0], "x"),
            ([0.7, 0.7, 0.0], "xy"),
            ([0.0, 1.0, 0.0], "y"),
        ]
        for vals, content in embeddings:
            e = _make_embedding(vals)
            r = _make_record(content=content, embedding=e)
            await repo.store("coll", r)

        query_emb = _make_embedding([1.0, 0.0, 0.0])
        results = await repo.search(["coll"], query_emb)

        scores = [s.value for _, s in results]
        assert scores == sorted(scores, reverse=True)

    async def test_search_across_multiple_collections(self):
        repo = InMemoryMemoryRepository()

        e1 = _make_embedding([1.0, 0.0, 0.0])
        r1 = _make_record(content="in coll_a", embedding=e1)
        await repo.store("coll_a", r1)

        e2 = _make_embedding([0.9, 0.1, 0.0])
        r2 = _make_record(content="in coll_b", embedding=e2)
        await repo.store("coll_b", r2)

        query_emb = _make_embedding([1.0, 0.0, 0.0])
        results = await repo.search(["coll_a", "coll_b"], query_emb)

        assert len(results) == 2
        contents = {r.entry.content for r, _ in results}
        assert contents == {"in coll_a", "in coll_b"}

    async def test_search_orthogonal_returns_zero_when_min_similarity_positive(self):
        repo = InMemoryMemoryRepository()

        e = _make_embedding([0.0, 1.0, 0.0])
        r = _make_record(content="y-only", embedding=e)
        await repo.store("coll", r)

        # Query orthogonal
        query_emb = _make_embedding([1.0, 0.0, 0.0])
        results = await repo.search(["coll"], query_emb, min_similarity=0.01)
        assert len(results) == 0

    async def test_search_zero_vector_returns_nothing_with_positive_min(self):
        repo = InMemoryMemoryRepository()

        e = _make_embedding([1.0, 0.0, 0.0])
        r = _make_record(content="c", embedding=e)
        await repo.store("coll", r)

        query_emb = _make_embedding([0.0, 0.0, 0.0])
        results = await repo.search(["coll"], query_emb, min_similarity=0.01)
        assert len(results) == 0


# =========================================================================
# InMemoryMemoryRepository — delete
# =========================================================================


@_async_mark
class TestDelete:
    async def test_delete_existing(self):
        repo = InMemoryMemoryRepository()
        record = _make_record()
        await repo.store("coll", record)
        deleted = await repo.delete("coll", record.record_id)
        assert deleted is True
        assert (await repo.get_by_id("coll", record.record_id)) is None

    async def test_delete_nonexistent_record(self):
        repo = InMemoryMemoryRepository()
        record = _make_record()
        await repo.store("coll", record)
        deleted = await repo.delete("coll", "wrong-id")
        assert deleted is False

    async def test_delete_nonexistent_collection(self):
        repo = InMemoryMemoryRepository()
        deleted = await repo.delete("no_coll", "no-id")
        assert deleted is False

    async def test_delete_does_not_affect_other_records(self):
        repo = InMemoryMemoryRepository()
        r1 = _make_record(content="first")
        r2 = _make_record(content="second")
        await repo.store("coll", r1)
        await repo.store("coll", r2)

        await repo.delete("coll", r1.record_id)
        assert (await repo.get_by_id("coll", r2.record_id)) is not None
        assert (await repo.count("coll")) == 1


# =========================================================================
# InMemoryMemoryRepository — delete_by_age
# =========================================================================


@_async_mark
class TestDeleteByAge:
    async def test_delete_by_age_removes_old(self):
        repo = InMemoryMemoryRepository()

        old_record = _make_record(content="old")
        old_record.accessed_at = datetime.now() - timedelta(days=100)
        await repo.store("coll", old_record)

        new_record = _make_record(content="new")
        await repo.store("coll", new_record)

        removed = await repo.delete_by_age("coll", max_age_days=50.0)
        assert removed == 1
        assert (await repo.get_by_id("coll", old_record.record_id)) is None
        assert (await repo.get_by_id("coll", new_record.record_id)) is not None

    async def test_delete_by_age_no_matches(self):
        repo = InMemoryMemoryRepository()
        record = _make_record()
        await repo.store("coll", record)

        removed = await repo.delete_by_age("coll", max_age_days=1.0)
        assert removed == 0

    async def test_delete_by_age_nonexistent_collection(self):
        repo = InMemoryMemoryRepository()
        removed = await repo.delete_by_age("no_coll", max_age_days=1.0)
        assert removed == 0

    async def test_delete_by_age_all_stale(self):
        repo = InMemoryMemoryRepository()
        for i in range(5):
            r = _make_record(content=f"rec-{i}")
            r.accessed_at = datetime.now() - timedelta(days=200)
            await repo.store("coll", r)

        removed = await repo.delete_by_age("coll", max_age_days=90.0)
        assert removed == 5
        assert (await repo.count("coll")) == 0


# =========================================================================
# InMemoryMemoryRepository — ensure_collection + collection_stats
# =========================================================================


@_async_mark
class TestEnsureCollectionAndStats:
    async def test_ensure_collection_creates_empty(self):
        repo = InMemoryMemoryRepository()
        await repo.ensure_collection("new_coll", 256)
        stats = await repo.collection_stats("new_coll")
        assert stats is not None
        assert stats["record_count"] == 0
        assert stats["dimension"] == 256

    async def test_ensure_collection_idempotent(self):
        repo = InMemoryMemoryRepository()
        await repo.ensure_collection("coll", 256)
        await repo.ensure_collection("coll", 384)  # Updates dimension
        stats = await repo.collection_stats("coll")
        assert stats["dimension"] == 384

    async def test_ensure_collection_does_not_clear_records(self):
        repo = InMemoryMemoryRepository()
        record = _make_record()
        await repo.store("coll", record)
        await repo.ensure_collection("coll", 256)
        assert (await repo.count("coll")) == 1

    async def test_collection_stats_nonexistent(self):
        repo = InMemoryMemoryRepository()
        stats = await repo.collection_stats("no_coll")
        assert stats is None

    async def test_collection_stats_with_records(self):
        repo = InMemoryMemoryRepository()
        await repo.ensure_collection("coll", 256)
        r1 = _make_record()
        r2 = _make_record()
        await repo.store("coll", r1)
        await repo.store("coll", r2)

        stats = await repo.collection_stats("coll")
        assert stats["collection_name"] == "coll"
        assert stats["record_count"] == 2
        assert stats["dimension"] == 256


# =========================================================================
# InMemoryMemoryRepository — count
# =========================================================================


@_async_mark
class TestCount:
    async def test_count_empty(self):
        repo = InMemoryMemoryRepository()
        assert (await repo.count()) == 0

    async def test_count_specific_collection(self):
        repo = InMemoryMemoryRepository()
        r1 = _make_record()
        r2 = _make_record()
        await repo.store("coll_a", r1)
        await repo.store("coll_b", r2)

        assert (await repo.count("coll_a")) == 1
        assert (await repo.count("coll_b")) == 1

    async def test_count_global(self):
        repo = InMemoryMemoryRepository()
        await repo.store("coll_a", _make_record())
        await repo.store("coll_a", _make_record())
        await repo.store("coll_b", _make_record())

        assert (await repo.count()) == 3

    async def test_count_nonexistent_collection(self):
        repo = InMemoryMemoryRepository()
        assert (await repo.count("no_coll")) == 0

    async def test_count_after_delete(self):
        repo = InMemoryMemoryRepository()
        r = _make_record()
        await repo.store("coll", r)
        assert (await repo.count("coll")) == 1

        await repo.delete("coll", r.record_id)
        assert (await repo.count("coll")) == 0
        assert (await repo.count()) == 0


# =========================================================================
# InMemoryMemoryRepository — _cosine_similarity static method
# =========================================================================


class TestCosineStaticMethod:
    def test_identical_vectors(self):
        sim = InMemoryMemoryRepository._cosine_similarity(
            (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)
        )
        assert sim == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        sim = InMemoryMemoryRepository._cosine_similarity(
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
        )
        assert sim == pytest.approx(0.0)

    def test_opposite_vectors_clamped_to_zero(self):
        # Opposite vectors have cosine = -1, but clamped to 0.0
        sim = InMemoryMemoryRepository._cosine_similarity(
            (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)
        )
        assert sim == 0.0

    def test_zero_vector_returns_zero(self):
        sim = InMemoryMemoryRepository._cosine_similarity(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
        )
        assert sim == 0.0

    def test_both_zero_vectors(self):
        sim = InMemoryMemoryRepository._cosine_similarity(
            (0.0, 0.0), (0.0, 0.0)
        )
        assert sim == 0.0

    def test_clamped_to_max_1(self):
        # Due to floating-point, result should never exceed 1.0
        sim = InMemoryMemoryRepository._cosine_similarity(
            (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)
        )
        assert sim <= 1.0

    def test_similar_vectors(self):
        sim = InMemoryMemoryRepository._cosine_similarity(
            (1.0, 0.0, 0.0), (0.9, 0.1, 0.0)
        )
        assert 0.9 < sim <= 1.0

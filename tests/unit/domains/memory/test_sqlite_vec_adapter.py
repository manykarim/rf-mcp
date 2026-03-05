"""Tests for SqliteVecRepository adapter.

Covers: lazy connection, WAL mode, ensure_collection, store/get_by_id
roundtrip, search with cosine similarity, min_similarity filter, top_k
limit, delete, delete_by_age, collection_stats, count, and graceful
handling of nonexistent collections.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator

import pytest

from robotmcp.domains.memory.entities import MemoryRecord
from robotmcp.domains.memory.value_objects import (
    EmbeddingVector,
    MemoryEntry,
    MemoryType,
    SimilarityScore,
)

try:
    import sqlite_vec  # noqa: F401

    SQLITE_VEC_AVAILABLE = True
except ImportError:
    SQLITE_VEC_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SQLITE_VEC_AVAILABLE, reason="sqlite-vec not installed"
)

_async_mark = pytest.mark.asyncio(loop_scope="module")

# =========================================================================
# Fixtures
# =========================================================================

_DIMENSION = 4


@pytest.fixture()
def db_dir() -> Generator[str, None, None]:
    """Create a temporary directory for the database file."""
    tmpdir = tempfile.mkdtemp(prefix="rfmcp_sqlvec_test_")
    yield tmpdir
    # Cleanup: remove any leftover files
    for f in Path(tmpdir).iterdir():
        try:
            f.unlink()
        except OSError:
            pass
    try:
        Path(tmpdir).rmdir()
    except OSError:
        pass


@pytest.fixture()
def db_path(db_dir: str) -> str:
    return os.path.join(db_dir, "test_memory.db")


@pytest.fixture()
def repo(db_path: str):
    from robotmcp.domains.memory.adapters.sqlite_vec_adapter import (
        SqliteVecRepository,
    )

    r = SqliteVecRepository(db_path)
    yield r
    r.close()


# =========================================================================
# Helpers
# =========================================================================


def _make_embedding(
    values: list[float] | None = None, model: str = "test-model"
) -> EmbeddingVector:
    if values is None:
        values = [1.0, 0.0, 0.0, 0.0]
    return EmbeddingVector.from_list(values, model)


def _make_entry(
    content: str = "test content",
    memory_type: MemoryType | None = None,
    embedding: EmbeddingVector | None = None,
    tags: tuple[str, ...] = (),
) -> MemoryEntry:
    mt = memory_type or MemoryType.keywords()
    if embedding is None:
        embedding = _make_embedding()
    return MemoryEntry(
        content=content,
        memory_type=mt,
        embedding=embedding,
        tags=tags,
    )


def _make_record(
    content: str = "test content",
    memory_type: MemoryType | None = None,
    embedding: EmbeddingVector | None = None,
    session_id: str | None = None,
    tags: tuple[str, ...] = (),
) -> MemoryRecord:
    if embedding is None:
        embedding = _make_embedding()
    entry = _make_entry(
        content=content, memory_type=memory_type, embedding=embedding, tags=tags
    )
    return MemoryRecord.create(entry, session_id=session_id)


# =========================================================================
# Constructor and connection
# =========================================================================


@_async_mark
class TestConstructor:
    async def test_creates_db_file_on_first_use(self, repo, db_path):
        """Lazy connection should create the database file."""
        await repo.ensure_collection("test_coll", _DIMENSION)
        assert Path(db_path).exists()

    async def test_creates_parent_directories(self, db_dir):
        from robotmcp.domains.memory.adapters.sqlite_vec_adapter import (
            SqliteVecRepository,
        )

        nested_path = os.path.join(db_dir, "sub", "dir", "test.db")
        r = SqliteVecRepository(nested_path)
        try:
            await r.ensure_collection("test_coll", _DIMENSION)
            assert Path(nested_path).exists()
        finally:
            r.close()

    async def test_wal_mode_enabled(self, repo, db_path):
        """WAL journal mode should be set on connection."""
        await repo.ensure_collection("test_coll", _DIMENSION)
        conn = repo._get_connection()
        row = conn.execute("PRAGMA journal_mode").fetchone()
        mode = row["journal_mode"] if hasattr(row, "keys") else row[0]
        assert mode == "wal"


# =========================================================================
# ensure_collection
# =========================================================================


@_async_mark
class TestEnsureCollection:
    async def test_creates_vec0_virtual_table(self, repo):
        await repo.ensure_collection("my_coll", _DIMENSION)
        conn = repo._get_connection()
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='my_coll_vec'"
        ).fetchone()
        assert row is not None

    async def test_creates_metadata_table(self, repo):
        await repo.ensure_collection("my_coll", _DIMENSION)
        conn = repo._get_connection()
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='my_coll_meta'"
        ).fetchone()
        assert row is not None

    async def test_idempotent(self, repo):
        """Calling ensure_collection twice should not raise."""
        await repo.ensure_collection("coll", _DIMENSION)
        await repo.ensure_collection("coll", _DIMENSION)
        stats = await repo.collection_stats("coll")
        assert stats is not None
        assert stats["record_count"] == 0

    async def test_sanitizes_name_with_special_chars(self, repo):
        await repo.ensure_collection("my-coll.with spaces!", _DIMENSION)
        conn = repo._get_connection()
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='my_coll_with_spaces__meta'"
        ).fetchone()
        assert row is not None

    async def test_sanitizes_name_starting_with_digit(self, repo):
        await repo.ensure_collection("123abc", _DIMENSION)
        conn = repo._get_connection()
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='c_123abc_meta'"
        ).fetchone()
        assert row is not None


# =========================================================================
# store + get_by_id roundtrip
# =========================================================================


@_async_mark
class TestStoreAndGetById:
    async def test_store_and_retrieve_roundtrip(self, repo):
        record = _make_record(content="hello world", tags=("tag1", "tag2"))
        await repo.ensure_collection("coll", _DIMENSION)
        await repo.store("coll", record)

        retrieved = await repo.get_by_id("coll", record.record_id)
        assert retrieved is not None
        assert retrieved.record_id == record.record_id
        assert retrieved.entry.content == "hello world"
        assert retrieved.entry.memory_type.value == MemoryType.KEYWORDS
        assert set(retrieved.entry.tags) == {"tag1", "tag2"}

    async def test_store_preserves_metadata(self, repo):
        entry = MemoryEntry(
            content="with metadata",
            memory_type=MemoryType.documentation(),
            embedding=_make_embedding(),
            metadata={"key": "value", "num": 42},
        )
        record = MemoryRecord.create(entry)
        await repo.ensure_collection("coll", _DIMENSION)
        await repo.store("coll", record)

        retrieved = await repo.get_by_id("coll", record.record_id)
        assert retrieved is not None
        assert retrieved.entry.metadata["key"] == "value"
        assert retrieved.entry.metadata["num"] == 42

    async def test_store_preserves_session_id(self, repo):
        record = _make_record(session_id="sess-123")
        await repo.ensure_collection("coll", _DIMENSION)
        await repo.store("coll", record)

        retrieved = await repo.get_by_id("coll", record.record_id)
        assert retrieved is not None
        assert retrieved.session_id == "sess-123"

    async def test_store_preserves_timestamps(self, repo):
        record = _make_record()
        await repo.ensure_collection("coll", _DIMENSION)
        await repo.store("coll", record)

        retrieved = await repo.get_by_id("coll", record.record_id)
        assert retrieved is not None
        # ISO roundtrip may lose microsecond precision, so check to the second
        assert retrieved.created_at.replace(microsecond=0) == record.created_at.replace(
            microsecond=0
        )

    async def test_store_skips_record_without_embedding(self, repo):
        entry = MemoryEntry(
            content="no embedding",
            memory_type=MemoryType.keywords(),
        )
        record = MemoryRecord.create(entry)
        await repo.ensure_collection("coll", _DIMENSION)
        await repo.store("coll", record)

        retrieved = await repo.get_by_id("coll", record.record_id)
        assert retrieved is None

    async def test_get_by_id_nonexistent_collection(self, repo):
        result = await repo.get_by_id("nonexistent_coll", "no-such-id")
        assert result is None

    async def test_get_by_id_nonexistent_record(self, repo):
        record = _make_record()
        await repo.ensure_collection("coll", _DIMENSION)
        await repo.store("coll", record)

        result = await repo.get_by_id("coll", "wrong-id")
        assert result is None

    async def test_store_multiple_records(self, repo):
        await repo.ensure_collection("coll", _DIMENSION)
        r1 = _make_record(content="first")
        r2 = _make_record(content="second")
        await repo.store("coll", r1)
        await repo.store("coll", r2)

        assert (await repo.get_by_id("coll", r1.record_id)) is not None
        assert (await repo.get_by_id("coll", r2.record_id)) is not None
        assert (await repo.count("coll")) == 2


# =========================================================================
# search with cosine similarity
# =========================================================================


@_async_mark
class TestSearch:
    async def test_search_basic_cosine(self, repo):
        await repo.ensure_collection("coll", _DIMENSION)

        e1 = _make_embedding([1.0, 0.0, 0.0, 0.0])
        r1 = _make_record(content="x-axis", embedding=e1)
        await repo.store("coll", r1)

        e2 = _make_embedding([0.0, 1.0, 0.0, 0.0])
        r2 = _make_record(content="y-axis", embedding=e2)
        await repo.store("coll", r2)

        query_emb = _make_embedding([0.9, 0.1, 0.0, 0.0])
        results = await repo.search(["coll"], query_emb)

        assert len(results) == 2
        # x-axis should be first (closer to query)
        assert results[0][0].entry.content == "x-axis"
        assert results[0][1].value > results[1][1].value

    async def test_search_identical_vector_returns_high_similarity(self, repo):
        await repo.ensure_collection("coll", _DIMENSION)

        emb = _make_embedding([1.0, 0.0, 0.0, 0.0])
        record = _make_record(content="exact", embedding=emb)
        await repo.store("coll", record)

        results = await repo.search(["coll"], emb)
        assert len(results) == 1
        _, score = results[0]
        assert isinstance(score, SimilarityScore)
        # sqlite-vec cosine distance for identical vectors should yield
        # similarity close to 1.0
        assert score.value >= 0.95

    async def test_search_with_min_similarity_filter(self, repo):
        await repo.ensure_collection("coll", _DIMENSION)

        e1 = _make_embedding([1.0, 0.0, 0.0, 0.0])
        r1 = _make_record(content="close", embedding=e1)
        await repo.store("coll", r1)

        e2 = _make_embedding([0.0, 0.0, 0.0, 1.0])
        r2 = _make_record(content="far", embedding=e2)
        await repo.store("coll", r2)

        query_emb = _make_embedding([1.0, 0.0, 0.0, 0.0])
        results = await repo.search(["coll"], query_emb, min_similarity=0.5)

        # Only the close vector should pass the filter
        contents = [r.entry.content for r, _ in results]
        assert "close" in contents
        assert "far" not in contents

    async def test_search_top_k_limit(self, repo):
        await repo.ensure_collection("coll", _DIMENSION)

        for i in range(10):
            emb = _make_embedding([float(i + 1), 1.0, 0.0, 0.0])
            r = _make_record(content=f"rec-{i}", embedding=emb)
            await repo.store("coll", r)

        query_emb = _make_embedding([5.0, 1.0, 0.0, 0.0])
        results = await repo.search(["coll"], query_emb, top_k=3)
        assert len(results) <= 3

    async def test_search_empty_collection(self, repo):
        await repo.ensure_collection("empty_coll", _DIMENSION)
        query_emb = _make_embedding()
        results = await repo.search(["empty_coll"], query_emb)
        assert results == []

    async def test_search_nonexistent_collection(self, repo):
        query_emb = _make_embedding()
        results = await repo.search(["no_such_coll"], query_emb)
        assert results == []

    async def test_search_sorted_descending(self, repo):
        await repo.ensure_collection("coll", _DIMENSION)

        vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [0.7, 0.7, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
        for vals in vectors:
            emb = _make_embedding(vals)
            r = _make_record(embedding=emb)
            await repo.store("coll", r)

        query_emb = _make_embedding([1.0, 0.0, 0.0, 0.0])
        results = await repo.search(["coll"], query_emb)
        scores = [s.value for _, s in results]
        assert scores == sorted(scores, reverse=True)

    async def test_search_across_multiple_collections(self, repo):
        await repo.ensure_collection("coll_a", _DIMENSION)
        await repo.ensure_collection("coll_b", _DIMENSION)

        e1 = _make_embedding([1.0, 0.0, 0.0, 0.0])
        r1 = _make_record(content="in A", embedding=e1)
        await repo.store("coll_a", r1)

        e2 = _make_embedding([0.9, 0.1, 0.0, 0.0])
        r2 = _make_record(content="in B", embedding=e2)
        await repo.store("coll_b", r2)

        query_emb = _make_embedding([1.0, 0.0, 0.0, 0.0])
        results = await repo.search(["coll_a", "coll_b"], query_emb)

        contents = {r.entry.content for r, _ in results}
        assert "in A" in contents
        assert "in B" in contents


# =========================================================================
# delete
# =========================================================================


@_async_mark
class TestDelete:
    async def test_delete_existing_record(self, repo):
        await repo.ensure_collection("coll", _DIMENSION)
        record = _make_record()
        await repo.store("coll", record)

        deleted = await repo.delete("coll", record.record_id)
        assert deleted is True
        assert (await repo.get_by_id("coll", record.record_id)) is None

    async def test_delete_nonexistent_record(self, repo):
        await repo.ensure_collection("coll", _DIMENSION)
        record = _make_record()
        await repo.store("coll", record)

        deleted = await repo.delete("coll", "wrong-id")
        assert deleted is False

    async def test_delete_nonexistent_collection(self, repo):
        deleted = await repo.delete("no_coll", "no-id")
        assert deleted is False

    async def test_delete_does_not_affect_other_records(self, repo):
        await repo.ensure_collection("coll", _DIMENSION)
        r1 = _make_record(content="keep")
        r2 = _make_record(content="remove")
        await repo.store("coll", r1)
        await repo.store("coll", r2)

        await repo.delete("coll", r2.record_id)
        assert (await repo.get_by_id("coll", r1.record_id)) is not None
        assert (await repo.count("coll")) == 1

    async def test_delete_also_removes_from_vec_table(self, repo):
        """After deleting, the vector row should also be gone."""
        await repo.ensure_collection("coll", _DIMENSION)
        record = _make_record()
        await repo.store("coll", record)

        # Confirm vec row exists
        conn = repo._get_connection()
        vec_count_before = conn.execute(
            "SELECT COUNT(*) FROM [coll_vec]"
        ).fetchone()[0]
        assert vec_count_before == 1

        await repo.delete("coll", record.record_id)

        vec_count_after = conn.execute(
            "SELECT COUNT(*) FROM [coll_vec]"
        ).fetchone()[0]
        assert vec_count_after == 0


# =========================================================================
# delete_by_age
# =========================================================================


@_async_mark
class TestDeleteByAge:
    async def test_removes_old_records(self, repo):
        await repo.ensure_collection("coll", _DIMENSION)

        old_record = _make_record(content="old")
        old_record.accessed_at = datetime.now() - timedelta(days=100)
        await repo.store("coll", old_record)

        new_record = _make_record(content="new")
        await repo.store("coll", new_record)

        removed = await repo.delete_by_age("coll", max_age_days=50.0)
        assert removed == 1
        assert (await repo.get_by_id("coll", old_record.record_id)) is None
        assert (await repo.get_by_id("coll", new_record.record_id)) is not None

    async def test_no_matches_returns_zero(self, repo):
        await repo.ensure_collection("coll", _DIMENSION)
        record = _make_record()
        await repo.store("coll", record)

        removed = await repo.delete_by_age("coll", max_age_days=1.0)
        assert removed == 0

    async def test_nonexistent_collection_returns_zero(self, repo):
        removed = await repo.delete_by_age("no_coll", max_age_days=1.0)
        assert removed == 0

    async def test_removes_all_stale(self, repo):
        await repo.ensure_collection("coll", _DIMENSION)
        for i in range(5):
            r = _make_record(content=f"stale-{i}")
            r.accessed_at = datetime.now() - timedelta(days=200)
            await repo.store("coll", r)

        removed = await repo.delete_by_age("coll", max_age_days=90.0)
        assert removed == 5
        assert (await repo.count("coll")) == 0

    async def test_preserves_fresh_records(self, repo):
        await repo.ensure_collection("coll", _DIMENSION)

        fresh = _make_record(content="fresh")
        await repo.store("coll", fresh)

        stale = _make_record(content="stale")
        stale.accessed_at = datetime.now() - timedelta(days=60)
        await repo.store("coll", stale)

        removed = await repo.delete_by_age("coll", max_age_days=30.0)
        assert removed == 1
        assert (await repo.get_by_id("coll", fresh.record_id)) is not None


# =========================================================================
# collection_stats
# =========================================================================


@_async_mark
class TestCollectionStats:
    async def test_returns_correct_counts(self, repo):
        await repo.ensure_collection("coll", _DIMENSION)
        r1 = _make_record(content="first")
        r2 = _make_record(content="second")
        await repo.store("coll", r1)
        await repo.store("coll", r2)

        stats = await repo.collection_stats("coll")
        assert stats is not None
        assert stats["collection_name"] == "coll"
        assert stats["record_count"] == 2
        assert stats["dimension"] == _DIMENSION

    async def test_empty_collection(self, repo):
        await repo.ensure_collection("empty", _DIMENSION)
        stats = await repo.collection_stats("empty")
        assert stats is not None
        assert stats["record_count"] == 0

    async def test_nonexistent_collection_returns_none(self, repo):
        stats = await repo.collection_stats("no_such_coll")
        assert stats is None


# =========================================================================
# count
# =========================================================================


@_async_mark
class TestCount:
    async def test_count_global_empty(self, repo):
        count = await repo.count()
        assert count == 0

    async def test_count_specific_collection(self, repo):
        await repo.ensure_collection("coll_a", _DIMENSION)
        await repo.ensure_collection("coll_b", _DIMENSION)

        await repo.store("coll_a", _make_record())
        await repo.store("coll_b", _make_record())
        await repo.store("coll_b", _make_record())

        assert (await repo.count("coll_a")) == 1
        assert (await repo.count("coll_b")) == 2

    async def test_count_global_sums_all(self, repo):
        await repo.ensure_collection("coll_a", _DIMENSION)
        await repo.ensure_collection("coll_b", _DIMENSION)

        await repo.store("coll_a", _make_record(content="a1"))
        await repo.store("coll_a", _make_record(content="a2"))
        await repo.store("coll_b", _make_record(content="b1"))

        total = await repo.count()
        assert total == 3

    async def test_count_nonexistent_collection(self, repo):
        assert (await repo.count("no_coll")) == 0

    async def test_count_after_delete(self, repo):
        await repo.ensure_collection("coll", _DIMENSION)
        r = _make_record()
        await repo.store("coll", r)
        assert (await repo.count("coll")) == 1

        await repo.delete("coll", r.record_id)
        assert (await repo.count("coll")) == 0
        assert (await repo.count()) == 0


# =========================================================================
# close + reuse
# =========================================================================


@_async_mark
class TestClose:
    async def test_close_and_reopen(self, db_path):
        """After close, a new instance should read persisted data."""
        from robotmcp.domains.memory.adapters.sqlite_vec_adapter import (
            SqliteVecRepository,
        )

        r1 = SqliteVecRepository(db_path)
        try:
            await r1.ensure_collection("coll", _DIMENSION)
            record = _make_record(content="persisted")
            await r1.store("coll", record)
        finally:
            r1.close()

        r2 = SqliteVecRepository(db_path)
        try:
            retrieved = await r2.get_by_id("coll", record.record_id)
            assert retrieved is not None
            assert retrieved.entry.content == "persisted"
        finally:
            r2.close()


# =========================================================================
# _sanitize_collection_name helper
# =========================================================================


class TestSanitizeCollectionName:
    def test_basic_alphanumeric(self):
        from robotmcp.domains.memory.adapters.sqlite_vec_adapter import (
            _sanitize_collection_name,
        )

        assert _sanitize_collection_name("hello_world") == "hello_world"

    def test_special_characters_replaced(self):
        from robotmcp.domains.memory.adapters.sqlite_vec_adapter import (
            _sanitize_collection_name,
        )

        assert _sanitize_collection_name("my-coll.v2") == "my_coll_v2"

    def test_leading_digit_prefixed(self):
        from robotmcp.domains.memory.adapters.sqlite_vec_adapter import (
            _sanitize_collection_name,
        )

        result = _sanitize_collection_name("123abc")
        assert result.startswith("c_")
        assert not result[0].isdigit()

    def test_empty_string(self):
        from robotmcp.domains.memory.adapters.sqlite_vec_adapter import (
            _sanitize_collection_name,
        )

        result = _sanitize_collection_name("")
        assert result.startswith("c_")


# =========================================================================
# _pack_vector helper
# =========================================================================


class TestPackVector:
    def test_roundtrip_pack(self):
        import struct

        from robotmcp.domains.memory.adapters.sqlite_vec_adapter import _pack_vector

        values = (1.0, 2.0, 3.0, 4.0)
        packed = _pack_vector(values)
        unpacked = struct.unpack(f"{len(values)}f", packed)
        assert unpacked == pytest.approx(values)

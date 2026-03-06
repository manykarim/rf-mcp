"""Tests for memory domain entities.

Covers: MemoryRecord, MemoryCollection.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from robotmcp.domains.memory.entities import MemoryCollection, MemoryRecord
from robotmcp.domains.memory.value_objects import (
    EmbeddingVector,
    MemoryEntry,
    MemoryType,
)


# =========================================================================
# MemoryRecord
# =========================================================================


class TestMemoryRecord:
    def _make_entry(self, **kwargs):
        defaults = {
            "content": "test content",
            "memory_type": MemoryType.keywords(),
        }
        defaults.update(kwargs)
        return MemoryEntry(**defaults)

    # -- create() ------------------------------------------------------------

    def test_create_generates_uuid(self):
        entry = self._make_entry()
        record = MemoryRecord.create(entry)
        assert record.record_id is not None
        assert len(record.record_id) == 36  # UUID format

    def test_create_unique_ids(self):
        entry = self._make_entry()
        r1 = MemoryRecord.create(entry)
        r2 = MemoryRecord.create(entry)
        assert r1.record_id != r2.record_id

    def test_create_sets_entry(self):
        entry = self._make_entry()
        record = MemoryRecord.create(entry)
        assert record.entry is entry

    def test_create_sets_timestamps(self):
        before = datetime.now()
        entry = self._make_entry()
        record = MemoryRecord.create(entry)
        after = datetime.now()
        assert before <= record.created_at <= after
        assert before <= record.accessed_at <= after

    def test_create_access_count_zero(self):
        record = MemoryRecord.create(self._make_entry())
        assert record.access_count == 0

    def test_create_session_id_none(self):
        record = MemoryRecord.create(self._make_entry())
        assert record.session_id is None

    def test_create_with_session_id(self):
        record = MemoryRecord.create(self._make_entry(), session_id="sess-42")
        assert record.session_id == "sess-42"

    # -- record_access() -----------------------------------------------------

    def test_record_access_increments_count(self):
        record = MemoryRecord.create(self._make_entry())
        assert record.access_count == 0
        record.record_access()
        assert record.access_count == 1
        record.record_access()
        assert record.access_count == 2

    def test_record_access_updates_accessed_at(self):
        record = MemoryRecord.create(self._make_entry())
        old_accessed = record.accessed_at
        # Ensure a small time difference
        record.record_access()
        assert record.accessed_at >= old_accessed

    # -- age_days ------------------------------------------------------------

    def test_age_days_recent(self):
        record = MemoryRecord.create(self._make_entry())
        # Just created, age should be very small
        assert record.age_days < 0.01

    def test_age_days_old_record(self):
        entry = self._make_entry()
        record = MemoryRecord.create(entry)
        record.created_at = datetime.now() - timedelta(days=10)
        assert 9.9 < record.age_days < 10.1

    # -- days_since_access ---------------------------------------------------

    def test_days_since_access_recent(self):
        record = MemoryRecord.create(self._make_entry())
        assert record.days_since_access < 0.01

    def test_days_since_access_old(self):
        record = MemoryRecord.create(self._make_entry())
        record.accessed_at = datetime.now() - timedelta(days=5)
        assert 4.9 < record.days_since_access < 5.1

    # -- is_stale() ----------------------------------------------------------

    def test_is_stale_false_when_recent(self):
        record = MemoryRecord.create(self._make_entry())
        assert record.is_stale(max_age_days=1.0) is False

    def test_is_stale_true_when_old(self):
        record = MemoryRecord.create(self._make_entry())
        record.accessed_at = datetime.now() - timedelta(days=100)
        assert record.is_stale(max_age_days=90.0) is True

    def test_is_stale_boundary(self):
        record = MemoryRecord.create(self._make_entry())
        record.accessed_at = datetime.now() - timedelta(days=90, seconds=1)
        assert record.is_stale(max_age_days=90.0) is True

    # -- to_dict() -----------------------------------------------------------

    def test_to_dict_basic(self):
        entry = self._make_entry(metadata={"key": "val"}, tags=("t1", "t2"))
        record = MemoryRecord.create(entry, session_id="sess-1")
        d = record.to_dict()

        assert d["record_id"] == record.record_id
        assert d["content"] == "test content"
        assert d["memory_type"] == "keywords"
        assert d["metadata"] == {"key": "val"}
        assert d["tags"] == ["t1", "t2"]
        assert "created_at" in d
        assert "accessed_at" in d
        assert d["access_count"] == 0
        assert d["session_id"] == "sess-1"

    def test_to_dict_no_session(self):
        record = MemoryRecord.create(self._make_entry())
        d = record.to_dict()
        assert "session_id" not in d

    def test_to_dict_empty_metadata(self):
        record = MemoryRecord.create(self._make_entry())
        d = record.to_dict()
        assert d["metadata"] == {}

    def test_to_dict_empty_tags(self):
        record = MemoryRecord.create(self._make_entry())
        d = record.to_dict()
        assert d["tags"] == []

    # -- from_dict() ---------------------------------------------------------

    def test_from_dict_roundtrip(self):
        entry = self._make_entry(metadata={"x": 1}, tags=("a", "b"))
        original = MemoryRecord.create(entry, session_id="s1")
        d = original.to_dict()
        restored = MemoryRecord.from_dict(d)

        assert restored.record_id == original.record_id
        assert restored.entry.content == original.entry.content
        assert restored.entry.memory_type == original.entry.memory_type
        assert restored.entry.metadata == original.entry.metadata
        assert restored.entry.tags == original.entry.tags
        assert restored.session_id == original.session_id
        assert restored.access_count == original.access_count

    def test_from_dict_minimal(self):
        d = {
            "record_id": "abc-123",
            "content": "hello",
            "memory_type": "documentation",
            "created_at": "2025-01-15T10:30:00",
            "accessed_at": "2025-01-15T10:30:00",
        }
        record = MemoryRecord.from_dict(d)
        assert record.record_id == "abc-123"
        assert record.entry.content == "hello"
        assert record.entry.memory_type.value == "documentation"
        assert record.access_count == 0
        assert record.session_id is None

    def test_from_dict_with_all_fields(self):
        d = {
            "record_id": "r1",
            "content": "c",
            "memory_type": "keywords",
            "metadata": {"source": "test"},
            "tags": ["tag1"],
            "created_at": "2025-06-01T12:00:00",
            "accessed_at": "2025-06-02T12:00:00",
            "access_count": 5,
            "session_id": "sess-x",
        }
        record = MemoryRecord.from_dict(d)
        assert record.entry.metadata == {"source": "test"}
        assert record.entry.tags == ("tag1",)
        assert record.access_count == 5
        assert record.session_id == "sess-x"

    # -- Equality and hash ---------------------------------------------------

    def test_equality_same_id(self):
        entry = self._make_entry()
        r1 = MemoryRecord.create(entry)
        r2 = MemoryRecord(
            record_id=r1.record_id,
            entry=self._make_entry(content="different"),
            created_at=datetime.now(),
            accessed_at=datetime.now(),
        )
        assert r1 == r2

    def test_inequality_different_id(self):
        r1 = MemoryRecord.create(self._make_entry())
        r2 = MemoryRecord.create(self._make_entry())
        assert r1 != r2

    def test_equality_not_implemented_for_other_types(self):
        r = MemoryRecord.create(self._make_entry())
        result = r.__eq__("not a record")
        assert result is NotImplemented

    def test_hash_same_id(self):
        entry = self._make_entry()
        r1 = MemoryRecord.create(entry)
        r2 = MemoryRecord(
            record_id=r1.record_id,
            entry=entry,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
        )
        assert hash(r1) == hash(r2)

    def test_hash_can_be_used_in_set(self):
        r1 = MemoryRecord.create(self._make_entry())
        r2 = MemoryRecord.create(self._make_entry())
        s = {r1, r2}
        assert len(s) == 2

    def test_hash_deduplicates_same_id(self):
        r = MemoryRecord.create(self._make_entry())
        r_copy = MemoryRecord(
            record_id=r.record_id,
            entry=self._make_entry(),
            created_at=datetime.now(),
            accessed_at=datetime.now(),
        )
        s = {r, r_copy}
        assert len(s) == 1

    # -- __test__ suppression ------------------------------------------------

    def test_test_flag_suppression(self):
        assert MemoryRecord.__test__ is False


# =========================================================================
# MemoryCollection
# =========================================================================


class TestMemoryCollection:
    # -- for_type() ----------------------------------------------------------

    def test_for_type_default_dimension(self):
        mt = MemoryType.keywords()
        coll = MemoryCollection.for_type(mt)
        assert coll.collection_id == "rfmcp_keywords"
        assert coll.memory_type == mt
        assert coll.dimension == 256
        assert coll.record_count == 0
        assert coll.last_updated_at is None

    def test_for_type_custom_dimension(self):
        mt = MemoryType.documentation()
        coll = MemoryCollection.for_type(mt, dimension=384)
        assert coll.dimension == 384

    def test_for_type_different_types(self):
        for mt in MemoryType.all_types():
            coll = MemoryCollection.for_type(mt)
            assert coll.collection_id == mt.collection_name
            assert coll.memory_type == mt

    # -- increment_count / decrement_count -----------------------------------

    def test_increment_count_default(self):
        coll = MemoryCollection.for_type(MemoryType.keywords())
        coll.increment_count()
        assert coll.record_count == 1

    def test_increment_count_by_n(self):
        coll = MemoryCollection.for_type(MemoryType.keywords())
        coll.increment_count(5)
        assert coll.record_count == 5

    def test_increment_count_updates_last_updated(self):
        coll = MemoryCollection.for_type(MemoryType.keywords())
        assert coll.last_updated_at is None
        coll.increment_count()
        assert coll.last_updated_at is not None

    def test_decrement_count_default(self):
        coll = MemoryCollection.for_type(MemoryType.keywords())
        coll.increment_count(3)
        coll.decrement_count()
        assert coll.record_count == 2

    def test_decrement_count_by_n(self):
        coll = MemoryCollection.for_type(MemoryType.keywords())
        coll.increment_count(10)
        coll.decrement_count(4)
        assert coll.record_count == 6

    def test_decrement_count_floor_at_zero(self):
        coll = MemoryCollection.for_type(MemoryType.keywords())
        coll.decrement_count(5)
        assert coll.record_count == 0

    def test_decrement_count_updates_last_updated(self):
        coll = MemoryCollection.for_type(MemoryType.keywords())
        coll.decrement_count()
        assert coll.last_updated_at is not None

    # -- is_empty ------------------------------------------------------------

    def test_is_empty_true(self):
        coll = MemoryCollection.for_type(MemoryType.keywords())
        assert coll.is_empty is True

    def test_is_empty_false(self):
        coll = MemoryCollection.for_type(MemoryType.keywords())
        coll.increment_count()
        assert coll.is_empty is False

    # -- to_dict() -----------------------------------------------------------

    def test_to_dict(self):
        coll = MemoryCollection.for_type(MemoryType.keywords())
        coll.increment_count(3)
        d = coll.to_dict()

        assert d["collection_id"] == "rfmcp_keywords"
        assert d["memory_type"] == "keywords"
        assert d["dimension"] == 256
        assert d["record_count"] == 3
        assert "created_at" in d
        assert d["last_updated_at"] is not None

    def test_to_dict_no_updates(self):
        coll = MemoryCollection.for_type(MemoryType.documentation())
        d = coll.to_dict()
        assert d["last_updated_at"] is None

    # -- __test__ suppression ------------------------------------------------

    def test_test_flag_suppression(self):
        assert MemoryCollection.__test__ is False

"""Tests for memory domain events.

Covers: MemoryStored, MemoryRecalled, MemoryDecayed, CollectionCreated, MemoryPruned.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from robotmcp.domains.memory.events import (
    CollectionCreated,
    MemoryDecayed,
    MemoryPruned,
    MemoryRecalled,
    MemoryStored,
)


# =========================================================================
# MemoryStored
# =========================================================================


class TestMemoryStored:
    def test_creation(self):
        ev = MemoryStored(
            record_id="r1",
            memory_type="keywords",
            content_preview="hello world",
            collection_id="rfmcp_keywords",
        )
        assert ev.record_id == "r1"
        assert ev.memory_type == "keywords"
        assert ev.content_preview == "hello world"
        assert ev.collection_id == "rfmcp_keywords"
        assert ev.session_id is None
        assert isinstance(ev.timestamp, datetime)

    def test_creation_with_session(self):
        ev = MemoryStored(
            record_id="r1",
            memory_type="keywords",
            content_preview="hello",
            collection_id="c1",
            session_id="sess-1",
        )
        assert ev.session_id == "sess-1"

    def test_frozen(self):
        ev = MemoryStored(
            record_id="r1",
            memory_type="keywords",
            content_preview="hello",
            collection_id="c1",
        )
        with pytest.raises(AttributeError):
            ev.record_id = "other"  # type: ignore[misc]

    def test_to_dict(self):
        ev = MemoryStored(
            record_id="r1",
            memory_type="keywords",
            content_preview="hello world test",
            collection_id="rfmcp_keywords",
            session_id="s1",
        )
        d = ev.to_dict()
        assert d["event_type"] == "MemoryStored"
        assert d["record_id"] == "r1"
        assert d["memory_type"] == "keywords"
        assert d["content_preview"] == "hello world test"
        assert d["collection_id"] == "rfmcp_keywords"
        assert d["session_id"] == "s1"
        assert "timestamp" in d

    def test_to_dict_truncates_content_preview(self):
        long_preview = "x" * 200
        ev = MemoryStored(
            record_id="r1",
            memory_type="keywords",
            content_preview=long_preview,
            collection_id="c1",
        )
        d = ev.to_dict()
        assert len(d["content_preview"]) == 100

    def test_to_dict_no_session(self):
        ev = MemoryStored(
            record_id="r1",
            memory_type="keywords",
            content_preview="hello",
            collection_id="c1",
        )
        d = ev.to_dict()
        assert d["session_id"] is None


# =========================================================================
# MemoryRecalled
# =========================================================================


class TestMemoryRecalled:
    def test_creation(self):
        ev = MemoryRecalled(
            query_text="test query",
            memory_type="keywords",
            result_count=5,
            top_similarity=0.9,
            query_time_ms=150.0,
        )
        assert ev.query_text == "test query"
        assert ev.memory_type == "keywords"
        assert ev.result_count == 5
        assert ev.top_similarity == 0.9
        assert ev.query_time_ms == 150.0
        assert ev.session_id is None

    def test_creation_with_session(self):
        ev = MemoryRecalled(
            query_text="q",
            memory_type=None,
            result_count=0,
            top_similarity=0.0,
            query_time_ms=10.0,
            session_id="s1",
        )
        assert ev.session_id == "s1"

    def test_creation_no_memory_type(self):
        ev = MemoryRecalled(
            query_text="q",
            memory_type=None,
            result_count=0,
            top_similarity=0.0,
            query_time_ms=10.0,
        )
        assert ev.memory_type is None

    def test_frozen(self):
        ev = MemoryRecalled(
            query_text="q",
            memory_type="keywords",
            result_count=0,
            top_similarity=0.0,
            query_time_ms=10.0,
        )
        with pytest.raises(AttributeError):
            ev.result_count = 5  # type: ignore[misc]

    # -- Properties ----------------------------------------------------------

    def test_is_empty_result_true(self):
        ev = MemoryRecalled(
            query_text="q",
            memory_type=None,
            result_count=0,
            top_similarity=0.0,
            query_time_ms=10.0,
        )
        assert ev.is_empty_result is True

    def test_is_empty_result_false(self):
        ev = MemoryRecalled(
            query_text="q",
            memory_type=None,
            result_count=1,
            top_similarity=0.5,
            query_time_ms=10.0,
        )
        assert ev.is_empty_result is False

    def test_is_slow_query_true(self):
        ev = MemoryRecalled(
            query_text="q",
            memory_type=None,
            result_count=0,
            top_similarity=0.0,
            query_time_ms=501.0,
        )
        assert ev.is_slow_query is True

    def test_is_slow_query_false(self):
        ev = MemoryRecalled(
            query_text="q",
            memory_type=None,
            result_count=0,
            top_similarity=0.0,
            query_time_ms=500.0,
        )
        assert ev.is_slow_query is False

    def test_is_slow_query_boundary(self):
        ev = MemoryRecalled(
            query_text="q",
            memory_type=None,
            result_count=0,
            top_similarity=0.0,
            query_time_ms=499.9,
        )
        assert ev.is_slow_query is False

    # -- to_dict -------------------------------------------------------------

    def test_to_dict(self):
        ev = MemoryRecalled(
            query_text="test query",
            memory_type="keywords",
            result_count=3,
            top_similarity=0.8567,
            query_time_ms=123.456,
            session_id="s1",
        )
        d = ev.to_dict()
        assert d["event_type"] == "MemoryRecalled"
        assert d["query_text"] == "test query"
        assert d["memory_type"] == "keywords"
        assert d["result_count"] == 3
        assert d["top_similarity"] == 0.8567
        assert d["query_time_ms"] == 123.46  # Rounded to 2dp
        assert d["session_id"] == "s1"
        assert "timestamp" in d

    def test_to_dict_truncates_query_text(self):
        long_query = "q" * 300
        ev = MemoryRecalled(
            query_text=long_query,
            memory_type=None,
            result_count=0,
            top_similarity=0.0,
            query_time_ms=10.0,
        )
        d = ev.to_dict()
        assert len(d["query_text"]) == 200


# =========================================================================
# MemoryDecayed
# =========================================================================


class TestMemoryDecayed:
    def test_creation(self):
        ev = MemoryDecayed(
            record_id="r1",
            original_similarity=0.9,
            decayed_similarity=0.7,
            age_days=30.5,
        )
        assert ev.record_id == "r1"
        assert ev.original_similarity == 0.9
        assert ev.decayed_similarity == 0.7
        assert ev.age_days == 30.5

    def test_frozen(self):
        ev = MemoryDecayed(
            record_id="r1",
            original_similarity=0.9,
            decayed_similarity=0.7,
            age_days=30.0,
        )
        with pytest.raises(AttributeError):
            ev.age_days = 60.0  # type: ignore[misc]

    # -- decay_ratio ---------------------------------------------------------

    def test_decay_ratio(self):
        ev = MemoryDecayed(
            record_id="r1",
            original_similarity=0.8,
            decayed_similarity=0.4,
            age_days=30.0,
        )
        assert ev.decay_ratio == pytest.approx(0.5)

    def test_decay_ratio_no_decay(self):
        ev = MemoryDecayed(
            record_id="r1",
            original_similarity=0.8,
            decayed_similarity=0.8,
            age_days=0.0,
        )
        assert ev.decay_ratio == pytest.approx(1.0)

    def test_decay_ratio_zero_original(self):
        ev = MemoryDecayed(
            record_id="r1",
            original_similarity=0.0,
            decayed_similarity=0.0,
            age_days=30.0,
        )
        assert ev.decay_ratio == 0.0

    # -- to_dict -------------------------------------------------------------

    def test_to_dict(self):
        ev = MemoryDecayed(
            record_id="r1",
            original_similarity=0.85678,
            decayed_similarity=0.62349,
            age_days=15.789,
        )
        d = ev.to_dict()
        assert d["event_type"] == "MemoryDecayed"
        assert d["record_id"] == "r1"
        assert d["original_similarity"] == round(0.85678, 4)
        assert d["decayed_similarity"] == round(0.62349, 4)
        assert d["age_days"] == 15.8  # Rounded to 1dp
        assert "decay_ratio" in d
        assert "timestamp" in d


# =========================================================================
# CollectionCreated
# =========================================================================


class TestCollectionCreated:
    def test_creation(self):
        ev = CollectionCreated(
            collection_id="rfmcp_keywords",
            memory_type="keywords",
            dimension=256,
        )
        assert ev.collection_id == "rfmcp_keywords"
        assert ev.memory_type == "keywords"
        assert ev.dimension == 256
        assert isinstance(ev.timestamp, datetime)

    def test_frozen(self):
        ev = CollectionCreated(
            collection_id="c1",
            memory_type="keywords",
            dimension=256,
        )
        with pytest.raises(AttributeError):
            ev.dimension = 384  # type: ignore[misc]

    def test_to_dict(self):
        ev = CollectionCreated(
            collection_id="rfmcp_documentation",
            memory_type="documentation",
            dimension=384,
        )
        d = ev.to_dict()
        assert d["event_type"] == "CollectionCreated"
        assert d["collection_id"] == "rfmcp_documentation"
        assert d["memory_type"] == "documentation"
        assert d["dimension"] == 384
        assert "timestamp" in d


# =========================================================================
# MemoryPruned
# =========================================================================


class TestMemoryPruned:
    def test_creation(self):
        ev = MemoryPruned(
            collection_id="rfmcp_keywords",
            memory_type="keywords",
            records_removed=10,
            max_age_days=90.0,
        )
        assert ev.collection_id == "rfmcp_keywords"
        assert ev.memory_type == "keywords"
        assert ev.records_removed == 10
        assert ev.max_age_days == 90.0
        assert isinstance(ev.timestamp, datetime)

    def test_frozen(self):
        ev = MemoryPruned(
            collection_id="c1",
            memory_type="keywords",
            records_removed=0,
            max_age_days=30.0,
        )
        with pytest.raises(AttributeError):
            ev.records_removed = 5  # type: ignore[misc]

    def test_to_dict(self):
        ev = MemoryPruned(
            collection_id="rfmcp_common_errors",
            memory_type="common_errors",
            records_removed=42,
            max_age_days=60.0,
        )
        d = ev.to_dict()
        assert d["event_type"] == "MemoryPruned"
        assert d["collection_id"] == "rfmcp_common_errors"
        assert d["memory_type"] == "common_errors"
        assert d["records_removed"] == 42
        assert d["max_age_days"] == 60.0
        assert "timestamp" in d

    def test_to_dict_zero_removed(self):
        ev = MemoryPruned(
            collection_id="c1",
            memory_type="keywords",
            records_removed=0,
            max_age_days=90.0,
        )
        d = ev.to_dict()
        assert d["records_removed"] == 0

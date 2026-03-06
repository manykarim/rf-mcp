"""Tests for Delta State Retrieval Service (ADR-018)."""
from __future__ import annotations

__test__ = True

import pytest

from robotmcp.domains.snapshot.delta_service import DeltaStateService
from robotmcp.domains.snapshot.delta_aggregates import VersionedStateCache
from robotmcp.domains.snapshot.delta_value_objects import (
    SectionChangeType,
    StateDelta,
    StateVersion,
)


# ---------------------------------------------------------------------------
# get_or_create_cache
# ---------------------------------------------------------------------------
class TestGetOrCreateCache:
    def test_creates_new_cache(self):
        service = DeltaStateService()
        cache = service.get_or_create_cache("sess1")
        assert isinstance(cache, VersionedStateCache)
        assert cache.session_id == "sess1"

    def test_returns_existing_cache(self):
        service = DeltaStateService()
        cache1 = service.get_or_create_cache("sess1")
        cache2 = service.get_or_create_cache("sess1")
        assert cache1 is cache2

    def test_different_sessions_get_different_caches(self):
        service = DeltaStateService()
        cache1 = service.get_or_create_cache("sess1")
        cache2 = service.get_or_create_cache("sess2")
        assert cache1 is not cache2
        assert cache1.session_id == "sess1"
        assert cache2.session_id == "sess2"

    def test_cache_inherits_service_config(self):
        service = DeltaStateService(
            max_versions_per_session=10,
            ttl_seconds=120.0,
        )
        cache = service.get_or_create_cache("sess1")
        assert cache.max_versions == 10
        assert cache.ttl_seconds == 120.0

    def test_access_moves_to_end_lru(self):
        service = DeltaStateService(max_sessions=3)
        service.get_or_create_cache("s1")
        service.get_or_create_cache("s2")
        service.get_or_create_cache("s3")
        # Access s1 to move it to end (most recently used)
        service.get_or_create_cache("s1")
        # Now create s4, which should evict s2 (oldest unused)
        service.get_or_create_cache("s4")
        assert service.active_sessions == 3
        # s2 should have been evicted
        assert "s2" not in service._caches
        # s1, s3, s4 should remain
        assert "s1" in service._caches
        assert "s3" in service._caches
        assert "s4" in service._caches


# ---------------------------------------------------------------------------
# record_full_state
# ---------------------------------------------------------------------------
class TestRecordFullState:
    def test_filters_sections(self):
        service = DeltaStateService()
        sections = {"a": 1, "b": 2, "c": 3}
        section_names = ["a", "c"]
        v = service.record_full_state("sess1", sections, section_names)
        assert isinstance(v, StateVersion)
        assert v.session_id == "sess1"
        assert v.version_number == 0
        # Verify only filtered sections were stored
        cache = service.get_or_create_cache("sess1")
        snap = cache.get_version(0)
        assert snap is not None
        assert set(snap.section_data.keys()) == {"a", "c"}

    def test_filters_nonexistent_section_names(self):
        service = DeltaStateService()
        sections = {"a": 1, "b": 2}
        section_names = ["a", "z"]  # z not in sections
        v = service.record_full_state("sess1", sections, section_names)
        cache = service.get_or_create_cache("sess1")
        snap = cache.get_version(v.version_number)
        assert snap is not None
        assert set(snap.section_data.keys()) == {"a"}

    def test_empty_section_names_stores_nothing(self):
        service = DeltaStateService()
        sections = {"a": 1, "b": 2}
        v = service.record_full_state("sess1", sections, [])
        cache = service.get_or_create_cache("sess1")
        snap = cache.get_version(v.version_number)
        assert snap is not None
        assert snap.section_data == {}

    def test_increments_version(self):
        service = DeltaStateService()
        v0 = service.record_full_state("sess1", {"a": 1}, ["a"])
        v1 = service.record_full_state("sess1", {"a": 2}, ["a"])
        assert v0.version_number == 0
        assert v1.version_number == 1


# ---------------------------------------------------------------------------
# compute_delta
# ---------------------------------------------------------------------------
class TestComputeDelta:
    def test_returns_delta_and_version(self):
        service = DeltaStateService()
        v0 = service.record_full_state("sess1", {"a": 1, "b": 2}, ["a", "b"])
        delta, v1 = service.compute_delta(
            "sess1", v0.version_number, {"a": 1, "b": 99}, ["a", "b"]
        )
        assert isinstance(delta, StateDelta)
        assert isinstance(v1, StateVersion)
        assert v1.version_number == v0.version_number + 1

    def test_delta_detects_modification(self):
        service = DeltaStateService()
        v0 = service.record_full_state("sess1", {"a": 1, "b": 2}, ["a", "b"])
        delta, _ = service.compute_delta(
            "sess1", v0.version_number, {"a": 1, "b": 99}, ["a", "b"]
        )
        assert delta.has_changes is True
        modified = [
            s for s in delta.changed_sections
            if s.change_type == SectionChangeType.MODIFIED
        ]
        assert len(modified) == 1
        assert modified[0].section_name == "b"

    def test_delta_filters_sections(self):
        service = DeltaStateService()
        v0 = service.record_full_state("sess1", {"a": 1, "b": 2}, ["a", "b"])
        # Only request "a" in delta -- "b" was in v0 but not in filtered current
        delta, _ = service.compute_delta(
            "sess1", v0.version_number, {"a": 99, "b": 2}, ["a"]
        )
        changed_names = {s.section_name for s in delta.changed_sections}
        # "a" should be MODIFIED (value changed), "b" should be REMOVED (not in filtered)
        assert "a" in changed_names
        assert "b" in changed_names

    def test_delta_no_changes(self):
        service = DeltaStateService()
        v0 = service.record_full_state("sess1", {"a": 1}, ["a"])
        delta, _ = service.compute_delta(
            "sess1", v0.version_number, {"a": 1}, ["a"]
        )
        assert delta.has_changes is False
        assert delta.change_count == 0

    def test_delta_on_new_session(self):
        service = DeltaStateService()
        # No prior state -- compute_delta creates v0, then looks up since_version=0
        # which is the version just stored, so it finds it.
        delta, v = service.compute_delta(
            "new_sess", 0, {"a": 1, "b": 2}, ["a", "b"]
        )
        assert isinstance(delta, StateDelta)
        assert isinstance(v, StateVersion)


# ---------------------------------------------------------------------------
# get_current_version
# ---------------------------------------------------------------------------
class TestGetCurrentVersion:
    def test_unknown_session_returns_zero(self):
        service = DeltaStateService()
        assert service.get_current_version("unknown") == 0

    def test_after_record(self):
        service = DeltaStateService()
        service.record_full_state("sess1", {"a": 1}, ["a"])
        assert service.get_current_version("sess1") == 0
        service.record_full_state("sess1", {"a": 2}, ["a"])
        assert service.get_current_version("sess1") == 1

    def test_after_compute_delta(self):
        service = DeltaStateService()
        v0 = service.record_full_state("sess1", {"a": 1}, ["a"])
        service.compute_delta("sess1", v0.version_number, {"a": 2}, ["a"])
        assert service.get_current_version("sess1") == 1


# ---------------------------------------------------------------------------
# clear_session
# ---------------------------------------------------------------------------
class TestClearSession:
    def test_clear_existing_session(self):
        service = DeltaStateService()
        service.record_full_state("sess1", {"a": 1}, ["a"])
        assert service.active_sessions == 1
        service.clear_session("sess1")
        assert service.active_sessions == 0

    def test_clear_nonexistent_session(self):
        service = DeltaStateService()
        # Should not raise
        service.clear_session("nonexistent")
        assert service.active_sessions == 0

    def test_clear_does_not_affect_other_sessions(self):
        service = DeltaStateService()
        service.record_full_state("sess1", {"a": 1}, ["a"])
        service.record_full_state("sess2", {"b": 2}, ["b"])
        service.clear_session("sess1")
        assert service.active_sessions == 1
        assert service.get_current_version("sess2") == 0


# ---------------------------------------------------------------------------
# clear_all
# ---------------------------------------------------------------------------
class TestClearAll:
    def test_clears_all_sessions(self):
        service = DeltaStateService()
        service.record_full_state("sess1", {"a": 1}, ["a"])
        service.record_full_state("sess2", {"b": 2}, ["b"])
        service.record_full_state("sess3", {"c": 3}, ["c"])
        assert service.active_sessions == 3
        service.clear_all()
        assert service.active_sessions == 0

    def test_clear_all_on_empty_service(self):
        service = DeltaStateService()
        service.clear_all()
        assert service.active_sessions == 0


# ---------------------------------------------------------------------------
# active_sessions property
# ---------------------------------------------------------------------------
class TestActiveSessions:
    def test_initially_zero(self):
        service = DeltaStateService()
        assert service.active_sessions == 0

    def test_increments_with_new_sessions(self):
        service = DeltaStateService()
        service.get_or_create_cache("s1")
        assert service.active_sessions == 1
        service.get_or_create_cache("s2")
        assert service.active_sessions == 2

    def test_same_session_does_not_increment(self):
        service = DeltaStateService()
        service.get_or_create_cache("s1")
        service.get_or_create_cache("s1")
        assert service.active_sessions == 1


# ---------------------------------------------------------------------------
# LRU session eviction
# ---------------------------------------------------------------------------
class TestSessionLRUEviction:
    def test_evicts_oldest_session(self):
        service = DeltaStateService(max_sessions=2)
        service.get_or_create_cache("s1")
        service.get_or_create_cache("s2")
        assert service.active_sessions == 2
        # Adding s3 should evict s1 (oldest)
        service.get_or_create_cache("s3")
        assert service.active_sessions == 2
        assert "s1" not in service._caches
        assert "s2" in service._caches
        assert "s3" in service._caches

    def test_eviction_preserves_data_of_remaining(self):
        service = DeltaStateService(max_sessions=2)
        service.record_full_state("s1", {"a": 1}, ["a"])
        service.record_full_state("s2", {"b": 2}, ["b"])
        # Evict s1 by adding s3
        service.get_or_create_cache("s3")
        # s2's data should still be intact
        assert service.get_current_version("s2") == 0
        cache_s2 = service.get_or_create_cache("s2")
        snap = cache_s2.get_version(0)
        assert snap is not None
        assert snap.section_data == {"b": 2}

    def test_max_sessions_one(self):
        service = DeltaStateService(max_sessions=1)
        service.get_or_create_cache("s1")
        service.get_or_create_cache("s2")
        assert service.active_sessions == 1
        assert "s2" in service._caches
        assert "s1" not in service._caches

    def test_access_refreshes_lru_order(self):
        service = DeltaStateService(max_sessions=2)
        service.get_or_create_cache("s1")
        service.get_or_create_cache("s2")
        # Access s1 so it becomes most recently used
        service.get_or_create_cache("s1")
        # Now add s3; s2 should be evicted (it's now the oldest)
        service.get_or_create_cache("s3")
        assert "s1" in service._caches
        assert "s2" not in service._caches
        assert "s3" in service._caches


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestServiceEdgeCases:
    def test_record_and_delta_on_same_session(self):
        service = DeltaStateService()
        v0 = service.record_full_state("s1", {"x": "old"}, ["x"])
        v1 = service.record_full_state("s1", {"x": "mid"}, ["x"])
        delta, v2 = service.compute_delta(
            "s1", v1.version_number, {"x": "new"}, ["x"]
        )
        assert delta.has_changes is True
        assert delta.change_count == 1
        assert v2.version_number == 2

    def test_many_sections(self):
        service = DeltaStateService()
        sections = {f"sec_{i}": f"val_{i}" for i in range(50)}
        names = list(sections.keys())
        v = service.record_full_state("s1", sections, names)
        assert v.version_number == 0
        cache = service.get_or_create_cache("s1")
        snap = cache.get_version(0)
        assert snap is not None
        assert len(snap.section_data) == 50

    def test_default_config_values(self):
        service = DeltaStateService()
        assert service.max_versions_per_session == 5
        assert service.ttl_seconds == 300.0
        assert service.max_sessions == 50

"""Tests for Delta State Retrieval Aggregates (ADR-018)."""
from __future__ import annotations

__test__ = True

import time
from datetime import datetime, timedelta

import pytest

from robotmcp.domains.snapshot.delta_aggregates import VersionedStateCache
from robotmcp.domains.snapshot.delta_value_objects import (
    SectionChangeType,
    StateDelta,
    StateVersion,
)
from robotmcp.domains.snapshot.delta_entities import VersionedSnapshot


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------
class TestVersionedStateCacheInit:
    def test_default_params(self):
        cache = VersionedStateCache(session_id="s1")
        assert cache.session_id == "s1"
        assert cache.max_versions == 5
        assert cache.ttl_seconds == 300.0

    def test_custom_params(self):
        cache = VersionedStateCache(
            session_id="s2", max_versions=3, ttl_seconds=60.0
        )
        assert cache.max_versions == 3
        assert cache.ttl_seconds == 60.0

    def test_max_versions_zero_raises(self):
        with pytest.raises(ValueError, match="max_versions must be >= 1"):
            VersionedStateCache(session_id="s", max_versions=0)

    def test_max_versions_negative_raises(self):
        with pytest.raises(ValueError, match="max_versions must be >= 1"):
            VersionedStateCache(session_id="s", max_versions=-1)

    def test_ttl_zero_raises(self):
        with pytest.raises(ValueError, match="ttl_seconds must be > 0"):
            VersionedStateCache(session_id="s", ttl_seconds=0)

    def test_ttl_negative_raises(self):
        with pytest.raises(ValueError, match="ttl_seconds must be > 0"):
            VersionedStateCache(session_id="s", ttl_seconds=-1.0)


# ---------------------------------------------------------------------------
# store_version
# ---------------------------------------------------------------------------
class TestStoreVersion:
    def test_first_version_is_zero(self):
        cache = VersionedStateCache(session_id="s1")
        v = cache.store_version({"a": 1})
        assert v.version_number == 0
        assert v.session_id == "s1"

    def test_increments_version(self):
        cache = VersionedStateCache(session_id="s1")
        v0 = cache.store_version({"a": 1})
        v1 = cache.store_version({"a": 2})
        v2 = cache.store_version({"a": 3})
        assert v0.version_number == 0
        assert v1.version_number == 1
        assert v2.version_number == 2

    def test_content_hash_changes(self):
        cache = VersionedStateCache(session_id="s1")
        v0 = cache.store_version({"a": 1})
        v1 = cache.store_version({"a": 2})
        assert v0.content_hash != v1.content_hash

    def test_content_hash_same_for_same_content(self):
        cache = VersionedStateCache(session_id="s1")
        v0 = cache.store_version({"a": 1})
        v1 = cache.store_version({"a": 1})
        assert v0.content_hash == v1.content_hash

    def test_lru_eviction(self):
        cache = VersionedStateCache(session_id="s1", max_versions=3)
        for i in range(5):
            cache.store_version({"a": i})
        # Only versions 2, 3, 4 should remain
        assert cache.get_version(0) is None
        assert cache.get_version(1) is None
        assert cache.get_version(2) is not None
        assert cache.get_version(3) is not None
        assert cache.get_version(4) is not None

    def test_lru_eviction_max_one(self):
        cache = VersionedStateCache(session_id="s1", max_versions=1)
        cache.store_version({"a": 0})
        cache.store_version({"a": 1})
        cache.store_version({"a": 2})
        assert cache.get_version(0) is None
        assert cache.get_version(1) is None
        assert cache.get_version(2) is not None

    def test_returns_state_version(self):
        cache = VersionedStateCache(session_id="s1")
        v = cache.store_version({"x": "hello"})
        assert isinstance(v, StateVersion)


# ---------------------------------------------------------------------------
# get_version
# ---------------------------------------------------------------------------
class TestGetVersion:
    def test_existing_version(self):
        cache = VersionedStateCache(session_id="s1")
        cache.store_version({"a": 1})
        snap = cache.get_version(0)
        assert snap is not None
        assert snap.version.version_number == 0

    def test_nonexistent_version(self):
        cache = VersionedStateCache(session_id="s1")
        assert cache.get_version(99) is None

    def test_ttl_expiry(self):
        cache = VersionedStateCache(session_id="s1", ttl_seconds=0.01)
        cache.store_version({"a": 1})
        time.sleep(0.02)
        assert cache.get_version(0) is None

    def test_ttl_not_expired(self):
        cache = VersionedStateCache(session_id="s1", ttl_seconds=60.0)
        cache.store_version({"a": 1})
        assert cache.get_version(0) is not None


# ---------------------------------------------------------------------------
# get_latest
# ---------------------------------------------------------------------------
class TestGetLatest:
    def test_empty_cache(self):
        cache = VersionedStateCache(session_id="s1")
        assert cache.get_latest() is None

    def test_single_version(self):
        cache = VersionedStateCache(session_id="s1")
        cache.store_version({"a": 1})
        latest = cache.get_latest()
        assert latest is not None
        assert latest.version.version_number == 0

    def test_multiple_versions(self):
        cache = VersionedStateCache(session_id="s1")
        cache.store_version({"a": 1})
        cache.store_version({"a": 2})
        cache.store_version({"a": 3})
        latest = cache.get_latest()
        assert latest is not None
        assert latest.version.version_number == 2


# ---------------------------------------------------------------------------
# get_current_version_number
# ---------------------------------------------------------------------------
class TestGetCurrentVersionNumber:
    def test_empty_cache(self):
        cache = VersionedStateCache(session_id="s1")
        assert cache.get_current_version_number() == 0

    def test_after_stores(self):
        cache = VersionedStateCache(session_id="s1")
        cache.store_version({"a": 1})
        assert cache.get_current_version_number() == 0
        cache.store_version({"a": 2})
        assert cache.get_current_version_number() == 1
        cache.store_version({"a": 3})
        assert cache.get_current_version_number() == 2

    def test_after_eviction(self):
        cache = VersionedStateCache(session_id="s1", max_versions=2)
        cache.store_version({"a": 0})
        cache.store_version({"a": 1})
        cache.store_version({"a": 2})
        # Version 0 evicted, but current is still 2
        assert cache.get_current_version_number() == 2


# ---------------------------------------------------------------------------
# compute_delta - all unchanged
# ---------------------------------------------------------------------------
class TestComputeDeltaAllUnchanged:
    def test_no_changes(self):
        cache = VersionedStateCache(session_id="s1")
        sections = {"a": {"x": 1}, "b": "hello"}
        v0 = cache.store_version(sections)
        delta, v1 = cache.compute_delta(v0.version_number, sections)
        assert delta.has_changes is False
        assert len(delta.unchanged_sections) == 2
        assert delta.change_count == 0
        assert delta.saved_tokens >= 0


# ---------------------------------------------------------------------------
# compute_delta - modified section
# ---------------------------------------------------------------------------
class TestComputeDeltaModified:
    def test_one_modified(self):
        cache = VersionedStateCache(session_id="s1")
        v0 = cache.store_version({"a": 1, "b": 2})
        delta, v1 = cache.compute_delta(v0.version_number, {"a": 1, "b": 99})
        assert delta.has_changes is True
        assert delta.change_count == 1
        changed_names = {s.section_name for s in delta.changed_sections}
        assert "b" in changed_names
        assert "a" in delta.unchanged_sections

    def test_modified_section_has_new_value(self):
        cache = VersionedStateCache(session_id="s1")
        v0 = cache.store_version({"a": "old"})
        delta, _ = cache.compute_delta(v0.version_number, {"a": "new"})
        assert delta.changed_sections[0].value == "new"
        assert delta.changed_sections[0].change_type == SectionChangeType.MODIFIED

    def test_all_modified(self):
        cache = VersionedStateCache(session_id="s1")
        v0 = cache.store_version({"a": 1, "b": 2})
        delta, _ = cache.compute_delta(v0.version_number, {"a": 99, "b": 99})
        assert delta.change_count == 2
        assert len(delta.unchanged_sections) == 0


# ---------------------------------------------------------------------------
# compute_delta - section added
# ---------------------------------------------------------------------------
class TestComputeDeltaSectionAdded:
    def test_new_section(self):
        cache = VersionedStateCache(session_id="s1")
        v0 = cache.store_version({"a": 1})
        delta, _ = cache.compute_delta(v0.version_number, {"a": 1, "b": 2})
        added = [s for s in delta.changed_sections if s.change_type == SectionChangeType.ADDED]
        assert len(added) == 1
        assert added[0].section_name == "b"
        assert added[0].value == 2

    def test_multiple_new_sections(self):
        cache = VersionedStateCache(session_id="s1")
        v0 = cache.store_version({"a": 1})
        delta, _ = cache.compute_delta(v0.version_number, {"a": 1, "b": 2, "c": 3})
        added = [s for s in delta.changed_sections if s.change_type == SectionChangeType.ADDED]
        assert len(added) == 2


# ---------------------------------------------------------------------------
# compute_delta - section removed
# ---------------------------------------------------------------------------
class TestComputeDeltaSectionRemoved:
    def test_removed_section(self):
        cache = VersionedStateCache(session_id="s1")
        v0 = cache.store_version({"a": 1, "b": 2})
        delta, _ = cache.compute_delta(v0.version_number, {"a": 1})
        removed = [s for s in delta.changed_sections if s.change_type == SectionChangeType.REMOVED]
        assert len(removed) == 1
        assert removed[0].section_name == "b"
        assert removed[0].value is None

    def test_all_removed(self):
        cache = VersionedStateCache(session_id="s1")
        v0 = cache.store_version({"a": 1, "b": 2})
        delta, _ = cache.compute_delta(v0.version_number, {})
        removed = [s for s in delta.changed_sections if s.change_type == SectionChangeType.REMOVED]
        assert len(removed) == 2


# ---------------------------------------------------------------------------
# compute_delta - expired since_version (fallback)
# ---------------------------------------------------------------------------
class TestComputeDeltaExpiredFallback:
    def test_expired_version_fallback(self):
        cache = VersionedStateCache(session_id="s1", ttl_seconds=0.01)
        cache.store_version({"a": 1})
        time.sleep(0.02)
        delta, new_v = cache.compute_delta(0, {"a": 1, "b": 2})
        # All sections should be ADDED (fallback)
        assert all(
            s.change_type == SectionChangeType.ADDED
            for s in delta.changed_sections
        )
        assert len(delta.changed_sections) == 2
        assert len(delta.unchanged_sections) == 0

    def test_evicted_version_fallback(self):
        cache = VersionedStateCache(session_id="s1", max_versions=1)
        cache.store_version({"a": 1})  # v0
        cache.store_version({"a": 2})  # v1, evicts v0
        delta, _ = cache.compute_delta(0, {"a": 3})  # v0 gone
        assert all(
            s.change_type == SectionChangeType.ADDED
            for s in delta.changed_sections
        )

    def test_nonexistent_version_fallback(self):
        cache = VersionedStateCache(session_id="s1")
        delta, new_v = cache.compute_delta(99, {"a": 1})
        assert all(
            s.change_type == SectionChangeType.ADDED
            for s in delta.changed_sections
        )
        # from_version is clamped to <= to_version
        assert delta.from_version <= delta.to_version


# ---------------------------------------------------------------------------
# compute_delta - token estimates
# ---------------------------------------------------------------------------
class TestComputeDeltaTokens:
    def test_delta_tokens_less_than_full_when_partial_change(self):
        cache = VersionedStateCache(session_id="s1")
        big_content = "x" * 1000
        v0 = cache.store_version({"big": big_content, "small": "y"})
        delta, _ = cache.compute_delta(v0.version_number, {"big": big_content, "small": "z"})
        assert delta.delta_token_estimate < delta.full_token_estimate

    def test_delta_includes_overhead(self):
        cache = VersionedStateCache(session_id="s1")
        v0 = cache.store_version({"a": "x"})
        delta, _ = cache.compute_delta(v0.version_number, {"a": "y"})
        # Should include the 45-token overhead
        assert delta.delta_token_estimate >= 45

    def test_fallback_delta_equals_full(self):
        cache = VersionedStateCache(session_id="s1")
        delta, _ = cache.compute_delta(99, {"a": "hello"})
        assert delta.delta_token_estimate == delta.full_token_estimate
        assert delta.from_version <= delta.to_version


# ---------------------------------------------------------------------------
# compute_delta - version advancement
# ---------------------------------------------------------------------------
class TestComputeDeltaVersions:
    def test_version_advances(self):
        cache = VersionedStateCache(session_id="s1")
        v0 = cache.store_version({"a": 1})
        delta, v1 = cache.compute_delta(v0.version_number, {"a": 2})
        assert v1.version_number == v0.version_number + 1

    def test_from_to_versions_in_delta(self):
        cache = VersionedStateCache(session_id="s1")
        v0 = cache.store_version({"a": 1})
        delta, v1 = cache.compute_delta(v0.version_number, {"a": 2})
        assert delta.from_version == v0.version_number
        assert delta.to_version == v1.version_number


# ---------------------------------------------------------------------------
# compute_delta - mixed changes
# ---------------------------------------------------------------------------
class TestComputeDeltaMixed:
    def test_add_remove_modify_unchanged(self):
        cache = VersionedStateCache(session_id="s1")
        v0 = cache.store_version({"keep": "same", "modify": "old", "remove": "bye"})
        delta, _ = cache.compute_delta(
            v0.version_number,
            {"keep": "same", "modify": "new", "add": "hi"},
        )
        changes_by_type = {}
        for s in delta.changed_sections:
            changes_by_type[s.section_name] = s.change_type

        assert changes_by_type["add"] == SectionChangeType.ADDED
        assert changes_by_type["remove"] == SectionChangeType.REMOVED
        assert changes_by_type["modify"] == SectionChangeType.MODIFIED
        assert "keep" in delta.unchanged_sections

    def test_delta_to_dict_roundtrip(self):
        cache = VersionedStateCache(session_id="s1")
        v0 = cache.store_version({"a": 1})
        delta, _ = cache.compute_delta(v0.version_number, {"a": 2, "b": 3})
        d = delta.to_dict()
        assert d["from_version"] == 0
        assert d["to_version"] == 1
        assert len(d["changed_sections"]) >= 1


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------
class TestClear:
    def test_clear_empties_cache(self):
        cache = VersionedStateCache(session_id="s1")
        cache.store_version({"a": 1})
        cache.store_version({"a": 2})
        cache.clear()
        assert cache.get_latest() is None
        assert cache.get_current_version_number() == 0

    def test_clear_resets_counter(self):
        cache = VersionedStateCache(session_id="s1")
        cache.store_version({"a": 1})
        cache.store_version({"a": 2})
        cache.clear()
        v = cache.store_version({"a": 3})
        assert v.version_number == 0


# ---------------------------------------------------------------------------
# drain_events
# ---------------------------------------------------------------------------
class TestDrainEvents:
    def test_drain_returns_empty_initially(self):
        cache = VersionedStateCache(session_id="s1")
        assert cache.drain_events() == []

    def test_drain_clears_events(self):
        cache = VersionedStateCache(session_id="s1")
        cache._pending_events.append("test_event")
        events = cache.drain_events()
        assert events == ["test_event"]
        assert cache.drain_events() == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_sections(self):
        cache = VersionedStateCache(session_id="s1")
        v = cache.store_version({})
        assert v.version_number == 0
        snap = cache.get_version(0)
        assert snap is not None
        assert snap.section_data == {}

    def test_large_number_of_sections(self):
        cache = VersionedStateCache(session_id="s1")
        sections = {f"section_{i}": f"value_{i}" for i in range(100)}
        v = cache.store_version(sections)
        snap = cache.get_version(v.version_number)
        assert snap is not None
        assert len(snap.section_data) == 100

    def test_nested_dict_sections(self):
        cache = VersionedStateCache(session_id="s1")
        sections = {"deep": {"level1": {"level2": {"level3": [1, 2, 3]}}}}
        v = cache.store_version(sections)
        snap = cache.get_version(v.version_number)
        assert snap is not None
        assert snap.get_section("deep") == sections["deep"]

    def test_compute_delta_with_empty_old_and_new(self):
        cache = VersionedStateCache(session_id="s1")
        v0 = cache.store_version({})
        delta, _ = cache.compute_delta(v0.version_number, {})
        assert delta.has_changes is False
        assert delta.change_count == 0

    def test_rapid_store_and_compute(self):
        cache = VersionedStateCache(session_id="s1", max_versions=3)
        versions = []
        for i in range(10):
            v = cache.store_version({"counter": i})
            versions.append(v)
        # Should still work with the latest
        delta, new_v = cache.compute_delta(
            versions[-1].version_number,
            {"counter": 10},
        )
        assert new_v.version_number == 10

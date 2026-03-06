"""Tests for Artifact Output Domain Aggregates (ADR-015)."""
from __future__ import annotations

__test__ = True

import pathlib
from datetime import datetime, timedelta

import pytest

from robotmcp.domains.artifact_output.aggregates import ArtifactStore
from robotmcp.domains.artifact_output.entities import Artifact, ArtifactSlice
from robotmcp.domains.artifact_output.events import ArtifactCreated, ArtifactExpired
from robotmcp.domains.artifact_output.value_objects import ArtifactPolicy


# ---------------------------------------------------------------------------
# create_artifact
# ---------------------------------------------------------------------------


class TestCreateArtifact:
    """create_artifact stores content and returns Artifact."""

    def test_returns_artifact_instance(self):
        store = ArtifactStore.create()
        art = store.create_artifact("hello world", "tool", "field", "sess-1")
        assert isinstance(art, Artifact)

    def test_stores_content_retrievable(self):
        store = ArtifactStore.create()
        art = store.create_artifact("hello world", "tool", "field", "sess-1")
        assert store.read_content(str(art.id)) == "hello world"

    def test_artifact_has_valid_id(self):
        store = ArtifactStore.create()
        art = store.create_artifact("data", "tool", "field", "sess-1")
        assert art.id.value.startswith("art_")
        assert len(art.id.value) == 16  # "art_" + 12 hex chars

    def test_artifact_reference_byte_size(self):
        store = ArtifactStore.create()
        art = store.create_artifact("hello world", "tool", "field", "sess-1")
        assert art.reference.byte_size == len("hello world".encode())

    def test_artifact_reference_token_estimate(self):
        store = ArtifactStore.create()
        art = store.create_artifact("hello world", "tool", "field", "sess-1")
        assert art.reference.token_estimate == len("hello world") // 4


# ---------------------------------------------------------------------------
# get_artifact - missing id
# ---------------------------------------------------------------------------


class TestGetArtifactMissing:
    """get_artifact returns None for missing id."""

    def test_nonexistent_id_returns_none(self):
        store = ArtifactStore.create()
        assert store.get_artifact("nonexistent") is None

    def test_empty_string_id_returns_none(self):
        store = ArtifactStore.create()
        assert store.get_artifact("") is None


# ---------------------------------------------------------------------------
# get_artifact - expired
# ---------------------------------------------------------------------------


class TestGetArtifactExpired:
    """get_artifact returns None for expired artifact (policy with ttl=0)."""

    def test_expired_artifact_returns_none(self):
        policy = ArtifactPolicy(retention_ttl_seconds=0)
        store = ArtifactStore.create(policy)
        art = store.create_artifact("data", "tool", "field", "sess-1")
        # Backdate creation so it is definitely past TTL=0
        art.created_at = datetime.now() - timedelta(seconds=10)
        assert store.get_artifact(str(art.id)) is None

    def test_expired_artifact_removed_from_store(self):
        policy = ArtifactPolicy(retention_ttl_seconds=0)
        store = ArtifactStore.create(policy)
        art = store.create_artifact("data", "tool", "field", "sess-1")
        art.created_at = datetime.now() - timedelta(seconds=10)
        store.get_artifact(str(art.id))
        # After get removes it, read_content should also return None
        assert store.read_content(str(art.id)) is None


# ---------------------------------------------------------------------------
# read_content
# ---------------------------------------------------------------------------


class TestReadContent:
    """read_content returns raw content."""

    def test_returns_exact_content(self):
        store = ArtifactStore.create()
        art = store.create_artifact("raw content here", "t", "f", "s")
        assert store.read_content(str(art.id)) == "raw content here"

    def test_returns_none_for_missing(self):
        store = ArtifactStore.create()
        assert store.read_content("nonexistent") is None

    def test_preserves_whitespace_and_special_chars(self):
        store = ArtifactStore.create()
        content = "  line1\n\ttab\r\nend  "
        art = store.create_artifact(content, "t", "f", "s")
        assert store.read_content(str(art.id)) == content


# ---------------------------------------------------------------------------
# read_slice
# ---------------------------------------------------------------------------


class TestReadSlice:
    """read_slice returns ArtifactSlice with pagination."""

    def test_returns_artifact_slice_type(self):
        store = ArtifactStore.create()
        art = store.create_artifact("abcdefghij", "t", "f", "s")
        slc = store.read_slice(str(art.id), offset=0, limit=5)
        assert isinstance(slc, ArtifactSlice)

    def test_slice_content_matches_offset_and_limit(self):
        store = ArtifactStore.create()
        art = store.create_artifact("abcdefghij", "t", "f", "s")
        slc = store.read_slice(str(art.id), offset=2, limit=4)
        assert slc is not None
        assert slc.content == "cdef"

    def test_slice_has_more_when_content_remains(self):
        store = ArtifactStore.create()
        content = "x" * 100
        art = store.create_artifact(content, "t", "f", "s")
        slc = store.read_slice(str(art.id), offset=0, limit=10)
        assert slc is not None
        assert slc.has_more is True

    def test_slice_no_more_when_fully_read(self):
        store = ArtifactStore.create()
        art = store.create_artifact("short", "t", "f", "s")
        slc = store.read_slice(str(art.id), offset=0, limit=4000)
        assert slc is not None
        assert slc.has_more is False

    def test_slice_total_size_matches_full_content(self):
        store = ArtifactStore.create()
        content = "hello world"
        art = store.create_artifact(content, "t", "f", "s")
        slc = store.read_slice(str(art.id), offset=0, limit=5)
        assert slc is not None
        assert slc.total_size == len(content)

    def test_slice_returns_none_for_missing_id(self):
        store = ArtifactStore.create()
        assert store.read_slice("nonexistent") is None


# ---------------------------------------------------------------------------
# cleanup_expired
# ---------------------------------------------------------------------------


class TestCleanupExpired:
    """cleanup_expired removes expired artifacts."""

    def test_removes_expired_returns_count(self):
        policy = ArtifactPolicy(retention_ttl_seconds=0)
        store = ArtifactStore.create(policy)
        art = store.create_artifact("data", "t", "f", "s")
        art.created_at = datetime.now() - timedelta(seconds=10)
        count = store.cleanup_expired()
        assert count == 1
        assert store.list_artifacts() == []

    def test_keeps_fresh_artifacts(self):
        store = ArtifactStore.create()
        store.create_artifact("data", "t", "f", "s")
        count = store.cleanup_expired()
        assert count == 0
        assert len(store.list_artifacts()) == 1

    def test_removes_only_expired_mixed(self):
        policy = ArtifactPolicy(retention_ttl_seconds=3600)
        store = ArtifactStore.create(policy)
        expired_art = store.create_artifact("old", "t", "f", "s")
        # Backdate the artifact so it exceeds the 1-hour TTL
        expired_art.created_at = datetime.now() - timedelta(seconds=7200)
        fresh_art = store.create_artifact("new", "t", "f", "s")
        count = store.cleanup_expired()
        assert count == 1
        remaining = store.list_artifacts()
        assert len(remaining) == 1
        assert str(remaining[0].id) == str(fresh_art.id)


# ---------------------------------------------------------------------------
# cleanup_session
# ---------------------------------------------------------------------------


class TestCleanupSession:
    """cleanup_session removes session artifacts."""

    def test_removes_all_session_artifacts(self):
        store = ArtifactStore.create()
        store.create_artifact("a", "t", "f", "sess-1")
        store.create_artifact("b", "t", "f", "sess-1")
        store.create_artifact("c", "t", "f", "sess-2")
        count = store.cleanup_session("sess-1")
        assert count == 2
        assert len(store.list_artifacts()) == 1

    def test_returns_zero_for_unknown_session(self):
        store = ArtifactStore.create()
        assert store.cleanup_session("nonexistent") == 0

    def test_other_sessions_unaffected(self):
        store = ArtifactStore.create()
        store.create_artifact("a", "t", "f", "sess-1")
        art_b = store.create_artifact("b", "t", "f", "sess-2")
        store.cleanup_session("sess-1")
        assert store.get_artifact(str(art_b.id)) is not None


# ---------------------------------------------------------------------------
# list_artifacts
# ---------------------------------------------------------------------------


class TestListArtifacts:
    """list_artifacts filters by session."""

    def test_list_all_no_filter(self):
        store = ArtifactStore.create()
        store.create_artifact("a", "t", "f", "s1")
        store.create_artifact("b", "t", "f", "s2")
        assert len(store.list_artifacts()) == 2

    def test_filter_by_session(self):
        store = ArtifactStore.create()
        store.create_artifact("a", "t", "f", "s1")
        store.create_artifact("b", "t", "f", "s1")
        store.create_artifact("c", "t", "f", "s2")
        assert len(store.list_artifacts("s1")) == 2
        assert len(store.list_artifacts("s2")) == 1

    def test_empty_session_returns_empty_list(self):
        store = ArtifactStore.create()
        assert store.list_artifacts("nonexistent") == []

    def test_returns_artifact_instances(self):
        store = ArtifactStore.create()
        store.create_artifact("a", "t", "f", "s1")
        arts = store.list_artifacts("s1")
        assert all(isinstance(a, Artifact) for a in arts)


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestLRUEviction:
    """LRU eviction when max_artifacts is reached."""

    def test_evicts_oldest_when_full(self):
        policy = ArtifactPolicy(max_artifacts=3)
        store = ArtifactStore.create(policy)
        a1 = store.create_artifact("one", "t", "f", "s")
        store.create_artifact("two", "t", "f", "s")
        store.create_artifact("three", "t", "f", "s")
        # 4th creation should evict the first
        a4 = store.create_artifact("four", "t", "f", "s")
        assert store.get_artifact(str(a1.id)) is None
        assert store.get_artifact(str(a4.id)) is not None

    def test_max_artifacts_one(self):
        policy = ArtifactPolicy(max_artifacts=1)
        store = ArtifactStore.create(policy)
        a1 = store.create_artifact("first", "t", "f", "s")
        a2 = store.create_artifact("second", "t", "f", "s")
        assert store.get_artifact(str(a1.id)) is None
        assert store.get_artifact(str(a2.id)) is not None

    def test_keeps_newest_after_eviction(self):
        policy = ArtifactPolicy(max_artifacts=2)
        store = ArtifactStore.create(policy)
        store.create_artifact("1", "t", "f", "s")
        a2 = store.create_artifact("2", "t", "f", "s")
        a3 = store.create_artifact("3", "t", "f", "s")
        assert store.get_artifact(str(a2.id)) is not None
        assert store.get_artifact(str(a3.id)) is not None


# ---------------------------------------------------------------------------
# drain_events
# ---------------------------------------------------------------------------


class TestDrainEvents:
    """drain_events returns and clears events."""

    def test_returns_accumulated_events(self):
        store = ArtifactStore.create()
        store.create_artifact("a", "t", "f", "s")
        store.create_artifact("b", "t", "f", "s")
        events = store.drain_events()
        assert len(events) == 2

    def test_clears_events_after_drain(self):
        store = ArtifactStore.create()
        store.create_artifact("a", "t", "f", "s")
        first = store.drain_events()
        assert len(first) == 1
        second = store.drain_events()
        assert len(second) == 0

    def test_empty_store_returns_empty_list(self):
        store = ArtifactStore.create()
        assert store.drain_events() == []


# ---------------------------------------------------------------------------
# ArtifactCreated event emitted on create
# ---------------------------------------------------------------------------


class TestArtifactCreatedEvent:
    """ArtifactCreated event emitted on create."""

    def test_created_event_type(self):
        store = ArtifactStore.create()
        store.create_artifact("data", "my_tool", "my_field", "sess-1")
        events = store.drain_events()
        assert len(events) == 1
        assert isinstance(events[0], ArtifactCreated)

    def test_created_event_fields(self):
        store = ArtifactStore.create()
        art = store.create_artifact("data", "my_tool", "my_field", "sess-1")
        events = store.drain_events()
        ev = events[0]
        assert ev.artifact_id == str(art.id)
        assert ev.tool_name == "my_tool"
        assert ev.field_name == "my_field"
        assert ev.session_id == "sess-1"
        assert ev.byte_size == len("data".encode())
        assert ev.token_estimate == len("data") // 4


# ---------------------------------------------------------------------------
# ArtifactExpired event emitted on removal
# ---------------------------------------------------------------------------


class TestArtifactExpiredEvent:
    """ArtifactExpired event emitted on removal."""

    def test_expired_event_on_eviction(self):
        policy = ArtifactPolicy(max_artifacts=1)
        store = ArtifactStore.create(policy)
        store.create_artifact("first", "t", "f", "s")
        store.drain_events()  # clear creation event
        store.create_artifact("second", "t", "f", "s")
        events = store.drain_events()
        expired_events = [e for e in events if isinstance(e, ArtifactExpired)]
        assert len(expired_events) == 1

    def test_expired_event_on_cleanup(self):
        policy = ArtifactPolicy(retention_ttl_seconds=0)
        store = ArtifactStore.create(policy)
        art = store.create_artifact("data", "tool_x", "field_y", "sess-1")
        art.created_at = datetime.now() - timedelta(seconds=10)
        store.drain_events()  # clear creation event
        store.cleanup_expired()
        events = store.drain_events()
        expired_events = [e for e in events if isinstance(e, ArtifactExpired)]
        assert len(expired_events) == 1
        assert expired_events[0].tool_name == "tool_x"
        assert expired_events[0].session_id == "sess-1"

    def test_expired_event_on_get_expired(self):
        policy = ArtifactPolicy(retention_ttl_seconds=0)
        store = ArtifactStore.create(policy)
        art = store.create_artifact("data", "t", "f", "s")
        art.created_at = datetime.now() - timedelta(seconds=10)
        store.drain_events()  # clear creation event
        store.get_artifact(str(art.id))  # triggers removal
        events = store.drain_events()
        expired_events = [e for e in events if isinstance(e, ArtifactExpired)]
        assert len(expired_events) == 1


# ---------------------------------------------------------------------------
# Disk write/delete
# ---------------------------------------------------------------------------


class TestDiskWrite:
    """Tests for artifact disk persistence."""

    def test_create_artifact_writes_to_disk(self, tmp_path):
        policy = ArtifactPolicy(artifact_dir=str(tmp_path / "arts"))
        store = ArtifactStore.create(policy)
        art = store.create_artifact("hello disk", "tool", "field", "sess-1")
        fp = pathlib.Path(art.reference.file_path)
        assert fp.exists()
        assert fp.read_text(encoding="utf-8") == "hello disk"

    def test_file_extension_txt_for_plain(self, tmp_path):
        policy = ArtifactPolicy(artifact_dir=str(tmp_path / "arts"))
        store = ArtifactStore.create(policy)
        art = store.create_artifact("data", "t", "f", "s", mime_type="text/plain")
        assert art.reference.file_path.endswith(".txt")

    def test_file_extension_json(self, tmp_path):
        policy = ArtifactPolicy(artifact_dir=str(tmp_path / "arts"))
        store = ArtifactStore.create(policy)
        art = store.create_artifact("{}", "t", "f", "s", mime_type="application/json")
        assert art.reference.file_path.endswith(".json")

    def test_file_extension_html(self, tmp_path):
        policy = ArtifactPolicy(artifact_dir=str(tmp_path / "arts"))
        store = ArtifactStore.create(policy)
        art = store.create_artifact("<html/>", "t", "f", "s", mime_type="text/html")
        assert art.reference.file_path.endswith(".html")

    def test_file_extension_robot(self, tmp_path):
        policy = ArtifactPolicy(artifact_dir=str(tmp_path / "arts"))
        store = ArtifactStore.create(policy)
        art = store.create_artifact("***", "t", "f", "s", mime_type="text/x-robot")
        assert art.reference.file_path.endswith(".robot")

    def test_file_extension_unknown_defaults_to_txt(self, tmp_path):
        policy = ArtifactPolicy(artifact_dir=str(tmp_path / "arts"))
        store = ArtifactStore.create(policy)
        art = store.create_artifact("data", "t", "f", "s", mime_type="application/octet-stream")
        assert art.reference.file_path.endswith(".txt")

    def test_remove_artifact_deletes_from_disk(self, tmp_path):
        policy = ArtifactPolicy(artifact_dir=str(tmp_path / "arts"))
        store = ArtifactStore.create(policy)
        art = store.create_artifact("data", "t", "f", "s")
        fp = pathlib.Path(art.reference.file_path)
        assert fp.exists()
        store._remove_artifact(str(art.id))
        assert not fp.exists()

    def test_cleanup_session_deletes_files(self, tmp_path):
        policy = ArtifactPolicy(artifact_dir=str(tmp_path / "arts"))
        store = ArtifactStore.create(policy)
        art1 = store.create_artifact("a", "t", "f", "s1")
        art2 = store.create_artifact("b", "t", "f", "s1")
        fp1 = pathlib.Path(art1.reference.file_path)
        fp2 = pathlib.Path(art2.reference.file_path)
        assert fp1.exists() and fp2.exists()
        store.cleanup_session("s1")
        assert not fp1.exists() and not fp2.exists()

    def test_eviction_deletes_oldest_file(self, tmp_path):
        policy = ArtifactPolicy(artifact_dir=str(tmp_path / "arts"), max_artifacts=1)
        store = ArtifactStore.create(policy)
        art1 = store.create_artifact("first", "t", "f", "s")
        fp1 = pathlib.Path(art1.reference.file_path)
        assert fp1.exists()
        store.create_artifact("second", "t", "f", "s")
        assert not fp1.exists()

    def test_create_factory_resolves_relative_path(self):
        policy = ArtifactPolicy(artifact_dir=".robotmcp_artifacts")
        store = ArtifactStore.create(policy)
        assert pathlib.Path(store.policy.artifact_dir).is_absolute()

    def test_create_factory_preserves_absolute_path(self, tmp_path):
        policy = ArtifactPolicy(artifact_dir=str(tmp_path))
        store = ArtifactStore.create(policy)
        assert store.policy.artifact_dir == str(tmp_path)

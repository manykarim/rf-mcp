"""Tests for Delta State Retrieval Events (ADR-018)."""
from __future__ import annotations

__test__ = True

from datetime import datetime

import pytest

from robotmcp.domains.snapshot.delta_events import (
    DeltaComputed,
    StateVersionCreated,
    StateVersionExpired,
)


# ---------------------------------------------------------------------------
# StateVersionCreated
# ---------------------------------------------------------------------------
class TestStateVersionCreated:
    def test_construction(self):
        ts = datetime(2026, 3, 6, 12, 0, 0)
        event = StateVersionCreated(
            session_id="sess1",
            version_number=3,
            content_hash="abcdef1234567890",
            timestamp=ts,
        )
        assert event.session_id == "sess1"
        assert event.version_number == 3
        assert event.content_hash == "abcdef1234567890"
        assert event.timestamp == ts

    def test_default_timestamp(self):
        before = datetime.now()
        event = StateVersionCreated(
            session_id="s1",
            version_number=0,
            content_hash="h",
        )
        after = datetime.now()
        assert before <= event.timestamp <= after

    def test_to_dict_structure(self):
        ts = datetime(2026, 3, 6, 14, 30, 0)
        event = StateVersionCreated(
            session_id="sess_abc",
            version_number=5,
            content_hash="abcdef1234567890",
            timestamp=ts,
        )
        d = event.to_dict()
        assert d["event_type"] == "StateVersionCreated"
        assert d["session"] == "sess_abc"
        assert d["version"] == 5
        assert d["hash"] == "abcdef12"  # first 8 chars
        assert d["ts"] == ts.isoformat()

    def test_to_dict_short_hash(self):
        event = StateVersionCreated(
            session_id="s1",
            version_number=0,
            content_hash="abc",
        )
        assert event.to_dict()["hash"] == "abc"

    def test_frozen(self):
        event = StateVersionCreated(
            session_id="s1",
            version_number=0,
            content_hash="h",
        )
        with pytest.raises(AttributeError):
            event.session_id = "other"  # type: ignore[misc]

    def test_equality(self):
        ts = datetime(2026, 1, 1)
        a = StateVersionCreated("s1", 0, "h", ts)
        b = StateVersionCreated("s1", 0, "h", ts)
        assert a == b

    def test_inequality_different_version(self):
        ts = datetime(2026, 1, 1)
        a = StateVersionCreated("s1", 0, "h", ts)
        b = StateVersionCreated("s1", 1, "h", ts)
        assert a != b


# ---------------------------------------------------------------------------
# DeltaComputed
# ---------------------------------------------------------------------------
class TestDeltaComputed:
    def test_construction(self):
        ts = datetime(2026, 3, 6, 12, 0, 0)
        event = DeltaComputed(
            session_id="sess1",
            from_version=0,
            to_version=3,
            changed_count=2,
            unchanged_count=5,
            saved_tokens=150,
            timestamp=ts,
        )
        assert event.session_id == "sess1"
        assert event.from_version == 0
        assert event.to_version == 3
        assert event.changed_count == 2
        assert event.unchanged_count == 5
        assert event.saved_tokens == 150
        assert event.timestamp == ts

    def test_default_timestamp(self):
        before = datetime.now()
        event = DeltaComputed(
            session_id="s1",
            from_version=0,
            to_version=1,
            changed_count=1,
            unchanged_count=0,
            saved_tokens=0,
        )
        after = datetime.now()
        assert before <= event.timestamp <= after

    def test_to_dict_structure(self):
        ts = datetime(2026, 3, 6, 15, 0, 0)
        event = DeltaComputed(
            session_id="sess_xyz",
            from_version=2,
            to_version=5,
            changed_count=3,
            unchanged_count=7,
            saved_tokens=500,
            timestamp=ts,
        )
        d = event.to_dict()
        assert d["event_type"] == "DeltaComputed"
        assert d["session"] == "sess_xyz"
        assert d["from"] == 2
        assert d["to"] == 5
        assert d["changed"] == 3
        assert d["unchanged"] == 7
        assert d["saved_tokens"] == 500
        assert d["ts"] == ts.isoformat()

    def test_frozen(self):
        event = DeltaComputed(
            session_id="s1",
            from_version=0,
            to_version=1,
            changed_count=1,
            unchanged_count=0,
            saved_tokens=0,
        )
        with pytest.raises(AttributeError):
            event.from_version = 99  # type: ignore[misc]

    def test_zero_changes(self):
        event = DeltaComputed(
            session_id="s1",
            from_version=0,
            to_version=1,
            changed_count=0,
            unchanged_count=10,
            saved_tokens=0,
        )
        d = event.to_dict()
        assert d["changed"] == 0
        assert d["unchanged"] == 10

    def test_equality(self):
        ts = datetime(2026, 1, 1)
        a = DeltaComputed("s1", 0, 1, 2, 3, 100, ts)
        b = DeltaComputed("s1", 0, 1, 2, 3, 100, ts)
        assert a == b


# ---------------------------------------------------------------------------
# StateVersionExpired
# ---------------------------------------------------------------------------
class TestStateVersionExpired:
    def test_construction_ttl(self):
        ts = datetime(2026, 3, 6, 12, 0, 0)
        event = StateVersionExpired(
            session_id="sess1",
            version_number=2,
            reason="ttl",
            timestamp=ts,
        )
        assert event.session_id == "sess1"
        assert event.version_number == 2
        assert event.reason == "ttl"
        assert event.timestamp == ts

    def test_construction_lru(self):
        event = StateVersionExpired(
            session_id="sess1",
            version_number=0,
            reason="lru",
        )
        assert event.reason == "lru"

    def test_default_timestamp(self):
        before = datetime.now()
        event = StateVersionExpired(
            session_id="s1",
            version_number=0,
            reason="ttl",
        )
        after = datetime.now()
        assert before <= event.timestamp <= after

    def test_to_dict_structure(self):
        ts = datetime(2026, 3, 6, 16, 0, 0)
        event = StateVersionExpired(
            session_id="sess_old",
            version_number=7,
            reason="lru",
            timestamp=ts,
        )
        d = event.to_dict()
        assert d["event_type"] == "StateVersionExpired"
        assert d["session"] == "sess_old"
        assert d["version"] == 7
        assert d["reason"] == "lru"
        assert d["ts"] == ts.isoformat()

    def test_to_dict_ttl_reason(self):
        event = StateVersionExpired(
            session_id="s1",
            version_number=0,
            reason="ttl",
        )
        d = event.to_dict()
        assert d["reason"] == "ttl"

    def test_frozen(self):
        event = StateVersionExpired(
            session_id="s1",
            version_number=0,
            reason="ttl",
        )
        with pytest.raises(AttributeError):
            event.reason = "lru"  # type: ignore[misc]

    def test_equality(self):
        ts = datetime(2026, 1, 1)
        a = StateVersionExpired("s1", 0, "ttl", ts)
        b = StateVersionExpired("s1", 0, "ttl", ts)
        assert a == b

    def test_inequality_different_reason(self):
        ts = datetime(2026, 1, 1)
        a = StateVersionExpired("s1", 0, "ttl", ts)
        b = StateVersionExpired("s1", 0, "lru", ts)
        assert a != b

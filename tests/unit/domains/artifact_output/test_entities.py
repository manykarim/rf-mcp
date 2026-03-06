"""Tests for Artifact Output Domain Entities (ADR-015)."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from robotmcp.domains.artifact_output.entities import Artifact, ArtifactSlice
from robotmcp.domains.artifact_output.value_objects import (
    ArtifactId,
    ArtifactReference,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ref(**kwargs):
    defaults = {
        "artifact_id": "art_aabbccddeeff",
        "file_path": ".artifacts/art_aabbccddeeff",
        "content_hash": "abc123def456",
        "byte_size": 1024,
        "token_estimate": 256,
        "mime_type": "text/plain",
    }
    defaults.update(kwargs)
    return ArtifactReference(**defaults)


def _make_artifact(**kwargs):
    defaults = {
        "id": ArtifactId("art_aabbccddeeff"),
        "reference": _make_ref(),
        "tool_name": "build_test_suite",
        "field_name": "rf_text",
        "session_id": "sess-1",
    }
    defaults.update(kwargs)
    return Artifact(**defaults)


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------

class TestArtifact:
    """Tests for Artifact entity."""

    def test_construction(self):
        a = _make_artifact()
        assert a.tool_name == "build_test_suite"
        assert a.session_id == "sess-1"
        assert a.accessed_at is None

    def test_created_at_default(self):
        a = _make_artifact()
        assert isinstance(a.created_at, datetime)

    def test_record_access(self):
        a = _make_artifact()
        assert a.accessed_at is None
        a.record_access()
        assert a.accessed_at is not None
        assert isinstance(a.accessed_at, datetime)

    def test_record_access_updates(self):
        a = _make_artifact()
        a.record_access()
        first = a.accessed_at
        time.sleep(0.01)
        a.record_access()
        assert a.accessed_at >= first  # type: ignore[operator]

    def test_is_expired_false_when_fresh(self):
        a = _make_artifact()
        assert not a.is_expired(3600)

    def test_is_expired_true_when_old(self):
        old_time = datetime.now() - timedelta(seconds=100)
        a = _make_artifact()
        a.created_at = old_time
        assert a.is_expired(50)

    def test_is_expired_boundary(self):
        # Just within TTL
        a = _make_artifact()
        a.created_at = datetime.now() - timedelta(seconds=9)
        assert not a.is_expired(10)

    def test_to_dict_structure(self):
        a = _make_artifact()
        d = a.to_dict()
        assert d["id"] == "art_aabbccddeeff"
        assert d["tool"] == "build_test_suite"
        assert d["field"] == "rf_text"
        assert d["session"] == "sess-1"
        assert "ref" in d
        assert "created" in d
        assert d["accessed"] is None

    def test_to_dict_with_access(self):
        a = _make_artifact()
        a.record_access()
        d = a.to_dict()
        assert d["accessed"] is not None

    def test_to_dict_ref_is_inline_dict(self):
        a = _make_artifact()
        d = a.to_dict()
        ref = d["ref"]
        assert "artifact_id" in ref
        assert "size" in ref

    def test_mutable(self):
        a = _make_artifact()
        a.tool_name = "other_tool"
        assert a.tool_name == "other_tool"

    def test_different_sessions(self):
        a1 = _make_artifact(session_id="sess-a")
        a2 = _make_artifact(session_id="sess-b")
        assert a1.session_id != a2.session_id


# ---------------------------------------------------------------------------
# ArtifactSlice
# ---------------------------------------------------------------------------

class TestArtifactSlice:
    """Tests for ArtifactSlice entity."""

    def test_construction(self):
        s = ArtifactSlice(artifact_id="art_aabbccddeeff", offset=0, limit=100, content="hello", total_size=100)
        assert s.content == "hello"
        assert s.total_size == 100

    def test_frozen(self):
        s = ArtifactSlice(artifact_id="art_aabbccddeeff", offset=0, limit=100, content="hello", total_size=100)
        with pytest.raises(AttributeError):
            s.content = "x"  # type: ignore[misc]

    def test_negative_offset_rejected(self):
        with pytest.raises(ValueError, match="offset cannot be negative"):
            ArtifactSlice(artifact_id="x", offset=-1, limit=10, content="a", total_size=10)

    def test_negative_limit_rejected(self):
        with pytest.raises(ValueError, match="limit cannot be negative"):
            ArtifactSlice(artifact_id="x", offset=0, limit=-1, content="a", total_size=10)

    def test_end_offset(self):
        s = ArtifactSlice(artifact_id="x", offset=10, limit=100, content="hello", total_size=100)
        assert s.end_offset == 15  # 10 + len("hello")

    def test_end_offset_from_zero(self):
        s = ArtifactSlice(artifact_id="x", offset=0, limit=100, content="abc", total_size=50)
        assert s.end_offset == 3

    def test_has_more_true(self):
        s = ArtifactSlice(artifact_id="x", offset=0, limit=10, content="hello", total_size=100)
        assert s.has_more is True

    def test_has_more_false_exact(self):
        s = ArtifactSlice(artifact_id="x", offset=0, limit=10, content="hello", total_size=5)
        assert s.has_more is False

    def test_has_more_false_beyond(self):
        s = ArtifactSlice(artifact_id="x", offset=95, limit=10, content="12345", total_size=100)
        assert s.has_more is False

    def test_to_dict(self):
        s = ArtifactSlice(artifact_id="art_aabbccddeeff", offset=10, limit=50, content="data", total_size=200)
        d = s.to_dict()
        assert d["artifact_id"] == "art_aabbccddeeff"
        assert d["offset"] == 10
        assert d["length"] == 4
        assert d["total_size"] == 200
        assert d["has_more"] is True
        assert d["content"] == "data"

    def test_to_dict_no_more(self):
        s = ArtifactSlice(artifact_id="x", offset=0, limit=10, content="ab", total_size=2)
        d = s.to_dict()
        assert d["has_more"] is False

    def test_empty_content(self):
        s = ArtifactSlice(artifact_id="x", offset=0, limit=10, content="", total_size=0)
        assert s.end_offset == 0
        assert s.has_more is False

    def test_zero_offset_zero_limit(self):
        s = ArtifactSlice(artifact_id="x", offset=0, limit=0, content="", total_size=0)
        assert s.end_offset == 0

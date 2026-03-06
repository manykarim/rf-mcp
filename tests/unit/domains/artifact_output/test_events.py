"""Tests for Artifact Output Domain Events (ADR-015)."""

from __future__ import annotations

from datetime import datetime

import pytest

from robotmcp.domains.artifact_output.events import (
    ArtifactAccessed,
    ArtifactCreated,
    ArtifactExpired,
    LargeFieldExternalized,
)


class TestArtifactCreated:
    """Tests for ArtifactCreated event."""

    def test_construction(self):
        e = ArtifactCreated(artifact_id="art_aabbccddeeff", tool_name="build_test_suite", field_name="rf_text", session_id="sess-1", byte_size=1024, token_estimate=256)
        assert e.artifact_id == "art_aabbccddeeff"
        assert e.tool_name == "build_test_suite"

    def test_timestamp_default(self):
        e = ArtifactCreated(artifact_id="x", tool_name="t", field_name="f", session_id="s", byte_size=0, token_estimate=0)
        assert isinstance(e.timestamp, datetime)

    def test_frozen(self):
        e = ArtifactCreated(artifact_id="x", tool_name="t", field_name="f", session_id="s", byte_size=0, token_estimate=0)
        with pytest.raises(AttributeError):
            e.artifact_id = "y"  # type: ignore[misc]

    def test_to_dict_event_type(self):
        e = ArtifactCreated(artifact_id="x", tool_name="t", field_name="f", session_id="s", byte_size=100, token_estimate=25)
        d = e.to_dict()
        assert d["event_type"] == "ArtifactCreated"
        assert d["artifact_id"] == "x"
        assert d["tool"] == "t"
        assert d["field"] == "f"
        assert d["session"] == "s"
        assert d["size"] == 100
        assert d["tokens"] == 25
        assert "ts" in d


class TestArtifactAccessed:
    """Tests for ArtifactAccessed event."""

    def test_construction(self):
        e = ArtifactAccessed(artifact_id="art_112233445566", tool_name="fetch_artifact")
        assert e.artifact_id == "art_112233445566"

    def test_timestamp_default(self):
        e = ArtifactAccessed(artifact_id="x", tool_name="t")
        assert isinstance(e.timestamp, datetime)

    def test_frozen(self):
        e = ArtifactAccessed(artifact_id="x", tool_name="t")
        with pytest.raises(AttributeError):
            e.artifact_id = "y"  # type: ignore[misc]

    def test_to_dict_event_type(self):
        e = ArtifactAccessed(artifact_id="x", tool_name="t")
        d = e.to_dict()
        assert d["event_type"] == "ArtifactAccessed"
        assert d["artifact_id"] == "x"
        assert d["tool"] == "t"
        assert "ts" in d


class TestArtifactExpired:
    """Tests for ArtifactExpired event."""

    def test_construction(self):
        e = ArtifactExpired(artifact_id="x", tool_name="t", session_id="s", age_seconds=3600.5)
        assert e.age_seconds == 3600.5

    def test_timestamp_default(self):
        e = ArtifactExpired(artifact_id="x", tool_name="t", session_id="s", age_seconds=0)
        assert isinstance(e.timestamp, datetime)

    def test_frozen(self):
        e = ArtifactExpired(artifact_id="x", tool_name="t", session_id="s", age_seconds=0)
        with pytest.raises(AttributeError):
            e.artifact_id = "y"  # type: ignore[misc]

    def test_to_dict_event_type(self):
        e = ArtifactExpired(artifact_id="x", tool_name="t", session_id="s", age_seconds=123.456)
        d = e.to_dict()
        assert d["event_type"] == "ArtifactExpired"
        assert d["artifact_id"] == "x"
        assert d["session"] == "s"
        assert d["age_s"] == 123.5  # rounded to 1 decimal
        assert "ts" in d


class TestLargeFieldExternalized:
    """Tests for LargeFieldExternalized event."""

    def test_construction(self):
        e = LargeFieldExternalized(tool_name="build_test_suite", field_path="rf_text", artifact_id="art_aabbccddeeff", original_tokens=1000, saved_tokens=800)
        assert e.saved_tokens == 800

    def test_timestamp_default(self):
        e = LargeFieldExternalized(tool_name="t", field_path="f", artifact_id="x", original_tokens=0, saved_tokens=0)
        assert isinstance(e.timestamp, datetime)

    def test_frozen(self):
        e = LargeFieldExternalized(tool_name="t", field_path="f", artifact_id="x", original_tokens=0, saved_tokens=0)
        with pytest.raises(AttributeError):
            e.tool_name = "y"  # type: ignore[misc]

    def test_to_dict_event_type(self):
        e = LargeFieldExternalized(tool_name="build_test_suite", field_path="rf_text", artifact_id="art_aabbccddeeff", original_tokens=1000, saved_tokens=800)
        d = e.to_dict()
        assert d["event_type"] == "LargeFieldExternalized"
        assert d["tool"] == "build_test_suite"
        assert d["field"] == "rf_text"
        assert d["artifact_id"] == "art_aabbccddeeff"
        assert d["original_tokens"] == 1000
        assert d["saved_tokens"] == 800
        assert "ts" in d

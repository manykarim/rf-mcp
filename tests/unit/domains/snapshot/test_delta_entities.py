"""Tests for Delta State Retrieval Entities (ADR-018)."""
from __future__ import annotations

__test__ = True

import hashlib
import json
from datetime import datetime

import pytest

from robotmcp.domains.snapshot.delta_entities import VersionedSnapshot
from robotmcp.domains.snapshot.delta_value_objects import StateVersion


@pytest.fixture
def version():
    return StateVersion.initial("sess1", "abc123")


@pytest.fixture
def sections():
    return {
        "variables": {"x": 1, "y": "hello"},
        "libraries": ["Browser", "Collections"],
        "page_state": "some page content here",
    }


class TestVersionedSnapshot:
    def test_create(self, version, sections):
        snap = VersionedSnapshot.create(version, sections)
        assert snap.version == version
        assert snap.section_data == sections
        assert len(snap.section_hashes) == 3

    def test_section_hashes_deterministic(self, version, sections):
        snap1 = VersionedSnapshot.create(version, sections)
        snap2 = VersionedSnapshot.create(version, sections)
        assert snap1.section_hashes == snap2.section_hashes

    def test_section_hashes_change_with_content(self, version):
        s1 = {"variables": {"x": 1}}
        s2 = {"variables": {"x": 2}}
        snap1 = VersionedSnapshot.create(version, s1)
        snap2 = VersionedSnapshot.create(version, s2)
        assert snap1.section_hashes["variables"] != snap2.section_hashes["variables"]

    def test_string_section_hash(self, version):
        snap = VersionedSnapshot.create(version, {"text": "hello world"})
        expected = hashlib.md5("hello world".encode()).hexdigest()
        assert snap.section_hashes["text"] == expected

    def test_dict_section_hash_uses_sorted_keys(self, version):
        s1 = {"data": {"b": 2, "a": 1}}
        s2 = {"data": {"a": 1, "b": 2}}
        snap1 = VersionedSnapshot.create(version, s1)
        snap2 = VersionedSnapshot.create(version, s2)
        assert snap1.section_hashes["data"] == snap2.section_hashes["data"]

    def test_get_section_hash_present(self, version, sections):
        snap = VersionedSnapshot.create(version, sections)
        assert snap.get_section_hash("variables") is not None

    def test_get_section_hash_absent(self, version, sections):
        snap = VersionedSnapshot.create(version, sections)
        assert snap.get_section_hash("nonexistent") is None

    def test_has_section_true(self, version, sections):
        snap = VersionedSnapshot.create(version, sections)
        assert snap.has_section("variables") is True

    def test_has_section_false(self, version, sections):
        snap = VersionedSnapshot.create(version, sections)
        assert snap.has_section("nonexistent") is False

    def test_get_section_present(self, version, sections):
        snap = VersionedSnapshot.create(version, sections)
        assert snap.get_section("variables") == {"x": 1, "y": "hello"}

    def test_get_section_absent(self, version, sections):
        snap = VersionedSnapshot.create(version, sections)
        assert snap.get_section("nonexistent") is None

    def test_estimate_tokens(self, version, sections):
        snap = VersionedSnapshot.create(version, sections)
        tokens = snap.estimate_tokens()
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_estimate_tokens_empty_sections(self, version):
        snap = VersionedSnapshot.create(version, {})
        assert snap.estimate_tokens() == 0

    def test_to_dict(self, version, sections):
        snap = VersionedSnapshot.create(version, sections)
        d = snap.to_dict()
        assert "version" in d
        assert set(d["sections"]) == set(sections.keys())
        assert d["tokens"] == snap.estimate_tokens()

    def test_created_at_default(self, version, sections):
        before = datetime.now()
        snap = VersionedSnapshot.create(version, sections)
        after = datetime.now()
        assert before <= snap.created_at <= after

"""Tests for Delta State Retrieval Value Objects (ADR-018)."""
from __future__ import annotations

__test__ = True

import pytest
from datetime import datetime

from robotmcp.domains.snapshot.delta_value_objects import (
    DeltaSection,
    SectionChangeType,
    StateDelta,
    StateRetrievalMode,
    StateVersion,
)


# ---------------------------------------------------------------------------
# StateRetrievalMode
# ---------------------------------------------------------------------------
class TestStateRetrievalMode:
    def test_full_value(self):
        assert StateRetrievalMode.FULL.value == "full"

    def test_delta_value(self):
        assert StateRetrievalMode.DELTA.value == "delta"

    def test_none_value(self):
        assert StateRetrievalMode.NONE.value == "none"

    def test_all_members(self):
        assert set(StateRetrievalMode) == {
            StateRetrievalMode.FULL,
            StateRetrievalMode.DELTA,
            StateRetrievalMode.NONE,
        }


# ---------------------------------------------------------------------------
# SectionChangeType
# ---------------------------------------------------------------------------
class TestSectionChangeType:
    def test_added_value(self):
        assert SectionChangeType.ADDED.value == "added"

    def test_removed_value(self):
        assert SectionChangeType.REMOVED.value == "removed"

    def test_modified_value(self):
        assert SectionChangeType.MODIFIED.value == "modified"

    def test_unchanged_value(self):
        assert SectionChangeType.UNCHANGED.value == "unchanged"

    def test_all_members(self):
        assert len(SectionChangeType) == 4


# ---------------------------------------------------------------------------
# StateVersion
# ---------------------------------------------------------------------------
class TestStateVersion:
    def test_create_valid(self):
        ts = datetime(2026, 1, 1, 12, 0, 0)
        v = StateVersion(
            version_number=3,
            timestamp=ts,
            session_id="sess1",
            content_hash="abc123def456",
        )
        assert v.version_number == 3
        assert v.timestamp == ts
        assert v.session_id == "sess1"
        assert v.content_hash == "abc123def456"

    def test_negative_version_raises(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            StateVersion(
                version_number=-1,
                timestamp=datetime.now(),
                session_id="s",
                content_hash="h",
            )

    def test_empty_session_id_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            StateVersion(
                version_number=0,
                timestamp=datetime.now(),
                session_id="",
                content_hash="h",
            )

    def test_initial_factory(self):
        v = StateVersion.initial("sess_abc", "hash123")
        assert v.version_number == 0
        assert v.session_id == "sess_abc"
        assert v.content_hash == "hash123"
        assert isinstance(v.timestamp, datetime)

    def test_next_increments_version(self):
        v0 = StateVersion.initial("s1", "h0")
        v1 = v0.next("h1")
        assert v1.version_number == 1
        assert v1.session_id == "s1"
        assert v1.content_hash == "h1"

    def test_next_chain(self):
        v = StateVersion.initial("s1", "h0")
        for i in range(1, 5):
            v = v.next(f"h{i}")
        assert v.version_number == 4

    def test_to_dict_structure(self):
        ts = datetime(2026, 3, 6, 10, 0, 0)
        v = StateVersion(
            version_number=2,
            timestamp=ts,
            session_id="sess1",
            content_hash="abcdef1234567890",
        )
        d = v.to_dict()
        assert d["version"] == 2
        assert d["ts"] == ts.isoformat()
        assert d["session"] == "sess1"
        assert d["hash"] == "abcdef12"  # first 8 chars

    def test_to_dict_short_hash(self):
        v = StateVersion(
            version_number=0,
            timestamp=datetime.now(),
            session_id="s",
            content_hash="abc",
        )
        assert v.to_dict()["hash"] == "abc"

    def test_frozen(self):
        v = StateVersion.initial("s", "h")
        with pytest.raises(AttributeError):
            v.version_number = 99  # type: ignore[misc]

    def test_equality(self):
        ts = datetime(2026, 1, 1)
        a = StateVersion(0, ts, "s", "h")
        b = StateVersion(0, ts, "s", "h")
        assert a == b

    def test_inequality_different_version(self):
        ts = datetime(2026, 1, 1)
        a = StateVersion(0, ts, "s", "h")
        b = StateVersion(1, ts, "s", "h")
        assert a != b


# ---------------------------------------------------------------------------
# DeltaSection
# ---------------------------------------------------------------------------
class TestDeltaSection:
    def test_create_added(self):
        ds = DeltaSection(
            section_name="variables",
            change_type=SectionChangeType.ADDED,
            value={"x": 1},
        )
        assert ds.section_name == "variables"
        assert ds.change_type == SectionChangeType.ADDED
        assert ds.value == {"x": 1}

    def test_create_removed(self):
        ds = DeltaSection(
            section_name="old",
            change_type=SectionChangeType.REMOVED,
        )
        assert ds.value is None

    def test_create_unchanged(self):
        ds = DeltaSection(
            section_name="config",
            change_type=SectionChangeType.UNCHANGED,
        )
        assert ds.value is None

    def test_added_without_value_raises(self):
        with pytest.raises(ValueError, match="value required"):
            DeltaSection(
                section_name="x",
                change_type=SectionChangeType.ADDED,
            )

    def test_modified_without_value_raises(self):
        with pytest.raises(ValueError, match="value required"):
            DeltaSection(
                section_name="x",
                change_type=SectionChangeType.MODIFIED,
            )

    def test_removed_with_value_allowed(self):
        # REMOVED with a value is allowed (no constraint)
        ds = DeltaSection(
            section_name="x",
            change_type=SectionChangeType.REMOVED,
            value="old_data",
        )
        assert ds.value == "old_data"

    def test_is_changed_added(self):
        ds = DeltaSection("s", SectionChangeType.ADDED, value="v")
        assert ds.is_changed is True

    def test_is_changed_removed(self):
        ds = DeltaSection("s", SectionChangeType.REMOVED)
        assert ds.is_changed is True

    def test_is_changed_modified(self):
        ds = DeltaSection("s", SectionChangeType.MODIFIED, value="v")
        assert ds.is_changed is True

    def test_is_changed_unchanged(self):
        ds = DeltaSection("s", SectionChangeType.UNCHANGED)
        assert ds.is_changed is False

    def test_to_dict_with_value(self):
        ds = DeltaSection("vars", SectionChangeType.MODIFIED, value={"a": 1})
        d = ds.to_dict()
        assert d["section"] == "vars"
        assert d["change"] == "modified"
        assert d["value"] == {"a": 1}

    def test_to_dict_without_value(self):
        ds = DeltaSection("vars", SectionChangeType.UNCHANGED)
        d = ds.to_dict()
        assert d["section"] == "vars"
        assert d["change"] == "unchanged"
        assert "value" not in d

    def test_to_dict_removed_no_value(self):
        ds = DeltaSection("old", SectionChangeType.REMOVED)
        d = ds.to_dict()
        assert "value" not in d

    def test_frozen(self):
        ds = DeltaSection("s", SectionChangeType.UNCHANGED)
        with pytest.raises(AttributeError):
            ds.section_name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# StateDelta
# ---------------------------------------------------------------------------
class TestStateDelta:
    @pytest.fixture
    def sample_delta(self):
        changed = (
            DeltaSection("variables", SectionChangeType.MODIFIED, value={"x": 2}),
            DeltaSection("new_section", SectionChangeType.ADDED, value="hello"),
        )
        return StateDelta(
            from_version=1,
            to_version=3,
            changed_sections=changed,
            unchanged_sections=("config", "libraries"),
            delta_token_estimate=50,
            full_token_estimate=200,
        )

    def test_create_valid(self, sample_delta):
        assert sample_delta.from_version == 1
        assert sample_delta.to_version == 3
        assert len(sample_delta.changed_sections) == 2
        assert len(sample_delta.unchanged_sections) == 2

    def test_to_version_less_than_from_raises(self):
        with pytest.raises(ValueError, match="to_version.*< from_version"):
            StateDelta(
                from_version=5,
                to_version=2,
                changed_sections=(),
                unchanged_sections=(),
                delta_token_estimate=0,
                full_token_estimate=0,
            )

    def test_same_version_allowed(self):
        # Edge case: from == to (no changes scenario)
        d = StateDelta(
            from_version=3,
            to_version=3,
            changed_sections=(),
            unchanged_sections=("a",),
            delta_token_estimate=0,
            full_token_estimate=100,
        )
        assert d.from_version == d.to_version

    def test_negative_delta_tokens_raises(self):
        with pytest.raises(ValueError, match="delta_token_estimate cannot be negative"):
            StateDelta(
                from_version=0,
                to_version=1,
                changed_sections=(),
                unchanged_sections=(),
                delta_token_estimate=-1,
                full_token_estimate=0,
            )

    def test_negative_full_tokens_raises(self):
        with pytest.raises(ValueError, match="full_token_estimate cannot be negative"):
            StateDelta(
                from_version=0,
                to_version=1,
                changed_sections=(),
                unchanged_sections=(),
                delta_token_estimate=0,
                full_token_estimate=-1,
            )

    def test_has_changes_true(self, sample_delta):
        assert sample_delta.has_changes is True

    def test_has_changes_false(self):
        d = StateDelta(
            from_version=0,
            to_version=1,
            changed_sections=(),
            unchanged_sections=("a",),
            delta_token_estimate=10,
            full_token_estimate=100,
        )
        assert d.has_changes is False

    def test_change_count(self, sample_delta):
        assert sample_delta.change_count == 2

    def test_change_count_zero(self):
        d = StateDelta(0, 1, (), ("a",), 0, 0)
        assert d.change_count == 0

    def test_saved_tokens(self, sample_delta):
        assert sample_delta.saved_tokens == 150  # 200 - 50

    def test_saved_tokens_no_savings(self):
        d = StateDelta(0, 1, (), (), 100, 50)
        # delta > full, so saved_tokens = max(0, 50-100) = 0
        assert d.saved_tokens == 0

    def test_savings_ratio(self, sample_delta):
        assert sample_delta.savings_ratio == pytest.approx(0.75)

    def test_savings_ratio_zero_full(self):
        d = StateDelta(0, 0, (), (), 0, 0)
        assert d.savings_ratio == 0.0

    def test_savings_ratio_full_savings(self):
        d = StateDelta(0, 1, (), ("a",), 0, 100)
        assert d.savings_ratio == pytest.approx(1.0)

    def test_to_dict_structure(self, sample_delta):
        d = sample_delta.to_dict()
        assert d["from_version"] == 1
        assert d["to_version"] == 3
        assert len(d["changed_sections"]) == 2
        assert d["unchanged_sections"] == ["config", "libraries"]
        ts = d["token_savings"]
        assert ts["delta_tokens"] == 50
        assert ts["full_tokens"] == 200
        assert ts["saved_tokens"] == 150
        assert ts["savings_ratio"] == 0.75

    def test_to_dict_savings_ratio_rounded(self):
        d = StateDelta(
            from_version=0,
            to_version=1,
            changed_sections=(),
            unchanged_sections=(),
            delta_token_estimate=1,
            full_token_estimate=3,
        )
        result = d.to_dict()
        # savings = 2/3 = 0.6666... rounded to 0.667
        assert result["token_savings"]["savings_ratio"] == 0.667

    def test_frozen(self, sample_delta):
        with pytest.raises(AttributeError):
            sample_delta.from_version = 99  # type: ignore[misc]

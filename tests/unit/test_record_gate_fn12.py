"""F-N12 record gate: build_test_suite produces a clean narrative.

Auto-classifies read-only inspection keywords as record=False so the
generated suite doesn't include every Get Title / Log probe the LLM
calls between actions. Two CARVE-OUTS preserve a step even when the
keyword is inspection-only:

1. ``assign_to`` is set — the recorded suite needs ``${var}= Get Text``
   so subsequent assertions compile.
2. A named test is currently open (after ``start_test``) — the user
   explicitly opened a multi-test scope; silently dropping
   inspection-only steps inside it produced empty test cases in CI
   (F3/F4 regression that motivated this gate).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from robotmcp.components.execution.keyword_executor import (
    _INSPECTION_ONLY_KEYWORDS,
    _resolve_record_gate,
)


def _make_session(*, current_test=None):
    """Build a minimal session stub whose ``test_registry.get_current_test``
    returns ``current_test``."""
    session = MagicMock()
    registry = MagicMock()
    registry.get_current_test.return_value = current_test
    session.test_registry = registry
    return session


class TestInspectionKeywordSet:
    """The set is the contract for what gets dropped — must include the
    common page-state reads the executor sees on every probe."""

    @pytest.mark.parametrize("kw", [
        "get title",
        "get url",
        "get text",
        "get attribute",
        "get element count",
        "log",
    ])
    def test_well_known_inspection_keywords_present(self, kw):
        assert kw in _INSPECTION_ONLY_KEYWORDS

    @pytest.mark.parametrize("kw", [
        "click",
        "fill text",
        "go to",
        "new page",
        "evaluate javascript",  # intentionally NOT inspection-only
    ])
    def test_action_keywords_not_in_set(self, kw):
        assert kw not in _INSPECTION_ONLY_KEYWORDS


class TestExplicitRecordOverride:
    """Explicit record=True/False bypasses all auto-classification."""

    def test_record_true_records_inspection_keyword(self):
        assert _resolve_record_gate(
            keyword="Get Title", record=True, assign_to=None,
            session=_make_session(),
        ) is True

    def test_record_false_drops_action_keyword(self):
        assert _resolve_record_gate(
            keyword="Click", record=False, assign_to=None,
            session=_make_session(),
        ) is False

    def test_record_false_overrides_assign_to(self):
        # Explicit drop wins even with assign_to set.
        assert _resolve_record_gate(
            keyword="Get Text", record=False, assign_to="result",
            session=_make_session(),
        ) is False

    def test_record_false_overrides_named_test(self):
        assert _resolve_record_gate(
            keyword="Get Title", record=False, assign_to=None,
            session=_make_session(current_test=MagicMock(name="Login Test")),
        ) is False


class TestAutoClassification:
    """record=None (default): auto-classify based on the keyword."""

    @pytest.mark.parametrize("kw", [
        "Get Title", "Get Url", "Get Text", "Log", "Get Attribute",
    ])
    def test_inspection_keyword_dropped(self, kw):
        assert _resolve_record_gate(
            keyword=kw, record=None, assign_to=None,
            session=_make_session(),
        ) is False

    @pytest.mark.parametrize("kw", [
        "Click", "Fill Text", "Go To", "New Page", "Wait For Elements State",
    ])
    def test_action_keyword_recorded(self, kw):
        assert _resolve_record_gate(
            keyword=kw, record=None, assign_to=None,
            session=_make_session(),
        ) is True

    @pytest.mark.parametrize("kw", [
        "GET TITLE", "get title", "Get Title", "  Get Title  ",
    ])
    def test_case_and_outer_whitespace_insensitive(self, kw):
        # Inspection set is lower-case single-spaced; matcher applies
        # .lower().strip() so any common casing of "Get Title" matches.
        assert _resolve_record_gate(
            keyword=kw, record=None, assign_to=None, session=_make_session(),
        ) is False


class TestAssignToCarveOut:
    """assign_to is load-bearing: the recorded suite needs ``${var}= ...``."""

    def test_get_text_with_assign_to_is_recorded(self):
        assert _resolve_record_gate(
            keyword="Get Text", record=None, assign_to="result",
            session=_make_session(),
        ) is True

    def test_list_assign_to_also_carves_out(self):
        assert _resolve_record_gate(
            keyword="Get Title", record=None, assign_to=["a", "b"],
            session=_make_session(),
        ) is True


class TestNamedTestCarveOut:
    """The CI F3/F4 fix: when a named test is open, never drop steps."""

    def test_get_title_inside_named_test_is_recorded(self):
        session = _make_session(current_test=MagicMock(name="Login Test"))
        assert _resolve_record_gate(
            keyword="Get Title", record=None, assign_to=None, session=session,
        ) is True

    def test_log_inside_named_test_is_recorded(self):
        session = _make_session(current_test=MagicMock(name="Smoke"))
        assert _resolve_record_gate(
            keyword="Log", record=None, assign_to=None, session=session,
        ) is True

    def test_get_title_outside_named_test_is_dropped(self):
        # Sanity check the carve-out's negative side: no current test ->
        # auto-classification fires and drops inspection-only keywords.
        session = _make_session(current_test=None)
        assert _resolve_record_gate(
            keyword="Get Title", record=None, assign_to=None, session=session,
        ) is False


class TestDefensiveBehaviour:
    """A malformed session must never crash the executor."""

    def test_missing_test_registry_falls_through(self):
        session = MagicMock(spec=[])  # no test_registry attribute
        # Should treat as "no current test" and fall through to
        # classification.
        assert _resolve_record_gate(
            keyword="Click", record=None, assign_to=None, session=session,
        ) is True
        assert _resolve_record_gate(
            keyword="Get Title", record=None, assign_to=None, session=session,
        ) is False

    def test_test_registry_raises_falls_through(self):
        session = MagicMock()
        session.test_registry.get_current_test.side_effect = RuntimeError("boom")
        # Exception in registry must not propagate.
        assert _resolve_record_gate(
            keyword="Get Title", record=None, assign_to=None, session=session,
        ) is False

"""F-N12 record gate: build_test_suite produces a clean narrative.

Auto-classifies AMBIENT-STATE reads (Get Title, Get Url, Get Viewport
Size, ...) as record=False so the generated suite doesn't include
every page-state probe the LLM calls between actions.

Crucially, locator-taking getters (Get Text, Get Value, Get Attribute,
Get Element Count, ...) are NOT in the auto-drop set, even though
they "read" — in Robot Framework they double as implicit existence
assertions (raise on missing element) and as explicit assertions via
the RF assertion-engine pattern (``Get Text  id=foo  ==  bar``).
Silently dropping such calls would remove load-bearing assertions
from the generated suite. See TestImplicitAssertionsArePreserved.

Two CARVE-OUTS preserve a step even when the keyword is in the
ambient-state set:

1. ``assign_to`` is set — the recorded suite needs ``${var}= ...``
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
    """The set is the contract for what gets auto-dropped.

    Only ambient-state reads (no locator, never throw, no implicit
    assertion semantics) belong in this set — see the module-level
    rationale in keyword_executor.py."""

    @pytest.mark.parametrize("kw", [
        "get title",          # Browser page title (no locator)
        "get url",            # Browser page URL (no locator)
        "get viewport size",  # Browser viewport (no locator)
        "get location",       # SeleniumLibrary page URL (no locator)
        "get capability",     # AppiumLibrary session metadata
        "get window size",    # AppiumLibrary window metadata
    ])
    def test_ambient_reads_present(self, kw):
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

    @pytest.mark.parametrize("kw", [
        # Locator-taking getters double as implicit existence assertions
        # and as RF assertion-engine targets (Get Text  loc  ==  val).
        # They must NOT be auto-dropped.
        "get text",
        "get value",
        "get attribute",
        "get element count",
        "get element states",
        "get property",
        "get classes",
        "get bounding box",
        "get element attribute",
        "get element size",
        "get list selected labels",
    ])
    def test_locator_taking_getters_not_in_set(self, kw):
        assert kw not in _INSPECTION_ONLY_KEYWORDS, (
            f"{kw!r} is a locator-taking getter — it doubles as an implicit "
            "existence assertion (raises on missing element) and as an RF "
            "assertion-engine target. Dropping it would silently remove "
            "load-bearing assertions from the generated suite."
        )

    @pytest.mark.parametrize("kw", ["log", "log to console", "log many"])
    def test_log_keywords_not_in_set(self, kw):
        # Log emits intentional narrative; existing tests seed sessions
        # with Log calls. Dropping it breaks both narrative and tests.
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
        "Get Title", "Get Url", "Get Viewport Size",
        "Get Location", "Get Capability", "Get Window Size",
    ])
    def test_inspection_keyword_dropped(self, kw):
        assert _resolve_record_gate(
            keyword=kw, record=None, assign_to=None,
            session=_make_session(),
        ) is False

    @pytest.mark.parametrize("kw", [
        "Click", "Fill Text", "Go To", "New Page", "Wait For Elements State",
        # Action keywords PLUS locator-taking getters (implicit assertions):
        "Get Text", "Get Value", "Get Attribute", "Get Element Count",
        "Get Element States", "Get Property", "Get Classes",
        "Get Element Attribute", "Get List Selected Labels",
        # Log is intentional narrative, not inspection:
        "Log", "Log To Console", "Log Many",
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


class TestImplicitAssertionsArePreserved:
    """Locator-taking getters are recorded by default — they double as
    implicit existence assertions (raise on missing element) and as
    explicit assertions via the RF assertion-engine pattern
    (``Get Text  id=foo  ==  bar``). Silently dropping them would
    remove load-bearing test logic from the generated suite.

    Pinned here as a separate class (not just inside
    test_action_keyword_recorded) so the intent is unmistakable to
    anyone tempted to "tidy up" the gate by re-adding these keywords
    to _INSPECTION_ONLY_KEYWORDS."""

    @pytest.mark.parametrize("kw", [
        "Get Text",
        "Get Value",
        "Get Attribute",
        "Get Element Count",
        "Get Element Attribute",
        "Get Element Size",
        "Get Element Tag Name",
        "Get List Selected Labels",
        "Get List Selected Values",
        "Get Property",
        "Get Style",
        "Get Classes",
        "Get Bounding Box",
        "Get Table Cell Element",
        "Get Element States",
    ])
    def test_locator_taking_getter_is_recorded(self, kw):
        # No assign_to, no named test — pure auto-classification.
        # MUST record because the keyword may be performing an implicit
        # or explicit assertion.
        assert _resolve_record_gate(
            keyword=kw, record=None, assign_to=None,
            session=_make_session(),
        ) is True

    @pytest.mark.parametrize("kw", ["Log", "Log To Console", "Log Many"])
    def test_log_keywords_are_recorded(self, kw):
        # Log is intentional narrative, not a probe. Existing tests rely
        # on Log to seed a session (test_build_test_suite_escapes_hash_
        # locators) — dropping it broke them in CI.
        assert _resolve_record_gate(
            keyword=kw, record=None, assign_to=None,
            session=_make_session(),
        ) is True


class TestAssignToCarveOut:
    """assign_to is load-bearing: the recorded suite needs ``${var}= ...``.

    With the tightened inspection set most getters are already recorded
    by default (implicit-assertion semantics), so these tests use the
    ambient-state getters that ARE dropped by default. The carve-out
    must force-record them when assign_to is set."""

    def test_get_title_with_assign_to_is_recorded(self):
        # Get Title would be dropped without assign_to (ambient state);
        # carve-out preserves it because the suite needs ``${title}= ...``.
        assert _resolve_record_gate(
            keyword="Get Title", record=None, assign_to="result",
            session=_make_session(),
        ) is True

    def test_get_url_with_list_assign_to_also_carves_out(self):
        assert _resolve_record_gate(
            keyword="Get Url", record=None, assign_to=["a", "b"],
            session=_make_session(),
        ) is True


class TestNamedTestCarveOut:
    """The CI F3/F4 fix: when a named test is open, never drop steps."""

    def test_get_title_inside_named_test_is_recorded(self):
        session = _make_session(current_test=MagicMock(name="Login Test"))
        assert _resolve_record_gate(
            keyword="Get Title", record=None, assign_to=None, session=session,
        ) is True

    def test_get_url_inside_named_test_is_recorded(self):
        # Get Url is ambient-state (dropped by default) — the carve-out
        # must preserve it inside a named-test scope.
        session = _make_session(current_test=MagicMock(name="Smoke"))
        assert _resolve_record_gate(
            keyword="Get Url", record=None, assign_to=None, session=session,
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

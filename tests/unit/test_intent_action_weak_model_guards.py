"""Tests for the intent_action weak-model guards.

Every case here is taken from a REAL trace: weekly E2E run 36120985462, where
MiniMax-M2.7 made 23 consecutive intent_action calls of the shape

    {'intent': 'navigate', 'target': None, 'session_id': None}

having already created a session and imported Browser into it. The model's own
reasoning trace identified the cause ("I've been passing the literal string "null"
instead of the JSON null value") and misdiagnosed the effect, because the error it
got back was about library imports rather than about its arguments.

Run: uv run pytest tests/unit/test_intent_action_weak_model_guards.py -q
"""
from __future__ import annotations

import pytest

from robotmcp.domains.intent import weak_model_guards as g


class _Session:
    def __init__(self, libs):
        self.imported_libraries = list(libs)


# =============================================================================
# coerce_absent — a textual null is not a locator
# =============================================================================


@pytest.mark.parametrize(
    "raw", ["null", "None", "NULL", "none", " null ", "", "  ", "undefined", "nil", "N/A"]
)
def test_textual_nulls_become_real_none(raw):
    """These are TRUTHY strings, so without coercion `session_id or "default"` keeps
    them and `target is None` is False - a literal "None" was accepted as a locator."""
    assert g.coerce_absent(raw) is None


@pytest.mark.parametrize(
    "raw",
    [
        "text=None found",       # a locator that merely CONTAINS the word
        "#null-banner",          # an id that happens to say null
        "https://example.com",
        "None of the above",     # visible link text
        0,
        False,
    ],
)
def test_real_values_are_untouched(raw):
    """Only a whole-value sentinel is absence. Over-eager matching would break real
    locators, which is a worse failure than the one being fixed."""
    assert g.coerce_absent(raw) == raw


# =============================================================================
# requires_target — reachable BEFORE library resolution
# =============================================================================


@pytest.mark.parametrize(
    "verb", ["navigate", "click", "fill", "hover", "select", "assert_visible", "wait_for"]
)
def test_action_verbs_require_a_target(verb):
    assert g.requires_target(verb) is True


@pytest.mark.parametrize("mode", ["url", "title", "text", "attribute", "count", None])
def test_extract_is_left_to_the_mapping_layer_whatever_the_mode(mode):
    """This assertion originally said extract REQUIRES a target for text/attribute/
    count/value. Checking the guard against the 33 real mappings proved that wrong:
    all four libraries declare requires_target=False for extract, because the
    requirement depends on `mode`, which a mapping cannot see.

    Pre-rejecting it would have made this guard refuse calls the system accepts - a
    worse bug than the one being fixed. The guard defers; the mapping still validates.
    """
    assert g.requires_target("extract", mode) is False


def test_the_table_is_derived_not_hardcoded():
    """Guards the reasoning. A hand-written list drifted from the mappings in BOTH
    directions - it over-required `extract` and missed `ensure_focused` entirely."""
    table = g._target_required_by_verb()

    assert table, "derivation must actually find the built-in mappings"
    assert table.get("ensure_focused") is True, "a verb the hand-written list missed"
    assert table.get("extract") is False
    for verb in ("navigate", "click", "fill", "hover", "select", "assert_visible"):
        assert table.get(verb) is True, verb


def test_an_unknown_verb_is_never_pre_rejected():
    """Only the mapping layer can speak for a verb this guard has no data on."""
    assert g.requires_target("some_future_verb") is False


def test_missing_target_error_names_the_parameter_and_shows_a_usable_example():
    """The point of this message is that it replaces 'Cannot determine target library
    ... ensure a library is imported', which pointed at the wrong thing entirely."""
    msg = g.missing_target_error("navigate")

    assert "target" in msg
    assert "https://example.com" in msg, "must show a concrete value, not just a name"
    assert "null" in msg.lower(), "the literal-null case is why we are here"
    assert "library" not in msg.lower(), "must not repeat the misleading diagnosis"


def test_extract_error_explains_the_one_exemption():
    msg = g.missing_target_error("extract", "text")
    assert 'mode="url"' in msg and 'mode="title"' in msg


# =============================================================================
# resolve_session_id — never silently target the wrong session
# =============================================================================


def test_the_exact_failure_from_the_trace_is_now_resolved():
    """M2.7's situation: it created a session with Browser loaded, then passed
    session_id=None. That used to resolve to 'default' - which has no libraries - and
    the resulting error blamed library imports."""
    sessions = {"babd990f": _Session(["Browser", "BuiltIn"])}

    sid, note, err = g.resolve_session_id(None, sessions)

    assert err is None
    assert sid == "babd990f", "must reach the session that can actually resolve intents"
    assert note and "babd990f" in note, "the caller has to learn what happened"


def test_a_literal_null_string_behaves_as_omission_not_as_a_session_name():
    sessions = {"s1": _Session(["Browser"])}
    assert g.resolve_session_id("null", sessions)[0] == "s1"
    assert g.resolve_session_id("None", sessions)[0] == "s1"


def test_an_explicit_session_is_honoured_without_a_note():
    sessions = {"s1": _Session(["Browser"]), "default": _Session(["Browser"])}
    sid, note, err = g.resolve_session_id("s1", sessions)
    assert (sid, note, err) == ("s1", None, None)


def test_default_is_used_when_it_is_genuinely_the_loaded_one():
    sessions = {"default": _Session(["Browser"])}
    sid, note, err = g.resolve_session_id(None, sessions)
    assert sid == "default" and err is None and note is None


def test_ambiguity_is_refused_rather_than_guessed():
    """Guessing among several loaded sessions would trade a clear failure for a silent
    wrong-session success - strictly worse, and much harder to debug."""
    sessions = {"a": _Session(["Browser"]), "b": _Session(["SeleniumLibrary"])}

    sid, note, err = g.resolve_session_id(None, sessions)

    assert sid is None
    assert err and "a" in err and "b" in err, "name the candidates so it is actionable"


def test_sessions_without_libraries_are_never_inferred():
    """An empty session cannot resolve any intent, so offering it would reproduce the
    original confusion rather than fix it."""
    sessions = {"empty": _Session([]), "loaded": _Session(["Browser"])}

    sid, _, err = g.resolve_session_id(None, sessions)

    assert sid == "loaded" and err is None


def test_unknown_session_error_lists_what_does_exist():
    sessions = {"real": _Session(["Browser"])}

    sid, _, err = g.resolve_session_id("typo-session", sessions)

    assert sid is None
    assert "typo-session" in err and "real" in err
    assert "analyze_scenario" in err, "point at where a valid id comes from"


def test_no_sessions_at_all_falls_through_to_default():
    """With nothing loaded, the existing 'import a library' guidance IS the right
    message - so this must not short-circuit it with a different error."""
    sid, note, err = g.resolve_session_id(None, {})
    assert sid == "default" and err is None and note is None


# =============================================================================
# End-to-end through the real tool — the shapes MiniMax-M2.7 actually sent
# =============================================================================


@pytest.fixture
def clean_sessions():
    """The session manager is process-global; leaking sessions between tests would
    change which branch resolve_session_id takes."""
    from robotmcp.server import execution_engine

    sm = execution_engine.session_manager
    saved = dict(sm.sessions)
    sm.sessions.clear()
    try:
        yield sm
    finally:
        sm.sessions.clear()
        sm.sessions.update(saved)


def _call(**kwargs):
    import asyncio

    from robotmcp.compat.fastmcp_compat import get_tool_fn
    from robotmcp.server import intent_action

    fn = get_tool_fn(intent_action)
    # new_event_loop + close, never get_event_loop: the latter fails inside the full
    # suite on 3.13 once another test has closed the ambient loop.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(fn(**kwargs))
    finally:
        loop.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(intent="navigate", target=None, session_id=None),
        dict(intent="navigate", target="null", session_id="null"),
        dict(intent="click", target=None, session_id=None),
        dict(intent="fill", target=None, value=None, session_id=None),
    ],
)
def test_missing_target_reports_the_target_not_the_library(kwargs, clean_sessions):
    """Regression for the trace. Before this change these returned "Cannot determine
    target library ... ensure a web or mobile library is imported", which sent the
    model off importing libraries it had already imported."""
    result = _call(**kwargs)

    assert result["success"] is False
    err = result["error"]
    assert "requires a target" in err, err
    assert "target=" in err, "must show a concrete example"
    assert "library" not in err.lower(), "the old misleading diagnosis must be gone"


def test_inferred_session_is_reported_even_when_the_call_then_fails(clean_sessions):
    """A caller who omitted session_id has to be able to tell 'ran in the wrong
    session' from 'the keyword failed' - so the note belongs on the failure path."""
    s = clean_sessions.create_session("only-loaded")
    s.imported_libraries = ["BuiltIn", "Browser"]

    result = _call(intent="click", target="#submit", session_id=None)

    assert result.get("session_id") == "only-loaded"
    assert "only-loaded" in (result.get("session_note") or "")


def test_unknown_session_is_refused_with_the_real_ones_listed(clean_sessions):
    s = clean_sessions.create_session("real-session")
    s.imported_libraries = ["BuiltIn", "Browser"]

    result = _call(intent="click", target="#submit", session_id="made-up")

    assert result["success"] is False
    assert "made-up" in result["error"] and "real-session" in result["error"]

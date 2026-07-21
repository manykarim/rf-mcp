"""Tests for change: fix-suite-arg-escape-roundtrip.

Escape-aware backslash doubling: RF drops the backslash for every escape it does not
recognize (\\d -> d, \\W -> W), so a literal backslash written into a generated .robot is
corrupted on parse. Double only the unrecognized ones; preserve RF's real escapes
(\\n \\r \\t, \\x \\u \\U), \\\\, and the syntax escapes the tool relies on (\\= \\# \\$ ...).
Idempotent, and variable references are untouched.
"""
from __future__ import annotations

from robot.utils.escaping import unescape

from robotmcp.components.test_builder import TestBuilder, _escape_aware_backslashes


def _esc(v: str) -> str:
    return TestBuilder()._escape_robot_argument(v)


def _roundtrip(v: str) -> str:
    return unescape(_esc(v))


# ── regexes and relative paths round-trip (the fix) ─────────────────────────
def test_regex_backslash_survives():
    for rgx in [r"\d+", r"\w\s", r"\W", r"\D\S", r"\b"]:
        assert _roundtrip(rgx) == rgx, rgx


def test_relative_windows_path_round_trips():
    # \o / \f are unrecognized -> doubled -> survive
    assert _roundtrip(r"data\output.txt") == r"data\output.txt"
    assert _roundtrip(r"logs\app.log") == r"logs\app.log"


def test_variable_reference_with_literal_backslash_preserved():
    assert _roundtrip(r"${dir}\file.txt") == r"${dir}\file.txt"


def test_escaped_variable_stays_literal():
    # \$ is an intended syntax escape -> preserved -> RF yields a literal ${x}
    assert _roundtrip(r"\${x}") == r"${x}"


# ── idempotency + existing contract preserved (must not regress) ────────────
def test_idempotent():
    for v in [r"\d+", r"data\output.txt", r"-env:X\=Y", "a\\nb", r"${dir}\file"]:
        once = _esc(v)
        assert _esc(once) == once, v


def test_dash_guard_and_control_escapes_unchanged():
    # dash-guard \= preserved; real newline still escaped to \n text
    assert _esc(r"-env:X\=Y") == r"-env:X\=Y"
    assert _esc("line1\nline2") == "line1\\nline2"
    # \n TEXT preserved as an escape (RF -> newline)
    assert _esc("a\\nb") == "a\\nb"


def test_drive_letter_path_still_forward_slashed():
    # prior change still applies BEFORE this pass (no backslashes left to double)
    assert _esc(r"C:\WINDOWS\system32\calc.exe") == "C:/WINDOWS/system32/calc.exe"


def test_urls_and_flags_untouched():
    for v in ["https://example.com/a?b=c", "/w", "alias=proc", "@{items}", "${{ {'k': 1} }}"]:
        assert _esc(v) == v


# ── the helper directly (idempotency of the pass itself) ────────────────────
def test_helper_pass_idempotent_and_targeted():
    assert _escape_aware_backslashes(r"\d") == r"\\d"
    assert _escape_aware_backslashes(r"\\d") == r"\\d"       # already-doubled kept
    assert _escape_aware_backslashes(r"\n") == r"\n"          # real escape preserved
    assert _escape_aware_backslashes("plain") == "plain"      # no backslash -> no-op

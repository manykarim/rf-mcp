"""Tests for change: fix-suite-path-escaping.

RF's escape processing corrupts a literal Windows path written into a generated .robot
(C:\\WINDOWS -> C:WINDOWS, C:\\name -> embedded newline). The fix normalizes a Windows
DRIVE-LETTER path to forward slashes — valid on Windows (Start Process/subprocess +
OperatingSystem keywords) and POSIX, and RF-escape-safe. It touches ONLY the drive-letter
shape; everything else is left to the pre-existing escaping (unchanged).
"""
from __future__ import annotations

from robot.utils.escaping import unescape

from robotmcp.components.test_builder import TestBuilder


def _esc(v: str) -> str:
    return TestBuilder()._escape_robot_argument(v)


def _roundtrip(v: str) -> str:
    """What RF passes to the keyword after our escaping + RF's own unescape."""
    return unescape(_esc(v))


# ── drive-letter Windows path -> forward slashes (the corruption fix) ────────
def test_windows_drive_path_becomes_forward_slash():
    assert _esc(r"C:\WINDOWS\system32\calc.exe") == "C:/WINDOWS/system32/calc.exe"
    # and RF round-trips the forward-slash form unchanged (no corruption)
    assert _roundtrip(r"C:\WINDOWS\system32\calc.exe") == "C:/WINDOWS/system32/calc.exe"


def test_windows_path_with_n_and_t_no_longer_corrupts():
    # \Users\name\report would embed a newline (\n=newline) without the fix
    assert _roundtrip(r"C:\Users\name\report.txt") == "C:/Users/name/report.txt"


def test_already_forward_slash_drive_path_unchanged():
    assert _esc("C:/data/report.txt") == "C:/data/report.txt"


def test_lowercase_drive_letter_normalized():
    assert _esc(r"d:\tmp\out.log") == "d:/tmp/out.log"


# ── the normalization is narrow: non-drive-letter values are NOT rewritten ──
def test_flags_urls_and_relative_values_not_separator_rewritten(monkeypatch):
    # None of these are a drive-letter path, so the path-normalization step leaves them
    # exactly as the pre-existing escaping did (this fix must not regress them).
    for v in ["/w", "-flag", "https://example.com/a?b=c", "alias=proc"]:
        assert _esc(v) == v


def test_variable_and_inline_eval_refs_untouched():
    for v in ["${resp.json()}", "@{items}", "&{headers}", "%{HOME}", "json=${{ {'k': 1} }}"]:
        assert _esc(v) == v


# ── regression guard: the pre-existing escape behavior is unchanged ─────────
def test_dash_arg_escaping_still_idempotent():
    # dash-guard escape of the first '=' still applies once and is idempotent
    assert _esc("-env:UserInstallation=file:///tmp/p") == "-env:UserInstallation\\=file:///tmp/p"
    assert _esc("-env:UserInstallation\\=file:///tmp/p") == "-env:UserInstallation\\=file:///tmp/p"


def test_real_control_chars_still_escaped():
    assert _esc("line1\nline2") == "line1\\nline2"
    assert _esc("a\tb") == "a\\tb"


# NOTE (deferred): round-trip escaping for non-drive-letter backslash values — regexes
# (\d+ -> d+), relative Windows paths (data\file), and UNC (\\srv) — is NOT fixed here.
# Blanket backslash-doubling is not idempotent with this function's existing escape
# contract (it re-corrupts the dash-guard's \= and intentional \n). That needs an
# escape-aware design; see the change's design.md.

"""Unit tests for the RobotCode-native error hinting engine."""

from __future__ import annotations

from pathlib import Path

from robotmcp.hints import HintContext, HintEngine


def test_browser_evaluate_js_single_arg_hint() -> None:
    engine = HintEngine()
    hints = engine.diagnose(
        HintContext(
            keyword="Evaluate JavaScript    () => document.title",
            error_text="[ FAIL ] Error: Evaluating JavaScript: Unexpected token '=' in selector",
        )
    )
    assert len(hints) >= 1
    assert "Evaluate JavaScript    ${None}    () => document.title" in hints[0].fix_examples


def test_browser_strict_mode_violation_hint() -> None:
    engine = HintEngine()
    hints = engine.diagnose(
        HintContext(
            keyword="Click    css=.btn",
            error_text="Error: locator.click: Error: strict mode violation: locator('.btn') resolved to 3 elements",
        )
    )
    assert len(hints) >= 1
    assert hints[0].id == "browser-strict-mode-violation"
    assert any("Get Aria Snapshot" in ex for ex in hints[0].fix_examples)
    assert any("nth=0" in ex for ex in hints[0].fix_examples)


def test_repl_section_header_hint() -> None:
    engine = HintEngine()
    hints = engine.diagnose(
        HintContext(
            keyword="*** Settings ***",
            error_text="[ FAIL ] No keyword with name '*** Settings ***' found.",
        )
    )
    assert len(hints) >= 1
    assert hints[0].id == "repl-section-header"
    assert any("Import Library" in ex for ex in hints[0].fix_examples)


def test_requests_and_dict_bracket_hint() -> None:
    engine = HintEngine()
    hints = engine.diagnose(
        HintContext(
            keyword="Log    ${resp.body}",
            error_text="Resolving variable '${resp.body}' failed: AttributeError: 'dict' object has no attribute 'body'",
        )
    )
    assert len(hints) >= 1
    assert hints[0].id == "rf-dict-bracket-access"
    assert any("${data}[key]" in ex for ex in hints[0].fix_examples)


def test_custom_project_hints_toml_override(tmp_path: Path) -> None:
    hints_toml = tmp_path / "hints.toml"
    hints_toml.write_text(
        """
[[hints]]
id = "corp-sso-expired"
title = "Corporate SSO token expired"
error_pattern = "CorpAuthExpiredError"
keyword_pattern = ".*"
message = "Refresh the SSO token using the Auth resource."
fix_examples = ["Import Resource    resources/auth.resource", "Refresh SSO Token"]
relevance = 99
""",
        encoding="utf-8",
    )
    engine = HintEngine(hints_file=hints_toml)
    hints = engine.diagnose(
        HintContext(
            keyword="Call Internal Endpoint",
            error_text="[ FAIL ] CorpAuthExpiredError: token TTL exceeded",
        )
    )
    assert len(hints) >= 1
    assert hints[0].id == "corp-sso-expired"
    assert hints[0].relevance == 99
    assert "Refresh SSO Token" in hints[0].fix_examples

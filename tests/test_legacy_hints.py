"""Hint rules carried over from the legacy 0.36.0 engine, checked with the legacy error texts."""

from __future__ import annotations

import pytest

from robotmcp.hints import HintContext, HintEngine

ENGINE = HintEngine()


def _ids(keyword: str, error_text: str, *, arguments=None, libraries=None) -> list[str]:
    hints = ENGINE.diagnose(
        HintContext(
            keyword=keyword,
            arguments=arguments or [],
            error_text=error_text,
            libraries=libraries or [],
        ),
        max_hints=5,
    )
    return [h.id for h in hints]


@pytest.mark.parametrize(
    ("keyword", "error_text", "rule"),
    [
        ("Click Element    xpath://div[", "invalid selector: not a valid XPath expression", "locator-invalid-selector"),
        ("Click    css=a::b", "Unsupported token while parsing css selector", "locator-invalid-selector"),
        ("Click    id=hidden", "element is outside of the viewport", "element-outside-viewport"),
        ("Click Element    id:row", "stale element reference: element is not attached to the page document", "element-stale"),
        ("Click    id=row", "element is not attached", "element-stale"),
        ("Click Element    id:save", "unexpected alert open", "unexpected-alert"),
        ("Select Frame    id:pay", "NoSuchFrameException: no such frame", "frame-not-found"),
        ("Switch Window    NEW", "NoSuchWindowException: no such window", "window-not-found"),
    ],
)
def test_legacy_error_text_triggers_rule(keyword: str, error_text: str, rule: str) -> None:
    assert rule in _ids(keyword, error_text)


def test_browser_role_prefix_fires_for_browser_only() -> None:
    assert "browser-role-prefix-misuse" in _ids(
        "Click", "element not visible", arguments=["button=Calculate"], libraries=["Browser"]
    )
    assert "browser-role-prefix-misuse" in _ids(
        "Click    link=Click here", "TimeoutError: waiting for locator('link=Click here')"
    )
    assert "browser-role-prefix-misuse" not in _ids(
        "Click Link    link=Click here", "element not visible", libraries=["SeleniumLibrary"]
    )
    assert "browser-role-prefix-misuse" not in _ids(
        "Click", "element not visible", arguments=["css=button.calc"], libraries=["Browser"]
    )


def test_role_prefix_outranks_invalid_selector() -> None:
    ids = _ids(
        "Click", "Unsupported token while parsing css selector", arguments=["button=Calculate"], libraries=["Browser"]
    )
    assert ids.index("browser-role-prefix-misuse") < ids.index("locator-invalid-selector")


def test_process_named_argument_misparse() -> None:
    libre = "-env:UserInstallation=file:///tmp/profile"
    hints = ENGINE.diagnose(
        HintContext(keyword="Start Process", arguments=["soffice", libre], error_text="Process launch failed")
    )
    hint = next(h for h in hints if h.id == "process-named-arg-misparse")
    assert "named argument" in hint.title
    assert "\\=" in hint.message
    assert any("-env:UserInstallation\\=file:///tmp/profile" in ex for ex in hint.fix_examples)

    assert "process-named-arg-misparse" not in _ids(
        "Start Process", "Process launch failed", arguments=["soffice", "--writer"]
    )
    assert "process-named-arg-misparse" not in _ids(
        "Start Process", "Process launch failed", arguments=["soffice", "-env:X\\=file:///tmp"]
    )
    assert "process-named-arg-misparse" not in _ids(
        "Start Process", "Process launch failed", arguments=["app", "cwd=/tmp", "shell=True"]
    )


def test_mobile_context_mismatch_needs_appium() -> None:
    assert "mobile-context-mismatch" in _ids(
        "Click Element    id=login", "UnknownCommandError: unknown command", libraries=["AppiumLibrary"]
    )
    assert "mobile-context-mismatch" not in _ids(
        "Click Element    id=login", "UnknownCommandError: unknown command", libraries=["SeleniumLibrary"]
    )

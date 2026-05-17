"""LocatorArgIntrospector: library-aware locator-arg detection.

Replaces a hardcoded "no-locator keywords" list with introspection of
each keyword's actual argument signature. Library-specific canonical
arg names:

    Browser           any arg starting with "selector"
    SeleniumLibrary   exact arg "locator"
    AppiumLibrary     exact arg "locator" or "element"

Used as a CONFIDENT VETO over the positive list of element-interaction
keywords in ``KeywordExecutor._requires_pre_validation``. The veto fires
only when the introspector returns a definitive False (the keyword
resolved to a specific library whose signature has no locator-style
arg). Ambiguous / unresolved lookups (None) do NOT veto — preserves
the curated positive list against false negatives like the
``clear element value`` over-veto case.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from robotmcp.components.execution.locator_arg_introspection import (
    LIBRARY_LOCATOR_PATTERNS,
    LocatorArgIntrospector,
    _strip_arg_annotation,
    args_contain_locator,
)


class TestStripArgAnnotation:
    """Robot Framework keyword discovery stores arg names with type hints
    and defaults — strip them to get the bare name."""

    @pytest.mark.parametrize("raw,expected", [
        ("selector", "selector"),
        ("selector: str", "selector"),
        ("button: MouseButton = left", "button"),
        ("reason=None", "reason"),
        ("*varargs", "varargs"),
        ("**kwargs", "kwargs"),
        ("  selector  ", "selector"),
        ("", ""),
    ])
    def test_strip_arg_annotation(self, raw, expected):
        assert _strip_arg_annotation(raw) == expected

    def test_non_string_input_returns_empty(self):
        assert _strip_arg_annotation(None) == ""
        assert _strip_arg_annotation(42) == ""


class TestArgsContainLocator:
    """Library-specific arg-name pattern matching."""

    def test_browser_recognises_selector(self):
        assert args_contain_locator("Browser", ["selector", "value"]) is True

    def test_browser_recognises_selector_with_type_annotation(self):
        assert args_contain_locator(
            "Browser", ["selector: str", "button: MouseButton = left"]
        ) is True

    def test_browser_recognises_selector_prefix(self):
        # e.g. Drag And Drop has selector_from and selector_to
        assert args_contain_locator(
            "Browser", ["selector_from", "selector_to"]
        ) is True

    def test_browser_rejects_unrelated_args(self):
        assert args_contain_locator("Browser", ["key", "action"]) is False
        assert args_contain_locator("Browser", ["url"]) is False
        assert args_contain_locator("Browser", []) is False

    def test_selenium_recognises_locator(self):
        assert args_contain_locator("SeleniumLibrary", ["locator"]) is True

    def test_selenium_rejects_unrelated_args(self):
        assert args_contain_locator("SeleniumLibrary", ["url"]) is False
        # selector is NOT the canonical Selenium arg name
        assert args_contain_locator("SeleniumLibrary", ["selector"]) is False

    def test_appium_recognises_locator_or_element(self):
        assert args_contain_locator("AppiumLibrary", ["locator"]) is True
        assert args_contain_locator("AppiumLibrary", ["element"]) is True

    def test_unknown_library_returns_false(self):
        assert args_contain_locator("UnknownLib", ["locator"]) is False
        assert args_contain_locator("", ["locator"]) is False

    def test_patterns_dict_is_exposed_for_inspection(self):
        assert "Browser" in LIBRARY_LOCATOR_PATTERNS
        assert "SeleniumLibrary" in LIBRARY_LOCATOR_PATTERNS
        assert "AppiumLibrary" in LIBRARY_LOCATOR_PATTERNS


class TestLocatorArgIntrospectorVetoBehaviour:
    """The CONFIDENT-VETO contract: only definitive False vetoes; None
    is "I don't know" and falls through to the caller's policy."""

    def _make_kd(self, *, find_keyword_return):
        kd = MagicMock()
        kd.find_keyword.return_value = find_keyword_return
        kd.get_all_keywords.return_value = []
        return kd

    def test_definitive_true_when_locator_arg_present(self):
        kd = self._make_kd(find_keyword_return=MagicMock(
            library="Browser", args=["selector: str", "button"],
        ))
        intro = LocatorArgIntrospector(keyword_discovery=kd)
        assert intro.keyword_takes_locator("Click") is True

    def test_definitive_false_when_keyword_resolved_no_locator_arg(self):
        kd = self._make_kd(find_keyword_return=MagicMock(
            library="Browser", args=["key", "action"],  # Keyboard Key signature
        ))
        intro = LocatorArgIntrospector(keyword_discovery=kd)
        assert intro.keyword_takes_locator("Keyboard Key") is False

    def test_none_when_keyword_unresolved(self):
        kd = self._make_kd(find_keyword_return=None)
        intro = LocatorArgIntrospector(keyword_discovery=kd)
        # No fallback either:
        assert intro.keyword_takes_locator("Some Made Up Keyword") is None

    def test_none_when_no_keyword_discovery(self):
        intro = LocatorArgIntrospector(keyword_discovery=None)
        assert intro.keyword_takes_locator("Click") is None

    def test_swallows_find_keyword_exception(self):
        kd = MagicMock()
        kd.find_keyword.side_effect = RuntimeError("transient")
        kd.get_all_keywords.return_value = []
        intro = LocatorArgIntrospector(keyword_discovery=kd)
        assert intro.keyword_takes_locator("Click") is None

    def test_falls_back_to_get_all_when_find_fails(self):
        # Simulates a library that was loaded but not bound to the session.
        kd = MagicMock()
        kd.find_keyword.return_value = None
        kd.get_all_keywords.return_value = [
            MagicMock(name_for_test="Click", library="Browser",
                      args=["selector"]),
        ]
        # mock setup quirk: `.name` is reserved, so we use name_for_test
        # then alias it:
        for k in kd.get_all_keywords.return_value:
            k.name = k.name_for_test
        intro = LocatorArgIntrospector(keyword_discovery=kd)
        assert intro.keyword_takes_locator("Click") is True

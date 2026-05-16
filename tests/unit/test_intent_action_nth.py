"""Tests for P6: nth= parameter on intent_action.

Verifies that nth=0 appends ">> nth=0" to Browser locators and
":nth-of-type(1)" to CSS SeleniumLibrary locators.
"""
from __future__ import annotations

__test__ = True

from typing import List, Optional

import pytest

from robotmcp.domains.intent.adapters.mcp_tool import (
    IntentActionAdapter,
    _apply_nth_to_locator,
)
from robotmcp.domains.intent.aggregates import IntentRegistry
from robotmcp.domains.intent.services import IntentResolver
from robotmcp.domains.intent.value_objects import IntentTarget, NormalizedLocator


class _SessionLookup:
    def __init__(self, library: str = "Browser"):
        self._lib = library

    def get_active_web_library(self, session_id: str) -> Optional[str]:
        return self._lib

    def get_imported_libraries(self, session_id: str) -> List[str]:
        return [self._lib]

    def get_platform_type(self, session_id: str) -> str:
        return "web"


class _PassthroughNormalizer:
    def normalize(self, target: IntentTarget, target_library: str) -> NormalizedLocator:
        return NormalizedLocator(
            value=target.locator,
            source_locator=target.locator,
            target_library=target_library,
            strategy_applied="pass_through",
            was_transformed=False,
        )


def _adapter(library: str = "Browser") -> IntentActionAdapter:
    registry = IntentRegistry.with_builtins()
    resolver = IntentResolver(
        registry=registry,
        session_lookup=_SessionLookup(library),
        normalizer=_PassthroughNormalizer(),
    )
    return IntentActionAdapter(resolver=resolver)


# ── _apply_nth_to_locator unit tests ──────────────────────────────────

def test_apply_nth_browser_appends_nth_suffix():
    result = _apply_nth_to_locator("id=nav_automobile", 0, "Browser")
    assert result == "id=nav_automobile >> nth=0"


def test_apply_nth_browser_nth_1():
    result = _apply_nth_to_locator("#btn", 1, "Browser")
    assert result == "#btn >> nth=1"


def test_apply_nth_selenium_css_appends_nth_of_type():
    result = _apply_nth_to_locator("css=.nav-link", 0, "SeleniumLibrary")
    assert result == "css=.nav-link:nth-of-type(1)"


def test_apply_nth_selenium_css_colon_prefix():
    result = _apply_nth_to_locator("css:.nav-link", 0, "SeleniumLibrary")
    assert result == "css:.nav-link:nth-of-type(1)"


def test_apply_nth_selenium_css_nth_1():
    result = _apply_nth_to_locator("css=button", 2, "SeleniumLibrary")
    assert result == "css=button:nth-of-type(3)"


def test_apply_nth_selenium_xpath_unchanged():
    # XPath locators: nth not supported for SL, returns locator unchanged
    result = _apply_nth_to_locator("xpath=//a[@id='nav']", 0, "SeleniumLibrary")
    assert result == "xpath=//a[@id='nav']"


def test_apply_nth_selenium_id_unchanged():
    result = _apply_nth_to_locator("id=nav_automobile", 0, "SeleniumLibrary")
    assert result == "id=nav_automobile"


def test_apply_nth_unknown_library_uses_browser_style():
    result = _apply_nth_to_locator("id=foo", 3, "UnknownLib")
    assert result == "id=foo >> nth=3"


# ── IntentActionAdapter nth= integration tests ────────────────────────

def test_click_browser_nth_0_appends_suffix():
    adapter = _adapter("Browser")
    result = adapter.resolve_intent(
        intent="click",
        target="id=nav_automobile",
        nth=0,
    )
    assert result["arguments"][0] == "id=nav_automobile >> nth=0"


def test_click_browser_nth_1_appends_correct_suffix():
    adapter = _adapter("Browser")
    result = adapter.resolve_intent(
        intent="click",
        target="text=Login",
        nth=1,
    )
    assert result["arguments"][0] == "text=Login >> nth=1"


def test_fill_browser_nth_appends_to_locator():
    adapter = _adapter("Browser")
    result = adapter.resolve_intent(
        intent="fill",
        target="css=input.name",
        value="John",
        nth=0,
    )
    assert result["arguments"][0] == "css=input.name >> nth=0"


def test_click_selenium_css_nth_0():
    adapter = _adapter("SeleniumLibrary")
    result = adapter.resolve_intent(
        intent="click",
        target="css=a.nav-item",
        nth=0,
    )
    assert result["arguments"][0] == "css=a.nav-item:nth-of-type(1)"


def test_nth_none_does_not_modify_locator():
    adapter = _adapter("Browser")
    result = adapter.resolve_intent(
        intent="click",
        target="id=submit",
        nth=None,
    )
    assert result["arguments"][0] == "id=submit"

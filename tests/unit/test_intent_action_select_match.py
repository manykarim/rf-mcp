"""Tests for P6: select intent match= parameter.

Verifies that match="value" dispatches Select Options By with "value" attribute
for Browser Library and uses Select From List By Value for SeleniumLibrary.
"""
from __future__ import annotations

__test__ = True

from typing import Dict, List, Optional

import pytest

from robotmcp.domains.intent.adapters.mcp_tool import IntentActionAdapter
from robotmcp.domains.intent.aggregates import (
    IntentRegistry,
    _resolve_select_match,
)
from robotmcp.domains.intent.services import IntentResolver
from robotmcp.domains.intent.value_objects import IntentTarget, NormalizedLocator


# ── helpers ──────────────────────────────────────────────────────────

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


def _make_adapter(library: str = "Browser") -> IntentActionAdapter:
    registry = IntentRegistry.with_builtins()
    resolver = IntentResolver(
        registry=registry,
        session_lookup=_SessionLookup(library),
        normalizer=_PassthroughNormalizer(),
    )
    return IntentActionAdapter(resolver=resolver)


# ── _resolve_select_match unit tests ──────────────────────────────────

def test_resolve_select_match_label():
    assert _resolve_select_match("label", "Preferred") == "label"


def test_resolve_select_match_value():
    assert _resolve_select_match("value", "5000000") == "value"


def test_resolve_select_match_index():
    assert _resolve_select_match("index", "2") == "index"


def test_resolve_select_match_text():
    assert _resolve_select_match("text", "foo") == "text"


def test_resolve_select_match_auto_numeric_returns_value():
    assert _resolve_select_match("auto", "5000000") == "value"


def test_resolve_select_match_auto_negative_numeric_returns_value():
    assert _resolve_select_match("auto", "-10") == "value"


def test_resolve_select_match_auto_non_numeric_returns_label():
    assert _resolve_select_match("auto", "Comprehensive") == "label"


def test_resolve_select_match_auto_none_value_returns_label():
    assert _resolve_select_match("auto", None) == "label"


def test_resolve_select_match_auto_empty_string_returns_label():
    assert _resolve_select_match("auto", "") == "label"


# ── Browser Library select match= tests ───────────────────────────────

def test_select_browser_match_value():
    adapter = _make_adapter("Browser")
    result = adapter.resolve_intent(
        intent="select",
        target="id=insurancesum",
        value="5000000",
        match="value",
    )
    assert result["keyword"] == "Select Options By"
    assert result["arguments"][1] == "value"
    assert result["arguments"][2] == "5000000"


def test_select_browser_match_label():
    adapter = _make_adapter("Browser")
    result = adapter.resolve_intent(
        intent="select",
        target="id=myselect",
        value="Comprehensive",
        match="label",
    )
    assert result["keyword"] == "Select Options By"
    assert result["arguments"][1] == "label"


def test_select_browser_match_index():
    adapter = _make_adapter("Browser")
    result = adapter.resolve_intent(
        intent="select",
        target="id=myselect",
        value="2",
        match="index",
    )
    assert result["keyword"] == "Select Options By"
    assert result["arguments"][1] == "index"


def test_select_browser_auto_numeric_uses_value():
    adapter = _make_adapter("Browser")
    result = adapter.resolve_intent(
        intent="select",
        target="id=insurancesum",
        value="5000000",
        match="auto",
    )
    assert result["arguments"][1] == "value"


def test_select_browser_auto_text_uses_label():
    adapter = _make_adapter("Browser")
    result = adapter.resolve_intent(
        intent="select",
        target="id=myselect",
        value="Basic Coverage",
        match="auto",
    )
    assert result["arguments"][1] == "label"


# ── SeleniumLibrary select match= tests ───────────────────────────────

def test_select_selenium_match_value_uses_correct_keyword():
    adapter = _make_adapter("SeleniumLibrary")
    result = adapter.resolve_intent(
        intent="select",
        target="id=insurancesum",
        value="5000000",
        match="value",
    )
    assert result["keyword"] == "Select From List By Value"
    assert "5000000" in result["arguments"]


def test_select_selenium_match_label_uses_label_keyword():
    adapter = _make_adapter("SeleniumLibrary")
    result = adapter.resolve_intent(
        intent="select",
        target="id=myselect",
        value="Comprehensive",
        match="label",
    )
    assert result["keyword"] == "Select From List By Label"


def test_select_selenium_match_index_uses_index_keyword():
    adapter = _make_adapter("SeleniumLibrary")
    result = adapter.resolve_intent(
        intent="select",
        target="id=myselect",
        value="2",
        match="index",
    )
    assert result["keyword"] == "Select From List By Index"


def test_select_selenium_auto_numeric_uses_value_keyword():
    adapter = _make_adapter("SeleniumLibrary")
    result = adapter.resolve_intent(
        intent="select",
        target="id=insurancesum",
        value="5000000",
        match="auto",
    )
    assert result["keyword"] == "Select From List By Value"


def test_select_selenium_auto_text_uses_label_keyword():
    adapter = _make_adapter("SeleniumLibrary")
    result = adapter.resolve_intent(
        intent="select",
        target="id=myselect",
        value="Basic Coverage",
        match="auto",
    )
    assert result["keyword"] == "Select From List By Label"

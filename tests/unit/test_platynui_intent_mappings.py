"""Unit tests for PlatynUI intent mappings (ADR-012).

Tests cover: IntentRegistry integration, mapping resolution,
argument transformers, and IntentVerb enum completeness.

Run with: uv run pytest tests/unit/test_platynui_intent_mappings.py -v
"""

__test__ = True

import pytest

from robotmcp.domains.intent import IntentRegistry, IntentVerb, IntentTarget, NormalizedLocator
from robotmcp.domains.intent.aggregates import (
    _extract_text_platynui_transformer,
    _assert_visible_platynui_transformer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry() -> IntentRegistry:
    return IntentRegistry.with_builtins()


def _target(locator: str = "/Window/Button[@Name='OK']") -> IntentTarget:
    return IntentTarget(locator=locator)


def _normalized(locator: str = "/Window/Button[@Name='OK']") -> NormalizedLocator:
    return NormalizedLocator(
        value=locator,
        source_locator=locator,
        target_library="PlatynUI.BareMetal",
        strategy_applied="platynui_xpath",
        was_transformed=False,
    )


# ===========================================================================
# Registry tests
# ===========================================================================

class TestRegistryPlatynUI:
    """Tests for PlatynUI.BareMetal in the IntentRegistry."""

    def test_platynui_baremetal_is_supported_library(self, registry):
        assert "PlatynUI.BareMetal" in registry.get_supported_libraries()

    @pytest.mark.parametrize("verb", [
        IntentVerb.CLICK,
        IntentVerb.FILL,
        IntentVerb.HOVER,
        IntentVerb.EXTRACT_TEXT,
        IntentVerb.ASSERT_VISIBLE,
        IntentVerb.FOCUS,
        IntentVerb.ACTIVATE,
        IntentVerb.MAXIMIZE,
        IntentVerb.MINIMIZE,
        IntentVerb.CLOSE_WINDOW,
        IntentVerb.RESTORE,
        IntentVerb.INSPECT,
    ])
    def test_has_mapping_for_verb(self, registry, verb):
        assert registry.has_mapping(verb, "PlatynUI.BareMetal"), (
            f"Expected mapping for ({verb.name}, PlatynUI.BareMetal)"
        )

    @pytest.mark.parametrize("verb", [
        IntentVerb.NAVIGATE,
        IntentVerb.WAIT_FOR,
        IntentVerb.SELECT,
    ])
    def test_does_not_have_mapping_for_web_only_verbs(self, registry, verb):
        assert not registry.has_mapping(verb, "PlatynUI.BareMetal"), (
            f"Unexpected mapping for ({verb.name}, PlatynUI.BareMetal)"
        )

    def test_platynui_supported_intents_count(self, registry):
        intents = registry.get_supported_intents("PlatynUI.BareMetal")
        assert len(intents) == 12

    def test_platynui_does_not_have_navigate(self, registry):
        assert not registry.has_mapping(IntentVerb.NAVIGATE, "PlatynUI.BareMetal")


# ===========================================================================
# Mapping resolution tests
# ===========================================================================

class TestMappingResolution:
    """Tests for resolving PlatynUI mappings to concrete keywords."""

    @pytest.mark.parametrize("verb,expected_keyword", [
        (IntentVerb.CLICK, "Pointer Click"),
        (IntentVerb.FILL, "Keyboard Type"),
        (IntentVerb.HOVER, "Pointer Move To"),
        (IntentVerb.FOCUS, "Focus"),
        (IntentVerb.ACTIVATE, "Activate"),
        (IntentVerb.INSPECT, "Query"),
        (IntentVerb.CLOSE_WINDOW, "Close"),
        (IntentVerb.MAXIMIZE, "Maximize"),
        (IntentVerb.MINIMIZE, "Minimize"),
        (IntentVerb.RESTORE, "Restore"),
    ])
    def test_verb_resolves_to_keyword(self, registry, verb, expected_keyword):
        mapping = registry.resolve(verb, "PlatynUI.BareMetal")
        assert mapping is not None, f"No mapping for {verb.name}"
        assert mapping.keyword == expected_keyword

    def test_extract_text_resolves_to_get_attribute(self, registry):
        mapping = registry.resolve(IntentVerb.EXTRACT_TEXT, "PlatynUI.BareMetal")
        assert mapping.keyword == "Get Attribute"

    def test_assert_visible_resolves_to_get_attribute(self, registry):
        mapping = registry.resolve(IntentVerb.ASSERT_VISIBLE, "PlatynUI.BareMetal")
        assert mapping.keyword == "Get Attribute"

    def test_click_requires_target_no_value(self, registry):
        mapping = registry.resolve(IntentVerb.CLICK, "PlatynUI.BareMetal")
        assert mapping.requires_target is True
        assert mapping.requires_value is False

    def test_fill_requires_target_and_value(self, registry):
        mapping = registry.resolve(IntentVerb.FILL, "PlatynUI.BareMetal")
        assert mapping.requires_target is True
        assert mapping.requires_value is True

    def test_inspect_timeout_category_is_read(self, registry):
        mapping = registry.resolve(IntentVerb.INSPECT, "PlatynUI.BareMetal")
        assert mapping.timeout_category == "read"

    def test_click_timeout_category_is_action(self, registry):
        mapping = registry.resolve(IntentVerb.CLICK, "PlatynUI.BareMetal")
        assert mapping.timeout_category == "action"


# ===========================================================================
# Argument transformer tests
# ===========================================================================

class TestArgumentTransformers:
    """Tests for PlatynUI argument transformer functions."""

    def test_extract_text_default_attribute_name(self):
        target = _target("/Window/Text[@Name='Status']")
        args = _extract_text_platynui_transformer(
            target, None, None, None
        )
        assert args == ["/Window/Text[@Name='Status']", "Name"]

    def test_extract_text_custom_attribute(self):
        target = _target("/Window/Edit[@Name='Input']")
        args = _extract_text_platynui_transformer(
            target, None, None, {"attribute_name": "Value"}
        )
        assert args == ["/Window/Edit[@Name='Input']", "Value"]

    def test_extract_text_with_normalized_locator(self):
        nlocator = _normalized("/Window/Text[@Name='Status']")
        args = _extract_text_platynui_transformer(
            None, None, nlocator, None
        )
        assert args == ["/Window/Text[@Name='Status']", "Name"]

    def test_assert_visible_produces_isoffscreen_check(self):
        target = _target("/Window/Button[@Name='Submit']")
        args = _assert_visible_platynui_transformer(
            target, None, None, None
        )
        assert args == ["/Window/Button[@Name='Submit']", "IsOffscreen", "==", "False"]

    def test_assert_visible_with_normalized_locator(self):
        nlocator = _normalized("/Window/Button[@Name='OK']")
        args = _assert_visible_platynui_transformer(
            None, None, nlocator, None
        )
        assert args == ["/Window/Button[@Name='OK']", "IsOffscreen", "==", "False"]

    def test_extract_text_via_mapping_build_arguments(self, registry):
        mapping = registry.resolve(IntentVerb.EXTRACT_TEXT, "PlatynUI.BareMetal")
        target = _target("/Window/Text")
        args = mapping.build_arguments(target, None, None, None)
        assert args == ["/Window/Text", "Name"]

    def test_assert_visible_via_mapping_build_arguments(self, registry):
        mapping = registry.resolve(IntentVerb.ASSERT_VISIBLE, "PlatynUI.BareMetal")
        target = _target("/Window/Button")
        args = mapping.build_arguments(target, None, None, None)
        assert args == ["/Window/Button", "IsOffscreen", "==", "False"]

    def test_click_uses_default_build_arguments(self, registry):
        mapping = registry.resolve(IntentVerb.CLICK, "PlatynUI.BareMetal")
        target = _target("/Window/Button[@Name='OK']")
        nlocator = _normalized("/Window/Button[@Name='OK']")
        args = mapping.build_arguments(target, None, nlocator, None)
        assert args == ["/Window/Button[@Name='OK']"]

    def test_fill_uses_default_build_arguments(self, registry):
        mapping = registry.resolve(IntentVerb.FILL, "PlatynUI.BareMetal")
        target = _target("/Window/Edit")
        nlocator = _normalized("/Window/Edit")
        args = mapping.build_arguments(target, "Hello", nlocator, None)
        assert args == ["/Window/Edit", "Hello"]


# ===========================================================================
# IntentVerb enum tests
# ===========================================================================

class TestIntentVerbEnum:
    """Tests for IntentVerb enum completeness."""

    def test_intent_verb_has_15_values(self):
        assert len(IntentVerb) == 15

    @pytest.mark.parametrize("verb_name", [
        "ACTIVATE", "MAXIMIZE", "MINIMIZE", "RESTORE",
        "FOCUS", "CLOSE_WINDOW", "INSPECT",
    ])
    def test_desktop_verbs_exist(self, verb_name):
        assert hasattr(IntentVerb, verb_name)
        assert isinstance(IntentVerb[verb_name], IntentVerb)

    @pytest.mark.parametrize("verb_name", [
        "NAVIGATE", "CLICK", "FILL", "HOVER", "SELECT",
        "ASSERT_VISIBLE", "EXTRACT_TEXT", "WAIT_FOR",
    ])
    def test_original_verbs_exist(self, verb_name):
        assert hasattr(IntentVerb, verb_name)

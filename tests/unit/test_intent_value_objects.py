"""Comprehensive unit tests for Intent domain value objects (ADR-007).

Tests cover: IntentVerb, LocatorStrategy, IntentTarget, NormalizedLocator, ResolvedIntent.

Run with: uv run pytest tests/unit/test_intent_value_objects.py -v
"""

__test__ = True

import pytest

from robotmcp.domains.intent.value_objects import (
    IntentTarget,
    IntentVerb,
    LocatorStrategy,
    NormalizedLocator,
    ResolvedIntent,
)


# =============================================================================
# IntentVerb
# =============================================================================


class TestIntentVerb:
    """Test IntentVerb enum."""

    def test_has_exactly_15_values(self):
        assert len(IntentVerb) == 15

    def test_navigate_value(self):
        assert IntentVerb.NAVIGATE.value == "navigate"

    def test_click_value(self):
        assert IntentVerb.CLICK.value == "click"

    def test_fill_value(self):
        assert IntentVerb.FILL.value == "fill"

    def test_hover_value(self):
        assert IntentVerb.HOVER.value == "hover"

    def test_select_value(self):
        assert IntentVerb.SELECT.value == "select"

    def test_assert_visible_value(self):
        assert IntentVerb.ASSERT_VISIBLE.value == "assert_visible"

    def test_extract_text_value(self):
        assert IntentVerb.EXTRACT_TEXT.value == "extract_text"

    def test_wait_for_value(self):
        assert IntentVerb.WAIT_FOR.value == "wait_for"

    def test_string_subclass(self):
        """IntentVerb is a str enum so it can be used directly as a string."""
        assert isinstance(IntentVerb.CLICK, str)
        assert IntentVerb.CLICK == "click"


# =============================================================================
# LocatorStrategy
# =============================================================================


class TestLocatorStrategy:
    """Test LocatorStrategy enum."""

    def test_has_exactly_10_values(self):
        assert len(LocatorStrategy) == 10

    def test_includes_auto(self):
        assert LocatorStrategy.AUTO.value == "auto"

    def test_includes_css(self):
        assert LocatorStrategy.CSS.value == "css"

    def test_includes_xpath(self):
        assert LocatorStrategy.XPATH.value == "xpath"

    def test_includes_text(self):
        assert LocatorStrategy.TEXT.value == "text"

    def test_includes_id(self):
        assert LocatorStrategy.ID.value == "id"

    def test_includes_name(self):
        assert LocatorStrategy.NAME.value == "name"

    def test_includes_link(self):
        assert LocatorStrategy.LINK.value == "link"

    def test_includes_partial_link(self):
        assert LocatorStrategy.PARTIAL_LINK.value == "partial_link"

    def test_includes_accessibility_id(self):
        assert LocatorStrategy.ACCESSIBILITY_ID.value == "accessibility_id"


# =============================================================================
# IntentTarget
# =============================================================================


class TestIntentTarget:
    """Test IntentTarget value object."""

    def test_basic_construction(self):
        target = IntentTarget(locator="text=Login")
        assert target.locator == "text=Login"
        assert target.strategy == LocatorStrategy.AUTO
        assert target.original_locator is None

    def test_construction_with_strategy(self):
        target = IntentTarget(locator="#submit", strategy=LocatorStrategy.CSS)
        assert target.strategy == LocatorStrategy.CSS

    def test_null_byte_raises(self):
        with pytest.raises(ValueError, match="null bytes"):
            IntentTarget(locator="bad\x00locator")

    def test_exceeds_max_length_raises(self):
        with pytest.raises(ValueError, match="exceeds max length"):
            IntentTarget(locator="x" * 10001)

    def test_exactly_max_length_is_valid(self):
        target = IntentTarget(locator="x" * 10000)
        assert len(target.locator) == 10000

    def test_has_explicit_strategy_true(self):
        target = IntentTarget(locator="btn", strategy=LocatorStrategy.CSS)
        assert target.has_explicit_strategy is True

    def test_has_explicit_strategy_false_for_auto(self):
        target = IntentTarget(locator="btn")
        assert target.has_explicit_strategy is False

    def test_has_prefix_css_equals(self):
        assert IntentTarget(locator="css=.btn").has_prefix is True

    def test_has_prefix_xpath_equals(self):
        assert IntentTarget(locator="xpath=//div").has_prefix is True

    def test_has_prefix_text_equals(self):
        assert IntentTarget(locator="text=Login").has_prefix is True

    def test_has_prefix_id_equals(self):
        assert IntentTarget(locator="id=submit").has_prefix is True

    def test_has_prefix_id_colon(self):
        assert IntentTarget(locator="id:submit").has_prefix is True

    def test_has_prefix_link_colon(self):
        assert IntentTarget(locator="link:Home").has_prefix is True

    def test_has_prefix_partial_link_colon(self):
        assert IntentTarget(locator="partial link:Ho").has_prefix is True

    def test_has_prefix_accessibility_id(self):
        assert IntentTarget(locator="accessibility_id=Login").has_prefix is True

    def test_has_prefix_bare_text_false(self):
        assert IntentTarget(locator="Login").has_prefix is False

    def test_has_prefix_hash_selector_false(self):
        """#submit does not start with a known prefix."""
        assert IntentTarget(locator="#submit").has_prefix is False

    def test_has_prefix_xpath_without_prefix_false(self):
        assert IntentTarget(locator="//button").has_prefix is False

    def test_frozen(self):
        target = IntentTarget(locator="x")
        with pytest.raises(AttributeError):
            target.locator = "y"


# =============================================================================
# NormalizedLocator
# =============================================================================


class TestNormalizedLocator:
    """Test NormalizedLocator value object."""

    def test_construction(self):
        loc = NormalizedLocator(
            value="text=Login",
            source_locator="Login",
            target_library="Browser",
            strategy_applied="bare_text",
            was_transformed=True,
        )
        assert loc.value == "text=Login"
        assert loc.was_transformed is True

    def test_frozen(self):
        loc = NormalizedLocator(
            value="x", source_locator="x",
            target_library="Browser", strategy_applied="pass_through",
            was_transformed=False,
        )
        with pytest.raises(AttributeError):
            loc.value = "y"

    def test_equality_by_value(self):
        a = NormalizedLocator("text=Login", "Login", "Browser", "bare_text", True)
        b = NormalizedLocator("text=Login", "Login", "Browser", "bare_text", True)
        assert a == b


# =============================================================================
# ResolvedIntent
# =============================================================================


class TestResolvedIntent:
    """Test ResolvedIntent value object."""

    def test_construction(self):
        resolved = ResolvedIntent(
            keyword="Click",
            arguments=["text=Login"],
            library="Browser",
            intent_verb=IntentVerb.CLICK,
        )
        assert resolved.keyword == "Click"
        assert resolved.arguments == ["text=Login"]
        assert resolved.library == "Browser"
        assert resolved.intent_verb == IntentVerb.CLICK
        assert resolved.normalized_locator is None
        assert resolved.metadata == {}

    def test_with_metadata(self):
        resolved = ResolvedIntent(
            keyword="Click",
            arguments=["text=Login"],
            library="Browser",
            intent_verb=IntentVerb.CLICK,
            metadata={"timeout_category": "action", "session_id": "s1"},
        )
        assert resolved.metadata["timeout_category"] == "action"

    def test_frozen(self):
        resolved = ResolvedIntent(
            keyword="Click", arguments=[], library="Browser",
            intent_verb=IntentVerb.CLICK,
        )
        with pytest.raises(AttributeError):
            resolved.keyword = "Other"

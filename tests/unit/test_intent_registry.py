"""Comprehensive unit tests for IntentRegistry aggregate (ADR-007).

Tests cover: with_builtins factory, register, resolve, has_mapping,
get_supported_intents, get_supported_libraries.

Run with: uv run pytest tests/unit/test_intent_registry.py -v
"""

__test__ = True

import pytest

from robotmcp.domains.intent.aggregates import IntentRegistry
from robotmcp.domains.intent.entities import IntentMapping
from robotmcp.domains.intent.value_objects import IntentVerb


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def registry():
    """Pre-populated registry with builtin mappings."""
    return IntentRegistry.with_builtins()


# =============================================================================
# with_builtins factory
# =============================================================================


class TestWithBuiltins:
    """Test the with_builtins factory method."""

    def test_creates_registry(self, registry):
        assert isinstance(registry, IntentRegistry)

    def test_browser_has_8_mappings(self, registry):
        intents = registry.get_supported_intents("Browser")
        assert len(intents) == 8

    def test_selenium_has_8_mappings(self, registry):
        intents = registry.get_supported_intents("SeleniumLibrary")
        assert len(intents) == 8

    def test_appium_has_6_mappings(self, registry):
        """AppiumLibrary has no HOVER and no SELECT mappings."""
        intents = registry.get_supported_intents("AppiumLibrary")
        assert len(intents) == 6

    def test_appium_missing_hover(self, registry):
        assert not registry.has_mapping(IntentVerb.HOVER, "AppiumLibrary")

    def test_appium_missing_select(self, registry):
        assert not registry.has_mapping(IntentVerb.SELECT, "AppiumLibrary")


# =============================================================================
# resolve
# =============================================================================


class TestResolve:
    """Test mapping lookup by (verb, library) pair."""

    def test_click_browser(self, registry):
        mapping = registry.resolve(IntentVerb.CLICK, "Browser")
        assert mapping is not None
        assert mapping.keyword == "Click"

    def test_click_selenium(self, registry):
        mapping = registry.resolve(IntentVerb.CLICK, "SeleniumLibrary")
        assert mapping is not None
        assert mapping.keyword == "Click Element"

    def test_click_appium(self, registry):
        mapping = registry.resolve(IntentVerb.CLICK, "AppiumLibrary")
        assert mapping is not None
        assert mapping.keyword == "Click Element"

    def test_fill_browser(self, registry):
        mapping = registry.resolve(IntentVerb.FILL, "Browser")
        assert mapping.keyword == "Fill Text"
        assert mapping.requires_value is True

    def test_fill_selenium(self, registry):
        mapping = registry.resolve(IntentVerb.FILL, "SeleniumLibrary")
        assert mapping.keyword == "Input Text"

    def test_navigate_browser(self, registry):
        mapping = registry.resolve(IntentVerb.NAVIGATE, "Browser")
        assert mapping.keyword == "Go To"

    def test_navigate_appium(self, registry):
        mapping = registry.resolve(IntentVerb.NAVIGATE, "AppiumLibrary")
        assert mapping.keyword == "Go To Url"

    def test_hover_browser(self, registry):
        mapping = registry.resolve(IntentVerb.HOVER, "Browser")
        assert mapping.keyword == "Hover"

    def test_hover_selenium(self, registry):
        mapping = registry.resolve(IntentVerb.HOVER, "SeleniumLibrary")
        assert mapping.keyword == "Mouse Over"

    def test_hover_appium_returns_none(self, registry):
        assert registry.resolve(IntentVerb.HOVER, "AppiumLibrary") is None

    def test_select_browser(self, registry):
        mapping = registry.resolve(IntentVerb.SELECT, "Browser")
        assert mapping.keyword == "Select Options By"

    def test_select_selenium(self, registry):
        mapping = registry.resolve(IntentVerb.SELECT, "SeleniumLibrary")
        assert mapping.keyword == "Select From List By Label"

    def test_assert_visible_browser(self, registry):
        mapping = registry.resolve(IntentVerb.ASSERT_VISIBLE, "Browser")
        assert mapping.keyword == "Get Element States"

    def test_assert_visible_selenium(self, registry):
        mapping = registry.resolve(IntentVerb.ASSERT_VISIBLE, "SeleniumLibrary")
        assert mapping.keyword == "Element Should Be Visible"

    def test_extract_text_browser(self, registry):
        mapping = registry.resolve(IntentVerb.EXTRACT_TEXT, "Browser")
        assert mapping.keyword == "Get Text"

    def test_wait_for_browser(self, registry):
        mapping = registry.resolve(IntentVerb.WAIT_FOR, "Browser")
        assert mapping.keyword == "Wait For Elements State"

    def test_wait_for_selenium(self, registry):
        mapping = registry.resolve(IntentVerb.WAIT_FOR, "SeleniumLibrary")
        assert mapping.keyword == "Wait Until Element Is Visible"

    def test_unmapped_verb_library_returns_none(self, registry):
        assert registry.resolve(IntentVerb.CLICK, "UnknownLibrary") is None


# =============================================================================
# has_mapping
# =============================================================================


class TestHasMapping:
    """Test has_mapping convenience method."""

    def test_true_for_existing(self, registry):
        assert registry.has_mapping(IntentVerb.CLICK, "Browser") is True

    def test_false_for_missing(self, registry):
        assert registry.has_mapping(IntentVerb.HOVER, "AppiumLibrary") is False

    def test_false_for_unknown_library(self, registry):
        assert registry.has_mapping(IntentVerb.CLICK, "NoSuchLib") is False


# =============================================================================
# get_supported_intents
# =============================================================================


class TestGetSupportedIntents:
    """Test intent enumeration per library."""

    def test_browser_all_8_web_verbs(self, registry):
        intents = registry.get_supported_intents("Browser")
        web_verbs = {
            IntentVerb.NAVIGATE, IntentVerb.CLICK, IntentVerb.FILL,
            IntentVerb.HOVER, IntentVerb.SELECT, IntentVerb.ASSERT_VISIBLE,
            IntentVerb.EXTRACT_TEXT, IntentVerb.WAIT_FOR,
        }
        assert set(intents) == web_verbs

    def test_selenium_all_8_web_verbs(self, registry):
        intents = registry.get_supported_intents("SeleniumLibrary")
        web_verbs = {
            IntentVerb.NAVIGATE, IntentVerb.CLICK, IntentVerb.FILL,
            IntentVerb.HOVER, IntentVerb.SELECT, IntentVerb.ASSERT_VISIBLE,
            IntentVerb.EXTRACT_TEXT, IntentVerb.WAIT_FOR,
        }
        assert set(intents) == web_verbs

    def test_appium_6(self, registry):
        intents = registry.get_supported_intents("AppiumLibrary")
        expected = {
            IntentVerb.NAVIGATE, IntentVerb.CLICK, IntentVerb.FILL,
            IntentVerb.ASSERT_VISIBLE, IntentVerb.EXTRACT_TEXT, IntentVerb.WAIT_FOR,
        }
        assert set(intents) == expected

    def test_unknown_library_empty(self, registry):
        assert registry.get_supported_intents("NoSuchLib") == []


# =============================================================================
# get_supported_libraries
# =============================================================================


class TestGetSupportedLibraries:
    """Test library enumeration."""

    def test_returns_four_libraries(self, registry):
        libs = registry.get_supported_libraries()
        assert libs == {"Browser", "SeleniumLibrary", "AppiumLibrary", "PlatynUI.BareMetal"}


# =============================================================================
# register with validation
# =============================================================================


class TestRegisterValidation:
    """Test register method input validation."""

    def test_empty_keyword_raises(self):
        registry = IntentRegistry()
        with pytest.raises(ValueError, match="keyword must not be empty"):
            registry.register(IntentMapping(
                intent_verb=IntentVerb.CLICK,
                library="Browser",
                keyword="",
            ))

    def test_empty_library_raises(self):
        registry = IntentRegistry()
        with pytest.raises(ValueError, match="library must not be empty"):
            registry.register(IntentMapping(
                intent_verb=IntentVerb.CLICK,
                library="",
                keyword="Click",
            ))

    def test_register_overrides_existing(self, registry):
        """Registering the same (verb, library) key overrides the previous mapping."""
        registry.register(IntentMapping(
            intent_verb=IntentVerb.CLICK,
            library="Browser",
            keyword="New Click Keyword",
        ))
        mapping = registry.resolve(IntentVerb.CLICK, "Browser")
        assert mapping.keyword == "New Click Keyword"

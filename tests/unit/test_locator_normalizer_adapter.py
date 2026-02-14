"""Comprehensive unit tests for LocatorNormalizerAdapter (ADR-007).

Tests cover: bare text normalization, CSS shorthand, XPath auto-detect,
cross-library prefix translation, URL pass-through, idempotency.

Run with: uv run pytest tests/unit/test_locator_normalizer_adapter.py -v
"""

__test__ = True

import pytest

from robotmcp.domains.intent.adapters.locator_normalizer_adapter import (
    LocatorNormalizerAdapter,
)
from robotmcp.domains.intent.value_objects import IntentTarget, LocatorStrategy


@pytest.fixture
def normalizer():
    return LocatorNormalizerAdapter()


# =============================================================================
# Bare text normalization
# =============================================================================


class TestBareTextNormalization:
    """Test bare text -> library-appropriate locator."""

    def test_bare_text_to_browser(self, normalizer):
        result = normalizer.normalize(IntentTarget(locator="Login"), "Browser")
        assert result.value == "text=Login"
        assert result.was_transformed is True
        assert result.strategy_applied == "bare_text"

    def test_bare_text_to_selenium_short(self, normalizer):
        result = normalizer.normalize(IntentTarget(locator="Login"), "SeleniumLibrary")
        assert result.value == "link:Login"
        assert result.was_transformed is True

    def test_bare_text_to_selenium_long(self, normalizer):
        """Long text (50+ chars) uses xpath contains()."""
        long_text = "A" * 55
        result = normalizer.normalize(IntentTarget(locator=long_text), "SeleniumLibrary")
        assert result.value.startswith("xpath=")
        assert "contains(text()" in result.value
        assert result.was_transformed is True

    def test_bare_text_to_appium(self, normalizer):
        result = normalizer.normalize(IntentTarget(locator="Login"), "AppiumLibrary")
        assert result.value == "accessibility_id=Login"
        assert result.was_transformed is True


# =============================================================================
# CSS shorthand normalization
# =============================================================================


class TestCSSShorthandNormalization:
    """Test CSS shorthand (#id, .class) normalization."""

    def test_hash_id_to_browser(self, normalizer):
        """Browser Library handles #id natively -- no transformation."""
        result = normalizer.normalize(IntentTarget(locator="#submit"), "Browser")
        assert result.value == "#submit"
        assert result.was_transformed is False

    def test_hash_id_to_selenium(self, normalizer):
        """#submit -> id=submit for SeleniumLibrary."""
        result = normalizer.normalize(IntentTarget(locator="#submit"), "SeleniumLibrary")
        assert result.value == "id=submit"
        assert result.was_transformed is True
        assert result.strategy_applied == "css_auto_detect"

    def test_dot_class_to_selenium(self, normalizer):
        """.nav-link -> css=.nav-link for SeleniumLibrary."""
        result = normalizer.normalize(IntentTarget(locator=".nav-link"), "SeleniumLibrary")
        assert result.value == "css=.nav-link"
        assert result.was_transformed is True

    def test_dot_class_to_browser(self, normalizer):
        """Browser handles .class natively."""
        result = normalizer.normalize(IntentTarget(locator=".nav-link"), "Browser")
        assert result.value == ".nav-link"
        assert result.was_transformed is False

    def test_data_testid_attribute_to_selenium(self, normalizer):
        """[data-testid="x"] -> css=[data-testid="x"] for SeleniumLibrary."""
        result = normalizer.normalize(
            IntentTarget(locator='[data-testid="x"]'), "SeleniumLibrary"
        )
        assert result.value == 'css=[data-testid="x"]'
        assert result.was_transformed is True


# =============================================================================
# XPath auto-detection
# =============================================================================


class TestXPathAutoDetect:
    """Test XPath expression normalization."""

    def test_xpath_to_browser(self, normalizer):
        """Browser handles // natively."""
        result = normalizer.normalize(IntentTarget(locator="//button"), "Browser")
        assert result.value == "//button"
        assert result.was_transformed is False

    def test_xpath_to_selenium(self, normalizer):
        """// -> xpath=// for SeleniumLibrary."""
        result = normalizer.normalize(IntentTarget(locator="//button"), "SeleniumLibrary")
        assert result.value == "xpath=//button"
        assert result.was_transformed is True

    def test_parent_xpath_to_selenium(self, normalizer):
        """.. -> xpath=.. for SeleniumLibrary."""
        result = normalizer.normalize(IntentTarget(locator="../div"), "SeleniumLibrary")
        assert result.value == "xpath=../div"
        assert result.was_transformed is True


# =============================================================================
# Cross-library prefix translation
# =============================================================================


class TestPrefixTranslation:
    """Test cross-library locator prefix translation."""

    def test_text_equals_to_selenium(self, normalizer):
        """text=Login -> link:Login for SeleniumLibrary."""
        result = normalizer.normalize(IntentTarget(locator="text=Login"), "SeleniumLibrary")
        assert result.value == "link:Login"
        assert result.was_transformed is True
        assert result.strategy_applied == "prefix_translation"

    def test_link_colon_to_browser(self, normalizer):
        """link:Login -> text=Login for Browser."""
        result = normalizer.normalize(IntentTarget(locator="link:Login"), "Browser")
        assert result.value == "text=Login"
        assert result.was_transformed is True

    def test_link_equals_to_browser(self, normalizer):
        """link=Login is not in IntentTarget.known_prefixes, so LocatorNormalizerAdapter
        treats it as bare text and prepends 'text='. The _translate_prefix path
        is only reached when target.has_prefix is True."""
        result = normalizer.normalize(IntentTarget(locator="link=Login"), "Browser")
        # link= is NOT a known prefix in IntentTarget, so it falls into bare text
        assert result.value == "text=link=Login"
        assert result.was_transformed is True

    def test_partial_link_colon_to_browser(self, normalizer):
        """partial link:Ho -> text=Ho for Browser."""
        result = normalizer.normalize(IntentTarget(locator="partial link:Ho"), "Browser")
        assert result.value == "text=Ho"
        assert result.was_transformed is True

    def test_id_colon_to_browser(self, normalizer):
        """id:submit -> id=submit for Browser."""
        result = normalizer.normalize(IntentTarget(locator="id:submit"), "Browser")
        assert result.value == "id=submit"
        assert result.was_transformed is True

    def test_css_colon_to_browser(self, normalizer):
        """css:.btn -> css=.btn for Browser."""
        result = normalizer.normalize(IntentTarget(locator="css:.btn"), "Browser")
        assert result.value == "css=.btn"
        assert result.was_transformed is True

    def test_text_equals_to_appium(self, normalizer):
        """text=Login -> accessibility_id=Login for AppiumLibrary."""
        result = normalizer.normalize(IntentTarget(locator="text=Login"), "AppiumLibrary")
        assert result.value == "accessibility_id=Login"
        assert result.was_transformed is True


# =============================================================================
# Already-prefixed / pass-through
# =============================================================================


class TestPassThrough:
    """Test that already-correct locators pass through unchanged."""

    def test_css_equals_any_library(self, normalizer):
        """css=.btn passes through for all libraries."""
        result = normalizer.normalize(IntentTarget(locator="css=.btn"), "Browser")
        assert result.value == "css=.btn"
        assert result.was_transformed is False

    def test_id_equals_any_library(self, normalizer):
        result = normalizer.normalize(IntentTarget(locator="id=submit"), "SeleniumLibrary")
        assert result.value == "id=submit"
        assert result.was_transformed is False

    def test_url_pass_through(self, normalizer):
        """URLs pass through unchanged."""
        result = normalizer.normalize(
            IntentTarget(locator="https://example.com"), "Browser"
        )
        assert result.value == "https://example.com"
        assert result.was_transformed is False
        assert result.strategy_applied == "url_pass_through"

    def test_relative_url_pass_through(self, normalizer):
        """/ path passes through as URL."""
        result = normalizer.normalize(IntentTarget(locator="/login"), "Browser")
        assert result.value == "/login"
        assert result.was_transformed is False


# =============================================================================
# Idempotency
# =============================================================================


class TestIdempotency:
    """Test that already-prefixed locators are not double-prefixed."""

    def test_text_equals_to_browser_not_doubled(self, normalizer):
        """text=Login should stay text=Login for Browser (not text=text=Login)."""
        result = normalizer.normalize(IntentTarget(locator="text=Login"), "Browser")
        assert result.value == "text=Login"
        assert result.was_transformed is False

    def test_xpath_equals_not_doubled(self, normalizer):
        result = normalizer.normalize(
            IntentTarget(locator="xpath=//button"), "SeleniumLibrary"
        )
        assert result.value == "xpath=//button"
        assert result.was_transformed is False

    def test_css_equals_not_doubled(self, normalizer):
        result = normalizer.normalize(
            IntentTarget(locator="css=.btn"), "SeleniumLibrary"
        )
        assert result.value == "css=.btn"
        assert result.was_transformed is False


# =============================================================================
# Explicit strategy
# =============================================================================


class TestExplicitStrategy:
    """Test explicit strategy hint application."""

    def test_explicit_css_strategy(self, normalizer):
        result = normalizer.normalize(
            IntentTarget(locator="btn-class", strategy=LocatorStrategy.CSS),
            "Browser",
        )
        assert result.value == "css=btn-class"
        assert result.was_transformed is True

    def test_explicit_id_strategy(self, normalizer):
        result = normalizer.normalize(
            IntentTarget(locator="submit", strategy=LocatorStrategy.ID),
            "Browser",
        )
        assert result.value == "id=submit"
        assert result.was_transformed is True

    def test_explicit_text_strategy_browser(self, normalizer):
        result = normalizer.normalize(
            IntentTarget(locator="Login", strategy=LocatorStrategy.TEXT),
            "Browser",
        )
        assert result.value == "text=Login"

    def test_explicit_text_strategy_selenium(self, normalizer):
        result = normalizer.normalize(
            IntentTarget(locator="Login", strategy=LocatorStrategy.TEXT),
            "SeleniumLibrary",
        )
        assert result.value == "link:Login"

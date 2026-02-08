"""Appium locator pre-validation tests.

Tests Appium-specific locator prefixes (accessibility_id=, android=, ios=,
uiautomator=) in the context of pre-validation. Documents the behavior
where Appium prefixes are NOT in _GENERIC_LOCATOR_PREFIXES.

POTENTIAL BUG DOCUMENTED: For keyword-specific keywords (Click Button, Click
Link, Click Image), Appium locators are treated as bare text and skip
pre-validation entirely. This is defensively safe (no false rejects) but
means Appium element actionability is never checked for these keywords.

Run with: uv run pytest tests/unit/test_appium_prevalidation.py -v
"""

__test__ = True

import pytest
from robotmcp.components.execution.keyword_executor import KeywordExecutor
from robotmcp.models.config_models import ExecutionConfig


@pytest.fixture
def executor():
    return KeywordExecutor(config=ExecutionConfig())


# Appium locator prefixes not in _GENERIC_LOCATOR_PREFIXES
APPIUM_LOCATOR_PREFIXES = [
    "accessibility_id=",
    "android=",
    "ios=",
    "uiautomator=",
    # Common Appium UiSelector and UiAutomator variants
]

# Standard Appium locator examples
APPIUM_LOCATORS = [
    "accessibility_id=login_button",
    "android=new UiSelector().text(\"Login\")",
    "ios=type == 'XCUIElementTypeButton' AND name == 'Login'",
    "uiautomator=text(\"Submit\")",
]


class TestAppiumLocatorsNotInGenericPrefixes:
    """Document that Appium locator prefixes are NOT in _GENERIC_LOCATOR_PREFIXES."""

    @pytest.mark.parametrize("prefix", APPIUM_LOCATOR_PREFIXES)
    def test_appium_prefix_not_in_generic_prefixes(self, executor, prefix):
        """Appium prefix is NOT in _GENERIC_LOCATOR_PREFIXES."""
        assert not prefix.lower().startswith(executor._GENERIC_LOCATOR_PREFIXES)

    @pytest.mark.parametrize("prefix", APPIUM_LOCATOR_PREFIXES)
    def test_appium_prefix_not_in_skip_prefixes(self, executor, prefix):
        """Appium prefix is NOT in _SKIP_PRE_VALIDATION_LOCATOR_PREFIXES."""
        assert not prefix.lower().startswith(executor._SKIP_PRE_VALIDATION_LOCATOR_PREFIXES)


class TestAppiumLocatorsWithRegularKeywords:
    """Test Appium locators with regular element interaction keywords."""

    @pytest.mark.parametrize("locator", APPIUM_LOCATORS)
    def test_appium_locator_extracted_for_click_element(self, executor, locator):
        """Regular keywords (click element) SHOULD extract Appium locators."""
        result = executor._extract_locator_from_args("click element", [locator])
        # Appium locators don't match any skip prefix, so they should be returned
        assert result == locator

    @pytest.mark.parametrize("locator", APPIUM_LOCATORS)
    def test_appium_locator_extracted_for_input_text(self, executor, locator):
        """Regular keywords (input text) SHOULD extract Appium locators."""
        result = executor._extract_locator_from_args("input text", [locator])
        assert result == locator

    @pytest.mark.parametrize("locator", APPIUM_LOCATORS)
    def test_appium_locator_extracted_for_tap(self, executor, locator):
        """Tap keyword SHOULD extract Appium locators."""
        result = executor._extract_locator_from_args("tap", [locator])
        assert result == locator


class TestAppiumLocatorsWithKeywordSpecificKeywords:
    """Test Appium locators with keyword-specific keywords (Click Button/Link/Image).

    DOCUMENTS POTENTIAL BUG: Appium locators with keyword-specific keywords
    are treated as bare text because they don't match _GENERIC_LOCATOR_PREFIXES.
    This means pre-validation is SKIPPED (defensively safe, but no actionability check).
    """

    @pytest.mark.parametrize("keyword", ["click button", "click link", "click image"])
    @pytest.mark.parametrize("locator", APPIUM_LOCATORS)
    def test_appium_locator_skipped_for_keyword_specific_keywords(self, executor, keyword, locator):
        """DOCUMENTS BUG: Appium locators treated as bare text for keyword-specific keywords.

        Appium locators like 'accessibility_id=btn' don't start with any
        _GENERIC_LOCATOR_PREFIXES, so they're treated as bare text and
        pre-validation is skipped (returns None).
        """
        result = executor._extract_locator_from_args(keyword, [locator])
        # Returns None because Appium prefix not in _GENERIC_LOCATOR_PREFIXES
        # and keyword is in _KEYWORD_SPECIFIC_LOCATOR_KEYWORDS
        assert result is None


class TestAppiumLocatorsWithGenericPrefixes:
    """Test that standard prefixes DO work with keyword-specific keywords."""

    @pytest.mark.parametrize("keyword", ["click button", "click link", "click image"])
    @pytest.mark.parametrize("locator", [
        "id=login_button",
        "css=button#submit",
        "xpath=//button[@text='Login']",
        "//button[@id='submit']",
    ])
    def test_generic_prefix_extracted_for_keyword_specific(self, executor, keyword, locator):
        """Generic prefixes (id=, css=, xpath=) ARE extracted for keyword-specific keywords."""
        result = executor._extract_locator_from_args(keyword, [locator])
        assert result == locator


class TestAppiumRequiresPreValidation:
    """Test that Appium-specific keywords are in ELEMENT_INTERACTION_KEYWORDS."""

    @pytest.mark.parametrize("keyword", [
        "tap",
        "long press",
        "input value",
    ])
    def test_appium_keywords_require_pre_validation(self, executor, keyword):
        """Appium interaction keywords should require pre-validation."""
        assert executor._requires_pre_validation(keyword) is True


class TestAppiumEmptyArguments:
    """Test edge cases with empty or missing arguments."""

    def test_empty_arguments_returns_none(self, executor):
        result = executor._extract_locator_from_args("tap", [])
        assert result is None

    def test_none_first_arg_returns_none(self, executor):
        result = executor._extract_locator_from_args("tap", [None])
        assert result is None

    def test_non_string_first_arg_returns_none(self, executor):
        result = executor._extract_locator_from_args("tap", [42])
        assert result is None

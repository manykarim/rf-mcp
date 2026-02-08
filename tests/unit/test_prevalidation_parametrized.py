"""Parametrized pre-validation prefix and keyword coverage tests.

Systematically validates all locator prefixes in _GENERIC_LOCATOR_PREFIXES and
all keywords in ELEMENT_INTERACTION_KEYWORDS using parametrized test generation.

Run with: uv run pytest tests/unit/test_prevalidation_parametrized.py -v
"""

__test__ = True

import pytest
from robotmcp.components.execution.keyword_executor import KeywordExecutor
from robotmcp.models.config_models import ExecutionConfig


@pytest.fixture
def executor():
    return KeywordExecutor(config=ExecutionConfig())


# All 25 generic locator prefixes from _GENERIC_LOCATOR_PREFIXES
GENERIC_PREFIXES = [
    "id=", "id:", "css=", "css:", "xpath=", "xpath:",
    "name=", "name:", "class=", "class:", "tag=", "tag:",
    "dom=", "dom:", "jquery=", "jquery:", "sizzle=", "sizzle:",
    "data=", "data:", "identifier=", "identifier:",
    "//",
]

# Skip prefixes from _SKIP_PRE_VALIDATION_LOCATOR_PREFIXES
SKIP_PREFIXES = [
    "link=", "partial link=", "link:", "partial link:",
]

# Keyword-specific keywords from _KEYWORD_SPECIFIC_LOCATOR_KEYWORDS
KEYWORD_SPECIFIC_KEYWORDS = [
    "click link", "click image", "click button",
]

# Sample of ELEMENT_INTERACTION_KEYWORDS for bare-text extraction
ELEMENT_INTERACTION_SAMPLE = [
    "click", "click element", "double click", "right click",
    "fill text", "type text", "input text",
    "check checkbox", "uncheck checkbox",
    "select options", "select from list by value",
    "press keys", "hover", "focus",
    "scroll to element", "drag and drop",
    "tap", "submit form", "choose file",
]


# =============================================================================
# GAP-1: All generic prefixes x keyword-specific keywords
# =============================================================================


class TestGenericPrefixesWithKeywordSpecificKeywords:
    """All 25 _GENERIC_LOCATOR_PREFIXES should be extracted by keyword-specific keywords."""

    @pytest.mark.parametrize("prefix", GENERIC_PREFIXES)
    @pytest.mark.parametrize("keyword", KEYWORD_SPECIFIC_KEYWORDS)
    def test_generic_prefix_extracted(self, executor, prefix, keyword):
        """Generic prefix '{prefix}' should be extracted for keyword '{keyword}'."""
        if prefix == "//":
            locator = "//div[@id='test']"
        else:
            locator = f"{prefix}test_element"

        result = executor._extract_locator_from_args(keyword, [locator])
        assert result == locator, (
            f"Expected '{locator}' to be extracted for '{keyword}', "
            f"but got {result}"
        )


# =============================================================================
# GAP-1b: Skip prefixes always return None
# =============================================================================


class TestSkipPrefixesAlwaysSkipped:
    """Link-text prefixes should always skip pre-validation (return None)."""

    @pytest.mark.parametrize("prefix", SKIP_PREFIXES)
    @pytest.mark.parametrize("keyword", ["click element", "click link", "click button"])
    def test_skip_prefix_returns_none(self, executor, prefix, keyword):
        """Skip prefix '{prefix}' should return None for '{keyword}'."""
        locator = f"{prefix}some text"
        result = executor._extract_locator_from_args(keyword, [locator])
        assert result is None


# =============================================================================
# GAP-8: ELEMENT_INTERACTION_KEYWORDS bare-text extraction
# =============================================================================


class TestElementInteractionKeywordsCoverage:
    """All ELEMENT_INTERACTION_KEYWORDS should require pre-validation."""

    @pytest.mark.parametrize("keyword", ELEMENT_INTERACTION_SAMPLE)
    def test_keyword_requires_prevalidation(self, executor, keyword):
        """'{keyword}' should be in ELEMENT_INTERACTION_KEYWORDS."""
        assert executor._requires_pre_validation(keyword), (
            f"'{keyword}' should require pre-validation"
        )

    @pytest.mark.parametrize("keyword", ELEMENT_INTERACTION_SAMPLE)
    def test_keyword_extracts_generic_locator(self, executor, keyword):
        """Element interaction keyword extracts generic locators."""
        # Skip keyword-specific keywords for bare text test
        if keyword in ("click link", "click image", "click button"):
            pytest.skip("Keyword-specific keywords have different bare text handling")

        locator = "id=my_element"
        result = executor._extract_locator_from_args(keyword, [locator])
        assert result == locator


class TestNonInteractionKeywordsSkipPrevalidation:
    """Keywords NOT in ELEMENT_INTERACTION_KEYWORDS should not require pre-validation."""

    @pytest.mark.parametrize("keyword", [
        "log", "should be equal", "get text", "get title",
        "wait for elements state", "set variable", "convert to string",
        "get source", "get url", "go to",
    ])
    def test_non_interaction_keyword_skips(self, executor, keyword):
        """'{keyword}' should NOT require pre-validation."""
        assert not executor._requires_pre_validation(keyword)


# =============================================================================
# GAP-15: Case-insensitive skip prefix matching
# =============================================================================


class TestCaseInsensitiveSkipPrefixes:
    """Skip prefixes should match case-insensitively."""

    @pytest.mark.parametrize("locator", [
        "link=some text",
        "Link=Some Text",
        "LINK=SOME TEXT",
        "partial link=some text",
        "Partial Link=Some Text",
        "PARTIAL LINK=SOME TEXT",
        "link:some text",
        "Link:Some Text",
        "partial link:some text",
        "Partial Link:Some Text",
    ])
    def test_skip_prefix_case_insensitive(self, executor, locator):
        """Skip prefix matching should be case-insensitive."""
        result = executor._extract_locator_from_args("click element", [locator])
        assert result is None, f"Expected None for '{locator}' but got '{result}'"


# =============================================================================
# GAP-4: Unicode locators
# =============================================================================


class TestUnicodeLocators:
    """Test locators with Unicode characters."""

    @pytest.mark.parametrize("locator", [
        "id=\u65e5\u672c\u8a9e\u30c6\u30b9\u30c8",
        "css=.\u00f1o\u00f1o-element",
        "xpath=//div[text()='\u00fcber']",
        "name=\u4e2d\u6587\u540d\u79f0",
    ])
    def test_unicode_locator_extracted(self, executor, locator):
        """Unicode locators should be properly extracted."""
        result = executor._extract_locator_from_args("click element", [locator])
        assert result == locator


# =============================================================================
# GAP-6: Empty/whitespace locators
# =============================================================================


class TestEmptyWhitespaceLocators:
    """Test empty and whitespace-only locators."""

    def test_empty_string_locator(self, executor):
        """Empty string locator should still be returned (not crash)."""
        result = executor._extract_locator_from_args("click element", [""])
        # Empty string doesn't match any skip prefix, so it's returned
        assert result == ""

    def test_whitespace_locator(self, executor):
        """Whitespace-only locator should be returned."""
        result = executor._extract_locator_from_args("click element", ["   "])
        assert result == "   "

    def test_empty_list_returns_none(self, executor):
        """Empty arguments list returns None."""
        result = executor._extract_locator_from_args("click element", [])
        assert result is None


# =============================================================================
# Action type classification
# =============================================================================


class TestActionTypeClassification:
    """Test _get_action_type_from_keyword_for_states."""

    @pytest.mark.parametrize("keyword,expected_action", [
        ("click element", "click"),
        ("double click element", "click"),
        ("fill text", "fill"),
        ("input text", "fill"),
        ("type text", "fill"),
        ("check checkbox", "check"),
        ("uncheck checkbox", "uncheck"),
        ("select from list by value", "select"),
        ("press keys", "press"),
        ("hover", "hover"),
        ("focus", "focus"),
        ("scroll to element", "scroll"),
        ("drag and drop", "drag"),
        ("tap", "tap"),
        ("submit form", "submit"),
        ("choose file", "upload"),
        ("clear element text", "clear"),
    ])
    def test_action_type_classification(self, executor, keyword, expected_action):
        """Keyword '{keyword}' should classify as action type '{expected_action}'."""
        result = executor._get_action_type_from_keyword_for_states(keyword)
        assert result == expected_action

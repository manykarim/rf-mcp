"""Tests for keyword classification for automatic timeout selection.

This module tests the mapping between Robot Framework keywords and
ActionType values for automatic timeout configuration.
"""

import pytest

from robotmcp.domains.timeout import (
    ActionType,
    TimeoutPolicy,
)
from robotmcp.domains.timeout.keyword_classifier import (
    classify_keyword,
    get_timeout_for_keyword,
    normalize_keyword,
    get_all_keywords_for_action,
    is_navigation_keyword,
    is_read_keyword,
    is_wait_keyword,
    CLICK_KEYWORDS,
    FILL_KEYWORDS,
    NAVIGATION_KEYWORDS,
    SELECT_KEYWORDS,
    READ_KEYWORDS,
    WAIT_KEYWORDS,
)


class TestNormalizeKeyword:
    """Tests for keyword normalization."""

    def test_lowercase_conversion(self):
        """Test that keywords are converted to lowercase."""
        assert normalize_keyword("Click Element") == "click_element"
        assert normalize_keyword("GO TO") == "go_to"
        assert normalize_keyword("GET TEXT") == "get_text"

    def test_space_to_underscore(self):
        """Test that spaces are converted to underscores."""
        assert normalize_keyword("click element") == "click_element"
        assert normalize_keyword("go to") == "go_to"
        assert normalize_keyword("input text") == "input_text"

    def test_hyphen_to_underscore(self):
        """Test that hyphens are converted to underscores."""
        assert normalize_keyword("click-element") == "click_element"
        assert normalize_keyword("double-click") == "double_click"

    def test_combined_normalization(self):
        """Test combined normalization scenarios."""
        assert normalize_keyword("Click-Element") == "click_element"
        assert normalize_keyword("DOUBLE CLICK ELEMENT") == "double_click_element"
        assert normalize_keyword("Wait-For-Element") == "wait_for_element"

    def test_already_normalized(self):
        """Test that already normalized keywords are unchanged."""
        assert normalize_keyword("click") == "click"
        assert normalize_keyword("go_to") == "go_to"


class TestClassifyKeyword:
    """Tests for keyword classification."""

    @pytest.mark.parametrize("keyword", [
        "click",
        "click_element",
        "Click Element",
        "click_button",
        "click_link",
        "double_click",
        "right_click",
        "hover",
        "mouse_over",
    ])
    def test_click_keywords_return_click_action(self, keyword):
        """Test that click keywords return CLICK ActionType."""
        assert classify_keyword(keyword) == ActionType.CLICK

    @pytest.mark.parametrize("keyword", [
        "fill_text",
        "input_text",
        "type_text",
        "Input Text",
        "fill_secret",
        "input_password",
        "clear_element_text",
        "press_keys",
    ])
    def test_fill_keywords_return_fill_action(self, keyword):
        """Test that fill keywords return FILL ActionType."""
        assert classify_keyword(keyword) == ActionType.FILL

    @pytest.mark.parametrize("keyword", [
        "go_to",
        "Go To",
        "go_back",
        "go_forward",
        "reload",
        "new_page",
        "new_browser",
        "open_browser",
        "close_browser",
        "wait_for_navigation",
        "wait_until_network_is_idle",
    ])
    def test_navigation_keywords_return_navigate_action(self, keyword):
        """Test that navigation keywords return NAVIGATE ActionType."""
        assert classify_keyword(keyword) == ActionType.NAVIGATE

    @pytest.mark.parametrize("keyword", [
        "select_options",
        "select_from_list",
        "Select From List",
        "select_from_list_by_value",
        "select_from_list_by_label",
        "select_checkbox",
        "unselect_checkbox",
        "check_checkbox",
    ])
    def test_select_keywords_return_select_action(self, keyword):
        """Test that select keywords return SELECT ActionType."""
        assert classify_keyword(keyword) == ActionType.SELECT

    @pytest.mark.parametrize("keyword", [
        "get_text",
        "Get Text",
        "get_value",
        "get_attribute",
        "get_element_states",
        "get_title",
        "get_url",
        "is_visible",
        "is_enabled",
        "take_screenshot",
    ])
    def test_read_keywords_return_get_text_action(self, keyword):
        """Test that read keywords return GET_TEXT ActionType."""
        assert classify_keyword(keyword) == ActionType.GET_TEXT

    @pytest.mark.parametrize("keyword", [
        "wait_for_elements_state",
        "wait_for_element",
        "Wait For Element",
        "wait_until_element_is_visible",
        "wait_until_element_is_enabled",
        "wait_until_page_contains",
        "sleep",
        "wait",
    ])
    def test_wait_keywords_return_wait_for_element_action(self, keyword):
        """Test that wait keywords return WAIT_FOR_ELEMENT ActionType."""
        assert classify_keyword(keyword) == ActionType.WAIT_FOR_ELEMENT

    def test_unknown_keyword_returns_click_action(self):
        """Test that unknown keywords default to CLICK ActionType."""
        assert classify_keyword("some_unknown_keyword") == ActionType.CLICK
        assert classify_keyword("custom_action") == ActionType.CLICK

    def test_case_insensitive_classification(self):
        """Test that classification is case-insensitive."""
        assert classify_keyword("CLICK") == classify_keyword("click")
        assert classify_keyword("Go To") == classify_keyword("GO_TO")
        assert classify_keyword("GET TEXT") == classify_keyword("get_text")


class TestGetTimeoutForKeyword:
    """Tests for timeout retrieval by keyword."""

    @pytest.fixture
    def policy(self):
        """Create a default timeout policy for testing."""
        return TimeoutPolicy.create_default("test_session")

    def test_click_keyword_gets_action_timeout(self, policy):
        """Test that click keywords get action timeout (5s)."""
        timeout = get_timeout_for_keyword("click", policy)
        assert timeout == 5000

    def test_navigation_keyword_gets_navigation_timeout(self, policy):
        """Test that navigation keywords get navigation timeout (60s)."""
        timeout = get_timeout_for_keyword("go_to", policy)
        assert timeout == 60000

    def test_read_keyword_gets_read_timeout(self, policy):
        """Test that read keywords get read timeout (2s)."""
        timeout = get_timeout_for_keyword("get_text", policy)
        assert timeout == 2000

    def test_wait_keyword_gets_assertion_timeout(self, policy):
        """Test that wait keywords get assertion timeout (10s)."""
        timeout = get_timeout_for_keyword("wait_for_element", policy)
        assert timeout == 10000

    def test_fill_keyword_gets_action_timeout(self, policy):
        """Test that fill keywords get action timeout (5s)."""
        timeout = get_timeout_for_keyword("input_text", policy)
        assert timeout == 5000

    def test_select_keyword_gets_action_timeout(self, policy):
        """Test that select keywords get action timeout (5s)."""
        timeout = get_timeout_for_keyword("select_from_list", policy)
        assert timeout == 5000


class TestHelperFunctions:
    """Tests for helper classification functions."""

    def test_is_navigation_keyword_true_for_navigation(self):
        """Test is_navigation_keyword returns True for navigation keywords."""
        assert is_navigation_keyword("go_to") is True
        assert is_navigation_keyword("Go To") is True
        assert is_navigation_keyword("reload") is True
        assert is_navigation_keyword("open_browser") is True

    def test_is_navigation_keyword_false_for_non_navigation(self):
        """Test is_navigation_keyword returns False for non-navigation keywords."""
        assert is_navigation_keyword("click") is False
        assert is_navigation_keyword("get_text") is False
        assert is_navigation_keyword("wait_for_element") is False

    def test_is_read_keyword_true_for_read(self):
        """Test is_read_keyword returns True for read keywords."""
        assert is_read_keyword("get_text") is True
        assert is_read_keyword("Get Text") is True
        assert is_read_keyword("get_attribute") is True
        assert is_read_keyword("is_visible") is True

    def test_is_read_keyword_false_for_non_read(self):
        """Test is_read_keyword returns False for non-read keywords."""
        assert is_read_keyword("click") is False
        assert is_read_keyword("go_to") is False

    def test_is_wait_keyword_true_for_wait(self):
        """Test is_wait_keyword returns True for wait keywords."""
        assert is_wait_keyword("wait_for_element") is True
        assert is_wait_keyword("Wait For Element") is True
        assert is_wait_keyword("sleep") is True

    def test_is_wait_keyword_false_for_non_wait(self):
        """Test is_wait_keyword returns False for non-wait keywords."""
        assert is_wait_keyword("click") is False
        assert is_wait_keyword("go_to") is False


class TestGetAllKeywordsForAction:
    """Tests for get_all_keywords_for_action function."""

    def test_get_click_keywords(self):
        """Test retrieving all CLICK keywords."""
        keywords = get_all_keywords_for_action(ActionType.CLICK)
        assert "click" in keywords
        assert "hover" in keywords
        assert "double_click" in keywords

    def test_get_navigate_keywords(self):
        """Test retrieving all NAVIGATE keywords."""
        keywords = get_all_keywords_for_action(ActionType.NAVIGATE)
        assert "go_to" in keywords
        assert "reload" in keywords
        assert "open_browser" in keywords

    def test_get_fill_keywords(self):
        """Test retrieving all FILL keywords."""
        keywords = get_all_keywords_for_action(ActionType.FILL)
        assert "fill_text" in keywords
        assert "input_text" in keywords

    def test_returns_set(self):
        """Test that result is a set."""
        keywords = get_all_keywords_for_action(ActionType.CLICK)
        assert isinstance(keywords, set)


class TestKeywordSetsCompleteness:
    """Tests to verify keyword sets are properly defined."""

    def test_click_keywords_not_empty(self):
        """Test that CLICK_KEYWORDS has entries."""
        assert len(CLICK_KEYWORDS) > 0
        assert "click" in CLICK_KEYWORDS

    def test_fill_keywords_not_empty(self):
        """Test that FILL_KEYWORDS has entries."""
        assert len(FILL_KEYWORDS) > 0
        assert "fill_text" in FILL_KEYWORDS

    def test_navigation_keywords_not_empty(self):
        """Test that NAVIGATION_KEYWORDS has entries."""
        assert len(NAVIGATION_KEYWORDS) > 0
        assert "go_to" in NAVIGATION_KEYWORDS

    def test_select_keywords_not_empty(self):
        """Test that SELECT_KEYWORDS has entries."""
        assert len(SELECT_KEYWORDS) > 0
        assert "select_options" in SELECT_KEYWORDS

    def test_read_keywords_not_empty(self):
        """Test that READ_KEYWORDS has entries."""
        assert len(READ_KEYWORDS) > 0
        assert "get_text" in READ_KEYWORDS

    def test_wait_keywords_not_empty(self):
        """Test that WAIT_KEYWORDS has entries."""
        assert len(WAIT_KEYWORDS) > 0
        assert "wait_for_element" in WAIT_KEYWORDS

    def test_no_keyword_overlap_between_sets(self):
        """Test that keywords are not in multiple sets."""
        all_sets = [
            CLICK_KEYWORDS,
            FILL_KEYWORDS,
            NAVIGATION_KEYWORDS,
            SELECT_KEYWORDS,
            READ_KEYWORDS,
            WAIT_KEYWORDS,
        ]

        all_keywords = []
        for kw_set in all_sets:
            all_keywords.extend(kw_set)

        # Check for duplicates
        assert len(all_keywords) == len(set(all_keywords)), \
            "Some keywords appear in multiple sets"


class TestEdgeCases:
    """Edge case tests for keyword classifier."""

    def test_empty_keyword(self):
        """Test handling of empty keyword string."""
        # Should default to CLICK for unknown
        result = classify_keyword("")
        assert result == ActionType.CLICK

    def test_whitespace_only_keyword(self):
        """Test handling of whitespace-only keyword."""
        result = classify_keyword("   ")
        assert result == ActionType.CLICK

    def test_keyword_with_extra_spaces(self):
        """Test keyword with extra internal spaces."""
        # "click  element" normalizes to "click__element" which is unknown
        result = classify_keyword("click  element")
        assert result == ActionType.CLICK

    def test_keyword_with_numbers(self):
        """Test keyword containing numbers."""
        result = classify_keyword("custom_action_123")
        assert result == ActionType.CLICK  # Unknown defaults to CLICK

    def test_very_long_keyword(self):
        """Test handling of very long keyword names."""
        long_keyword = "click_" + "a" * 1000
        result = classify_keyword(long_keyword)
        assert result == ActionType.CLICK  # Unknown defaults to CLICK

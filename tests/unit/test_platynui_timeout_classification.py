"""Unit tests for PlatynUI keyword classifications in timeout/keyword_classifier.py (ADR-012).

Tests cover: pointer keywords, keyboard keywords, window keywords,
read keywords, screenshot keywords, and highlight keywords.

Run with: uv run pytest tests/unit/test_platynui_timeout_classification.py -v
"""

__test__ = True

import pytest

from robotmcp.domains.timeout.keyword_classifier import (
    classify_keyword,
    PLATYNUI_POINTER_KEYWORDS,
    PLATYNUI_KEYBOARD_KEYWORDS,
    PLATYNUI_WINDOW_KEYWORDS,
    PLATYNUI_READ_KEYWORDS,
    PLATYNUI_SCREENSHOT_KEYWORDS,
    PLATYNUI_HIGHLIGHT_KEYWORDS,
)
from robotmcp.domains.timeout.entities import ActionType


# ===========================================================================
# Pointer keywords -> CLICK
# ===========================================================================

class TestPointerKeywordsClassification:
    """PlatynUI pointer keywords should classify as CLICK."""

    @pytest.mark.parametrize("keyword", [
        "pointer_click",
        "pointer_multi_click",
        "pointer_press",
        "pointer_release",
        "pointer_move_to",
    ])
    def test_pointer_keyword_is_click(self, keyword):
        assert classify_keyword(keyword) == ActionType.CLICK

    @pytest.mark.parametrize("keyword", [
        "Pointer Click",
        "Pointer Multi Click",
        "Pointer Press",
        "Pointer Release",
        "Pointer Move To",
    ])
    def test_pointer_keyword_with_spaces(self, keyword):
        """classify_keyword normalizes spaces to underscores."""
        assert classify_keyword(keyword) == ActionType.CLICK

    def test_pointer_keywords_set_has_5_entries(self):
        assert len(PLATYNUI_POINTER_KEYWORDS) == 5


# ===========================================================================
# Keyboard keywords -> FILL
# ===========================================================================

class TestKeyboardKeywordsClassification:
    """PlatynUI keyboard keywords should classify as FILL."""

    @pytest.mark.parametrize("keyword", [
        "keyboard_type",
        "keyboard_press",
        "keyboard_release",
    ])
    def test_keyboard_keyword_is_fill(self, keyword):
        assert classify_keyword(keyword) == ActionType.FILL

    @pytest.mark.parametrize("keyword", [
        "Keyboard Type",
        "Keyboard Press",
        "Keyboard Release",
    ])
    def test_keyboard_keyword_with_spaces(self, keyword):
        assert classify_keyword(keyword) == ActionType.FILL

    def test_keyboard_keywords_set_has_3_entries(self):
        assert len(PLATYNUI_KEYBOARD_KEYWORDS) == 3


# ===========================================================================
# Window keywords -> CLICK
# ===========================================================================

class TestWindowKeywordsClassification:
    """PlatynUI window keywords should classify as CLICK."""

    @pytest.mark.parametrize("keyword", [
        "activate",
        "restore",
        "maximize",
        "minimize",
        "close",
        "focus",
    ])
    def test_window_keyword_is_click(self, keyword):
        assert classify_keyword(keyword) == ActionType.CLICK

    @pytest.mark.parametrize("keyword", [
        "Activate",
        "Restore",
        "Maximize",
        "Minimize",
        "Close",
        "Focus",
    ])
    def test_window_keyword_title_case(self, keyword):
        assert classify_keyword(keyword) == ActionType.CLICK

    def test_window_keywords_set_has_6_entries(self):
        assert len(PLATYNUI_WINDOW_KEYWORDS) == 6


# ===========================================================================
# Read keywords -> GET_TEXT
# ===========================================================================

class TestReadKeywordsClassification:
    """PlatynUI read keywords should classify as GET_TEXT."""

    @pytest.mark.parametrize("keyword", [
        "query",
        "get_pointer_position",
    ])
    def test_read_keyword_is_get_text(self, keyword):
        assert classify_keyword(keyword) == ActionType.GET_TEXT

    def test_get_attribute_is_get_text(self):
        # get_attribute appears in both READ_KEYWORDS and PLATYNUI_READ_KEYWORDS,
        # both map to GET_TEXT so the result is consistent.
        assert classify_keyword("get_attribute") == ActionType.GET_TEXT

    @pytest.mark.parametrize("keyword", [
        "Query",
        "Get Pointer Position",
        "Get Attribute",
    ])
    def test_read_keyword_with_spaces(self, keyword):
        assert classify_keyword(keyword) == ActionType.GET_TEXT

    def test_read_keywords_set_has_3_entries(self):
        assert len(PLATYNUI_READ_KEYWORDS) == 3


# ===========================================================================
# Screenshot keyword -> SCREENSHOT
# ===========================================================================

class TestScreenshotKeywordClassification:
    """PlatynUI screenshot keyword classification."""

    def test_take_screenshot_is_screenshot(self):
        # take_screenshot appears in READ_KEYWORDS (GET_TEXT) first,
        # but PLATYNUI_SCREENSHOT_KEYWORDS overrides it to SCREENSHOT
        # because PlatynUI sets are processed after generic sets.
        assert classify_keyword("take_screenshot") == ActionType.SCREENSHOT

    def test_take_screenshot_title_case(self):
        assert classify_keyword("Take Screenshot") == ActionType.SCREENSHOT

    def test_screenshot_keywords_set_has_1_entry(self):
        assert len(PLATYNUI_SCREENSHOT_KEYWORDS) == 1


# ===========================================================================
# Highlight keyword -> GET_TEXT
# ===========================================================================

class TestHighlightKeywordClassification:
    """PlatynUI highlight keyword classification."""

    def test_highlight_is_get_text(self):
        assert classify_keyword("highlight") == ActionType.GET_TEXT

    def test_highlight_title_case(self):
        assert classify_keyword("Highlight") == ActionType.GET_TEXT

    def test_highlight_keywords_set_has_1_entry(self):
        assert len(PLATYNUI_HIGHLIGHT_KEYWORDS) == 1


# ===========================================================================
# Cross-cutting tests
# ===========================================================================

class TestPlatynUIKeywordSetCompleteness:
    """Verify all PlatynUI keyword sets together cover expected keywords."""

    def test_all_platynui_sets_total(self):
        """Union of all PlatynUI keyword sets should match expected count.

        Note: get_attribute and take_screenshot overlap with generic READ_KEYWORDS.
        The PlatynUI-specific sets contain these unique keywords:
        - Pointer: 5, Keyboard: 3, Window: 6, Read: 3, Screenshot: 1, Highlight: 1
        Some keywords (get_attribute, take_screenshot) are in both generic and
        PlatynUI sets but the PlatynUI mapping overrides the generic one.
        """
        all_platynui = (
            PLATYNUI_POINTER_KEYWORDS
            | PLATYNUI_KEYBOARD_KEYWORDS
            | PLATYNUI_WINDOW_KEYWORDS
            | PLATYNUI_READ_KEYWORDS
            | PLATYNUI_SCREENSHOT_KEYWORDS
            | PLATYNUI_HIGHLIGHT_KEYWORDS
        )
        # 5 + 3 + 6 + 3 + 1 + 1 = 19, but get_attribute overlaps with
        # PLATYNUI_READ_KEYWORDS (still 19 unique since no overlap within
        # PlatynUI sets themselves)
        assert len(all_platynui) == 19

    def test_unknown_keyword_defaults_to_click(self):
        """Unknown keywords should fall back to CLICK."""
        assert classify_keyword("some_unknown_keyword") == ActionType.CLICK

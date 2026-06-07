"""Unit tests for PlatynUI keyword timeout classification (ADR-025).

Covers PLATYNUI_ACTION_KEYWORDS -> CLICK, PLATYNUI_KEYBOARD_KEYWORDS -> FILL,
PLATYNUI_READ_KEYWORDS -> GET_TEXT via the public classify_keyword /
normalize_keyword functions.

Run with: uv run pytest tests/unit/test_platynui_newcore_timeout.py -q
"""

__test__ = True

import pytest

from robotmcp.domains.timeout import ActionType
from robotmcp.domains.timeout.keyword_classifier import (
    PLATYNUI_ACTION_KEYWORDS,
    PLATYNUI_KEYBOARD_KEYWORDS,
    PLATYNUI_READ_KEYWORDS,
    classify_keyword,
    normalize_keyword,
)


# =============================================================================
# Normalization of PlatynUI RF keyword names
# =============================================================================


class TestNormalization:
    def test_pointer_click_normalizes(self):
        assert normalize_keyword("Pointer Click") == "pointer_click"

    def test_keyboard_type_normalizes(self):
        assert normalize_keyword("Keyboard Type") == "keyboard_type"

    def test_move_and_resize_window_normalizes(self):
        assert normalize_keyword("Move And Resize Window") == "move_and_resize_window"


# =============================================================================
# Action keywords -> CLICK
# =============================================================================


class TestActionKeywords:
    @pytest.mark.parametrize(
        "keyword",
        ["Pointer Click", "Activate Window", "Bring To Front", "Focus"],
    )
    def test_spot_check_action_keywords_are_click(self, keyword):
        assert classify_keyword(keyword) == ActionType.CLICK

    def test_all_action_keywords_classify_as_click(self):
        for kw in PLATYNUI_ACTION_KEYWORDS:
            # set entries are already normalized (underscored)
            assert classify_keyword(kw) == ActionType.CLICK


# =============================================================================
# Keyboard keywords -> FILL
# =============================================================================


class TestKeyboardKeywords:
    @pytest.mark.parametrize(
        "keyword", ["Keyboard Type", "Keyboard Press", "Keyboard Release"]
    )
    def test_keyboard_keywords_are_fill(self, keyword):
        assert classify_keyword(keyword) == ActionType.FILL

    def test_all_keyboard_keywords_classify_as_fill(self):
        for kw in PLATYNUI_KEYBOARD_KEYWORDS:
            assert classify_keyword(kw) == ActionType.FILL


# =============================================================================
# Read keywords -> GET_TEXT
# =============================================================================


class TestReadKeywords:
    @pytest.mark.parametrize(
        "keyword",
        ["Query", "Set Root", "Get Pointer Position", "Highlight"],
    )
    def test_read_keywords_are_get_text(self, keyword):
        assert classify_keyword(keyword) == ActionType.GET_TEXT

    def test_all_read_keywords_classify_as_get_text(self):
        for kw in PLATYNUI_READ_KEYWORDS:
            assert classify_keyword(kw) == ActionType.GET_TEXT


# =============================================================================
# Set membership sanity
# =============================================================================


class TestSetMembership:
    def test_action_set_includes_window_management(self):
        for kw in (
            "maximize_window",
            "minimize_window",
            "restore_window",
            "close_window",
            "move_window",
            "resize_window",
            "move_and_resize_window",
        ):
            assert kw in PLATYNUI_ACTION_KEYWORDS

    def test_no_overlap_between_platynui_sets(self):
        assert PLATYNUI_ACTION_KEYWORDS.isdisjoint(PLATYNUI_KEYBOARD_KEYWORDS)
        assert PLATYNUI_ACTION_KEYWORDS.isdisjoint(PLATYNUI_READ_KEYWORDS)
        assert PLATYNUI_KEYBOARD_KEYWORDS.isdisjoint(PLATYNUI_READ_KEYWORDS)

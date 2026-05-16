"""Tests for F-N12: _INSPECTION_ONLY_KEYWORDS set.

Verifies content invariants — inspection reads are in the set,
action keywords are not.
"""

__test__ = True

import pytest

from robotmcp.components.execution.keyword_executor import _INSPECTION_ONLY_KEYWORDS


class TestInspectionOnlyKeywordsSet:

    def test_get_title_is_inspection_only(self):
        assert "get title" in _INSPECTION_ONLY_KEYWORDS

    def test_get_url_is_inspection_only(self):
        assert "get url" in _INSPECTION_ONLY_KEYWORDS

    def test_get_text_is_inspection_only(self):
        assert "get text" in _INSPECTION_ONLY_KEYWORDS

    def test_log_is_inspection_only(self):
        assert "log" in _INSPECTION_ONLY_KEYWORDS

    def test_log_to_console_is_inspection_only(self):
        assert "log to console" in _INSPECTION_ONLY_KEYWORDS

    def test_get_location_is_inspection_only(self):
        assert "get location" in _INSPECTION_ONLY_KEYWORDS

    def test_get_element_count_is_inspection_only(self):
        assert "get element count" in _INSPECTION_ONLY_KEYWORDS

    def test_get_attribute_is_inspection_only(self):
        assert "get attribute" in _INSPECTION_ONLY_KEYWORDS

    # Action keywords must NOT appear in the set.

    def test_click_is_not_inspection_only(self):
        assert "click" not in _INSPECTION_ONLY_KEYWORDS

    def test_fill_text_is_not_inspection_only(self):
        assert "fill text" not in _INSPECTION_ONLY_KEYWORDS

    def test_go_to_is_not_inspection_only(self):
        assert "go to" not in _INSPECTION_ONLY_KEYWORDS

    def test_new_page_is_not_inspection_only(self):
        assert "new page" not in _INSPECTION_ONLY_KEYWORDS

    def test_input_text_is_not_inspection_only(self):
        assert "input text" not in _INSPECTION_ONLY_KEYWORDS

    def test_evaluate_javascript_is_not_inspection_only(self):
        # Evaluate JavaScript semantics depend on the JS — must be explicit.
        assert "evaluate javascript" not in _INSPECTION_ONLY_KEYWORDS

    def test_set_is_frozenset(self):
        assert isinstance(_INSPECTION_ONLY_KEYWORDS, frozenset)

    def test_all_entries_are_lowercase(self):
        for kw in _INSPECTION_ONLY_KEYWORDS:
            assert kw == kw.lower(), f"Entry not lowercase: {kw!r}"

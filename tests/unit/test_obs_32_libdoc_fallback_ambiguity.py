"""OBS-32 — LibDoc-fallback ambiguity collapse fix.

The inspection-based fallback path in
``execution_coordinator.get_keyword_documentation`` (rare path, only
triggers when LibDoc isn't available) silently returned the first
match for ambiguous unscoped lookups. The LibDoc primary path
correctly returns a ``matches[]`` array for the same case. OBS-32
makes the fallback path return the same shape.

These tests pin:

1. ``find_all_keywords`` returns every keyword matching a name
   across loaded libraries (the new method on KeywordDiscovery).
2. With LibDoc unavailable, ``get_keyword_documentation("Go To")``
   (no library_name) returns ``matches[]`` not a single ``keyword``.
3. When only one library has the keyword, ``matches[]`` has one entry
   (no special-casing).
4. The LibDoc path is unchanged (the existing matches[] shape there
   continues to work).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_keyword_info(name: str, library: str):
    """Minimal stub matching the KeywordInfo dataclass shape."""
    ki = MagicMock()
    ki.name = name
    ki.library = library
    ki.args = []
    ki.arg_types = []
    ki.doc = f"{name} doc from {library}"
    ki.short_doc = f"{name} short"
    ki.tags = []
    ki.source = ""
    ki.lineno = 0
    return ki


class TestFindAllKeywordsMethod:
    """The new KeywordDiscovery.find_all_keywords method must return
    every matching keyword across all loaded libraries — not just the
    first."""

    def test_returns_all_matches_for_ambiguous_name(self):
        from robotmcp.core.keyword_discovery import KeywordDiscovery
        kd = KeywordDiscovery()
        # Inject two keywords with the same name from different libraries.
        kd.keyword_cache = {
            "browser.go to": _make_keyword_info("Go To", "Browser"),
            "seleniumlibrary.go to": _make_keyword_info("Go To", "SeleniumLibrary"),
        }
        results = kd.find_all_keywords("Go To")
        assert len(results) == 2
        libs = {r.library for r in results}
        assert libs == {"Browser", "SeleniumLibrary"}

    def test_returns_single_entry_when_not_ambiguous(self):
        from robotmcp.core.keyword_discovery import KeywordDiscovery
        kd = KeywordDiscovery()
        kd.keyword_cache = {
            "browser.click": _make_keyword_info("Click", "Browser"),
        }
        results = kd.find_all_keywords("Click")
        assert len(results) == 1
        assert results[0].library == "Browser"

    def test_returns_empty_when_no_match(self):
        from robotmcp.core.keyword_discovery import KeywordDiscovery
        kd = KeywordDiscovery()
        kd.keyword_cache = {
            "browser.click": _make_keyword_info("Click", "Browser"),
        }
        assert kd.find_all_keywords("NoSuchKeyword") == []

    def test_returns_empty_for_empty_name(self):
        from robotmcp.core.keyword_discovery import KeywordDiscovery
        kd = KeywordDiscovery()
        kd.keyword_cache = {
            "browser.click": _make_keyword_info("Click", "Browser"),
        }
        assert kd.find_all_keywords("") == []

    def test_result_order_is_stable_and_sorted_by_library(self):
        from robotmcp.core.keyword_discovery import KeywordDiscovery
        kd = KeywordDiscovery()
        kd.keyword_cache = {
            "seleniumlibrary.go to": _make_keyword_info("Go To", "SeleniumLibrary"),
            "browser.go to": _make_keyword_info("Go To", "Browser"),
        }
        results = kd.find_all_keywords("Go To")
        # Sort key is library name → Browser before SeleniumLibrary.
        assert [r.library for r in results] == ["Browser", "SeleniumLibrary"]

    def test_deduplicates_identical_keyword_instances(self):
        """Same KeywordInfo object indexed under multiple cache keys
        must not produce duplicate result entries."""
        from robotmcp.core.keyword_discovery import KeywordDiscovery
        kd = KeywordDiscovery()
        shared = _make_keyword_info("Click", "Browser")
        kd.keyword_cache = {
            "click": shared,
            "browser.click": shared,
        }
        results = kd.find_all_keywords("Click")
        assert len(results) == 1


class TestInspectionFallbackUsesAllMatches:
    """When LibDoc is unavailable, the inspection-based fallback path
    in ``ExecutionCoordinator.get_keyword_documentation`` must return
    a ``matches[]`` array (same shape as the LibDoc path) rather than
    collapsing to a single keyword."""

    def test_unscoped_lookup_returns_matches_array(self):
        from robotmcp.components.execution.execution_coordinator import (
            ExecutionCoordinator,
        )
        # Don't run __init__ (heavy); build the object surgically.
        coord = ExecutionCoordinator.__new__(ExecutionCoordinator)
        # LibDoc not available → fall through to inspection path.
        coord.rf_doc_storage = MagicMock()
        coord.rf_doc_storage.is_available = MagicMock(return_value=False)
        # Inspection keyword_discovery returns 2 matches for ambiguous name.
        ki_browser = _make_keyword_info("Go To", "Browser")
        ki_sl = _make_keyword_info("Go To", "SeleniumLibrary")
        coord.keyword_executor = MagicMock()
        coord.keyword_executor.keyword_discovery = MagicMock()
        coord.keyword_executor.keyword_discovery.find_all_keywords = MagicMock(
            return_value=[ki_browser, ki_sl]
        )

        result = coord.get_keyword_documentation("Go To")
        assert result["success"] is True
        # matches[] shape (matching LibDoc path), NOT a singular `keyword` field.
        assert "matches" in result
        assert "keyword" not in result
        assert len(result["matches"]) == 2
        libs = {m["library"] for m in result["matches"]}
        assert libs == {"Browser", "SeleniumLibrary"}

    def test_unscoped_lookup_returns_matches_array_when_only_one(self):
        from robotmcp.components.execution.execution_coordinator import (
            ExecutionCoordinator,
        )
        coord = ExecutionCoordinator.__new__(ExecutionCoordinator)
        coord.rf_doc_storage = MagicMock()
        coord.rf_doc_storage.is_available = MagicMock(return_value=False)
        ki = _make_keyword_info("Click", "Browser")
        coord.keyword_executor = MagicMock()
        coord.keyword_executor.keyword_discovery = MagicMock()
        coord.keyword_executor.keyword_discovery.find_all_keywords = MagicMock(
            return_value=[ki]
        )

        result = coord.get_keyword_documentation("Click")
        assert result["success"] is True
        # Single match — STILL returned as matches[] (no special case).
        assert "matches" in result
        assert len(result["matches"]) == 1
        assert result["matches"][0]["library"] == "Browser"

    def test_unscoped_lookup_not_found_returns_error(self):
        from robotmcp.components.execution.execution_coordinator import (
            ExecutionCoordinator,
        )
        coord = ExecutionCoordinator.__new__(ExecutionCoordinator)
        coord.rf_doc_storage = MagicMock()
        coord.rf_doc_storage.is_available = MagicMock(return_value=False)
        coord.keyword_executor = MagicMock()
        coord.keyword_executor.keyword_discovery = MagicMock()
        coord.keyword_executor.keyword_discovery.find_all_keywords = MagicMock(
            return_value=[]
        )
        result = coord.get_keyword_documentation("NoSuchKeyword")
        assert result["success"] is False
        assert "NoSuchKeyword" in result["error"]

    def test_scoped_lookup_path_unchanged(self):
        """When library_name is provided, behaviour is the existing
        per-library path — single keyword shape, not matches[]."""
        from robotmcp.components.execution.execution_coordinator import (
            ExecutionCoordinator,
        )
        coord = ExecutionCoordinator.__new__(ExecutionCoordinator)
        coord.rf_doc_storage = MagicMock()
        coord.rf_doc_storage.is_available = MagicMock(return_value=False)
        ki = _make_keyword_info("Click", "Browser")
        coord.keyword_executor = MagicMock()
        coord.keyword_executor.keyword_discovery = MagicMock()
        coord.keyword_executor.keyword_discovery.find_keyword = MagicMock(
            return_value=ki
        )
        result = coord.get_keyword_documentation("Click", "Browser")
        # Scoped path retains the singular `keyword` shape (unchanged).
        assert result["success"] is True
        assert "keyword" in result
        assert "matches" not in result

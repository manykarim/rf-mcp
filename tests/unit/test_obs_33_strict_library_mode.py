"""OBS-33 — strict_library=True filter mode.

Pre-fix: ``find_keywords(strategy="pattern", query="Get*",
library_name="Browser")`` returned 85 results across 10 libraries
because the library_name filter only excluded plugin-table-
incompatible siblings (SeleniumLibrary). Compatible/neutral
libraries (BuiltIn, Collections, etc.) passed through.

The fix: add ``strict_library: bool = False`` parameter to
``find_keywords``. When True AND a preference is set, exclude EVERY
library that isn't the preferred one. Compatible "neutral" libraries
like BuiltIn / Collections / String / DateTime are dropped too.

These tests pin:
1. Default (strict_library=False) preserves the existing behaviour
   — compatible siblings remain visible. BACKWARDS-COMPAT.
2. strict_library=True + library_name="Browser" → ONLY Browser
   keywords. Neutral helpers excluded.
3. strict_library=True + no library_name → ignored (no library to be
   strict about).
4. Response carries ``library_filter.mode`` field
   ("strict" | "compatible") so agents see which rule fired.
5. Works across all three strategies that use the filter (semantic,
   pattern, catalog).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotmcp.server import (
    _filter_keywords_by_session_library,
    find_keywords,
)


def _fn(tool):
    return getattr(tool, "fn", tool)


def _stub_catalog(library_distribution):
    """Build a synthetic catalog where ``library_distribution`` is
    a list of (library, count) tuples."""
    catalog = []
    counter = 0
    for lib, count in library_distribution:
        for i in range(count):
            catalog.append({
                "name": f"{lib}_kw_{counter}",
                "library": lib,
                "args": [],
                "short_doc": f"kw {counter}",
            })
            counter += 1
    return catalog


class TestFilterBackwardCompat:
    """Default strict_library=False MUST preserve existing behaviour:
    only plugin-incompatible libraries excluded; neutral libraries
    pass through."""

    def test_default_compatible_mode_keeps_neutral_libraries(self):
        catalog = _stub_catalog([
            ("Browser", 5),
            ("BuiltIn", 3),
            ("SeleniumLibrary", 2),  # incompatible with Browser
            ("Collections", 1),
        ])
        filtered, excluded = _filter_keywords_by_session_library(
            catalog, session_id="t", session_library_preference="Browser",
        )
        # SL excluded; Browser + BuiltIn + Collections kept
        kept_libs = {kw["library"] for kw in filtered}
        assert kept_libs == {"Browser", "BuiltIn", "Collections"}
        assert len(excluded) == 2  # 2 SL keywords


class TestStrictLibraryMode:
    """strict_library=True excludes ALL non-preferred libraries."""

    def test_strict_browser_excludes_everything_else(self):
        catalog = _stub_catalog([
            ("Browser", 5),
            ("BuiltIn", 3),
            ("SeleniumLibrary", 2),
            ("Collections", 1),
            ("String", 1),
        ])
        filtered, excluded = _filter_keywords_by_session_library(
            catalog, session_id="t", session_library_preference="Browser",
            strict_library=True,
        )
        # ONLY Browser keywords kept
        kept_libs = {kw["library"] for kw in filtered}
        assert kept_libs == {"Browser"}
        assert len(filtered) == 5
        # All non-Browser keywords excluded (7 total: 3+2+1+1)
        assert len(excluded) == 7

    def test_strict_selenium_excludes_browser_and_neutrals(self):
        """Symmetric test: strict_library=True works for any preference."""
        catalog = _stub_catalog([
            ("Browser", 5),
            ("BuiltIn", 3),
            ("SeleniumLibrary", 4),
        ])
        filtered, excluded = _filter_keywords_by_session_library(
            catalog, session_id="t",
            session_library_preference="SeleniumLibrary",
            strict_library=True,
        )
        kept_libs = {kw["library"] for kw in filtered}
        assert kept_libs == {"SeleniumLibrary"}
        assert len(filtered) == 4

    def test_strict_mode_no_preference_no_op(self):
        """strict_library=True with no preference is a no-op (no library
        to be strict about)."""
        catalog = _stub_catalog([("Browser", 5), ("BuiltIn", 3)])
        filtered, excluded = _filter_keywords_by_session_library(
            catalog, session_id="t", session_library_preference=None,
            strict_library=True,
        )
        # No filter applied — all keywords kept
        assert len(filtered) == 8
        assert len(excluded) == 0

    def test_strict_mode_preserves_keyword_count_diagnostic(self):
        """Excluded count includes all dropped keywords, not just
        plugin-incompatible ones."""
        catalog = _stub_catalog([
            ("Browser", 3),
            ("BuiltIn", 10),  # would normally pass; now excluded
        ])
        _, excluded = _filter_keywords_by_session_library(
            catalog, session_id="t", session_library_preference="Browser",
            strict_library=True,
        )
        assert len(excluded) == 10


@pytest.mark.asyncio
class TestFindKeywordsStrictParameter:
    """find_keywords accepts strict_library and passes it through."""

    async def test_pattern_strategy_strict_library_filters(self):
        """S16-style scenario: Get* + Browser + strict → only Browser
        Get* keywords."""
        async def _ensure_loaded():
            return None

        # Synthetic engine output: 5 Browser Get*, 8 BuiltIn Get*, 2 SL Get*
        results = [
            {"name": f"Get Browser{i}", "library": "Browser",
             "args": [], "short_doc": ""}
            for i in range(5)
        ] + [
            {"name": f"Get BuiltIn{i}", "library": "BuiltIn",
             "args": [], "short_doc": ""}
            for i in range(8)
        ] + [
            {"name": f"Get SL{i}", "library": "SeleniumLibrary",
             "args": [], "short_doc": ""}
            for i in range(2)
        ]
        sess_mgr = MagicMock()
        sess_mgr.get_session = MagicMock(return_value=None)
        engine = MagicMock()
        engine.search_keywords = MagicMock(return_value=results)
        engine.session_manager = sess_mgr

        with patch(
            "robotmcp.server._ensure_all_session_libraries_loaded", _ensure_loaded,
        ), patch(
            "robotmcp.server.execution_engine", engine,
        ), patch(
            "robotmcp.server._externalize_response", side_effect=lambda *a: a[-1],
        ), patch(
            "robotmcp.server._track_tool_result", MagicMock(),
        ):
            # Without strict mode — Browser + BuiltIn kept, SL excluded
            result_compat = await _fn(find_keywords)(
                query="Get*", strategy="pattern", library_name="Browser",
            )
            kept_libs_compat = {kw["library"] for kw in result_compat["results"]}
            assert "BuiltIn" in kept_libs_compat  # neutral kept

            # With strict mode — ONLY Browser
            result_strict = await _fn(find_keywords)(
                query="Get*", strategy="pattern", library_name="Browser",
                strict_library=True,
            )
            kept_libs_strict = {kw["library"] for kw in result_strict["results"]}
            assert kept_libs_strict == {"Browser"}
            assert len(result_strict["results"]) == 5

    async def test_library_filter_mode_field_surfaces(self):
        """The library_filter.mode field tells the agent which rule
        fired ("strict" or "compatible")."""
        async def _ensure_loaded():
            return None
        results = [
            {"name": "Click", "library": "Browser", "args": [], "short_doc": ""},
            {"name": "Log", "library": "BuiltIn", "args": [], "short_doc": ""},
            {"name": "Click Element", "library": "SeleniumLibrary",
             "args": [], "short_doc": ""},
        ]
        sess_mgr = MagicMock()
        sess_mgr.get_session = MagicMock(return_value=None)
        engine = MagicMock()
        engine.search_keywords = MagicMock(return_value=results)
        engine.session_manager = sess_mgr

        with patch(
            "robotmcp.server._ensure_all_session_libraries_loaded", _ensure_loaded,
        ), patch(
            "robotmcp.server.execution_engine", engine,
        ), patch(
            "robotmcp.server._externalize_response", side_effect=lambda *a: a[-1],
        ), patch(
            "robotmcp.server._track_tool_result", MagicMock(),
        ):
            # Compatible mode
            r1 = await _fn(find_keywords)(
                query="Click*", strategy="pattern", library_name="Browser",
            )
            assert r1["library_filter"]["mode"] == "compatible"

            # Strict mode
            r2 = await _fn(find_keywords)(
                query="Click*", strategy="pattern", library_name="Browser",
                strict_library=True,
            )
            assert r2["library_filter"]["mode"] == "strict"

    async def test_strict_library_no_library_name_no_op(self):
        """strict_library=True without library_name (and no session
        preference) is a no-op — no preference to be strict about."""
        async def _ensure_loaded():
            return None
        results = [
            {"name": "Click", "library": "Browser",
             "args": [], "short_doc": ""},
            {"name": "Log", "library": "BuiltIn",
             "args": [], "short_doc": ""},
        ]
        sess_mgr = MagicMock()
        sess_mgr.get_session = MagicMock(return_value=None)
        engine = MagicMock()
        engine.search_keywords = MagicMock(return_value=results)
        engine.session_manager = sess_mgr

        with patch(
            "robotmcp.server._ensure_all_session_libraries_loaded", _ensure_loaded,
        ), patch(
            "robotmcp.server.execution_engine", engine,
        ), patch(
            "robotmcp.server._externalize_response", side_effect=lambda *a: a[-1],
        ), patch(
            "robotmcp.server._track_tool_result", MagicMock(),
        ):
            result = await _fn(find_keywords)(
                query="*", strategy="pattern", strict_library=True,
            )
        # No filter applied — both libraries' keywords remain
        kept_libs = {kw["library"] for kw in result["results"]}
        assert "Browser" in kept_libs
        assert "BuiltIn" in kept_libs

    async def test_default_strict_library_false_unchanged(self):
        """When strict_library is omitted from the call, behaviour is
        identical to pre-OBS-33 (compatible-only filter)."""
        async def _ensure_loaded():
            return None
        results = [
            {"name": "Click", "library": "Browser",
             "args": [], "short_doc": ""},
            {"name": "Log", "library": "BuiltIn",
             "args": [], "short_doc": ""},
            {"name": "Click Element", "library": "SeleniumLibrary",
             "args": [], "short_doc": ""},
        ]
        sess_mgr = MagicMock()
        sess_mgr.get_session = MagicMock(return_value=None)
        engine = MagicMock()
        engine.search_keywords = MagicMock(return_value=results)
        engine.session_manager = sess_mgr

        with patch(
            "robotmcp.server._ensure_all_session_libraries_loaded", _ensure_loaded,
        ), patch(
            "robotmcp.server.execution_engine", engine,
        ), patch(
            "robotmcp.server._externalize_response", side_effect=lambda *a: a[-1],
        ), patch(
            "robotmcp.server._track_tool_result", MagicMock(),
        ):
            result = await _fn(find_keywords)(
                query="*", strategy="pattern", library_name="Browser",
            )
        # SL excluded; Browser + BuiltIn kept (existing compat behaviour)
        kept_libs = {kw["library"] for kw in result["results"]}
        assert "Browser" in kept_libs
        assert "BuiltIn" in kept_libs
        assert "SeleniumLibrary" not in kept_libs

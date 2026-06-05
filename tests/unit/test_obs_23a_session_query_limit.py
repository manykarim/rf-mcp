"""OBS-23A — find_keywords(strategy="session") honours query + limit.

Pre-fix: ``find_keywords(strategy="session", query="click", limit=5)``
silently dumped the entire session namespace because the lower-level
``list_available_keywords`` didn't accept query / limit. Verified
end-to-end Codex round-2: query was decorative, echoed in response
but never consulted.

The fix: ``list_available_keywords`` gains ``name_filter`` and
``limit`` parameters. ``find_keywords`` session branch passes the
caller's ``query`` and ``limit_value`` through.

These tests pin:
1. ``name_filter`` substring-matches keyword names case-insensitively.
2. ``limit`` trims the union (library + resource) to N entries.
3. No filter + no limit → identical pre-fix behaviour (backwards
   compat).
4. ``total_before_trim`` / ``total_after_filter`` diagnostic fields
   surface so callers can detect truncation.
5. find_keywords session branch passes query → name_filter through.
6. Limit application order: filter first, then limit (so ``query="click",
   limit=5`` returns up to 5 click-matching keywords, not 5 arbitrary
   keywords).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotmcp.server import find_keywords


def _fn(tool):
    return getattr(tool, "fn", tool)


def _stub_namespace(library_keywords, resource_keywords=None):
    """Build a synthetic namespace returning library/resource keywords."""
    kw_objs = []
    for spec in library_keywords:
        kw_obj = MagicMock()
        kw_obj.name = spec["name"]
        kw_obj.full_name = spec.get("full_name", spec["name"])
        kw_objs.append((kw_obj, spec.get("library", "Browser")))
    return kw_objs


class TestListAvailableKeywordsFilter:
    """Unit tests against ``RobotFrameworkNativeContextManager.list_available_keywords``
    directly (the backend method) — independent of the MCP wrapper."""

    def _build_mgr_with_session(self, library_keywords, resource_keywords=None):
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
        )
        mgr = RobotFrameworkNativeContextManager()

        # Build mock namespace.libraries — each library has .keywords + .name
        libs = []
        # Group keywords by library
        by_library = {}
        for kw_spec in library_keywords:
            lib_name = kw_spec.get("library", "Browser")
            by_library.setdefault(lib_name, []).append(kw_spec)
        for lib_name, specs in by_library.items():
            lib = MagicMock()
            lib.name = lib_name
            lib.keywords = []
            for spec in specs:
                kw = MagicMock()
                kw.name = spec["name"]
                kw.full_name = spec.get("full_name", spec["name"])
                lib.keywords.append(kw)
            libs.append(lib)

        namespace = MagicMock()
        namespace.libraries = libs
        mgr._session_contexts["test-sess"] = {
            "namespace": namespace,
            "resources": resource_keywords or [],
        }
        return mgr

    def test_no_filter_no_limit_returns_all(self):
        mgr = self._build_mgr_with_session([
            {"name": "Click", "library": "Browser"},
            {"name": "Fill Text", "library": "Browser"},
            {"name": "Go To", "library": "Browser"},
            {"name": "Log", "library": "BuiltIn"},
        ])
        result = mgr.list_available_keywords("test-sess")
        assert result["success"] is True
        assert len(result["library_keywords"]) == 4
        assert result["total_before_trim"] == 4
        assert result["total_after_filter"] == 4

    def test_name_filter_substring_match(self):
        mgr = self._build_mgr_with_session([
            {"name": "Click", "library": "Browser"},
            {"name": "Click Element", "library": "SeleniumLibrary"},
            {"name": "Fill Text", "library": "Browser"},
            {"name": "Go To", "library": "Browser"},
        ])
        result = mgr.list_available_keywords("test-sess", name_filter="click")
        names = [kw["name"] for kw in result["library_keywords"]]
        assert sorted(names) == ["Click", "Click Element"]
        # Pre-trim diagnostic still shows the full count.
        assert result["total_before_trim"] == 4
        # Post-filter shows the filtered count (before any limit applied).
        assert result["total_after_filter"] == 2

    def test_name_filter_case_insensitive(self):
        mgr = self._build_mgr_with_session([
            {"name": "Click", "library": "Browser"},
            {"name": "Fill Text", "library": "Browser"},
        ])
        # Mixed-case filter must still match
        result = mgr.list_available_keywords("test-sess", name_filter="CLICK")
        assert len(result["library_keywords"]) == 1
        assert result["library_keywords"][0]["name"] == "Click"

    def test_name_filter_whitespace_only_treated_as_no_filter(self):
        mgr = self._build_mgr_with_session([
            {"name": "Click", "library": "Browser"},
            {"name": "Fill Text", "library": "Browser"},
        ])
        result = mgr.list_available_keywords("test-sess", name_filter="   ")
        # Whitespace-only filter → no filtering applied
        assert len(result["library_keywords"]) == 2

    def test_limit_trims_to_n(self):
        mgr = self._build_mgr_with_session([
            {"name": f"Keyword{i}", "library": "Browser"} for i in range(20)
        ])
        result = mgr.list_available_keywords("test-sess", limit=5)
        assert len(result["library_keywords"]) == 5
        # Pre-trim diagnostic
        assert result["total_before_trim"] == 20
        # After-filter equals pre-trim when no name_filter
        assert result["total_after_filter"] == 20

    def test_filter_then_limit_order(self):
        """When both filter + limit are present, filter applies FIRST,
        then limit. So ``query="click", limit=3`` returns up to 3 click
        matches (not 3 arbitrary keywords)."""
        mgr = self._build_mgr_with_session(
            [{"name": "Click", "library": "Browser"}]
            + [{"name": f"Click{i}", "library": "Browser"} for i in range(10)]
            + [{"name": f"Other{i}", "library": "Browser"} for i in range(5)]
        )
        result = mgr.list_available_keywords(
            "test-sess", name_filter="click", limit=3,
        )
        assert len(result["library_keywords"]) == 3
        # Every returned entry contains "click"
        for kw in result["library_keywords"]:
            assert "click" in kw["name"].lower()
        # Diagnostics show the picture: 16 total, 11 click-matches,
        # trimmed to 3
        assert result["total_before_trim"] == 16
        assert result["total_after_filter"] == 11

    def test_invalid_limit_zero_no_trim(self):
        mgr = self._build_mgr_with_session([
            {"name": "Click", "library": "Browser"},
            {"name": "Fill Text", "library": "Browser"},
        ])
        # limit=0 is invalid; behaves as no limit
        result = mgr.list_available_keywords("test-sess", limit=0)
        assert len(result["library_keywords"]) == 2

    def test_invalid_limit_negative_no_trim(self):
        mgr = self._build_mgr_with_session([
            {"name": "Click", "library": "Browser"},
            {"name": "Fill Text", "library": "Browser"},
        ])
        result = mgr.list_available_keywords("test-sess", limit=-5)
        assert len(result["library_keywords"]) == 2

    def test_missing_session_returns_error(self):
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
        )
        mgr = RobotFrameworkNativeContextManager()
        result = mgr.list_available_keywords("nonexistent")
        assert result["success"] is False
        assert "No RF context" in result["error"]


@pytest.mark.asyncio
class TestSessionStrategyEndToEnd:
    """End-to-end: find_keywords(strategy="session", query=..., limit=...)
    passes query → name_filter and limit → list_available_keywords."""

    async def test_session_strategy_passes_query_as_name_filter(self):
        mock_mgr = MagicMock()
        mock_mgr.list_available_keywords = MagicMock(
            return_value={
                "success": True,
                "library_keywords": [
                    {"name": "Click", "library": "Browser"},
                ],
                "resource_keywords": [],
                "total_before_trim": 50,
                "total_after_filter": 1,
                "libraries_count": 2,
            },
        )
        with patch(
            "robotmcp.server.get_rf_native_context_manager",
            return_value=mock_mgr,
        ), patch(
            "robotmcp.server._externalize_response",
            side_effect=lambda *args: args[-1],
        ), patch(
            "robotmcp.server._track_tool_result", MagicMock(),
        ):
            result = await _fn(find_keywords)(
                query="click",
                strategy="session",
                session_id="sess-x",
                limit=5,
            )
        # Verify the backend was called with the filter+limit
        _, kwargs = mock_mgr.list_available_keywords.call_args
        assert kwargs.get("name_filter") == "click"
        assert kwargs.get("limit") == 5
        # Response carries the data + strategy + query echo
        assert result["success"] is True
        assert result["strategy"] == "session"
        assert result["query"] == "click"
        # Diagnostic fields surface under result.* (OBS-23B-impl
        # moved these under the unified ``result`` envelope per
        # ``docs/proposals/session_strategy_schema.md`` v2).
        assert result["result"]["total_before_trim"] == 50
        assert result["result"]["total_after_filter"] == 1

    async def test_session_strategy_empty_query_no_filter(self):
        """Empty query → no name_filter passed to backend."""
        mock_mgr = MagicMock()
        mock_mgr.list_available_keywords = MagicMock(
            return_value={
                "success": True,
                "library_keywords": [],
                "resource_keywords": [],
                "total_before_trim": 0,
                "total_after_filter": 0,
                "libraries_count": 0,
            },
        )
        with patch(
            "robotmcp.server.get_rf_native_context_manager",
            return_value=mock_mgr,
        ), patch(
            "robotmcp.server._externalize_response",
            side_effect=lambda *args: args[-1],
        ), patch(
            "robotmcp.server._track_tool_result", MagicMock(),
        ):
            await _fn(find_keywords)(
                query="",
                strategy="session",
                session_id="sess-x",
            )
        _, kwargs = mock_mgr.list_available_keywords.call_args
        # Empty query → name_filter is None
        assert kwargs.get("name_filter") is None

    async def test_session_strategy_no_session_id_errors(self):
        result = await _fn(find_keywords)(
            query="click",
            strategy="session",
        )
        assert result["success"] is False
        assert "session_id is required" in result["error"]

"""OBS-23B-impl Phase 1 — session strategy unified shape + legacy dual-emit.

The session strategy's previous response shape diverged from
semantic/pattern/catalog strategies:
- Top-level ``library_keywords`` + ``resource_keywords`` siblings
- No unified ``result.matches[]``
- No ``recommendations`` prose
- Legacy fields uncovered by existing externalisation rules

Phase 1 dual-emit (this story):
- ADDS the unified ``result`` envelope with ``matches``, ``library_count``,
  ``resource_count``, ``recommendations``, ``total_before_trim``,
  ``total_after_filter``.
- PRESERVES the legacy top-level fields for backwards-compat during
  Phase 2 (deprecation warning, OBS-34, v0.34) → Phase 3 (removal,
  OBS-35, v0.36).
- ADDS externalisation rules for ``library_keywords`` /
  ``resource_keywords`` so the legacy mirror also externalises.

Match shape uses ``match_type: "exact_substring"`` instead of v1's
fake ``confidence=1.00``.

These tests pin:
1. Unified ``result.matches`` is populated from library+resource
   keywords, with correct ``source_type`` per entry.
2. Legacy top-level fields still present.
3. ``library_count``/``resource_count`` count distinct values.
4. ``recommendations`` prose uses honest match-type language.
5. Recommendations exact-match path when query matches a keyword
   name; substring path otherwise.
6. ``total_before_trim`` + ``total_after_filter`` surface under
   ``result``.
7. Externalisation rules added to ``DEFAULT_RULES``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from robotmcp.server import (
    _build_session_recommendations,
    find_keywords,
)


def _fn(tool):
    return getattr(tool, "fn", tool)


class TestExternalizationRules:
    def test_library_keywords_rule_added(self):
        from robotmcp.domains.artifact_output.services import DEFAULT_RULES
        pairs = {(r.tool_name, r.field_path) for r in DEFAULT_RULES}
        assert ("find_keywords", "library_keywords") in pairs

    def test_resource_keywords_rule_added(self):
        from robotmcp.domains.artifact_output.services import DEFAULT_RULES
        pairs = {(r.tool_name, r.field_path) for r in DEFAULT_RULES}
        assert ("find_keywords", "resource_keywords") in pairs


class TestSessionRecommendations:
    """The recommendations helper produces honest prose without
    fake confidence numbers."""

    def test_empty_matches_returns_no_match_prose(self):
        recs = _build_session_recommendations([], "click")
        assert any("No keywords match" in r for r in recs)

    def test_exact_match_naming(self):
        matches = [
            {"keyword_name": "Click", "library": "Browser"},
            {"keyword_name": "Click Element", "library": "SeleniumLibrary"},
        ]
        recs = _build_session_recommendations(matches, "click")
        # Exact match wins (case-insensitive)
        assert recs[0].startswith("Exact match: Click")
        assert "(Browser)" in recs[0]
        # No fake confidence number
        assert "confidence" not in recs[0].lower()

    def test_substring_match_naming(self):
        matches = [
            {"keyword_name": "Click Element", "library": "SeleniumLibrary"},
            {"keyword_name": "Click Button", "library": "SeleniumLibrary"},
        ]
        recs = _build_session_recommendations(matches, "click")
        # No exact match; falls back to substring
        assert "Substring match" in recs[0]
        assert "Click Element" in recs[0]
        # Mentions the failed-exact-match query
        assert "click" in recs[0]

    def test_empty_query_uses_first_in_session(self):
        matches = [
            {"keyword_name": "Log", "library": "BuiltIn"},
            {"keyword_name": "Click", "library": "Browser"},
        ]
        recs = _build_session_recommendations(matches, "")
        assert "First in session" in recs[0]
        assert "Log" in recs[0]

    def test_alternatives_section_when_multiple_matches(self):
        matches = [
            {"keyword_name": "Click", "library": "Browser"},
            {"keyword_name": "Click Element", "library": "SeleniumLibrary"},
            {"keyword_name": "Click Button", "library": "SeleniumLibrary"},
            {"keyword_name": "Tap", "library": "Browser"},
        ]
        recs = _build_session_recommendations(matches, "click")
        # Alternative options appears
        alt_line = next((r for r in recs if r.startswith("Alternative")), None)
        assert alt_line is not None
        # Names matches[1:4]
        assert "Click Element" in alt_line
        assert "Click Button" in alt_line
        assert "Tap" in alt_line

    def test_required_arguments_NOT_present(self):
        """Per design v2 — session strategy doesn't have per-keyword
        arg info, so 'Required arguments' line is intentionally
        omitted (was a v1 design bug)."""
        matches = [
            {"keyword_name": "Click", "library": "Browser"},
        ]
        recs = _build_session_recommendations(matches, "click")
        assert not any(
            "Required arguments" in r or "Required Arguments" in r
            for r in recs
        )


@pytest.mark.asyncio
class TestUnifiedShape:
    """find_keywords(strategy='session') response gains the unified
    ``result.matches`` envelope on top of the existing legacy fields."""

    async def _call(self, library_keywords=None, resource_keywords=None,
                    query="", limit=None, total_before=None, total_after=None):
        mock_mgr = MagicMock()
        mock_mgr.list_available_keywords = MagicMock(return_value={
            "success": True,
            "library_keywords": library_keywords or [],
            "resource_keywords": resource_keywords or [],
            "libraries_count": len({kw.get("library") for kw in (library_keywords or [])}),
            "total_before_trim": total_before if total_before is not None
                                 else len(library_keywords or []) + len(resource_keywords or []),
            "total_after_filter": total_after if total_after is not None
                                  else len(library_keywords or []) + len(resource_keywords or []),
        })
        with patch(
            "robotmcp.server.get_rf_native_context_manager",
            return_value=mock_mgr,
        ), patch(
            "robotmcp.server._externalize_response",
            side_effect=lambda *a: a[-1],
        ), patch(
            "robotmcp.server._track_tool_result", MagicMock(),
        ):
            return await _fn(find_keywords)(
                query=query, strategy="session",
                session_id="sess-x", limit=limit,
            )

    async def test_unified_result_matches_populated(self):
        result = await self._call(
            library_keywords=[
                {"name": "Click", "library": "Browser", "full_name": "Browser.Click"},
                {"name": "Log", "library": "BuiltIn", "full_name": "BuiltIn.Log"},
            ],
            resource_keywords=[
                {"name": "Login User", "resource": "/path/to/keywords.resource",
                 "full_name": "Login User"},
            ],
            query="click",
        )
        # Unified result envelope
        assert "result" in result
        assert "matches" in result["result"]
        # 3 entries total (2 library + 1 resource)
        assert len(result["result"]["matches"]) == 3

    async def test_match_shape_uses_match_type_not_confidence(self):
        result = await self._call(
            library_keywords=[
                {"name": "Click", "library": "Browser"},
            ],
        )
        matches = result["result"]["matches"]
        # match_type field present, NO confidence field (honest shape)
        assert matches[0]["match_type"] == "exact_substring"
        assert "confidence" not in matches[0]

    async def test_source_type_distinguishes_library_from_resource(self):
        result = await self._call(
            library_keywords=[
                {"name": "Click", "library": "Browser"},
            ],
            resource_keywords=[
                {"name": "Login", "resource": "/foo.resource"},
            ],
        )
        matches = result["result"]["matches"]
        library_matches = [m for m in matches if m["source_type"] == "library"]
        resource_matches = [m for m in matches if m["source_type"] == "resource"]
        assert len(library_matches) == 1
        assert library_matches[0]["library"] == "Browser"
        assert library_matches[0]["source"] is None
        assert len(resource_matches) == 1
        assert resource_matches[0]["library"] is None
        assert resource_matches[0]["source"] == "/foo.resource"

    async def test_counts_reflect_distinct_libraries_and_resources(self):
        result = await self._call(
            library_keywords=[
                {"name": "Click", "library": "Browser"},
                {"name": "Tap", "library": "Browser"},  # same lib
                {"name": "Log", "library": "BuiltIn"},  # different lib
            ],
            resource_keywords=[
                {"name": "Login", "resource": "/a.resource"},
                {"name": "Logout", "resource": "/a.resource"},  # same res
                {"name": "Greet", "resource": "/b.resource"},  # different res
            ],
        )
        assert result["result"]["library_count"] == 2  # Browser + BuiltIn
        assert result["result"]["resource_count"] == 2  # /a.resource + /b.resource

    async def test_recommendations_present_in_result(self):
        result = await self._call(
            library_keywords=[
                {"name": "Click", "library": "Browser"},
            ],
            query="click",
        )
        assert "recommendations" in result["result"]
        recs = result["result"]["recommendations"]
        assert recs[0].startswith("Exact match: Click")

    async def test_total_before_trim_surfaces(self):
        result = await self._call(
            library_keywords=[
                {"name": "Click", "library": "Browser"},
            ],
            query="click",
            total_before=50,
            total_after=1,
        )
        # Diagnostic fields under result for agent visibility
        assert result["result"]["total_before_trim"] == 50
        assert result["result"]["total_after_filter"] == 1


@pytest.mark.asyncio
class TestBackwardsCompat:
    """Phase 1 dual-emit MUST preserve legacy top-level fields so
    existing readers (tests/e2e/test_openai_fastmcp.py:294,
    tests/benchmarks/test_robustness_token_overhead.py:124) keep
    working."""

    async def test_legacy_library_keywords_still_top_level(self):
        result = await self._call_helper([
            {"name": "Click", "library": "Browser"},
        ])
        # Legacy field at top level (NOT under result)
        assert "library_keywords" in result
        assert len(result["library_keywords"]) == 1
        assert result["library_keywords"][0]["name"] == "Click"

    async def test_legacy_resource_keywords_still_top_level(self):
        result = await self._call_helper(
            library_keywords=[],
            resource_keywords=[
                {"name": "Login", "resource": "/foo.resource"},
            ],
        )
        assert "resource_keywords" in result
        assert len(result["resource_keywords"]) == 1

    async def test_legacy_libraries_count_still_present(self):
        result = await self._call_helper([
            {"name": "Click", "library": "Browser"},
            {"name": "Log", "library": "BuiltIn"},
        ])
        # Legacy alias for library_count
        assert "libraries_count" in result

    async def test_strategy_and_query_echoed(self):
        result = await self._call_helper(query="click")
        assert result["strategy"] == "session"
        assert result["query"] == "click"

    async def _call_helper(self, library_keywords=None, resource_keywords=None, query=""):
        mock_mgr = MagicMock()
        mock_mgr.list_available_keywords = MagicMock(return_value={
            "success": True,
            "library_keywords": library_keywords or [],
            "resource_keywords": resource_keywords or [],
            "libraries_count": 1,
            "total_before_trim": len(library_keywords or []) + len(resource_keywords or []),
            "total_after_filter": len(library_keywords or []) + len(resource_keywords or []),
        })
        with patch(
            "robotmcp.server.get_rf_native_context_manager",
            return_value=mock_mgr,
        ), patch(
            "robotmcp.server._externalize_response",
            side_effect=lambda *a: a[-1],
        ), patch(
            "robotmcp.server._track_tool_result", MagicMock(),
        ):
            return await _fn(find_keywords)(
                query=query, strategy="session",
                session_id="sess-x",
            )


@pytest.mark.asyncio
class TestRfNativeFailurePropagation:
    """When the backend returns success=False (e.g., 'No RF context'),
    the wrapper carries the error through."""

    async def test_no_rf_context_error_propagated(self):
        mock_mgr = MagicMock()
        mock_mgr.list_available_keywords = MagicMock(return_value={
            "success": False,
            "error": "No RF context for session",
            "session_id": "sess-x",
        })
        with patch(
            "robotmcp.server.get_rf_native_context_manager",
            return_value=mock_mgr,
        ), patch(
            "robotmcp.server._externalize_response",
            side_effect=lambda *a: a[-1],
        ), patch(
            "robotmcp.server._track_tool_result", MagicMock(),
        ):
            result = await _fn(find_keywords)(
                query="", strategy="session", session_id="sess-x",
            )
        assert result["success"] is False
        assert "No RF context" in result["error"]

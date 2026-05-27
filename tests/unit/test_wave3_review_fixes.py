"""Wave-3 round-1 review fixes.

Four cross-LLM reviewers (Codex CLI, Copilot CLI sonnet-4.6, Kilo CLI
MiniMax-M2.7, Claude sub-agent) converged on these blockers in the
initial Wave-3 implementation:

1. **S12 regression**: compound query "wait for X and click Y" was
   classified as ``wait`` (first trigger match wins), causing the
   reranker to penalise the click-class top match. Pre-Wave-3 top
   was ``Browser.Click @ 0.80``, post-Wave-3 became
   ``Browser.Wait For Condition @ 0.54``. Violated design AC #4
   ("no regression on S12") AND the user's hard requirement
   ("rf-mcp does not decrease in keyword finding/search").

2. **Cap-then-rebuild ordering**: ``_rebuild_post_filter_recommendations``
   ran BEFORE the post-filter cap. Result: ``matches[0].confidence``
   was 0.5 but ``recommendations[0]`` read "confidence: 0.72" —
   contradictory output the agent had to reconcile.

3. **OBS-33 session-strategy bypass**: ``strict_library=True`` had no
   effect on ``strategy="session"`` because the session branch never
   invoked the filter helper. Violated OBS-33 AC #3 ("works across
   all strategies").

4. **OBS-23B-impl session_id dropped**: the legacy backend payload
   carries top-level ``session_id``. The unified-shape wrapper
   dropped it. Backwards-compat break.

5. **Stale low_confidence_top_match**: pre-filter cap on the matcher
   could set the flag, then filtering shuffled the top match into a
   non-divergent set, but the flag persisted as a false positive.

These tests pin all five fixes.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotmcp.components.keyword_matcher import (
    KeywordMatch,
    _classify_query_action_class,
    apply_action_class_reranker,
    apply_confidence_cap,
    apply_confidence_cap_dict,
    classify_keyword_action,
)
from robotmcp.config import library_registry
from robotmcp.server import find_keywords


@pytest.fixture(autouse=True)
def _ensure_plugin_state():
    """Tests in this file call ``find_keywords`` end-to-end, which
    hits ``_filter_keywords_by_session_library`` → the global plugin
    manager. ``test_plugin_keyword_validation.py`` has an autouse
    fixture that resets the plugin manager state; running it before
    this file leaves the manager empty for our tests. Re-initialize
    explicitly so the Browser/SeleniumLibrary incompatibility table
    is populated."""
    library_registry._ensure_plugins_registered()
    yield


def _fn(tool):
    return getattr(tool, "fn", tool)


def _mk_match(name, library, confidence, tags=None):
    return KeywordMatch(
        keyword_name=name, library=library, confidence=confidence,
        arguments=[], argument_types=[], documentation="",
        usage_example=None, tags=list(tags or []),
    )


# ---------------------------------------------------------------------------
# Fix 1 — S12 regression: ambiguous compound queries
# ---------------------------------------------------------------------------


class TestAmbiguousQueryClassification:
    """Compound queries matching multiple opinionated classes
    classify as ``ambiguous`` (not first-match-wins). The reranker
    abstains on ambiguous classes."""

    @pytest.mark.parametrize("query,expected_class", [
        # S12-style: wait + click compound
        ("wait for the modal dialog to appear and then click the "
         "confirm button at the bottom of the form", "ambiguous"),
        # Simpler ambiguous combinations
        ("wait for element and click it", "ambiguous"),
        ("click button and verify result", "ambiguous"),
        ("navigate to page and fill form", "ambiguous"),
        ("go to url and wait for load", "ambiguous"),
        # Single-class queries remain unambiguous
        ("click submit button", "click"),
        ("wait for element visible", "wait"),
        ("fill form input field", "fill"),
        ("navigate to url", "navigate"),
        ("select dropdown option", "select"),
        ("verify text is shown", "assert"),
        ("get element text", "query"),
        # No-trigger queries
        ("banana telephone", "unknown"),
        ("send http post request", "unknown"),
        ("", "unknown"),
    ])
    def test_compound_queries_classify_as_ambiguous(self, query, expected_class):
        assert _classify_query_action_class(query) == expected_class

    def test_reranker_abstains_on_ambiguous(self):
        """When query class is ``ambiguous``, the reranker must NOT
        down-weight any class — both wait-class AND click-class
        candidates must keep their pre-rerank confidence."""
        matches = [
            _mk_match("Click", "Browser", 0.85,
                      tags=["PageContent", "Setter"]),
            _mk_match("Wait For Elements State", "Browser", 0.80,
                      tags=["PageContent", "Wait"]),
            _mk_match("Get Text", "Browser", 0.75,
                      tags=["Assertion", "Getter", "PageContent"]),
        ]
        out = apply_action_class_reranker(matches, "ambiguous")
        # Confidences unchanged
        confs = [m.confidence for m in out]
        assert confs == [0.85, 0.80, 0.75]

    def test_cap_trigger_b_does_not_fire_on_ambiguous(self):
        """Trigger B fires only on ``unknown``. ``ambiguous`` queries
        have meaning — no cap."""
        matches = [
            _mk_match("New Persistent Context", "Browser", 0.72,
                      tags=["BrowserControl", "Setter"]),
        ]
        # unknown query → Trigger B fires (existing behaviour pinned)
        _, flag_unknown = apply_confidence_cap(matches, "unknown")
        assert flag_unknown is True
        # ambiguous query → Trigger B does NOT fire
        _, flag_ambig = apply_confidence_cap(matches, "ambiguous")
        assert flag_ambig is False


# ---------------------------------------------------------------------------
# Fix 2 — Cap-then-rebuild ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCapBeforeRebuild:
    """The post-filter confidence cap MUST apply BEFORE recommendations
    are rebuilt. Pre-fix ordering produced contradictory output:
    ``matches[0].confidence`` showed the capped value but
    ``recommendations[0]`` showed the uncapped value."""

    async def test_recommendations_show_capped_confidence(self):
        """Inject 3 matches (1 Browser + 2 SL). Browser filter excludes
        SL → rebuild fires. Query is ``unknown`` + opinionated top
        Browser match at 0.72 → Trigger B → cap to 0.5. Recommendation
        must show the CAPPED 0.5, not the uncapped 0.72."""
        async def _ensure_loaded():
            return None
        discover_mock = AsyncMock(return_value={
            "success": True,
            "matches": [
                {
                    "keyword_name": "New Persistent Context",
                    "library": "Browser",
                    "confidence": 0.72,
                    "arguments": ["userDataDir"],
                    "argument_types": ["str"],
                    "documentation": "",
                    "usage_example": "",
                    "tags": ["BrowserControl", "Setter"],
                },
                {
                    "keyword_name": "Click Element",
                    "library": "SeleniumLibrary",
                    "confidence": 0.80,
                    "arguments": ["locator"],
                    "argument_types": ["str"],
                    "documentation": "",
                    "usage_example": "",
                    "tags": [],
                },
                {
                    "keyword_name": "Set Window Position",
                    "library": "SeleniumLibrary",
                    "confidence": 0.75,
                    "arguments": ["x", "y"],
                    "argument_types": ["int", "int"],
                    "documentation": "",
                    "usage_example": "",
                    "tags": [],
                },
            ],
            "total_matches": 3,
            "recommendations": [
                "Best match: Click Element (confidence: 0.80)",  # pre-filter
            ],
        })
        sess_mgr = MagicMock()
        sess_mgr.get_session = MagicMock(return_value=None)
        engine = MagicMock()
        engine.session_manager = sess_mgr
        engine.search_keywords = MagicMock(return_value=[])
        with patch.dict(os.environ, {"ROBOTMCP_MATCHER_RERANK": "1"}), patch(
            "robotmcp.server.keyword_matcher.discover_keywords", discover_mock,
        ), patch(
            "robotmcp.server._ensure_all_session_libraries_loaded",
            _ensure_loaded,
        ), patch(
            "robotmcp.server._externalize_response",
            side_effect=lambda *a: a[-1],
        ), patch(
            "robotmcp.server._track_tool_result", MagicMock(),
        ), patch(
            "robotmcp.server.execution_engine", engine,
        ):
            result = await _fn(find_keywords)(
                query="send http post request",  # unknown class
                strategy="semantic",
                library_name="Browser",
            )
        res = result["result"]
        # Top match capped at 0.5 (Trigger B fires on unknown query +
        # opinionated Browser top match above cap).
        assert res["matches"][0]["keyword_name"] == "New Persistent Context"
        assert res["matches"][0]["confidence"] == 0.5
        # Flag fires.
        assert res.get("low_confidence_top_match") is True
        # Recommendations rebuilt POST-CAP — must show 0.50, NOT 0.72.
        assert "0.50" in res["recommendations"][0]
        assert "0.72" not in res["recommendations"][0]

    async def test_flag_cleared_when_filter_changes_top_match(self):
        """Pre-filter cap may set ``low_confidence_top_match: True``,
        but if the library filter shuffles top to a non-divergent set,
        the flag must be CLEARED post-filter (not persist as false
        positive)."""
        async def _ensure_loaded():
            return None
        # Pre-filter: 3 matches spanning 3 classes (Trigger A would fire).
        # Library filter excludes 2 SL matches → only 1 Browser match
        # remains → Trigger A can't fire (need ≥3 matches). Trigger B
        # also doesn't fire (query is opinionated 'click').
        discover_mock = AsyncMock(return_value={
            "success": True,
            "matches": [
                {
                    "keyword_name": "Click",
                    "library": "Browser",
                    "confidence": 0.85,
                    "arguments": [],
                    "argument_types": [],
                    "documentation": "",
                    "usage_example": "",
                    "tags": ["PageContent", "Setter"],
                },
                {
                    "keyword_name": "Get Text",
                    "library": "SeleniumLibrary",
                    "confidence": 0.7,
                    "arguments": [],
                    "argument_types": [],
                    "documentation": "",
                    "usage_example": "",
                    "tags": [],
                },
                {
                    "keyword_name": "Wait Until Element Is Visible",
                    "library": "SeleniumLibrary",
                    "confidence": 0.65,
                    "arguments": [],
                    "argument_types": [],
                    "documentation": "",
                    "usage_example": "",
                    "tags": [],
                },
            ],
            "total_matches": 3,
            "recommendations": [],
            # Matcher set the flag based on pre-filter top-3 divergence
            "low_confidence_top_match": True,
        })
        sess_mgr = MagicMock()
        sess_mgr.get_session = MagicMock(return_value=None)
        engine = MagicMock()
        engine.session_manager = sess_mgr
        engine.search_keywords = MagicMock(return_value=[])
        with patch.dict(os.environ, {"ROBOTMCP_MATCHER_RERANK": "1"}), patch(
            "robotmcp.server.keyword_matcher.discover_keywords", discover_mock,
        ), patch(
            "robotmcp.server._ensure_all_session_libraries_loaded",
            _ensure_loaded,
        ), patch(
            "robotmcp.server._externalize_response",
            side_effect=lambda *a: a[-1],
        ), patch(
            "robotmcp.server._track_tool_result", MagicMock(),
        ), patch(
            "robotmcp.server.execution_engine", engine,
        ):
            result = await _fn(find_keywords)(
                query="click button",  # opinionated class
                strategy="semantic",
                library_name="Browser",
            )
        res = result["result"]
        # Only 1 Browser match remains after filter
        assert len(res["matches"]) == 1
        # Flag CLEARED (Trigger A can't fire on <3 matches; Trigger B
        # only on unknown queries).
        assert res.get("low_confidence_top_match") is None or res.get("low_confidence_top_match") is False


# ---------------------------------------------------------------------------
# Fix 3 — OBS-33 wired into session strategy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSessionStrategyStrictLibrary:
    """OBS-33 ``strict_library=True`` must filter the unified
    ``result.matches`` for the session strategy (not silently bypass)."""

    async def test_session_strict_library_excludes_neutrals(self):
        """Browser session with strict_library=True excludes all
        non-Browser libraries (including BuiltIn neutrals)."""
        mock_mgr = MagicMock()
        mock_mgr.list_available_keywords = MagicMock(return_value={
            "success": True,
            "session_id": "sess-x",
            "library_keywords": [
                {"name": "Click", "library": "Browser",
                 "full_name": "Browser.Click"},
                {"name": "Log", "library": "BuiltIn",
                 "full_name": "BuiltIn.Log"},
                {"name": "Click Element", "library": "SeleniumLibrary",
                 "full_name": "SeleniumLibrary.Click Element"},
            ],
            "resource_keywords": [],
            "libraries_count": 3,
            "total_before_trim": 3,
            "total_after_filter": 3,
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
                query="", strategy="session",
                session_id="sess-x",
                library_name="Browser",
                strict_library=True,
            )
        libs = {m["library"] for m in result["result"]["matches"]}
        # Only Browser in matches; BuiltIn + SL excluded.
        assert libs == {"Browser"}
        # library_filter.mode signals strict mode.
        lf = result.get("library_filter")
        assert lf is not None
        assert lf["mode"] == "strict"

    async def test_session_default_keeps_neutrals(self):
        """strict_library=False (default) keeps BuiltIn alongside the
        preferred library (only plugin-incompatible siblings excluded)."""
        mock_mgr = MagicMock()
        mock_mgr.list_available_keywords = MagicMock(return_value={
            "success": True,
            "session_id": "sess-x",
            "library_keywords": [
                {"name": "Click", "library": "Browser"},
                {"name": "Log", "library": "BuiltIn"},
                {"name": "Click Element", "library": "SeleniumLibrary"},
            ],
            "resource_keywords": [],
            "libraries_count": 3,
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
                query="", strategy="session",
                session_id="sess-x",
                library_name="Browser",
                # strict_library defaults to False
            )
        libs = {m["library"] for m in result["result"]["matches"]}
        # SL excluded (incompatible); Browser + BuiltIn kept.
        assert libs == {"Browser", "BuiltIn"}
        lf = result.get("library_filter")
        assert lf is not None
        assert lf["mode"] == "compatible"


# ---------------------------------------------------------------------------
# Fix 4 — session_id preserved in unified-shape response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSessionIdPreserved:
    """The legacy backend payload carries ``session_id`` at top level.
    The unified-shape wrapper must continue surfacing it for
    backwards-compat (pre-fix dropped it)."""

    async def test_session_id_present_at_top_level(self):
        mock_mgr = MagicMock()
        mock_mgr.list_available_keywords = MagicMock(return_value={
            "success": True,
            "session_id": "my-session-id",  # backend top-level field
            "library_keywords": [
                {"name": "Click", "library": "Browser"},
            ],
            "resource_keywords": [],
            "libraries_count": 1,
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
                query="", strategy="session",
                session_id="my-session-id",
            )
        # session_id preserved at top level (legacy callers read it
        # to identify the session).
        assert result.get("session_id") == "my-session-id"

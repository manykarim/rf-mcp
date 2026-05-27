"""find_keywords — ``library_name`` parameter must filter ALL strategies.

The 2026-05-17 defect report exposed a silent failure: an MCP call

    {
      "query": "select dropdown option by label or text",
      "strategy": "semantic",
      "context": "web",
      "library_name": "Browser",
      "limit": 10
    }

in a Browser-only session returned 7 SeleniumLibrary keywords among the
top 10 matches (with "Set Window Position" — completely unrelated — at
the top of the list with confidence 0.82). The ``library_name`` parameter
was accepted at the API surface, documented as "Optional library filter
for catalog search", and silently ignored for semantic/pattern. Agents
reading the response reasonably assume Browser library doesn't have a
``Select Dropdown`` keyword and write a SL-style call that breaks at
execute_step.

The fix wires ``library_name`` through to the existing
``_filter_keywords_by_session_library`` filter for all strategies, with
precedence ``library_name > session.explicit_library_preference > none``.
The filter uses the plugin-driven incompatibility table — for
``library_name="Browser"`` this excludes ``SeleniumLibrary`` and passes
through ``BuiltIn``, ``Collections``, ``String``, etc.

These tests pin:

(1) Reproducer: semantic + library_name="Browser", no session →
    SeleniumLibrary matches excluded; ``excluded_keywords`` populated;
    ``library_filter.source="library_name"``
(2) Symmetry: semantic + library_name="SeleniumLibrary" →
    Browser matches excluded
(3) Strategy parity: pattern + library_name="Browser" → same filter
(4) Precedence: library_name="Browser" + session preference
    "SeleniumLibrary" → library_name wins
(5) Fall-through: no library_name + session preference "Browser" →
    session wins; library_filter.source="session"
(6) Idempotency: neither set → unchanged behaviour, no library_filter
    field, no excluded_keywords
(7) Unknown library: library_name="FakeLib" → no exclusions
(8) Sibling libraries preserved: library_name="Browser" + a BuiltIn
    keyword in results → BuiltIn kept
(9) Docstring contract: parameter still documented and reachable
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotmcp.server import find_keywords


def _async_tool_fn(tool):
    """Unwrap a FastMCP tool to its underlying async function."""
    return getattr(tool, "fn", tool)


# ---------------------------------------------------------------------------
# Synthetic keyword fixtures — match the shape returned by the real engines
# ---------------------------------------------------------------------------


def _semantic_matches() -> List[Dict[str, Any]]:
    """Shape returned by ``KeywordMatcher.discover_keywords`` (under .matches)."""
    return [
        {
            "keyword_name": "Set Window Position",
            "library": "SeleniumLibrary",
            "confidence": 0.82,
            "arguments": ["x", "y"],
            "argument_types": ["int", "int"],
            "documentation": "Sets window position.",
            "usage_example": "Set Window Position    arg1    arg2",
        },
        {
            "keyword_name": "Click",
            "library": "Browser",
            "confidence": 0.80,
            "arguments": ["selector", "button"],
            "argument_types": ["str", "MouseButton"],
            "documentation": "Simulates click on element.",
            "usage_example": "Click    arg1    arg2",
        },
        {
            "keyword_name": "Click Element",
            "library": "SeleniumLibrary",
            "confidence": 0.80,
            "arguments": ["locator"],
            "argument_types": ["str"],
            "documentation": "Clicks element.",
            "usage_example": "Click Element    id=x",
        },
        {
            "keyword_name": "Select Options By",
            "library": "Browser",
            "confidence": 0.77,
            "arguments": ["selector", "attribute", "values"],
            "argument_types": ["str", "SelectAttribute", "str"],
            "documentation": "Select <option>.",
            "usage_example": "Select Options By    arg1    arg2    arg3",
        },
        {
            "keyword_name": "Select From List By Label",
            "library": "SeleniumLibrary",
            "confidence": 0.76,
            "arguments": ["locator", "labels"],
            "argument_types": ["str", "str"],
            "documentation": "Select by label.",
            "usage_example": "Select From List By Label    id=x    arg2",
        },
        {
            "keyword_name": "Log",
            "library": "BuiltIn",
            "confidence": 0.60,
            "arguments": ["message"],
            "argument_types": ["str"],
            "documentation": "Logs a message.",
            "usage_example": "Log    arg1",
        },
    ]


def _pattern_matches() -> List[Dict[str, Any]]:
    """Shape returned by ``execution_engine.search_keywords``."""
    return [
        {"name": "Click", "library": "Browser", "arguments": ["selector"]},
        {"name": "Click Element", "library": "SeleniumLibrary", "arguments": ["locator"]},
        {"name": "Click Button", "library": "SeleniumLibrary", "arguments": ["locator"]},
        {"name": "Tap", "library": "Browser", "arguments": ["selector"]},
        {"name": "Log", "library": "BuiltIn", "arguments": ["message"]},
    ]


# ---------------------------------------------------------------------------
# Test infrastructure: patch the matcher + engine + session lookup
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_engines():
    """Stub the semantic matcher, pattern engine, and ensure-all-libraries
    helper so tests don't hit the real RF runtime.

    Yields a context dict the tests can use to override per-case (e.g.
    to inject a session with a preference)."""
    semantic_payload = {
        "success": True,
        "action_description": "test action",
        "action_type": "click",
        "matches": _semantic_matches(),
        "total_matches": len(_semantic_matches()),
        "recommendations": [],
    }

    async def _ensure_loaded():
        return None

    discover_mock = AsyncMock(return_value=semantic_payload)
    search_mock = MagicMock(return_value=_pattern_matches())
    externalize_mock = MagicMock(side_effect=lambda _tool, _sid, r: r)
    track_mock = MagicMock()

    # session_manager.get_session returns None by default (no session).
    # Tests that need a session preference re-patch this.
    session_manager_mock = MagicMock()
    session_manager_mock.get_session = MagicMock(return_value=None)

    execution_engine_mock = MagicMock()
    execution_engine_mock.session_manager = session_manager_mock
    execution_engine_mock.search_keywords = search_mock

    with patch(
        "robotmcp.server.keyword_matcher.discover_keywords", discover_mock,
    ), patch(
        "robotmcp.server._ensure_all_session_libraries_loaded", _ensure_loaded,
    ), patch(
        "robotmcp.server._externalize_response", externalize_mock,
    ), patch(
        "robotmcp.server._track_tool_result", track_mock,
    ), patch(
        "robotmcp.server.execution_engine", execution_engine_mock,
    ):
        yield {
            "session_manager": session_manager_mock,
            "discover_mock": discover_mock,
            "search_mock": search_mock,
        }


# ---------------------------------------------------------------------------
# 1. Reproducer: the exact failing user payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReproducerSemanticBrowser:
    """The exact call from the 2026-05-17 defect report must now exclude
    SeleniumLibrary matches and surface ``library_filter``."""

    async def test_semantic_browser_filter_excludes_selenium(self, patched_engines):
        result = await _async_tool_fn(find_keywords)(
            query="select dropdown option by label or text",
            strategy="semantic",
            context="web",
            library_name="Browser",
            limit=10,
        )
        matches = result["result"]["matches"]
        libs = sorted({m["library"] for m in matches})
        assert "SeleniumLibrary" not in libs, (
            f"Browser library_name must exclude SL; got libs={libs!r}"
        )
        # Sibling libraries (BuiltIn) are preserved.
        assert "Browser" in libs
        assert "BuiltIn" in libs

    async def test_library_filter_reports_count_and_from_library(self, patched_engines):
        """The compact response carries an exclusion count and the
        from_library label, without re-listing every excluded keyword
        name (those would already be missing from ``matches``, so the
        list is informationally redundant and just inflates tokens)."""
        result = await _async_tool_fn(find_keywords)(
            query="select dropdown option by label or text",
            strategy="semantic",
            context="web",
            library_name="Browser",
            limit=10,
        )
        lf = result.get("library_filter")
        assert lf is not None
        assert lf["count"] == 3  # 3 SL keywords in the fixture
        assert lf["from_library"] == "SeleniumLibrary"

    async def test_library_filter_indicates_source(self, patched_engines):
        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            library_name="Browser",
        )
        lf = result.get("library_filter")
        assert lf is not None
        assert lf["applied"] == "Browser"
        assert lf["source"] == "library_name"

    async def test_verbose_legacy_fields_dropped(self, patched_engines):
        """Both ``excluded_keywords`` (verbose per-entry list) and
        ``session_library`` (redundant with library_filter.applied) are
        dropped from the response. Token budget: ~75 vs ~1100 in the
        pre-compaction shape."""
        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            library_name="Browser",
        )
        assert "excluded_keywords" not in result, (
            "verbose excluded_keywords list must be dropped"
        )
        assert "session_library" not in result, (
            "redundant session_library field must be dropped"
        )


# ---------------------------------------------------------------------------
# 2. Symmetry: SeleniumLibrary filter excludes Browser
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSymmetrySeleniumFilter:
    """The filter must work symmetrically — library_name="SeleniumLibrary"
    excludes Browser, same plugin-table logic."""

    async def test_semantic_selenium_filter_excludes_browser(self, patched_engines):
        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            library_name="SeleniumLibrary",
        )
        matches = result["result"]["matches"]
        libs = sorted({m["library"] for m in matches})
        assert "Browser" not in libs, libs
        assert "SeleniumLibrary" in libs


# ---------------------------------------------------------------------------
# 3. Strategy parity: pattern branch honours library_name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStrategyParityPattern:
    """Pattern strategy must apply the same filter."""

    async def test_pattern_browser_filter_excludes_selenium(self, patched_engines):
        result = await _async_tool_fn(find_keywords)(
            query="click*",
            strategy="pattern",
            library_name="Browser",
        )
        matches = result.get("results", [])
        libs = sorted({m["library"] for m in matches})
        assert "SeleniumLibrary" not in libs, libs
        assert "Browser" in libs
        assert "BuiltIn" in libs
        # Response shape consistent with semantic branch (compact shape).
        lf = result.get("library_filter")
        assert lf is not None
        assert lf["applied"] == "Browser"
        assert lf["source"] == "library_name"
        assert lf["count"] >= 1
        assert lf["from_library"] == "SeleniumLibrary"


# ---------------------------------------------------------------------------
# 4. Precedence: library_name wins over session preference
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPrecedenceLibraryNameOverSession:
    """When BOTH library_name parameter AND session.explicit_library_preference
    are set, the parameter takes precedence (per-call override)."""

    async def test_library_name_overrides_session_preference(self, patched_engines):
        # Session preference set to SeleniumLibrary; per-call library_name
        # passed as Browser. Browser wins → SL excluded.
        sess = MagicMock()
        sess.explicit_library_preference = "SeleniumLibrary"
        patched_engines["session_manager"].get_session = MagicMock(return_value=sess)

        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            session_id="sess-1",
            library_name="Browser",
        )
        libs = sorted({m["library"] for m in result["result"]["matches"]})
        assert "SeleniumLibrary" not in libs, libs
        assert "Browser" in libs
        # library_filter clearly attributes the source.
        assert result["library_filter"]["source"] == "library_name"
        assert result["library_filter"]["applied"] == "Browser"


# ---------------------------------------------------------------------------
# 5. Fall-through: no library_name → session preference applies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFallthroughSessionPreference:
    """The existing session-preference path must still work when
    library_name is not supplied."""

    async def test_session_preference_applies_when_no_library_name(self, patched_engines):
        sess = MagicMock()
        sess.explicit_library_preference = "Browser"
        patched_engines["session_manager"].get_session = MagicMock(return_value=sess)

        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            session_id="sess-1",
        )
        libs = sorted({m["library"] for m in result["result"]["matches"]})
        assert "SeleniumLibrary" not in libs, libs
        assert "Browser" in libs
        # Source attributed to session, not library_name.
        assert result["library_filter"]["source"] == "session"
        assert result["library_filter"]["applied"] == "Browser"


# ---------------------------------------------------------------------------
# 6. Idempotency: no filter when neither is set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestIdempotencyNoFilter:
    """When no library_name AND no session preference, behaviour is
    byte-for-byte the pre-fix shape: no library_filter, no
    excluded_keywords."""

    async def test_no_filter_when_neither_set(self, patched_engines):
        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
        )
        # Both library_name and SL keywords remain.
        libs = sorted({m["library"] for m in result["result"]["matches"]})
        assert "SeleniumLibrary" in libs
        assert "Browser" in libs
        assert "BuiltIn" in libs
        # No filter side-channels.
        assert "library_filter" not in result
        assert "excluded_keywords" not in result
        assert "session_library" not in result

    async def test_no_filter_when_session_has_no_preference(self, patched_engines):
        sess = MagicMock()
        sess.explicit_library_preference = None
        patched_engines["session_manager"].get_session = MagicMock(return_value=sess)
        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            session_id="sess-1",
        )
        libs = sorted({m["library"] for m in result["result"]["matches"]})
        assert "SeleniumLibrary" in libs
        assert "library_filter" not in result


# ---------------------------------------------------------------------------
# 7. Unknown library: no exclusions (graceful)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestUnknownLibrary:
    """An unknown library_name produces no exclusions (the plugin table
    returns an empty incompatibility list). The filter is a no-op rather
    than an error — agents are not blocked by a typo."""

    async def test_unknown_library_no_exclusions(self, patched_engines):
        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            library_name="NoSuchLibrary",
        )
        libs = sorted({m["library"] for m in result["result"]["matches"]})
        # All original libraries remain — no incompatibility table for
        # the unknown name.
        assert "SeleniumLibrary" in libs
        assert "Browser" in libs
        # No exclusions surface, so the response carries no library_filter
        # diagnostic (parallel to the no-filter idempotency case).
        assert "excluded_keywords" not in result
        assert "library_filter" not in result


# ---------------------------------------------------------------------------
# 8. Sibling libraries preserved (BuiltIn, Collections, String)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSiblingLibrariesPreserved:
    """library_name="Browser" must NOT exclude BuiltIn/Collections/etc. —
    only the explicitly-incompatible siblings (SeleniumLibrary)."""

    async def test_builtin_preserved_under_browser_filter(self, patched_engines):
        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            library_name="Browser",
        )
        names = {m["keyword_name"] for m in result["result"]["matches"]}
        # ``Log`` is BuiltIn — must remain after filtering.
        assert "Log" in names, (
            f"BuiltIn keywords must not be excluded by Browser filter; "
            f"got names={sorted(names)!r}"
        )


# ---------------------------------------------------------------------------
# 9. Compaction: only actionable alternatives surface; verbose fields dropped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExcludedKeywordsCompaction:
    """The pre-compaction response carried 7+ verbose entries × ~150
    tokens each (with redundant ``incompatible_library`` /
    ``session_library`` / ``reason`` fields repeated identically). The
    compact response keeps only the actionable translations (entries
    that carry an ``alternative`` from the plugin's KEYWORD_ALTERNATIVES
    table) and surfaces the rest as a single ``count`` field.

    The fixture data above has no ``alternative`` entries because the
    synthetic matches don't trip any plugin-table mappings. We patch
    the filter function to inject a realistic actionable entry and
    pin the response shape."""

    async def test_alternatives_surface_when_plugin_provides_them(self, patched_engines):
        # Inject an actionable alternative via the underlying filter.
        # KEYWORD_ALTERNATIVES on browser_plugin maps "close all browsers"
        # → {"alternative": "Close Browser ALL", "example": "Close Browser    ALL"}
        with patch(
            "robotmcp.server._filter_keywords_by_session_library",
        ) as mock_filter:
            mock_filter.return_value = (
                [{"name": "Click", "library": "Browser"}],
                [
                    {
                        "keyword": "Close All Browsers",
                        "incompatible_library": "SeleniumLibrary",
                        "session_library": "Browser",
                        "reason": "...",
                        "alternative": "Close Browser    ALL",
                        "example": "Close Browser    ALL",
                    },
                    {
                        "keyword": "Set Window Position",
                        "incompatible_library": "SeleniumLibrary",
                        "session_library": "Browser",
                        "reason": "...",
                        # NB: no alternative key — should NOT appear in the
                        # compact ``excluded_alternatives`` list.
                    },
                ],
            )
            result = await _async_tool_fn(find_keywords)(
                query="anything",
                strategy="pattern",
                library_name="Browser",
            )
        # library_filter carries count for ALL excluded entries (2).
        assert result["library_filter"]["count"] == 2
        assert result["library_filter"]["from_library"] == "SeleniumLibrary"
        # excluded_alternatives only carries the actionable one.
        alts = result.get("excluded_alternatives", [])
        assert len(alts) == 1
        assert alts[0]["keyword"] == "Close All Browsers"
        assert alts[0]["alternative"] == "Close Browser    ALL"
        # The non-actionable entry's bare keyword name does NOT appear
        # anywhere in the response — the agent learns it was filtered
        # from its absence in the matches list.
        flat = repr(result)
        assert "Set Window Position" not in flat

    async def test_excluded_alternatives_filters_by_actionability(self, patched_engines):
        """The fixture's SL keywords are:
          - Set Window Position           (no plugin mapping → silent)
          - Click Element                 (mapped → "Click", surfaces)
          - Select From List By Label     (no plugin mapping → silent)

        Only the mapped entry appears in ``excluded_alternatives``.
        The other two are silently dropped from the inline response —
        the agent learns they were filtered from their absence in the
        ``matches`` list."""
        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            library_name="Browser",
        )
        # 3 total excluded entries (matches the fixture count).
        assert result["library_filter"]["count"] == 3
        assert result["library_filter"]["from_library"] == "SeleniumLibrary"
        # Only the Click Element mapping surfaces.
        alts = result.get("excluded_alternatives", [])
        assert len(alts) == 1, (
            f"only Click Element has a plugin mapping; got {alts!r}"
        )
        assert alts[0]["keyword"] == "Click Element"
        assert alts[0]["alternative"] == "Click"
        # The non-actionable SL names do NOT appear in the response.
        flat = repr(result)
        assert "Set Window Position" not in flat
        assert "Select From List By Label" not in flat

    async def test_excluded_alternatives_absent_when_no_mappings_at_all(self, patched_engines):
        """If NO excluded keyword has a plugin mapping, the
        ``excluded_alternatives`` key is absent (not present-but-empty).
        Pinned by injecting two non-mapped SL keywords."""
        with patch(
            "robotmcp.server._filter_keywords_by_session_library",
        ) as mock_filter:
            mock_filter.return_value = (
                [{"name": "Click", "library": "Browser"}],
                [
                    {
                        "keyword": "Set Window Position",
                        "incompatible_library": "SeleniumLibrary",
                        "session_library": "Browser",
                        "reason": "...",
                    },
                    {
                        "keyword": "Maximize Browser Window",
                        "incompatible_library": "SeleniumLibrary",
                        "session_library": "Browser",
                        "reason": "...",
                    },
                ],
            )
            result = await _async_tool_fn(find_keywords)(
                query="anything",
                strategy="pattern",
                library_name="Browser",
            )
        assert "excluded_alternatives" not in result
        # But the count + from_library are still surfaced.
        assert result["library_filter"]["count"] == 2
        assert result["library_filter"]["from_library"] == "SeleniumLibrary"


# ---------------------------------------------------------------------------
# 10. Recommendations rebuild — recommendations follow post-filter matches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRecommendationsRebuild:
    """``KeywordMatcher.discover_keywords`` computes ``recommendations``
    BEFORE the library filter runs in ``find_keywords``. Without a
    rebuild, the "Best match: X" line can name an excluded keyword,
    contradicting the post-filter ``matches`` list and pointing the
    agent at an unavailable keyword.

    Reproduced 2026-05-17 with ``library_name="Browser"`` and query
    "open browser page navigate to url": matches contained only
    Browser keywords but recommendations[0] was "Best match: Get
    Browser Aliases (confidence: 0.84)" — a SeleniumLibrary keyword
    that had just been filtered out."""

    async def test_recommendations_reference_post_filter_top_match(self, patched_engines):
        # Build a discovery payload where the pre-filter top match is
        # an SL keyword (Set Window Position, the matcher's first
        # entry in ``_semantic_matches``) and the pre-filter
        # recommendations name it.
        payload = {
            "success": True,
            "action_description": "test",
            "action_type": "click",
            "matches": _semantic_matches(),  # SL top, Browser later
            "total_matches": len(_semantic_matches()),
            "recommendations": [
                "Best match: Set Window Position (confidence: 0.82)",
                "Required arguments: x, y",
                "Alternative options: Click, Click Element, Select Options By",
            ],
        }
        patched_engines["discover_mock"].return_value = payload

        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            library_name="Browser",
        )
        recs = result["result"]["recommendations"]
        # Top recommendation must reference a kept keyword, not an
        # excluded one.
        assert recs, "recommendations must not be empty"
        assert "Set Window Position" not in recs[0], (
            f"recommendations[0] still references excluded keyword: {recs[0]!r}"
        )
        # The post-filter top match is the Browser ``Click`` entry.
        # OBS-18B (Wave 3 round-1 fix): query="anything" classifies as
        # "unknown"; ``Click`` is an opinionated class → Trigger B
        # fires → confidence capped at 0.50. Recommendation
        # correctly reports the CAPPED value (cap runs BEFORE rebuild
        # per Wave-3 round-1 ordering fix).
        assert "Click" in recs[0]
        assert "confidence: 0.50" in recs[0]

    async def test_recommendations_carry_post_filter_arguments(self, patched_engines):
        """The "Required arguments" line in recommendations must
        reflect the post-filter top match's arguments — not the
        pre-filter top match's arguments."""
        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            library_name="Browser",
        )
        recs = result["result"]["recommendations"]
        # Browser ``Click`` takes (selector, button); SL ``Set Window
        # Position`` took (x, y). After filtering, the args line must
        # be Click's, not Set Window Position's.
        args_line = next((r for r in recs if r.startswith("Required arguments")), None)
        assert args_line is not None
        assert "selector" in args_line
        assert "x, y" not in args_line, (
            f"args line still reflects pre-filter top match: {args_line!r}"
        )

    async def test_recommendations_alternatives_only_kept_keywords(self, patched_engines):
        """The "Alternative options" line must only contain post-filter
        keyword names — no excluded SL keywords leaking in."""
        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            library_name="Browser",
        )
        recs = result["result"]["recommendations"]
        alt_line = next((r for r in recs if r.startswith("Alternative options")), None)
        if alt_line is not None:
            for excluded_name in (
                "Set Window Position", "Click Element",
                "Select From List By Label",
            ):
                assert excluded_name not in alt_line, (
                    f"alt line leaks excluded keyword {excluded_name!r}: {alt_line!r}"
                )

    async def test_recommendations_empty_no_matches_message_when_all_filtered(
        self, patched_engines,
    ):
        """When the filter excludes ALL matches, recommendations must
        show the no-matches guidance (matching the matcher's behaviour
        for the empty case) — NOT a stale top match from the pre-filter
        list."""
        # Build a payload where every match is SL.
        all_sl = [m for m in _semantic_matches() if m["library"] == "SeleniumLibrary"]
        payload = {
            "success": True,
            "action_description": "test",
            "action_type": "click",
            "matches": all_sl,
            "total_matches": len(all_sl),
            "recommendations": [
                f"Best match: {all_sl[0]['keyword_name']} (confidence: 0.82)",
            ],
        }
        patched_engines["discover_mock"].return_value = payload

        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
            library_name="Browser",
        )
        recs = result["result"]["recommendations"]
        # All matches filtered out — recommendations now show the
        # no-matches guidance.
        assert any("No matching keywords found" in r for r in recs), recs
        # The pre-filter top match name must NOT appear.
        for sl in all_sl:
            assert sl["keyword_name"] not in " ".join(recs), (
                f"stale pre-filter name {sl['keyword_name']!r} in recs: {recs!r}"
            )

    async def test_recommendations_unchanged_when_no_filter_applied(self, patched_engines):
        """When no filter applies (no library_name + no session
        preference), the matcher's original recommendations pass
        through unchanged — no rebuild work."""
        sentinel_recs = ["Best match: Set Window Position (confidence: 0.82)"]
        payload = {
            "success": True,
            "action_description": "test",
            "action_type": "click",
            "matches": _semantic_matches(),
            "total_matches": len(_semantic_matches()),
            "recommendations": list(sentinel_recs),
        }
        patched_engines["discover_mock"].return_value = payload

        result = await _async_tool_fn(find_keywords)(
            query="anything",
            strategy="semantic",
        )
        # No filter → recommendations are byte-identical to what the
        # matcher emitted. Pinned via the sentinel string.
        assert result["result"]["recommendations"] == sentinel_recs


# ---------------------------------------------------------------------------
# 11. Docstring contract — the parameter is reachable and named
# ---------------------------------------------------------------------------


class TestDocstringContract:
    """Light contract check: the parameter exists in the signature and
    the docstring no longer says "catalog search" (the old narrow
    promise)."""

    def test_library_name_parameter_present(self):
        import inspect
        sig = inspect.signature(_async_tool_fn(find_keywords))
        assert "library_name" in sig.parameters
        assert sig.parameters["library_name"].default is None

    def test_docstring_announces_all_strategies(self):
        fn = _async_tool_fn(find_keywords)
        doc = (fn.__doc__ or "").lower()
        # Old narrow docstring said "for catalog search". The new doc
        # explicitly says "ALL strategies".
        assert "all strategies" in doc, (
            f"docstring should announce all-strategies coverage; got: {doc[:400]}"
        )

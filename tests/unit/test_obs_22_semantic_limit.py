"""OBS-22 — find_keywords(strategy="semantic") must honour the
``limit`` parameter.

Pre-fix behaviour: ``find_keywords(strategy="semantic", limit=20)``
silently capped at 10 because:
- ``server.py`` parsed ``limit`` into ``limit_value`` and applied it
  in the pattern/catalog branches, but NOT semantic.
- ``KeywordMatcher.discover_keywords`` hard-capped internally at 10
  (``keyword_matcher.py:307``).

Round-2 Codex review caught my original "apply post-filter slice"
proposal: it doesn't honour ``limit > 10`` because the matcher
already capped. The fix threads ``limit`` into ``discover_keywords``
so the matcher can return up to the requested count.

These tests pin:
1. limit=3 → 3 matches
2. limit=20 → up to 20 matches (if 20 exist)
3. limit omitted → default 10 (unchanged from pre-fix)
4. limit=0 / negative → falls back to default 10
5. matcher receives the limit value
6. limit + filter: returned count is the filter-survived subset of
   the (matcher-limit) ranked list
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotmcp.server import find_keywords


def _fn(tool):
    return getattr(tool, "fn", tool)


def _stub_matches(n: int, library: str = "Browser"):
    """Build a synthetic matcher response with `n` entries."""
    return {
        "success": True,
        "action_description": "test",
        "action_type": "click",
        "matches": [
            {
                "keyword_name": f"Keyword{i:03d}",
                "library": library,
                "confidence": 0.9 - (i * 0.01),
                "arguments": ["arg1"],
                "argument_types": ["str"],
                "documentation": f"Doc {i}",
                "usage_example": f"Keyword{i:03d}    arg1",
            }
            for i in range(n)
        ],
        "total_matches": n,
        "recommendations": [],
    }


@pytest.fixture
def patched_engines():
    """Stub matcher + engine + externalisation so tests don't hit RF runtime."""
    discover_mock = AsyncMock()

    async def _ensure_loaded():
        return None

    sess_mgr = MagicMock()
    sess_mgr.get_session = MagicMock(return_value=None)
    engine_mock = MagicMock()
    engine_mock.session_manager = sess_mgr
    engine_mock.search_keywords = MagicMock(return_value=[])

    with patch(
        "robotmcp.server.keyword_matcher.discover_keywords", discover_mock,
    ), patch(
        "robotmcp.server._ensure_all_session_libraries_loaded", _ensure_loaded,
    ), patch(
        "robotmcp.server._externalize_response", side_effect=lambda *a: a[-1],
    ), patch(
        "robotmcp.server._track_tool_result", MagicMock(),
    ), patch(
        "robotmcp.server.execution_engine", engine_mock,
    ):
        yield {"discover_mock": discover_mock}


@pytest.mark.asyncio
class TestLimitHonoured:
    """The caller-supplied ``limit`` reaches the matcher (instead of
    being silently overridden by the matcher's hard cap)."""

    async def test_limit_passed_to_matcher(self, patched_engines):
        patched_engines["discover_mock"].return_value = _stub_matches(20)
        await _fn(find_keywords)(
            query="click", strategy="semantic", limit=20,
        )
        # Matcher receives the limit kwarg.
        _, kwargs = patched_engines["discover_mock"].call_args
        assert kwargs.get("limit") == 20

    async def test_limit_3_returns_3_matches(self, patched_engines):
        # Matcher honours the limit by capping its own output.
        patched_engines["discover_mock"].return_value = _stub_matches(3)
        result = await _fn(find_keywords)(
            query="click", strategy="semantic", limit=3,
        )
        assert len(result["result"]["matches"]) == 3

    async def test_limit_20_can_return_more_than_10(self, patched_engines):
        patched_engines["discover_mock"].return_value = _stub_matches(20)
        result = await _fn(find_keywords)(
            query="click", strategy="semantic", limit=20,
        )
        assert len(result["result"]["matches"]) == 20

    async def test_limit_omitted_defaults_to_None_to_matcher(self, patched_engines):
        """When the caller doesn't pass ``limit``, the matcher
        receives ``limit=None`` and uses its own default of 10."""
        patched_engines["discover_mock"].return_value = _stub_matches(10)
        await _fn(find_keywords)(query="click", strategy="semantic")
        _, kwargs = patched_engines["discover_mock"].call_args
        assert kwargs.get("limit") is None


@pytest.mark.asyncio
class TestLimitEdgeCases:
    """Invalid limit values fall back to default behaviour."""

    async def test_limit_zero_defaults(self, patched_engines):
        """``limit=0`` is invalid; matcher falls back to default 10."""
        patched_engines["discover_mock"].return_value = _stub_matches(10)
        await _fn(find_keywords)(
            query="click", strategy="semantic", limit=0,
        )
        # Server passes the value through; matcher's internal guard
        # (``isinstance(limit, int) and limit > 0``) handles fallback.
        _, kwargs = patched_engines["discover_mock"].call_args
        assert kwargs.get("limit") == 0

    async def test_limit_negative_defaults(self, patched_engines):
        patched_engines["discover_mock"].return_value = _stub_matches(10)
        await _fn(find_keywords)(
            query="click", strategy="semantic", limit=-5,
        )
        _, kwargs = patched_engines["discover_mock"].call_args
        assert kwargs.get("limit") == -5


class TestMatcherLimitGuard:
    """The matcher itself must honour the limit (or fall back to 10
    when invalid)."""

    @pytest.mark.asyncio
    async def test_matcher_returns_up_to_limit(self):
        from robotmcp.components.keyword_matcher import KeywordMatcher
        m = KeywordMatcher()
        # Bypass real initialization; the matcher branches we exercise
        # don't need it.
        m._initialized = True
        m._initialization_lock = MagicMock()

        # Mock the internal matching methods.
        m._pattern_based_matching = AsyncMock(return_value=[])
        m._context_aware_matching = AsyncMock(return_value=[])

        # Inject 30 synthetic deduplicated/ranked matches by mocking
        # the rank step. We override the internal pipeline so the test
        # is deterministic.
        from robotmcp.components.keyword_matcher import KeywordMatch
        fake_ranked = [
            KeywordMatch(
                keyword_name=f"Kw{i:03d}",
                library="Browser",
                confidence=0.9 - i*0.01,
                arguments=[], argument_types=[],
                documentation="", usage_example="",
            )
            for i in range(30)
        ]
        m._rank_matches = MagicMock(return_value=fake_ranked)
        m._deduplicate_matches = MagicMock(return_value=fake_ranked)
        m._normalize_action = MagicMock(return_value="click")
        m._classify_action = MagicMock(return_value="click")
        m._generate_usage_recommendations = MagicMock(return_value=[])

        async def _ensure(): return None
        m._ensure_initialized = _ensure

        # limit=20 → matcher returns 20.
        result = await m.discover_keywords("click", limit=20)
        assert len(result["matches"]) == 20

        # limit=5 → matcher returns 5.
        result = await m.discover_keywords("click", limit=5)
        assert len(result["matches"]) == 5

        # limit omitted → matcher's default 10.
        result = await m.discover_keywords("click")
        assert len(result["matches"]) == 10

        # limit=0 → matcher's default 10 (invalid, falls back).
        result = await m.discover_keywords("click", limit=0)
        assert len(result["matches"]) == 10

        # limit=-3 → matcher's default 10.
        result = await m.discover_keywords("click", limit=-3)
        assert len(result["matches"]) == 10


@pytest.mark.asyncio
class TestLimitInteractsWithFilter:
    """When library filter excludes some matches, the response
    contains the filter-survived subset (which can be less than
    the requested limit). This is intentional — "up to N matches
    that survived filtering" rather than "exactly N matches"."""

    async def test_limit_20_filter_excludes_some(self, patched_engines):
        # 20 matches: 10 Browser, 10 SeleniumLibrary.
        mixed = _stub_matches(10, "Browser")["matches"] + _stub_matches(
            10, "SeleniumLibrary"
        )["matches"]
        patched_engines["discover_mock"].return_value = {
            "success": True, "matches": mixed,
            "total_matches": 20, "recommendations": [],
        }
        result = await _fn(find_keywords)(
            query="click", strategy="semantic",
            library_name="Browser", limit=20,
        )
        libs = {m["library"] for m in result["result"]["matches"]}
        # SL keywords excluded.
        assert "SeleniumLibrary" not in libs
        # Returned count is the filter-survived subset (10 Browser),
        # less than the requested limit of 20.
        assert len(result["result"]["matches"]) == 10

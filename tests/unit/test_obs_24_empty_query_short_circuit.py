"""OBS-24 — find_keywords short-circuits empty/whitespace queries
on semantic + pattern strategies.

Empty queries against the matcher produce a single bogus low-confidence
hit (S09 in the benchmark: ``New Persistent Context`` at confidence
0.35, with 40 arguments in the response = ~838 tokens) because every
keyword scores >0 against an empty action description.

This guard returns a clear error before the matcher runs. Catalog +
session strategies handle empty queries by design (list everything in
scope) and are intentionally not gated.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotmcp.server import find_keywords


def _fn(tool):
    return getattr(tool, "fn", tool)


@pytest.mark.asyncio
class TestEmptyQueryGated:
    """Empty / whitespace-only queries must NOT reach the matcher /
    pattern engine for semantic / pattern strategies."""

    @pytest.mark.parametrize("query", ["", " ", "  ", "\t", "\n", " \t \n "])
    @pytest.mark.parametrize("strategy", ["semantic", "intent", "pattern", "search"])
    async def test_empty_query_returns_error(self, query, strategy):
        result = await _fn(find_keywords)(
            query=query,
            strategy=strategy,
        )
        assert result["success"] is False
        assert result["strategy"] == strategy
        assert "Query string is required" in result["error"]
        assert "hint" in result
        assert "catalog" in result["hint"].lower()

    async def test_matcher_not_invoked_on_empty_query(self):
        """Behaviour pinned via mock: the matcher's discover_keywords
        must NOT be called when the guard fires."""
        with patch(
            "robotmcp.server.keyword_matcher.discover_keywords",
            new_callable=AsyncMock,
        ) as mock_discover:
            result = await _fn(find_keywords)(query="", strategy="semantic")
        assert result["success"] is False
        mock_discover.assert_not_called()


@pytest.mark.asyncio
class TestNonEmptyQueryPassesGuard:
    """Non-empty queries continue through to the matcher / pattern
    engine as before."""

    async def test_one_character_query_passes(self):
        with patch(
            "robotmcp.server.keyword_matcher.discover_keywords",
            new_callable=AsyncMock,
        ) as mock_discover:
            mock_discover.return_value = {
                "success": True,
                "matches": [],
                "total_matches": 0,
                "recommendations": [],
            }
            result = await _fn(find_keywords)(query="x", strategy="semantic")
        assert result["success"] is True
        mock_discover.assert_called_once()

    async def test_whitespace_around_real_query_passes(self):
        """``  click  `` (real word with whitespace padding) must NOT
        be treated as empty — only purely whitespace queries hit the
        guard."""
        with patch(
            "robotmcp.server.keyword_matcher.discover_keywords",
            new_callable=AsyncMock,
        ) as mock_discover:
            mock_discover.return_value = {
                "success": True, "matches": [], "total_matches": 0,
                "recommendations": [],
            }
            result = await _fn(find_keywords)(
                query="  click  ", strategy="semantic",
            )
        assert result["success"] is True
        mock_discover.assert_called_once()


@pytest.mark.asyncio
class TestCatalogAndSessionUnaffected:
    """Catalog + session strategies handle empty queries by design —
    they must NOT be gated by the OBS-24 guard."""

    async def test_catalog_empty_query_passes_guard(self):
        with patch(
            "robotmcp.server._ensure_all_session_libraries_loaded",
            new_callable=AsyncMock,
        ), patch(
            "robotmcp.server.execution_engine"
        ) as mock_engine:
            mock_engine.get_available_keywords = MagicMock(return_value=[])
            mock_engine.session_manager.get_session = MagicMock(return_value=None)
            result = await _fn(find_keywords)(query="", strategy="catalog")
        # Catalog with empty query is allowed (lists all available).
        assert result["success"] is True
        assert result["strategy"] == "catalog"

    async def test_session_empty_query_passes_guard(self):
        """Session strategy with no session_id surfaces its own error
        (different from the OBS-24 guard message)."""
        result = await _fn(find_keywords)(query="", strategy="session")
        assert result["success"] is False
        # The error comes from the session branch's own validation,
        # NOT the OBS-24 guard. Pin by checking it's the session-id
        # error, not the "Query string is required" error.
        assert "session_id is required" in result["error"]

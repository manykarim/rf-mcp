"""OBS-25 — Hard-cap unscoped catalog dumps.

Benchmark S20 showed ``find_keywords(strategy="catalog")`` with no
``session_id``, no ``library_name``, no ``query`` returning 658
keywords across 10 libraries = 97,212 inline tokens. Externalisation
is session-gated, so the no-session call has no rescue path.

OBS-25 adds a hard cap (default 100, env-configurable via
ROBOTMCP_CATALOG_HARD_CAP) that fires ONLY when all three scoping
mechanisms are absent. Any scoped call is intentional and not
truncated.

Tests pin:
1. Unscoped large catalog → truncated + diagnostic hint
2. Scoped catalog (library_name OR query OR session_id) → no cap
3. Hard cap value configurable via env var
4. Caller-supplied limit interacts correctly with the hard cap
5. Small catalog (≤ cap) → no truncation
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotmcp.server import find_keywords


def _fn(tool):
    return getattr(tool, "fn", tool)


def _stub_catalog(n: int, libraries=("Browser", "BuiltIn", "Collections")):
    """Build a synthetic catalog of `n` keywords distributed across
    `libraries`."""
    catalog = []
    for i in range(n):
        lib = libraries[i % len(libraries)]
        catalog.append({
            "name": f"Keyword{i:03d}",
            "library": lib,
            "args": [],
            "short_doc": f"Keyword {i}",
        })
    return catalog


@pytest.fixture
def patched_catalog():
    """Stub the catalog engine + ensure-libraries-loaded hook so tests
    don't hit the real RF runtime."""
    catalog_data = []

    async def _ensure_loaded():
        return None

    catalog_mock = MagicMock(return_value=catalog_data)
    track_mock = MagicMock()
    sess_mgr = MagicMock()
    sess_mgr.get_session = MagicMock(return_value=None)
    exec_engine_mock = MagicMock()
    exec_engine_mock.get_available_keywords = catalog_mock
    exec_engine_mock.session_manager = sess_mgr

    with patch(
        "robotmcp.server._ensure_all_session_libraries_loaded", _ensure_loaded,
    ), patch(
        "robotmcp.server._track_tool_result", track_mock,
    ), patch(
        "robotmcp.server.execution_engine", exec_engine_mock,
    ):
        yield {"catalog_mock": catalog_mock, "catalog_data": catalog_data}


@pytest.mark.asyncio
class TestHardCapFires:
    """When all three scoping mechanisms are absent AND catalog
    exceeds the cap, truncate + diagnostic hint."""

    async def test_unscoped_658_keyword_dump_truncated(self, patched_catalog):
        patched_catalog["catalog_mock"].return_value = _stub_catalog(658)
        result = await _fn(find_keywords)(query="", strategy="catalog")
        assert result["success"] is True
        # Truncation diagnostic surfaces.
        assert result.get("catalog_truncated") is True
        assert result.get("full_catalog_size") == 658
        # match_count is the truncated count, NOT the full count.
        assert result["match_count"] == 100  # default cap
        assert len(result["results"]) == 100
        # Hint mentions the full count + scoping options.
        assert "658" in result["hint"]
        assert "library_name" in result["hint"]
        assert "query" in result["hint"]
        assert "session_id" in result["hint"]

    async def test_cap_value_env_configurable(self, patched_catalog, monkeypatch):
        monkeypatch.setenv("ROBOTMCP_CATALOG_HARD_CAP", "50")
        patched_catalog["catalog_mock"].return_value = _stub_catalog(200)
        result = await _fn(find_keywords)(query="", strategy="catalog")
        assert result["catalog_truncated"] is True
        assert len(result["results"]) == 50
        assert "50" in result["hint"]  # cap value mentioned

    async def test_top_matches_reflects_truncated_list(self, patched_catalog):
        patched_catalog["catalog_mock"].return_value = _stub_catalog(658)
        result = await _fn(find_keywords)(query="", strategy="catalog")
        # top_matches summary is from the truncated list, not the full one.
        assert all(
            n.startswith("Keyword0") or n.startswith("Keyword00")
            for n in result["top_matches"]
        )


@pytest.mark.asyncio
class TestScopedCallsNotCapped:
    """Any scoping mechanism present → no cap, even on huge catalogs."""

    async def test_library_name_scoped_not_capped(self, patched_catalog):
        # 200 keywords against a library_name-scoped call → no cap.
        patched_catalog["catalog_mock"].return_value = _stub_catalog(200, ("Browser",))
        result = await _fn(find_keywords)(
            query="", strategy="catalog", library_name="Browser",
        )
        assert result.get("catalog_truncated") is not True
        assert len(result["results"]) == 200

    async def test_query_filter_scoped_not_capped(self, patched_catalog):
        patched_catalog["catalog_mock"].return_value = _stub_catalog(200)
        result = await _fn(find_keywords)(
            query="Keyword", strategy="catalog",
        )
        # Query filter matches all 200; cap does NOT fire.
        assert result.get("catalog_truncated") is not True
        assert len(result["results"]) == 200

    async def test_session_id_scoped_not_capped(self, patched_catalog):
        sess = MagicMock()
        sess.explicit_library_preference = None
        patched_catalog["catalog_mock"].return_value = _stub_catalog(200)
        # Inject the session so the manager finds one.
        with patch(
            "robotmcp.server.execution_engine.session_manager.get_session",
            return_value=sess,
        ):
            result = await _fn(find_keywords)(
                query="", strategy="catalog", session_id="sess-x",
            )
        assert result.get("catalog_truncated") is not True


@pytest.mark.asyncio
class TestSmallCatalogNotCapped:
    """Catalogs at or below the cap pass through untouched."""

    async def test_catalog_below_cap_unaffected(self, patched_catalog):
        patched_catalog["catalog_mock"].return_value = _stub_catalog(50)
        result = await _fn(find_keywords)(query="", strategy="catalog")
        assert result.get("catalog_truncated") is not True
        assert len(result["results"]) == 50

    async def test_catalog_exactly_at_cap_unaffected(self, patched_catalog):
        patched_catalog["catalog_mock"].return_value = _stub_catalog(100)
        result = await _fn(find_keywords)(query="", strategy="catalog")
        # Exactly at the cap → still not truncated.
        assert result.get("catalog_truncated") is not True
        assert len(result["results"]) == 100


@pytest.mark.asyncio
class TestCapInteractionWithLimit:
    """Caller-supplied ``limit`` interacts predictably with the cap:
    hard cap applies first (truncating to 100); limit then trims the
    truncated set."""

    async def test_limit_below_cap_applies_to_truncated_set(self, patched_catalog):
        patched_catalog["catalog_mock"].return_value = _stub_catalog(658)
        result = await _fn(find_keywords)(
            query="", strategy="catalog", limit=20,
        )
        assert len(result["results"]) == 20
        # Truncation diagnostic still surfaces — the limit trim is on
        # top of the hard cap, not a substitute.
        assert result.get("catalog_truncated") is True
        assert result.get("full_catalog_size") == 658

    async def test_limit_above_cap_caps_at_cap(self, patched_catalog):
        patched_catalog["catalog_mock"].return_value = _stub_catalog(658)
        result = await _fn(find_keywords)(
            query="", strategy="catalog", limit=200,
        )
        # Limit > cap: still capped at 100.
        assert len(result["results"]) == 100

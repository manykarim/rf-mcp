"""E2E test for the unified library recommendation tool using FastMCP Client."""

import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_recommend_libraries_unified_applies_search_order(mcp_client):
    session_id = "unified_reco_sess"
    scenario = "Manipulate data structures and parse XML documents"

    res = await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": scenario,
            "context": "data",
            "max_recommendations": 4,
            "session_id": session_id,
            "check_availability": True,
            "apply_search_order": True,
        },
    )

    assert res.data.get("success") is True
    recs = res.data.get("recommended_libraries", [])
    assert isinstance(recs, list) and len(recs) >= 1

    # Expect data-related libraries to be present
    assert any(name in recs for name in ("XML", "Collections", "String"))

    # Session search order should be applied
    sess = await mcp_client.call_tool("get_session_info", {"session_id": session_id})
    assert sess.data.get("success") is True
    new_order = sess.data.get("session_info", {}).get("search_order", [])
    assert isinstance(new_order, list) and len(new_order) >= 1
    # Ensure recommended items appear in the search order
    assert any(name in new_order for name in recs)

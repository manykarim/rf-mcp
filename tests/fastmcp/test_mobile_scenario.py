import os
import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import mcp

pytestmark = pytest.mark.skipif(
    not os.environ.get("ROBOTMCP_MOBILE_TESTS"),
    reason="Set ROBOTMCP_MOBILE_TESTS=1 to run mobile scenario tests",
)


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_mobile_recommendations(mcp_client):
    scenario_text = "Launch the weather app on Android and capture a screenshot"
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": scenario_text, "context": "mobile"},
    )
    assert analyze.data["success"] is True
    session_id = analyze.data["session_id"]

    rec = await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": scenario_text,
            "context": "mobile",
            "session_id": session_id,
        },
    )
    assert rec.data["success"] is True
    recommended = rec.data["recommended_libraries"]
    assert "AppiumLibrary" in recommended or "Browser" in recommended

    state = await mcp_client.call_tool(
        "get_session_state",
        {"session_id": session_id, "sections": ["summary"]},
    )
    assert state.data["success"] is True

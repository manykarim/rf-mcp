import os
import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import mcp

pytestmark = pytest.mark.skipif(
    not os.environ.get("ROBOTMCP_BROWSER_TESTS"),
    reason="Set ROBOTMCP_BROWSER_TESTS=1 to run browser scenario tests",
)


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_web_scenario_workflow(mcp_client):
    scenario_text = "Open the example.com homepage and verify the title"
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": scenario_text, "context": "web"},
    )
    assert analyze.data["success"] is True
    session_id = analyze.data["session_id"]

    rec = await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": scenario_text,
            "context": "web",
            "session_id": session_id,
        },
    )
    assert rec.data["success"] is True
    recommended = rec.data["recommended_libraries"]
    assert "Browser" in recommended or "SeleniumLibrary" in recommended

    keywords = await mcp_client.call_tool(
        "find_keywords",
        {"query": "open browser", "strategy": "pattern", "limit": 5},
    )
    assert keywords.data["success"] is True

    state = await mcp_client.call_tool(
        "get_session_state",
        {"session_id": session_id, "sections": ["summary"]},
    )
    assert state.data["success"] is True
    assert state.data["sections"]["summary"]["success"] is True

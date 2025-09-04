"""FastMCP E2E tests for sampling-enabled recommendation tools.

Validates the wrapper tools return callable payloads without invoking
decorated prompt objects directly (avoids 'FunctionPrompt not callable' issues).
"""

import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_list_available_libraries_for_prompt(mcp_client):
    res = await mcp_client.call_tool("list_available_libraries_for_prompt", {})
    assert res.data.get("success") is True
    libs = res.data.get("available_libraries")
    assert isinstance(libs, list)
    assert len(libs) >= 3  # BuiltIn, Collections, String at minimum
    # Check expected fields
    sample = libs[0]
    for field in [
        "name",
        "description",
        "categories",
        "requires_setup",
        "setup_commands",
        "use_cases",
    ]:
        assert field in sample


@pytest.mark.asyncio
async def test_recommend_libraries_sampling_tool_returns_prompt(mcp_client):
    avail = await mcp_client.call_tool("list_available_libraries_for_prompt", {})
    libs = avail.data.get("available_libraries")

    res = await mcp_client.call_tool(
        "recommend_libraries_sampling_tool",
        {
            "scenario": "Web UI login and basic API checks; some data parsing",
            "k": 3,
            "available_libraries": libs,
        },
    )
    assert res.data.get("success") is True
    assert isinstance(res.data.get("prompt"), str)
    sampling = res.data.get("recommended_sampling")
    assert sampling and sampling.get("count") == 3


@pytest.mark.asyncio
async def test_choose_recommendations_tool_returns_prompt(mcp_client):
    candidates = [
        {
            "recommendations": [
                {"name": "Browser", "reason": "modern web", "score": 0.8},
                {"name": "RequestsLibrary", "reason": "api", "score": 0.7},
            ],
            "conflicts": [
                {"conflict_set": ["Browser", "SeleniumLibrary"], "chosen": "Browser", "reason": "Playwright"}
            ],
        },
        {
            "recommendations": [
                {"name": "SeleniumLibrary", "reason": "web alt", "score": 0.5},
                {"name": "XML", "reason": "xml parsing", "score": 0.6},
            ]
        },
    ]

    res = await mcp_client.call_tool(
        "choose_recommendations_tool",
        {"candidates": candidates},
    )
    assert res.data.get("success") is True
    assert isinstance(res.data.get("prompt"), str)

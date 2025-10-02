"""Desktop automation integration tests focusing on PlatynUI support."""

from pathlib import Path

import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


DESKTOP_SCENARIO = "Launch the desktop calculator application, click buttons, and assert the result"


@pytest.mark.asyncio
async def test_recommend_libraries_desktop_context(mcp_client):
    response = await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": DESKTOP_SCENARIO,
            "context": "desktop",
            "max_recommendations": 5,
            "check_availability": False,
            "apply_search_order": False,
        },
    )
    data = response.data
    assert data.get("success") is True
    libraries = data.get("recommended_libraries", [])
    assert "PlatynUI" in libraries
    assert "Browser" not in libraries
    assert "AppiumLibrary" not in libraries


@pytest.mark.asyncio
async def test_desktop_environment_diagnostics_available(mcp_client):
    response = await mcp_client.call_tool(
        "diagnose_desktop_environment",
        {"session_id": "desktop_diag"},
    )
    data = response.data
    assert data.get("success") is True
    assert "issues" in data
    assert "detected_backends" in data


@pytest.mark.asyncio
async def test_capture_desktop_tree_file_backend(mcp_client, tmp_path):
    sample_path = Path("test_data/desktop/spy_sample.json").resolve()

    response = await mcp_client.call_tool(
        "capture_desktop_tree",
        {
            "backend": "file",
            "input_path": str(sample_path),
            "attribute_set": "essential",
        },
    )
    data = response.data
    assert data.get("success") is True
    tree = data.get("tree")
    assert isinstance(tree, dict)
    assert tree.get("name") == "Calculator"
    assert tree.get("children")

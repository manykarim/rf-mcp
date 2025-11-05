"""End-to-end tests validating DocTest.VisualTest execution behaviour."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import pytest_asyncio

from fastmcp import Client
from fastmcp.exceptions import ToolError
from robotmcp.server import mcp


# Skip entire module if DocTest.VisualTest is not available in this environment.
try:
    importlib.import_module("DocTest.VisualTest")
except ImportError:  # pragma: no cover - optional dependency missing on some CI jobs
    pytest.skip(
        "DocTest.VisualTest is not installed; skipping DocTest Visual integration tests",
        allow_module_level=True,
    )


SCREENSHOT_DIR = Path(__file__).resolve().parents[2] / "screenshots"
REFERENCE_IMAGE = SCREENSHOT_DIR / "firefox_home.png"
CANDIDATE_IMAGE = SCREENSHOT_DIR / "firefox_home.png"


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


async def _prepare_session(mcp_client: Client, session_id: str) -> None:
    scenario_text = (
        "Perform visual regression comparisons using DocTest.VisualTest between reference and candidate images."
    )
    await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": scenario_text, "context": "desktop", "session_id": session_id},
    )
    await mcp_client.call_tool(
        "recommend_libraries",
        {
            "scenario": scenario_text,
            "context": "desktop",
            "session_id": session_id,
            "max_recommendations": 5,
            "check_availability": True,
            "apply_search_order": True,
        },
    )


@pytest.mark.asyncio
async def test_compare_images_valid_paths_returns_success(mcp_client: Client):
    session_id = "doctest_visual_valid"
    await _prepare_session(mcp_client, session_id)
    await mcp_client.call_tool(
        "import_custom_library",
        {"session_id": session_id, "name_or_path": "DocTest.VisualTest"},
    )

    res = await mcp_client.call_tool(
        "execute_step",
        {
            "session_id": session_id,
            "keyword": "Compare Images",
            "arguments": [str(REFERENCE_IMAGE), str(CANDIDATE_IMAGE)],
            "use_context": True,
        },
    )

    payload = res.data
    assert payload.get("success") is True
    assert payload.get("status") == "pass"
    assert payload.get("error") in (None, "")


@pytest.mark.asyncio
async def test_compare_images_invalid_paths_returns_failure_quickly(mcp_client: Client):
    session_id = "doctest_visual_invalid"
    await _prepare_session(mcp_client, session_id)
    await mcp_client.call_tool(
        "import_custom_library",
        {"session_id": session_id, "name_or_path": "DocTest.VisualTest"},
    )

    invalid_path = str(REFERENCE_IMAGE) + ".missing"

    with pytest.raises(ToolError) as exc:
        await mcp_client.call_tool(
            "execute_step",
            {
                "session_id": session_id,
                "keyword": "Compare Images",
                "arguments": [invalid_path, str(CANDIDATE_IMAGE)],
                "use_context": True,
            },
        )

    msg = str(exc.value)
    assert "Cannot load image" in msg
    assert invalid_path in msg

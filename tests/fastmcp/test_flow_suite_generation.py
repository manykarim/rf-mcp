"""E2E tests that exercise flow tools and then build a test suite.

These ensure that steps executed via flow tools are recorded and can be rendered
into a Robot Framework suite without errors.
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
async def test_flow_for_each_then_build_suite(mcp_client):
    session_id = "flow_suite_for_each"
    # Run a FOR_EACH that logs items
    res = await mcp_client.call_tool(
        "execute_for_each",
        {
            "session_id": session_id,
            "items": [1, 2],
            "steps": [
                {"keyword": "Log", "arguments": ["loop"]},
                {"keyword": "Evaluate", "arguments": ["int(${item}) + 100"], "assign_to": "VAL"},
            ],
        },
    )
    assert res.data.get("count") == 2

    # Build suite
    suite = await mcp_client.call_tool(
        "build_test_suite", {"test_name": "Flow ForEach", "session_id": session_id}
    )
    assert suite.data.get("success") is True
    rf_text = suite.data.get("rf_text", "")
    assert "Log    loop" in rf_text
    # assignment should be rendered with Evaluate line
    assert "Evaluate    int(${item}) + 100" in rf_text or "Evaluate    int(${item}) + 100" in rf_text


@pytest.mark.asyncio
async def test_flow_if_then_build_suite(mcp_client):
    session_id = "flow_suite_if"
    # Set variable and run IF
    await mcp_client.call_tool(
        "set_variables", {"session_id": session_id, "variables": {"X": 1}}
    )
    res = await mcp_client.call_tool(
        "execute_if",
        {
            "session_id": session_id,
            "condition": "int(${X}) == 1",
            "then_steps": [{"keyword": "Log", "arguments": ["then-branch"]}],
            "else_steps": [{"keyword": "Log", "arguments": ["else-branch"]}],
        },
    )
    assert res.data.get("branch_taken") == "then"

    # Build suite
    suite = await mcp_client.call_tool(
        "build_test_suite", {"test_name": "Flow If", "session_id": session_id}
    )
    assert suite.data.get("success") is True
    rf_text = suite.data.get("rf_text", "")
    assert "Log    then-branch" in rf_text


@pytest.mark.asyncio
async def test_flow_try_then_build_suite(mcp_client):
    session_id = "flow_suite_try"
    # Execute TRY/EXCEPT using BuiltIn keywords to avoid external deps
    res = await mcp_client.call_tool(
        "execute_try_except",
        {
            "session_id": session_id,
            "try_steps": [{"keyword": "Fail", "arguments": ["boom"]}],
            "except_patterns": ["*"],
            "except_steps": [{"keyword": "Log", "arguments": ["handled"]}],
        },
    )
    assert res.data.get("success") is True
    assert res.data.get("handled") is True

    # Build suite
    suite = await mcp_client.call_tool(
        "build_test_suite", {"test_name": "Flow Try/Except", "session_id": session_id}
    )
    assert suite.data.get("success") is True
    rf_text = suite.data.get("rf_text", "")
    assert "TRY" in rf_text and "EXCEPT" in rf_text and "END" in rf_text
    assert "Log    handled" in rf_text

    # Validate structured steps include control nodes
    cases = suite.data.get("suite", {}).get("test_cases", [])
    assert cases
    ssteps = cases[0].get("structured_steps", [])
    kinds = [s.get("control") for s in ssteps if s.get("type") == "control"]
    assert "TRY" in kinds and "EXCEPT" in kinds and "END" in kinds

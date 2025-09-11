"""Reproduce execute_step failure after only a suite dry run (no full run).

This is a lighter-weight variant to avoid potential environment hangs during
`run_test_suite` while still exercising RF CLI dry run and then calling
`execute_step` again on the same session.
"""

import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import mcp


TIMEOUT = 60


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_execute_step_after_dry_run_only(mcp_client):
    session_id = "post_dry_run_context"

    # 1) Seed the session with a simple BuiltIn keyword
    r1 = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Log",
            "arguments": ["Seed step"],
            "session_id": session_id,
        },
        timeout=TIMEOUT,
    )
    assert r1.data.get("success") is True

    # 2) Build a suite
    build = await mcp_client.call_tool(
        "build_test_suite",
        {
            "session_id": session_id,
            "test_name": "Minimal Dry Run Repro",
            "documentation": "Suite generated for dry run reproduction",
            "tags": ["regression", "context"],
        },
        timeout=TIMEOUT,
    )
    assert build.data.get("success") is True

    # 3) Dry run the suite
    dry = await mcp_client.call_tool(
        "run_test_suite_dry",
        {"session_id": session_id, "validation_level": "standard"},
        timeout=TIMEOUT,
    )
    assert dry.data.get("success") is True

    # 4) Try another execute_step post-dry-run
    r2 = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Log",
            "arguments": ["Post-dry-run step"],
            "session_id": session_id,
        },
        timeout=TIMEOUT,
    )
    assert r2.data.get("success") is True, (
        "Expected post-dry-run execute_step to succeed; got error: %s" % r2.data
    )


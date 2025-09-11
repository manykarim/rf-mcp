"""Reproduce execute_step failure after suite dry run and execution.

This test simulates a minimal, network-free scenario:
- Execute a simple BuiltIn step (Log)
- Build a suite from the session
- Validate with run_test_suite_dry
- Execute with run_test_suite
- Try another execute_step and assert it still succeeds

Historically, some environments reported a failure like:
  'NoneType' object has no attribute 'is_logged'
on the execute_step after running a suite. This test aims to catch that regression.
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
async def test_execute_step_after_suite_run(mcp_client):
    # Create a named session to make identification clearer
    session_id = "post_suite_exec_context"

    # 1) Execute a simple BuiltIn keyword to seed the session
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

    # 2) Build a Robot Framework test suite from the session steps
    build = await mcp_client.call_tool(
        "build_test_suite",
        {
            "session_id": session_id,
            "test_name": "Minimal Post-Run Repro",
            "documentation": "Suite generated for post-run execute_step reproduction",
            "tags": ["regression", "context"]
        },
        timeout=TIMEOUT,
    )
    assert build.data.get("success") is True
    assert isinstance(build.data.get("rf_text"), str) and len(build.data.get("rf_text")) > 0

    # 3) Validate suite (dry run)
    dry = await mcp_client.call_tool(
        "run_test_suite_dry",
        {
            "session_id": session_id,
            "validation_level": "standard",
        },
        timeout=TIMEOUT,
    )
    assert isinstance(dry.data, dict)
    assert dry.data.get("tool") == "run_test_suite_dry"
    assert dry.data.get("success") is True, f"Dry run failed: {dry.data}"

    # 4) Execute the suite normally
    run = await mcp_client.call_tool(
        "run_test_suite",
        {
            "session_id": session_id,
            "output_level": "minimal",
        },
        timeout=TIMEOUT,
    )
    assert isinstance(run.data, dict)
    assert run.data.get("tool") == "run_test_suite"
    assert run.data.get("success") is True, f"Suite execution failed: {run.data}"

    # 5) Try another execute_step on the same session after suite execution
    #    This is where the NoneType.is_logged error was observed in the wild.
    r2 = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Log",
            "arguments": ["Post-run step"],
            "session_id": session_id,
        },
        timeout=TIMEOUT,
    )

    assert r2.data.get("success") is True, (
        "Expected post-run execute_step to succeed; got error: %s" % r2.data
    )


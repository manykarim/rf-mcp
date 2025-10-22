"""Reproduce execute_step failure after full suite run (RequestsLibrary session).

This test mirrors the reported scenario where after building a suite and
executing it, a subsequent execute_step fails with:
    'NoneType' object has no attribute 'is_logged'

It uses a minimal RequestsLibrary flow against Restful Booker, then runs
build_test_suite → run_test_suite_dry → run_test_suite, followed by another
execute_step on the same session. The final step should succeed.

Network-gated: set RUN_NETWORK_TESTS=1 to enable.
"""

import os
import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import mcp

from tests.utils.dependency_matrix import requires_extras

pytestmark = [
    requires_extras("api"),
    pytest.mark.optional_dependency("api"),
    pytest.mark.optional_api,
]



BASE_URL = "https://restful-booker.herokuapp.com"
TIMEOUT = 90


network_enabled = os.environ.get("RUN_NETWORK_TESTS") == "1"


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.skipif(not network_enabled, reason="Network tests disabled; set RUN_NETWORK_TESTS=1 to enable")
@pytest.mark.asyncio
async def test_execute_step_after_full_suite_run_requests(mcp_client):
    # Create API-focused session
    analyze = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "HTTP API testing with RequestsLibrary", "context": "api"},
    )
    session = analyze.data["session_info"]["session_id"]

    # Ensure RequestsLibrary is first in search order
    await mcp_client.call_tool(
        "set_library_search_order",
        {"libraries": ["RequestsLibrary", "BuiltIn", "Collections", "String"], "session_id": session},
    )

    # 1) Seed with a simple step: create a named session
    create = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Create Session",
            "arguments": ["booker", BASE_URL],
            "session_id": session,
        },
        timeout=TIMEOUT,
    )
    assert create.data.get("success") is True

    # Optional: sanity GET to confirm the session works
    get_list = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Get On Session",
            "arguments": ["booker", "/booking"],
            "session_id": session,
            "assign_to": "resp",
        },
        timeout=TIMEOUT,
    )
    assert get_list.data.get("success") is True

    # 2) Build a suite from recorded steps
    build = await mcp_client.call_tool(
        "build_test_suite",
        {
            "session_id": session,
            "test_name": "Requests Post-Run ExecuteStep Repro",
            "documentation": "Repro for execute_step after suite run",
            "tags": ["regression", "requests"],
        },
        timeout=TIMEOUT,
    )
    assert build.data.get("success") is True

    # 3) Dry run
    dry = await mcp_client.call_tool(
        "run_test_suite_dry",
        {"session_id": session, "validation_level": "standard"},
        timeout=TIMEOUT,
    )
    assert dry.data.get("success") is True

    # 4) Full run
    run = await mcp_client.call_tool(
        "run_test_suite",
        {"session_id": session, "output_level": "standard"},
        timeout=TIMEOUT,
    )
    assert run.data.get("success") is True, f"Suite execution failed: {run.data}"

    # 5) Execute another step on the same session after suite execution
    post = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Log",
            "arguments": ["Post-suite execute_step should succeed"],
            "session_id": session,
        },
        timeout=TIMEOUT,
    )
    assert post.data.get("success") is True, (
        "Expected post-suite execute_step to succeed; got error: %s" % post.data
    )

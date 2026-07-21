"""End-to-end tests for the API init-guidance injection (change: refactor-mcp-instructions §4).

manage_session(action="init") must attach a compact `api_guidance` bundle when
RequestsLibrary is loaded — mirroring the existing desktop_guidance injection —
so RequestsLibrary's non-obvious response-access rules land in the init response
the agent always reads. It must NOT fire for non-API sessions, and must respect
the ROBOTMCP_API_GUIDANCE=off opt-out.
"""
from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import mcp


def _sid(prefix: str = "apiguid") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_requests_init_returns_api_guidance(mcp_client):
    sid = _sid()
    result = await mcp_client.call_tool(
        "manage_session",
        {"action": "init", "session_id": sid,
         "libraries": ["RequestsLibrary", "BuiltIn"]},
    )
    data = result.data
    assert data.get("success") is not False
    assert "api_guidance" in data, "RequestsLibrary init must attach api_guidance"
    g = data["api_guidance"]
    blob = " ".join(g.get("rules", [])) + " " + g.get("more", "")
    assert "On Session" in blob and "${resp.json()}" in blob
    assert 'get_locator_guidance(library="requests")' in g.get("more", "")


@pytest.mark.asyncio
async def test_non_api_init_has_no_api_guidance(mcp_client):
    sid = _sid()
    result = await mcp_client.call_tool(
        "manage_session",
        {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
    )
    assert "api_guidance" not in result.data


@pytest.mark.asyncio
async def test_analyze_scenario_api_flow_returns_api_guidance(mcp_client):
    # The lean default steers agents through analyze_scenario (NOT manage_session init),
    # so the api_guidance bundle must ride the analyze_scenario response for API scenarios.
    result = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "Send a GET request to the /users REST API endpoint and check the status is 200",
         "context": "api"},
    )
    data = result.data
    # RequestsLibrary is only *recommended* at analyze time, so key off the API signal.
    assert data.get("session_type") == "api_testing"
    assert "api_guidance" in data, "API analyze_scenario must attach api_guidance"
    assert "On Session" in " ".join(data["api_guidance"].get("rules", []))


@pytest.mark.asyncio
async def test_analyze_scenario_non_api_has_no_api_guidance(mcp_client):
    result = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "Create a list of three fruits and verify its length", "context": "generic"},
    )
    assert "api_guidance" not in result.data


@pytest.mark.asyncio
async def test_api_guidance_opt_out(mcp_client, monkeypatch):
    monkeypatch.setenv("ROBOTMCP_API_GUIDANCE", "off")
    sid = _sid()
    result = await mcp_client.call_tool(
        "manage_session",
        {"action": "init", "session_id": sid,
         "libraries": ["RequestsLibrary", "BuiltIn"]},
    )
    assert "api_guidance" not in result.data

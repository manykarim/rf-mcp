"""End-to-end tests that integrate FastMCP client with OpenAI sampling."""

from __future__ import annotations

import json
import os
from datetime import timedelta

import pytest
import pytest_asyncio
from fastmcp import Client
from fastmcp.client import FastMCPTransport
from openai import BadRequestError, OpenAI

from robotmcp.server import mcp

_PLACEHOLDER_VALUES = {"", "changeme", "placeholder"}


@pytest.fixture(scope="session")
def openai_model() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key in _PLACEHOLDER_VALUES:
        pytest.skip("OPENAI_API_KEY not configured for OpenAI-driven tests")
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@pytest.fixture(scope="session")
def openai_client(openai_model: str) -> OpenAI:  # noqa: ARG001 - ensure dependency ordering
    return OpenAI()


@pytest_asyncio.fixture
async def openai_mcp_client() -> Client:
    client = Client(
        FastMCPTransport(mcp, raise_exceptions=True),
        timeout=timedelta(seconds=90),
        init_timeout=90,
    )
    async with client:
        yield client


def _extract_text_from_openai_response(response) -> str:
    """Return the assistant text from an OpenAI chat completion response."""
    choices = getattr(response, "choices", None)
    if not choices:  # pragma: no cover - defensive
        return ""
    content = choices[0].message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        block = content[0]
        if isinstance(block, str):
            return block
        return getattr(block, "text", "")
    return ""


@pytest.mark.asyncio
async def test_openai_driven_workflow(openai_client: OpenAI, openai_model: str, openai_mcp_client: Client):
    """Use OpenAI to craft a scenario and exercise the primary MCP tooling workflow."""

    seed_prompt = (
        "Create a concise Robot Framework scenario focused on data processing. "
        "Use only BuiltIn and Collections library keywords. Return strict JSON with keys "
        "'scenario' (string) and 'context' (string)."
    )

    request_kwargs = {
        "model": openai_model,
        "messages": [
            {"role": "system", "content": "You design high-quality Robot Framework automation plans."},
            {"role": "user", "content": seed_prompt},
        ],
    }

    try:
        completion = openai_client.chat.completions.create(
            **request_kwargs,
            max_completion_tokens=300,
        )
    except (BadRequestError, TypeError) as error:
        message = str(error)
        if "max_completion_tokens" not in message and "max_tokens" not in message:
            raise
        completion = openai_client.chat.completions.create(
            **request_kwargs,
            max_tokens=300,
        )

    scenario_payload = _extract_text_from_openai_response(completion).strip()
    assert scenario_payload, "OpenAI response should not be empty"

    try:
        parsed = json.loads(scenario_payload)
    except json.JSONDecodeError:
        parsed = {"scenario": scenario_payload, "context": "process"}

    scenario_text = parsed.get("scenario", scenario_payload)
    scenario_context = parsed.get("context", "process")

    analysis = await openai_mcp_client.call_tool(
        "analyze_scenario", {"scenario": scenario_text, "context": scenario_context}
    )
    assert analysis.data.get("success") is True
    session_info = analysis.data.get("session_info", {})
    session_id = session_info.get("session_id")
    assert session_id, "analyze_scenario should return a session_id"

    recommendations = await openai_mcp_client.call_tool(
        "recommend_libraries",
        {"scenario": scenario_text, "context": scenario_context, "session_id": session_id},
    )
    recommended_libraries = recommendations.data.get("recommended_libraries") or session_info.get(
        "recommended_libraries", []
    )
    if not recommended_libraries:
        recommended_libraries = ["BuiltIn", "Collections"]

    availability = await openai_mcp_client.call_tool(
        "check_library_availability", {"libraries": recommended_libraries}
    )
    assert isinstance(availability.data, dict)

    init_result = await openai_mcp_client.call_tool(
        "manage_session",
        {
            "action": "init",
            "session_id": session_id,
            "variables": {"SCENARIO": "openai"},
            "libraries": recommended_libraries,
        },
    )
    assert init_result.data.get("success") is True

    keywords = await openai_mcp_client.call_tool(
        "find_keywords",
        {"strategy": "session", "session_id": session_id, "query": ""},
    )
    if keywords.data.get("success"):
        session_keywords = keywords.data.get("library_keywords", [])
        assert session_keywords is not None
    else:
        fallback_keywords = await openai_mcp_client.call_tool(
            "find_keywords",
            {"strategy": "catalog", "library_name": "BuiltIn", "query": ""},
        )
        assert fallback_keywords.data.get("success") is True

    context_snapshot = await openai_mcp_client.call_tool(
        "get_session_state",
        {"session_id": session_id, "sections": ["rf_context"]},
    )
    assert context_snapshot.data.get("success") is True

    execution = await openai_mcp_client.call_tool(
        "execute_step",
        {
            "session_id": session_id,
            "keyword": "Create List",
            "arguments": ["alpha", "beta"],
            "assign_to": "items",
        },
    )
    assert execution.data.get("success") is True

    keyword_doc = await openai_mcp_client.call_tool(
        "get_keyword_info",
        {
            "mode": "session",
            "session_id": session_id,
            "keyword_name": "Create List",
        },
    )
    assert keyword_doc.data.get("success") is True

    suite = await openai_mcp_client.call_tool(
        "build_test_suite",
        {"session_id": session_id, "test_name": "OpenAI Scenario", "documentation": scenario_text},
    )
    assert suite.data.get("success") is True
    assert "rf_text" in suite.data

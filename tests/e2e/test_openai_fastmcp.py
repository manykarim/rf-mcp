"""End-to-end tests that integrate FastMCP client with OpenAI sampling."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import timedelta
from typing import List, Tuple

import pytest
import pytest_asyncio
from fastmcp import Client
from fastmcp.client import FastMCPTransport
from openai import BadRequestError, OpenAI, APIError

from robotmcp.server import mcp

logger = logging.getLogger(__name__)

_PLACEHOLDER_VALUES = {"", "changeme", "placeholder"}


@pytest.fixture(scope="session")
def openai_model() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key in _PLACEHOLDER_VALUES:
        pytest.skip("OPENAI_API_KEY not configured for OpenAI-driven tests")
    return os.getenv("OPENAI_MODEL", "gpt-5-mini")


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


def _get_scenario_prompts() -> List[Tuple[str, str]]:
    """Return a list of (system_prompt, user_prompt) tuples to try.

    Multiple prompts are provided to handle cases where certain prompts
    may trigger content filtering or produce empty responses.
    """
    return [
        # Primary prompt - detailed instruction
        (
            "You design high-quality Robot Framework automation plans.",
            "Create a concise Robot Framework scenario focused on data processing. "
            "Use only BuiltIn and Collections library keywords. Return strict JSON with keys "
            "'scenario' (string) and 'context' (string).",
        ),
        # Fallback 1 - simpler instruction
        (
            "You are a helpful assistant that returns JSON.",
            "Generate a simple test scenario that creates a list and logs it. "
            'Return JSON: {"scenario": "your scenario description", "context": "process"}',
        ),
        # Fallback 2 - most minimal
        (
            "Return only valid JSON.",
            '{"scenario": "Create a list with items A and B, then log the list", "context": "process"}',
        ),
    ]


def _try_openai_completion_once(
    client: OpenAI, model: str, system_prompt: str, user_prompt: str
) -> str:
    """Try to get a single completion from OpenAI with the given prompts.

    Args:
        client: OpenAI client
        model: Model name to use
        system_prompt: System message content
        user_prompt: User message content

    Returns:
        Extracted text from response, or empty string if failed
    """
    request_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    completion = None

    # Try max_completion_tokens first (newer models like gpt-5-mini)
    try:
        completion = client.chat.completions.create(
            **request_kwargs,
            max_completion_tokens=500,
        )
    except BadRequestError as error:
        message = str(error).lower()
        # If max_completion_tokens is not supported, try max_tokens
        if "max_completion_tokens" in message or "unsupported parameter" in message:
            logger.info("Model %s doesn't support max_completion_tokens, trying max_tokens", model)
            try:
                completion = client.chat.completions.create(
                    **request_kwargs,
                    max_tokens=500,
                )
            except BadRequestError as fallback_error:
                # If max_tokens also fails, try without any token limit
                fallback_message = str(fallback_error).lower()
                if "max_tokens" in fallback_message or "unsupported parameter" in fallback_message:
                    logger.info("Model %s doesn't support max_tokens either, trying without limit", model)
                    completion = client.chat.completions.create(**request_kwargs)
                else:
                    raise
        else:
            raise
    except TypeError as error:
        # Handle older SDK versions that might not support max_completion_tokens
        message = str(error)
        if "max_completion_tokens" in message:
            completion = client.chat.completions.create(
                **request_kwargs,
                max_tokens=500,
            )
        else:
            raise

    if completion is None:
        return ""

    text = _extract_text_from_openai_response(completion).strip()

    if not text:
        # Log diagnostic info for debugging
        finish_reason = None
        if hasattr(completion, "choices") and completion.choices:
            finish_reason = getattr(completion.choices[0], "finish_reason", None)
        logger.warning(
            "Empty OpenAI response. Model: %s, Finish reason: %s",
            model,
            finish_reason,
        )

    return text


def _try_openai_completion(
    client: OpenAI, model: str, system_prompt: str, user_prompt: str,
    max_retries: int = 3
) -> str:
    """Try to get a completion from OpenAI with retry logic for empty responses.

    Args:
        client: OpenAI client
        model: Model name to use
        system_prompt: System message content
        user_prompt: User message content
        max_retries: Maximum number of retry attempts for empty responses

    Returns:
        Extracted text from response, or empty string if all retries exhausted
    """
    for attempt in range(max_retries):
        try:
            text = _try_openai_completion_once(client, model, system_prompt, user_prompt)
            if text:
                return text

            # Wait before retry with exponential backoff
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    "Empty response from OpenAI, retrying in %ds (attempt %d/%d)",
                    wait_time, attempt + 1, max_retries
                )
                time.sleep(wait_time)
        except APIError as e:
            # Retry on transient API errors
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(
                    "OpenAI API error: %s, retrying in %ds (attempt %d/%d)",
                    str(e), wait_time, attempt + 1, max_retries
                )
                time.sleep(wait_time)
            else:
                raise

    return ""  # All retries exhausted


@pytest.mark.asyncio
async def test_openai_driven_workflow(openai_client: OpenAI, openai_model: str, openai_mcp_client: Client):
    """Use OpenAI to craft a scenario and exercise the primary MCP tooling workflow.

    This test tries multiple prompts to handle cases where certain prompts
    may trigger content filtering or produce empty responses.
    """
    # Try multiple prompts until we get a valid response
    scenario_payload = None
    prompts = _get_scenario_prompts()

    for i, (system_prompt, user_prompt) in enumerate(prompts):
        scenario_payload = _try_openai_completion(
            openai_client, openai_model, system_prompt, user_prompt
        )
        if scenario_payload:
            logger.info("Got valid response on attempt %d/%d", i + 1, len(prompts))
            break
        logger.warning("Empty response on attempt %d/%d, trying next prompt", i + 1, len(prompts))

    if not scenario_payload:
        pytest.fail(
            f"OpenAI returned empty responses for all {len(prompts)} prompt variations. "
            f"Model: {openai_model}. This indicates a persistent API issue or content filtering."
        )

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

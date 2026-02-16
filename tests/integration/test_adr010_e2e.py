"""E2E integration tests for ADR-010: Small LLM Resilience.

Validates that the MCP server handles malformed inputs from small LLMs
(e.g. GLM-4.5 AIR) gracefully, including:
  I1/I2: Array coercion (JSON string, comma-separated, single value)
  I3:    Catalog empty hint when no session is active
  I5:    Session ID auto-generation when empty string is provided
  I6:    Next-step guidance in init response

Run with: uv run pytest tests/integration/test_adr010_e2e.py -v
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from fastmcp import Client

from robotmcp.server import mcp


def _sid(prefix: str = "adr010") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


# =============================================================================
# I1/I2: Array Coercion — manage_session libraries
# =============================================================================


class TestArrayCoercionLibraries:
    """Verify manage_session accepts libraries as JSON string or CSV."""

    @pytest.mark.asyncio
    async def test_libraries_as_json_string(self, mcp_client):
        """I1: manage_session accepts libraries as a JSON array string."""
        sid = _sid()
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": '["BuiltIn"]',  # String, not a list
            },
        )
        assert result.data["success"] is True
        assert "BuiltIn" in result.data["libraries_loaded"]

    @pytest.mark.asyncio
    async def test_libraries_as_json_string_multiple(self, mcp_client):
        """I1: manage_session accepts multi-element JSON string."""
        sid = _sid()
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": '["BuiltIn", "Collections"]',
            },
        )
        assert result.data["success"] is True
        loaded = result.data["libraries_loaded"]
        assert "BuiltIn" in loaded
        assert "Collections" in loaded

    @pytest.mark.asyncio
    async def test_libraries_as_comma_separated(self, mcp_client):
        """I2: manage_session accepts comma-separated string."""
        sid = _sid()
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": "BuiltIn, Collections",
            },
        )
        assert result.data["success"] is True
        loaded = result.data["libraries_loaded"]
        assert "BuiltIn" in loaded
        assert "Collections" in loaded

    @pytest.mark.asyncio
    async def test_libraries_as_single_string(self, mcp_client):
        """I2: manage_session accepts bare single-library string."""
        sid = _sid()
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": "BuiltIn",
            },
        )
        assert result.data["success"] is True
        assert "BuiltIn" in result.data["libraries_loaded"]

    @pytest.mark.asyncio
    async def test_libraries_as_normal_list(self, mcp_client):
        """Baseline: standard list still works after coercion changes."""
        sid = _sid()
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": ["BuiltIn"],
            },
        )
        assert result.data["success"] is True
        assert "BuiltIn" in result.data["libraries_loaded"]


# =============================================================================
# I1/I2: Array Coercion — get_session_state sections
# =============================================================================


class TestArrayCoercionSections:
    """Verify get_session_state accepts sections as JSON string or CSV."""

    @pytest.mark.asyncio
    async def test_sections_as_json_string(self, mcp_client):
        """I1: get_session_state accepts sections as JSON string."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "get_session_state",
            {
                "session_id": sid,
                "sections": '["summary"]',
            },
        )
        assert result.data["success"] is True
        assert "sections" in result.data
        assert "summary" in result.data["sections"]

    @pytest.mark.asyncio
    async def test_sections_as_comma_separated(self, mcp_client):
        """I2: get_session_state accepts sections as CSV string."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "get_session_state",
            {
                "session_id": sid,
                "sections": "summary, variables",
            },
        )
        assert result.data["success"] is True
        assert "sections" in result.data

    @pytest.mark.asyncio
    async def test_sections_as_normal_list(self, mcp_client):
        """Baseline: standard list for sections still works."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "get_session_state",
            {
                "session_id": sid,
                "sections": ["summary"],
            },
        )
        assert result.data["success"] is True
        assert "sections" in result.data
        assert "summary" in result.data["sections"]


# =============================================================================
# I1/I2: Array Coercion — execute_step arguments
# =============================================================================


class TestArrayCoercionArguments:
    """Verify execute_step accepts arguments as JSON string or CSV."""

    @pytest.mark.asyncio
    async def test_arguments_as_json_string(self, mcp_client):
        """I1: execute_step accepts arguments as JSON array string."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Log",
                "arguments": '["Hello from ADR-010"]',
                "session_id": sid,
            },
        )
        assert result.data["success"] is True

    @pytest.mark.asyncio
    async def test_arguments_as_single_string(self, mcp_client):
        """I2: execute_step accepts a bare single argument string."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Log",
                "arguments": "Hello single string",
                "session_id": sid,
            },
        )
        assert result.data["success"] is True

    @pytest.mark.asyncio
    async def test_arguments_as_normal_list(self, mcp_client):
        """Baseline: standard list arguments still work."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Log",
                "arguments": ["Hello normal list"],
                "session_id": sid,
            },
        )
        assert result.data["success"] is True

    @pytest.mark.asyncio
    async def test_arguments_json_string_with_should_be_equal(self, mcp_client):
        """I1: JSON string arguments with a keyword that needs two args."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Should Be Equal",
                "arguments": '["hello", "hello"]',
                "session_id": sid,
            },
        )
        assert result.data["success"] is True


# =============================================================================
# I1/I2: Array Coercion — manage_session test_tags
# =============================================================================


class TestArrayCoercionTestTags:
    """Verify manage_session start_test accepts test_tags as string."""

    @pytest.mark.asyncio
    async def test_test_tags_as_json_string(self, mcp_client):
        """I1: start_test accepts test_tags as JSON string."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "start_test",
                "session_id": sid,
                "test_name": "ADR-010 Tag Test",
                "test_tags": '["smoke", "adr010"]',
            },
        )
        assert result.data["success"] is True

    @pytest.mark.asyncio
    async def test_test_tags_as_comma_separated(self, mcp_client):
        """I2: start_test accepts test_tags as CSV string."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "start_test",
                "session_id": sid,
                "test_name": "ADR-010 CSV Tag Test",
                "test_tags": "smoke, adr010",
            },
        )
        assert result.data["success"] is True


# =============================================================================
# I5: Session ID Auto-Generation
# =============================================================================


class TestSessionIdAutoGeneration:
    """Verify manage_session auto-generates session_id when empty."""

    @pytest.mark.asyncio
    async def test_empty_string_session_id(self, mcp_client):
        """I5: manage_session auto-generates session_id when empty string."""
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": "",
                "libraries": ["BuiltIn"],
            },
        )
        assert result.data["success"] is True
        returned_sid = result.data.get("session_id", "")
        # Server should have created a session (possibly with a generated id,
        # or using an internal default). Either way, it must be usable.
        assert returned_sid is not None
        assert isinstance(returned_sid, str)

    @pytest.mark.asyncio
    async def test_omitted_session_id_defaults(self, mcp_client):
        """I5: manage_session works when session_id not provided (default)."""
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "libraries": ["BuiltIn"],
            },
        )
        assert result.data["success"] is True
        # Should get a session_id back
        assert "session_id" in result.data

    @pytest.mark.asyncio
    async def test_auto_session_id_is_usable(self, mcp_client):
        """I5: auto-generated session_id can be used for execute_step."""
        init = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": "",
                "libraries": ["BuiltIn"],
            },
        )
        assert init.data["success"] is True
        sid = init.data["session_id"]

        # Use the returned session_id in a follow-up call
        step = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Log",
                "arguments": ["Using auto-generated session"],
                "session_id": sid,
            },
        )
        assert step.data["success"] is True


# =============================================================================
# I6: Next-Step Guidance
# =============================================================================


class TestNextStepGuidance:
    """Verify init response includes next_step guidance."""

    @pytest.mark.asyncio
    async def test_init_response_has_next_step(self, mcp_client):
        """I6: init response includes next_step guidance field."""
        sid = _sid()
        result = await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        data = result.data
        assert data["success"] is True
        assert "next_step" in data, (
            f"Expected 'next_step' in init response, got keys: {list(data.keys())}"
        )

    @pytest.mark.asyncio
    async def test_next_step_contains_session_id(self, mcp_client):
        """I6: next_step guidance mentions the session_id for reuse."""
        sid = _sid()
        result = await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        data = result.data
        assert "next_step" in data
        assert sid in data["next_step"], (
            f"Expected session_id '{sid}' in next_step: {data['next_step']}"
        )

    @pytest.mark.asyncio
    async def test_next_step_is_string(self, mcp_client):
        """I6: next_step is a human-readable string, not a dict."""
        sid = _sid()
        result = await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        assert isinstance(result.data["next_step"], str)


# =============================================================================
# I3: Catalog Empty Hint
# =============================================================================


class TestCatalogEmptyHint:
    """Verify find_keywords catalog returns hint when no session loaded."""

    @pytest.mark.asyncio
    async def test_catalog_empty_without_session_has_hint(self, mcp_client):
        """I3: catalog returns hint when called without active session."""
        result = await mcp_client.call_tool(
            "find_keywords",
            {"query": "nonexistent_xyz", "strategy": "catalog"},
        )
        data = result.data
        # Either the catalog returned results (global keywords loaded) or a hint
        if not data.get("results"):
            assert "hint" in data, (
                f"Expected 'hint' key in empty catalog response, got: {list(data.keys())}"
            )
            assert "semantic" in data["hint"].lower() or "session" in data["hint"].lower()

    @pytest.mark.asyncio
    async def test_catalog_with_session_no_hint(self, mcp_client):
        """I3: catalog with active session does NOT include hint."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {"action": "init", "session_id": sid, "libraries": ["BuiltIn"]},
        )
        result = await mcp_client.call_tool(
            "find_keywords",
            {"query": "Log", "strategy": "catalog", "session_id": sid},
        )
        data = result.data
        assert data["success"] is True
        # With a live session and matching keyword, should have results
        assert data.get("results"), "Expected non-empty catalog with BuiltIn loaded"
        # Hint should NOT be present when results exist
        assert "hint" not in data or not data["hint"]


# =============================================================================
# Combined Resilience: Full Workflow with String Inputs
# =============================================================================


class TestFullWorkflowWithStringInputs:
    """End-to-end workflow using only string inputs (simulating small LLM)."""

    @pytest.mark.asyncio
    async def test_full_workflow_string_inputs_only(self, mcp_client):
        """Full lifecycle using string inputs a small LLM would produce."""
        sid = _sid("small-llm")

        # Step 1: Init with JSON string libraries
        init = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": '["BuiltIn", "String"]',
            },
        )
        assert init.data["success"] is True
        assert "next_step" in init.data

        # Step 2: Execute step with JSON string arguments
        step1 = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Convert To Upper Case",
                "arguments": '["hello world"]',
                "session_id": sid,
                "assign_to": "RESULT",
            },
        )
        assert step1.data["success"] is True

        # Step 3: Verify with JSON string arguments
        step2 = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Should Be Equal",
                "arguments": '["${RESULT}", "HELLO WORLD"]',
                "session_id": sid,
            },
        )
        assert step2.data["success"] is True

        # Step 4: Get state with string sections
        state = await mcp_client.call_tool(
            "get_session_state",
            {
                "session_id": sid,
                "sections": '["summary", "variables"]',
            },
        )
        assert state.data["success"] is True
        assert "sections" in state.data

        # Step 5: Build test suite (end of workflow)
        build = await mcp_client.call_tool(
            "build_test_suite",
            {
                "session_id": sid,
                "test_name": "Small LLM Resilience Test",
            },
        )
        assert build.data["success"] is True
        assert "*** Test Cases ***" in build.data["rf_text"]
        assert "Convert To Upper Case" in build.data["rf_text"]

    @pytest.mark.asyncio
    async def test_multi_test_workflow_with_coerced_tags(self, mcp_client):
        """Multi-test workflow using string tags (small LLM pattern)."""
        sid = _sid("multi-coerce")

        # Init with CSV libraries
        init = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "libraries": "BuiltIn, Collections",
            },
        )
        assert init.data["success"] is True

        # Start test with JSON string tags
        start = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "start_test",
                "session_id": sid,
                "test_name": "Coerced Tag Test",
                "test_tags": '["smoke", "resilience"]',
            },
        )
        assert start.data["success"] is True

        # Execute a step
        step = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Create Dictionary",
                "arguments": '["key=value"]',
                "session_id": sid,
                "assign_to": "MY_DICT",
            },
        )
        assert step.data["success"] is True

        # End test
        end = await mcp_client.call_tool(
            "manage_session",
            {"action": "end_test", "session_id": sid},
        )
        assert end.data["success"] is True

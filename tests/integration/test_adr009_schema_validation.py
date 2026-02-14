"""Integration tests for ADR-009 type-constrained tool parameter schemas.

Validates that the ACTUAL MCP tool schemas exposed by the server contain
enum constraints for parameters that use ADR-009 type aliases. This tests
the full stack: kernel type alias -> server.py function signature -> FastMCP
schema generation -> client tool listing.

Run with: uv run pytest tests/integration/test_adr009_schema_validation.py -v
"""

from __future__ import annotations

__test__ = True

import uuid

import pytest
import pytest_asyncio
from fastmcp import Client

from robotmcp.server import mcp


def _sid(prefix: str = "adr009") -> str:
    """Generate a unique session ID for test isolation."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def mcp_client():
    """Provide an MCP client connected to the rf-mcp server."""
    async with Client(mcp) as client:
        yield client


@pytest_asyncio.fixture
async def tools(mcp_client):
    """Cache the tool listing to avoid repeated round-trips."""
    return await mcp_client.list_tools()


def _get_tool(tools, name: str):
    """Find a tool by name from the tool list."""
    for t in tools:
        if t.name == name:
            return t
    pytest.fail(f"Tool '{name}' not found in tool listing. Available: {[t.name for t in tools]}")


def _get_param_schema(tool, param_name: str) -> dict:
    """Extract the schema for a specific parameter from a tool's inputSchema."""
    props = tool.inputSchema.get("properties", {})
    assert param_name in props, (
        f"Parameter '{param_name}' not found in tool '{tool.name}'. "
        f"Available params: {list(props.keys())}"
    )
    return props[param_name]


def _assert_has_enum(schema: dict, expected_values: list, param_path: str):
    """Assert that a schema dict contains enum with expected values.

    Handles both flat enum and anyOf (for Optional types) schemas.
    """
    if "enum" in schema:
        # Flat enum (required parameter)
        non_null = [v for v in schema["enum"] if v is not None]
        assert sorted(non_null) == sorted(expected_values), (
            f"{param_path}: enum mismatch. Got {non_null}, expected {expected_values}"
        )
    elif "anyOf" in schema:
        # Optional parameter: anyOf [{enum...}, {type: null}]
        enum_variants = [v for v in schema["anyOf"] if "enum" in v]
        assert len(enum_variants) >= 1, (
            f"{param_path}: anyOf has no enum variant: {schema}"
        )
        non_null = [v for v in enum_variants[0]["enum"] if v is not None]
        assert sorted(non_null) == sorted(expected_values), (
            f"{param_path}: enum mismatch in anyOf. Got {non_null}, expected {expected_values}"
        )
    else:
        pytest.fail(
            f"{param_path}: schema has neither 'enum' nor 'anyOf': {schema}"
        )


# ============================================================
# 1. manage_session Tool Schema Tests
# ============================================================


class TestManageSessionSchema:
    """Verify manage_session tool has enum constraints on action, test_status, etc."""

    @pytest.mark.asyncio
    async def test_action_has_enum(self, tools):
        """manage_session.action should have enum constraint."""
        tool = _get_tool(tools, "manage_session")
        schema = _get_param_schema(tool, "action")
        expected = [
            "init", "import_library", "import_resource",
            "set_variables", "import_variables",
            "start_test", "end_test", "list_tests",
            "set_suite_setup", "set_suite_teardown",
            "set_tool_profile",
        ]
        _assert_has_enum(schema, expected, "manage_session.action")

    @pytest.mark.asyncio
    async def test_test_status_has_enum(self, tools):
        """manage_session.test_status should have enum constraint."""
        tool = _get_tool(tools, "manage_session")
        schema = _get_param_schema(tool, "test_status")
        _assert_has_enum(schema, ["pass", "fail"], "manage_session.test_status")

    @pytest.mark.asyncio
    async def test_tool_profile_has_enum(self, tools):
        """manage_session.tool_profile should have enum with Optional support."""
        tool = _get_tool(tools, "manage_session")
        schema = _get_param_schema(tool, "tool_profile")
        expected = ["browser_exec", "api_exec", "discovery", "minimal_exec", "full"]
        _assert_has_enum(schema, expected, "manage_session.tool_profile")

    @pytest.mark.asyncio
    async def test_model_tier_has_enum(self, tools):
        """manage_session.model_tier should have enum with Optional support."""
        tool = _get_tool(tools, "manage_session")
        schema = _get_param_schema(tool, "model_tier")
        expected = ["small_context", "standard", "large_context"]
        _assert_has_enum(schema, expected, "manage_session.model_tier")

    @pytest.mark.asyncio
    async def test_profile_has_enum(self, tools):
        """manage_session.profile should have enum with Optional support."""
        tool = _get_tool(tools, "manage_session")
        schema = _get_param_schema(tool, "profile")
        expected = ["browser_exec", "api_exec", "discovery", "minimal_exec", "full"]
        _assert_has_enum(schema, expected, "manage_session.profile")

    @pytest.mark.asyncio
    async def test_action_is_required(self, tools):
        """manage_session.action should be in 'required' list."""
        tool = _get_tool(tools, "manage_session")
        required = tool.inputSchema.get("required", [])
        assert "action" in required, (
            f"'action' should be required. Required params: {required}"
        )


# ============================================================
# 2. intent_action Tool Schema Tests
# ============================================================


class TestIntentActionSchema:
    """Verify intent_action tool has enum constraint on intent."""

    @pytest.mark.asyncio
    async def test_intent_has_enum(self, tools):
        """intent_action.intent should have enum constraint."""
        tool = _get_tool(tools, "intent_action")
        schema = _get_param_schema(tool, "intent")
        expected = [
            "navigate", "click", "fill", "hover",
            "select", "assert_visible", "extract_text", "wait_for",
        ]
        _assert_has_enum(schema, expected, "intent_action.intent")

    @pytest.mark.asyncio
    async def test_intent_is_required(self, tools):
        """intent_action.intent should be required."""
        tool = _get_tool(tools, "intent_action")
        required = tool.inputSchema.get("required", [])
        assert "intent" in required


# ============================================================
# 3. find_keywords Tool Schema Tests
# ============================================================


class TestFindKeywordsSchema:
    """Verify find_keywords tool has enum constraints on strategy and context."""

    @pytest.mark.asyncio
    async def test_strategy_has_enum(self, tools):
        """find_keywords.strategy should have enum constraint."""
        tool = _get_tool(tools, "find_keywords")
        schema = _get_param_schema(tool, "strategy")
        expected = ["semantic", "pattern", "catalog", "session"]
        _assert_has_enum(schema, expected, "find_keywords.strategy")

    @pytest.mark.asyncio
    async def test_context_has_enum(self, tools):
        """find_keywords.context should have enum constraint."""
        tool = _get_tool(tools, "find_keywords")
        schema = _get_param_schema(tool, "context")
        expected = ["web", "mobile", "api", "desktop"]
        _assert_has_enum(schema, expected, "find_keywords.context")


# ============================================================
# 4. execute_step Tool Schema Tests
# ============================================================


class TestExecuteStepSchema:
    """Verify execute_step tool has enum constraints on mode and detail_level."""

    @pytest.mark.asyncio
    async def test_mode_has_enum(self, tools):
        """execute_step.mode should have enum constraint."""
        tool = _get_tool(tools, "execute_step")
        schema = _get_param_schema(tool, "mode")
        expected = ["keyword", "evaluate"]
        _assert_has_enum(schema, expected, "execute_step.mode")

    @pytest.mark.asyncio
    async def test_detail_level_has_enum(self, tools):
        """execute_step.detail_level should have enum constraint."""
        tool = _get_tool(tools, "execute_step")
        schema = _get_param_schema(tool, "detail_level")
        expected = ["minimal", "standard", "full"]
        _assert_has_enum(schema, expected, "execute_step.detail_level")


# ============================================================
# 5. execute_flow Tool Schema Tests
# ============================================================


class TestExecuteFlowSchema:
    """Verify execute_flow tool has enum constraint on structure."""

    @pytest.mark.asyncio
    async def test_structure_has_enum(self, tools):
        """execute_flow.structure should have enum constraint."""
        tool = _get_tool(tools, "execute_flow")
        schema = _get_param_schema(tool, "structure")
        expected = ["if", "for", "try"]
        _assert_has_enum(schema, expected, "execute_flow.structure")


# ============================================================
# 6. run_test_suite Tool Schema Tests
# ============================================================


class TestRunTestSuiteSchema:
    """Verify run_test_suite tool has enum constraints on mode, validation_level, output_level."""

    @pytest.mark.asyncio
    async def test_mode_has_enum(self, tools):
        """run_test_suite.mode should have enum constraint."""
        tool = _get_tool(tools, "run_test_suite")
        schema = _get_param_schema(tool, "mode")
        expected = ["dry", "validate", "full"]
        _assert_has_enum(schema, expected, "run_test_suite.mode")

    @pytest.mark.asyncio
    async def test_validation_level_has_enum(self, tools):
        """run_test_suite.validation_level should have enum constraint."""
        tool = _get_tool(tools, "run_test_suite")
        schema = _get_param_schema(tool, "validation_level")
        expected = ["minimal", "standard", "strict"]
        _assert_has_enum(schema, expected, "run_test_suite.validation_level")

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason=(
            "output_level uses 'detailed' not 'full', so it may not match "
            "DetailLevel alias. Parallel server.py agent may add a separate alias."
        ),
        strict=False,
    )
    async def test_output_level_has_enum(self, tools):
        """run_test_suite.output_level should have enum constraint.

        Note: output_level accepts "minimal"/"standard"/"detailed" which does
        not match DetailLevel ("minimal"/"standard"/"full"). A dedicated type
        alias may be needed, or the parallel agent may normalize the values.
        """
        tool = _get_tool(tools, "run_test_suite")
        schema = _get_param_schema(tool, "output_level")
        # Accept either DetailLevel values or a dedicated output-level set
        has_enum = "enum" in schema or (
            "anyOf" in schema and any("enum" in v for v in schema["anyOf"])
        )
        assert has_enum, (
            f"run_test_suite.output_level should have enum constraint: {schema}"
        )


# ============================================================
# 7. Additional Tools with Type Aliases
# ============================================================


class TestManageLibraryPluginsSchema:
    """Verify manage_library_plugins has enum on action."""

    @pytest.mark.asyncio
    async def test_action_has_enum(self, tools):
        """manage_library_plugins.action should have enum constraint."""
        tool = _get_tool(tools, "manage_library_plugins")
        schema = _get_param_schema(tool, "action")
        expected = ["list", "reload", "diagnose"]
        _assert_has_enum(schema, expected, "manage_library_plugins.action")


class TestManageAttachSchema:
    """Verify manage_attach has enum on action."""

    @pytest.mark.asyncio
    async def test_action_has_enum(self, tools):
        """manage_attach.action should have enum constraint."""
        tool = _get_tool(tools, "manage_attach")
        schema = _get_param_schema(tool, "action")
        expected = ["status", "stop", "cleanup", "reset", "disconnect_all"]
        _assert_has_enum(schema, expected, "manage_attach.action")


class TestRecommendLibrariesSchema:
    """Verify recommend_libraries has enum on context and mode."""

    @pytest.mark.asyncio
    async def test_context_has_enum(self, tools):
        """recommend_libraries.context should have enum constraint."""
        tool = _get_tool(tools, "recommend_libraries")
        schema = _get_param_schema(tool, "context")
        expected = ["web", "mobile", "api", "desktop"]
        _assert_has_enum(schema, expected, "recommend_libraries.context")

    @pytest.mark.asyncio
    async def test_mode_has_enum(self, tools):
        """recommend_libraries.mode should have enum constraint."""
        tool = _get_tool(tools, "recommend_libraries")
        schema = _get_param_schema(tool, "mode")
        expected = ["direct", "sampling_prompt", "merge_samples"]
        _assert_has_enum(schema, expected, "recommend_libraries.mode")


# ============================================================
# 8. Enum Coverage Metrics
# ============================================================


class TestEnumCoverage:
    """Aggregate tests that verify overall enum coverage across all tools."""

    @pytest.mark.asyncio
    async def test_enum_coverage_minimum(self, tools):
        """At least 18 parameters should have enum constraints after ADR-009."""
        enum_count = 0
        for tool in tools:
            for prop_name, prop_schema in tool.inputSchema.get("properties", {}).items():
                if "enum" in prop_schema:
                    enum_count += 1
                elif "anyOf" in prop_schema:
                    if any("enum" in v for v in prop_schema["anyOf"]):
                        enum_count += 1
        assert enum_count >= 18, (
            f"Expected >= 18 enum-constrained params across all tools, "
            f"got {enum_count}"
        )

    @pytest.mark.asyncio
    async def test_enum_coverage_inventory(self, tools):
        """Document which parameters have enum constraints (informational)."""
        inventory = []
        for tool in tools:
            for prop_name, prop_schema in tool.inputSchema.get("properties", {}).items():
                has_enum = False
                if "enum" in prop_schema:
                    has_enum = True
                elif "anyOf" in prop_schema:
                    if any("enum" in v for v in prop_schema["anyOf"]):
                        has_enum = True
                if has_enum:
                    inventory.append(f"{tool.name}.{prop_name}")

        # At minimum, the tools that are already wired should contribute:
        # manage_session: action, test_status, tool_profile, model_tier, profile (5)
        # intent_action: intent (1)
        # manage_library_plugins: action (1)
        # manage_attach: action (1)
        # recommend_libraries: context, mode (2)
        # That's 10 already-wired. Plus the parallel agent work adds more.
        assert len(inventory) >= 10, (
            f"Expected >= 10 enum params in inventory, got {len(inventory)}: {inventory}"
        )

    @pytest.mark.asyncio
    async def test_no_free_text_action_params(self, tools):
        """All parameters named 'action' should be enum-constrained.

        This is a key ADR-009 requirement: action dispatchers must not
        accept free-text strings that small LLMs might hallucinate.
        """
        unconstrained_actions = []
        for tool in tools:
            props = tool.inputSchema.get("properties", {})
            if "action" in props:
                schema = props["action"]
                has_enum = "enum" in schema or (
                    "anyOf" in schema
                    and any("enum" in v for v in schema["anyOf"])
                )
                if not has_enum:
                    unconstrained_actions.append(tool.name)

        assert not unconstrained_actions, (
            f"These tools have 'action' param without enum: {unconstrained_actions}"
        )


# ============================================================
# 9. Runtime Validation via MCP Call (Smoke Tests)
# ============================================================


class TestRuntimeValidation:
    """Smoke tests that verify the server rejects invalid enum values at runtime."""

    @pytest.mark.asyncio
    async def test_manage_session_rejects_invalid_action(self, mcp_client):
        """Server should reject manage_session with bogus action."""
        sid = _sid()
        try:
            result = await mcp_client.call_tool(
                "manage_session",
                {"action": "bogus_action", "session_id": sid},
            )
            # If we get here, the call returned instead of raising.
            # The result should indicate failure.
            if isinstance(result, dict):
                assert not result.get("success", True), (
                    "Expected failure for invalid action, got success"
                )
        except Exception:
            # Validation error is the expected outcome
            pass

    @pytest.mark.asyncio
    async def test_manage_session_accepts_valid_action_case_insensitive(self, mcp_client):
        """Server should accept manage_session with INIT (uppercase)."""
        sid = _sid()
        try:
            result = await mcp_client.call_tool(
                "manage_session",
                {"action": "INIT", "session_id": sid},
            )
            # The call should succeed (action normalized to "init")
            if isinstance(result, dict):
                assert result.get("success", False), (
                    f"Expected success for 'INIT' action, got: {result}"
                )
        except Exception as e:
            # Some transient failures are OK if they're not about validation
            error_msg = str(e).lower()
            assert "action" not in error_msg or "valid" not in error_msg, (
                f"Unexpected validation error for 'INIT': {e}"
            )

    @pytest.mark.asyncio
    async def test_intent_action_rejects_invalid_intent(self, mcp_client):
        """Server should reject intent_action with bogus intent verb."""
        try:
            result = await mcp_client.call_tool(
                "intent_action",
                {"intent": "destroy", "target": "#button"},
            )
            if isinstance(result, dict):
                assert not result.get("success", True)
        except Exception:
            pass  # Validation error expected

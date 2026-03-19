"""Full RF test-creation workflow tests using the Copilot CLI.

Replaces ``test_openai_fastmcp.py`` by driving the entire
analyze -> recommend -> init -> find -> execute -> build pipeline
through the ``copilot`` CLI binary instead of direct FastMCP + OpenAI SDK
calls.

All tests are marked with ``copilot_cli`` and will be skipped
automatically when the Copilot CLI binary is not available or
authentication is missing.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pytest

from tests.e2e.copilot_cli_runner import (
    CopilotRunResult,
    is_copilot_authenticated,
    is_copilot_available,
    run_copilot_cli,
    save_scenario_result,
)
from tests.e2e.models import ExpectedToolCall, Scenario
from tests.e2e.conftest import skip_if_rate_limited

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

skip_no_copilot = pytest.mark.skipif(
    not is_copilot_available(),
    reason="copilot CLI binary not found in PATH",
)

skip_no_auth = pytest.mark.skipif(
    not is_copilot_authenticated(),
    reason="No Copilot authentication (set COPILOT_GITHUB_TOKEN or install copilot CLI)",
)

# ---------------------------------------------------------------------------
# Environment / CI detection
# ---------------------------------------------------------------------------

_IS_CI = os.environ.get("CI") == "true" or os.environ.get("COPILOT_GITHUB_TOKEN")
_MCP_CONFIG = "auto" if _IS_CI else None

_MODEL = os.environ.get("COPILOT_MODEL", "gpt-5-mini")

METRICS_DIR = Path(__file__).parent / "metrics" / "copilot_workflow"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_metrics(result: CopilotRunResult, test_name: str) -> Path:
    """Persist raw run metrics to a JSON file and return the path."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = METRICS_DIR / f"{test_name}_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    return filepath


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.copilot_cli
@skip_no_copilot
@skip_no_auth
class TestCopilotWorkflow:
    """Full RF test-creation workflow driven via the Copilot CLI."""

    def test_full_rf_test_creation(self) -> None:
        """End-to-end: single prompt drives the full 6-tool RF pipeline.

        The prompt asks Copilot to create an XML parsing test going through
        analyze_scenario, recommend_libraries, manage_session (init),
        find_keywords, execute_step, and build_test_suite.

        We verify at least 4 of the 6 expected tools were called (LLM
        non-determinism means we tolerate up to 2 misses).
        """
        prompt = (
            "You are a Robot Framework test automation assistant using RobotMCP tools. "
            "Please perform ALL of the following steps in order:\n"
            "1. Call analyze_scenario with scenario='Parse an XML document to extract book titles "
            "and verify the count' and context='generic'\n"
            "2. Call recommend_libraries with scenario='XML parsing and verification' "
            "and context='generic'\n"
            "3. Call manage_session with action='init', scenario='XML parsing test', "
            "libraries=['BuiltIn', 'XML', 'Collections']\n"
            "4. Call find_keywords with query='Parse XML' to discover XML keywords\n"
            "5. Call execute_step with keyword='Log' and arguments=['Starting XML test']\n"
            "6. Call build_test_suite with test_name='XML Parsing Test' "
            "and documentation='Verify XML parsing of book titles'\n"
            "Return the generated Robot Framework test code."
        )

        expected_tools = [
            "analyze_scenario",
            "recommend_libraries",
            "manage_session",
            "find_keywords",
            "execute_step",
            "build_test_suite",
        ]

        result = run_copilot_cli(
            prompt=prompt,
            model=_MODEL,
            timeout=180,
            mcp_config_path=_MCP_CONFIG,
        )
        skip_if_rate_limited(result)

        # Always save metrics
        metrics_path = _save_metrics(result, "full_rf_test_creation")
        print(f"Metrics saved to: {metrics_path}")

        # Also save as a ScenarioResult for the shared metrics pipeline
        scenario = Scenario(
            id="full_rf_test_creation",
            name="Full RF Test Creation Pipeline",
            description="Single prompt drives analyze, recommend, init, find, execute, build",
            context="generic",
            prompt=prompt,
            expected_tools=[
                ExpectedToolCall(tool_name=t, min_calls=1) for t in expected_tools
            ],
            expected_outcome="Robot Framework test suite for XML parsing",
            min_tool_hit_rate=0.66,
            tags=["workflow", "xml", "full-pipeline"],
        )
        scenario_result = result.to_scenario_result(scenario)
        sr_path = save_scenario_result(scenario_result, METRICS_DIR, prefix="sr_")
        print(f"ScenarioResult saved to: {sr_path}")

        # Diagnostic output
        called = result.get_tool_names()
        mcp_called = result.get_mcp_tool_names()
        hits = [t for t in expected_tools if result.has_tool_call(t)]
        misses = [t for t in expected_tools if not result.has_tool_call(t)]

        print(f"Exit code: {result.exit_code}")
        print(f"All tool calls: {called}")
        print(f"MCP tool calls: {mcp_called}")
        print(f"Expected tools hit: {hits}")
        print(f"Expected tools missed: {misses}")
        print(f"Premium requests: {result.premium_requests}")
        print(f"Session duration: {result.session_duration_ms}ms")
        if result.stderr:
            print(f"Stderr (first 500 chars): {result.stderr[:500]}")

        # Assertions
        assert result.success, f"Copilot CLI exited with code {result.exit_code}"

        hit_count = len(hits)
        assert hit_count >= 4, (
            f"Only {hit_count}/6 expected tools were called (need >= 4). "
            f"Hits: {hits}, misses: {misses}"
        )

    def test_builtin_keyword_execution(self) -> None:
        """Multi-step BuiltIn workflow: init, Create List, Length Should Be, build.

        Verifies that manage_session, execute_step, and build_test_suite
        are called by Copilot when given explicit instructions.
        """
        prompt = (
            "Use RobotMCP to do the following steps:\n"
            "1. Call manage_session with action='init', scenario='BuiltIn list test', "
            "libraries=['BuiltIn']\n"
            "2. Call execute_step with keyword='Create List' and "
            "arguments=['apple', 'banana', 'cherry'] and assign_to='fruits'\n"
            "3. Call execute_step with keyword='Length Should Be' and "
            "arguments=['${fruits}', '3']\n"
            "4. Call build_test_suite to generate the final Robot Framework test suite\n"
            "Return the generated code."
        )

        required_tools = ["manage_session", "execute_step", "build_test_suite"]

        result = run_copilot_cli(
            prompt=prompt,
            model=_MODEL,
            timeout=180,
            mcp_config_path=_MCP_CONFIG,
        )
        skip_if_rate_limited(result)

        metrics_path = _save_metrics(result, "builtin_keyword_execution")
        print(f"Metrics saved to: {metrics_path}")

        # Diagnostic output
        called = result.get_tool_names()
        mcp_called = result.get_mcp_tool_names()
        hits = [t for t in required_tools if result.has_tool_call(t)]
        misses = [t for t in required_tools if not result.has_tool_call(t)]

        print(f"Exit code: {result.exit_code}")
        print(f"All tool calls: {called}")
        print(f"MCP tool calls: {mcp_called}")
        print(f"Expected tools hit: {hits}")
        print(f"Expected tools missed: {misses}")
        if result.stderr:
            print(f"Stderr (first 500 chars): {result.stderr[:500]}")

        # Assertions
        assert result.success, f"Copilot CLI exited with code {result.exit_code}"

        for tool in required_tools:
            assert result.has_tool_call(tool), (
                f"Required tool '{tool}' was not called. "
                f"Tools called: {mcp_called}"
            )

        # execute_step should have been called at least twice
        exec_calls = [
            tc for tc in result.tool_calls if tc.mcp_tool_name == "execute_step"
        ]
        assert len(exec_calls) >= 2, (
            f"Expected at least 2 execute_step calls (Create List + Length Should Be), "
            f"got {len(exec_calls)}"
        )

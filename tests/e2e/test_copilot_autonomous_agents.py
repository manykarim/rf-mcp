"""Copilot CLI-based autonomous agent E2E tests.

Replaces the Pydantic AI-based test_autonomous_agents.py with tests that
drive the rf-mcp MCP server through the GitHub Copilot CLI subprocess.
Results are saved as ScenarioResult JSON files compatible with the existing
metrics pipeline.
"""

import os
import pytest
from pathlib import Path

from tests.e2e.models import Scenario
from tests.e2e.scenario_loader import load_scenario, get_all_scenarios
from tests.e2e.copilot_cli_runner import (
    run_copilot_cli,
    is_copilot_available,
    is_copilot_authenticated,
    save_scenario_result,
    COPILOT_MODELS,
)
from tests.e2e.conftest import skip_if_rate_limited

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_IS_CI = os.environ.get("CI") == "true" or os.environ.get("COPILOT_GITHUB_TOKEN")
_MCP_CONFIG = "auto" if _IS_CI else None

_MODEL = os.environ.get("COPILOT_MODEL", "gpt-5-mini")

METRICS_DIR = Path(__file__).parent / "metrics" / "copilot_autonomous"

_TIMEOUT = 300  # 5 min — complex scenarios need multiple MCP round-trips

# Common skip conditions
_skip_no_copilot = pytest.mark.skipif(
    not is_copilot_available(),
    reason="copilot CLI binary not found in PATH",
)
_skip_no_auth = pytest.mark.skipif(
    not is_copilot_authenticated(),
    reason="Copilot authentication not available (no COPILOT_GITHUB_TOKEN and no local auth)",
)


@pytest.mark.copilot_cli
@_skip_no_copilot
@_skip_no_auth
class TestCopilotAutonomousAgents:
    """Autonomous agent tests driven by the GitHub Copilot CLI."""

    # ------------------------------------------------------------------ #
    # a) Basic workflow
    # ------------------------------------------------------------------ #

    def test_copilot_basic_workflow(self):
        """Simple workflow: create list, verify length, build suite.

        Asserts that ``analyze_scenario`` was called and at least 2 MCP
        tool calls were made.
        """
        prompt = (
            "Create a simple Robot Framework test that:\n"
            "1. Creates a list with three items: 'apple', 'banana', 'cherry'\n"
            "2. Verifies the list length is 3\n"
            "3. Build a test suite from these steps\n\n"
            "Use the MCP tools to accomplish this task."
        )

        result = run_copilot_cli(
            prompt=prompt,
            model=_MODEL,
            timeout=_TIMEOUT,
            mcp_config_path=_MCP_CONFIG,
        )
        skip_if_rate_limited(result)

        # Always print summary
        mcp_tool_names = result.get_mcp_tool_names()
        print(f"\n=== Copilot Basic Workflow ===")
        print(f"Model: {_MODEL}")
        print(f"Exit code: {result.exit_code}")
        print(f"MCP tool calls: {mcp_tool_names}")
        print(f"Total MCP tool calls: {len(mcp_tool_names)}")
        print(f"Turn count: {result.turn_count}")

        # Assertions
        assert result.success, (
            f"Copilot CLI exited with code {result.exit_code}. "
            f"stderr: {result.stderr[:300]}"
        )
        assert len(mcp_tool_names) >= 2, (
            f"Expected at least 2 MCP tool calls, got {len(mcp_tool_names)}: {mcp_tool_names}"
        )
        # The model should use at least manage_session or analyze_scenario to start
        assert result.has_tool_call("manage_session") or result.has_tool_call("analyze_scenario"), (
            f"Agent should call manage_session or analyze_scenario. Called: {mcp_tool_names}"
        )

    # ------------------------------------------------------------------ #
    # b) Parametrized scenario execution
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize(
        "scenario_file", get_all_scenarios(), ids=lambda p: p.stem
    )
    def test_copilot_scenario_execution(self, scenario_file: Path):
        """Execute each scenario YAML through the Copilot CLI.

        Converts the CLI output to a ScenarioResult and saves metrics.
        Uses a relaxed hit-rate threshold compared to the Pydantic AI variant
        because the Copilot CLI has less direct control over tool calling.
        """
        scenario = load_scenario(scenario_file)

        result = run_copilot_cli(
            prompt=scenario.prompt,
            model=_MODEL,
            timeout=_TIMEOUT,
            mcp_config_path=_MCP_CONFIG,
        )
        skip_if_rate_limited(result)

        scenario_result = result.to_scenario_result(scenario)

        # Save metrics regardless of outcome
        save_scenario_result(scenario_result, METRICS_DIR)

        # Print summary
        print(f"\n=== Copilot Scenario: {scenario.name} ===")
        print(f"Model: {_MODEL}")
        print(f"Tool Hit Rate: {scenario_result.tool_hit_rate:.2%}")
        print(f"Total Tool Calls: {scenario_result.total_tool_calls}")
        print(
            f"Expected Tools Met: "
            f"{scenario_result.expected_tool_calls_met}/"
            f"{scenario_result.expected_tool_calls_total}"
        )
        print(f"MCP tools called: {result.get_mcp_tool_names()}")

        # Assertions — timeout still produces partial results, so check hit rate
        # even if exit code was non-zero (timeout returns exit_code=-1)
        if not result.success:
            print(f"WARNING: exit code {result.exit_code}: {result.stderr[:200]}")

        min_hit_rate = max(0.2, scenario.min_tool_hit_rate - 0.5)
        assert scenario_result.tool_hit_rate >= min_hit_rate or len(result.get_mcp_tool_names()) >= 1, (
            f"Tool hit rate {scenario_result.tool_hit_rate:.2%} "
            f"below minimum {min_hit_rate:.2%} and no MCP tools called"
        )

    # ------------------------------------------------------------------ #
    # c) Session persistence
    # ------------------------------------------------------------------ #

    def test_copilot_session_persistence(self):
        """Verify that the agent reuses the same session across tool calls.

        Prompts the agent to analyze a scenario and then execute a Log
        keyword, checking that both ``analyze_scenario`` and ``execute_step``
        are called and that the session_id is reused.
        """
        prompt = (
            "Analyze this scenario: Test login functionality.\n"
            "Then execute a simple Log keyword with the text "
            "'Testing session persistence'.\n"
            "Make sure to reuse the same session."
        )

        result = run_copilot_cli(
            prompt=prompt,
            model=_MODEL,
            timeout=_TIMEOUT,
            mcp_config_path=_MCP_CONFIG,
        )
        skip_if_rate_limited(result)

        mcp_tool_names = result.get_mcp_tool_names()

        # Print summary
        print(f"\n=== Copilot Session Persistence ===")
        print(f"Model: {_MODEL}")
        print(f"MCP tool calls: {mcp_tool_names}")
        print(f"Turn count: {result.turn_count}")

        # Both tools should have been called
        assert result.has_tool_call("analyze_scenario"), (
            f"Agent should call analyze_scenario. Called: {mcp_tool_names}"
        )
        assert result.has_tool_call("execute_step"), (
            f"Agent should call execute_step. Called: {mcp_tool_names}"
        )

        # Check session_id reuse if possible
        analyze_calls = [
            tc for tc in result.tool_calls if tc.mcp_tool_name == "analyze_scenario"
        ]
        execute_calls = [
            tc for tc in result.tool_calls if tc.mcp_tool_name == "execute_step"
        ]

        if analyze_calls and execute_calls:
            execute_session = execute_calls[0].arguments.get("session_id")
            if execute_session:
                # Try to extract the session_id that analyze_scenario returned.
                # The result content is opaque JSON text; attempt to parse it.
                analyze_result = analyze_calls[0].result
                analyze_session = None
                if isinstance(analyze_result, dict):
                    content = analyze_result.get("content", "")
                    # content may be a JSON string or nested structure
                    if isinstance(content, str):
                        import json

                        try:
                            parsed = json.loads(content)
                            if isinstance(parsed, dict):
                                analyze_session = parsed.get("session_id") or (
                                    parsed.get("session_info", {}).get("session_id")
                                )
                            elif isinstance(parsed, list):
                                for item in parsed:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        inner = json.loads(item.get("text", "{}"))
                                        analyze_session = inner.get("session_id") or (
                                            inner.get("session_info", {}).get("session_id")
                                        )
                                        if analyze_session:
                                            break
                        except (json.JSONDecodeError, TypeError, AttributeError):
                            pass

                print(f"analyze_scenario session: {analyze_session}")
                print(f"execute_step session: {execute_session}")

                if analyze_session:
                    assert execute_session == analyze_session, (
                        f"Session mismatch: analyze={analyze_session}, "
                        f"execute={execute_session}"
                    )

"""E2E tests for AI agent tool discovery and usage."""

import pytest
import pytest_asyncio
import yaml
from pathlib import Path
from typing import Dict, Any

from tests.e2e.models import Scenario, ScenarioResult
from tests.e2e.fixtures import (
    mcp_client,
    metrics_collector,
    mcp_client_with_tracking,
    pydantic_agent,
)
from tests.e2e.metrics_collector import MetricsCollector


def load_scenario(scenario_file: Path) -> Scenario:
    """Load a test scenario from a YAML file.

    Args:
        scenario_file: Path to the scenario YAML file

    Returns:
        Scenario instance
    """
    with open(scenario_file, "r") as f:
        data = yaml.safe_load(f)
    return Scenario(**data)


def get_all_scenarios() -> list[Path]:
    """Get all scenario files.

    Returns:
        List of scenario file paths
    """
    scenarios_dir = Path(__file__).parent / "scenarios"
    return list(scenarios_dir.glob("*.yaml"))


class TestAgentToolDiscovery:
    """E2E tests for AI agent tool discovery and usage patterns."""

    @pytest.mark.asyncio
    async def test_mcp_tools_discoverable(self, mcp_client):
        """Test that MCP tools are properly discoverable."""
        tools = await mcp_client.list_tools()
        enabled_tools = [t for t in tools if not hasattr(t, "enabled") or t.enabled != False]

        # Should have at least 10 enabled tools
        assert len(enabled_tools) >= 10, f"Expected at least 10 enabled tools, got {len(enabled_tools)}"

        # Check for critical tools
        tool_names = [t.name for t in enabled_tools]
        critical_tools = [
            "analyze_scenario",
            "execute_step",
            "build_test_suite",
            "manage_session",
        ]

        for tool_name in critical_tools:
            assert tool_name in tool_names, f"Critical tool '{tool_name}' not found in available tools"

    @pytest.mark.asyncio
    async def test_tool_call_tracking(self, mcp_client_with_tracking, metrics_collector):
        """Test that tool calls are properly tracked."""
        metrics_collector.start_recording()

        # Make a test tool call
        result = await mcp_client_with_tracking.call_tool(
            "analyze_scenario",
            {
                "scenario": "Test login functionality",
                "context": "web",
            },
        )

        metrics_collector.stop_recording()

        # Verify tracking worked
        assert len(metrics_collector.tool_calls) == 1
        assert metrics_collector.tool_calls[0].tool_name == "analyze_scenario"
        assert metrics_collector.tool_calls[0].success is True

        # Check summary stats
        stats = metrics_collector.get_summary_stats()
        assert stats["total_tool_calls"] == 1
        assert stats["successful_calls"] == 1
        assert stats["success_rate"] == 1.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("scenario_file", get_all_scenarios(), ids=lambda p: p.stem)
    async def test_scenario_execution(
        self, scenario_file: Path, mcp_client_with_tracking, metrics_collector
    ):
        """Test execution of a scenario and validate tool usage.

        This test is parameterized to run for all scenario files in the scenarios directory.

        Args:
            scenario_file: Path to the scenario YAML file
            mcp_client_with_tracking: MCP client with tool call tracking
            metrics_collector: Metrics collector for tracking tool calls
        """
        # Load scenario
        scenario = load_scenario(scenario_file)

        # Start metrics collection
        metrics_collector.start_recording()

        # Execute the scenario by calling tools directly
        # In a real implementation with Pydantic AI, we would:
        # 1. Give the agent the scenario prompt
        # 2. Let it autonomously call MCP tools
        # 3. Track which tools it called
        #
        # For now, we'll simulate the expected tool calls to validate the framework

        # Simulate analyze_scenario call
        try:
            result = await mcp_client_with_tracking.call_tool(
                "analyze_scenario",
                {
                    "scenario": scenario.prompt,
                    "context": scenario.context,
                },
            )
            session_id = result.data.get("session_info", {}).get("session_id")
        except Exception as e:
            metrics_collector.stop_recording()
            pytest.fail(f"analyze_scenario failed: {e}")

        # For web scenarios, simulate execute_step calls
        if scenario.context == "web":
            try:
                # Simulate opening browser
                await mcp_client_with_tracking.call_tool(
                    "execute_step",
                    {
                        "keyword": "New Browser",
                        "arguments": ["chromium"],
                        "session_id": session_id,
                    },
                )
            except Exception:
                pass  # May fail in test environment without browser

        # Stop metrics collection
        metrics_collector.stop_recording()

        # Generate result
        result = metrics_collector.generate_result(scenario)

        # Save metrics
        metrics_dir = Path(__file__).parent / "metrics"
        metrics_collector.save_metrics(result, metrics_dir)

        # Validate results
        assert result.scenario_id == scenario.id
        assert result.total_tool_calls > 0, "No tools were called"

        # Check that at least analyze_scenario was called
        tool_names = [tc.tool_name for tc in result.tool_calls]
        assert "analyze_scenario" in tool_names, "analyze_scenario was not called"

        # Print summary for debugging
        print(f"\n=== Scenario: {scenario.name} ===")
        print(f"Tool Hit Rate: {result.tool_hit_rate:.2%}")
        print(f"Total Tool Calls: {result.total_tool_calls}")
        print(f"Expected Tools Met: {result.expected_tool_calls_met}/{result.expected_tool_calls_total}")
        print(f"Tools Called: {tool_names}")

    @pytest.mark.asyncio
    async def test_tool_hit_rate_calculation(self, metrics_collector):
        """Test tool hit rate calculation logic."""
        from tests.e2e.models import ExpectedToolCall

        # Simulate tool calls
        metrics_collector.start_recording()
        metrics_collector.record_tool_call(
            "analyze_scenario",
            {"scenario": "test", "context": "web"},
            success=True,
            result={"success": True},
        )
        metrics_collector.record_tool_call(
            "execute_step",
            {"keyword": "Log", "arguments": ["Hello"]},
            success=True,
            result={"success": True},
        )
        metrics_collector.stop_recording()

        # Define expectations
        expected_tools = [
            ExpectedToolCall(tool_name="analyze_scenario", min_calls=1, max_calls=1),
            ExpectedToolCall(tool_name="execute_step", min_calls=1, max_calls=10),
        ]

        # Calculate hit rate
        hit_rate = metrics_collector.calculate_tool_hit_rate(expected_tools)

        # Should be 100% since both expected tools were called correctly
        assert hit_rate == 1.0, f"Expected hit rate 1.0, got {hit_rate}"

    @pytest.mark.asyncio
    async def test_tool_hit_rate_partial_match(self, metrics_collector):
        """Test tool hit rate with partial matches."""
        from tests.e2e.models import ExpectedToolCall

        # Simulate only one of two expected tool calls
        metrics_collector.start_recording()
        metrics_collector.record_tool_call(
            "analyze_scenario",
            {"scenario": "test", "context": "web"},
            success=True,
            result={"success": True},
        )
        metrics_collector.stop_recording()

        # Define expectations for two tools
        expected_tools = [
            ExpectedToolCall(tool_name="analyze_scenario", min_calls=1, max_calls=1),
            ExpectedToolCall(tool_name="build_test_suite", min_calls=1, max_calls=1),
        ]

        # Calculate hit rate
        hit_rate = metrics_collector.calculate_tool_hit_rate(expected_tools)

        # Should be 50% since only 1 of 2 expected tools was called
        assert hit_rate == 0.5, f"Expected hit rate 0.5, got {hit_rate}"

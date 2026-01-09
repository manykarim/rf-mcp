"""Metrics collector for tracking tool calls in E2E tests."""

import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from tests.e2e.models import ToolCallRecord, ScenarioResult, Scenario, ExpectedToolCall


class MetricsCollector:
    """Collects and analyzes metrics from E2E test execution."""

    def __init__(self):
        """Initialize the metrics collector."""
        self.tool_calls: List[ToolCallRecord] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start_recording(self) -> None:
        """Start recording metrics."""
        self.start_time = time.time()
        self.tool_calls = []

    def record_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        success: bool,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Record a tool call.

        Args:
            tool_name: Name of the tool called
            arguments: Arguments passed to the tool
            success: Whether the tool call succeeded
            result: Tool call result (if successful)
            error: Error message (if failed)
        """
        record = ToolCallRecord(
            tool_name=tool_name,
            arguments=arguments,
            success=success,
            result=result,
            error=error,
            timestamp=time.time(),
        )
        self.tool_calls.append(record)

    def stop_recording(self) -> None:
        """Stop recording metrics."""
        self.end_time = time.time()

    def calculate_tool_hit_rate(self, expected_tools: List[ExpectedToolCall]) -> float:
        """Calculate the tool hit rate.

        The tool hit rate is the percentage of expected tools that were called
        with the correct frequency and parameters.

        Args:
            expected_tools: List of expected tool calls

        Returns:
            Tool hit rate as a float between 0.0 and 1.0
        """
        if not expected_tools:
            return 1.0

        met_expectations = 0
        total_expectations = len(expected_tools)

        for expected in expected_tools:
            # Count how many times this tool was called
            actual_calls = [tc for tc in self.tool_calls if tc.tool_name == expected.tool_name]
            call_count = len(actual_calls)

            # Check if call count is within expected range
            min_met = call_count >= expected.min_calls
            max_met = expected.max_calls is None or call_count <= expected.max_calls

            # Check required parameters
            params_met = True
            if expected.required_params:
                for call in actual_calls:
                    for key, expected_value in expected.required_params.items():
                        if key not in call.arguments or call.arguments[key] != expected_value:
                            params_met = False
                            break
                    if not params_met:
                        break

            # If all conditions met, increment counter
            if min_met and max_met and params_met:
                met_expectations += 1

        return met_expectations / total_expectations if total_expectations > 0 else 1.0

    def generate_result(
        self,
        scenario: Scenario,
        agent_output: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ScenarioResult:
        """Generate a scenario result from collected metrics.

        Args:
            scenario: The test scenario that was executed
            agent_output: Optional output from the AI agent
            metadata: Optional additional metadata

        Returns:
            ScenarioResult with all metrics
        """
        execution_time = (
            (self.end_time - self.start_time)
            if self.start_time and self.end_time
            else 0.0
        )

        tool_hit_rate = self.calculate_tool_hit_rate(scenario.expected_tools)

        # Count how many expected tool calls were met
        met_count = 0
        for expected in scenario.expected_tools:
            actual_calls = [tc for tc in self.tool_calls if tc.tool_name == expected.tool_name]
            call_count = len(actual_calls)
            if call_count >= expected.min_calls and (
                expected.max_calls is None or call_count <= expected.max_calls
            ):
                met_count += 1

        # Collect errors
        errors = [tc.error for tc in self.tool_calls if tc.error]

        # Determine overall success
        success = (
            tool_hit_rate >= scenario.min_tool_hit_rate
            and len(errors) == 0
        )

        return ScenarioResult(
            scenario_id=scenario.id,
            success=success,
            tool_calls=self.tool_calls,
            tool_hit_rate=tool_hit_rate,
            total_tool_calls=len(self.tool_calls),
            expected_tool_calls_met=met_count,
            expected_tool_calls_total=len(scenario.expected_tools),
            errors=errors,
            execution_time_seconds=execution_time,
            agent_output=agent_output,
            metadata=metadata or {},
        )

    def save_metrics(self, result: ScenarioResult, output_dir: Path) -> None:
        """Save metrics to a JSON file.

        Args:
            result: ScenarioResult to save
            output_dir: Directory to save metrics to
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.scenario_id}_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of tool calls.

        Returns:
            Dictionary with summary statistics
        """
        total_calls = len(self.tool_calls)
        successful_calls = sum(1 for tc in self.tool_calls if tc.success)
        failed_calls = total_calls - successful_calls

        tool_call_counts: Dict[str, int] = {}
        for tc in self.tool_calls:
            tool_call_counts[tc.tool_name] = tool_call_counts.get(tc.tool_name, 0) + 1

        return {
            "total_tool_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "success_rate": successful_calls / total_calls if total_calls > 0 else 0.0,
            "tool_call_counts": tool_call_counts,
            "unique_tools_called": len(tool_call_counts),
        }

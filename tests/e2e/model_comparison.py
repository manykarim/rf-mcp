"""Model comparison functionality for E2E testing."""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from tests.e2e.models import Scenario, ScenarioResult
from tests.e2e.agent_integration import MCPAgentIntegration
from tests.e2e.metrics_collector import MetricsCollector
from fastmcp import Client, FastMCP


@dataclass
class ModelComparisonResult:
    """Result of comparing multiple models on the same scenario."""

    scenario_id: str
    scenario_name: str
    timestamp: str
    models_tested: List[str]
    results: Dict[str, ScenarioResult]
    comparison_metrics: Dict[str, Any]


class ModelComparator:
    """Compares performance of different AI models on scenarios."""

    # Allowed models based on user specification
    ALLOWED_MODELS = [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "o4-mini",
        "gpt-5-nano",
        "gpt-5",
        "o3-mini",
        "gpt-4o",
        "gpt-5-mini",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-4o-mini",
        "gpt-5.1",
    ]

    def __init__(self, mcp_server_or_client):
        """Initialize model comparator.

        Args:
            mcp_server_or_client: FastMCP server instance or Client instance.
                MCPAgentIntegration requires a FastMCP server for FastMCPToolset.
        """
        self.mcp_server_or_client = mcp_server_or_client

    async def compare_models_on_scenario(
        self,
        scenario: Scenario,
        models: List[str],
        max_concurrent: int = 1,
    ) -> ModelComparisonResult:
        """Compare multiple models on the same scenario.

        Args:
            scenario: Test scenario to run
            models: List of model names to compare
            max_concurrent: Maximum concurrent model executions

        Returns:
            Comparison result with metrics for each model
        """
        # Validate models
        for model in models:
            if model not in self.ALLOWED_MODELS:
                raise ValueError(
                    f"Model '{model}' not in allowed list. "
                    f"Allowed models: {', '.join(self.ALLOWED_MODELS)}"
                )

        results = {}

        # Run scenario with each model
        for model_name in models:
            print(f"\n{'=' * 80}")
            print(f"Testing model: {model_name}")
            print(f"Scenario: {scenario.name}")
            print(f"{'=' * 80}\n")

            # Create new metrics collector for this model
            metrics_collector = MetricsCollector()
            integration = MCPAgentIntegration(self.mcp_server_or_client, metrics_collector)

            # Create agent with this model
            agent = integration.create_agent_with_mcp_tools(
                model_name=model_name, use_test_model=False
            )

            # Run scenario
            metrics_collector.start_recording()

            try:
                output, messages = await integration.run_agent_with_scenario(
                    agent, scenario.prompt
                )
            except Exception as e:
                print(f"Error with model {model_name}: {e}")
                metrics_collector.stop_recording()
                # Create a failed result
                result = ScenarioResult(
                    scenario_id=scenario.id,
                    success=False,
                    tool_calls=[],
                    tool_hit_rate=0.0,
                    total_tool_calls=0,
                    expected_tool_calls_met=0,
                    expected_tool_calls_total=len(scenario.expected_tools),
                    errors=[str(e)],
                    execution_time_seconds=0.0,
                    agent_output=None,
                )
                results[model_name] = result
                continue

            metrics_collector.stop_recording()

            # Generate result
            result = metrics_collector.generate_result(
                scenario,
                agent_output=output,
                metadata={"model": model_name, "messages_count": len(messages)},
            )

            results[model_name] = result

            # Print summary
            print(f"\nModel: {model_name}")
            print(f"Tool Hit Rate: {result.tool_hit_rate:.2%}")
            print(f"Total Tool Calls: {result.total_tool_calls}")
            print(f"Success: {result.success}")
            print(f"Execution Time: {result.execution_time_seconds:.2f}s")

        # Calculate comparison metrics
        comparison_metrics = self._calculate_comparison_metrics(results)

        return ModelComparisonResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            timestamp=datetime.now().isoformat(),
            models_tested=models,
            results=results,
            comparison_metrics=comparison_metrics,
        )

    def _calculate_comparison_metrics(
        self, results: Dict[str, ScenarioResult]
    ) -> Dict[str, Any]:
        """Calculate comparison metrics across models.

        Args:
            results: Results for each model

        Returns:
            Dictionary of comparison metrics
        """
        metrics = {
            "total_models": len(results),
            "successful_models": sum(1 for r in results.values() if r.success),
            "average_tool_hit_rate": sum(r.tool_hit_rate for r in results.values())
            / len(results)
            if results
            else 0.0,
            "average_tool_calls": sum(r.total_tool_calls for r in results.values())
            / len(results)
            if results
            else 0.0,
            "average_execution_time": sum(
                r.execution_time_seconds for r in results.values()
            )
            / len(results)
            if results
            else 0.0,
            "best_model_by_hit_rate": None,
            "best_model_by_speed": None,
            "model_rankings": {},
        }

        if results:
            # Find best model by hit rate
            best_by_hit_rate = max(results.items(), key=lambda x: x[1].tool_hit_rate)
            metrics["best_model_by_hit_rate"] = {
                "model": best_by_hit_rate[0],
                "hit_rate": best_by_hit_rate[1].tool_hit_rate,
            }

            # Find best model by speed
            best_by_speed = min(
                results.items(), key=lambda x: x[1].execution_time_seconds
            )
            metrics["best_model_by_speed"] = {
                "model": best_by_speed[0],
                "time": best_by_speed[1].execution_time_seconds,
            }

            # Create rankings
            for model_name, result in results.items():
                metrics["model_rankings"][model_name] = {
                    "tool_hit_rate": result.tool_hit_rate,
                    "total_tool_calls": result.total_tool_calls,
                    "execution_time": result.execution_time_seconds,
                    "success": result.success,
                    "errors": len(result.errors),
                }

        return metrics

    def save_comparison_result(
        self, result: ModelComparisonResult, output_dir: Path
    ) -> None:
        """Save comparison result to a JSON file.

        Args:
            result: Comparison result to save
            output_dir: Directory to save to
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{result.scenario_id}_{timestamp}.json"
        filepath = output_dir / filename

        # Convert to serializable format
        data = {
            "scenario_id": result.scenario_id,
            "scenario_name": result.scenario_name,
            "timestamp": result.timestamp,
            "models_tested": result.models_tested,
            "results": {
                model: result_obj.model_dump()
                for model, result_obj in result.results.items()
            },
            "comparison_metrics": result.comparison_metrics,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"\nComparison results saved to: {filepath}")

    def generate_comparison_report(
        self, result: ModelComparisonResult
    ) -> str:
        """Generate a human-readable comparison report.

        Args:
            result: Comparison result

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append(f"MODEL COMPARISON REPORT")
        report.append("=" * 80)
        report.append(f"Scenario: {result.scenario_name}")
        report.append(f"Scenario ID: {result.scenario_id}")
        report.append(f"Timestamp: {result.timestamp}")
        report.append(f"Models Tested: {len(result.models_tested)}")
        report.append("")

        # Overall metrics
        report.append("OVERALL METRICS")
        report.append("-" * 80)
        metrics = result.comparison_metrics
        report.append(
            f"Average Tool Hit Rate: {metrics['average_tool_hit_rate']:.2%}"
        )
        report.append(
            f"Average Tool Calls: {metrics['average_tool_calls']:.1f}"
        )
        report.append(
            f"Average Execution Time: {metrics['average_execution_time']:.2f}s"
        )
        report.append(
            f"Successful Models: {metrics['successful_models']}/{metrics['total_models']}"
        )
        report.append("")

        # Best performers
        report.append("BEST PERFORMERS")
        report.append("-" * 80)
        if metrics["best_model_by_hit_rate"]:
            best_hr = metrics["best_model_by_hit_rate"]
            report.append(
                f"Best Tool Hit Rate: {best_hr['model']} ({best_hr['hit_rate']:.2%})"
            )
        if metrics["best_model_by_speed"]:
            best_speed = metrics["best_model_by_speed"]
            report.append(
                f"Fastest Execution: {best_speed['model']} ({best_speed['time']:.2f}s)"
            )
        report.append("")

        # Individual model results
        report.append("INDIVIDUAL MODEL RESULTS")
        report.append("-" * 80)

        # Sort by tool hit rate (descending)
        sorted_models = sorted(
            result.results.items(), key=lambda x: x[1].tool_hit_rate, reverse=True
        )

        for rank, (model_name, model_result) in enumerate(sorted_models, 1):
            report.append(f"\n{rank}. {model_name}")
            report.append(f"   Tool Hit Rate: {model_result.tool_hit_rate:.2%}")
            report.append(f"   Total Tool Calls: {model_result.total_tool_calls}")
            report.append(
                f"   Expected Tools Met: {model_result.expected_tool_calls_met}/{model_result.expected_tool_calls_total}"
            )
            report.append(
                f"   Execution Time: {model_result.execution_time_seconds:.2f}s"
            )
            report.append(f"   Success: {model_result.success}")
            if model_result.errors:
                report.append(f"   Errors: {len(model_result.errors)}")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

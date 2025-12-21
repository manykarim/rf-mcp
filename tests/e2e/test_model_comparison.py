"""Tests for model comparison functionality."""

import os
import pytest
import pytest_asyncio
from pathlib import Path

from tests.e2e.fixtures import mcp_client
from tests.e2e.model_comparison import ModelComparator
from tests.e2e.test_agent_tool_discovery import load_scenario


def should_run_comparison() -> bool:
    """Check if model comparison tests should run."""
    return os.getenv("RUN_MODEL_COMPARISON", "false").lower() in ("true", "1", "yes")


def get_comparison_models() -> list[str]:
    """Get list of models to compare.

    Returns from COMPARISON_MODELS env var (comma-separated) or defaults.
    """
    models_str = os.getenv("COMPARISON_MODELS", "")
    if models_str:
        return [m.strip() for m in models_str.split(",")]

    # Default to comparing a few fast models
    return ["gpt-4o-mini", "gpt-3.5-turbo"]


@pytest.mark.asyncio
@pytest.mark.skipif(
    not should_run_comparison(),
    reason="Requires RUN_MODEL_COMPARISON=true and OPENAI_API_KEY",
)
async def test_model_comparison_simple_scenario(mcp_client):
    """Compare models on a simple test scenario."""
    from tests.e2e.models import Scenario, ExpectedToolCall

    # Create a simple test scenario
    scenario = Scenario(
        id="simple_comparison_test",
        name="Simple Comparison Test",
        description="A simple scenario for model comparison",
        context="generic",
        prompt="""Create a simple Robot Framework test that:
1. Creates a list with items: "test1", "test2", "test3"
2. Verifies the list has 3 items
3. Build a test suite from these steps

Use the MCP tools to accomplish this.""",
        expected_tools=[
            ExpectedToolCall(tool_name="analyze_scenario", min_calls=1, max_calls=1),
            ExpectedToolCall(tool_name="execute_step", min_calls=2, max_calls=10),
            ExpectedToolCall(tool_name="build_test_suite", min_calls=1, max_calls=1),
        ],
        expected_outcome="Successfully creates a test with list operations",
        min_tool_hit_rate=0.7,
        tags=["comparison", "simple"],
    )

    # Get models to compare
    models = get_comparison_models()
    print(f"\nComparing models: {models}")

    # Create comparator
    comparator = ModelComparator(mcp_client)

    # Run comparison
    result = await comparator.compare_models_on_scenario(scenario, models)

    # Save results
    output_dir = Path(__file__).parent / "metrics" / "comparisons"
    comparator.save_comparison_result(result, output_dir)

    # Generate and print report
    report = comparator.generate_comparison_report(result)
    print("\n" + report)

    # Validate results
    assert len(result.results) == len(models), "Should have results for all models"
    assert result.comparison_metrics["total_models"] == len(
        models
    ), "Metrics should reflect all models"

    # At least one model should succeed
    assert (
        result.comparison_metrics["successful_models"] > 0
    ), "At least one model should successfully complete the scenario"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not should_run_comparison(),
    reason="Requires RUN_MODEL_COMPARISON=true and OPENAI_API_KEY",
)
async def test_model_comparison_with_real_scenario(mcp_client):
    """Compare models on a real scenario from scenarios directory."""
    # Load XML testing scenario (simpler than web scenarios)
    scenario_file = Path(__file__).parent / "scenarios" / "xml_testing.yaml"

    if not scenario_file.exists():
        pytest.skip("xml_testing.yaml scenario not found")

    scenario = load_scenario(scenario_file)

    # Get models to compare
    models = get_comparison_models()
    print(f"\nComparing models on '{scenario.name}': {models}")

    # Create comparator
    comparator = ModelComparator(mcp_client)

    # Run comparison
    result = await comparator.compare_models_on_scenario(scenario, models)

    # Save results
    output_dir = Path(__file__).parent / "metrics" / "comparisons"
    comparator.save_comparison_result(result, output_dir)

    # Generate and print report
    report = comparator.generate_comparison_report(result)
    print("\n" + report)

    # Validate results
    assert len(result.results) == len(models)

    # Check that comparison metrics are calculated
    assert result.comparison_metrics["average_tool_hit_rate"] >= 0.0
    assert result.comparison_metrics["best_model_by_hit_rate"] is not None


@pytest.mark.asyncio
async def test_allowed_models_validation(mcp_client):
    """Test that only allowed models can be used."""
    from tests.e2e.models import Scenario, ExpectedToolCall

    scenario = Scenario(
        id="test_validation",
        name="Test Validation",
        description="Test scenario",
        context="generic",
        prompt="Simple test",
        expected_tools=[],
        expected_outcome="Test",
        min_tool_hit_rate=0.5,
        tags=["test"],
    )

    comparator = ModelComparator(mcp_client)

    # Should reject invalid model
    with pytest.raises(ValueError, match="not in allowed list"):
        await comparator.compare_models_on_scenario(
            scenario, ["invalid-model-name"]
        )


@pytest.mark.asyncio
async def test_comparison_report_generation(mcp_client):
    """Test report generation functionality."""
    from tests.e2e.models import Scenario, ExpectedToolCall, ScenarioResult, ToolCallRecord
    from tests.e2e.model_comparison import ModelComparisonResult

    # Create mock results
    scenario = Scenario(
        id="mock_scenario",
        name="Mock Scenario",
        description="Mock",
        context="generic",
        prompt="Test",
        expected_tools=[],
        expected_outcome="Test",
        min_tool_hit_rate=0.5,
        tags=["test"],
    )

    result1 = ScenarioResult(
        scenario_id="mock_scenario",
        success=True,
        tool_calls=[],
        tool_hit_rate=0.8,
        total_tool_calls=5,
        expected_tool_calls_met=4,
        expected_tool_calls_total=5,
        errors=[],
        execution_time_seconds=10.5,
    )

    result2 = ScenarioResult(
        scenario_id="mock_scenario",
        success=True,
        tool_calls=[],
        tool_hit_rate=0.6,
        total_tool_calls=4,
        expected_tool_calls_met=3,
        expected_tool_calls_total=5,
        errors=[],
        execution_time_seconds=8.2,
    )

    comparison_result = ModelComparisonResult(
        scenario_id="mock_scenario",
        scenario_name="Mock Scenario",
        timestamp="2025-01-01T00:00:00",
        models_tested=["gpt-4o-mini", "gpt-3.5-turbo"],
        results={"gpt-4o-mini": result1, "gpt-3.5-turbo": result2},
        comparison_metrics={
            "total_models": 2,
            "successful_models": 2,
            "average_tool_hit_rate": 0.7,
            "average_tool_calls": 4.5,
            "average_execution_time": 9.35,
            "best_model_by_hit_rate": {"model": "gpt-4o-mini", "hit_rate": 0.8},
            "best_model_by_speed": {"model": "gpt-3.5-turbo", "time": 8.2},
            "model_rankings": {},
        },
    )

    comparator = ModelComparator(mcp_client)
    report = comparator.generate_comparison_report(comparison_result)

    # Verify report contains key information
    assert "MODEL COMPARISON REPORT" in report
    assert "Mock Scenario" in report
    assert "gpt-4o-mini" in report
    assert "gpt-3.5-turbo" in report
    assert "80.00%" in report  # Hit rate for gpt-4o-mini
    assert "60.00%" in report  # Hit rate for gpt-3.5-turbo
    assert "Best Tool Hit Rate" in report
    assert "Fastest Execution" in report

    print("\n" + report)

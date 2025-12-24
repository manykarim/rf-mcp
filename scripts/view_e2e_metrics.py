#!/usr/bin/env python3
"""View E2E test metrics from JSON files."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List


def load_latest_metrics(metrics_dir: Path, scenario_id: str = None) -> List[Dict[str, Any]]:
    """Load latest metrics for each scenario or a specific scenario.

    Args:
        metrics_dir: Path to metrics directory
        scenario_id: Optional scenario ID to filter by

    Returns:
        List of metric dictionaries
    """
    metrics_files = list(metrics_dir.glob("*.json"))

    if scenario_id:
        # Filter by scenario ID
        metrics_files = [f for f in metrics_files if scenario_id in f.name]

    # Group by scenario and get latest
    scenarios = {}
    for file in metrics_files:
        # Extract scenario ID from filename (e.g., "todomvc_browser_basic_20251222_085424.json")
        parts = file.stem.rsplit('_', 2)
        if len(parts) >= 3:
            sid = '_'.join(parts[:-2])
            timestamp = f"{parts[-2]}_{parts[-1]}"

            if sid not in scenarios or timestamp > scenarios[sid]['timestamp']:
                scenarios[sid] = {'file': file, 'timestamp': timestamp}

    # Load metrics
    results = []
    for sid, info in sorted(scenarios.items()):
        try:
            with open(info['file'], 'r') as f:
                data = json.load(f)
                data['_metrics_file'] = info['file'].name
                results.append(data)
        except Exception as e:
            print(f"Error loading {info['file']}: {e}", file=sys.stderr)

    return results


def print_scenario_summary(metrics: Dict[str, Any]):
    """Print summary for a single scenario."""
    print(f"\n{'='*80}")
    print(f"Scenario: {metrics.get('scenario_id', 'Unknown')}")
    print(f"File: {metrics.get('_metrics_file', 'Unknown')}")
    print(f"{'='*80}")
    print(f"Success: {'✅' if metrics.get('success') else '❌'}")
    print(f"Tool Hit Rate: {metrics.get('tool_hit_rate', 0):.1%}")
    print(f"Total Tool Calls: {metrics.get('total_tool_calls', 0)}")
    print(f"Expected Tools Met: {metrics.get('expected_tool_calls_met', 0)}/{metrics.get('expected_tool_calls_total', 0)}")

    if metrics.get('errors'):
        print(f"\nErrors: {len(metrics['errors'])}")
        for error in metrics['errors'][:3]:  # Show first 3 errors
            print(f"  - {error}")

    print(f"\nTool Calls:")
    print(f"{'#':<4} {'Tool Name':<30} {'Success':<10} {'Arguments':<30}")
    print("-" * 80)

    for i, call in enumerate(metrics.get('tool_calls', []), 1):
        tool_name = call.get('tool_name', 'Unknown')
        success = '✅' if call.get('success') else '❌'
        args = call.get('arguments', {})

        # Format arguments
        if isinstance(args, dict):
            arg_str = ', '.join(f"{k}" for k in args.keys())
        else:
            arg_str = str(args)

        if len(arg_str) > 28:
            arg_str = arg_str[:25] + "..."

        print(f"{i:<4} {tool_name:<30} {success:<10} {arg_str:<30}")

    print()


def print_detailed_tool_calls(metrics: Dict[str, Any]):
    """Print detailed tool call information."""
    print(f"\n{'='*80}")
    print(f"Detailed Tool Calls for: {metrics.get('scenario_id', 'Unknown')}")
    print(f"{'='*80}\n")

    for i, call in enumerate(metrics.get('tool_calls', []), 1):
        print(f"Call #{i}: {call.get('tool_name', 'Unknown')}")
        print(f"  Success: {'✅' if call.get('success') else '❌'}")

        # Arguments
        args = call.get('arguments', {})
        if args:
            print(f"  Arguments:")
            for key, value in args.items():
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                print(f"    {key}: {value_str}")

        # Result
        result = call.get('result')
        if result:
            print(f"  Result:")
            if isinstance(result, dict):
                for key, value in list(result.items())[:5]:  # Show first 5 keys
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:100] + "..."
                    print(f"    {key}: {value_str}")
                if len(result) > 5:
                    print(f"    ... ({len(result) - 5} more keys)")
            else:
                result_str = str(result)
                if len(result_str) > 200:
                    result_str = result_str[:200] + "..."
                print(f"    {result_str}")

        # Error
        if call.get('error'):
            print(f"  Error: {call['error']}")

        print()


def main():
    """Main entry point."""
    metrics_dir = Path(__file__).parent.parent / "tests" / "e2e" / "metrics"

    if not metrics_dir.exists():
        print(f"Metrics directory not found: {metrics_dir}")
        print("Run E2E tests first to generate metrics:")
        print("  uv run pytest tests/e2e/test_agent_tool_discovery.py::TestAgentToolDiscovery::test_scenario_execution -v")
        return 1

    # Parse command line arguments
    scenario_id = sys.argv[1] if len(sys.argv) > 1 else None
    detailed = "--detailed" in sys.argv or "-d" in sys.argv

    # Load metrics
    metrics_list = load_latest_metrics(metrics_dir, scenario_id)

    if not metrics_list:
        print(f"No metrics found{' for scenario: ' + scenario_id if scenario_id else ''}")
        return 1

    # Print summary or detailed view
    for metrics in metrics_list:
        if detailed:
            print_detailed_tool_calls(metrics)
        else:
            print_scenario_summary(metrics)

    # Print overall summary
    if len(metrics_list) > 1 and not detailed:
        print(f"\n{'='*80}")
        print(f"Overall Summary ({len(metrics_list)} scenarios)")
        print(f"{'='*80}")

        total_calls = sum(m.get('total_tool_calls', 0) for m in metrics_list)
        avg_hit_rate = sum(m.get('tool_hit_rate', 0) for m in metrics_list) / len(metrics_list) if metrics_list else 0
        successful = sum(1 for m in metrics_list if m.get('success'))

        print(f"Successful Scenarios: {successful}/{len(metrics_list)}")
        print(f"Average Tool Hit Rate: {avg_hit_rate:.1%}")
        print(f"Total Tool Calls: {total_calls}")
        print()

        print("Use --detailed or -d flag to see full tool call details")
        print("Use scenario_id as argument to filter by scenario")
        print()
        print("Examples:")
        print("  python scripts/view_e2e_metrics.py")
        print("  python scripts/view_e2e_metrics.py todomvc_browser_basic")
        print("  python scripts/view_e2e_metrics.py --detailed")
        print("  python scripts/view_e2e_metrics.py variable_file_loading -d")

    return 0


if __name__ == "__main__":
    sys.exit(main())

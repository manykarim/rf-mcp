#!/usr/bin/env python3
"""Run realistic E2E tests from test_data/prompts with multiple small LLM models.

Evaluates tool call efficiency across models using real-world test scenarios.

Usage:
    uv run python tests/e2e/run_realistic_e2e.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.e2e.test_intent_action_models import (
    MODELS,
    ModelResult,
    ToolCall,
    _run_model,
    _parse_opencode_json,
    _print_result,
)

# Realistic prompts extracted from test_data/prompts/
REALISTIC_PROMPTS = {
    "web_todo": {
        "name": "Web Testing Todo App",
        "category": "web",
        "prompt": (
            "Use RobotMCP to create a test suite and execute it step wise.\n"
            "It shall:\n"
            "- Open https://todomvc.com/examples/react/dist/\n"
            "- Add a new todo item\n"
            "- Assert the item was added to the list\n"
            "- Add another todo item\n"
            "- Assert the second item was added to the list\n"
            "- Mark the first item as completed\n"
            "- Assert the first item is marked as completed\n"
            "Execute step by step and build final test suite afterwards"
        ),
        "expected_min_tools": 5,
        "expected_tool_pattern": [
            "robotmcp_analyze_scenario",
            "robotmcp_manage_session",
            "robotmcp_execute_step",
            "robotmcp_build_test_suite",
        ],
    },
    "api_restful_booker": {
        "name": "REST API Testing",
        "category": "api",
        "prompt": (
            "Use RobotMCP and create a test suite for a REST API and execute it step wise.\n"
            "The API is at https://restful-booker.herokuapp.com/apidoc/index.html\n"
            "It shall cover: GET, POST, DELETE with assertions.\n"
            "Use a suite setup for authorization as admin.\n"
            "Execute step by step and build and save final test suite."
        ),
        "expected_min_tools": 4,
        "expected_tool_pattern": [
            "robotmcp_analyze_scenario",
            "robotmcp_manage_session",
            "robotmcp_execute_step",
            "robotmcp_build_test_suite",
        ],
    },
    "web_demoshop": {
        "name": "Web Testing Demoshop",
        "category": "web",
        "prompt": (
            "Use RobotMCP to create a test suite and execute it step wise.\n"
            "It shall:\n"
            "- Open https://demoshop.makrocode.de/\n"
            "- Add item to cart\n"
            "- Assert item was added to cart\n"
            "- Checkout\n"
            "- Assert checkout was successful\n"
            "Execute step by step and build final test suite afterwards"
        ),
        "expected_min_tools": 5,
        "expected_tool_pattern": [
            "robotmcp_analyze_scenario",
            "robotmcp_manage_session",
            "robotmcp_execute_step",
            "robotmcp_build_test_suite",
        ],
    },
}


@dataclass
class ToolEfficiency:
    """Tool call efficiency metrics for a single model+prompt run."""

    model: str
    prompt_id: str
    prompt_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    unique_tools: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)
    expected_tools_hit: int = 0
    expected_tools_total: int = 0
    used_intent_action: bool = False
    used_execute_step: bool = False
    used_analyze_scenario: bool = False
    used_build_test_suite: bool = False
    used_find_keywords: bool = False
    used_get_session_state: bool = False
    duration_seconds: float = 0.0
    total_tokens: int = 0
    cost: float = 0.0
    error: str | None = None

    @property
    def success_rate(self) -> float:
        return self.successful_calls / self.total_calls if self.total_calls else 0.0

    @property
    def tool_hit_rate(self) -> float:
        return self.expected_tools_hit / self.expected_tools_total if self.expected_tools_total else 0.0

    @property
    def calls_per_second(self) -> float:
        return self.total_calls / self.duration_seconds if self.duration_seconds else 0.0


def _analyze_efficiency(result: ModelResult, prompt_id: str, prompt_info: dict) -> ToolEfficiency:
    """Analyze tool call efficiency from a model result."""
    eff = ToolEfficiency(
        model=result.model,
        prompt_id=prompt_id,
        prompt_name=prompt_info["name"],
        duration_seconds=result.duration_seconds,
        total_tokens=result.total_tokens,
        cost=result.cost,
        error=result.error,
    )

    if result.error:
        return eff

    eff.total_calls = len(result.tool_calls)
    eff.successful_calls = sum(1 for tc in result.tool_calls if tc.success)
    eff.failed_calls = eff.total_calls - eff.successful_calls

    # Count tools
    for tc in result.tool_calls:
        eff.tool_counts[tc.name] = eff.tool_counts.get(tc.name, 0) + 1
    eff.unique_tools = len(eff.tool_counts)

    # Check expected tools
    expected = prompt_info.get("expected_tool_pattern", [])
    eff.expected_tools_total = len(expected)
    eff.expected_tools_hit = sum(1 for t in expected if t in eff.tool_counts)

    # Check specific tool usage
    eff.used_intent_action = "robotmcp_intent_action" in eff.tool_counts
    eff.used_execute_step = "robotmcp_execute_step" in eff.tool_counts
    eff.used_analyze_scenario = "robotmcp_analyze_scenario" in eff.tool_counts
    eff.used_build_test_suite = "robotmcp_build_test_suite" in eff.tool_counts
    eff.used_find_keywords = "robotmcp_find_keywords" in eff.tool_counts
    eff.used_get_session_state = "robotmcp_get_session_state" in eff.tool_counts

    return eff


def _print_efficiency_report(all_efficiencies: list[ToolEfficiency]) -> None:
    """Print comprehensive efficiency comparison report."""
    print("\n\n" + "=" * 120)
    print("TOOL CALL EFFICIENCY REPORT — REALISTIC E2E PROMPTS")
    print("=" * 120)

    # Group by prompt
    prompts = {}
    for eff in all_efficiencies:
        if eff.prompt_id not in prompts:
            prompts[eff.prompt_id] = []
        prompts[eff.prompt_id].append(eff)

    for prompt_id, effs in prompts.items():
        prompt_name = effs[0].prompt_name
        print(f"\n{'─' * 120}")
        print(f"Prompt: {prompt_name} ({prompt_id})")
        print(f"{'─' * 120}")
        print(f"{'Model':<22} {'Calls':<6} {'OK':<5} {'Fail':<5} {'Rate':<6} "
              f"{'Unique':<7} {'Hit':<5} {'a_s':<4} {'m_s':<4} {'e_s':<4} {'b_s':<4} "
              f"{'i_a':<4} {'f_k':<4} {'s_s':<4} {'Time':<8} {'Tokens':<8} {'Cost':<8}")
        print(f"{'─' * 120}")

        for eff in sorted(effs, key=lambda e: e.total_calls, reverse=True):
            model_short = eff.model.split("/")[-1][:20]
            if eff.error:
                print(f"  {model_short:<20} {'ERROR':<6} {'-':<5} {'-':<5} {'-':<6} "
                      f"{'-':<7} {'-':<5} {'-':<4} {'-':<4} {'-':<4} {'-':<4} "
                      f"{'-':<4} {'-':<4} {'-':<4} {'-':<8} {'-':<8} {'-':<8}"
                      f"  {eff.error[:40]}")
                continue
            print(f"  {model_short:<20} {eff.total_calls:<6} {eff.successful_calls:<5} "
                  f"{eff.failed_calls:<5} {eff.success_rate:<6.0%} "
                  f"{eff.unique_tools:<7} {eff.expected_tools_hit}/{eff.expected_tools_total:<3} "
                  f"{'Y' if eff.used_analyze_scenario else '-':<4} "
                  f"{'Y' if eff.used_execute_step or eff.used_intent_action else '-':<4} "  # m_s = manage/exec
                  f"{'Y' if eff.used_execute_step else '-':<4} "
                  f"{'Y' if eff.used_build_test_suite else '-':<4} "
                  f"{'Y' if eff.used_intent_action else '-':<4} "
                  f"{'Y' if eff.used_find_keywords else '-':<4} "
                  f"{'Y' if eff.used_get_session_state else '-':<4} "
                  f"{eff.duration_seconds:<8.1f} {eff.total_tokens:<8} ${eff.cost:<7.4f}")

    # Overall summary
    working = [e for e in all_efficiencies if not e.error]
    print(f"\n{'=' * 120}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 120}")
    if working:
        avg_calls = sum(e.total_calls for e in working) / len(working)
        avg_success = sum(e.success_rate for e in working) / len(working)
        avg_hit = sum(e.tool_hit_rate for e in working) / len(working)
        avg_time = sum(e.duration_seconds for e in working) / len(working)
        total_cost = sum(e.cost for e in working)

        print(f"  Working runs:          {len(working)}/{len(all_efficiencies)}")
        print(f"  Avg tool calls/run:    {avg_calls:.1f}")
        print(f"  Avg success rate:      {avg_success:.0%}")
        print(f"  Avg tool hit rate:     {avg_hit:.0%}")
        print(f"  Avg execution time:    {avg_time:.1f}s")
        print(f"  Total cost:            ${total_cost:.4f}")

        # Tool usage summary
        tool_counts: dict[str, int] = {}
        for e in working:
            for tool, count in e.tool_counts.items():
                tool_counts[tool] = tool_counts.get(tool, 0) + count
        print(f"\n  Tool usage across all runs:")
        for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
            print(f"    {tool}: {count}")

        # intent_action vs execute_step
        ia_count = sum(1 for e in working if e.used_intent_action)
        es_count = sum(1 for e in working if e.used_execute_step)
        print(f"\n  Models using intent_action: {ia_count}/{len(working)}")
        print(f"  Models using execute_step:  {es_count}/{len(working)}")

    print(f"{'=' * 120}")

    # Legend
    print("\nLegend: a_s=analyze_scenario m_s=manage_session/exec e_s=execute_step "
          "b_s=build_test_suite i_a=intent_action f_k=find_keywords s_s=get_session_state")


def main():
    print("=" * 100)
    print("REALISTIC E2E TESTS — MULTI-MODEL TOOL CALL EFFICIENCY")
    print("=" * 100)
    print(f"Models: {len(MODELS)}")
    print(f"Prompts: {len(REALISTIC_PROMPTS)}")
    print(f"Total runs: {len(MODELS) * len(REALISTIC_PROMPTS)}")
    print()

    all_efficiencies: list[ToolEfficiency] = []
    all_results: list[dict[str, Any]] = []
    start_time = time.time()

    for prompt_id, prompt_info in REALISTIC_PROMPTS.items():
        print(f"\n{'#' * 100}")
        print(f"# Prompt: {prompt_info['name']} ({prompt_id})")
        print(f"{'#' * 100}")

        for i, model in enumerate(MODELS, 1):
            model_short = model.split("/")[-1]
            print(f"\n  [{i}/{len(MODELS)}] {model_short} — {prompt_info['name']}")
            print(f"  {'─' * 60}")

            result = _run_model(model, prompt_info["prompt"], timeout_seconds=300)
            _print_result(result)

            eff = _analyze_efficiency(result, prompt_id, prompt_info)
            all_efficiencies.append(eff)

            all_results.append({
                "model": model,
                "prompt_id": prompt_id,
                "prompt_name": prompt_info["name"],
                "total_calls": eff.total_calls,
                "successful_calls": eff.successful_calls,
                "failed_calls": eff.failed_calls,
                "success_rate": eff.success_rate,
                "unique_tools": eff.unique_tools,
                "tool_counts": eff.tool_counts,
                "expected_tools_hit": eff.expected_tools_hit,
                "expected_tools_total": eff.expected_tools_total,
                "tool_hit_rate": eff.tool_hit_rate,
                "used_intent_action": eff.used_intent_action,
                "used_execute_step": eff.used_execute_step,
                "used_analyze_scenario": eff.used_analyze_scenario,
                "used_build_test_suite": eff.used_build_test_suite,
                "duration_seconds": eff.duration_seconds,
                "total_tokens": eff.total_tokens,
                "cost": eff.cost,
                "error": eff.error,
            })

    total_time = time.time() - start_time

    # Print efficiency report
    _print_efficiency_report(all_efficiencies)

    # Save results
    metrics_dir = Path(__file__).parent / "metrics" / "realistic_e2e"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = metrics_dir / f"realistic_e2e_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "total_time_seconds": total_time,
            "models": [m.split("/")[-1] for m in MODELS],
            "prompts": list(REALISTIC_PROMPTS.keys()),
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"Total time: {total_time:.0f}s")
    print(f"Total cost: ${sum(e.cost for e in all_efficiencies):.4f}")


if __name__ == "__main__":
    main()

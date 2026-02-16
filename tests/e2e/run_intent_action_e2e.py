#!/usr/bin/env python3
"""Run intent_action E2E tests across multiple small LLM models.

Usage:
    uv run python tests/e2e/run_intent_action_e2e.py

This script runs opencode with each model and analyzes intent_action tool usage.
Results are printed as a comparison table and saved to tests/e2e/metrics/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.e2e.test_intent_action_models import (
    MODELS,
    INTENT_ACTION_PROMPT,
    SHORT_INTENT_PROMPT,
    ModelResult,
    _run_model,
    _print_result,
)

# Models that work better with shorter prompts (need more time for tool execution)
_SLOW_MODELS = {"openrouter/meta-llama/llama-4-scout"}


def main():
    print("=" * 90)
    print("INTENT_ACTION E2E TEST — MULTI-MODEL COMPARISON")
    print("=" * 90)
    print(f"Models: {len(MODELS)}")
    for m in MODELS:
        print(f"  - {m}")
    print()

    all_results: list[ModelResult] = []
    start_time = time.time()

    for i, model in enumerate(MODELS, 1):
        model_short = model.split("/")[-1]
        print(f"\n{'#' * 70}")
        print(f"# [{i}/{len(MODELS)}] Testing: {model_short}")
        print(f"{'#' * 70}")

        # Slow models get shorter prompt first (more reliable) with longer timeout
        if model in _SLOW_MODELS:
            result = _run_model(model, SHORT_INTENT_PROMPT, timeout_seconds=300)
        else:
            result = _run_model(model, INTENT_ACTION_PROMPT, timeout_seconds=300)
        _print_result(result)

        # Retry with shorter prompt if no intent_action usage
        if not result.used_intent_action and not result.error:
            print(f"\n  >>> Retrying {model_short} with shorter prompt...")
            result2 = _run_model(model, SHORT_INTENT_PROMPT, timeout_seconds=300)
            _print_result(result2)
            if result2.used_intent_action:
                result = result2
                result.model = model  # keep original model name

        all_results.append(result)

    total_time = time.time() - start_time

    # Print comparison table
    print("\n\n" + "=" * 100)
    print("FINAL COMPARISON TABLE")
    print("=" * 100)
    print(f"{'Model':<25} {'Status':<8} {'intent_action':<14} "
          f"{'Intents':<30} {'session_id':<12} {'Time':<8} {'Cost':<10}")
    print("-" * 100)

    for r in all_results:
        model_short = r.model.split("/")[-1]
        if r.error:
            status = "ERROR"
            ia_count = "-"
            intents = r.error[:28]
            sid = "-"
            time_s = "-"
            cost = "-"
        else:
            status = "PASS" if r.used_intent_action else "FAIL"
            ia_count = str(len(r.intent_action_calls))
            intents = ",".join(r.intent_action_intents)[:28] or "-"
            sid = "YES" if r.passed_session_id else "NO"
            time_s = f"{r.duration_seconds:.1f}s"
            cost = f"${r.cost:.4f}"
        print(f"  {model_short:<23} {status:<8} {ia_count:<14} "
              f"{intents:<30} {sid:<12} {time_s:<8} {cost:<10}")

    # Summary stats
    working = [r for r in all_results if not r.error]
    using_ia = [r for r in working if r.used_intent_action]
    print(f"\n  Total models:           {len(all_results)}")
    print(f"  Working models:         {len(working)}")
    print(f"  Using intent_action:    {len(using_ia)}")
    if working:
        print(f"  Discovery rate:         {len(using_ia)/len(working):.0%}")
    print(f"  Total time:             {total_time:.1f}s")
    print(f"  Total cost:             ${sum(r.cost for r in all_results):.4f}")
    print("=" * 100)

    # Save results to JSON
    metrics_dir = Path(__file__).parent / "metrics" / "intent_action"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = metrics_dir / f"intent_action_comparison_{timestamp}.json"

    results_data = {
        "timestamp": timestamp,
        "total_time_seconds": total_time,
        "models": [
            {
                "model": r.model,
                "used_intent_action": r.used_intent_action,
                "intent_action_count": len(r.intent_action_calls),
                "intents_used": r.intent_action_intents,
                "passed_session_id": r.passed_session_id,
                "used_analyze_scenario": r.used_analyze_scenario,
                "used_manage_session": r.used_manage_session,
                "total_tool_calls": len(r.tool_calls),
                "all_tools": [tc.name for tc in r.tool_calls],
                "duration_seconds": r.duration_seconds,
                "total_tokens": r.total_tokens,
                "cost": r.cost,
                "error": r.error,
            }
            for r in all_results
        ],
        "summary": {
            "total_models": len(all_results),
            "working_models": len(working),
            "using_intent_action": len(using_ia),
            "discovery_rate": len(using_ia) / len(working) if working else 0,
            "total_cost": sum(r.cost for r in all_results),
        },
    }

    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Exit code: 0 if at least half used intent_action, 1 otherwise
    if working and len(using_ia) / len(working) >= 0.5:
        print("\nPASS: >= 50% of models used intent_action")
        sys.exit(0)
    else:
        print("\nFAIL: < 50% of models used intent_action")
        sys.exit(1)


if __name__ == "__main__":
    main()

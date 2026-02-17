#!/usr/bin/env python3
"""Compare execute_batch vs execute_step for demoshop scenario.

Runs the same demoshop test scenario with two different prompts:
1. Original: "Execute step by step" → model uses repeated execute_step calls
2. Batch: "Use execute_batch" → model uses a single execute_batch call

Compares total MCP tool calls, round-trips, tokens, cost, and timing.

Usage:
    OPENCODE_MODELS="openrouter/qwen/qwen3-coder" uv run python tests/e2e/run_batch_vs_step_e2e.py
    uv run python tests/e2e/run_batch_vs_step_e2e.py  # runs all default models
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.e2e.test_intent_action_models import (
    ModelResult,
    ToolCall,
    _run_model,
    _parse_opencode_json,
)

# ── Models ────────────────────────────────────────────────────────────

_DEFAULT_MODELS = [
    "openrouter/qwen/qwen3-coder",
    "openrouter/z-ai/glm-4.7-flash",
]

_env_models = os.getenv("OPENCODE_MODELS", "").strip()
MODELS = [m.strip() for m in _env_models.split(",") if m.strip()] if _env_models else _DEFAULT_MODELS

# ── Prompts ───────────────────────────────────────────────────────────

STEP_PROMPT = """\
Use RobotMCP to test the demoshop at https://demoshop.makrocode.de/ using SeleniumLibrary.
Execute each action step by step using execute_step.

Steps:
1. Call analyze_scenario for "Selenium test for the demoshop"
2. Call manage_session to init with SeleniumLibrary and BuiltIn
3. Use execute_step to: Open Browser  https://demoshop.makrocode.de/  headlesschrome
4. Use execute_step to: Click Element  css=.product-card:first-child button
5. Use execute_step to: Get Text  css=.cart-count  (use assign_to="cart_count")
6. Use execute_step to: Should Not Be Equal  ${cart_count}  0
7. Use execute_step to: Click Element  css=a[href='/cart']
8. Use execute_step to: Get Text  css=.cart-total  (use assign_to="cart_total")
9. Call build_test_suite to generate the .robot file

Execute exactly these steps. Do NOT skip any. Use execute_step for steps 3-8.
"""

BATCH_PROMPT = """\
Use RobotMCP to test the demoshop at https://demoshop.makrocode.de/ using SeleniumLibrary.
Use execute_batch to run all keyword steps in a SINGLE call.

Steps:
1. Call analyze_scenario for "Selenium test for the demoshop"
2. Call manage_session to init with SeleniumLibrary and BuiltIn
3. Call execute_batch with session_id and these steps (in one call):
   [
     {"keyword": "Open Browser", "args": ["https://demoshop.makrocode.de/", "headlesschrome"], "label": "open demoshop"},
     {"keyword": "Click Element", "args": ["css=.product-card:first-child button"], "label": "add to cart"},
     {"keyword": "Get Text", "args": ["css=.cart-count"], "assign_to": "cart_count", "label": "get cart count"},
     {"keyword": "Should Not Be Equal", "args": ["${cart_count}", "0"], "label": "verify cart"},
     {"keyword": "Click Element", "args": ["css=a[href='/cart']"], "label": "go to cart"},
     {"keyword": "Get Text", "args": ["css=.cart-total"], "assign_to": "cart_total", "label": "get total"}
   ]
   Use on_failure="recover" and timeout_ms=60000.
4. Call build_test_suite to generate the .robot file

IMPORTANT: Use execute_batch for step 3. Do NOT use execute_step.
execute_batch accepts: session_id, steps (list of step dicts), on_failure, timeout_ms.
Each step dict has: keyword, args, label (optional), assign_to (optional), timeout (optional).
"""


# ── Analysis ──────────────────────────────────────────────────────────

@dataclass
class RunAnalysis:
    """Analysis of a single prompt run."""
    model: str
    prompt_type: str  # "step" or "batch"
    total_tool_calls: int = 0
    execute_step_calls: int = 0
    execute_batch_calls: int = 0
    analyze_scenario_calls: int = 0
    manage_session_calls: int = 0
    build_test_suite_calls: int = 0
    other_calls: int = 0
    tool_counts: dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    total_tokens: int = 0
    cost: float = 0.0
    error: str | None = None
    # Batch-specific
    batch_steps_count: int = 0  # Number of steps inside execute_batch

    @property
    def mcp_round_trips(self) -> int:
        """Effective MCP round-trips (each tool call = 1 round-trip)."""
        return self.total_tool_calls


def _analyze_run(result: ModelResult, prompt_type: str) -> RunAnalysis:
    """Extract analysis metrics from a model run."""
    a = RunAnalysis(
        model=result.model,
        prompt_type=prompt_type,
        duration_seconds=result.duration_seconds,
        total_tokens=result.total_tokens,
        cost=result.cost,
        error=result.error,
    )
    if result.error:
        return a

    a.total_tool_calls = len(result.tool_calls)

    for tc in result.tool_calls:
        a.tool_counts[tc.name] = a.tool_counts.get(tc.name, 0) + 1

        if tc.name == "robotmcp_execute_step":
            a.execute_step_calls += 1
        elif tc.name == "robotmcp_execute_batch":
            a.execute_batch_calls += 1
            # Count steps inside the batch
            steps = tc.args.get("steps", [])
            if isinstance(steps, list):
                a.batch_steps_count += len(steps)
        elif tc.name == "robotmcp_analyze_scenario":
            a.analyze_scenario_calls += 1
        elif tc.name == "robotmcp_manage_session":
            a.manage_session_calls += 1
        elif tc.name == "robotmcp_build_test_suite":
            a.build_test_suite_calls += 1
        else:
            a.other_calls += 1

    return a


# ── Reporting ─────────────────────────────────────────────────────────

def _print_comparison(step_runs: list[RunAnalysis], batch_runs: list[RunAnalysis]) -> None:
    """Print side-by-side comparison of step vs batch runs."""
    print("\n\n" + "=" * 120)
    print("EXECUTE_BATCH vs EXECUTE_STEP — E2E COMPARISON")
    print("=" * 120)

    # Per-model comparison
    print(f"\n{'Model':<22} {'Mode':<7} {'Calls':<7} {'e_step':<8} {'e_batch':<9} "
          f"{'batch_steps':<13} {'a_scen':<8} {'m_sess':<8} {'b_suite':<9} "
          f"{'Time':<8} {'Tokens':<9} {'Cost':<8}")
    print("-" * 120)

    models_seen = set()
    all_runs = [(r, "step") for r in step_runs] + [(r, "batch") for r in batch_runs]
    # Sort by model name, then prompt type
    all_runs.sort(key=lambda x: (x[0].model, x[1]))

    for run, ptype in all_runs:
        model_short = run.model.split("/")[-1][:20]
        if run.error:
            print(f"  {model_short:<20} {ptype:<7} {'ERROR':<7} {'-':<8} {'-':<9} "
                  f"{'-':<13} {'-':<8} {'-':<8} {'-':<9} "
                  f"{'-':<8} {'-':<9} {'-':<8}  {run.error[:40]}")
            continue

        batch_steps_str = str(run.batch_steps_count) if run.execute_batch_calls > 0 else "-"
        print(f"  {model_short:<20} {ptype:<7} {run.total_tool_calls:<7} "
              f"{run.execute_step_calls:<8} {run.execute_batch_calls:<9} "
              f"{batch_steps_str:<13} "
              f"{run.analyze_scenario_calls:<8} {run.manage_session_calls:<8} "
              f"{run.build_test_suite_calls:<9} "
              f"{run.duration_seconds:<8.1f} {run.total_tokens:<9} "
              f"${run.cost:<7.4f}")

    # Summary
    working_step = [r for r in step_runs if not r.error]
    working_batch = [r for r in batch_runs if not r.error]

    print(f"\n{'=' * 120}")
    print("SUMMARY")
    print(f"{'=' * 120}")

    if working_step:
        avg_calls_step = sum(r.total_tool_calls for r in working_step) / len(working_step)
        avg_es = sum(r.execute_step_calls for r in working_step) / len(working_step)
        avg_tokens_step = sum(r.total_tokens for r in working_step) / len(working_step)
        avg_cost_step = sum(r.cost for r in working_step) / len(working_step)
        avg_time_step = sum(r.duration_seconds for r in working_step) / len(working_step)
        print(f"  execute_step mode:")
        print(f"    Avg total tool calls:   {avg_calls_step:.1f}")
        print(f"    Avg execute_step calls: {avg_es:.1f}")
        print(f"    Avg tokens:             {avg_tokens_step:.0f}")
        print(f"    Avg cost:               ${avg_cost_step:.4f}")
        print(f"    Avg time:               {avg_time_step:.1f}s")

    if working_batch:
        avg_calls_batch = sum(r.total_tool_calls for r in working_batch) / len(working_batch)
        avg_eb = sum(r.execute_batch_calls for r in working_batch) / len(working_batch)
        avg_bs = sum(r.batch_steps_count for r in working_batch) / len(working_batch)
        avg_tokens_batch = sum(r.total_tokens for r in working_batch) / len(working_batch)
        avg_cost_batch = sum(r.cost for r in working_batch) / len(working_batch)
        avg_time_batch = sum(r.duration_seconds for r in working_batch) / len(working_batch)
        print(f"\n  execute_batch mode:")
        print(f"    Avg total tool calls:   {avg_calls_batch:.1f}")
        print(f"    Avg execute_batch calls:{avg_eb:.1f}")
        print(f"    Avg steps in batch:     {avg_bs:.1f}")
        print(f"    Avg tokens:             {avg_tokens_batch:.0f}")
        print(f"    Avg cost:               ${avg_cost_batch:.4f}")
        print(f"    Avg time:               {avg_time_batch:.1f}s")

    if working_step and working_batch:
        reduction = 1 - (avg_calls_batch / avg_calls_step) if avg_calls_step > 0 else 0
        token_diff = avg_tokens_batch - avg_tokens_step
        cost_diff = avg_cost_batch - avg_cost_step
        time_diff = avg_time_batch - avg_time_step
        print(f"\n  Comparison (batch vs step):")
        print(f"    Tool call reduction:    {reduction:.0%} ({avg_calls_step:.1f} → {avg_calls_batch:.1f})")
        print(f"    Token difference:       {token_diff:+.0f} ({token_diff/avg_tokens_step:+.0%})" if avg_tokens_step else "")
        print(f"    Cost difference:        ${cost_diff:+.4f}")
        print(f"    Time difference:        {time_diff:+.1f}s")

        # Batch discovery rate
        using_batch = sum(1 for r in working_batch if r.execute_batch_calls > 0)
        print(f"    Batch discovery rate:   {using_batch}/{len(working_batch)} "
              f"({using_batch/len(working_batch):.0%})")

    print(f"{'=' * 120}")
    print("\nLegend: e_step=execute_step calls, e_batch=execute_batch calls, "
          "batch_steps=steps inside batch, a_scen=analyze_scenario, "
          "m_sess=manage_session, b_suite=build_test_suite")


def main():
    print("=" * 100)
    print("EXECUTE_BATCH vs EXECUTE_STEP — DEMOSHOP E2E COMPARISON")
    print("=" * 100)
    print(f"Models: {len(MODELS)}")
    for m in MODELS:
        print(f"  - {m}")
    print(f"Runs: {len(MODELS) * 2} (step + batch for each model)")
    print()

    step_runs: list[RunAnalysis] = []
    batch_runs: list[RunAnalysis] = []
    all_raw: list[dict[str, Any]] = []
    start_time = time.time()

    for i, model in enumerate(MODELS, 1):
        model_short = model.split("/")[-1]

        # ── Run 1: execute_step prompt ──
        print(f"\n{'#' * 100}")
        print(f"# [{i}/{len(MODELS)}] {model_short} — STEP MODE")
        print(f"{'#' * 100}")
        result_step = _run_model(model, STEP_PROMPT, timeout_seconds=300)
        analysis_step = _analyze_run(result_step, "step")
        step_runs.append(analysis_step)

        if result_step.error:
            print(f"  ERROR: {result_step.error}")
        else:
            print(f"  Duration: {result_step.duration_seconds:.1f}s")
            print(f"  Total tool calls: {len(result_step.tool_calls)}")
            tc = {}
            for t in result_step.tool_calls:
                tc[t.name] = tc.get(t.name, 0) + 1
            print(f"  Tools: {tc}")
            print(f"  Tokens: {result_step.total_tokens}")
            print(f"  Cost: ${result_step.cost:.4f}")

        all_raw.append({
            "model": model, "prompt_type": "step",
            "total_calls": analysis_step.total_tool_calls,
            "execute_step_calls": analysis_step.execute_step_calls,
            "execute_batch_calls": analysis_step.execute_batch_calls,
            "batch_steps_count": analysis_step.batch_steps_count,
            "tool_counts": analysis_step.tool_counts,
            "duration_seconds": analysis_step.duration_seconds,
            "total_tokens": analysis_step.total_tokens,
            "cost": analysis_step.cost,
            "error": analysis_step.error,
        })

        # ── Run 2: execute_batch prompt ──
        print(f"\n{'#' * 100}")
        print(f"# [{i}/{len(MODELS)}] {model_short} — BATCH MODE")
        print(f"{'#' * 100}")
        result_batch = _run_model(model, BATCH_PROMPT, timeout_seconds=300)
        analysis_batch = _analyze_run(result_batch, "batch")
        batch_runs.append(analysis_batch)

        if result_batch.error:
            print(f"  ERROR: {result_batch.error}")
        else:
            print(f"  Duration: {result_batch.duration_seconds:.1f}s")
            print(f"  Total tool calls: {len(result_batch.tool_calls)}")
            tc = {}
            for t in result_batch.tool_calls:
                tc[t.name] = tc.get(t.name, 0) + 1
            print(f"  Tools: {tc}")
            if analysis_batch.execute_batch_calls > 0:
                print(f"  Steps inside batch: {analysis_batch.batch_steps_count}")
            print(f"  Tokens: {result_batch.total_tokens}")
            print(f"  Cost: ${result_batch.cost:.4f}")

        all_raw.append({
            "model": model, "prompt_type": "batch",
            "total_calls": analysis_batch.total_tool_calls,
            "execute_step_calls": analysis_batch.execute_step_calls,
            "execute_batch_calls": analysis_batch.execute_batch_calls,
            "batch_steps_count": analysis_batch.batch_steps_count,
            "tool_counts": analysis_batch.tool_counts,
            "duration_seconds": analysis_batch.duration_seconds,
            "total_tokens": analysis_batch.total_tokens,
            "cost": analysis_batch.cost,
            "error": analysis_batch.error,
        })

    total_time = time.time() - start_time

    # Print comparison
    _print_comparison(step_runs, batch_runs)

    # Save results
    metrics_dir = Path(__file__).parent / "metrics" / "batch_vs_step"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = metrics_dir / f"batch_vs_step_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "total_time_seconds": total_time,
            "models": [m.split("/")[-1] for m in MODELS],
            "step_prompt": STEP_PROMPT,
            "batch_prompt": BATCH_PROMPT,
            "results": all_raw,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"Total time: {total_time:.0f}s")
    print(f"Total cost: ${sum(r.cost for r in step_runs + batch_runs):.4f}")


if __name__ == "__main__":
    main()

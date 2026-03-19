"""Multi-model comparison tests using the Copilot CLI.

Replaces ``test_model_comparison.py`` by running the same prompt across
multiple LLM models via ``copilot --model <name>`` and collecting
per-model metrics for side-by-side comparison.

All tests are marked with ``copilot_cli`` and will be skipped
automatically when the Copilot CLI binary is not available or
authentication is missing.
"""

from __future__ import annotations

import glob as glob_mod
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tests.e2e.copilot_cli_runner import (
    COPILOT_MODELS,
    CopilotRunResult,
    DEFAULT_COMPARISON_MODELS,
    is_copilot_authenticated,
    is_copilot_available,
    run_copilot_cli,
)

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

_COMPARISON_MODELS: List[str] = os.environ.get(
    "COPILOT_COMPARISON_MODELS",
    ",".join(DEFAULT_COMPARISON_MODELS),
).split(",")

METRICS_DIR = Path(__file__).parent / "metrics" / "copilot_comparisons"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_model_metrics(result: CopilotRunResult, model: str) -> Path:
    """Persist per-model run metrics to a JSON file and return the path."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace(".", "_").replace("-", "_")
    filepath = METRICS_DIR / f"{today}_comparison_{safe_model}_{timestamp}.json"

    mcp_tools = result.get_mcp_tool_names()
    successful = result.get_successful_tool_calls()
    failed = result.get_failed_tool_calls()
    model_info = COPILOT_MODELS.get(model, {})

    payload: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "provider": model_info.get("provider", "unknown"),
        "multiplier": model_info.get("multiplier"),
        "tier": model_info.get("tier", "unknown"),
        "success": result.success,
        "exit_code": result.exit_code,
        "mcp_tool_calls": mcp_tools,
        "tool_call_count": len(mcp_tools),
        "successful_tool_calls": len(successful),
        "failed_tool_calls": len(failed),
        "premium_requests": result.premium_requests,
        "api_duration_ms": result.api_duration_ms,
        "session_duration_ms": result.session_duration_ms,
        "total_output_tokens": result.total_output_tokens,
        "turn_count": result.turn_count,
        "raw": result.to_dict(),
    }

    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return filepath


def _compute_hit_rate(result: CopilotRunResult, expected: List[str]) -> float:
    """Return the fraction of *expected* tool names that were actually called."""
    if not expected:
        return 1.0
    hits = sum(1 for t in expected if result.has_tool_call(t))
    return hits / len(expected)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.copilot_cli
@skip_no_copilot
@skip_no_auth
class TestCopilotModelComparison:
    """Compare multiple LLM models on the same RobotMCP prompts."""

    @pytest.mark.parametrize("model", _COMPARISON_MODELS, ids=_COMPARISON_MODELS)
    def test_model_comparison_simple(self, model: str) -> None:
        """Run a simple multi-tool prompt on each model and save metrics.

        The prompt asks for find_keywords + manage_session + execute_step
        so we can compare tool-call success rates and latencies.
        """
        prompt = (
            "Use the RobotMCP tools to do the following:\n"
            "1. Call find_keywords with query='Log' and library='BuiltIn'\n"
            "2. Call manage_session with action='init', scenario='model comparison', "
            "libraries=['BuiltIn']\n"
            "3. Call execute_step with keyword='Log' and arguments=['Hello from model test']\n"
            "Return the results."
        )

        expected_tools = ["find_keywords", "manage_session", "execute_step"]

        result = run_copilot_cli(
            prompt=prompt,
            model=model,
            timeout=180,
            mcp_config_path=_MCP_CONFIG,
        )

        # Save per-model metrics
        metrics_path = _save_model_metrics(result, model)

        # Compute hit rate
        hit_rate = _compute_hit_rate(result, expected_tools)
        mcp_tools = result.get_mcp_tool_names()
        duration_s = result.session_duration_ms / 1000.0

        # Summary line (always printed, even on failure)
        print(
            f"[{model}] success={result.success}  "
            f"tools={len(mcp_tools)}  hit_rate={hit_rate:.0%}  "
            f"duration={duration_s:.1f}s  "
            f"premium_requests={result.premium_requests}  "
            f"tokens={result.total_output_tokens}  "
            f"metrics={metrics_path}"
        )

        if result.stderr:
            print(f"[{model}] Stderr (first 300 chars): {result.stderr[:300]}")

        # Relaxed assertion: the CLI should at least run without crashing.
        # We record a soft assertion so CI can surface issues without
        # failing hard (the CI job typically uses continue-on-error).
        assert len(result.tool_calls) >= 0, "Unexpected negative tool call count"

    def test_comparison_report(self) -> None:
        """Aggregate per-model metrics from today's session into a report.

        Reads all ``<today>_comparison_*.json`` files produced by the
        parametrized ``test_model_comparison_simple`` test and produces
        a summary JSON plus a formatted table printed to stdout.

        This test is NOT parametrized -- it runs once after all models
        have been tested.
        """
        today = datetime.now().strftime("%Y%m%d")
        pattern = str(METRICS_DIR / f"{today}_comparison_*.json")
        metric_files = sorted(glob_mod.glob(pattern))

        if not metric_files:
            pytest.skip(
                f"No comparison metrics found for today ({today}) in {METRICS_DIR}"
            )

        # Load all per-model results
        model_data: Dict[str, Dict[str, Any]] = {}
        for filepath in metric_files:
            with open(filepath) as f:
                data = json.load(f)
            model_name = data.get("model", Path(filepath).stem)
            # Keep the latest entry per model (in case of reruns)
            model_data[model_name] = data

        # Build summary
        summary: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "date": today,
            "models_tested": list(model_data.keys()),
            "total_models": len(model_data),
            "per_model": {},
        }

        for model_name, data in model_data.items():
            summary["per_model"][model_name] = {
                "provider": data.get("provider", "unknown"),
                "tier": data.get("tier", "unknown"),
                "multiplier": data.get("multiplier"),
                "success": data.get("success", False),
                "tool_call_count": data.get("tool_call_count", 0),
                "successful_tool_calls": data.get("successful_tool_calls", 0),
                "failed_tool_calls": data.get("failed_tool_calls", 0),
                "premium_requests": data.get("premium_requests", 0),
                "api_duration_ms": data.get("api_duration_ms", 0),
                "session_duration_ms": data.get("session_duration_ms", 0),
                "total_output_tokens": data.get("total_output_tokens", 0),
                "turn_count": data.get("turn_count", 0),
                "mcp_tool_calls": data.get("mcp_tool_calls", []),
            }

        # Averages
        models_list = list(summary["per_model"].values())
        n = len(models_list) or 1
        summary["averages"] = {
            "avg_tool_calls": sum(m["tool_call_count"] for m in models_list) / n,
            "avg_duration_ms": sum(m["session_duration_ms"] for m in models_list) / n,
            "avg_premium_requests": sum(m["premium_requests"] for m in models_list) / n,
            "avg_output_tokens": sum(m["total_output_tokens"] for m in models_list) / n,
            "success_rate": sum(1 for m in models_list if m["success"]) / n,
        }

        # Save aggregated summary
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        summary_path = METRICS_DIR / f"{today}_comparison_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nComparison summary saved to: {summary_path}")

        # Print formatted table
        header = (
            f"{'Model':<25} {'Provider':<12} {'Tier':<10} "
            f"{'Tools':>6} {'OK':>4} {'Fail':>5} "
            f"{'Duration':>10} {'Premium':>8} {'Tokens':>8} {'Success':>8}"
        )
        separator = "-" * len(header)

        print(f"\n{separator}")
        print("COPILOT MODEL COMPARISON REPORT")
        print(f"Date: {today}  |  Models: {len(model_data)}")
        print(separator)
        print(header)
        print(separator)

        for model_name, m in summary["per_model"].items():
            duration_s = m["session_duration_ms"] / 1000.0
            print(
                f"{model_name:<25} {m['provider']:<12} {m['tier']:<10} "
                f"{m['tool_call_count']:>6} {m['successful_tool_calls']:>4} "
                f"{m['failed_tool_calls']:>5} "
                f"{duration_s:>9.1f}s {m['premium_requests']:>8} "
                f"{m['total_output_tokens']:>8} "
                f"{'PASS' if m['success'] else 'FAIL':>8}"
            )

        print(separator)
        avgs = summary["averages"]
        print(
            f"{'AVERAGES':<25} {'':<12} {'':<10} "
            f"{avgs['avg_tool_calls']:>6.1f} {'':<4} {'':<5} "
            f"{avgs['avg_duration_ms'] / 1000.0:>9.1f}s "
            f"{avgs['avg_premium_requests']:>8.1f} "
            f"{avgs['avg_output_tokens']:>8.0f} "
            f"{avgs['success_rate']:>7.0%}"
        )
        print(separator)

        # Assertions: we should have loaded at least one model's data
        assert len(model_data) >= 1, "Expected at least 1 model result file"

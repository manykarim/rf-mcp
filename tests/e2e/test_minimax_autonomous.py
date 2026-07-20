"""MiniMax autonomous-agent e2e with a per-model instruction-quality gate.

The PURPOSE of this test is to measure whether rf-mcp's tool descriptions + MCP
instructions let an agent DISCOVER and DRIVE the tools correctly — and to FAIL when that
degrades. Wrong / failed / absent tool calls are the signal (see tests/e2e/quality_gate.py
for the philosophy). The harness (agent_integration.py) uses a NEUTRAL system prompt and
injects the server's real MCP instructions, so the agent depends on rf-mcp's actual
guidance, not a test cheat-sheet.

Isolation of an rf-mcp regression from model non-determinism is done by pinning the model
and comparing N-run aggregates against THAT model's captured baseline
(tests/e2e/baselines/instruction_quality_baselines.json), with a tolerance band and
median aggregation.

Gated on MINIMAX_API_KEY. Modes:
- Gate (default): run each model N times, evaluate vs baseline, assert no regression.
- Capture: E2E_CAPTURE_BASELINE=1 writes/updates baselines instead of asserting.

    # capture the M3 reference baseline (5 runs)
    MINIMAX_API_KEY=... E2E_CAPTURE_BASELINE=1 E2E_RUNS=5 MINIMAX_MODELS=MiniMax-M3 \
        uv run pytest tests/e2e/test_minimax_autonomous.py -q
    # run the gate
    MINIMAX_API_KEY=... MINIMAX_MODELS=MiniMax-M3 \
        uv run pytest tests/e2e/test_minimax_autonomous.py -v -s
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.e2e.agent_integration import MCPAgentIntegration
from tests.e2e.fixtures import mcp_server, metrics_collector
import subprocess
from datetime import datetime, timezone

from tests.e2e.minimax_support import (
    REFERENCE_MODEL,
    provider_for,
    real_llm_available,
)
from tests.e2e.models import ExpectedToolCall, Scenario
from tests.e2e.quality_gate import (
    capture_entry,
    compute_run_metrics,
    evaluate,
    is_excluded_model,
    load_baselines,
    write_gate_report,
)

pytestmark = pytest.mark.skipif(
    not real_llm_available(),
    reason="Requires a model API key (MINIMAX_API_KEY or OPENROUTER_API_KEY)",
)

_METRICS_DIR = Path(__file__).parent / "metrics" / "minimax"
_BASELINE_PATH = Path(__file__).parent / "baselines" / "instruction_quality_baselines.json"

# Deterministic, non-browser scenario (no display server needed in CI). Completing it
# requires discovering the analyze -> execute -> build pipeline from rf-mcp's own
# descriptions/instructions.
_BASIC_SCENARIO = Scenario(
    id="minimax_basic_list",
    name="Basic list workflow",
    description="Create a 3-item list and verify its length via MCP tools.",
    context="generic",
    prompt=(
        "Create and run a Robot Framework test that builds a list containing the items "
        "'apple', 'banana', and 'cherry', then verifies the list has length 3. "
        "Use the available MCP tools to start a session, build the steps, and produce a "
        "final test suite."
    ),
    expected_tools=[
        ExpectedToolCall(tool_name="analyze_scenario", min_calls=1),
        ExpectedToolCall(tool_name="execute_step", min_calls=1),
        ExpectedToolCall(tool_name="build_test_suite", min_calls=1),
    ],
    expected_outcome="A suite that builds a 3-item list and asserts its length.",
    min_tool_hit_rate=0.5,
    tags=["minimax", "smoke", "generic"],
)

# Tools whose success marks the task pipeline complete (start + artifact).
_COMPLETION_TOOLS = ["analyze_scenario", "build_test_suite"]


def _runs_per_model() -> int:
    try:
        return max(1, int(os.getenv("E2E_RUNS", "3")))
    except ValueError:
        return 3


def _models_to_test() -> list[str]:
    """Models to run: E2E_MODELS / MINIMAX_MODELS env, else the baseline roster.

    The roster (reference + hard_gate + inform, minus excluded) lives in the baseline
    file so promoting/demoting a model is a reviewed config change.
    """
    for env in ("E2E_MODELS", "MINIMAX_MODELS"):
        raw = os.getenv(env, "").strip()
        if raw:
            return [m.strip() for m in raw.split(",") if m.strip()]
    b = load_baselines(_BASELINE_PATH)
    roster = (
        b.get("reference_models", []) + b.get("hard_gate_models", []) + b.get("inform_models", [])
    )
    excluded = set(b.get("excluded_models", []))
    return [m for m in roster if m not in excluded] or [REFERENCE_MODEL]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _capture_mode() -> bool:
    return os.getenv("E2E_CAPTURE_BASELINE", "").lower() in ("1", "true", "yes")


async def _registered_tool_names(mcp_server) -> list[str]:
    tools = await mcp_server.list_tools()
    return [getattr(t, "name", None) for t in tools if getattr(t, "name", None)]


async def _one_run(model_name: str, mcp_server, metrics_collector):
    """Execute the scenario once; return (RunMetrics, ScenarioResult, registered_tools)."""
    request_limit = 24
    integration = MCPAgentIntegration(mcp_server, metrics_collector)
    agent = integration.create_agent_with_mcp_tools(model_name=model_name, use_test_model=False)

    metrics_collector.start_recording()
    agent_output, run_error = "", None
    try:
        agent_output, _ = await integration.run_agent_with_scenario(
            agent, _BASIC_SCENARIO.prompt, request_limit=request_limit
        )
    except Exception as e:  # UsageLimitExceeded / transport / model framing
        run_error = f"{type(e).__name__}: {e}"
    finally:
        metrics_collector.stop_recording()

    result = metrics_collector.generate_result(_BASIC_SCENARIO, agent_output=agent_output)
    registered = await _registered_tool_names(mcp_server)
    run_metrics = compute_run_metrics(
        result,
        registered,
        expected_tool_names=[e.tool_name for e in _BASIC_SCENARIO.expected_tools],
        completion_tool_names=_COMPLETION_TOOLS,
        run_error=run_error,
    )
    return run_metrics, result, registered


def _write_baseline_entry(model_name: str, entry: dict) -> None:
    """Merge a captured entry for (model, scenario) into the baseline file, marking the
    scenario validated (a capture only succeeds when the reference completed cleanly)."""
    data = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
    scen = data.setdefault("scenarios", {}).setdefault(_BASIC_SCENARIO.id, {})
    scen.setdefault("_validated", True)
    scen[model_name] = entry
    _BASELINE_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", _models_to_test())
async def test_instruction_quality_gate(model_name, mcp_server, metrics_collector):
    """Run a model N times and enforce the instruction-quality gate (or capture)."""
    baselines = load_baselines(_BASELINE_PATH)
    if is_excluded_model(model_name, baselines):
        pytest.skip(f"{model_name} is excluded (broken tool-calling)")

    n = _runs_per_model()
    runs, last_result, registered = [], None, []
    for _ in range(n):
        run_metrics, last_result, registered = await _one_run(
            model_name, mcp_server, metrics_collector
        )
        runs.append(run_metrics)

    if last_result is not None:
        _METRICS_DIR.mkdir(parents=True, exist_ok=True)
        metrics_collector.save_metrics(last_result, _METRICS_DIR)

    if _capture_mode():
        provenance = {
            "captured_at": _utcnow(),
            "captured_pin": model_name,
            "rf_mcp_git_sha": _git_sha(),
        }
        entry = capture_entry(runs, provenance=provenance)
        print(f"\n=== CAPTURE {model_name} ({_BASIC_SCENARIO.id}) ===")
        if entry is None:
            pytest.fail(
                f"Capture invalid for {model_name}: infra fault or no valid quorum "
                f"(refusing to bless a broken baseline)."
            )
        _write_baseline_entry(model_name, entry)
        print(json.dumps(entry, indent=2))
        return

    verdict = evaluate(model_name, _BASIC_SCENARIO.id, runs, registered, baselines)

    safe = model_name.replace("/", "_")
    write_gate_report([verdict], _METRICS_DIR / f"gate_{safe}.json")

    print(f"\n=== Instruction-quality gate: {model_name} ===")
    print(f"Provider: {provider_for(model_name)} | tier: {verdict.details.get('tier')}")
    print(f"Aggregate: {json.dumps(verdict.details.get('aggregate'))}")
    print(f"Verdict: {verdict.overall}")
    for w in verdict.warnings:
        print(f"  WARN: {w}")
    for h in verdict.hard_failures:
        print(f"  HARD: {h}")

    # An rf-mcp instruction regression (or infra fault, or a reference model that can no
    # longer drive the tools) fails the build. Weak-tier framing flakiness on an inform
    # model only warns. INCONCLUSIVE (most runs died on transport) also fails — we cannot
    # prove "no decrease" from dead runs.
    assert verdict.overall not in ("fail", "inconclusive"), (
        f"Instruction-quality gate {verdict.overall.upper()} for {model_name}: "
        f"{verdict.hard_failures}"
    )

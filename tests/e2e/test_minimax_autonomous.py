"""MiniMax autonomous-agent e2e with a metrics-evaluation quality gate.

Drives the in-process pydantic-ai harness against MiniMax models (M2, M2.5, M2.7,
M3) over the MiniMax OpenAI-compatible endpoint, collects per-model metrics, and
evaluates the rf-mcp quality gate (tests/e2e/quality_gate.py).

The gate is designed so MiniMax's own tool-call-framing flakiness (weak tiers emit
dict-shaped args / omit required args) does NOT red the build, while a real rf-mcp
regression (tool registration broken, server faulting on well-formed calls, recursion,
handshake failure) DOES. See the module docstrings for the attribution rules.

Gated on MINIMAX_API_KEY (a repo secret; also read from the environment locally).
Run in CI via the ``minimax-autonomous`` job (e2e-weekly) and the ``e2e-minimax-smoke``
job (ci.yml). Locally::

    MINIMAX_API_KEY=... MINIMAX_MODELS=MiniMax-M3 \
        uv run pytest tests/e2e/test_minimax_autonomous.py -v -s
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.e2e.agent_integration import MCPAgentIntegration
from tests.e2e.fixtures import mcp_server, metrics_collector
from tests.e2e.minimax_support import (
    MINIMAX_BASE_URL,
    minimax_api_key,
    minimax_models_to_test,
)
from tests.e2e.models import ExpectedToolCall, Scenario
from tests.e2e.quality_gate import evaluate_scenario, write_gate_report

pytestmark = pytest.mark.skipif(
    not minimax_api_key(),
    reason="Requires MINIMAX_API_KEY (MiniMax platform key) to be set",
)

_METRICS_DIR = Path(__file__).parent / "metrics" / "minimax"

# A deterministic, non-browser scenario: exercises the analyze -> execute -> build
# pipeline with pure BuiltIn/Collections keywords (no display server needed in CI).
_BASIC_SCENARIO = Scenario(
    id="minimax_basic_list",
    name="Basic list workflow",
    description="Create a 3-item list and verify its length via MCP tools.",
    context="generic",
    prompt=(
        "Create a Robot Framework test using the MCP tools:\n"
        "1. Use analyze_scenario to start a session.\n"
        "2. Use execute_step to create a list with items 'apple', 'banana', 'cherry'.\n"
        "3. Use execute_step to verify the list length is 3.\n"
        "4. Use build_test_suite to generate the suite.\n"
        "Call keywords WITHOUT a library prefix (e.g. 'Create List', not "
        "'Collections.Create List'). Reuse the session_id from analyze_scenario."
    ),
    expected_tools=[
        ExpectedToolCall(tool_name="analyze_scenario", min_calls=1),
        ExpectedToolCall(tool_name="execute_step", min_calls=1),
    ],
    expected_outcome="A suite that builds a 3-item list and asserts its length.",
    min_tool_hit_rate=0.5,
    tags=["minimax", "smoke", "generic"],
)


async def _registered_tool_names(mcp_server) -> list[str]:
    """Return the tool names the server currently exposes (model-independent)."""
    tools = await mcp_server.list_tools()
    return [getattr(t, "name", None) for t in tools if getattr(t, "name", None)]


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name", minimax_models_to_test())
async def test_minimax_autonomous_quality_gate(
    model_name: str, mcp_server, metrics_collector
):
    """Run one MiniMax model through the basic workflow and enforce the quality gate.

    HARD failures (rf-mcp regressions) fail the test. SOFT warnings (model framing /
    weak-tier flakiness) are printed but keep the build green.
    """
    # Bound weaker models tightly; M3 gets a little more room for recovery.
    request_limit = 30 if model_name.endswith("M3") else 18

    integration = MCPAgentIntegration(mcp_server, metrics_collector)
    agent = integration.create_agent_with_mcp_tools(
        model_name=model_name, use_test_model=False
    )

    metrics_collector.start_recording()
    agent_output = ""
    run_error = None
    try:
        agent_output, _ = await integration.run_agent_with_scenario(
            agent, _BASIC_SCENARIO.prompt, request_limit=request_limit
        )
    except Exception as e:  # UsageLimitExceeded / model framing / transport blips
        run_error = f"{type(e).__name__}: {e}"
    finally:
        metrics_collector.stop_recording()

    result = metrics_collector.generate_result(_BASIC_SCENARIO, agent_output=agent_output)
    result.metadata.update({"model": model_name, "endpoint": MINIMAX_BASE_URL})
    if run_error:
        result.metadata["run_error"] = run_error

    registered = await _registered_tool_names(mcp_server)
    verdict = evaluate_scenario(
        result,
        registered,
        model=model_name,
        min_tool_hit_rate=_BASIC_SCENARIO.min_tool_hit_rate,
    )

    # Persist per-model metrics + gate verdict for the CI artifact / trend tracking.
    safe = model_name.replace("/", "_")
    metrics_collector.save_metrics(result, _METRICS_DIR)
    write_gate_report([verdict], _METRICS_DIR / f"gate_{safe}.json")

    print(f"\n=== MiniMax quality gate: {model_name} ===")
    print(f"Endpoint: {MINIMAX_BASE_URL}")
    print(f"Registered tools: {len(registered)} | tool calls: {result.total_tool_calls}")
    print(f"Tool hit rate: {result.tool_hit_rate:.2%} | verdict: {verdict.overall}")
    if run_error:
        print(f"Run ended with: {run_error}")
    if verdict.soft_warnings:
        print("SOFT warnings (not failing):")
        for w in verdict.soft_warnings:
            print(f"  - {w}")
    if verdict.hard_failures:
        print("HARD failures:")
        for h in verdict.hard_failures:
            print(f"  - {h}")

    # HARD gate: a real rf-mcp regression fails the build; model flakiness does not.
    assert verdict.overall != "fail", (
        f"rf-mcp quality gate FAILED for {model_name}: {verdict.hard_failures}"
    )

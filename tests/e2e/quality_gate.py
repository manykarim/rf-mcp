"""Metrics-evaluation quality gate for the agentic e2e harness.

Purpose: adding weaker external models (e.g. MiniMax M2/M2.5) must NOT red the build
for the model's own tool-call-framing flakiness, yet a genuine rf-mcp regression
(tool registration broken, server faulting on well-formed calls, recursion, handshake
failure) MUST be caught.

The crux is error ATTRIBUTION — separating model-agnostic rf-mcp health from
model-choice noise:

- ``RF_MCP`` fault: the call reached the server and the SERVER/infra faulted —
  unhandled exception/traceback, recursion, MCP protocol/registration error,
  handshake/libdoc failure, a tool that is not even registered. These are real
  rf-mcp regressions and gate HARD.
- ``MODEL_FRAMING`` fault: the model drove rf-mcp wrong — a graceful Robot Framework
  keyword/argument error the server reported *as a structured hint* (the server
  working as designed), malformed args rejected at the pydantic boundary, or the
  agent never calling a tool. These are reported but never fail the build.

HARD checks are model-independent — tool registration plus attributed rf-mcp/infra
faults. SOFT checks (tool-hit-rate with the ``max(0.40, threshold - 0.30)`` tolerance,
all-calls-failed, off-script behaviour) warn only and NEVER red the build, so a weak
model's framing flakiness cannot cause a false failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Core tools every profile-free rf-mcp server must expose. A missing member here is a
# registration/handshake regression (exactly what the fastmcp3 tool-profile bug hit).
CORE_TOOLS = frozenset(
    {
        "analyze_scenario",
        "execute_step",
        "build_test_suite",
        "manage_session",
        "recommend_libraries",
        "get_session_state",
        "find_keywords",
        "get_keyword_info",
    }
)

# Substrings that mark an rf-mcp / infra fault rather than a graceful RF keyword error.
# Deliberately narrow: graceful "No keyword with name 'X' found. Did you mean ..." hints
# are the server WORKING, so they are NOT listed here.
RF_MCP_INFRA_PATTERNS = (
    "maximum recursion",
    "recursionerror",
    "traceback (most recent call last)",
    "tool not found",
    "unknown tool",
    "not registered",
    "closedresource",
    "cancel scope",
    "handshake",
    "failed to load library documentation",
    "libdoc",
    "internal server error",
    "notimplementederror",
    "attributeerror: 'nonetype'",
)


def classify_error(error: Optional[str], tool_name: str, registered_tools: Sequence[str]) -> str:
    """Classify a failed tool call as ``"rf_mcp"`` or ``"model_framing"``.

    A call to a tool that is not even registered is an rf-mcp fault. Otherwise the
    error text decides: infra-fault patterns -> rf_mcp; everything else (graceful
    keyword/arg hints, wrong-library mistakes) -> model_framing.
    """
    if tool_name not in set(registered_tools):
        return "rf_mcp"
    if not error:
        return "model_framing"
    low = error.lower()
    if any(pat in low for pat in RF_MCP_INFRA_PATTERNS):
        return "rf_mcp"
    return "model_framing"


@dataclass
class GateVerdict:
    """Outcome of evaluating one scenario run against the quality gate."""

    model: str
    scenario_id: str
    overall: str  # "pass" | "warn" | "fail"
    hard_failures: List[str] = field(default_factory=list)
    soft_warnings: List[str] = field(default_factory=list)
    details: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "model": self.model,
            "scenario_id": self.scenario_id,
            "overall": self.overall,
            "hard_failures": self.hard_failures,
            "soft_warnings": self.soft_warnings,
            "details": self.details,
        }


def evaluate_scenario(
    result,
    registered_tools: Sequence[str],
    *,
    model: str,
    min_tool_hit_rate: float = 0.8,
) -> GateVerdict:
    """Evaluate a ScenarioResult against the rf-mcp quality gate.

    Args:
        result: a ScenarioResult (models.ScenarioResult) from MetricsCollector.
        registered_tools: tool names the server currently exposes (``list_tools``).
        model: model id under test (for reporting).
        min_tool_hit_rate: the scenario's declared threshold.

    Returns:
        GateVerdict. ``overall == "fail"`` iff a HARD check fails or a SOFT check
        escalated; ``"warn"`` absorbs pure model-framing flakiness (still green).
    """
    registered = list(registered_tools)
    tool_names = [tc.tool_name for tc in result.tool_calls]

    # Attribute every failed call.
    rf_mcp_faults: List[str] = []
    model_faults: List[str] = []
    for tc in result.tool_calls:
        if tc.success:
            continue
        kind = classify_error(tc.error, tc.tool_name, registered)
        target = rf_mcp_faults if kind == "rf_mcp" else model_faults
        target.append(f"{tc.tool_name}: {(tc.error or '')[:120]}")

    hard: List[str] = []
    soft: List[str] = []

    # HARD 1: tool registration is intact (model-independent).
    missing = sorted(CORE_TOOLS - set(registered))
    if missing:
        hard.append(f"core tools not registered: {missing}")

    # HARD 2: no rf-mcp/infra faults on calls that reached the server.
    if rf_mcp_faults:
        hard.append(f"rf-mcp server faults ({len(rf_mcp_faults)}): {rf_mcp_faults[:3]}")

    # The ONLY server-health (HARD) signals are tool registration (HARD 1) and
    # attributed rf-mcp/infra faults (HARD 2). Everything below is model-dependent and
    # stays SOFT: a weak model that calls the wrong tools, makes only malformed calls,
    # or goes off-script warns but never reds the build. A genuine "server faulting on
    # every call" still surfaces as an rf_mcp_fault (HARD 2), because an infra fault
    # produces an infra-classified error — so demoting these to SOFT loses no real
    # regression signal while removing false reds on model framing.
    analyze_ok = any(
        tc.tool_name == "analyze_scenario" and tc.success for tc in result.tool_calls
    )
    any_success = any(tc.success for tc in result.tool_calls)
    if tool_names and not any_success:
        soft.append("no tool call succeeded (weak model / all calls model-framing)")
    if not analyze_ok:
        soft.append("analyze_scenario was not successfully called")

    # SOFT: tool hit rate is model-dependent (did the model call the EXPECTED tools by
    # name?), so it never HARD-fails — a model solving via equivalent tools legitimately
    # scores low here. Reported with the established tolerance for visibility only.
    effective_floor = max(0.40, min_tool_hit_rate - 0.30)
    if result.tool_hit_rate < effective_floor:
        soft.append(
            f"tool_hit_rate {result.tool_hit_rate:.2f} < floor {effective_floor:.2f} "
            f"(scenario threshold {min_tool_hit_rate:.2f})"
        )

    # SOFT: an agent that never called a tool at all is pure model framing.
    if not tool_names:
        soft.append("agent made no tool calls (model framing / off-script)")

    overall = "fail" if hard else ("warn" if soft else "pass")
    return GateVerdict(
        model=model,
        scenario_id=result.scenario_id,
        overall=overall,
        hard_failures=hard,
        soft_warnings=soft,
        details={
            "tool_hit_rate": result.tool_hit_rate,
            "effective_floor": effective_floor,
            "total_tool_calls": result.total_tool_calls,
            "rf_mcp_fault_count": len(rf_mcp_faults),
            "model_framing_fault_count": len(model_faults),
            "registered_tool_count": len(registered),
            "analyze_scenario_ok": analyze_ok,
            "tool_names": tool_names,
        },
    )


def write_gate_report(verdicts: Sequence[GateVerdict], path: Path) -> Dict[str, object]:
    """Write a combined gate report JSON and return it.

    ``overall`` is ``fail`` if any verdict failed, else ``warn`` if any warned, else
    ``pass``. Callers assert on the returned ``overall`` for the HARD gate.
    """
    order = {"pass": 0, "warn": 1, "fail": 2}
    overall = "pass"
    for v in verdicts:
        if order[v.overall] > order[overall]:
            overall = v.overall
    report = {
        "overall": overall,
        "verdicts": [v.to_dict() for v in verdicts],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

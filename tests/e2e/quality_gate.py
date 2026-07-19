"""Instruction-quality gate for the agentic e2e tests.

PURPOSE (this is the point of the whole harness): wrong / failed / unexpected / absent
tool calls ARE a signal of rf-mcp's instruction quality — how well its tool
descriptions, MCP instructions, schemas and discovery outputs let an agent DISCOVER and
DRIVE the tools. When a FIXED, pinned model that previously produced clean calls starts
producing worse ones, only two things can have changed: the model (eliminated by pinning
the model id) or rf-mcp's instruction surface (the thing under test). So the gate MUST
fail on degraded tool-call quality.

The old design excused whole error classes as "model framing" (SOFT) — that inverted the
purpose. This one instead isolates an rf-mcp regression from model non-determinism by:
- comparing each metric against THAT SAME model's captured baseline (a drop = rf-mcp got
  worse for a fixed model),
- a tolerance band derived from the variance measured at capture,
- aggregating over N runs (median for rates, so one unlucky run can't false-red).

Model-independent infra faults (recursion, unregistered tool, handshake/libdoc failure,
traceback) stay a HARD fail on ANY single run — they never needed a baseline.

Tiers (membership lives in the baseline JSON):
- reference (e.g. MiniMax-M3): HARD on absolute floors AND regression-vs-baseline.
- hard_gate (e.g. M2.7/M2.5): HARD on regression-vs-baseline only.
- inform (e.g. M2, noisiest): regression -> WARN; only infra faults HARD-fail it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence

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

# Substrings that mark an rf-mcp / infra fault (server broke) rather than a graceful RF
# keyword hint (the server working). Narrow on purpose — graceful "No keyword with name
# ... Did you mean" hints are NOT infra faults (they lower success_rate, which the
# regression gate already measures).
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

_GATE_METRICS = ("tool_success_rate", "tool_hit_rate", "task_completion_rate")
_TOL_FLOOR = 0.10


def is_infra_fault(error: Optional[str], tool_name: str, registered_tools: Sequence[str]) -> bool:
    """True when a failed call reflects an rf-mcp/infra fault (not a graceful hint)."""
    if tool_name not in set(registered_tools):
        return True
    if not error:
        return False
    low = error.lower()
    return any(pat in low for pat in RF_MCP_INFRA_PATTERNS)


@dataclass
class RunMetrics:
    """Metrics for a single scenario run — every wrong/failed call lowers these."""

    tool_success_rate: float
    tool_hit_rate: float
    completed: bool
    total_calls: int
    successful_calls: int
    infra_faults: int
    unexpected_tool_calls: int
    aborted: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "tool_success_rate": round(self.tool_success_rate, 4),
            "tool_hit_rate": round(self.tool_hit_rate, 4),
            "completed": self.completed,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "infra_faults": self.infra_faults,
            "unexpected_tool_calls": self.unexpected_tool_calls,
            "aborted": self.aborted,
        }


def compute_run_metrics(
    result,
    registered_tools: Sequence[str],
    *,
    expected_tool_names: Sequence[str],
    completion_tool_names: Sequence[str],
    run_error: Optional[str] = None,
) -> RunMetrics:
    """Compute one run's metrics from a ScenarioResult.

    - tool_success_rate: successful calls / total calls (0 if no calls — no calls is the
      worst outcome, never a vacuous pass).
    - completed: every completion tool was called AND succeeded at least once.
    - infra_faults: failed calls attributed to rf-mcp/infra (see is_infra_fault).
    - aborted: the run made no tool calls and ended on a transport/usage error.
    """
    calls = list(result.tool_calls)
    total = len(calls)
    successful = sum(1 for tc in calls if tc.success)
    success_rate = (successful / total) if total else 0.0

    infra = sum(
        1 for tc in calls
        if not tc.success and is_infra_fault(tc.error, tc.tool_name, registered_tools)
    )

    expected = set(expected_tool_names)
    allowed = expected | CORE_TOOLS
    unexpected = sum(1 for tc in calls if tc.tool_name not in allowed)

    succeeded_by_tool = {tc.tool_name for tc in calls if tc.success}
    completed = bool(completion_tool_names) and all(
        name in succeeded_by_tool for name in completion_tool_names
    )

    aborted = total == 0 and bool(run_error)

    return RunMetrics(
        tool_success_rate=success_rate,
        tool_hit_rate=float(result.tool_hit_rate),
        completed=completed,
        total_calls=total,
        successful_calls=successful,
        infra_faults=infra,
        unexpected_tool_calls=unexpected,
        aborted=aborted,
    )


@dataclass
class AggMetrics:
    n_total: int
    n_valid: int
    tool_success_rate: float
    tool_hit_rate: float
    task_completion_rate: float
    completion_best_of_n: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_total": self.n_total,
            "n_valid": self.n_valid,
            "tool_success_rate": round(self.tool_success_rate, 4),
            "tool_hit_rate": round(self.tool_hit_rate, 4),
            "task_completion_rate": round(self.task_completion_rate, 4),
            "completion_best_of_n": self.completion_best_of_n,
        }


def _valid_runs(runs: Sequence[RunMetrics]) -> List[RunMetrics]:
    return [r for r in runs if not r.aborted and r.infra_faults == 0]


def aggregate(runs: Sequence[RunMetrics]) -> AggMetrics:
    """Aggregate valid runs: median for rates (outlier-robust), mean for completion."""
    valid = _valid_runs(runs)
    if not valid:
        return AggMetrics(len(runs), 0, 0.0, 0.0, 0.0, 0)
    return AggMetrics(
        n_total=len(runs),
        n_valid=len(valid),
        tool_success_rate=float(median(r.tool_success_rate for r in valid)),
        tool_hit_rate=float(median(r.tool_hit_rate for r in valid)),
        task_completion_rate=sum(1 for r in valid if r.completed) / len(valid),
        completion_best_of_n=1 if any(r.completed for r in valid) else 0,
    )


def iqr(values: Sequence[float]) -> float:
    """Interquartile range (P75-P25), used to widen tolerance for noisy metrics."""
    xs = sorted(values)
    if len(xs) < 2:
        return 0.0
    def _pct(p: float) -> float:
        k = (len(xs) - 1) * p
        lo = math.floor(k)
        hi = math.ceil(k)
        if lo == hi:
            return xs[int(k)]
        return xs[lo] * (hi - k) + xs[hi] * (k - lo)
    return max(0.0, _pct(0.75) - _pct(0.25))


# ── baselines ─────────────────────────────────────────────────────

def load_baselines(path: Path) -> dict:
    """Load the instruction-quality baseline JSON (empty skeleton if absent)."""
    if not path.exists():
        return {"scenarios": {}, "reference_models": [], "hard_gate_models": [], "inform_models": []}
    return json.loads(path.read_text(encoding="utf-8"))


def model_tier(model: str, baselines: dict) -> str:
    """Return 'reference' | 'hard_gate' | 'inform' | 'unknown' for a model."""
    if model in baselines.get("reference_models", []):
        return "reference"
    if model in baselines.get("hard_gate_models", []):
        return "hard_gate"
    if model in baselines.get("inform_models", []):
        return "inform"
    return "unknown"


def _tolerance(metric: str, base_entry: dict, baselines: dict) -> float:
    tol_cfg = baselines.get("tolerances", {})
    floor = float(tol_cfg.get(metric, _TOL_FLOOR))
    measured_iqr = float((base_entry.get("iqr") or {}).get(metric, 0.0))
    return max(floor, measured_iqr)


@dataclass
class GateVerdict:
    model: str
    scenario_id: str
    overall: str  # "pass" | "warn" | "fail" | "inconclusive"
    hard_failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "model": self.model,
            "scenario_id": self.scenario_id,
            "overall": self.overall,
            "hard_failures": self.hard_failures,
            "warnings": self.warnings,
            "details": self.details,
        }


def evaluate(
    model: str,
    scenario_id: str,
    runs: Sequence[RunMetrics],
    registered_tools: Sequence[str],
    baselines: dict,
) -> GateVerdict:
    """Evaluate N runs of one (model, scenario) against the instruction-quality gate."""
    tier = model_tier(model, baselines)
    hard: List[str] = []
    warn: List[str] = []
    n = len(runs)

    # STEP 1 — model-independent infra HARD (any single run), for every tier.
    missing = sorted(CORE_TOOLS - set(registered_tools))
    if missing:
        hard.append(f"registration regression: core tools missing {missing}")
    total_infra = sum(r.infra_faults for r in runs)
    if total_infra:
        hard.append(f"rf-mcp infra fault(s) across runs: {total_infra}")

    # STEP 2 — quorum of valid runs.
    agg = aggregate(runs)
    quorum = math.ceil(n / 2) if n else 1
    inconclusive = agg.n_valid < quorum

    # STEP 3/4/5 — regression + absolute floors (only if we have a quorum and a tier).
    base_entry = (baselines.get("scenarios", {}).get(scenario_id, {}) or {}).get(model)
    if not inconclusive and tier in ("reference", "hard_gate", "inform"):
        if base_entry is None:
            msg = f"no baseline for ({model}, {scenario_id})"
            if tier == "inform":
                warn.append(msg + " (inform tier)")
            else:
                hard.append(msg + " (fail closed — capture a baseline)")
        else:
            for m in _GATE_METRICS:
                tol = _tolerance(m, base_entry, baselines)
                agg_v = getattr(agg, m)
                base_v = float(base_entry.get(m, 0.0))
                if agg_v < base_v - tol:
                    line = (
                        f"regression {m}: {agg_v:.2f} < baseline {base_v:.2f} - tol {tol:.2f}"
                    )
                    (warn if tier == "inform" else hard).append(line)

        if tier == "reference":
            floors = (baselines.get("absolute_floors", {}) or {}).get(model, {})
            for m in ("tool_success_rate", "tool_hit_rate"):
                if m in floors and getattr(agg, m) < float(floors[m]):
                    hard.append(f"absolute floor breach {m}: {getattr(agg, m):.2f} < {floors[m]}")
            if "task_completion_best_of_n" in floors and agg.completion_best_of_n < int(
                floors["task_completion_best_of_n"]
            ):
                hard.append("reference model completed the task in ZERO of N runs")

    if tier == "unknown":
        warn.append(f"model '{model}' has no tier in baselines — not gated")

    if inconclusive:
        overall = "inconclusive"
    elif hard:
        overall = "fail"
    elif warn:
        overall = "warn"
    else:
        overall = "pass"

    return GateVerdict(
        model=model,
        scenario_id=scenario_id,
        overall=overall,
        hard_failures=hard,
        warnings=warn,
        details={"tier": tier, "aggregate": agg.to_dict(), "per_run": [r.to_dict() for r in runs]},
    )


def capture_entry(runs: Sequence[RunMetrics]) -> Optional[dict]:
    """Build a baseline entry from capture runs, or None if the capture is invalid.

    Refuses to bless a broken state: aborts (returns None) if any run had an infra fault
    or if there is no valid quorum. The aggregates + per-run values + IQR are stored so
    the gate can widen tolerance to measured noise and reviewers can audit a change.
    """
    if any(r.infra_faults for r in runs):
        return None
    valid = _valid_runs(runs)
    if not valid or len(valid) < math.ceil(len(runs) / 2):
        return None
    agg = aggregate(runs)
    return {
        "tool_success_rate": round(agg.tool_success_rate, 4),
        "tool_hit_rate": round(agg.tool_hit_rate, 4),
        "task_completion_rate": round(agg.task_completion_rate, 4),
        "n_runs": len(runs),
        "per_run": {
            m: [round(getattr(r, m), 4) for r in valid]
            for m in ("tool_success_rate", "tool_hit_rate")
        },
        "iqr": {
            "tool_success_rate": round(iqr([r.tool_success_rate for r in valid]), 4),
            "tool_hit_rate": round(iqr([r.tool_hit_rate for r in valid]), 4),
            "task_completion_rate": 0.0,
        },
    }


def overall_verdict(verdicts: Sequence[GateVerdict]) -> str:
    order = {"pass": 0, "warn": 1, "inconclusive": 2, "fail": 3}
    worst = "pass"
    for v in verdicts:
        if order[v.overall] > order[worst]:
            worst = v.overall
    return worst


def write_gate_report(verdicts: Sequence[GateVerdict], path: Path) -> Dict[str, object]:
    report = {"overall": overall_verdict(verdicts), "verdicts": [v.to_dict() for v in verdicts]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

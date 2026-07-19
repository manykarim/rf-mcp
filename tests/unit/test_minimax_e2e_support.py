"""Fast, no-network unit tests for the MiniMax e2e wiring + quality gate.

These run in the normal unit suite (no MINIMAX_API_KEY / no external calls) so the
model-resolution and quality-gate logic is guarded on every commit, independent of
the secret-gated e2e jobs.
"""

from __future__ import annotations

import pytest

from tests.e2e.minimax_support import (
    MINIMAX_BASE_URL,
    MINIMAX_MODELS,
    default_agent_model,
    is_minimax_model,
    minimax_models_to_test,
    real_llm_available,
    resolve_model,
)
from tests.e2e.models import ScenarioResult, ToolCallRecord
from tests.e2e.quality_gate import (
    CORE_TOOLS,
    aggregate,
    capture_entry,
    compute_run_metrics,
    evaluate,
    is_infra_fault,
)


# ── model resolution ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "name,expected",
    [
        ("MiniMax-M2", True),
        ("MiniMax-M3", True),
        ("minimax-m2.5", True),
        ("gpt-5-mini", False),
        ("claude-haiku-4.5", False),
    ],
)
def test_is_minimax_model(name, expected):
    assert is_minimax_model(name) is expected


def test_resolve_minimax_requires_key(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MINIMAX_API_KEY"):
        resolve_model("MiniMax-M2")


def test_resolve_minimax_points_at_minimax_endpoint(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "dummy-key")
    model = resolve_model("MiniMax-M2.7")
    assert getattr(model, "model_name", None) == "MiniMax-M2.7"
    base = str(getattr(getattr(model, "client", None), "base_url", ""))
    assert "api.minimax.io" in base and base.rstrip("/").endswith("/v1")
    assert MINIMAX_BASE_URL == "https://api.minimax.io/v1"


def test_default_agent_model_priority(monkeypatch):
    monkeypatch.delenv("E2E_AGENT_MODEL", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    assert default_agent_model() == "gpt-4o-mini"

    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    assert default_agent_model() == "MiniMax-M3"  # MiniMax wins when its key is set

    monkeypatch.setenv("E2E_AGENT_MODEL", "MiniMax-M2")
    assert default_agent_model() == "MiniMax-M2"  # explicit override wins


def test_default_agent_model_empty_openai_model_falls_back(monkeypatch):
    monkeypatch.delenv("E2E_AGENT_MODEL", raising=False)
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "   ")  # empty/whitespace secret in CI
    assert default_agent_model() == "gpt-5-mini"


def test_real_llm_available(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("USE_REAL_LLM", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert real_llm_available() is False

    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    assert real_llm_available() is True  # MiniMax key alone enables the harness


def test_minimax_models_to_test(monkeypatch):
    monkeypatch.delenv("MINIMAX_MODELS", raising=False)
    assert minimax_models_to_test() == list(MINIMAX_MODELS)
    monkeypatch.setenv("MINIMAX_MODELS", "MiniMax-M3, MiniMax-M2")
    assert minimax_models_to_test() == ["MiniMax-M3", "MiniMax-M2"]


# ── quality gate: metrics + per-model baseline regression ─────────

_ALL_TOOLS = sorted(CORE_TOOLS)
_EXPECTED = ["analyze_scenario", "execute_step", "build_test_suite"]
_COMPLETION = ["analyze_scenario", "build_test_suite"]


def _record(tool, success, error=None):
    return ToolCallRecord(
        tool_name=tool, arguments={}, success=success, result={}, error=error, timestamp=0.0
    )


def _result(calls, hit_rate=1.0):
    return ScenarioResult(
        scenario_id="s", success=True, tool_calls=calls, tool_hit_rate=hit_rate,
        total_tool_calls=len(calls), expected_tool_calls_met=1, expected_tool_calls_total=1,
        errors=[c.error for c in calls if c.error], execution_time_seconds=1.0,
    )


def _rm(calls, hit_rate=1.0, run_error=None, registered=_ALL_TOOLS):
    return compute_run_metrics(
        _result(calls, hit_rate), registered,
        expected_tool_names=_EXPECTED, completion_tool_names=_COMPLETION, run_error=run_error,
    )


# A healthy run: analyze + execute + build all succeed.
def _healthy():
    return _rm([_record("analyze_scenario", True), _record("execute_step", True), _record("build_test_suite", True)])


# A degraded run: half the calls fail with GRACEFUL keyword errors (the signature of a
# docstring/instruction regression that makes the agent pick wrong keywords) -> lower
# success_rate. NOT infra faults.
def _degraded():
    return _rm([
        _record("analyze_scenario", True),
        _record("execute_step", False, error="No keyword with name 'Xyz' found. Did you mean"),
        _record("execute_step", False, error="Setting list value to scalar variable"),
        _record("build_test_suite", True),
    ], hit_rate=0.6)


def _baselines(**over):
    b = {
        "reference_models": ["MiniMax-M3"],
        "hard_gate_models": ["MiniMax-M2.7"],
        "inform_models": ["MiniMax-M2"],
        "tolerances": {"tool_success_rate": 0.1, "tool_hit_rate": 0.1, "task_completion_rate": 0.1},
        "absolute_floors": {"MiniMax-M3": {"tool_success_rate": 0.7, "tool_hit_rate": 0.8, "task_completion_best_of_n": 1}},
        "scenarios": {"s": {
            "MiniMax-M3": {"tool_success_rate": 1.0, "tool_hit_rate": 1.0, "task_completion_rate": 1.0, "iqr": {}},
            "MiniMax-M2.7": {"tool_success_rate": 0.8, "tool_hit_rate": 0.7, "task_completion_rate": 0.8, "iqr": {}},
            "MiniMax-M2": {"tool_success_rate": 0.5, "tool_hit_rate": 0.4, "task_completion_rate": 0.5, "iqr": {}},
        }},
    }
    b.update(over)
    return b


# --- is_infra_fault ---
@pytest.mark.parametrize("error,tool,expected", [
    ("maximum recursion depth exceeded", "execute_step", True),
    ("Traceback (most recent call last): ...", "analyze_scenario", True),
    ("anyio ClosedResourceError", "execute_step", True),
    ("No keyword with name 'X' found. Did you mean", "execute_step", False),
    (None, "execute_step", False),
    ("whatever", "not_a_real_tool", True),  # unregistered tool = registration fault
])
def test_is_infra_fault(error, tool, expected):
    assert is_infra_fault(error, tool, _ALL_TOOLS) is expected


# --- compute_run_metrics ---
def test_compute_run_metrics_basic():
    rm = _healthy()
    assert rm.tool_success_rate == 1.0 and rm.completed is True and rm.infra_faults == 0
    rm2 = _degraded()
    assert rm2.tool_success_rate == 0.5 and rm2.completed is True  # analyze+build still succeeded
    rm3 = _rm([_record("execute_step", False, error="maximum recursion depth exceeded")])
    assert rm3.infra_faults == 1
    rm4 = _rm([], run_error="UsageLimitExceeded")
    assert rm4.aborted is True and rm4.tool_success_rate == 0.0


# --- aggregate ---
def test_aggregate_medians_and_excludes_invalid():
    runs = [_healthy(), _degraded(), _rm([], run_error="x")]  # 1 aborted excluded
    agg = aggregate(runs)
    assert agg.n_valid == 2
    assert agg.tool_success_rate == 0.75  # median(1.0, 0.5)
    assert agg.completion_best_of_n == 1


# --- evaluate: the philosophy fix ---
def test_gate_pass_when_matches_baseline():
    v = evaluate("MiniMax-M3", "s", [_healthy(), _healthy(), _healthy()], _ALL_TOOLS, _baselines())
    assert v.overall == "pass"


def test_gate_FAILS_on_instruction_regression():
    # THE core behaviour: a fixed reference model's tool-call quality drops (graceful
    # keyword failures from a degraded docstring) -> regression vs baseline -> HARD FAIL.
    v = evaluate("MiniMax-M3", "s", [_degraded(), _degraded(), _degraded()], _ALL_TOOLS, _baselines())
    assert v.overall == "fail"
    assert any("regression tool_success_rate" in h for h in v.hard_failures)


def test_gate_fails_on_absolute_floor_breach():
    # success 0.5 median < M3 floor 0.7 -> HARD even if baseline were lower.
    b = _baselines()
    b["scenarios"]["s"]["MiniMax-M3"]["tool_success_rate"] = 0.5  # pretend baseline low
    v = evaluate("MiniMax-M3", "s", [_degraded(), _degraded(), _degraded()], _ALL_TOOLS, b)
    assert v.overall == "fail"
    assert any("absolute floor" in h for h in v.hard_failures)


def test_gate_infra_fault_hard_fails_every_tier():
    infra = _rm([_record("execute_step", False, error="maximum recursion depth exceeded")])
    for model in ("MiniMax-M3", "MiniMax-M2"):  # even the inform tier
        v = evaluate(model, "s", [infra, _healthy(), _healthy()], _ALL_TOOLS, _baselines())
        assert v.overall == "fail"
        assert any("infra fault" in h for h in v.hard_failures)


def test_gate_inform_tier_regression_only_warns():
    # M2 (inform) drops below its baseline on graceful failures -> WARN, never red.
    v = evaluate("MiniMax-M2", "s", [_rm([_record("execute_step", False, error="No keyword X")], hit_rate=0.0)] * 3,
                 _ALL_TOOLS, _baselines())
    assert v.overall != "fail"


def test_gate_inconclusive_when_no_quorum():
    runs = [_rm([], run_error="x"), _rm([], run_error="x"), _healthy()]  # 2/3 aborted
    v = evaluate("MiniMax-M3", "s", runs, _ALL_TOOLS, _baselines())
    assert v.overall == "inconclusive"


def test_gate_reference_no_baseline_fails_closed():
    b = _baselines()
    b["scenarios"]["s"].pop("MiniMax-M3")
    v = evaluate("MiniMax-M3", "s", [_healthy()] * 3, _ALL_TOOLS, b)
    assert v.overall == "fail"


def test_gate_missing_registration_fails():
    reduced = [t for t in _ALL_TOOLS if t != "build_test_suite"]
    v = evaluate("MiniMax-M3", "s", [_healthy()] * 3, reduced, _baselines())
    assert v.overall == "fail"
    assert any("registration regression" in h for h in v.hard_failures)


# --- capture_entry ---
def test_capture_entry_valid_and_refuses_infra():
    entry = capture_entry([_healthy(), _healthy(), _degraded()])
    assert entry is not None and entry["tool_success_rate"] == 1.0  # median(1,1,0.5)
    infra = _rm([_record("execute_step", False, error="maximum recursion depth exceeded")])
    assert capture_entry([_healthy(), infra, _healthy()]) is None  # refuses to bless a broken state

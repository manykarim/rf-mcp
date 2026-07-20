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
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("USE_REAL_LLM", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert real_llm_available() is False

    monkeypatch.setenv("MINIMAX_API_KEY", "k")
    assert real_llm_available() is True  # MiniMax key alone enables the harness

    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert real_llm_available() is True  # OpenRouter key alone also enables it


def test_provider_routing(monkeypatch):
    from tests.e2e.minimax_support import provider_for, is_openrouter_model, REFERENCE_MODEL
    assert provider_for("MiniMax-M3") == "minimax"
    assert provider_for("qwen/qwen3-coder-30b-a3b-instruct") == "openrouter"
    assert provider_for("gpt-5-mini") == "openai"
    assert is_openrouter_model(REFERENCE_MODEL) is True
    assert is_openrouter_model("MiniMax-M3") is False


def test_resolve_openrouter_points_at_openrouter(monkeypatch):
    from tests.e2e.minimax_support import resolve_model, REFERENCE_MODEL
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    model = resolve_model(REFERENCE_MODEL)
    base = str(getattr(getattr(model, "client", None), "base_url", ""))
    assert "openrouter.ai/api/v1" in base


def test_resolve_openrouter_requires_key(monkeypatch):
    from tests.e2e.minimax_support import resolve_model, REFERENCE_MODEL
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        resolve_model(REFERENCE_MODEL)


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
        "absolute_floors": {"MiniMax-M3": {"tool_success_rate": 0.7, "first_try_selection_rate": 0.5, "task_completion_best_of_n": 1}},
        "scenarios": {"s": {
            "MiniMax-M3": {"tool_success_rate": 1.0, "tool_hit_rate": 1.0, "task_completion_rate": 1.0, "first_try_selection_rate": 1.0, "iqr": {}},
            "MiniMax-M2.7": {"tool_success_rate": 0.8, "tool_hit_rate": 0.7, "task_completion_rate": 0.8, "first_try_selection_rate": 0.8, "iqr": {}},
            "MiniMax-M2": {"tool_success_rate": 0.5, "tool_hit_rate": 0.4, "task_completion_rate": 0.5, "first_try_selection_rate": 0.5, "iqr": {}},
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


# ── new metrics + validation protocol (change: autonomous-e2e-coverage) ──

from tests.e2e.quality_gate import (  # noqa: E402
    aggregate as _aggregate,
    is_excluded_model,
    is_scenario_validated,
    staleness_warning,
    validate_scenario,
)


def test_first_try_metric_not_inflated_by_flailing():
    # First call is a wrong/unexpected tool -> first_try_ok False even if later calls hit.
    good = _rm([_record("analyze_scenario", True), _record("execute_step", True), _record("build_test_suite", True)])
    assert good.first_try_ok is True
    floundering = _rm([_record("get_session_state", True)] + [_record("execute_step", True)] * 6)
    assert floundering.first_try_ok is False  # first call not an expected tool


def test_artifact_executes_metric():
    without = _rm([_record("analyze_scenario", True), _record("build_test_suite", True)])
    assert without.artifact_executes is False
    with_run = _rm([_record("analyze_scenario", True), _record("build_test_suite", True), _record("run_test_suite", True)])
    assert with_run.artifact_executes is True


def test_validate_scenario_calibrated_and_sensitive():
    good = _aggregate([_healthy(), _healthy(), _healthy()])
    degraded = _aggregate([_degraded(), _degraded(), _degraded()])
    ok, reason = validate_scenario(good, degraded)
    assert ok is True, reason


def test_validate_scenario_rejects_uncalibrated():
    # reference never completes on "good" -> no headroom -> invalid probe
    notdone = _rm([_record("find_keywords", True)])  # completion tools never succeed
    good = _aggregate([notdone, notdone, notdone])
    degraded = _aggregate([notdone, notdone, notdone])
    ok, reason = validate_scenario(good, degraded)
    assert ok is False and "uncalibrated" in reason


def test_validate_scenario_rejects_insensitive():
    # degradation does not lower any metric (e.g. metric inverts/stays) -> invalid probe
    good = _aggregate([_healthy(), _healthy(), _healthy()])
    ok, reason = validate_scenario(good, good)  # degraded == good
    assert ok is False and "insensitive" in reason


def test_unvalidated_scenario_demotes_to_warn():
    b = _baselines()
    b["scenarios"]["s"]["_validated"] = False  # canary rejected this scenario
    v = evaluate("MiniMax-M3", "s", [_degraded(), _degraded(), _degraded()], _ALL_TOOLS, b)
    assert v.overall != "fail"  # regression warns, never hard-fails on an invalid probe


def test_excluded_model_is_not_gated():
    b = _baselines()
    b["excluded_models"] = ["broken-model"]
    assert is_excluded_model("broken-model", b) is True
    from tests.e2e.quality_gate import model_tier
    assert model_tier("broken-model", b) == "excluded"


def test_hit_rate_drop_hard_fails_on_validated_scenario():
    # On a VALIDATED scenario a hit-rate DROP is a legitimate regression (the validation
    # protocol guarantees monotonicity), so it hard-fails even if completion holds.
    b = _baselines()
    b["scenarios"]["s"]["MiniMax-M3"]["tool_hit_rate"] = 1.0
    low_hit = _rm([_record("analyze_scenario", True), _record("execute_step", True), _record("build_test_suite", True)], hit_rate=0.5)
    v = evaluate("MiniMax-M3", "s", [low_hit, low_hit, low_hit], _ALL_TOOLS, b)
    assert v.overall == "fail"
    assert any("tool_hit_rate" in h for h in v.hard_failures)


def test_staleness_warning_on_pin_mismatch():
    b = _baselines()
    b["reference_pin"] = {"model": "qwen/qwen3-coder-30b-a3b-instruct"}
    b["reference_models"] = ["qwen/qwen3-coder-30b-a3b-instruct"]
    entry = {"captured_pin": "old-model-slug"}
    w = staleness_warning(entry, "qwen/qwen3-coder-30b-a3b-instruct", b)
    assert w and "stale baseline" in w


def test_is_scenario_validated_default_and_explicit():
    b = _baselines()
    assert is_scenario_validated("s", b) is True          # missing marker => allowed
    b["scenarios"]["s"]["_validated"] = False
    assert is_scenario_validated("s", b) is False


def test_openrouter_provider_pinning(monkeypatch):
    # OPENROUTER_PROVIDER pins a single provider (reproducibility fix for routing variance).
    from tests.e2e.minimax_support import resolve_model, REFERENCE_MODEL
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setenv("OPENROUTER_PROVIDER", "Novita")
    model = resolve_model(REFERENCE_MODEL)
    eb = getattr(getattr(model, "settings", None), "get", lambda *_: None)("extra_body") \
        if isinstance(getattr(model, "settings", None), dict) else getattr(getattr(model, "settings", None), "extra_body", None)
    # settings may be a TypedDict-like mapping; assert the provider order is pinned
    settings = getattr(model, "settings", None) or {}
    body = settings.get("extra_body") if hasattr(settings, "get") else None
    assert body and body.get("provider", {}).get("order") == ["Novita"]
    assert body["provider"]["allow_fallbacks"] is False


def test_infra_fault_is_fail_not_inconclusive_even_without_quorum():
    # A single infra-fault run (n=1) invalidates the only run -> no quorum. HARD infra
    # must still make overall='fail', not 'inconclusive' (a crash is a fail, not "unknown").
    infra = _rm([_record("execute_step", False, error="maximum recursion depth exceeded")])
    v = evaluate("MiniMax-M3", "s", [infra], _ALL_TOOLS, _baselines())
    assert v.overall == "fail"
    assert any("infra fault" in h for h in v.hard_failures)


def test_missing_gate_metric_warns_on_reference():
    # A reference baseline lacking a gate metric (e.g. first_try) must WARN (ratchet
    # inactive), not silently skip.
    b = _baselines()
    b["scenarios"]["s"]["MiniMax-M3"] = {"tool_success_rate": 1.0, "task_completion_rate": 1.0, "iqr": {}}
    v = evaluate("MiniMax-M3", "s", [_healthy(), _healthy(), _healthy()], _ALL_TOOLS, b)
    assert any("missing gate metric" in w for w in v.warnings)

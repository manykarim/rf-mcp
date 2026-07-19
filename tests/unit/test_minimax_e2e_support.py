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
from tests.e2e.quality_gate import CORE_TOOLS, classify_error, evaluate_scenario


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


# ── quality-gate attribution ──────────────────────────────────────

_ALL_TOOLS = sorted(CORE_TOOLS)


@pytest.mark.parametrize(
    "error,tool,expected",
    [
        ("maximum recursion depth exceeded", "execute_step", "rf_mcp"),
        ("Traceback (most recent call last): ...", "analyze_scenario", "rf_mcp"),
        ("anyio ClosedResourceError", "execute_step", "rf_mcp"),
        ("No keyword with name 'Collections.Create List' found. Did you mean", "execute_step", "model_framing"),
        ("Setting list value to scalar variable is not supported", "execute_step", "model_framing"),
        (None, "execute_step", "model_framing"),
        # A call to an unregistered tool is an rf-mcp registration fault.
        ("whatever", "not_a_real_tool", "rf_mcp"),
    ],
)
def test_classify_error(error, tool, expected):
    assert classify_error(error, tool, _ALL_TOOLS) == expected


def _record(tool, success, error=None):
    return ToolCallRecord(
        tool_name=tool, arguments={}, success=success, result={}, error=error, timestamp=0.0
    )


def _result(calls, hit_rate=1.0):
    return ScenarioResult(
        scenario_id="s",
        success=True,
        tool_calls=calls,
        tool_hit_rate=hit_rate,
        total_tool_calls=len(calls),
        expected_tool_calls_met=1,
        expected_tool_calls_total=1,
        errors=[c.error for c in calls if c.error],
        execution_time_seconds=1.0,
    )


def test_gate_pass_on_healthy_run():
    calls = [_record("analyze_scenario", True), _record("execute_step", True)]
    v = evaluate_scenario(_result(calls), _ALL_TOOLS, model="MiniMax-M3", min_tool_hit_rate=0.5)
    assert v.overall == "pass"


def test_gate_fail_on_rf_mcp_infra_fault():
    # A recursion fault on a well-formed call = rf-mcp regression -> HARD fail.
    calls = [
        _record("analyze_scenario", True),
        _record("execute_step", False, error="maximum recursion depth exceeded"),
    ]
    v = evaluate_scenario(_result(calls), _ALL_TOOLS, model="MiniMax-M3", min_tool_hit_rate=0.5)
    assert v.overall == "fail"
    assert any("server fault" in h for h in v.hard_failures)


def test_gate_warns_not_fails_on_model_framing():
    # Weak model makes graceful keyword mistakes -> WARN/pass, never a HARD fail.
    calls = [
        _record("analyze_scenario", True),
        _record("execute_step", False, error="No keyword with name 'X' found. Did you mean"),
        _record("execute_step", True),
    ]
    v = evaluate_scenario(_result(calls, hit_rate=0.6), _ALL_TOOLS, model="MiniMax-M2", min_tool_hit_rate=0.5)
    assert v.overall != "fail"


def test_gate_warns_not_fails_when_all_calls_model_framing_fail():
    # Weakest tier: every call is malformed/off-script and fails, but none is an
    # rf-mcp/infra fault -> WARN, never a HARD fail (no false red on a weak model).
    calls = [
        _record("execute_step", False, error="Setting list value to scalar variable"),
        _record("execute_step", False, error="No keyword with name 'X' found"),
    ]
    v = evaluate_scenario(_result(calls, hit_rate=0.0), _ALL_TOOLS, model="MiniMax-M2", min_tool_hit_rate=0.5)
    assert v.overall == "warn"
    assert not v.hard_failures


def test_gate_low_hit_rate_never_hard_fails_via_alternate_tools():
    # Model solves via equivalent tools (manage_session/recommend_libraries) and never
    # calls the EXPECTED tools -> hit_rate 0.0 but healthy -> SOFT, not a guidance HARD.
    calls = [_record("manage_session", True), _record("recommend_libraries", True)]
    v = evaluate_scenario(_result(calls, hit_rate=0.0), _ALL_TOOLS, model="MiniMax-M3", min_tool_hit_rate=0.5)
    assert v.overall != "fail"


def test_gate_fail_on_missing_tool_registration():
    calls = [_record("analyze_scenario", True)]
    reduced = [t for t in _ALL_TOOLS if t != "build_test_suite"]
    v = evaluate_scenario(_result(calls), reduced, model="MiniMax-M3", min_tool_hit_rate=0.5)
    assert v.overall == "fail"
    assert any("not registered" in h for h in v.hard_failures)

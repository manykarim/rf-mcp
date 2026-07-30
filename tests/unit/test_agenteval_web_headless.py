"""Unit test for the agenteval harness's headless browser-mode override
(change: agenteval-web-headless-ci). The override lets the demoshop web scenario
run headless (no display) in CI when AGENTEVAL_BROWSER_HEADLESS is set, while the
scenario YAML keeps a visible browser as the local default.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCENARIO_LIB = Path(__file__).resolve().parents[1] / "agenteval" / "scenario_lib.py"
_PROMPT = "Call execute_step: keyword=\"New Browser\", arguments=[\"chromium\", \"headless=False\"]"


def _override():
    spec = importlib.util.spec_from_file_location("agenteval_scenario_lib", _SCENARIO_LIB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._headless_override


def test_headless_override_off_leaves_prompt_unchanged(monkeypatch):
    monkeypatch.delenv("AGENTEVAL_BROWSER_HEADLESS", raising=False)
    assert _override()(_PROMPT) == _PROMPT


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on"])
def test_headless_override_on_flips_to_headless(monkeypatch, val):
    monkeypatch.setenv("AGENTEVAL_BROWSER_HEADLESS", val)
    out = _override()(_PROMPT)
    assert "headless=True" in out
    assert "headless=False" not in out


def test_headless_override_falsey_value_is_off(monkeypatch):
    monkeypatch.setenv("AGENTEVAL_BROWSER_HEADLESS", "0")
    assert _override()(_PROMPT) == _PROMPT

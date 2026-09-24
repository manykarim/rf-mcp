"""Tests for the CI SARIF severity mapping (change: ci-signal-fidelity).

The capability being protected is `ci-quality-scanning`'s requirement that the analysis
never produces a failed check from a finding. Ruff publishes every finding at
level="error", GitHub code scanning fails its check on error-level alerts, so without a
deliberate mapping a pull request goes red for findings it did not introduce.

Run: uv run pytest tests/unit/test_ci_sarif_severity.py -q
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci_sarif_severity.py"


def _load():
    spec = importlib.util.spec_from_file_location("ci_sarif_severity", _MODULE_PATH)
    assert spec and spec.loader, f"cannot load {_MODULE_PATH}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sev = _load()


def _sarif(rule_ids, level="error"):
    """A SARIF shaped like ruff's: every result and rule at one uniform level."""
    return {
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "ruff",
                        "rules": [
                            {
                                "id": rid,
                                "properties": {"id": rid, "problem.severity": level},
                            }
                            for rid in sorted(set(rule_ids))
                        ],
                    }
                },
                "results": [
                    {
                        "ruleId": rid,
                        "level": level,
                        "message": {"text": rid},
                        "locations": [],
                    }
                    for rid in rule_ids
                ],
            }
        ]
    }


def test_uniform_error_input_is_split_by_the_declared_set():
    """The whole point: a constant level carries no severity, so it must be reassigned."""
    doc = _sarif(["S324", "S110", "S112", "F401", "S602"])

    n_high, n_non = sev.relevel(doc)

    assert (n_high, n_non) == (2, 3), "S324 and S602 are HIGH; S110/S112/F401 are not"

    levels = {r["ruleId"]: r["level"] for r in doc["runs"][0]["results"]}
    assert levels == {
        "S324": "error",
        "S602": "error",
        "S110": "warning",
        "S112": "warning",
        "F401": "warning",
    }


def test_both_severity_carriers_are_rewritten():
    """Result level AND rule severity: it is undocumented which one code scanning reads,
    and ruff sets both to the same constant, so leaving one behind risks no effect."""
    doc = _sarif(["S110", "S324"])

    sev.relevel(doc)

    rules = {r["id"]: r for r in doc["runs"][0]["tool"]["driver"]["rules"]}
    assert rules["S110"]["properties"]["problem.severity"] == "warning"
    assert rules["S110"]["defaultConfiguration"]["level"] == "warning"
    assert rules["S324"]["properties"]["problem.severity"] == "error"
    assert rules["S324"]["defaultConfiguration"]["level"] == "error"


def test_no_finding_is_ever_dropped():
    """Severity mapping must not become suppression. A fix that silenced the red check
    by removing findings would also make these tests pass unless the count is asserted."""
    rule_ids = ["S110"] * 160 + ["S112"] * 22 + ["S324"] * 6 + ["F401"] * 23
    doc = _sarif(rule_ids)
    before = len(doc["runs"][0]["results"])

    sev.relevel(doc)

    assert len(doc["runs"][0]["results"]) == before == 211


def test_note_fallback_is_available_without_touching_repo_settings():
    """If the repository's check threshold counts warnings too, the escape hatch is a
    quieter level - not an invisible repository setting that applies to every tool."""
    doc = _sarif(["S110", "S324"])

    sev.relevel(doc, non_high="note")

    levels = {r["ruleId"]: r["level"] for r in doc["runs"][0]["results"]}
    assert levels == {"S110": "note", "S324": "error"}


def test_count_reports_every_finding_not_just_the_high_ones(tmp_path):
    """The digest must keep reporting the full backlog; mapping changes severity, not
    what is counted."""
    path = tmp_path / "q.sarif"
    path.write_text(json.dumps(_sarif(["S110"] * 160 + ["S324"] * 6)), encoding="utf-8")

    total, high = sev.count(path)

    assert total == 166, "the digest reports all findings"
    assert high == 6, "…and separately how many are genuinely high"


def test_the_declared_set_is_not_merely_the_security_prefix():
    """Guards the reasoning, not just the data. `S` is a category, not a severity: of 222
    `S` findings on this source only 6 are what bandit rates HIGH, 160 being S110."""
    assert "S110" not in sev.HIGH, "try-except-pass is not a high-severity finding"
    assert "S112" not in sev.HIGH
    assert "S608" not in sev.HIGH, "hardcoded-sql is not in bandit's HIGH tier here"
    assert "S324" in sev.HIGH, "weak hash IS bandit HIGH (B324)"
    assert all(r.startswith("S") for r in sev.HIGH), "HIGH is drawn from flake8-bandit"


@pytest.mark.parametrize("level", ["error", "warning", "note"])
def test_input_level_is_ignored_entirely(level):
    """Whatever the analyzer claims, severity comes from the declared set."""
    doc = _sarif(["S324", "S110"], level=level)

    sev.relevel(doc)

    levels = {r["ruleId"]: r["level"] for r in doc["runs"][0]["results"]}
    assert levels == {"S324": "error", "S110": "warning"}

"""Tests for change: refactor-mcp-instructions.

Covers the lean order-explicit default spine, template selectability/rollback,
the unified session entry, and the compact API init-guidance bundle.
"""
from __future__ import annotations

import pytest


# ── §2 lean default template ────────────────────────────────────────────────
def _resolve(monkeypatch, template_env=None):
    monkeypatch.delenv("ROBOTMCP_INSTRUCTIONS", raising=False)
    monkeypatch.delenv("ROBOTMCP_INSTRUCTIONS_FILE", raising=False)
    if template_env is None:
        monkeypatch.delenv("ROBOTMCP_INSTRUCTIONS_TEMPLATE", raising=False)
    else:
        monkeypatch.setenv("ROBOTMCP_INSTRUCTIONS_TEMPLATE", template_env)
    from robotmcp.domains.instruction.adapters.fastmcp_adapter import (
        FastMCPInstructionAdapter,
    )

    a = FastMCPInstructionAdapter()
    cfg = a.create_config_from_env()
    return a, a.get_server_instructions(cfg, context=a.get_default_tools_context())


def test_default_is_lean_order_explicit_spine(monkeypatch):
    a, ins = _resolve(monkeypatch)
    assert a.get_template_type().value == "lean"
    # analyze_scenario named first + the churn-killer unified-entry line
    assert "analyze_scenario ONCE" in ins
    assert 'NEVER call manage_session(action="init")' in ins
    # canonical order present
    for tok in ("find_keywords", "execute_step", "build_test_suite"):
        assert tok in ins


def test_default_has_no_per_tool_catalog_echo(monkeypatch):
    _, ins = _resolve(monkeypatch)
    # the lean spine must NOT restate the "Available tools:" catalog the schemas carry
    assert "Available tools:" not in ins


def test_lean_default_materially_shorter_than_standard(monkeypatch):
    _, lean = _resolve(monkeypatch)
    _, standard = _resolve(monkeypatch, template_env="standard")
    assert len(lean) < len(standard)
    # standard was ~2900 chars; the lean spine must be well under half of it
    assert len(lean) < len(standard) * 0.6


def test_old_templates_remain_selectable_for_rollback(monkeypatch):
    _, standard = _resolve(monkeypatch, template_env="standard")
    assert "WORKFLOW GUIDE" in standard  # the standard template's signature line
    _, detailed = _resolve(monkeypatch, template_env="detailed")
    assert "STEP-BY-STEP GUIDE" in detailed


def test_lean_template_lookup_and_checklist_alias():
    from robotmcp.domains.instruction.value_objects import InstructionTemplate

    lean = InstructionTemplate.lean()
    assert lean.template_id == "lean"
    assert InstructionTemplate.get_by_name("lean").template_id == "lean"
    # 'checklist' is an alias for the lean spine (the winning experiment name)
    assert InstructionTemplate.get_by_name("checklist").template_id == "lean"


# ── §4 compact API init-guidance bundle ─────────────────────────────────────
def test_build_api_init_guidance_is_compact_and_load_bearing():
    from robotmcp.utils.requests_guidance import build_api_init_guidance

    g = build_api_init_guidance()
    assert isinstance(g["rules"], list) and g["rules"]
    blob = " ".join(g["rules"]) + " " + g["more"]
    # the load-bearing RequestsLibrary recipes
    assert "On Session" in blob
    assert "${resp.json()}" in blob
    assert "Status Should Be" in blob
    # points to the full cookbook rather than inlining it
    assert 'get_locator_guidance(library="requests")' in g["more"]
    # compact: a handful of rules, not the full cookbook's ~9 tips + examples
    assert len(g["rules"]) <= 6

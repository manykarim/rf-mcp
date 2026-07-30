"""Load rf-mcp agentic scenario definitions for the agenteval harness
(change: adopt-agenteval-harness).

Keeps the scenario DATA (the same YAML shape the bespoke tests/e2e harness used:
``prompt`` + ``expected_tools`` + optional ``min_tool_hit_rate``) while the RUNNER
becomes agenteval keywords. Only PyYAML is needed (a base agenteval dependency), so
this loads fine inside the isolated harness env with no rf-mcp import.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from robot.api.deco import keyword, library


def _headless_override(prompt: str) -> str:
    """When AGENTEVAL_BROWSER_HEADLESS is truthy, rewrite a scenario's authored
    ``headless=False`` to ``headless=True`` so the agent launches a display-less
    browser (change: agenteval-web-headless-ci). Opt-in only; the YAML keeps a
    visible browser as the local default."""
    if os.environ.get("AGENTEVAL_BROWSER_HEADLESS", "").strip().lower() in ("1", "true", "yes", "on"):
        return prompt.replace("headless=False", "headless=True")
    return prompt


@library(scope="GLOBAL")
class scenario_lib:
    @keyword("Load Agentic Scenario")
    def load_agentic_scenario(self, path: str) -> Dict[str, Any]:
        """Return a scenario as a dict the suite can drive and assert on:
        ``{id, name, context, prompt, expected_tool_names, min_tool_hit_rate}``."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        expected: List[str] = [
            t["tool_name"] for t in data.get("expected_tools", []) if t.get("tool_name")
        ]
        return {
            "id": data.get("id", ""),
            "name": data.get("name", ""),
            "context": data.get("context", "generic"),
            "prompt": _headless_override(data.get("prompt", "")),
            "expected_tool_names": expected,
            # Default matches the bespoke Scenario model's default gate.
            "min_tool_hit_rate": float(data.get("min_tool_hit_rate", 0.70)),
        }

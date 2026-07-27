"""Load rf-mcp agentic scenario definitions for the agenteval harness
(change: adopt-agenteval-harness).

Keeps the scenario DATA (the same YAML shape the bespoke tests/e2e harness used:
``prompt`` + ``expected_tools`` + optional ``min_tool_hit_rate``) while the RUNNER
becomes agenteval keywords. Only PyYAML is needed (a base agenteval dependency), so
this loads fine inside the isolated harness env with no rf-mcp import.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml
from robot.api.deco import keyword, library


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
            "prompt": data.get("prompt", ""),
            "expected_tool_names": expected,
            # Default matches the bespoke Scenario model's default gate.
            "min_tool_hit_rate": float(data.get("min_tool_hit_rate", 0.70)),
        }

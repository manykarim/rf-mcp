"""docs/MCP_TOOLS.md is generated from the published tool schemas and must stay current."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_tool_reference_is_current() -> None:
    spec = importlib.util.spec_from_file_location("generate_tool_reference", ROOT / "scripts" / "generate_tool_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    committed = (ROOT / "docs" / "MCP_TOOLS.md").read_text(encoding="utf-8")
    assert committed == module.generate(), "docs/MCP_TOOLS.md is out of date: run uv run python scripts/generate_tool_reference.py"

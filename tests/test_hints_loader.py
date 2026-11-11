"""Tolerant `.robotframework/hints.toml` loading and hint gating by output kind."""

from __future__ import annotations

from pathlib import Path

from fastmcp.client import Client

from robotmcp.config import ServerConfig
from robotmcp.hints import HintContext, HintEngine, load_custom_rules_with_errors
from robotmcp.onboarding import run_doctor
from robotmcp.server import create_server

MALFORMED = '[[hints]]\nid = "broken\nerror_pattern = "x"\n'

ONE_BAD_REGEX = """
[[hints]]
id = "good-one"
error_pattern = "CorpAuthExpiredError"
message = "Refresh the token."

[[hints]]
id = "bad-regex"
error_pattern = "unclosed (group"
message = "never loaded"

[[hints]]
id = "good-two"
error_pattern = "QuotaExceeded"
message = "Wait and retry."
"""


def _write_hints(root: Path, text: str) -> Path:
    path = root / ".robotframework" / "hints.toml"
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    return path


async def test_malformed_hints_toml_keeps_server_and_builtin_rules(tmp_path: Path) -> None:
    hints_file = _write_hints(tmp_path, MALFORMED)
    rules, errors = load_custom_rules_with_errors(hints_file)
    assert rules == []
    assert len(errors) == 1 and "hints.toml" in errors[0]

    engine = HintEngine(hints_file=hints_file)
    assert engine.load_errors == errors
    hints = engine.diagnose(HintContext(keyword="X", error_text="No keyword with name 'X' found."))
    assert hints and hints[0].id == "rf-keyword-not-found"

    mcp = create_server(ServerConfig(execution_mode="client", workspace_root=tmp_path))
    assert "Custom hints not loaded" in (mcp.instructions or "")
    async with Client(mcp) as client:
        assert {t.name for t in await client.list_tools()} >= {"rf_run", "rf_diagnose"}

    report = run_doctor(root=tmp_path, as_json=True)
    assert report["hints_status"] == "errors"
    assert report["hints_errors"] == errors


def test_one_bad_regex_is_skipped_with_location(tmp_path: Path) -> None:
    hints_file = _write_hints(tmp_path, ONE_BAD_REGEX)
    rules, errors = load_custom_rules_with_errors(hints_file)
    assert [r.id for r in rules] == ["good-one", "good-two"]
    assert len(errors) == 1
    assert "bad-regex" in errors[0]
    assert "hints.toml:8" in errors[0]
    assert "error_pattern" in errors[0]

    engine = HintEngine(hints_file=hints_file)
    hints = engine.diagnose(HintContext(keyword="Call", error_text="QuotaExceeded"))
    assert hints[0].id == "good-two"


def test_static_kind_excludes_runtime_only_rules() -> None:
    engine = HintEngine()
    ctx = HintContext(keyword="analyze code", error_text="Variable '${token}' not found.")
    runtime_ids = [h.id for h in engine.diagnose(ctx, kind="runtime")]
    static_ids = [h.id for h in engine.diagnose(ctx, kind="static")]
    assert "rf-variable-not-found" in runtime_ids
    assert "rf-variable-not-found" not in static_ids
    assert "rf-variable-not-found-static" in static_ids


def test_custom_rules_apply_to_both_kinds_by_default(tmp_path: Path) -> None:
    hints_file = _write_hints(tmp_path, ONE_BAD_REGEX)
    engine = HintEngine(hints_file=hints_file)
    ctx = HintContext(keyword="x", error_text="CorpAuthExpiredError")
    assert engine.diagnose(ctx, kind="static")[0].id == "good-one"
    assert engine.diagnose(ctx, kind="runtime")[0].id == "good-one"

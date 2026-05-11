"""Dry-run subprocess timeout returns structured diagnostic fields."""

from __future__ import annotations

import subprocess

import pytest

from robotmcp.components.execution.execution_coordinator import ExecutionCoordinator
from robotmcp.components.execution import suite_execution_service as ses_mod


def _minimal_suite(root) -> str:
    p = root / "t.robot"
    p.write_text(
        "*** Settings ***\n"
        "Library    BuiltIn\n"
        "*** Test Cases ***\n"
        "T\n"
        "    Log    x\n",
        encoding="utf-8",
    )
    return str(p)


@pytest.mark.asyncio
async def test_dry_run_subprocess_timeout_payload_includes_streams(tmp_path, monkeypatch):
    long_out = "O" * 9000
    long_err = "E" * 9000

    def fake_run(cmd, **_kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout=3, output=long_out, stderr=long_err)

    monkeypatch.setattr(ses_mod.subprocess, "run", fake_run)
    suite = _minimal_suite(tmp_path)
    ec = ExecutionCoordinator()
    result = await ec.run_suite_dry_run_from_file(
        suite,
        execution_options={"dry_run_timeout": 3},
    )
    assert result.get("timeout") is True
    assert result.get("timeout_seconds") == 3
    assert isinstance(result.get("command"), list)
    assert "-m" in result["command"] and "robot" in result["command"]
    assert result.get("cwd")
    out = result.get("timeout_stdout_tail") or ""
    err = result.get("timeout_stderr_tail") or ""
    assert len(out) == ses_mod._DRY_RUN_TIMEOUT_OUTPUT_TAIL_MAX
    assert len(err) == ses_mod._DRY_RUN_TIMEOUT_OUTPUT_TAIL_MAX
    assert out == long_out[-ses_mod._DRY_RUN_TIMEOUT_OUTPUT_TAIL_MAX :]
    assert err == long_err[-ses_mod._DRY_RUN_TIMEOUT_OUTPUT_TAIL_MAX :]
    assert "Dry run subprocess timed out" in (result.get("error") or "")

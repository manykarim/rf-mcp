import os
from pathlib import Path

import pytest

from robotmcp.components.execution.suite_execution_service import SuiteExecutionService
from robotmcp.models.config_models import ExecutionConfig


def _service():
    return SuiteExecutionService(ExecutionConfig())


def test_normalize_path_windows_to_wsl_posix():
    svc = _service()
    win_path = r"C:\\workspace\\rf-mcp\\tests\\suite.robot"
    normalized = svc._normalize_path(win_path)
    # Only translate on posix/WSL hosts
    if os.name == "posix":
        assert normalized == "/mnt/c/workspace/rf-mcp/tests/suite.robot"
    else:  # pragma: no cover - Windows CI
        assert normalized == win_path


def test_rewrite_browser_import_keeps_directory(tmp_path: Path):
    svc = _service()
    suite_path = tmp_path / "sample.robot"
    suite_path.write_text("*** Settings ***\nLibrary    Browser\n", encoding="utf-8")

    rewritten = svc._maybe_rewrite_browser_import(str(suite_path))

    # Rewritten file should live alongside the original to preserve relative imports
    assert Path(rewritten).parent == suite_path.parent
    content = Path(rewritten).read_text(encoding="utf-8")
    assert "pause_on_failure=False" in content


@pytest.mark.asyncio
async def test_run_robot_process_blocking_fallback(monkeypatch, tmp_path: Path):
    """Force the blocking runner to verify it exits quickly and returns output."""
    monkeypatch.setenv("ROBOTMCP_FORCE_SYNC_ROBOT", "1")
    svc = _service()

    suite_file = tmp_path / "mini.robot"
    suite_file.write_text(
        "*** Test Cases ***\nMini\n    Log    hello from robotmcp\n",
        encoding="utf-8",
    )

    rc, stdout, stderr, syslog = await svc._run_robot_process(
        ["--output", "NONE", "--report", "NONE", "--log", "NONE", str(suite_file)],
        timeout=15,
        execution_id="test",
    )

    assert rc == 0
    assert syslog.endswith("robot_syslog_test.log")

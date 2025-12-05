import os
import pytest

from robotmcp.components.execution.suite_execution_service import SuiteExecutionService
from robotmcp.server import execution_engine
from robotmcp.models.config_models import ExecutionConfig


def test_normalize_path_windows_to_wsl():
    service = SuiteExecutionService(ExecutionConfig())
    win_path = r"C:\\Users\\tester\\suite.robot"
    normalized = service._normalize_path(win_path)
    if os.name == "posix":
        assert normalized.startswith("/mnt/c/Users/tester/suite.robot")
    else:
        assert normalized == win_path


@pytest.mark.asyncio
async def test_run_suite_dry_run_missing_file_returns_fast_error():
    missing_path = r"C:\\totally_missing\\no_suite_here.robot"
    result = await execution_engine.run_suite_dry_run_from_file(
        suite_file_path=missing_path,
        validation_level="standard",
        include_warnings=True,
    )
    assert result["success"] is False
    assert result.get("error_type") == "file_not_found"
    assert "not found" in result.get("error", "").lower()

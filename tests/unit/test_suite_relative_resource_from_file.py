"""Regression: on-disk suite files must keep suite-relative Resource paths.

Uses a synthetic layout only (no product-specific names).
"""

from __future__ import annotations

import pytest

from robotmcp.components.execution.execution_coordinator import ExecutionCoordinator


def _write_relative_resource_fixture(root) -> str:
    res = root / "Resources" / "Common.robot"
    res.parent.mkdir(parents=True, exist_ok=True)
    res.write_text(
        "*** Keywords ***\n"
        "Common Keyword\n"
        "    Log    hello from common\n",
        encoding="utf-8",
    )
    suite = root / "TestSuites" / "SmokeLike.robot"
    suite.parent.mkdir(parents=True, exist_ok=True)
    suite.write_text(
        "*** Settings ***\n"
        "Resource    ../Resources/Common.robot\n"
        "\n"
        "*** Test Cases ***\n"
        "Example\n"
        "    Common Keyword\n",
        encoding="utf-8",
    )
    return str(suite.resolve())


@pytest.mark.asyncio
async def test_dry_run_from_file_resolves_relative_resource(tmp_path):
    suite_path = _write_relative_resource_fixture(tmp_path)
    ec = ExecutionCoordinator()
    result = await ec.run_suite_dry_run_from_file(suite_path)

    assert result.get("return_code") == 0, result
    assert result.get("success") is True
    vr = result.get("validation_results", {})
    issue_types = {i.get("type") for i in vr.get("issues", [])}
    assert "resource_error" not in issue_types
    assert "missing_keyword" not in issue_types
    assert vr.get("imports_valid") is True
    assert result.get("suite_info", {}).get("test_count") == 1


@pytest.mark.asyncio
async def test_full_run_from_file_resolves_relative_resource(tmp_path):
    suite_path = _write_relative_resource_fixture(tmp_path)
    ec = ExecutionCoordinator()
    result = await ec.run_suite_execution_from_file(suite_path)

    rc = result.get("return_code")
    if rc is None:
        rc = (result.get("execution_details") or {}).get("return_code")
    assert rc == 0, result
    assert result.get("success") is True
    stats = result.get("statistics", {})
    assert stats.get("total") == 1
    assert stats.get("failed") == 0


@pytest.mark.asyncio
async def test_dry_run_from_file_missing_file_returns_error(tmp_path):
    missing = str(tmp_path / "does_not_exist.robot")
    ec = ExecutionCoordinator()
    result = await ec.run_suite_dry_run_from_file(missing)

    assert result.get("success") is False
    assert "file_not_found" == result.get("error_type")
    assert result.get("session_id"), "error payload must include session_id"
    assert result.get("suite_file_path") == missing


@pytest.mark.asyncio
async def test_full_run_from_file_missing_file_returns_error(tmp_path):
    missing = str(tmp_path / "does_not_exist.robot")
    ec = ExecutionCoordinator()
    result = await ec.run_suite_execution_from_file(missing)

    assert result.get("success") is False
    assert "file_not_found" == result.get("error_type")
    assert result.get("session_id"), "error payload must include session_id"
    assert result.get("suite_file_path") == missing

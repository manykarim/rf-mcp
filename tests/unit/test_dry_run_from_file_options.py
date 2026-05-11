"""Dry-from-file: execution_options and subprocess timeout reach Robot."""

from __future__ import annotations

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
async def test_dry_run_from_file_default_subprocess_timeout(tmp_path, monkeypatch):
    captured = {}

    def fake_run(*_args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(ses_mod.subprocess, "run", fake_run)
    suite = _minimal_suite(tmp_path)
    ec = ExecutionCoordinator()
    await ec.run_suite_dry_run_from_file(suite)
    assert captured["timeout"] == 180


@pytest.mark.asyncio
async def test_dry_run_from_file_passes_subprocess_timeout(tmp_path, monkeypatch):
    captured = {}

    def fake_run(*_args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(ses_mod.subprocess, "run", fake_run)
    suite = _minimal_suite(tmp_path)
    ec = ExecutionCoordinator()
    await ec.run_suite_dry_run_from_file(
        suite,
        execution_options={"dry_run_timeout": 97},
    )
    assert captured["timeout"] == 97


@pytest.mark.asyncio
async def test_dry_run_from_file_timeout_alias(tmp_path, monkeypatch):
    captured = {}

    def fake_run(*_args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(ses_mod.subprocess, "run", fake_run)
    suite = _minimal_suite(tmp_path)
    ec = ExecutionCoordinator()
    await ec.run_suite_dry_run_from_file(suite, execution_options={"timeout": 88})
    assert captured["timeout"] == 88


@pytest.mark.asyncio
async def test_dry_run_from_file_forwards_cli_options(tmp_path, monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = list(cmd)

        class R:
            returncode = 0
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr(ses_mod.subprocess, "run", fake_run)
    suite = _minimal_suite(tmp_path)
    ec = ExecutionCoordinator()
    await ec.run_suite_dry_run_from_file(
        suite,
        execution_options={
            "variables": {"SUT_NAME": "acme"},
            "include_tags": ["solo"],
            "exclude_tags": ["slow"],
            "test": "T",
            "pythonpath": [str(tmp_path)],
            "loglevel": "ERROR",
        },
    )
    cmd = captured["cmd"]
    assert isinstance(cmd, list), cmd
    assert "--variable" in cmd
    assert "SUT_NAME:acme" in cmd
    assert cmd.count("--include") >= 1 and "solo" in cmd
    assert cmd.count("--exclude") >= 1 and "slow" in cmd
    assert "--test" in cmd and "T" in cmd
    assert "--pythonpath" in cmd
    assert "--loglevel" in cmd and "ERROR" in cmd

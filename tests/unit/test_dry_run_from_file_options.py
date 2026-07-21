"""Dry-from-file: execution_options and subprocess timeout reach Robot.

The dry-run spawns via subprocess.Popen (change: fix-mcp-subprocess-stdin-deadlock) with a
non-inheriting stdin; the effective timeout is applied to communicate(), and the RF CLI
options land on the spawned command.
"""

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


def _fake_popen(captured):
    class FakeProc:
        returncode = 0
        pid = 1

        def communicate(self, timeout=None):
            captured["timeout"] = timeout
            return ("", "")

        def kill(self):
            pass

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["popen_kwargs"] = kwargs
        return FakeProc()

    return fake_popen


@pytest.mark.asyncio
async def test_dry_run_from_file_default_subprocess_timeout(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(ses_mod.subprocess, "Popen", _fake_popen(captured))
    ec = ExecutionCoordinator()
    await ec.run_suite_dry_run_from_file(_minimal_suite(tmp_path))
    assert captured["timeout"] == 180
    # the deadlock fix: child never inherits the server's stdin
    import subprocess
    assert captured["popen_kwargs"].get("stdin") is subprocess.DEVNULL


@pytest.mark.asyncio
async def test_dry_run_from_file_passes_subprocess_timeout(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(ses_mod.subprocess, "Popen", _fake_popen(captured))
    ec = ExecutionCoordinator()
    await ec.run_suite_dry_run_from_file(
        _minimal_suite(tmp_path), execution_options={"dry_run_timeout": 97}
    )
    assert captured["timeout"] == 97


@pytest.mark.asyncio
async def test_dry_run_from_file_timeout_alias(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(ses_mod.subprocess, "Popen", _fake_popen(captured))
    ec = ExecutionCoordinator()
    await ec.run_suite_dry_run_from_file(
        _minimal_suite(tmp_path), execution_options={"timeout": 88}
    )
    assert captured["timeout"] == 88


@pytest.mark.asyncio
async def test_dry_run_from_file_forwards_cli_options(tmp_path, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(ses_mod.subprocess, "Popen", _fake_popen(captured))
    ec = ExecutionCoordinator()
    await ec.run_suite_dry_run_from_file(
        _minimal_suite(tmp_path),
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

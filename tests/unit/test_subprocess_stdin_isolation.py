"""Tests for change: fix-mcp-subprocess-stdin-deadlock.

The stdio MCP server holds a permanent pending read on its stdin; any child that inherits
that handle deadlocks on Windows. So serving-time subprocesses must be spawned with a
non-inheriting stdin, and dry-run timeouts must reap the whole process tree.
"""
from __future__ import annotations

import asyncio
import subprocess
import types

import pytest


def _svc():
    from robotmcp.components.execution.suite_execution_service import SuiteExecutionService
    # getattr-with-default config is enough (DRY_RUN_TIMEOUT etc. fall back)
    return SuiteExecutionService(types.SimpleNamespace())


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── §1 dry-run subprocess must NOT inherit stdin ────────────────────────────
def test_dry_run_spawn_uses_devnull_stdin(monkeypatch):
    from robotmcp.components.execution import suite_execution_service as ses

    captured = {}

    class FakeProc:
        returncode = 0
        pid = 4242
        def communicate(self, timeout=None):
            return ("dry-run ok", "")
        def kill(self):
            pass

    def fake_popen(cmd, **kwargs):
        captured.update(kwargs)
        return FakeProc()

    monkeypatch.setattr(ses.subprocess, "Popen", fake_popen)
    rc, out, err = _run(_svc()._execute_rf_dry_run("/tmp/suite.robot", {}))
    assert rc == 0
    assert captured.get("stdin") is subprocess.DEVNULL, (
        "dry-run child must not inherit the server's stdin"
    )


# ── §2 timeout reaps the whole process tree, no hang ────────────────────────
def test_dry_run_timeout_reaps_process_tree(monkeypatch):
    from robotmcp.components.execution import suite_execution_service as ses

    reaped = {"n": 0}
    monkeypatch.setattr(ses, "_kill_process_tree", lambda proc: reaped.__setitem__("n", reaped["n"] + 1))

    class HangingProc:
        returncode = None
        pid = 99
        _calls = 0
        def communicate(self, timeout=None):
            HangingProc._calls += 1
            if HangingProc._calls == 1:
                raise subprocess.TimeoutExpired(cmd="robot", timeout=timeout)
            return ("", "")  # post-kill drain returns promptly
        def kill(self):
            pass

    monkeypatch.setattr(ses.subprocess, "Popen", lambda cmd, **kw: HangingProc())

    with pytest.raises(Exception) as ei:
        _run(_svc()._execute_rf_dry_run("/tmp/suite.robot", {"dry_run_timeout": 1}))
    assert "timed out" in str(ei.value).lower()
    assert reaped["n"] >= 1, "the process tree must be reaped on timeout"


def test_kill_process_tree_never_raises(monkeypatch):
    from robotmcp.components.execution import suite_execution_service as ses

    class BadProc:
        pid = 123456789
        def kill(self):
            raise RuntimeError("boom")

    # os.killpg / taskkill will fail on a bogus pid; kill() raises too — must be swallowed.
    ses._kill_process_tree(BadProc())  # no exception == pass


# ── §3 the latent library-check subprocess must not inherit stdin ───────────
def test_pip_list_check_uses_devnull_stdin(monkeypatch):
    from robotmcp.utils import library_checker as lc

    captured = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(returncode=0, stdout="robotframework==7.0\n", stderr="")

    monkeypatch.setattr(lc.subprocess, "run", fake_run)
    lc.LibraryAvailabilityChecker().check_pip_package_installed("robotframework")
    assert captured.get("stdin") is subprocess.DEVNULL

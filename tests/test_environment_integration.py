"""A global rf-mcp runs a project's tests in the project's environment (project-independent-server, task 2.4).

Reproduces the 2026-11-11 measurements with a real uv: a project venv holding Robot Framework and RequestsLibrary but
no robotcode gets rf-mcp's robotcode layered on, analysis and runs see RequestsLibrary, a REPL session works across
calls, and a venv that pins robotcode 1.5.0 keeps it. Needs uv and network access on first run; skipped without uv.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from fastmcp.client import Client

from robotmcp.config import ServerConfig
from robotmcp.server import create_server

UV = shutil.which("uv")
pytestmark = pytest.mark.skipif(UV is None, reason="needs uv")

SUITE = """*** Settings ***
Library    RequestsLibrary

*** Test Cases ***
Sum
    ${s}=    Evaluate    1+1
    Should Be Equal As Integers    ${s}    2
"""


def _python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _project(path: Path, *packages: str) -> Path:
    path.mkdir(parents=True)
    (path / "robot.toml").write_text("", encoding="utf-8")
    (path / "t.robot").write_text(SUITE, encoding="utf-8")
    subprocess.run([UV, "venv", "-q", "--python", "3.12", str(path / ".venv")], check=True)
    subprocess.run([UV, "pip", "install", "-q", "--python", str(_python(path / ".venv")), *packages], check=True)
    return path


def _robotcode_version(venv: Path) -> str | None:
    proc = subprocess.run(
        [str(_python(venv)), "-c", "import importlib.metadata as m; print(m.version('robotcode'))"],
        capture_output=True, text=True,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


@pytest.fixture(scope="module")
def plain_project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _project(tmp_path_factory.mktemp("global") / "shop", "robotframework", "robotframework-requests")


async def test_analyze_run_and_repl_see_the_projects_library(plain_project: Path) -> None:
    async with Client(create_server(ServerConfig(execution_mode="server", workspace_root=plain_project))) as client:
        analyzed = (await client.call_tool("rf_analyze", {})).data
        assert "warning" not in analyzed, analyzed.get("warning")
        assert not analyzed.get("has_errors"), analyzed

        ran = (await client.call_tool("rf_run", {"timeout_seconds": 300})).data
        assert ran.get("exit_code") == 0, ran

        started = (await client.call_tool("rf_repl", {"action": "start"})).data
        sid = started["session_id"]
        try:
            imported = (await client.call_tool("rf_repl", {"action": "send", "session_id": sid,
                                                           "statement": "Import Library    RequestsLibrary"})).data
            assert imported["status"] == "passed", imported
            evaluated = (await client.call_tool("rf_repl", {"action": "send", "session_id": sid,
                                                            "statement": "${r}=    Evaluate    6*7"})).data
            assert evaluated["status"] == "passed" and "${r} = 42" in evaluated["output"], evaluated
        finally:
            await client.call_tool("rf_repl", {"action": "stop", "session_id": sid})
        listed = (await client.call_tool("rf_repl", {"action": "list"})).data
        assert sid not in str(listed.get("sessions") or [])

    assert _robotcode_version(plain_project / ".venv") is None


async def test_a_project_pinning_an_old_robotcode_keeps_it(tmp_path: Path) -> None:
    project = _project(tmp_path / "legacy", "robotframework", "robotframework-requests", "robotcode==1.5.0")
    async with Client(create_server(ServerConfig(execution_mode="server", workspace_root=project))) as client:
        analyzed = (await client.call_tool("rf_analyze", {})).data
        assert not analyzed.get("has_errors") and "warning" not in analyzed, analyzed
    assert _robotcode_version(project / ".venv") == "1.5.0"


def test_doctor_reports_the_layered_environment(plain_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Task 5.2: doctor warms the layer and says how long it took."""
    from robotmcp.onboarding import run_doctor
    from robotmcp.runner.environment import EnvironmentResolver

    environ = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    environ["PATH"] = os.pathsep.join(p for p in environ.get("PATH", "").split(os.pathsep) if ".venv" not in p)
    report = run_doctor(root=plain_project, resolver=EnvironmentResolver(environ=environ))
    out = capsys.readouterr().out
    assert report["ok"] is True and report["environment"]["layered"] is True, report["environment"]
    assert report["environment"]["source"] == "venv" and report["environment"]["python"].startswith("3.12")
    assert "added by uv, ready in" in out and "Project Root" in out

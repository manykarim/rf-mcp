"""Opt-in packaging smoke test (change: uv-tool-install-onboarding §6.6).

Builds the wheel and `uv tool install`s it into an isolated location, then asserts
the `robotmcp` executable is on PATH and the onboarding CLI runs. Off by default
(needs `uv`, network for deps, and time); enable with ROBOTMCP_PACKAGING_SMOKE=1.
Mirrors experiments/uv-tool-install/probe_toolinstall.sh.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("ROBOTMCP_PACKAGING_SMOKE") != "1",
    reason="set ROBOTMCP_PACKAGING_SMOKE=1 to run the uv tool install packaging smoke",
)

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not available")
def test_uv_tool_install_yields_working_command(tmp_path):
    dist = tmp_path / "dist"
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(dist)],
                   cwd=REPO, check=True, capture_output=True, text=True)
    wheel = next(dist.glob("rf_mcp-*.whl"))

    env = dict(os.environ)
    env["UV_TOOL_DIR"] = str(tmp_path / "tools")
    env["UV_TOOL_BIN_DIR"] = str(tmp_path / "bin")
    Path(env["UV_TOOL_BIN_DIR"]).mkdir(parents=True, exist_ok=True)

    subprocess.run(["uv", "tool", "install", f"{wheel}[api]"],
                   env=env, check=True, capture_output=True, text=True)
    robotmcp = Path(env["UV_TOOL_BIN_DIR"]) / "robotmcp"
    assert robotmcp.exists(), "robotmcp executable not placed on the tool bin dir"

    v = subprocess.run([str(robotmcp), "--version"], env=env,
                       capture_output=True, text=True)
    assert v.returncode == 0 and v.stdout.strip(), v.stderr

    lst = subprocess.run([str(robotmcp), "list"], env=env,
                         capture_output=True, text=True)
    assert lst.returncode == 0 and "Claude Code" in lst.stdout, lst.stderr

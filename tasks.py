"""Developer tasks powered by Invoke."""

from __future__ import annotations

import os
import pathlib
import subprocess
from typing import Iterable

from invoke import task

ROOT = pathlib.Path(__file__).parent.resolve()
ENV = {**os.environ, "UV_PROJECT_ENVIRONMENT": ".venv-wsl"}
RESULTS_DIR = ROOT / "results"


def _run(command: Iterable[str] | str) -> None:
    if isinstance(command, str):
        cmd = command
    else:
        cmd = " ".join(command)
    subprocess.run(cmd, shell=True, check=True, cwd=ROOT, env=ENV)


def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)


@task
def tests(_context):
    """Run test suite without coverage (quick feedback)."""
    _ensure_results_dir()
    _run(["uv", "run", "pytest", "tests/"])


@task
def coverage(_context):
    """Run tests under coverage and generate reports."""
    _ensure_results_dir()
    _run(["uv", "run", "coverage", "erase"])
    _run(
        [
            "uv",
            "run",
            "coverage",
            "run",
            "-m",
            "pytest",
            "tests/",
            "--junitxml=results/pytest.xml",
        ]
    )
    _run(["uv", "run", "coverage", "combine"])
    _run(["uv", "run", "coverage", "report"])
    _run(["uv", "run", "coverage", "html", "-d", "results/htmlcov"])
    _run(["uv", "run", "coverage", "xml", "-o", "results/coverage.xml"])


@task
def lint(_context):
    """Run formatting and type checks."""
    _run(["uv", "run", "black", "--check", "src", "tests"])
    _run(["uv", "run", "mypy", "src"])


@task
def build(_context):
    """Build distribution artifacts."""
    _run(["uv", "build"])

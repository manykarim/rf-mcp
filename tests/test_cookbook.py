"""Run the skill cookbook suites (tests/cookbook) against local fixtures.

Every snippet in the built-in skills' cookbooks is executed here, so a snippet that stops
working fails the build. The suites need libraries rf-mcp itself does not depend on, so the
interpreters are configurable:

- `RF_MCP_COOKBOOK_PYTHON`: Python with Robot Framework and the libraries under test
  (robotframework-requests; robotframework-browser with `rfbrowser init`; robotframework-seleniumlibrary
  with Chrome). Default: the interpreter running pytest.
- `RF_MCP_FIXTURE_PYTHON`: Python with fastapi, uvicorn, python-multipart and strawberry-graphql
  for the ledger API fixture. Default: the interpreter running pytest.

A suite whose libraries or fixture dependencies are missing is skipped, not failed.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

HERE = Path(__file__).resolve().parent
COOKBOOK = HERE / "cookbook"
FIXTURES = HERE / "fixtures"

ROBOT_PY = os.environ.get("RF_MCP_COOKBOOK_PYTHON", sys.executable)
FIXTURE_PY = os.environ.get("RF_MCP_FIXTURE_PYTHON", sys.executable)


def _has_modules(python: str, *modules: str) -> bool:
    code = "import importlib, sys; [importlib.import_module(m) for m in sys.argv[1:]]"
    return subprocess.run([python, "-c", code, *modules], capture_output=True, check=False).returncode == 0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http(url: str, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except OSError:
            time.sleep(0.3)
    raise RuntimeError(f"{url} did not come up")


@contextmanager
def _ledger() -> Iterator[str]:
    if not _has_modules(FIXTURE_PY, "fastapi", "uvicorn", "multipart", "strawberry"):
        pytest.skip("ledger fixture needs fastapi, uvicorn, python-multipart, strawberry-graphql (RF_MCP_FIXTURE_PYTHON)")
    port = _free_port()
    proc = subprocess.Popen(
        [FIXTURE_PY, str(FIXTURES / "ledger" / "ledger_ws.py"), str(port)],
        cwd=FIXTURES / "ledger",
        env={**os.environ, "PYTHONPATH": str(FIXTURES / "ledger")},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_http(f"http://127.0.0.1:{port}/")
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.kill()
        proc.wait(timeout=10)


def _robot(suite: Path, outdir: Path, *variables: str) -> subprocess.CompletedProcess:
    args = [ROBOT_PY, "-m", "robot", "--outputdir", str(outdir), "--console", "dotted"]
    for var in variables:
        args.extend(["-v", var])
    args.append(str(suite))
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        timeout=600,
        check=False,
    )


@pytest.mark.parametrize("suite", ["requests_import.robot", "requests_cookbook.robot", "ws_cookbook.robot"])
def test_requests_cookbook(suite: str, tmp_path: Path) -> None:
    if not _has_modules(ROBOT_PY, "robot", "RequestsLibrary"):
        pytest.skip("needs robotframework-requests (RF_MCP_COOKBOOK_PYTHON)")
    with _ledger() as url:
        res = _robot(COOKBOOK / suite, tmp_path, f"LEDGER_URL:{url}", f"LEDGER_USER:cb-{tmp_path.name[-8:]}")
    assert res.returncode == 0, res.stdout[-4000:] + res.stderr[-2000:]


# --- Browser / SeleniumLibrary locator cookbook (shadow DOM and iframes) -------------------

XORIGIN_PLACEHOLDER = "http://127.0.0.1:8702"


@contextmanager
def _serve(directory: Path) -> Iterator[str]:
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(directory)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_http(f"http://127.0.0.1:{port}/")
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.kill()
        proc.wait(timeout=10)


@contextmanager
def _widget_site(tmp_path: Path) -> Iterator[str]:
    """Serve the widget page and, on a second origin, its cross-origin frame."""
    import shutil

    site = tmp_path / "site"
    shutil.copytree(FIXTURES / "web" / "site", site)
    with _serve(FIXTURES / "web" / "xsite") as xorigin:
        index = site / "index.html"
        index.write_text(index.read_text(encoding="utf-8").replace(XORIGIN_PLACEHOLDER, xorigin), encoding="utf-8")
        with _serve(site) as origin:
            yield f"{origin}/index.html"


@pytest.mark.parametrize(
    ("suite", "library"),
    [
        ("browser_shadow_frames.robot", "Browser"),
        ("browser_closed_shadow.robot", "Browser"),
        ("browser_more.robot", "Browser"),
        ("selenium_shadow_frames.robot", "SeleniumLibrary"),
        ("selenium_cdp_read.robot", "SeleniumLibrary"),
    ],
)
def test_locator_cookbook(suite: str, library: str, tmp_path: Path) -> None:
    if not _has_modules(ROBOT_PY, "robot", library):
        pytest.skip(f"needs {library} (RF_MCP_COOKBOOK_PYTHON)")
    with _widget_site(tmp_path) as url:
        res = _robot(COOKBOOK / suite, tmp_path / "out", f"URL:{url}")
    output = res.stdout + res.stderr
    for marker in MISSING_BROWSER_MARKERS:
        if res.returncode != 0 and marker in output:
            pytest.skip(f"{library} is installed but its browser is not ({marker!r}); run `rfbrowser init` or set PLAYWRIGHT_BROWSERS_PATH")
    assert res.returncode == 0, res.stdout[-4000:] + res.stderr[-2000:]


MISSING_BROWSER_MARKERS = (
    "Executable doesn't exist",
    "rfbrowser init",
    "Unable to obtain driver",
    "cannot find Chrome binary",
)

"""Servers and remote systems the evaluated tasks run against.

- `WebFixture`: the Widget Lab page (tests/fixtures/web/site) plus its cross-origin frame (xsite) on a second origin.
- `LedgerFixture`: the ledger REST/SOAP/GraphQL API (tests/fixtures/ledger) plus one server per injected bug.
- `Demoshop`: the shared workshop shop (https://demoshop.makrocode.de), isolated per run by `X-Workshop-Space`.
- `demoshop_source()`: a checkout of github.com/manykarim/ai-engineering-robotframework (cached clone).
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .env import FIXTURE_PY, FIXTURES_DIR, has_modules

XORIGIN_PLACEHOLDER = "http://127.0.0.1:8702"
SHOP_URL = os.environ.get("RF_EVAL_SHOP_URL", "https://demoshop.makrocode.de")
DEMOSHOP_REPO = os.environ.get("RF_EVAL_DEMOSHOP_REPO", "https://github.com/manykarim/ai-engineering-robotframework")
CACHE_DIR = Path(os.environ.get("RF_EVAL_CACHE", Path(tempfile.gettempdir()) / "rf-mcp-agent-eval" / "cache"))


class FixtureError(RuntimeError):
    """A fixture or seed source is unavailable."""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url: str, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except OSError:
            time.sleep(0.3)
    raise FixtureError(f"{url} did not come up within {timeout}s")


@dataclass
class _Procs:
    procs: list[subprocess.Popen] = field(default_factory=list)

    def spawn(self, args: list[str], **kw) -> None:
        self.procs.append(subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw))

    def stop(self) -> None:
        for p in self.procs:
            p.kill()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        self.procs.clear()


class WebFixture(_Procs):
    """Serve the widget page; the cross-origin frame src is rewritten to the served xsite origin."""

    def __init__(self, workdir: Path):
        super().__init__()
        self.workdir = workdir
        self.url = "http://127.0.0.1:<port>/index.html"

    def _serve(self, directory: Path) -> str:
        port = free_port()
        self.spawn([sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(directory)])
        url = f"http://127.0.0.1:{port}"
        wait_http(url + "/")
        return url

    def start(self) -> "WebFixture":
        site = self.workdir / "web_site"
        if site.exists():
            shutil.rmtree(site)
        shutil.copytree(FIXTURES_DIR / "web" / "site", site)
        xorigin = self._serve(FIXTURES_DIR / "web" / "xsite")
        index = site / "index.html"
        index.write_text(index.read_text(encoding="utf-8").replace(XORIGIN_PLACEHOLDER, xorigin), encoding="utf-8")
        self.url = self._serve(site) + "/index.html"
        return self


class LedgerFixture(_Procs):
    """The ledger API and one bug server per name in `bugs` (see tests/fixtures/ledger/ledger_api.py)."""

    def __init__(self, bugs: list[str]):
        super().__init__()
        self.bugs = bugs
        self.url = "http://127.0.0.1:<port>"
        self.bug_urls: dict[str, str] = {b: f"http://127.0.0.1:<port-{b}>" for b in bugs}

    def _spawn_ledger(self, bug: str) -> str:
        port = free_port()
        ledger = FIXTURES_DIR / "ledger"
        self.spawn([FIXTURE_PY, str(ledger / "ledger_ws.py"), str(port), bug], cwd=ledger,
                   env={**os.environ, "PYTHONPATH": str(ledger)})
        return f"http://127.0.0.1:{port}"

    def start(self) -> "LedgerFixture":
        if not has_modules(FIXTURE_PY, "fastapi", "uvicorn", "multipart", "strawberry"):
            raise FixtureError("the ledger fixture needs fastapi, uvicorn, python-multipart and strawberry-graphql "
                               f"in RF_MCP_FIXTURE_PYTHON (now {FIXTURE_PY})")
        self.url = self._spawn_ledger("")
        self.bug_urls = {b: self._spawn_ledger(b) for b in self.bugs}
        for url in [self.url, *self.bug_urls.values()]:
            wait_http(url + "/")
        return self


class Demoshop:
    """Client for the workshop shop's admin API; every run uses its own `X-Workshop-Space`."""

    UA = {"User-Agent": "curl/8.0"}  # Cloudflare blocks the Python-urllib user agent

    def __init__(self, url: str = SHOP_URL):
        self.url = url

    def api(self, method: str, path: str, space: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={**self.UA, "X-Workshop-Space": space, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b"{}")

    def set_preset(self, space: str, preset: str) -> None:
        self.api("POST", "/api/workshop/reset", space)
        if preset != "clean":
            self.api("POST", "/api/workshop/preset", space, {"preset": preset})

    def reset(self, space: str) -> None:
        try:
            self.api("POST", "/api/workshop/reset", space)
        except OSError:
            pass

    def run_status(self, env, root: Path, preset: str, select=(), tag: str = "grade") -> dict:
        """Run (a selection of) the suite in a fresh space with `preset`; return {longname: status}."""
        from .graders import run_suite  # local import: graders does not depend on fixtures

        space = f"ev-{tag}-{uuid.uuid4().hex[:8]}"
        try:
            self.set_preset(space, preset)
            sel: list[str] = []
            for s in select:
                sel += ["-bl", s]
            return run_suite(env, root, *sel, key="fullName", extra_env={"SHOP_SPACE": space})
        except OSError as e:
            return {"_error": repr(e)}
        finally:
            self.reset(space)


_CLONE_LOCK = threading.Lock()
_CLONE_ERROR: list[FixtureError] = []


def demoshop_source() -> Path:
    """A checkout of the workshop repository: `RF_EVAL_DEMOSHOP_SRC`, else a cached shallow clone of main."""
    with _CLONE_LOCK:  # parallel runs share one clone; a failed clone is not retried for every run
        if _CLONE_ERROR:
            raise _CLONE_ERROR[0]
        try:
            return _demoshop_source()
        except FixtureError as e:
            _CLONE_ERROR.append(e)
            raise


def _demoshop_source() -> Path:
    local = os.environ.get("RF_EVAL_DEMOSHOP_SRC")
    if local:
        if not (Path(local) / "robot.toml").is_file():
            raise FixtureError(f"RF_EVAL_DEMOSHOP_SRC={local} is not a checkout of {DEMOSHOP_REPO} (no robot.toml)")
        return Path(local)
    dest = CACHE_DIR / "ai-engineering-robotframework"
    if (dest / "robot.toml").is_file():
        return dest
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_DIR / f"clone-{uuid.uuid4().hex[:6]}"
    try:
        p = subprocess.run(["git", "clone", "--depth", "1", "--branch", "main", DEMOSHOP_REPO, str(tmp)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise FixtureError(f"cannot clone {DEMOSHOP_REPO}: {e}. Set RF_EVAL_DEMOSHOP_SRC to a local checkout.") from e
    if p.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise FixtureError(f"cannot clone {DEMOSHOP_REPO} (offline?): {(p.stderr or p.stdout).strip()[-300:]}. "
                           "Set RF_EVAL_DEMOSHOP_SRC to a local checkout.")
    tmp.rename(dest)
    return dest

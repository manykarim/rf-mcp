from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)


def _wait_for_server(url: str, timeout: float = 20.0, process: subprocess.Popen | None = None) -> None:
    import urllib.request
    import urllib.error

    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url) as resp:
                if resp.status < 500:
                    return
        except urllib.error.URLError:
            time.sleep(0.2)
        if process is not None and process.poll() is not None:
            raise RuntimeError("Frontend server exited prematurely")
    raise RuntimeError(f"Frontend server not reachable at {url}")


@pytest.fixture(scope="module")
def frontend_process():
    env = os.environ.copy()
    cmd = [
        str(PYTHON),
        "-m",
        "robotmcp.frontend.devserver",
        "--host",
        "127.0.0.1",
        "--port",
        "8065",
    ]
    proc = subprocess.Popen(
        cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    try:
        _wait_for_server("http://127.0.0.1:8065/", process=proc)
    except Exception:
        proc.terminate()
        stdout, _ = proc.communicate(timeout=5)
        raise RuntimeError(f"Failed to start devserver. Output:\n{stdout}")

    yield proc

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_frontend_renders_dashboard(frontend_process):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            _wait_for_server("http://127.0.0.1:8065/", process=frontend_process)
            page.goto("http://127.0.0.1:8065/", wait_until="domcontentloaded")
            page.wait_for_selector("#summary-panel", timeout=5000)

            cards = page.query_selector_all(".session-card")
            if not cards:  # pragma: no cover - devserver fixture normally seeds sessions
                pytest.skip("Dev server did not expose sample sessions")

            cards[0].click()
            page.wait_for_function(
                "() => document.querySelectorAll('#session-meta .meta-chip').length > 0",
                timeout=10_000,
            )

            def chip_text(label: str) -> str:
                el = page.wait_for_selector(
                    f"#session-meta .meta-chip:has-text('{label}')",
                    timeout=10_000,
                )
                return el.text_content().strip()

            summary_values = {
                "Browser": chip_text("Browser"),
                "Current URL": chip_text("Current URL"),
                "Libraries": chip_text("Libraries"),
                "Active Library": chip_text("Active Library"),
            }

            for label, value in summary_values.items():
                assert "—" not in value, f"{label} chip is empty"
                assert "unknown" not in value.lower(), f"{label} chip shows unknown"
            assert "about:blank" not in summary_values["Current URL"].lower()
        finally:
            browser.close()



def test_frontend_session_summary(frontend_process):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            _wait_for_server("http://127.0.0.1:8065/", process=frontend_process)
            page.goto("http://127.0.0.1:8065/", wait_until="domcontentloaded")
            page.wait_for_selector("#summary-panel", timeout=5000)

            cards = page.query_selector_all(".session-card")
            if not cards:  # pragma: no cover - devserver fixture normally seeds sessions
                pytest.skip("Dev server did not expose sample sessions")

            card = cards[0]
            session_label = card.text_content().strip()
            card.click()

            page.wait_for_function(
                "() => document.querySelectorAll('#session-steps .step-card').length > 0",
                timeout=10_000,
            )

            steps = page.query_selector_all("#session-steps .step-card")
            assert steps, "Expected step cards to render for session summary"
            titles = [step.query_selector(".step-label").text_content() for step in steps]
            assert any("Open Browser" in title for title in titles)
            assert any("Go To" in title or "Go to" in title for title in titles)

            meta_summary = page.inner_text("#session-meta")
            assert session_label[:6] in meta_summary
        finally:
            browser.close()

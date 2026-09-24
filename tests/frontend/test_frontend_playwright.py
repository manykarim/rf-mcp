from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Error as PlaywrightError  # type: ignore
from playwright.sync_api import expect  # type: ignore

# These tests drive a LIVE dashboard that re-renders itself from streamed events, so
# `query_selector_all` + `.click()` is unsafe: the returned ElementHandles are bound to
# the DOM as it was at query time, and a re-render between the query and the click
# detaches them -
#     Error: ElementHandle.click: Element is not attached to the DOM
# which is what made this file the single largest source of false-red CI across two PRs.
# Locators re-resolve the selector at every action and auto-wait for actionability, so
# they survive a re-render. Prefer `page.locator(...)` over `query_selector*` here.

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
        cmd,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        _wait_for_server("http://127.0.0.1:8065/", process=proc)
    except Exception:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise RuntimeError("Failed to start devserver.")

    yield proc

    if sys.platform == "win32":
        proc.terminate()
    else:
        proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_frontend_renders_dashboard(frontend_process):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except PlaywrightError as exc:  # pragma: no cover - environment dependency
            if "Executable doesn't exist" in str(exc):
                pytest.skip("Playwright browsers unavailable on host; install via `playwright install`.")
            raise
        page = browser.new_page()
        try:
            _wait_for_server("http://127.0.0.1:8065/", process=frontend_process)
            page.goto("http://127.0.0.1:8065/", wait_until="domcontentloaded")
            page.wait_for_selector("#summary-panel", timeout=5000)

            cards = page.locator(".session-card")
            if cards.count() == 0:  # pragma: no cover - devserver normally seeds sessions
                pytest.skip("Dev server did not expose sample sessions")

            cards.first.click()
            page.wait_for_function(
                "() => document.querySelectorAll('#session-meta .meta-chip').length > 0",
                timeout=10_000,
            )

            def chip_text(label: str) -> str:
                """Value of the chip whose LABEL is exactly `label`.

                `:has-text('Browser')` was matching THREE chips - "Browser",
                "Active Library" (value "browser") and "Libraries" (value contains
                "Browser") - because :has-text is a case-insensitive substring match.
                `.first` then resolved to "Active Library", so this helper returned the
                wrong chip's text and the Browser chip was never asserted on at all.
                Proven by mutation: forcing browser_type="unknown" in bridge.py did NOT
                fail this test before the fix.

                Each chip is `div.meta-chip > span(label) + strong(value)`, so anchor on
                the span's FULL text and read the sibling strong.
                """
                chip = page.locator("#session-meta .meta-chip").filter(
                    has=page.locator(
                        "span", has_text=re.compile(rf"^\s*{re.escape(label)}\s*$")
                    )
                )
                expect(chip).to_have_count(1, timeout=10_000)
                value = chip.locator("strong")
                expect(value).to_be_visible(timeout=10_000)
                return (value.text_content() or "").strip()

            summary_values = {
                "Browser": chip_text("Browser"),
                "Current URL": chip_text("Current URL"),
                "Libraries": chip_text("Libraries"),
                "Active Library": chip_text("Active Library"),
            }

            for label, value in summary_values.items():
                # An EMPTY chip previously slipped through every check below - "—" is
                # not in "", and neither is "unknown" - so the assertion whose message
                # reads "chip is empty" could not actually catch an empty chip. Assert
                # the content exists before asserting what it is not.
                assert value, f"{label} chip rendered no text at all"
                assert "—" not in value, f"{label} chip is empty"
                assert "unknown" not in value.lower(), f"{label} chip shows unknown"
            # Guards the fabricated-metadata bug (frontend-dashboard-browser-state-fidelity):
            # a real session must not report the placeholder URL.
            assert "about:blank" not in summary_values["Current URL"].lower()
        finally:
            browser.close()



def test_frontend_session_summary(frontend_process):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except PlaywrightError as exc:  # pragma: no cover - environment dependency
            if "Executable doesn't exist" in str(exc):
                pytest.skip("Playwright browsers unavailable on host; install via `playwright install`.")
            raise
        page = browser.new_page()
        try:
            _wait_for_server("http://127.0.0.1:8065/", process=frontend_process)
            page.goto("http://127.0.0.1:8065/", wait_until="domcontentloaded")
            page.wait_for_selector("#summary-panel", timeout=5000)

            cards = page.locator(".session-card")
            if cards.count() == 0:  # pragma: no cover - devserver normally seeds sessions
                pytest.skip("Dev server did not expose sample sessions")

            card = cards.first
            expect(card).to_be_visible(timeout=10_000)
            session_label = (card.text_content() or "").strip()
            # An empty label would make the identity assertion at the end vacuous
            # (""[:6] == "" and "" is in every string), so the test would "pass" while
            # proving nothing about which session was opened.
            assert session_label, "Session card rendered no label"
            card.click()

            steps = page.locator("#session-steps .step-card")
            # Auto-retrying: replaces the wait_for_function + re-query pair, and keeps
            # the assertion that step cards actually render.
            expect(steps.first).to_be_visible(timeout=10_000)
            assert steps.count() > 0, "Expected step cards to render for session summary"

            # Resolved in one shot rather than by iterating handles that can detach.
            titles = steps.locator(".step-label").all_text_contents()
            assert titles, "Step cards rendered without any .step-label"
            assert any("Open Browser" in title for title in titles), titles
            assert any("Go To" in title or "Go to" in title for title in titles), titles

            # The selected session's identity must reach the meta panel - this is what
            # proves the click loaded THAT session rather than leaving stale content.
            meta_summary = page.inner_text("#session-meta")
            assert session_label[:6] in meta_summary, (
                f"selected session {session_label!r} not reflected in #session-meta: "
                f"{meta_summary!r}"
            )
        finally:
            browser.close()

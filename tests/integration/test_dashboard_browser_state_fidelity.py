"""Real end-to-end tests: the dashboard bridge must report browser/platform metadata
that reflects the session's ACTUAL technology, never fabricated defaults.

Regression for the data-fidelity bug (change: frontend-dashboard-browser-state-fidelity):
a pure BuiltIn/Requests session was shown as Platform=Web / Browser=Chromium /
Current URL=about:blank (a hardcoded literal + a laundered None.lower() crash).

These drive REAL sessions across different technologies through the real MCP tool layer
(fastmcp Client(mcp)) and assert on the real bridge output. Browser and SeleniumLibrary
cannot coexist in one process (web_automation exclusion), so SeleniumLibrary is covered in
test_dashboard_browser_state_fidelity_selenium.py; here we use Browser + Requests + BuiltIn.

Run: uv run pytest tests/integration/test_dashboard_browser_state_fidelity.py -v
"""

from __future__ import annotations

__test__ = True

import pytest
import pytest_asyncio
from fastmcp import Client

from robotmcp.server import mcp
from robotmcp.frontend.bridge import bridge


def _has(lib: str) -> bool:
    try:
        __import__(lib)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


async def _init(c, sid, libs):
    await c.call_tool("manage_session", {"action": "init", "session_id": sid, "libraries": libs})


async def _step(c, sid, kw, args, assign=None):
    p = {"keyword": kw, "arguments": args, "session_id": sid}
    if assign:
        p["assign_to"] = assign
    try:
        await c.call_tool("execute_step", p)
    except Exception:
        pass  # a failed action does not change the browser-vs-no-browser classification


class TestDashboardBrowserStateFidelity:
    async def test_builtin_session_reports_no_browser(self, mcp_client):
        """A BuiltIn-only session must NOT fabricate a browser/URL and must be 'generic'."""
        await _init(mcp_client, "fidelity-builtin", ["BuiltIn"])
        await _step(mcp_client, "fidelity-builtin", "Set Variable", ["hello"], "GREETING")
        d = await bridge.get_session_details("fidelity-builtin")
        bs = d.get("browser_state") or {}
        assert d.get("platform_type") == "generic", d.get("platform_type")
        assert not d.get("browser_type"), d.get("browser_type")
        assert not d.get("current_url"), d.get("current_url")
        # no fabricated chromium / about:blank leaks into browser_state
        assert not bs.get("browser_type"), bs
        assert bs.get("current_url") not in ("about:blank",), bs

    @pytest.mark.skipif(not _has("RequestsLibrary"), reason="RequestsLibrary not installed")
    async def test_requests_session_is_api_not_web(self, mcp_client):
        """An HTTP-request session must be 'api' with no browser/URL."""
        await _init(mcp_client, "fidelity-requests", ["RequestsLibrary"])
        await _step(mcp_client, "fidelity-requests", "Create Session", ["demo", "https://example.com"])
        d = await bridge.get_session_details("fidelity-requests")
        assert d.get("platform_type") == "api", d.get("platform_type")
        assert not d.get("browser_type"), d.get("browser_type")
        assert not d.get("current_url"), d.get("current_url")

    @pytest.mark.skipif(not _has("Browser"), reason="Browser Library not installed")
    async def test_browser_session_reports_real_state(self, mcp_client):
        """A real Browser session must be 'web' with the real engine + navigated URL."""
        await _init(mcp_client, "fidelity-browser", ["Browser"])
        await _step(mcp_client, "fidelity-browser", "New Browser", ["chromium", "headless=True"])
        await _step(mcp_client, "fidelity-browser", "New Page", ["https://example.com/"])
        d = await bridge.get_session_details("fidelity-browser")
        assert d.get("platform_type") == "web", d.get("platform_type")
        assert d.get("browser_type"), "expected a real browser engine"
        assert "example.com" in (d.get("current_url") or ""), d.get("current_url")
        # cleanup
        await _step(mcp_client, "fidelity-browser", "Close Browser", ["ALL"])

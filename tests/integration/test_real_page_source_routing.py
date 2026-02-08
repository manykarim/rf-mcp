"""Real browser integration tests for page source keyword routing.

Validates that the correct library-specific keywords are used for page source,
URL, and title retrieval based on which library is loaded in the session.

Phase 4 tests — verifies PageSourceService._keyword_candidates() routes to:
    Browser Library:    Get Page Source, Get Url, Get Title
    SeleniumLibrary:    Get Source, Get Location, Get Title

These tests complement the 46 unit tests in test_page_source_keyword_routing.py
by running against actual browsers to catch real-world integration issues.

NOTE: Browser and SeleniumLibrary tests each get their own browser session.
Since the exclusion group prevents them in the same session, they use separate
session IDs. The RF namespace is process-global, so library state persists.
"""

import os
import pytest
import pytest_asyncio
from fastmcp import Client

from robotmcp.server import mcp, execution_engine
from robotmcp.components.execution.page_source_service import PageSourceService

os.environ.setdefault("DISPLAY", ":0")

BROWSER_SESSION_ID = "ps_routing_browser"
SELENIUM_SESSION_ID = "ps_routing_selenium"

# Track whether sessions have been set up (module-level state)
_browser_session_ready = False
_selenium_session_ready = False


def _has_browser_library():
    try:
        import Browser  # noqa: F401
        return True
    except ImportError:
        return False


def _has_selenium_library():
    try:
        import SeleniumLibrary  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Phase 4: Page Source Keyword Routing - Static Logic Tests
# ---------------------------------------------------------------------------


class TestKeywordCandidatesRouting:
    """Verify _keyword_candidates selects correct keywords per library."""

    def test_browser_page_source_keyword(self):
        """Browser Library should use 'Get Page Source'."""
        candidates = PageSourceService._keyword_candidates(
            PageSourceService._SOURCE_KEYWORDS, ["Browser", "BuiltIn"]
        )
        assert candidates == ("Get Page Source",)

    def test_selenium_page_source_keyword(self):
        """SeleniumLibrary should use 'Get Source'."""
        candidates = PageSourceService._keyword_candidates(
            PageSourceService._SOURCE_KEYWORDS, ["SeleniumLibrary", "BuiltIn"]
        )
        assert candidates == ("Get Source",)

    def test_browser_url_keyword(self):
        """Browser Library should use 'Get Url'."""
        candidates = PageSourceService._keyword_candidates(
            PageSourceService._URL_KEYWORDS, ["Browser", "BuiltIn"]
        )
        assert candidates == ("Get Url",)

    def test_selenium_url_keyword(self):
        """SeleniumLibrary should use 'Get Location'."""
        candidates = PageSourceService._keyword_candidates(
            PageSourceService._URL_KEYWORDS, ["SeleniumLibrary", "BuiltIn"]
        )
        assert candidates == ("Get Location",)

    def test_browser_title_keyword(self):
        """Browser Library should use 'Get Title'."""
        candidates = PageSourceService._keyword_candidates(
            PageSourceService._TITLE_KEYWORDS, ["Browser", "BuiltIn"]
        )
        assert candidates == ("Get Title",)

    def test_selenium_title_keyword(self):
        """SeleniumLibrary should use 'Get Title'."""
        candidates = PageSourceService._keyword_candidates(
            PageSourceService._TITLE_KEYWORDS, ["SeleniumLibrary", "BuiltIn"]
        )
        assert candidates == ("Get Title",)

    def test_fallback_when_no_matching_library(self):
        """Fallback should return all unique keywords when no library matches."""
        candidates = PageSourceService._keyword_candidates(
            PageSourceService._SOURCE_KEYWORDS, ["BuiltIn", "Collections"]
        )
        assert "Get Page Source" in candidates
        assert "Get Source" in candidates

    def test_appium_page_source_keyword(self):
        """AppiumLibrary should use 'Get Source'."""
        candidates = PageSourceService._keyword_candidates(
            PageSourceService._SOURCE_KEYWORDS, ["AppiumLibrary", "BuiltIn"]
        )
        assert candidates == ("Get Source",)


# ---------------------------------------------------------------------------
# Phase 4: Real Browser Page Source Routing Tests
# Each test creates its own MCP client to avoid event loop scope issues.
# Browser sessions are created once and reused by session ID.
# ---------------------------------------------------------------------------


async def _ensure_browser_session(client):
    """Ensure Browser Library session exists (idempotent)."""
    global _browser_session_ready
    if _browser_session_ready:
        return
    init_res = await client.call_tool(
        "manage_session",
        {
            "action": "init",
            "session_id": BROWSER_SESSION_ID,
            "libraries": ["Browser", "BuiltIn"],
        },
    )
    assert init_res.data.get("success") is True
    await client.call_tool(
        "execute_step",
        {
            "keyword": "New Browser",
            "arguments": ["chromium", "headless=True"],
            "session_id": BROWSER_SESSION_ID,
        },
    )
    await client.call_tool(
        "execute_step",
        {
            "keyword": "New Page",
            "arguments": ["https://example.com"],
            "session_id": BROWSER_SESSION_ID,
        },
    )
    _browser_session_ready = True


async def _ensure_selenium_session(client):
    """Ensure SeleniumLibrary session exists (idempotent)."""
    global _selenium_session_ready
    if _selenium_session_ready:
        return
    init_res = await client.call_tool(
        "manage_session",
        {
            "action": "init",
            "session_id": SELENIUM_SESSION_ID,
            "libraries": ["SeleniumLibrary", "BuiltIn"],
        },
    )
    assert init_res.data.get("success") is True
    await client.call_tool(
        "execute_step",
        {
            "keyword": "Open Browser",
            "arguments": ["https://example.com", "headlesschrome"],
            "session_id": SELENIUM_SESSION_ID,
        },
    )
    _selenium_session_ready = True


@pytest.mark.skipif(
    not _has_browser_library(),
    reason="Browser Library not installed",
)
class TestRealBrowserPageSourceRouting:
    """Test page source routing with real Browser Library session."""

    @pytest.mark.asyncio
    async def test_browser_get_page_source_returns_html(self):
        """'Get Page Source' (Browser Library keyword) should return HTML."""
        async with Client(mcp) as client:
            await _ensure_browser_session(client)
            res = await client.call_tool(
                "execute_step",
                {
                    "keyword": "Get Page Source",
                    "arguments": [],
                    "session_id": BROWSER_SESSION_ID,
                    "assign_to": "html",
                },
            )
        assert res.data.get("success") is True
        html = res.data.get("assigned_variables", {}).get("${html}", "")
        assert "<html" in html.lower() or "<!doctype" in html.lower(), \
            f"Expected HTML, got: {html[:200]}"

    @pytest.mark.asyncio
    async def test_browser_get_url_returns_url(self):
        """'Get Url' (Browser Library keyword) should return URL string."""
        async with Client(mcp) as client:
            await _ensure_browser_session(client)
            res = await client.call_tool(
                "execute_step",
                {
                    "keyword": "Get Url",
                    "arguments": [],
                    "session_id": BROWSER_SESSION_ID,
                    "assign_to": "url",
                },
            )
        assert res.data.get("success") is True
        url = res.data.get("assigned_variables", {}).get("${url}", "")
        assert "example.com" in url

    @pytest.mark.asyncio
    async def test_browser_get_title_returns_title(self):
        """'Get Title' (Browser Library keyword) should return page title."""
        async with Client(mcp) as client:
            await _ensure_browser_session(client)
            res = await client.call_tool(
                "execute_step",
                {
                    "keyword": "Get Title",
                    "arguments": [],
                    "session_id": BROWSER_SESSION_ID,
                    "assign_to": "title",
                },
            )
        assert res.data.get("success") is True
        title = res.data.get("assigned_variables", {}).get("${title}", "")
        assert "example" in title.lower()


@pytest.mark.skipif(
    not _has_selenium_library(),
    reason="SeleniumLibrary not installed",
)
class TestRealSeleniumPageSourceRouting:
    """Test page source routing with real SeleniumLibrary session."""

    @pytest.mark.asyncio
    async def test_selenium_get_source_returns_html(self):
        """'Get Source' (SeleniumLibrary keyword) should return HTML."""
        async with Client(mcp) as client:
            await _ensure_selenium_session(client)
            res = await client.call_tool(
                "execute_step",
                {
                    "keyword": "Get Source",
                    "arguments": [],
                    "session_id": SELENIUM_SESSION_ID,
                    "assign_to": "html",
                },
            )
        assert res.data.get("success") is True
        html = res.data.get("assigned_variables", {}).get("${html}", "")
        assert "<html" in html.lower() or "<!doctype" in html.lower(), \
            f"Expected HTML, got: {html[:200]}"

    @pytest.mark.asyncio
    async def test_selenium_get_location_returns_url(self):
        """'Get Location' (SeleniumLibrary keyword) should return URL string."""
        async with Client(mcp) as client:
            await _ensure_selenium_session(client)
            res = await client.call_tool(
                "execute_step",
                {
                    "keyword": "Get Location",
                    "arguments": [],
                    "session_id": SELENIUM_SESSION_ID,
                    "assign_to": "url",
                },
            )
        assert res.data.get("success") is True
        url = res.data.get("assigned_variables", {}).get("${url}", "")
        assert "example.com" in url

    @pytest.mark.asyncio
    async def test_selenium_get_title_returns_title(self):
        """'Get Title' (SeleniumLibrary keyword) should return page title."""
        async with Client(mcp) as client:
            await _ensure_selenium_session(client)
            res = await client.call_tool(
                "execute_step",
                {
                    "keyword": "Get Title",
                    "arguments": [],
                    "session_id": SELENIUM_SESSION_ID,
                    "assign_to": "title",
                },
            )
        assert res.data.get("success") is True
        title = res.data.get("assigned_variables", {}).get("${title}", "")
        assert "example" in title.lower()

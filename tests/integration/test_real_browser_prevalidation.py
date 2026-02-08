"""Real browser integration tests for Browser Library pre-validation.

These tests use actual Chromium browser (headless) via Browser Library (Playwright)
to validate pre-validation, page source retrieval, and keyword routing end-to-end.

Requirements:
    - Chromium installed (e.g., /snap/bin/chromium or playwright-managed)
    - Browser Library (robotframework-browser) installed
    - Display server or headless mode

NOTE: Browser Library and SeleniumLibrary cannot coexist in the same process
due to the web_automation exclusion group. Run this file in a separate pytest
invocation from test_real_selenium_prevalidation.py.
"""

import os
import pytest
import pytest_asyncio
from fastmcp import Client

from robotmcp.server import mcp, execution_engine

# Ensure headless works even without X11
os.environ.setdefault("DISPLAY", ":0")

SESSION_ID = "real_browser_preval"


def _has_browser_library():
    """Check if Browser Library is importable."""
    try:
        import Browser  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = [
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.skipif(
        not _has_browser_library(),
        reason="Browser Library not installed",
    ),
]


@pytest_asyncio.fixture(scope="module")
async def browser_session():
    """Module-scoped fixture: one Browser Library session for all tests.

    Opens chromium headless to https://example.com.
    Yields (session, executor, client) tuple.
    Closes browser after all tests in this module.
    """
    async with Client(mcp) as client:
        # Init session with Browser + BuiltIn
        init_res = await client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": SESSION_ID,
                "libraries": ["Browser", "BuiltIn"],
            },
        )
        assert init_res.data.get("success") is True, f"Session init failed: {init_res.data}"

        # New Browser (headless chromium)
        browser_res = await client.call_tool(
            "execute_step",
            {
                "keyword": "New Browser",
                "arguments": ["chromium", "headless=True"],
                "session_id": SESSION_ID,
            },
        )
        assert browser_res.data.get("success") is True, f"New Browser failed: {browser_res.data}"

        # New Page -> example.com
        page_res = await client.call_tool(
            "execute_step",
            {
                "keyword": "New Page",
                "arguments": ["https://example.com"],
                "session_id": SESSION_ID,
            },
        )
        assert page_res.data.get("success") is True, f"New Page failed: {page_res.data}"

        session = execution_engine.session_manager.get_session(SESSION_ID)
        executor = execution_engine.keyword_executor

        yield session, executor, client

        # Cleanup: close all browsers
        try:
            await client.call_tool(
                "execute_step",
                {
                    "keyword": "Close Browser",
                    "arguments": ["ALL"],
                    "session_id": SESSION_ID,
                },
            )
        except Exception:
            pass  # Best effort cleanup


# ---------------------------------------------------------------------------
# Phase 2: Real Browser Pre-Validation Tests
# ---------------------------------------------------------------------------


class TestBrowserPreValidation:
    """Validate pre-validation with real Browser Library against live page."""

    async def test_visible_element_passes(self, browser_session):
        """Pre-validation should pass for a visible element (h1 on example.com)."""
        session, executor, _ = browser_session
        is_valid, error, details = await executor._pre_validate_element(
            "css=h1", session, "click"
        )
        assert is_valid is True, f"Expected valid, got error: {error}"
        assert error is None
        assert "visible" in details.get("current_states", [])

    async def test_missing_element_fails(self, browser_session):
        """Pre-validation should fail for a non-existent element."""
        session, executor, _ = browser_session
        is_valid, error, details = await executor._pre_validate_element(
            "css=#nonexistent-element-xyz", session, "click"
        )
        assert is_valid is False
        assert error is not None

    async def test_link_element_valid(self, browser_session):
        """Pre-validation should pass for a clickable link element."""
        session, executor, _ = browser_session
        # example.com has a link "More information..."
        is_valid, error, details = await executor._pre_validate_element(
            "css=a", session, "click"
        )
        assert is_valid is True, f"Expected valid link, got error: {error}"

    async def test_non_browser_keyword_skips(self, browser_session):
        """Pre-validation should skip for non-browser keywords like 'Log'."""
        session, executor, _ = browser_session
        is_valid, error, details = await executor._pre_validate_element(
            "css=h1", session, "log"
        )
        # "Log" is not in _requires_pre_validation, so this should not even
        # be called in production. But if called, it still validates against
        # the page. The key test is that it doesn't crash.
        # The executor checks _requires_pre_validation separately.
        assert is_valid is True

    async def test_pre_validation_returns_timing(self, browser_session):
        """Pre-validation should return elapsed_ms in details."""
        session, executor, _ = browser_session
        is_valid, error, details = await executor._pre_validate_element(
            "css=h1", session, "click"
        )
        assert is_valid is True
        assert "elapsed_ms" in details
        assert details["elapsed_ms"] > 0
        # Real browser pre-validation should complete within 5 seconds
        assert details["elapsed_ms"] < 5000


class TestBrowserKeywordExecution:
    """Validate actual keyword execution returns correct data."""

    async def test_get_title_returns_text(self, browser_session):
        """Get Title should return the page title from example.com."""
        _, _, client = browser_session
        res = await client.call_tool(
            "execute_step",
            {
                "keyword": "Get Title",
                "arguments": [],
                "session_id": SESSION_ID,
                "assign_to": "title",
            },
        )
        assert res.data.get("success") is True
        assigned = res.data.get("assigned_variables", {})
        title = assigned.get("${title}", "")
        assert "example" in title.lower(), f"Expected 'example' in title, got: {title}"

    async def test_get_url_returns_url(self, browser_session):
        """Get Url should return the current URL."""
        _, _, client = browser_session
        res = await client.call_tool(
            "execute_step",
            {
                "keyword": "Get Url",
                "arguments": [],
                "session_id": SESSION_ID,
                "assign_to": "url",
            },
        )
        assert res.data.get("success") is True
        assigned = res.data.get("assigned_variables", {})
        url = assigned.get("${url}", "")
        assert "example.com" in url, f"Expected 'example.com' in URL, got: {url}"

    async def test_get_page_source_returns_html(self, browser_session):
        """Get Page Source should return HTML content."""
        _, _, client = browser_session
        res = await client.call_tool(
            "execute_step",
            {
                "keyword": "Get Page Source",
                "arguments": [],
                "session_id": SESSION_ID,
                "assign_to": "source",
            },
        )
        assert res.data.get("success") is True
        assigned = res.data.get("assigned_variables", {})
        source = assigned.get("${source}", "")
        assert "<html" in source.lower() or "<body" in source.lower(), \
            f"Expected HTML in source, got: {source[:200]}"


class TestBrowserPageSourceService:
    """Validate page source retrieval via get_session_state MCP tool."""

    async def test_page_source_retrieval(self, browser_session):
        """get_session_state(page_source) should return HTML from the page."""
        _, _, client = browser_session
        res = await client.call_tool(
            "get_session_state",
            {
                "session_id": SESSION_ID,
                "sections": ["page_source"],
            },
        )
        data = res.data
        assert data.get("success") is True
        ps_section = data.get("sections", {}).get("page_source", {})
        assert ps_section.get("success") is True, f"Page source failed: {ps_section}"
        # Should have page source content
        assert ps_section.get("page_source_length", 0) > 0

    async def test_page_source_has_context(self, browser_session):
        """Page source context should contain title and url."""
        _, _, client = browser_session
        res = await client.call_tool(
            "get_session_state",
            {
                "session_id": SESSION_ID,
                "sections": ["page_source"],
            },
        )
        ps_section = res.data.get("sections", {}).get("page_source", {})
        context = ps_section.get("context", {})
        # Context uses "page_title" key from extract_page_context()
        title = context.get("page_title", "")
        assert title, f"Expected non-empty page_title in context, got: {context}"

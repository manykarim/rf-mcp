"""Real browser integration tests for SeleniumLibrary pre-validation.

These tests use actual Chromium browser (headless) via SeleniumLibrary (Selenium)
to validate pre-validation, page source retrieval, and keyword routing end-to-end.

Requirements:
    - Chromium + chromedriver installed
    - SeleniumLibrary installed
    - Display server or headless mode

NOTE: Browser Library and SeleniumLibrary cannot coexist in the same process
due to the web_automation exclusion group. Run this file in a separate pytest
invocation from test_real_browser_prevalidation.py.
"""

import os
import pytest
import pytest_asyncio
from fastmcp import Client

from robotmcp.server import mcp, execution_engine

# Ensure headless works even without X11
os.environ.setdefault("DISPLAY", ":0")

SESSION_ID = "real_selenium_preval"


def _has_selenium_library():
    """Check if SeleniumLibrary is importable."""
    try:
        import SeleniumLibrary  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = [
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.skipif(
        not _has_selenium_library(),
        reason="SeleniumLibrary not installed",
    ),
]


@pytest_asyncio.fixture(scope="module")
async def selenium_session():
    """Module-scoped fixture: one SeleniumLibrary session for all tests.

    Opens headlesschrome to https://example.com.
    Yields (session, executor, client) tuple.
    Closes browser after all tests in this module.
    """
    async with Client(mcp) as client:
        # Init session with SeleniumLibrary + BuiltIn
        init_res = await client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": SESSION_ID,
                "libraries": ["SeleniumLibrary", "BuiltIn"],
            },
        )
        assert init_res.data.get("success") is True, f"Session init failed: {init_res.data}"

        # Open Browser with headlesschrome
        browser_res = await client.call_tool(
            "execute_step",
            {
                "keyword": "Open Browser",
                "arguments": ["https://example.com", "headlesschrome"],
                "session_id": SESSION_ID,
            },
        )
        assert browser_res.data.get("success") is True, f"Open Browser failed: {browser_res.data}"

        session = execution_engine.session_manager.get_session(SESSION_ID)
        executor = execution_engine.keyword_executor

        yield session, executor, client

        # Cleanup: close all browsers
        try:
            await client.call_tool(
                "execute_step",
                {
                    "keyword": "Close All Browsers",
                    "arguments": [],
                    "session_id": SESSION_ID,
                },
            )
        except Exception:
            pass  # Best effort cleanup


# ---------------------------------------------------------------------------
# Phase 3: Real SeleniumLibrary Pre-Validation Tests
# ---------------------------------------------------------------------------


class TestSeleniumPreValidation:
    """Validate pre-validation with real SeleniumLibrary against live page."""

    async def test_visible_element_passes(self, selenium_session):
        """Pre-validation should pass for a visible element (h1 on example.com)."""
        session, executor, _ = selenium_session
        # Use "click element" — the actual SeleniumLibrary keyword name
        is_valid, error, details = await executor._pre_validate_element(
            "css=h1", session, "click element"
        )
        assert is_valid is True, f"Expected valid, got error: {error}"
        assert error is None

    async def test_missing_element_fails(self, selenium_session):
        """Pre-validation should fail for a non-existent element."""
        session, executor, _ = selenium_session
        is_valid, error, details = await executor._pre_validate_element(
            "css=#nonexistent-element-xyz", session, "click element"
        )
        assert is_valid is False
        assert error is not None

    async def test_link_element_valid(self, selenium_session):
        """Pre-validation should pass for a clickable link."""
        session, executor, _ = selenium_session
        is_valid, error, details = await executor._pre_validate_element(
            "css=a", session, "click element"
        )
        assert is_valid is True, f"Expected valid link, got error: {error}"

    async def test_pre_validation_returns_timing(self, selenium_session):
        """Pre-validation should return elapsed_ms in details."""
        session, executor, _ = selenium_session
        is_valid, error, details = await executor._pre_validate_element(
            "css=h1", session, "click element"
        )
        assert is_valid is True
        assert "elapsed_ms" in details
        assert details["elapsed_ms"] > 0
        assert details["elapsed_ms"] < 5000


class TestSeleniumKeywordExecution:
    """Validate actual SeleniumLibrary keyword execution returns correct data."""

    async def test_get_title_returns_text(self, selenium_session):
        """Get Title should return the page title from example.com."""
        _, _, client = selenium_session
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

    async def test_get_location_returns_url(self, selenium_session):
        """Get Location should return the current URL (SeleniumLibrary-specific)."""
        _, _, client = selenium_session
        res = await client.call_tool(
            "execute_step",
            {
                "keyword": "Get Location",
                "arguments": [],
                "session_id": SESSION_ID,
                "assign_to": "url",
            },
        )
        assert res.data.get("success") is True
        assigned = res.data.get("assigned_variables", {})
        url = assigned.get("${url}", "")
        assert "example.com" in url, f"Expected 'example.com' in URL, got: {url}"

    async def test_get_source_returns_html(self, selenium_session):
        """Get Source should return HTML content (SeleniumLibrary-specific)."""
        _, _, client = selenium_session
        res = await client.call_tool(
            "execute_step",
            {
                "keyword": "Get Source",
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

    async def test_timeout_not_injected_for_action_keywords(self, selenium_session):
        """Click Element should NOT receive timeout= argument (P0 fix validation)."""
        _, _, client = selenium_session
        # Click on the h1 — should succeed without timeout injection error
        res = await client.call_tool(
            "execute_step",
            {
                "keyword": "Click Element",
                "arguments": ["css=h1"],
                "session_id": SESSION_ID,
            },
        )
        assert res.data.get("success") is True, \
            f"Click Element failed (timeout injection?): {res.data.get('error', '')}"


class TestSeleniumPageSourceService:
    """Validate page source retrieval via get_session_state for SeleniumLibrary."""

    async def test_page_source_retrieval(self, selenium_session):
        """get_session_state(page_source) should return HTML from the page."""
        _, _, client = selenium_session
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
        assert ps_section.get("page_source_length", 0) > 0

    async def test_page_source_has_context(self, selenium_session):
        """Page source context should contain title."""
        _, _, client = selenium_session
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

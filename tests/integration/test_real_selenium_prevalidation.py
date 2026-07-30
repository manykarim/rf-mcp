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

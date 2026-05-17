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
from typing import Any, Optional

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
        """Pre-validation should pass for a visible element (h1 on example.com).

        Uses 'hover' action because h1 is non-interactive (not 'enabled'),
        so 'click' would fail since it requires {visible, enabled}.
        """
        session, executor, _ = browser_session
        is_valid, error, details = await executor._pre_validate_element(
            "css=h1", session, "hover"
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


# ---------------------------------------------------------------------------
# OBS-01 — id=X vs css=#X verdict equivalence (live Browser)
# ---------------------------------------------------------------------------


class TestBrowserIdLocatorEquivalenceOBS01:
    """OBS-01 — pre-validation verdicts must agree for ``id=X`` and
    ``css=#X`` against the same DOM element.

    The 2026-05-17 Tricentis benchmark exposed a real flake: the same
    ``<button id="generate">`` element was reported 'detached' when
    targeted as ``id=generate`` but passed pre-validation when targeted
    as ``css=#generate``. The fix
    (``_normalize_locator_for_browser_prevalidation``) rewrites
    ``id=X`` to ``[id="X"]`` for the pre-validation call only, forcing
    both forms through Playwright's CSS engine.

    This integration test runs against a real headless Chromium browser
    via Browser Library / Playwright using a ``data:text/html`` fixture
    URL with buttons in every id-shape variant the story's acceptance
    criteria #2 calls out. The unit-test counterpart lives in
    ``tests/unit/test_prevalidation_id_equivalence.py``.
    """

    # Fixture page with one button per id-shape variant. data: URL so
    # we don't need a static HTML file or a local web server.
    DATA_URL = (
        "data:text/html,"
        "<html><body>"
        "<button id='simple'>Simple</button>"
        "<button id='with-hyphen'>Hyphen</button>"
        "<button id='with_underscore'>Underscore</button>"
        "<button id='Camel123'>Camel</button>"
        "<button id='generate'>Generate</button>"
        "</body></html>"
    )

    @pytest.mark.parametrize("id_value", [
        "simple",
        "with-hyphen",       # the form most likely to trip a naive #X shortcut
        "with_underscore",
        "Camel123",
        "generate",          # Sonnet's actual repro id from Obstacle 3
    ])
    async def test_id_and_css_hash_produce_same_verdict(
        self, browser_session, id_value,
    ):
        session, executor, client = browser_session

        # Navigate to the fixture page. Per-test navigation keeps each
        # case isolated and stops earlier tests in this module (which
        # use example.com) from interacting with our fixture state.
        nav_res = await client.call_tool(
            "execute_step",
            {
                "keyword": "Go To",
                "arguments": [self.DATA_URL],
                "session_id": SESSION_ID,
            },
        )
        assert nav_res.data.get("success") is True, (
            f"Failed to navigate to fixture page: {nav_res.data}"
        )

        # ``hover`` requires only {visible}; using it keeps the test
        # focused on the locator-equivalence question without bringing
        # the {enabled} state into play (all the fixture buttons are
        # enabled but the test reads cleaner without that dependency).
        id_form_valid, id_err, id_details = await executor._pre_validate_element(
            f"id={id_value}", session, "hover",
        )
        css_form_valid, css_err, css_details = await executor._pre_validate_element(
            f"css=#{id_value}", session, "hover",
        )

        # Equivalence — the verdict MUST match across the two forms.
        # This is the OBS-01 acceptance #1 assertion.
        assert id_form_valid == css_form_valid, (
            f"Verdict mismatch for id={id_value!r}: "
            f"id-form valid={id_form_valid} (err={id_err!r}) vs "
            f"css-form valid={css_form_valid} (err={css_err!r}). "
            f"id-details={id_details}, css-details={css_details}"
        )
        # And both should pass — these fixture elements are visible.
        # If they don't pass it means the page didn't load or the data:
        # URL isn't being honoured, not that the equivalence is wrong.
        assert id_form_valid is True, (
            f"Both forms failed for id={id_value!r}; "
            f"check the data: URL navigation: err={id_err!r}"
        )

    async def test_missing_id_fails_consistently_across_forms(
        self, browser_session,
    ):
        """When the element genuinely does NOT exist, both forms must
        fail in the same way — equivalent failure mode is just as
        important as equivalent success mode (otherwise an LLM that
        observes 'css=#X failed' can't conclude anything about whether
        'id=X' would have succeeded)."""
        session, executor, client = browser_session
        nav_res = await client.call_tool(
            "execute_step",
            {
                "keyword": "Go To",
                "arguments": [self.DATA_URL],
                "session_id": SESSION_ID,
            },
        )
        assert nav_res.data.get("success") is True

        id_form_valid, id_err, _ = await executor._pre_validate_element(
            "id=definitely-not-on-this-page", session, "hover",
        )
        css_form_valid, css_err, _ = await executor._pre_validate_element(
            "css=#definitely-not-on-this-page", session, "hover",
        )
        assert id_form_valid is False and css_form_valid is False, (
            "Both forms must fail for a non-existent element"
        )


# ---------------------------------------------------------------------------
# OBS-06 — intent_action(intent="extract") end-to-end
# ---------------------------------------------------------------------------


class TestIntentActionExtractOBS06:
    """End-to-end test for the new extract verb against a real Browser
    session. Drives an Obstacle-3-equivalent scenario: read a dynamically-
    generated id-bearing value from the DOM, then assert against it.

    This is the integration test counterpart to
    tests/unit/test_intent_action_extract.py. The unit tests pin the
    wiring (transformers, keyword routers, adapter dispatch); this
    integration test confirms the wire-up dispatches successfully
    against real Playwright."""

    # Fixture page exercising every extract mode that takes a target.
    # The visible text is what we'll read; ``data-order-id`` is the
    # attribute we'll fetch; three `.item` cards make the count
    # assertable; the `input` lets us read mode=value.
    # NB: ``#`` in a data: URL is parsed as a fragment separator and
    # truncates the page content. Use ``Order`` rather than ``Card #``
    # to keep the body intact.
    DATA_URL = (
        "data:text/html,<html><head><title>OBS-06 Fixture</title></head>"
        "<body>"
        "<div id='order-display' data-order-id='ORD-1007696'>Order ORD-1007696</div>"
        "<div class='item'>A</div>"
        "<div class='item'>B</div>"
        "<div class='item'>C</div>"
        "<input id='user-name' value='alice' />"
        "</body></html>"
    )

    async def _navigate(self, client):
        nav_res = await client.call_tool(
            "execute_step",
            {
                "keyword": "Go To",
                "arguments": [self.DATA_URL],
                "session_id": SESSION_ID,
            },
        )
        assert nav_res.data.get("success") is True, (
            f"Navigation to OBS-06 fixture failed: {nav_res.data}"
        )

    async def test_extract_text_returns_element_text(self, browser_session):
        _, _, client = browser_session
        await self._navigate(client)
        res = await client.call_tool(
            "intent_action",
            {
                "intent": "extract",
                "target": "id=order-display",
                "mode": "text",
                "session_id": SESSION_ID,
            },
        )
        assert res.data.get("success") is True, f"extract text failed: {res.data}"
        # The extracted value is surfaced as a top-level field — the LLM
        # doesn't need to dig through result/output/assigned_values.
        assert res.data.get("extracted_value") == "Order ORD-1007696"

    async def test_extract_attribute_returns_attribute_value(self, browser_session):
        _, _, client = browser_session
        await self._navigate(client)
        res = await client.call_tool(
            "intent_action",
            {
                "intent": "extract",
                "target": "id=order-display",
                "mode": "attribute",
                "attribute_name": "data-order-id",
                "session_id": SESSION_ID,
            },
        )
        assert res.data.get("success") is True, f"extract attribute failed: {res.data}"
        assert res.data.get("extracted_value") == "ORD-1007696"

    async def test_extract_count_returns_match_count(self, browser_session):
        # The OBS-06 acceptance #3 case: mode=count must NOT strict-mode-
        # fail when the locator matches multiple elements. The fixture
        # has three .item divs.
        _, _, client = browser_session
        await self._navigate(client)
        res = await client.call_tool(
            "intent_action",
            {
                "intent": "extract",
                "target": "css=.item",
                "mode": "count",
                "session_id": SESSION_ID,
            },
        )
        assert res.data.get("success") is True, f"extract count failed: {res.data}"
        # Get Element Count returns an int.
        assert int(res.data.get("extracted_value")) == 3

    async def test_extract_value_returns_input_value(self, browser_session):
        # Browser's Get Property(selector, "value") reads the DOM value
        # property — used to read live <input> values. The transformer
        # appends the literal "value" attribute name.
        _, _, client = browser_session
        await self._navigate(client)
        res = await client.call_tool(
            "intent_action",
            {
                "intent": "extract",
                "target": "id=user-name",
                "mode": "value",
                "session_id": SESSION_ID,
            },
        )
        assert res.data.get("success") is True, f"extract value failed: {res.data}"
        assert res.data.get("extracted_value") == "alice"

    async def test_extract_title_returns_page_title(self, browser_session):
        # mode=title takes no target — exercises the
        # _EXTRACT_MODES_WITHOUT_TARGET path.
        _, _, client = browser_session
        await self._navigate(client)
        res = await client.call_tool(
            "intent_action",
            {
                "intent": "extract",
                "mode": "title",
                "session_id": SESSION_ID,
            },
        )
        assert res.data.get("success") is True, f"extract title failed: {res.data}"
        assert res.data.get("extracted_value") == "OBS-06 Fixture"

    async def test_extract_url_returns_current_url(self, browser_session):
        _, _, client = browser_session
        await self._navigate(client)
        res = await client.call_tool(
            "intent_action",
            {
                "intent": "extract",
                "mode": "url",
                "session_id": SESSION_ID,
            },
        )
        assert res.data.get("success") is True, f"extract url failed: {res.data}"
        # The URL came from the data: scheme — assert it round-trips
        # without checking the full URL-encoded form (some browsers
        # percent-escape, some don't).
        extracted = res.data.get("extracted_value") or ""
        assert "data:text/html" in extracted, (
            f"expected data: URL, got {extracted!r}"
        )

    async def test_extract_with_assign_to_captures_variable(self, browser_session):
        # OBS-06 acceptance #4: assign_to integration. The captured value
        # should be both in extracted_value AND in session variables so
        # the next step can reference ${order_id}.
        _, _, client = browser_session
        await self._navigate(client)
        res = await client.call_tool(
            "intent_action",
            {
                "intent": "extract",
                "target": "id=order-display",
                "mode": "attribute",
                "attribute_name": "data-order-id",
                "assign_to": "order_id",
                "session_id": SESSION_ID,
            },
        )
        assert res.data.get("success") is True
        assert res.data.get("extracted_value") == "ORD-1007696"
        # The intent_resolved block surfaces the extract_mode so callers
        # can confirm the right mode was applied.
        assert res.data.get("intent_resolved", {}).get("extract_mode") == "attribute"


# ---------------------------------------------------------------------------
# OBS-10 — Drag And Drop pre-scrolls off-screen source/target
# ---------------------------------------------------------------------------


class TestDragAndDropPrescrollOBS10:
    """OBS-10 — Browser Drag And Drop silently no-ops when source or
    target is outside the viewport (Playwright dispatches drag events
    but the drop doesn't fire). The 2026-05-17 post-OBS Tricentis
    benchmark Obstacle 10 (Todolist) cost Sonnet ~27 tool calls
    debugging this exact pattern.

    The fix: a Browser-plugin override pre-scrolls source + target
    into view via ``Browser.Scroll To Element`` BEFORE dispatching
    the actual Drag And Drop. Override returns ``None`` so normal
    dispatch proceeds — only difference is the elements are now
    guaranteed visible.

    This integration test runs against a real headless Chromium and
    uses a ``data:text/html`` fixture with:
      - A drop zone at the top of the page
      - A 2000px spacer pushing the draggable source below the viewport
    Without OBS-10's pre-scroll, the drag would silently no-op (the
    source's bounding box is off-screen at first call). With OBS-10,
    the override scrolls the source into view before the drag fires."""

    # Fixture: target at the top, then a 2200px spacer, then the
    # draggable source. The source's top edge starts ~2240px below
    # the viewport's top — well outside any reasonable browser height.
    # OBS-10's pre-scroll must bring it into view before Playwright's
    # drag mechanics can fire correctly.
    DATA_URL = (
        "data:text/html,<html><head><title>OBS-10 Fixture</title></head>"
        "<body>"
        "<div id='target' style='width:200px;height:80px;border:2px dashed "
        "blue;background:lightyellow'>drop here</div>"
        "<div style='height:2200px'></div>"
        "<div id='source' draggable='true' style='width:120px;height:40px;"
        "background:lightblue;cursor:grab'>drag me</div>"
        "</body></html>"
    )

    async def test_drag_and_drop_pre_scrolls_off_screen_source(
        self, browser_session,
    ):
        """OBS-10 contract: when the source element is off-screen,
        ``Drag And Drop`` must pre-scroll it into the viewport.

        We assert on the WINDOW scroll position (``top`` coord) before
        and after the drag — without OBS-10's pre-scroll, the page
        stays at scroll-top=0 even after Drag And Drop returns
        success (this was Sonnet's 27-call debug cycle on Tricentis
        Obstacle 10). With the pre-scroll override, the source
        scrolls into view → scroll-top > 0.

        We deliberately don't assert on drop-side semantics
        (HTML5 vs pointer events have library- and browser-specific
        quirks orthogonal to OBS-10). The unit tests in
        ``test_browser_drag_and_drop_prescroll.py`` pin the override
        wiring; this test confirms the wiring fires end-to-end
        against a real Browser session."""
        _, _, client = browser_session
        nav_res = await client.call_tool(
            "execute_step",
            {
                "keyword": "Go To",
                "arguments": [self.DATA_URL],
                "session_id": SESSION_ID,
            },
        )
        assert nav_res.data.get("success") is True

        # Pre-condition: the page is at scroll-top=0 (source is
        # below the viewport). Get Scroll Position returns a dict
        # ``{top, left, bottom, right}``.
        pre_scroll = await client.call_tool(
            "execute_step",
            {
                "keyword": "Get Scroll Position",
                "arguments": [],
                "session_id": SESSION_ID,
            },
        )
        assert pre_scroll.data.get("success") is True
        pre_top = _extract_scroll_top(pre_scroll.data.get("output"))
        assert pre_top is not None and pre_top < 100, (
            f"Fixture precondition: page should start at scroll-top=0; "
            f"got top={pre_top!r}"
        )

        # The OBS-10 call: Drag And Drop with off-screen source. The
        # override pre-scrolls; the drag itself can then proceed.
        drag_res = await client.call_tool(
            "execute_step",
            {
                "keyword": "Drag And Drop",
                "arguments": ["id=source", "id=target"],
                "session_id": SESSION_ID,
            },
        )
        assert drag_res.data.get("success") is True, (
            f"Drag And Drop should succeed; got {drag_res.data}"
        )

        # Post-condition: page has scrolled down to bring the
        # off-screen source into view. The source's top edge is
        # ~2280px down (target 80px + spacer 2200px); after
        # Scroll To Element, scroll-top should be a substantial
        # value > 1000 (Playwright scrolls to center the element).
        post_scroll = await client.call_tool(
            "execute_step",
            {
                "keyword": "Get Scroll Position",
                "arguments": [],
                "session_id": SESSION_ID,
            },
        )
        assert post_scroll.data.get("success") is True
        post_top = _extract_scroll_top(post_scroll.data.get("output"))
        assert post_top is not None, (
            f"Get Scroll Position failed: {post_scroll.data!r}"
        )
        assert post_top > 1000, (
            f"OBS-10 contract: page should scroll down to bring the "
            f"off-screen source into view. pre_top={pre_top}, "
            f"post_top={post_top}. Without the pre-scroll override "
            f"the page would stay at scroll-top=0 and Drag And Drop "
            f"would silently no-op on drop semantics."
        )


def _extract_scroll_top(scroll: Any) -> Optional[float]:
    """Browser.Get Scroll Position returns a dict ``{top, left, bottom,
    right}`` (or a stringified equivalent depending on the response
    serialisation path). Extract the ``top`` value defensively."""
    if isinstance(scroll, dict):
        v = scroll.get("top")
        if isinstance(v, (int, float)):
            return float(v)
    if isinstance(scroll, str):
        import re
        m = re.search(r"['\"]top['\"]\s*:\s*([\d.]+)", scroll)
        if m:
            return float(m.group(1))
    return None

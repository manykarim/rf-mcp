"""Tests for element interaction error hints.

These tests validate the hint generation for common element interaction
errors across Browser Library, SeleniumLibrary, and AppiumLibrary.
Each test creates a HintContext with a specific error_text and
session_search_order, then asserts that generate_hints returns
appropriate guidance.
"""

import pytest

from robotmcp.utils.hints import HintContext, generate_hints


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_ctx(
    keyword="Click",
    arguments=None,
    error_text="",
    search_order=None,
):
    """Create a HintContext for testing."""
    return HintContext(
        session_id="test-session",
        keyword=keyword,
        arguments=arguments or ["id=myElement"],
        error_text=error_text,
        session_search_order=search_order,
    )


def _titles(hints):
    """Return lowercased titles for easy assertions."""
    return [h["title"].lower() for h in hints]


def _all_example_text(hints):
    """Flatten all example values into a single lowercase string for searching."""
    parts = []
    for h in hints:
        for ex in h.get("examples", []):
            parts.append(str(ex).lower())
    return " ".join(parts)


def _all_message_text(hints):
    """Concatenate all message fields into a single lowercase string."""
    return " ".join(h["message"].lower() for h in hints)


# ===================================================================
# 1. Click Intercepted -- Browser Library
# ===================================================================


class TestClickInterceptedBrowser:
    def test_intercept_hint_returned(self):
        ctx = _make_ctx(
            error_text="Element <div> intercepts pointer events",
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "force" in combined

    def test_intercept_example_contains_click_with_options(self):
        ctx = _make_ctx(
            error_text="Element <div> intercepts pointer events",
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints)
        # Should suggest Click With Options or force=True
        assert "click with options" in combined or "force" in combined


# ===================================================================
# 2. Click Intercepted -- SeleniumLibrary
# ===================================================================


class TestClickInterceptedSelenium:
    def test_intercept_hint_returned(self):
        ctx = _make_ctx(
            error_text=(
                "element click intercepted: Element is not clickable at point "
                "(100, 200). Other element would receive the click"
            ),
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_intercept_suggests_js_click(self):
        ctx = _make_ctx(
            error_text=(
                "element click intercepted: Element is not clickable at point "
                "(100, 200). Other element would receive the click"
            ),
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "execute javascript" in combined or "arguments[0].click()" in combined


# ===================================================================
# 3. Click Intercepted -- AppiumLibrary
# ===================================================================


class TestClickInterceptedAppium:
    def test_intercept_hint_returned(self):
        ctx = _make_ctx(
            error_text="element click intercepted",
            search_order=["AppiumLibrary"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_intercept_suggests_hide_keyboard(self):
        ctx = _make_ctx(
            error_text="element click intercepted",
            search_order=["AppiumLibrary"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "hide keyboard" in combined


# ===================================================================
# 4. Strict Mode Violation -- Browser Library
# ===================================================================


class TestStrictModeViolationBrowser:
    def test_strict_mode_hint_returned(self):
        ctx = _make_ctx(
            error_text='strict mode violation: locator("#button") resolved to 3 elements',
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_strict_mode_suggests_narrowing(self):
        ctx = _make_ctx(
            error_text='strict mode violation: locator("#button") resolved to 3 elements',
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "visible" in combined or "nth=" in combined or "nth" in combined

    def test_resolved_to_elements_without_strict_mode_prefix(self):
        """'resolved to N elements' without 'strict mode violation' prefix should still trigger hint."""
        ctx = _make_ctx(
            error_text='locator("#btn") resolved to 5 elements',
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1
        assert any("multiple elements" in h["title"].lower() for h in hints)

    def test_hint_mentions_nth_zero_based(self):
        ctx = _make_ctx(
            error_text='strict mode violation: locator("button") resolved to 3 elements',
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        msg = _all_message_text(hints)
        assert "nth=0" in msg
        assert "zero-based" in msg

    def test_hint_examples_include_nth_0_and_nth_1(self):
        ctx = _make_ctx(
            error_text='strict mode violation: locator(".item") resolved to 4 elements',
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        examples_text = _all_example_text(hints)
        assert "nth=0" in examples_text
        assert "nth=1" in examples_text

    def test_hint_includes_element_count(self):
        ctx = _make_ctx(
            error_text='strict mode violation: locator("#x") resolved to 7 elements',
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        msg = _all_message_text(hints)
        assert "7" in msg

    def test_hint_uses_actual_keyword_in_examples(self):
        ctx = _make_ctx(
            keyword="Fill Text",
            error_text='strict mode violation: locator("input") resolved to 2 elements',
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        examples_text = _all_example_text(hints)
        assert "fill text" in examples_text

    def test_not_triggered_for_selenium(self):
        """Strict mode / nth= hint is Browser Library specific."""
        ctx = _make_ctx(
            error_text='strict mode violation: locator("#btn") resolved to 3 elements',
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        titles = _titles(hints)
        assert "selector matches multiple elements" not in titles


# ===================================================================
# 5. Invalid Selector
# ===================================================================


class TestInvalidSelector:
    def test_invalid_selector_hint_returned(self):
        ctx = _make_ctx(
            error_text="invalid selector: not a valid XPath expression",
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_invalid_selector_mentions_syntax(self):
        ctx = _make_ctx(
            error_text="invalid selector: not a valid XPath expression",
        )
        hints = generate_hints(ctx)
        combined = _all_message_text(hints)
        assert "syntax" in combined or "xpath" in combined or "selector" in combined


# ===================================================================
# 6. Element Outside Viewport -- Browser
# ===================================================================


class TestElementOutsideViewportBrowser:
    def test_viewport_hint_returned(self):
        ctx = _make_ctx(
            error_text="element is outside of the viewport",
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_viewport_suggests_scroll_to_element(self):
        ctx = _make_ctx(
            error_text="element is outside of the viewport",
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "scroll to element" in combined


# ===================================================================
# 7. Element Outside Viewport -- Appium
# ===================================================================


class TestElementOutsideViewportAppium:
    def test_viewport_hint_returned(self):
        ctx = _make_ctx(
            error_text="element is outside of the viewport",
            search_order=["AppiumLibrary"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_viewport_suggests_scroll_down(self):
        ctx = _make_ctx(
            error_text="element is outside of the viewport",
            search_order=["AppiumLibrary"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "scroll down" in combined or "scroll" in combined


# ===================================================================
# 8. Element Not Interactable -- Browser
# ===================================================================


class TestElementNotInteractableBrowser:
    def test_not_interactable_hint_returned(self):
        ctx = _make_ctx(
            error_text="element is not visible",
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_not_interactable_suggests_wait(self):
        ctx = _make_ctx(
            error_text="element is not visible",
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "wait for elements state" in combined


# ===================================================================
# 9. Element Not Interactable -- Selenium
# ===================================================================


class TestElementNotInteractableSelenium:
    def test_not_interactable_hint_returned(self):
        ctx = _make_ctx(
            error_text="ElementNotInteractableException: element not interactable",
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_not_interactable_suggests_wait_visible(self):
        ctx = _make_ctx(
            error_text="ElementNotInteractableException: element not interactable",
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "wait until element is visible" in combined


# ===================================================================
# 10. Element Not Interactable -- Appium
# ===================================================================


class TestElementNotInteractableAppium:
    def test_not_interactable_hint_returned(self):
        ctx = _make_ctx(
            error_text="element not interactable",
            search_order=["AppiumLibrary"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_not_interactable_suggests_hide_keyboard(self):
        ctx = _make_ctx(
            error_text="element not interactable",
            search_order=["AppiumLibrary"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "hide keyboard" in combined


# ===================================================================
# 11. Stale Element -- Browser
# ===================================================================


class TestStaleElementBrowser:
    def test_stale_hint_returned(self):
        ctx = _make_ctx(
            error_text="element is not attached",
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_stale_suggests_wait_attached(self):
        ctx = _make_ctx(
            error_text="element is not attached",
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "wait for elements state" in combined
        assert "attached" in combined


# ===================================================================
# 12. Stale Element -- Selenium
# ===================================================================


class TestStaleElementSelenium:
    def test_stale_hint_returned(self):
        ctx = _make_ctx(
            error_text="stale element reference: element is not attached to the page document",
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_stale_suggests_wait_page_contains(self):
        ctx = _make_ctx(
            error_text="stale element reference: element is not attached to the page document",
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "wait until page contains element" in combined


# ===================================================================
# 13. Stale Element -- Appium
# ===================================================================


class TestStaleElementAppium:
    def test_stale_hint_returned(self):
        ctx = _make_ctx(
            error_text=(
                "StaleElementReferenceError: The previously found element "
                "is not present in the current view anymore"
            ),
            search_order=["AppiumLibrary"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_stale_suggests_expect_element(self):
        ctx = _make_ctx(
            error_text=(
                "StaleElementReferenceError: The previously found element "
                "is not present in the current view anymore"
            ),
            search_order=["AppiumLibrary"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        # Appium should recommend re-finding the element
        assert "expect element" in combined or "wait" in combined


# ===================================================================
# 14. Unexpected Alert -- Selenium
# ===================================================================


class TestUnexpectedAlertSelenium:
    def test_alert_hint_returned(self):
        ctx = _make_ctx(
            error_text="unexpected alert open",
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_alert_suggests_handle_alert(self):
        ctx = _make_ctx(
            error_text="unexpected alert open",
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "handle alert" in combined


# ===================================================================
# 15. Unexpected Alert -- Appium
# ===================================================================


class TestUnexpectedAlertAppium:
    def test_alert_hint_returned(self):
        ctx = _make_ctx(
            error_text="unexpected alert open",
            search_order=["AppiumLibrary"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_alert_suggests_click_alert_button(self):
        ctx = _make_ctx(
            error_text="unexpected alert open",
            search_order=["AppiumLibrary"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "click alert button" in combined or "alert" in combined


# ===================================================================
# 16. Mobile Context Mismatch -- Appium
# ===================================================================


class TestMobileContextMismatchAppium:
    def test_context_mismatch_hint_returned(self):
        ctx = _make_ctx(
            error_text="UnknownCommandError: unknown command",
            search_order=["AppiumLibrary"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_context_mismatch_suggests_switch(self):
        ctx = _make_ctx(
            error_text="UnknownCommandError: unknown command",
            search_order=["AppiumLibrary"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "get current context" in combined or "switch to context" in combined or "context" in combined


# ===================================================================
# 17. Mobile Scroll Needed -- Appium
# ===================================================================


class TestMobileScrollNeededAppium:
    def test_scroll_hint_returned(self):
        ctx = _make_ctx(
            error_text=(
                "An element could not be located on the page using "
                "the given search parameters"
            ),
            search_order=["AppiumLibrary"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_scroll_suggests_scroll_down(self):
        ctx = _make_ctx(
            error_text=(
                "An element could not be located on the page using "
                "the given search parameters"
            ),
            search_order=["AppiumLibrary"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "scroll down" in combined or "scroll" in combined


# ===================================================================
# 18. Element Not Found -- Browser
# ===================================================================


class TestElementNotFoundBrowser:
    def test_not_found_hint_returned(self):
        ctx = _make_ctx(
            error_text="Timeout 10000ms exceeded. waiting for locator('id=myElement')",
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_not_found_suggests_wait_attached(self):
        ctx = _make_ctx(
            error_text="Timeout 10000ms exceeded. waiting for locator('id=myElement')",
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "wait for elements state" in combined
        assert "attached" in combined or "locator" in combined


# ===================================================================
# 19. Element Not Found -- Selenium
# ===================================================================


class TestElementNotFoundSelenium:
    def test_not_found_hint_returned(self):
        ctx = _make_ctx(
            error_text="no such element: Unable to locate element",
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_not_found_suggests_wait_and_frame(self):
        ctx = _make_ctx(
            error_text="no such element: Unable to locate element",
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "wait until page contains element" in combined
        assert "select frame" in combined


# ===================================================================
# 20. Invalid Element State -- Browser
# ===================================================================


class TestInvalidElementStateBrowser:
    def test_invalid_state_hint_returned(self):
        ctx = _make_ctx(
            error_text="element is not editable",
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_invalid_state_suggests_wait_editable(self):
        ctx = _make_ctx(
            error_text="element is not editable",
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "wait for elements state" in combined
        assert "editable" in combined


# ===================================================================
# 21. Invalid Element State -- Selenium
# ===================================================================


class TestInvalidElementStateSelenium:
    def test_invalid_state_hint_returned(self):
        ctx = _make_ctx(
            error_text="InvalidElementStateException: invalid element state",
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_invalid_state_suggests_wait_enabled(self):
        ctx = _make_ctx(
            error_text="InvalidElementStateException: invalid element state",
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "wait until element is enabled" in combined


# ===================================================================
# 22. Frame Error -- Selenium
# ===================================================================


class TestFrameErrorSelenium:
    def test_frame_error_hint_returned(self):
        ctx = _make_ctx(
            error_text="NoSuchFrameException: no such frame",
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_frame_error_suggests_unselect_and_select(self):
        ctx = _make_ctx(
            error_text="NoSuchFrameException: no such frame",
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "unselect frame" in combined
        assert "select frame" in combined


# ===================================================================
# 23. Window Error -- Selenium
# ===================================================================


class TestWindowErrorSelenium:
    def test_window_error_hint_returned(self):
        ctx = _make_ctx(
            error_text="NoSuchWindowException: no such window",
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_window_error_suggests_switch_window(self):
        ctx = _make_ctx(
            error_text="NoSuchWindowException: no such window",
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "switch window" in combined


# ===================================================================
# 24. Timeout (generic) -- Browser
# ===================================================================


class TestTimeoutBrowser:
    def test_generic_timeout_hint_returned(self):
        """Generic timeout without locator or intercept context."""
        ctx = _make_ctx(
            error_text="Timeout 10000ms exceeded",
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        # Should get at least a generic timeout hint
        assert len(hints) >= 1

    def test_generic_timeout_not_intercept(self):
        """A bare 'Timeout' error should NOT produce the click-intercept hint."""
        ctx = _make_ctx(
            error_text="Timeout 10000ms exceeded",
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        # Should not mention force-click or intercept since there is none
        assert "intercept" not in combined


# ===================================================================
# 25. Timeout -- Selenium
# ===================================================================


class TestTimeoutSelenium:
    def test_timeout_hint_returned(self):
        ctx = _make_ctx(
            error_text="TimeoutException: timed out",
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1

    def test_timeout_suggests_set_timeout(self):
        ctx = _make_ctx(
            error_text="TimeoutException: timed out",
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "set selenium timeout" in combined


# ===================================================================
# 26. No match -- returns empty
# ===================================================================


class TestNoMatch:
    def test_random_error_returns_empty(self):
        ctx = _make_ctx(
            error_text="some random error that does not match any pattern",
        )
        hints = generate_hints(ctx)
        # Should return empty or no element-interaction hints
        # (other hint generators may still fire, so we check there are
        # no element-interaction-related titles)
        element_titles = [
            t
            for t in _titles(hints)
            if any(
                kw in t
                for kw in [
                    "click",
                    "intercept",
                    "viewport",
                    "stale",
                    "alert",
                    "frame",
                    "window",
                    "timeout",
                    "not interactable",
                    "strict mode",
                    "selector",
                    "scroll",
                    "context mismatch",
                    "not found",
                    "invalid element",
                    "not editable",
                ]
            )
        ]
        assert len(element_titles) == 0


# ===================================================================
# 27. Locator substitution in examples
# ===================================================================


class TestLocatorSubstitution:
    def test_locator_appears_in_examples(self):
        ctx = _make_ctx(
            error_text="Element <div> intercepts pointer events",
            arguments=["css=.my-button"],
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1
        combined = _all_example_text(hints)
        assert "css=.my-button" in combined


# ===================================================================
# 28. Priority: intercept beats timeout
# ===================================================================


class TestPriorityInterceptBeatsTimeout:
    def test_intercept_before_timeout(self):
        ctx = _make_ctx(
            error_text=(
                "Timeout 10000ms exceeded. <span class='overlay'> "
                "intercepts pointer events"
            ),
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1
        # The first hint should be about the intercept, not generic timeout
        first_title = hints[0]["title"].lower()
        assert (
            "intercept" in first_title
            or "click" in first_title
            or "force" in hints[0]["message"].lower()
        )


# ===================================================================
# 29. Unknown library fallback
# ===================================================================


class TestUnknownLibraryFallback:
    """When no library is detected, element interaction hints are intentionally
    skipped because the advice is library-specific.  Providing incorrect
    library guidance would be worse than providing no guidance at all."""

    def test_no_search_order_returns_no_element_hints(self):
        ctx = _make_ctx(
            error_text="element not interactable",
            search_order=None,
        )
        hints = generate_hints(ctx)
        # No element-interaction hints because library is unknown
        assert len(hints) == 0

    def test_empty_search_order_returns_no_element_hints(self):
        ctx = _make_ctx(
            error_text="element not interactable",
            search_order=[],
        )
        hints = generate_hints(ctx)
        assert len(hints) == 0

    def test_recognised_library_does_return_hint(self):
        """Sanity check: the same error with a known library produces a hint."""
        ctx = _make_ctx(
            error_text="element not interactable",
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1


# ===================================================================
# 30. Relevance ordering
# ===================================================================


class TestRelevanceOrdering:
    def test_hints_sorted_by_relevance_descending(self):
        """When multiple patterns match, hints should be sorted highest-relevance first."""
        # Use an error that could match multiple patterns
        ctx = _make_ctx(
            error_text=(
                "Timeout 10000ms exceeded. <span class='overlay'> "
                "intercepts pointer events"
            ),
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        if len(hints) >= 2:
            # generate_hints strips the relevance field from the returned dicts,
            # but the ordering should still be highest-first. We verify that
            # the intercept hint (higher relevance) appears before any generic
            # timeout hint.
            first = hints[0]["title"].lower()
            assert "intercept" in first or "click" in first

    def test_max_three_hints_returned(self):
        """generate_hints should return at most 3 hints."""
        ctx = _make_ctx(
            error_text=(
                "Timeout 10000ms exceeded. <span class='overlay'> "
                "intercepts pointer events"
            ),
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        assert len(hints) <= 3


# ===================================================================
# 31. get_session_state tool suggestions for page inspection
# ===================================================================


class TestGetSessionStateSuggestion:
    """Hints for element-not-found and locator timeouts should suggest
    get_session_state to inspect the DOM/ARIA snapshot."""

    def test_element_not_found_browser_suggests_get_session_state(self):
        ctx = _make_ctx(
            error_text="Timeout 10000ms exceeded. waiting for locator('id=model')",
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints)
        assert "get_session_state" in combined

    def test_element_not_found_selenium_suggests_get_session_state(self):
        ctx = _make_ctx(
            error_text="no such element: Unable to locate element",
            search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints)
        assert "get_session_state" in combined

    def test_mobile_scroll_suggests_get_session_state(self):
        ctx = _make_ctx(
            error_text=(
                "An element could not be located on the page "
                "using the given search parameters"
            ),
            search_order=["AppiumLibrary"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints)
        assert "get_session_state" in combined

    def test_select_options_locator_timeout_suggests_inspection(self):
        """Real-world scenario: Select Options By fails because locator is wrong."""
        ctx = _make_ctx(
            keyword="Select Options By",
            arguments=["id=model", "value", "Scooter"],
            error_text=(
                "TimeoutError: locator.selectOption: Timeout 10000ms exceeded.\n"
                "Call log:\n  - waiting for locator(\"id=model\")"
            ),
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        assert len(hints) >= 1
        combined = _all_example_text(hints) + " " + _all_message_text(hints)
        assert "get_session_state" in combined
        assert "inspect" in combined or "aria" in combined or "dom" in combined

    def test_inspect_hint_includes_page_source_section(self):
        """The get_session_state suggestion should request page_source section."""
        ctx = _make_ctx(
            error_text="Timeout 10000ms exceeded. waiting for locator('id=model')",
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints)
        assert "page_source" in combined

    def test_generic_timeout_without_locator_no_inspection(self):
        """A generic timeout (no 'waiting for locator') should NOT suggest inspection."""
        ctx = _make_ctx(
            error_text="Timeout 10000ms exceeded",
            search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        combined = _all_example_text(hints)
        assert "get_session_state" not in combined

"""Tests for expanded error classification in instruction_learner.

Tests the _classify_error method of SessionTracker, covering all 15 error
categories, every pattern string, priority ordering, and case insensitivity.
"""

import pytest

from robotmcp.optimization.instruction_learner import SessionTracker


@pytest.fixture
def tracker():
    """Create a SessionTracker for testing."""
    return SessionTracker(
        session_id="test-session",
        instruction_mode="default",
        llm_type="test-llm",
        scenario_type="test",
    )


# ---------------------------------------------------------------------------
# 1. invalid_keyword
# ---------------------------------------------------------------------------
class TestInvalidKeyword:
    """Tests for the invalid_keyword error category."""

    def test_keyword_not_found(self, tracker):
        result = tracker._classify_error("keyword not found: Click Button")
        assert result == "invalid_keyword"

    def test_no_keyword(self, tracker):
        result = tracker._classify_error("no keyword matches 'Clicky'")
        assert result == "invalid_keyword"

    def test_unknown_keyword(self, tracker):
        result = tracker._classify_error("unknown keyword: Press")
        assert result == "invalid_keyword"


# ---------------------------------------------------------------------------
# 2. element_intercept
# ---------------------------------------------------------------------------
class TestElementIntercept:
    """Tests for the element_intercept error category."""

    def test_intercepts_pointer_events(self, tracker):
        result = tracker._classify_error(
            "Element <div> intercepts pointer events"
        )
        assert result == "element_intercept"

    def test_element_click_intercepted(self, tracker):
        result = tracker._classify_error(
            "element click intercepted: Element <div> is not clickable"
        )
        assert result == "element_intercept"

    def test_is_not_clickable_at_point(self, tracker):
        result = tracker._classify_error(
            "Element is not clickable at point (100, 200)"
        )
        assert result == "element_intercept"

    def test_other_element_would_receive_the_click(self, tracker):
        result = tracker._classify_error(
            "Other element would receive the click: <div class='overlay'>"
        )
        assert result == "element_intercept"


# ---------------------------------------------------------------------------
# 3. strict_mode
# ---------------------------------------------------------------------------
class TestStrictMode:
    """Tests for the strict_mode error category."""

    def test_strict_mode_violation(self, tracker):
        result = tracker._classify_error(
            'strict mode violation: locator("#btn") resolved to 3 elements'
        )
        assert result == "strict_mode"


# ---------------------------------------------------------------------------
# 4. invalid_selector
# ---------------------------------------------------------------------------
class TestInvalidSelector:
    """Tests for the invalid_selector error category."""

    def test_invalid_selector(self, tracker):
        result = tracker._classify_error("invalid selector: //div[")
        assert result == "invalid_selector"

    def test_not_a_valid_xpath(self, tracker):
        result = tracker._classify_error(
            "The expression '//div[' is not a valid xpath expression"
        )
        assert result == "invalid_selector"

    def test_selector_syntax_error(self, tracker):
        result = tracker._classify_error(
            "selector syntax error at position 5"
        )
        assert result == "invalid_selector"

    def test_unexpected_token(self, tracker):
        result = tracker._classify_error(
            "unexpected token in selector: '>>'"
        )
        assert result == "invalid_selector"


# ---------------------------------------------------------------------------
# 5. element_outside_viewport
# ---------------------------------------------------------------------------
class TestElementOutsideViewport:
    """Tests for the element_outside_viewport error category."""

    def test_element_is_outside_of_the_viewport(self, tracker):
        result = tracker._classify_error(
            "element is outside of the viewport"
        )
        assert result == "element_outside_viewport"

    def test_outside_and_viewport_combined(self, tracker):
        result = tracker._classify_error(
            "The element is positioned outside the viewport area"
        )
        assert result == "element_outside_viewport"


# ---------------------------------------------------------------------------
# 6. element_not_interactable
# ---------------------------------------------------------------------------
class TestElementNotInteractable:
    """Tests for the element_not_interactable error category."""

    def test_element_not_interactable(self, tracker):
        result = tracker._classify_error(
            "element not interactable: <input type='hidden'>"
        )
        assert result == "element_not_interactable"

    def test_elementnotinteractableexception(self, tracker):
        result = tracker._classify_error(
            "ElementNotInteractableException: Message: element not interactable"
        )
        assert result == "element_not_interactable"

    def test_not_currently_visible(self, tracker):
        result = tracker._classify_error(
            "Element is not currently visible and may not be interacted with"
        )
        assert result == "element_not_interactable"

    def test_element_is_not_visible(self, tracker):
        result = tracker._classify_error(
            "element is not visible so it cannot be clicked"
        )
        assert result == "element_not_interactable"

    def test_element_is_not_enabled(self, tracker):
        result = tracker._classify_error(
            "element is not enabled and cannot be interacted with"
        )
        assert result == "element_not_interactable"


# ---------------------------------------------------------------------------
# 7. stale_ref
# ---------------------------------------------------------------------------
class TestStaleRef:
    """Tests for the stale_ref error category."""

    def test_stale_element_reference(self, tracker):
        result = tracker._classify_error(
            "stale element reference: element is not attached to the page document"
        )
        assert result == "stale_ref"

    def test_element_is_not_attached(self, tracker):
        result = tracker._classify_error(
            "element is not attached to the DOM"
        )
        assert result == "stale_ref"

    def test_not_attached_to_the_page_document(self, tracker):
        result = tracker._classify_error(
            "The element is not attached to the page document"
        )
        assert result == "stale_ref"

    def test_not_present_in_the_current_view(self, tracker):
        result = tracker._classify_error(
            "Element is not present in the current view"
        )
        assert result == "stale_ref"

    def test_expired_from_the_internal_cache(self, tracker):
        result = tracker._classify_error(
            "Element handle has expired from the internal cache"
        )
        assert result == "stale_ref"

    def test_target_closed(self, tracker):
        result = tracker._classify_error(
            "target closed: page has been closed"
        )
        assert result == "stale_ref"


# ---------------------------------------------------------------------------
# 8. frame_error
# ---------------------------------------------------------------------------
class TestFrameError:
    """Tests for the frame_error error category."""

    def test_no_such_frame(self, tracker):
        result = tracker._classify_error(
            "no such frame: frame id 123 not found"
        )
        assert result == "frame_error"

    def test_nosuchframeexception(self, tracker):
        result = tracker._classify_error(
            "NoSuchFrameException: Unable to locate frame"
        )
        assert result == "frame_error"

    def test_frame_was_detached(self, tracker):
        result = tracker._classify_error(
            "frame was detached during navigation"
        )
        assert result == "frame_error"

    def test_err_aborted(self, tracker):
        result = tracker._classify_error(
            "net::ERR_ABORTED while loading resource"
        )
        assert result == "frame_error"


# ---------------------------------------------------------------------------
# 9. alert_dialog
# ---------------------------------------------------------------------------
class TestAlertDialog:
    """Tests for the alert_dialog error category."""

    def test_unexpected_alert_open(self, tracker):
        result = tracker._classify_error(
            "unexpected alert open: {Alert text: Are you sure?}"
        )
        assert result == "alert_dialog"

    def test_unexpectedalertpresentexception(self, tracker):
        result = tracker._classify_error(
            "UnexpectedAlertPresentException: Alert is present"
        )
        assert result == "alert_dialog"


# ---------------------------------------------------------------------------
# 10. window_error
# ---------------------------------------------------------------------------
class TestWindowError:
    """Tests for the window_error error category."""

    def test_no_such_window(self, tracker):
        result = tracker._classify_error(
            "no such window: target window already closed"
        )
        assert result == "window_error"

    def test_nosuchwindowexception(self, tracker):
        result = tracker._classify_error(
            "NoSuchWindowException: Unable to find window"
        )
        assert result == "window_error"


# ---------------------------------------------------------------------------
# 11. invalid_state
# ---------------------------------------------------------------------------
class TestInvalidState:
    """Tests for the invalid_state error category."""

    def test_invalid_element_state(self, tracker):
        result = tracker._classify_error(
            "invalid element state: cannot clear a non-editable element"
        )
        assert result == "invalid_state"

    def test_invalidelementstateexception(self, tracker):
        result = tracker._classify_error(
            "InvalidElementStateException: element not editable"
        )
        assert result == "invalid_state"

    def test_may_not_be_manipulated(self, tracker):
        result = tracker._classify_error(
            "Element may not be manipulated"
        )
        assert result == "invalid_state"

    def test_element_is_not_editable(self, tracker):
        result = tracker._classify_error(
            "element is not editable: <select> elements cannot be typed into"
        )
        assert result == "invalid_state"


# ---------------------------------------------------------------------------
# 12. element_not_found
# ---------------------------------------------------------------------------
class TestElementNotFound:
    """Tests for the element_not_found error category."""

    def test_no_such_element(self, tracker):
        result = tracker._classify_error(
            "no such element: Unable to locate element: #missing"
        )
        assert result == "element_not_found"

    def test_unable_to_locate_element(self, tracker):
        result = tracker._classify_error(
            "unable to locate element: //div[@id='gone']"
        )
        assert result == "element_not_found"

    def test_could_not_be_located(self, tracker):
        result = tracker._classify_error(
            "Element with locator '#btn' could not be located"
        )
        assert result == "element_not_found"

    def test_did_not_match_any_elements(self, tracker):
        result = tracker._classify_error(
            "Locator css=#missing did not match any elements"
        )
        assert result == "element_not_found"

    def test_page_should_have_contained_element(self, tracker):
        result = tracker._classify_error(
            "Page should have contained element 'id=main' but did not"
        )
        assert result == "element_not_found"

    def test_element_not_found(self, tracker):
        result = tracker._classify_error(
            "Element not found using selector: .btn-primary"
        )
        assert result == "element_not_found"


# ---------------------------------------------------------------------------
# 13. timeout
# ---------------------------------------------------------------------------
class TestTimeout:
    """Tests for the timeout error category."""

    def test_timeout(self, tracker):
        result = tracker._classify_error("timeout waiting for element")
        assert result == "timeout"

    def test_timed_out(self, tracker):
        result = tracker._classify_error(
            "Timed out after 30000ms waiting for condition"
        )
        assert result == "timeout"


# ---------------------------------------------------------------------------
# 14. connection_error
# ---------------------------------------------------------------------------
class TestConnectionError:
    """Tests for the connection_error error category."""

    def test_connection(self, tracker):
        result = tracker._classify_error("connection refused on port 4444")
        assert result == "connection_error"

    def test_network(self, tracker):
        result = tracker._classify_error("network error: ERR_CONNECTION_RESET")
        assert result == "connection_error"


# ---------------------------------------------------------------------------
# 15. other (default)
# ---------------------------------------------------------------------------
class TestOther:
    """Tests for the other/default error category."""

    def test_random_error(self, tracker):
        result = tracker._classify_error("some random error occurred")
        assert result == "other"

    def test_unrecognized_error(self, tracker):
        result = tracker._classify_error(
            "JavaScriptError: Cannot read properties of undefined"
        )
        assert result == "other"


# ---------------------------------------------------------------------------
# 16. None / empty input
# ---------------------------------------------------------------------------
class TestNoneAndEmpty:
    """Tests for None and empty string inputs."""

    def test_none_input(self, tracker):
        result = tracker._classify_error(None)
        assert result is None

    def test_empty_string(self, tracker):
        result = tracker._classify_error("")
        assert result is None


# ---------------------------------------------------------------------------
# Priority ordering tests
# ---------------------------------------------------------------------------
class TestPriorityOrdering:
    """Tests that more specific patterns win over generic ones.

    The _classify_error method checks patterns in order from most specific
    to least specific. These tests verify that ordering is correct.
    """

    def test_intercept_beats_timeout(self, tracker):
        """Error with both 'timeout' AND 'intercepts pointer events' should
        be element_intercept, not timeout."""
        result = tracker._classify_error(
            "Timeout: element intercepts pointer events during click"
        )
        assert result == "element_intercept"

    def test_element_not_found_beats_timeout(self, tracker):
        """Error with both 'timeout' AND 'no such element' should be
        element_not_found, not timeout."""
        result = tracker._classify_error(
            "Timeout waiting: no such element could be found"
        )
        assert result == "element_not_found"

    def test_stale_ref_beats_element_not_found(self, tracker):
        """Error with 'stale element reference' (contains 'element') should
        be stale_ref, not element_not_found."""
        result = tracker._classify_error(
            "stale element reference: element is no longer in the DOM"
        )
        assert result == "stale_ref"

    def test_element_not_found_vs_connection(self, tracker):
        """Error with 'element not found' AND 'connection' should be
        element_not_found because it is checked before connection_error."""
        result = tracker._classify_error(
            "element not found after connection was established"
        )
        assert result == "element_not_found"

    def test_invalid_keyword_beats_element_not_found(self, tracker):
        """Error with both 'keyword not found' and 'element' should be
        invalid_keyword because it is checked first."""
        result = tracker._classify_error(
            "keyword not found: the element keyword does not exist"
        )
        assert result == "invalid_keyword"

    def test_strict_mode_beats_invalid_selector(self, tracker):
        """strict_mode is checked before invalid_selector."""
        result = tracker._classify_error(
            "strict mode violation: invalid selector locator resolved to 2"
        )
        assert result == "strict_mode"

    def test_element_intercept_beats_element_not_interactable(self, tracker):
        """element_intercept is checked before element_not_interactable."""
        result = tracker._classify_error(
            "element click intercepted: element not interactable"
        )
        assert result == "element_intercept"

    def test_frame_error_beats_timeout(self, tracker):
        """frame_error is checked before timeout."""
        result = tracker._classify_error(
            "Timeout: no such frame available"
        )
        assert result == "frame_error"

    def test_invalid_state_beats_element_not_found(self, tracker):
        """invalid_state is checked before element_not_found."""
        result = tracker._classify_error(
            "invalid element state: element not found in editable mode"
        )
        assert result == "invalid_state"


# ---------------------------------------------------------------------------
# Case insensitivity tests
# ---------------------------------------------------------------------------
class TestCaseInsensitivity:
    """Tests that error classification is case-insensitive."""

    def test_uppercase_element_click_intercepted(self, tracker):
        result = tracker._classify_error("ELEMENT CLICK INTERCEPTED")
        assert result == "element_intercept"

    def test_mixed_case_strict_mode_violation(self, tracker):
        result = tracker._classify_error("Strict Mode Violation in test")
        assert result == "strict_mode"

    def test_uppercase_timeout(self, tracker):
        result = tracker._classify_error("TIMEOUT EXCEEDED")
        assert result == "timeout"

    def test_mixed_case_stale_element(self, tracker):
        result = tracker._classify_error("Stale Element Reference Exception")
        assert result == "stale_ref"

    def test_uppercase_no_such_element(self, tracker):
        result = tracker._classify_error("NO SUCH ELEMENT found on page")
        assert result == "element_not_found"

    def test_camelcase_nosuchframeexception(self, tracker):
        """NoSuchFrameException lowercases to nosuchframeexception, which
        matches the pattern."""
        result = tracker._classify_error("NoSuchFrameException: frame gone")
        assert result == "frame_error"

    def test_camelcase_elementnotinteractableexception(self, tracker):
        """ElementNotInteractableException lowercases to match the pattern."""
        result = tracker._classify_error(
            "ElementNotInteractableException: cannot interact"
        )
        assert result == "element_not_interactable"

    def test_uppercase_keyword_not_found(self, tracker):
        result = tracker._classify_error("KEYWORD NOT FOUND: MyKeyword")
        assert result == "invalid_keyword"

    def test_mixed_case_connection(self, tracker):
        result = tracker._classify_error("Connection Refused by server")
        assert result == "connection_error"

    def test_uppercase_invalid_selector(self, tracker):
        result = tracker._classify_error("INVALID SELECTOR: //broken")
        assert result == "invalid_selector"

    def test_uppercase_unexpected_alert_open(self, tracker):
        result = tracker._classify_error("UNEXPECTED ALERT OPEN in dialog")
        assert result == "alert_dialog"

    def test_uppercase_no_such_window(self, tracker):
        result = tracker._classify_error("NO SUCH WINDOW: handle invalid")
        assert result == "window_error"

    def test_uppercase_invalid_element_state(self, tracker):
        result = tracker._classify_error(
            "INVALID ELEMENT STATE: readonly field"
        )
        assert result == "invalid_state"

    def test_mixed_case_outside_viewport(self, tracker):
        result = tracker._classify_error(
            "Element Is Outside Of The Viewport"
        )
        assert result == "element_outside_viewport"

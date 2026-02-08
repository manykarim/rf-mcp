"""Unit tests for pre-validation feature in KeywordExecutor.

Tests the fast element visibility/state check that runs before keyword execution
to detect "element not visible/enabled" in ~500ms instead of waiting 10s.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from robotmcp.components.execution.keyword_executor import KeywordExecutor
from robotmcp.models.config_models import ExecutionConfig
from robotmcp.models.session_models import ExecutionSession, BrowserState
from tests.unit.helpers.rf_context_mock import rf_context_with_owner, no_rf_context


@pytest.fixture
def executor():
    """Create a KeywordExecutor with test configuration."""
    config = ExecutionConfig()
    config.PRE_VALIDATION_TIMEOUT = 500
    return KeywordExecutor(config=config)


@pytest.fixture
def mock_session():
    """Create a mock ExecutionSession with browser state."""
    session = MagicMock(spec=ExecutionSession)
    session.session_id = "test-session-123"
    session.variables = {}
    session.browser_state = MagicMock(spec=BrowserState)
    session.browser_state.active_library = "browser"
    session.imported_libraries = ["Browser", "BuiltIn"]
    return session


class TestPreValidationRequirements:
    """Tests for _requires_pre_validation method."""

    def test_click_requires_pre_validation(self, executor):
        """Click keyword should require pre-validation."""
        assert executor._requires_pre_validation("click") is True
        assert executor._requires_pre_validation("Click") is True
        assert executor._requires_pre_validation("CLICK") is True

    def test_click_element_requires_pre_validation(self, executor):
        """Click Element keyword should require pre-validation."""
        assert executor._requires_pre_validation("click element") is True

    def test_fill_text_requires_pre_validation(self, executor):
        """Fill Text keyword should require pre-validation."""
        assert executor._requires_pre_validation("fill text") is True

    def test_input_text_requires_pre_validation(self, executor):
        """Input Text keyword should require pre-validation."""
        assert executor._requires_pre_validation("input text") is True

    def test_select_options_requires_pre_validation(self, executor):
        """Select Options keyword should require pre-validation."""
        assert executor._requires_pre_validation("select options") is True

    def test_press_keys_requires_pre_validation(self, executor):
        """Press Keys keyword should require pre-validation."""
        assert executor._requires_pre_validation("press keys") is True

    def test_navigation_does_not_require_pre_validation(self, executor):
        """Navigation keywords should not require pre-validation."""
        assert executor._requires_pre_validation("go to") is False
        assert executor._requires_pre_validation("new page") is False
        assert executor._requires_pre_validation("new browser") is False

    def test_get_text_does_not_require_pre_validation(self, executor):
        """Read keywords should not require pre-validation."""
        assert executor._requires_pre_validation("get text") is False
        assert executor._requires_pre_validation("get title") is False

    def test_wait_keywords_do_not_require_pre_validation(self, executor):
        """Wait keywords should not require pre-validation (they have their own waits)."""
        assert executor._requires_pre_validation("wait for elements state") is False


class TestActionTypeMapping:
    """Tests for _get_action_type_from_keyword_for_states method."""

    def test_click_action_type(self, executor):
        """Click keywords should map to 'click' action type."""
        assert executor._get_action_type_from_keyword_for_states("click") == "click"
        assert executor._get_action_type_from_keyword_for_states("click element") == "click"
        assert executor._get_action_type_from_keyword_for_states("double click") == "click"

    def test_fill_action_type(self, executor):
        """Fill/input keywords should map to 'fill' action type."""
        assert executor._get_action_type_from_keyword_for_states("fill text") == "fill"
        assert executor._get_action_type_from_keyword_for_states("input text") == "fill"
        assert executor._get_action_type_from_keyword_for_states("type text") == "fill"

    def test_check_action_type(self, executor):
        """Check checkbox keywords should map to 'check' action type."""
        assert executor._get_action_type_from_keyword_for_states("check checkbox") == "check"
        assert executor._get_action_type_from_keyword_for_states("select checkbox") == "check"

    def test_uncheck_action_type(self, executor):
        """Uncheck checkbox keywords should map to 'uncheck' action type."""
        assert executor._get_action_type_from_keyword_for_states("uncheck checkbox") == "uncheck"

    def test_select_action_type(self, executor):
        """Select keywords should map to 'select' action type."""
        assert executor._get_action_type_from_keyword_for_states("select options") == "select"
        assert executor._get_action_type_from_keyword_for_states("select from list") == "select"


class TestLocatorExtraction:
    """Tests for _extract_locator_from_args method."""

    def test_extract_first_string_argument(self, executor):
        """Should extract first string argument as locator."""
        locator = executor._extract_locator_from_args("click", ["css=#button", "force=true"])
        assert locator == "css=#button"

    def test_empty_arguments_returns_none(self, executor):
        """Should return None for empty arguments."""
        locator = executor._extract_locator_from_args("click", [])
        assert locator is None

    def test_non_string_first_arg_returns_none(self, executor):
        """Should return None if first argument is not a string."""
        locator = executor._extract_locator_from_args("click", [123, "locator"])
        assert locator is None


class TestRequiredStates:
    """Tests for required states mapping."""

    def test_click_requires_visible_and_enabled(self, executor):
        """Click actions should require visible and enabled states."""
        states = executor.REQUIRED_STATES_FOR_ACTION["click"]
        assert "visible" in states
        assert "enabled" in states

    def test_fill_requires_visible_enabled_editable(self, executor):
        """Fill actions should require visible, enabled, and editable states."""
        states = executor.REQUIRED_STATES_FOR_ACTION["fill"]
        assert "visible" in states
        assert "enabled" in states
        assert "editable" in states

    def test_focus_requires_only_visible(self, executor):
        """Focus actions should only require visible state."""
        states = executor.REQUIRED_STATES_FOR_ACTION["focus"]
        assert "visible" in states
        assert "enabled" not in states

    def test_scroll_requires_only_attached(self, executor):
        """Scroll actions should only require attached state."""
        states = executor.REQUIRED_STATES_FOR_ACTION["scroll"]
        assert "attached" in states
        assert "visible" not in states


class TestPreValidationBrowserLibrary:
    """Tests for Browser Library pre-validation."""

    @pytest.mark.asyncio
    async def test_pre_validate_passes_for_visible_enabled_element(self, executor, mock_session):
        """Pre-validation should pass when element has required states."""
        with rf_context_with_owner("Browser"), \
             patch.object(executor, '_run_browser_get_states') as mock_get_states:
            mock_get_states.return_value = (["ElementState.visible", "ElementState.enabled"], None)

            is_valid, error, details = await executor._pre_validate_element(
                "css=#button", mock_session, "click"
            )

            assert is_valid is True
            assert error is None
            assert "visible" in details.get("current_states", [])
            assert "enabled" in details.get("current_states", [])

    @pytest.mark.asyncio
    async def test_pre_validate_fails_for_hidden_element(self, executor, mock_session):
        """Pre-validation should fail when element is hidden."""
        with rf_context_with_owner("Browser"), \
             patch.object(executor, '_run_browser_get_states') as mock_get_states:
            mock_get_states.return_value = (["ElementState.attached", "ElementState.enabled"], None)

            is_valid, error, details = await executor._pre_validate_element(
                "css=#hidden-button", mock_session, "click"
            )

            assert is_valid is False
            assert "visible" in error
            assert "visible" in details.get("missing_states", [])

    @pytest.mark.asyncio
    async def test_pre_validate_fails_for_disabled_element(self, executor, mock_session):
        """Pre-validation should fail when element is disabled."""
        with rf_context_with_owner("Browser"), \
             patch.object(executor, '_run_browser_get_states') as mock_get_states:
            mock_get_states.return_value = (["ElementState.visible", "ElementState.disabled"], None)

            is_valid, error, details = await executor._pre_validate_element(
                "css=#disabled-button", mock_session, "click"
            )

            assert is_valid is False
            assert "enabled" in error
            assert "enabled" in details.get("missing_states", [])

    @pytest.mark.asyncio
    async def test_pre_validate_fails_for_not_found_element(self, executor, mock_session):
        """Pre-validation should fail when element is not found."""
        with rf_context_with_owner("Browser"), \
             patch.object(executor, '_run_browser_get_states') as mock_get_states:
            mock_get_states.return_value = (None, "Element not found: css=#non-existent")

            is_valid, error, details = await executor._pre_validate_element(
                "css=#non-existent", mock_session, "click"
            )

            assert is_valid is False
            assert "not found" in error.lower()

    @pytest.mark.asyncio
    async def test_pre_validate_handles_multiple_elements_strict_mode(self, executor, mock_session):
        """Pre-validation should report multiple elements error instead of 'not found'."""
        with rf_context_with_owner("Browser"), \
             patch.object(executor, '_run_browser_get_states') as mock_get_states:
            mock_get_states.return_value = (
                None,
                "Multiple elements found for 'id=nav_automobile'. "
                "Tried visible filter and nth=0 but both failed. "
                "Original error: strict mode violation: resolved to 2 elements"
            )

            is_valid, error, details = await executor._pre_validate_element(
                "id=nav_automobile", mock_session, "click"
            )

            assert is_valid is False
            assert "multiple elements" in error.lower()
            assert "strict mode" in error.lower()


class TestPreValidationSkip:
    """Tests for pre-validation skip conditions."""

    @pytest.mark.asyncio
    async def test_pre_validate_skips_when_no_active_browser(self, executor, mock_session):
        """Pre-validation should be skipped when no browser is active."""
        mock_session.browser_state.active_library = None
        # Ensure imported_libraries doesn't contain any UI library
        mock_session.imported_libraries = ["BuiltIn", "Collections"]

        is_valid, error, details = await executor._pre_validate_element(
            "css=#button", mock_session, "click"
        )

        assert is_valid is True
        assert error is None
        assert details.get("skipped") is True
        assert details.get("reason") == "no_active_browser"

    def test_pre_validation_disabled_via_env(self):
        """Pre-validation should respect ROBOTMCP_PRE_VALIDATION env var."""
        with patch.dict('os.environ', {'ROBOTMCP_PRE_VALIDATION': '0'}):
            config = ExecutionConfig()
            executor = KeywordExecutor(config=config)
            assert executor.pre_validation_enabled is False


class TestPreValidationTimeout:
    """Tests for pre-validation timeout behavior."""

    def test_default_pre_validation_timeout(self, executor):
        """Should use PRE_VALIDATION_TIMEOUT from config."""
        assert executor.config.PRE_VALIDATION_TIMEOUT == 500

    @pytest.mark.asyncio
    async def test_custom_timeout_passed_to_browser_get_states(self, executor, mock_session):
        """Custom timeout should be passed to Browser Library."""
        with rf_context_with_owner("Browser"), \
             patch.object(executor, '_run_browser_get_states') as mock_get_states:
            mock_get_states.return_value = (["ElementState.visible", "ElementState.enabled"], None)

            await executor._pre_validate_element(
                "css=#button", mock_session, "click", timeout_ms=1000
            )

            mock_get_states.assert_called_once()
            call_args = mock_get_states.call_args
            assert "1000ms" in call_args[0]


# =============================================================================
# Tests for Actual Keyword Calls in Pre-Validation Methods
# =============================================================================


class TestBrowserGetStatesKeywordCalls:
    """Tests that verify the actual BuiltIn.run_keyword calls in _run_browser_get_states.

    These tests mock BuiltIn to verify:
    1. Correct keyword names are used
    2. Correct argument count and format is passed
    3. Return values are handled correctly
    """

    @pytest.fixture
    def executor(self):
        """Create a KeywordExecutor with test configuration."""
        config = ExecutionConfig()
        config.PRE_VALIDATION_TIMEOUT = 500
        return KeywordExecutor(config=config)

    def test_set_browser_timeout_returns_previous_value(self, executor):
        """Test that Set Browser Timeout returns previous value (no separate Get call needed).

        Note: Browser Library's Set Browser Timeout returns the previous timeout,
        so we don't need a separate Get Browser Timeout call.
        """
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            # Set up return values for the flow
            # Set Browser Timeout returns the previous timeout value
            mock_builtin.run_keyword.side_effect = [
                "30s",  # Set Browser Timeout returns original timeout
                ["ElementState.visible", "ElementState.enabled"],  # Get Element States
                None,   # Restore Browser Timeout (in finally)
            ]

            result, error = executor._run_browser_get_states("css=#button", "500ms")

            # Verify first call is Set Browser Timeout (which returns previous value)
            calls = mock_builtin.run_keyword.call_args_list
            assert calls[0] == (("Browser.Set Browser Timeout", "500ms"), {})

    def test_get_element_states_is_second_call(self, executor):
        """Test that Get Element States is the second call after Set Browser Timeout."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            mock_builtin.run_keyword.side_effect = [
                "30s",  # Set Browser Timeout returns previous timeout
                ["ElementState.visible"],  # Get Element States
                None,   # Restore timeout
            ]

            executor._run_browser_get_states("css=#button", "500ms")

            calls = mock_builtin.run_keyword.call_args_list
            # Second call should be Get Element States with the locator
            assert calls[1] == (("Browser.Get Element States", "css=#button"), {})

    def test_get_element_states_keyword_call_with_library_prefix(self, executor):
        """Test that Browser.Get Element States is called with library prefix."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            mock_builtin.run_keyword.side_effect = [
                "30s",  # Set Browser Timeout returns previous timeout
                ["ElementState.visible", "ElementState.enabled"],  # Get Element States with prefix
                None,   # Restore timeout
            ]

            result, error = executor._run_browser_get_states("css=#button", "500ms")

            calls = mock_builtin.run_keyword.call_args_list
            # Second call should be Get Element States with library prefix and locator
            assert calls[1] == (("Browser.Get Element States", "css=#button"), {})
            assert result == ["ElementState.visible", "ElementState.enabled"]
            assert error is None

    def test_get_element_states_fallback_without_prefix(self, executor):
        """Test that Get Element States without prefix is tried if prefixed version fails."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            set_timeout_call_count = [0]  # Use list for mutability in closure

            def run_keyword_side_effect(*args):
                if args[0] == "Browser.Set Browser Timeout":
                    set_timeout_call_count[0] += 1
                    if set_timeout_call_count[0] == 1:
                        return "30s"  # First call returns previous timeout
                    return None
                elif args[0] == "Browser.Get Element States":
                    raise Exception("Keyword 'Browser.Get Element States' not found")
                elif args[0] == "Get Element States":
                    return ["ElementState.visible"]
                return None

            mock_builtin.run_keyword.side_effect = run_keyword_side_effect

            result, error = executor._run_browser_get_states("css=#button", "500ms")

            # Verify both versions were tried
            calls = [call[0][0] for call in mock_builtin.run_keyword.call_args_list]
            assert "Browser.Get Element States" in calls
            assert "Get Element States" in calls
            assert result == ["ElementState.visible"]
            assert error is None

    def test_get_element_states_returns_error_on_failure(self, executor):
        """Test that error is returned when both Get Element States calls fail."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            set_timeout_call_count = [0]

            def run_keyword_side_effect(*args):
                if args[0] == "Browser.Set Browser Timeout":
                    set_timeout_call_count[0] += 1
                    if set_timeout_call_count[0] == 1:
                        return "30s"  # First call returns previous timeout
                    return None
                elif args[0] in ["Browser.Get Element States", "Get Element States"]:
                    raise Exception("Element not found: css=#missing")
                return None

            mock_builtin.run_keyword.side_effect = run_keyword_side_effect

            result, error = executor._run_browser_get_states("css=#missing", "500ms")

            assert result is None
            assert "Element not found" in error

    def test_timeout_restored_after_success(self, executor):
        """Test that original timeout is restored after successful Get Element States.

        Note: Set Browser Timeout returns the previous timeout value, so:
        1. First Set Browser Timeout sets new timeout, returns original
        2. Get Element States gets the states
        3. Second Set Browser Timeout restores original (in finally block)
        """
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            call_sequence = []
            original_timeout_returned = False

            def run_keyword_side_effect(*args):
                nonlocal original_timeout_returned
                call_sequence.append(args)
                if args[0] == "Browser.Set Browser Timeout":
                    if not original_timeout_returned:
                        # First call returns the original timeout
                        original_timeout_returned = True
                        return "original_timeout"
                    return None
                elif args[0] == "Browser.Get Element States":
                    return ["ElementState.visible"]
                return None

            mock_builtin.run_keyword.side_effect = run_keyword_side_effect

            executor._run_browser_get_states("css=#button", "500ms")

            # Verify 3 calls total: set timeout, get states, restore timeout
            assert len(call_sequence) >= 3
            # Should have exactly 2 Set Browser Timeout calls
            set_timeout_calls = [c for c in call_sequence if c[0] == "Browser.Set Browser Timeout"]
            assert len(set_timeout_calls) == 2
            # First call sets the new timeout
            assert set_timeout_calls[0] == ("Browser.Set Browser Timeout", "500ms")
            # Last Set Browser Timeout should restore original timeout
            assert set_timeout_calls[-1] == ("Browser.Set Browser Timeout", "original_timeout")

    def test_strict_mode_violation_triggers_visible_filter(self, executor):
        """Test that strict mode violation triggers visible=true filter retry."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            locators_tried = []
            set_timeout_call_count = [0]

            def run_keyword_side_effect(*args):
                if args[0] == "Browser.Set Browser Timeout":
                    set_timeout_call_count[0] += 1
                    if set_timeout_call_count[0] == 1:
                        return "30s"  # First call returns previous timeout
                    return None
                elif args[0] in ["Browser.Get Element States", "Get Element States"]:
                    locator = args[1]
                    locators_tried.append(locator)
                    if ">> visible=true" in locator:
                        return ["ElementState.visible"]
                    elif ">> nth=0" in locator:
                        return ["ElementState.visible"]
                    else:
                        raise Exception("strict mode violation: resolved to 2 elements")
                return None

            mock_builtin.run_keyword.side_effect = run_keyword_side_effect

            result, error = executor._run_browser_get_states("id=nav_automobile", "500ms")

            # Should have tried: original, with visible=true filter, and possibly nth=0
            assert "id=nav_automobile" in locators_tried
            assert any(">> visible=true" in loc for loc in locators_tried)
            assert result == ["ElementState.visible"]


class TestSeleniumStateCheckKeywordCalls:
    """Tests that verify the actual BuiltIn.run_keyword calls in _run_selenium_state_check.

    These tests mock BuiltIn to verify:
    1. Correct SeleniumLibrary keyword names are used
    2. Correct argument count and format is passed
    3. Return values are handled correctly
    """

    @pytest.fixture
    def executor(self):
        """Create a KeywordExecutor with test configuration."""
        config = ExecutionConfig()
        return KeywordExecutor(config=config)

    def test_set_selenium_implicit_wait_keyword_call(self, executor):
        """Test that SeleniumLibrary.Set Selenium Implicit Wait is called correctly."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            mock_element = MagicMock()
            mock_builtin.run_keyword.side_effect = [
                None,         # Set Selenium Implicit Wait
                [mock_element],  # Get WebElements
                ["attached", "visible", "enabled"],  # Execute JavaScript
            ]

            executor._run_selenium_state_check("css=#button", {"visible", "enabled"}, 5000)

            calls = mock_builtin.run_keyword.call_args_list
            # First call should be Set Selenium Implicit Wait with timeout in seconds
            assert calls[0] == (("SeleniumLibrary.Set Selenium Implicit Wait", "5.0s"), {})

    def test_get_webelements_keyword_call(self, executor):
        """Test that SeleniumLibrary.Get WebElements is called with correct locator."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            mock_element = MagicMock()
            mock_builtin.run_keyword.side_effect = [
                None,         # Set Selenium Implicit Wait
                [mock_element],  # Get WebElements
                ["attached", "visible", "enabled"],  # Execute JavaScript
            ]

            executor._run_selenium_state_check("xpath=//button[@id='submit']", {"visible"}, 5000)

            calls = mock_builtin.run_keyword.call_args_list
            # Second call should be Get WebElements with the locator
            assert calls[1] == (("SeleniumLibrary.Get WebElements", "xpath=//button[@id='submit']"), {})

    def test_get_webelement_fallback_on_empty_elements(self, executor):
        """Test that SeleniumLibrary.Get WebElement is tried when Get WebElements returns empty."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            mock_element = MagicMock()

            def run_keyword_side_effect(*args):
                if args[0] == "SeleniumLibrary.Set Selenium Implicit Wait":
                    return None
                elif args[0] == "SeleniumLibrary.Get WebElements":
                    return []  # Empty - no elements found
                elif args[0] == "SeleniumLibrary.Get WebElement":
                    return mock_element
                elif args[0] == "SeleniumLibrary.Execute Javascript":
                    return ["attached", "visible"]
                return None

            mock_builtin.run_keyword.side_effect = run_keyword_side_effect

            result = executor._run_selenium_state_check("css=#button", {"visible"}, 5000)

            calls = [call[0][0] for call in mock_builtin.run_keyword.call_args_list]
            assert "SeleniumLibrary.Get WebElements" in calls
            assert "SeleniumLibrary.Get WebElement" in calls
            assert result["valid"] is True

    def test_execute_javascript_for_state_check(self, executor):
        """Test that SeleniumLibrary.Execute Javascript is called with correct JS and element."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            mock_element = MagicMock()

            call_args_list = []

            def run_keyword_side_effect(*args):
                call_args_list.append(args)
                if args[0] == "SeleniumLibrary.Set Selenium Implicit Wait":
                    return None
                elif args[0] == "SeleniumLibrary.Get WebElements":
                    return [mock_element]
                elif args[0] == "SeleniumLibrary.Execute Javascript":
                    return ["attached", "visible", "enabled"]
                return None

            mock_builtin.run_keyword.side_effect = run_keyword_side_effect

            result = executor._run_selenium_state_check("css=#button", {"visible", "enabled"}, 5000)

            # Find the Execute Javascript call
            js_calls = [c for c in call_args_list if c[0] == "SeleniumLibrary.Execute Javascript"]
            assert len(js_calls) == 1

            # Verify it has JavaScript code, ARGUMENTS marker, and the element
            js_call = js_calls[0]
            assert "SeleniumLibrary.Execute Javascript" == js_call[0]
            assert "var el = arguments[0]" in js_call[1]  # The JavaScript code
            assert js_call[2] == "ARGUMENTS"  # Required marker for passing WebElements
            assert js_call[3] == mock_element  # The element passed as argument

    def test_execute_javascript_for_visibility_check_multiple_elements(self, executor):
        """Test that Execute JavaScript is called to check visibility for multiple elements."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            mock_element1 = MagicMock()
            mock_element2 = MagicMock()

            js_call_count = [0]

            def run_keyword_side_effect(*args):
                if args[0] == "SeleniumLibrary.Set Selenium Implicit Wait":
                    return None
                elif args[0] == "SeleniumLibrary.Get WebElements":
                    return [mock_element1, mock_element2]  # Multiple elements
                elif args[0] == "SeleniumLibrary.Execute Javascript":
                    js_call_count[0] += 1
                    js_code = args[1]
                    if "getBoundingClientRect" in js_code and "return style.display" in js_code:
                        # This is the visibility check JS
                        if js_call_count[0] == 1:
                            return True  # First element is visible
                        return False
                    else:
                        # This is the state check JS
                        return ["attached", "visible", "enabled"]
                return None

            mock_builtin.run_keyword.side_effect = run_keyword_side_effect

            result = executor._run_selenium_state_check("css=.button", {"visible"}, 5000)

            # Should have called Execute JavaScript for visibility check and state check
            assert js_call_count[0] >= 1
            assert result["valid"] is True

    def test_returns_error_when_element_not_found(self, executor):
        """Test that error dict is returned when element is not found."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            def run_keyword_side_effect(*args):
                if args[0] == "SeleniumLibrary.Set Selenium Implicit Wait":
                    return None
                elif args[0] == "SeleniumLibrary.Get WebElements":
                    return []
                elif args[0] == "SeleniumLibrary.Get WebElement":
                    raise Exception("Element not found")
                return None

            mock_builtin.run_keyword.side_effect = run_keyword_side_effect

            result = executor._run_selenium_state_check("css=#nonexistent", {"visible"}, 5000)

            assert result["valid"] is False
            assert "not found" in result["error"].lower()

    def test_returns_missing_states_correctly(self, executor):
        """Test that missing states are correctly reported."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            mock_element = MagicMock()

            def run_keyword_side_effect(*args):
                if args[0] == "SeleniumLibrary.Set Selenium Implicit Wait":
                    return None
                elif args[0] == "SeleniumLibrary.Get WebElements":
                    return [mock_element]
                elif args[0] == "SeleniumLibrary.Execute Javascript":
                    return ["attached"]  # Only attached, missing visible and enabled
                return None

            mock_builtin.run_keyword.side_effect = run_keyword_side_effect

            result = executor._run_selenium_state_check("css=#button", {"visible", "enabled"}, 5000)

            assert result["valid"] is False
            assert "visible" in result["missing"]
            assert "enabled" in result["missing"]


class TestAppiumStateCheckKeywordCalls:
    """Tests that verify the actual BuiltIn.run_keyword calls in _run_appium_state_check.

    These tests mock BuiltIn to verify:
    1. Correct AppiumLibrary keyword names are used
    2. Correct argument count and format is passed
    3. Return values are handled correctly
    """

    @pytest.fixture
    def executor(self):
        """Create a KeywordExecutor with test configuration."""
        config = ExecutionConfig()
        return KeywordExecutor(config=config)

    def test_get_webelements_appium_keyword_call(self, executor):
        """Test that AppiumLibrary.Get Webelements is called with correct locator.

        Note: The code uses 'Get Webelements' (lowercase 'e') which is valid since
        Robot Framework keywords are case-insensitive.
        """
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            mock_element = MagicMock()
            mock_element.is_displayed.return_value = True
            mock_element.is_enabled.return_value = True
            mock_element.tag_name = "button"

            mock_builtin.run_keyword.return_value = [mock_element]

            executor._run_appium_state_check("accessibility_id=Submit", {"visible"}, 5000)

            calls = mock_builtin.run_keyword.call_args_list
            # First call is Set Appium Implicit Wait (P0-3 fix: timeout control)
            assert calls[0] == (("AppiumLibrary.Set Appium Implicit Wait", "5.0"), {})
            # Second call is Get Webelements (lowercase 'e' - RF keywords are case-insensitive)
            assert calls[1] == (("AppiumLibrary.Get Webelements", "accessibility_id=Submit"), {})

    def test_get_webelement_appium_fallback(self, executor):
        """Test that AppiumLibrary.Get Webelement is tried when Get Webelements returns empty.

        Note: The code uses 'Get Webelements'/'Get Webelement' (lowercase 'e') which is valid
        since Robot Framework keywords are case-insensitive.
        """
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            mock_element = MagicMock()
            mock_element.is_displayed.return_value = True
            mock_element.is_enabled.return_value = True
            mock_element.tag_name = "button"

            def run_keyword_side_effect(*args):
                # Note: Code uses 'Get Webelements' (lowercase 'e')
                if args[0] == "AppiumLibrary.Get Webelements":
                    return []  # Empty
                elif args[0] == "AppiumLibrary.Get Webelement":
                    return mock_element
                return None

            mock_builtin.run_keyword.side_effect = run_keyword_side_effect

            result = executor._run_appium_state_check("accessibility_id=Submit", {"visible"}, 5000)

            calls = [call[0][0] for call in mock_builtin.run_keyword.call_args_list]
            # Note: Code uses lowercase 'e' in 'elements' and 'element'
            assert "AppiumLibrary.Get Webelements" in calls
            assert "AppiumLibrary.Get Webelement" in calls
            assert result["valid"] is True

    def test_element_is_displayed_method_called(self, executor):
        """Test that element.is_displayed() is called for visibility check."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            mock_element = MagicMock()
            mock_element.is_displayed.return_value = True
            mock_element.is_enabled.return_value = True
            mock_element.tag_name = "button"

            mock_builtin.run_keyword.return_value = [mock_element]

            result = executor._run_appium_state_check("accessibility_id=Submit", {"visible"}, 5000)

            mock_element.is_displayed.assert_called()
            assert "visible" in result["states"]

    def test_element_is_enabled_method_called(self, executor):
        """Test that element.is_enabled() is called for enabled check."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            mock_element = MagicMock()
            mock_element.is_displayed.return_value = True
            mock_element.is_enabled.return_value = True
            mock_element.tag_name = "button"

            mock_builtin.run_keyword.return_value = [mock_element]

            result = executor._run_appium_state_check("accessibility_id=Submit", {"enabled"}, 5000)

            mock_element.is_enabled.assert_called()
            assert "enabled" in result["states"]

    def test_multiple_elements_finds_first_visible(self, executor):
        """Test that with multiple elements, the first visible one is used."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            mock_element1 = MagicMock()
            mock_element1.is_displayed.return_value = False  # Not visible
            mock_element1.is_enabled.return_value = True
            mock_element1.tag_name = "button"

            mock_element2 = MagicMock()
            mock_element2.is_displayed.return_value = True  # Visible
            mock_element2.is_enabled.return_value = True
            mock_element2.tag_name = "button"

            mock_builtin.run_keyword.return_value = [mock_element1, mock_element2]

            result = executor._run_appium_state_check("class=android.widget.Button", {"visible"}, 5000)

            # Should have checked both elements' is_displayed()
            mock_element1.is_displayed.assert_called()
            mock_element2.is_displayed.assert_called()
            # Result should be valid since second element is visible
            assert result["valid"] is True

    def test_editable_state_for_edittext_elements(self, executor):
        """Test that editable state is detected for EditText elements."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            mock_element = MagicMock()
            mock_element.is_displayed.return_value = True
            mock_element.is_enabled.return_value = True
            mock_element.tag_name = "edittext"  # Android EditText

            mock_builtin.run_keyword.return_value = [mock_element]

            result = executor._run_appium_state_check("class=android.widget.EditText", {"editable"}, 5000)

            assert result["valid"] is True
            assert "editable" in result["states"]

    def test_returns_error_when_element_not_found(self, executor):
        """Test that error dict is returned when Appium element is not found."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            def run_keyword_side_effect(*args):
                if args[0] == "AppiumLibrary.Get WebElements":
                    return []
                elif args[0] == "AppiumLibrary.Get WebElement":
                    raise Exception("Element not found")
                return None

            mock_builtin.run_keyword.side_effect = run_keyword_side_effect

            result = executor._run_appium_state_check("accessibility_id=NonExistent", {"visible"}, 5000)

            assert result["valid"] is False
            assert "not found" in result["error"].lower()

    def test_returns_missing_states_correctly(self, executor):
        """Test that missing states are correctly reported for Appium."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            mock_element = MagicMock()
            mock_element.is_displayed.return_value = False  # Not visible
            mock_element.is_enabled.return_value = True
            mock_element.tag_name = "button"

            mock_builtin.run_keyword.return_value = [mock_element]

            result = executor._run_appium_state_check("accessibility_id=Submit", {"visible", "enabled"}, 5000)

            assert result["valid"] is False
            assert "visible" in result["missing"]

    def test_handles_exception_in_is_displayed(self, executor):
        """Test that exceptions in is_displayed() are handled gracefully."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            mock_element = MagicMock()
            mock_element.is_displayed.side_effect = Exception("Stale element")
            mock_element.is_enabled.return_value = True
            mock_element.tag_name = "button"

            mock_builtin.run_keyword.return_value = [mock_element]

            # Should not raise, but handle gracefully
            result = executor._run_appium_state_check("accessibility_id=Submit", {"visible"}, 5000)

            # When is_displayed fails, element is treated as not visible
            # The method should still return a valid dict
            assert "valid" in result


class TestKeywordSignatureErrors:
    """Tests specifically designed to catch keyword signature errors.

    These tests verify that the correct number of arguments are passed
    to each keyword and in the correct format.
    """

    @pytest.fixture
    def executor(self):
        """Create a KeywordExecutor with test configuration."""
        config = ExecutionConfig()
        return KeywordExecutor(config=config)

    def test_set_browser_timeout_returns_previous_for_restore(self, executor):
        """Verify Set Browser Timeout returns previous value which is used for restore.

        Note: Browser Library's Set Browser Timeout returns the previous timeout,
        so we don't need a separate Get Browser Timeout call.
        """
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            call_args_captured = []
            set_call_count = [0]

            def capture_calls(*args, **kwargs):
                call_args_captured.append((args, kwargs))
                if args[0] == "Browser.Set Browser Timeout":
                    set_call_count[0] += 1
                    if set_call_count[0] == 1:
                        return "original_30s"  # First call returns previous timeout
                    return None
                elif args[0] == "Browser.Get Element States":
                    return ["ElementState.visible"]
                return None

            mock_builtin.run_keyword.side_effect = capture_calls

            executor._run_browser_get_states("css=#button", "500ms")

            # Verify no Get Browser Timeout call was made
            get_timeout_calls = [c for c in call_args_captured if c[0][0] == "Browser.Get Browser Timeout"]
            assert len(get_timeout_calls) == 0

            # Verify Set Browser Timeout was called twice
            set_timeout_calls = [c for c in call_args_captured if c[0][0] == "Browser.Set Browser Timeout"]
            assert len(set_timeout_calls) == 2
            # First call sets new timeout
            assert set_timeout_calls[0][0] == ("Browser.Set Browser Timeout", "500ms")
            # Second call restores original (from return value of first call)
            assert set_timeout_calls[1][0] == ("Browser.Set Browser Timeout", "original_30s")

    def test_browser_set_timeout_takes_one_argument(self, executor):
        """Verify Browser.Set Browser Timeout is called with exactly one argument (the timeout)."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            call_args_captured = []
            set_call_count = [0]

            def capture_calls(*args, **kwargs):
                call_args_captured.append((args, kwargs))
                if args[0] == "Browser.Set Browser Timeout":
                    set_call_count[0] += 1
                    if set_call_count[0] == 1:
                        return "30s"  # First call returns previous timeout
                    return None
                elif args[0] == "Browser.Get Element States":
                    return ["ElementState.visible"]
                return None

            mock_builtin.run_keyword.side_effect = capture_calls

            executor._run_browser_get_states("css=#button", "500ms")

            # Find the Set Browser Timeout calls
            set_timeout_calls = [c for c in call_args_captured if c[0][0] == "Browser.Set Browser Timeout"]
            assert len(set_timeout_calls) >= 1
            # First call should set the new timeout - 2 args (keyword name + timeout value)
            assert len(set_timeout_calls[0][0]) == 2
            assert set_timeout_calls[0][0][1] == "500ms"

    def test_browser_get_element_states_takes_one_argument(self, executor):
        """Verify Browser.Get Element States is called with exactly one argument (the locator)."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            call_args_captured = []
            set_call_count = [0]

            def capture_calls(*args, **kwargs):
                call_args_captured.append((args, kwargs))
                if args[0] == "Browser.Set Browser Timeout":
                    set_call_count[0] += 1
                    if set_call_count[0] == 1:
                        return "30s"  # First call returns previous timeout
                    return None
                elif args[0] == "Browser.Get Element States":
                    return ["ElementState.visible"]
                return None

            mock_builtin.run_keyword.side_effect = capture_calls

            executor._run_browser_get_states("css=#my-button", "500ms")

            # Find the Get Element States call
            get_states_calls = [c for c in call_args_captured if c[0][0] == "Browser.Get Element States"]
            assert len(get_states_calls) == 1
            # Should have exactly 2 args (keyword name + locator)
            assert len(get_states_calls[0][0]) == 2
            assert get_states_calls[0][0][1] == "css=#my-button"

    def test_selenium_implicit_wait_argument_format(self, executor):
        """Verify Set Selenium Implicit Wait receives timeout in correct format (e.g., '5.0s')."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            call_args_captured = []

            def capture_calls(*args, **kwargs):
                call_args_captured.append((args, kwargs))
                if args[0] == "SeleniumLibrary.Set Selenium Implicit Wait":
                    # P0-1 fix: Set Selenium Implicit Wait returns previous value
                    return "10 seconds"
                elif args[0] == "SeleniumLibrary.Get WebElements":
                    mock_element = MagicMock()
                    return [mock_element]
                elif args[0] == "SeleniumLibrary.Execute Javascript":
                    return ["visible"]
                return None

            mock_builtin.run_keyword.side_effect = capture_calls

            executor._run_selenium_state_check("css=#button", {"visible"}, 5000)

            # Find the Set Selenium Implicit Wait calls (should have 2: set and restore)
            wait_calls = [c for c in call_args_captured if c[0][0] == "SeleniumLibrary.Set Selenium Implicit Wait"]
            assert len(wait_calls) == 2  # One to set, one to restore
            # First call should set the new timeout
            assert len(wait_calls[0][0]) == 2
            timeout_arg = wait_calls[0][0][1]
            assert timeout_arg == "5.0s"
            # Second call should restore to captured previous value (P0-1 fix)
            assert wait_calls[1][0][1] == "10 seconds"

    def test_selenium_execute_javascript_argument_order(self, executor):
        """Verify Execute JavaScript receives (js_code, element) in correct order."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            mock_element = MagicMock()
            call_args_captured = []

            def capture_calls(*args, **kwargs):
                call_args_captured.append((args, kwargs))
                if args[0] == "SeleniumLibrary.Set Selenium Implicit Wait":
                    return None
                elif args[0] == "SeleniumLibrary.Get WebElements":
                    return [mock_element]
                elif args[0] == "SeleniumLibrary.Execute Javascript":
                    return ["visible", "enabled"]
                return None

            mock_builtin.run_keyword.side_effect = capture_calls

            executor._run_selenium_state_check("css=#button", {"visible"}, 5000)

            # Find the Execute JavaScript call for state checking
            js_calls = [c for c in call_args_captured
                       if c[0][0] == "SeleniumLibrary.Execute Javascript" and "var el = arguments[0]" in c[0][1]]
            assert len(js_calls) >= 1

            # Should have 4 args (keyword name, JS code, "ARGUMENTS", element)
            js_call = js_calls[0]
            assert len(js_call[0]) == 4
            assert js_call[0][0] == "SeleniumLibrary.Execute Javascript"
            assert isinstance(js_call[0][1], str)  # JS code
            assert js_call[0][2] == "ARGUMENTS"  # Required marker for passing WebElements
            assert js_call[0][3] == mock_element  # Element object

    def test_appium_get_webelements_takes_one_argument(self, executor):
        """Verify AppiumLibrary.Get Webelements is called with exactly one argument (locator).

        Note: The code uses 'Get Webelements' (lowercase 'e') which is valid since
        Robot Framework keywords are case-insensitive.
        """
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            call_args_captured = []
            mock_element = MagicMock()
            mock_element.is_displayed.return_value = True
            mock_element.is_enabled.return_value = True
            mock_element.tag_name = "button"

            def capture_calls(*args, **kwargs):
                call_args_captured.append((args, kwargs))
                # Note: Code uses 'Get Webelements' (lowercase 'e')
                if args[0] == "AppiumLibrary.Get Webelements":
                    return [mock_element]
                return None

            mock_builtin.run_keyword.side_effect = capture_calls

            executor._run_appium_state_check("accessibility_id=Submit", {"visible"}, 5000)

            # Find the Get Webelements call (lowercase 'e')
            get_elements_calls = [c for c in call_args_captured if c[0][0] == "AppiumLibrary.Get Webelements"]
            assert len(get_elements_calls) == 1
            # Should have 2 args (keyword name + locator)
            assert len(get_elements_calls[0][0]) == 2
            assert get_elements_calls[0][0][1] == "accessibility_id=Submit"
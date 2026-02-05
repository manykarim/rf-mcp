"""Tests for timeout injection in KeywordExecutor.

This module tests the integration of TimeoutPolicy with KeywordExecutor,
ensuring that timeouts are properly classified and injected into keyword
arguments for Browser Library and SeleniumLibrary keywords.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from robotmcp.components.execution.keyword_executor import KeywordExecutor
from robotmcp.models.session_models import ExecutionSession
from robotmcp.models.config_models import ExecutionConfig
from robotmcp.domains.timeout import ActionType, TimeoutPolicy


class TestTimeoutInjection:
    """Tests for the _inject_timeout_into_arguments method."""

    @pytest.fixture
    def executor(self):
        """Create a KeywordExecutor instance for testing."""
        return KeywordExecutor(config=ExecutionConfig())

    @pytest.fixture
    def mock_session(self):
        """Create a mock ExecutionSession."""
        session = MagicMock(spec=ExecutionSession)
        session.session_id = "test_session"
        session.browser_state = MagicMock()
        session.browser_state.active_library = "browser"
        return session

    def test_no_injection_when_timeout_is_none(self, executor, mock_session):
        """Test that no timeout is injected when timeout_ms is None."""
        arguments = ["#button", "Click"]
        result = executor._inject_timeout_into_arguments(
            "click", arguments, None, mock_session
        )
        assert result == arguments

    def test_no_injection_when_timeout_already_present(self, executor, mock_session):
        """Test that no timeout is injected when already present in arguments."""
        arguments = ["#button", "timeout=3000"]
        result = executor._inject_timeout_into_arguments(
            "click", arguments, 5000, mock_session
        )
        assert result == arguments

    def test_browser_library_click_no_timeout_injection(self, executor, mock_session):
        """Test that Browser Library click does NOT get timeout injected.

        Browser Library action keywords use global browser timeout,
        not per-keyword timeout parameter.
        """
        arguments = ["#button"]
        result = executor._inject_timeout_into_arguments(
            "click", arguments, 5000, mock_session
        )
        # Click does NOT accept timeout parameter
        assert result == arguments

    def test_browser_library_wait_for_elements_state_timeout_injection(self, executor, mock_session):
        """Test timeout injection for Browser Library wait_for_elements_state keyword."""
        arguments = ["#element", "visible"]
        result = executor._inject_timeout_into_arguments(
            "wait_for_elements_state", arguments, 5000, mock_session
        )
        # Wait keywords DO accept timeout parameter
        assert "timeout=5000ms" in result

    def test_selenium_library_click_element_timeout_injection(self, executor, mock_session):
        """Test timeout injection for SeleniumLibrary click_element keyword."""
        mock_session.browser_state.active_library = "selenium"
        arguments = ["id:button"]
        result = executor._inject_timeout_into_arguments(
            "click_element", arguments, 5000, mock_session
        )
        # SeleniumLibrary uses seconds, so 5000ms = 5.0s
        assert "timeout=5.0" in result

    def test_selenium_library_input_text_timeout_injection(self, executor, mock_session):
        """Test timeout injection for SeleniumLibrary input_text keyword."""
        mock_session.browser_state.active_library = "selenium"
        arguments = ["id:input", "some text"]
        result = executor._inject_timeout_into_arguments(
            "input_text", arguments, 3000, mock_session
        )
        # SeleniumLibrary uses seconds, so 3000ms = 3.0s
        assert "timeout=3.0" in result

    def test_no_injection_for_unsupported_keyword(self, executor, mock_session):
        """Test that no timeout is injected for keywords that don't support it."""
        arguments = ["some argument"]
        result = executor._inject_timeout_into_arguments(
            "some_unknown_keyword", arguments, 5000, mock_session
        )
        assert result == arguments
        assert "timeout=" not in str(result)

    def test_library_prefix_stripped(self, executor, mock_session):
        """Test that library prefix is stripped for keyword matching."""
        arguments = ["#element", "visible"]
        result = executor._inject_timeout_into_arguments(
            "Browser.wait_for_elements_state", arguments, 5000, mock_session
        )
        assert "timeout=5000ms" in result

    def test_case_insensitive_matching(self, executor, mock_session):
        """Test that keyword matching is case insensitive."""
        arguments = ["#element", "visible"]
        result = executor._inject_timeout_into_arguments(
            "Wait_For_Elements_State", arguments, 5000, mock_session
        )
        assert "timeout=5000ms" in result

    def test_space_to_underscore_normalization(self, executor, mock_session):
        """Test that spaces are converted to underscores for matching."""
        arguments = ["#element", "visible"]
        result = executor._inject_timeout_into_arguments(
            "wait for elements state", arguments, 5000, mock_session
        )
        assert "timeout=5000ms" in result


class TestTimeoutClassificationInExecutor:
    """Tests for timeout classification integration in execute_keyword."""

    @pytest.fixture
    def executor(self):
        """Create a KeywordExecutor instance for testing."""
        return KeywordExecutor(config=ExecutionConfig())

    def test_click_keyword_uses_action_timeout(self, executor):
        """Test that click keywords use action timeout (5s)."""
        from robotmcp.domains.timeout.keyword_classifier import classify_keyword
        action_type = classify_keyword("click")
        assert action_type == ActionType.CLICK

        policy = TimeoutPolicy.create_default("test")
        timeout = policy.get_timeout_for(action_type)
        assert timeout.value == 5000

    def test_navigation_keyword_uses_navigation_timeout(self, executor):
        """Test that navigation keywords use navigation timeout (60s)."""
        from robotmcp.domains.timeout.keyword_classifier import classify_keyword
        action_type = classify_keyword("go_to")
        assert action_type == ActionType.NAVIGATE

        policy = TimeoutPolicy.create_default("test")
        timeout = policy.get_timeout_for(action_type)
        assert timeout.value == 60000

    def test_read_keyword_uses_read_timeout(self, executor):
        """Test that read keywords use read timeout (2s)."""
        from robotmcp.domains.timeout.keyword_classifier import classify_keyword
        action_type = classify_keyword("get_text")
        assert action_type == ActionType.GET_TEXT

        policy = TimeoutPolicy.create_default("test")
        timeout = policy.get_timeout_for(action_type)
        assert timeout.value == 2000


class TestBrowserLibraryTimeoutKeywords:
    """Tests for Browser Library keyword timeout support.

    NOTE: Most Browser Library keywords do NOT accept a timeout parameter.
    They use the global browser timeout set via "Set Browser Timeout".
    Only explicit wait keywords accept timeout parameter.
    """

    @pytest.fixture
    def executor(self):
        return KeywordExecutor(config=ExecutionConfig())

    @pytest.fixture
    def mock_session(self):
        session = MagicMock(spec=ExecutionSession)
        session.session_id = "test_session"
        session.browser_state = MagicMock()
        session.browser_state.active_library = "browser"
        return session

    @pytest.mark.parametrize("keyword", [
        # Only wait keywords accept timeout parameter
        "wait_for_elements_state",
        "wait_for_condition",
        "wait_for_navigation",
        "wait_for_request",
        "wait_for_response",
        "wait_for_function",
        "wait_for_load_state",
        "wait_until_network_is_idle",
    ])
    def test_browser_library_wait_keywords_support_timeout(self, executor, mock_session, keyword):
        """Test that Browser Library wait keywords get timeout injected."""
        arguments = ["#element"]
        result = executor._inject_timeout_into_arguments(
            keyword, arguments, 5000, mock_session
        )
        assert "timeout=5000ms" in result

    @pytest.mark.parametrize("keyword", [
        # Action keywords do NOT accept timeout parameter
        "click", "fill_text", "fill_secret", "type_text", "press_keys",
        "check_checkbox", "uncheck_checkbox", "select_options", "hover",
        "focus", "scroll_to_element",
        # Read keywords also do NOT accept timeout parameter
        "get_text", "get_attribute", "get_property", "get_element_count",
    ])
    def test_browser_library_action_keywords_no_timeout_injection(self, executor, mock_session, keyword):
        """Test that Browser Library action keywords do NOT get timeout injected."""
        arguments = ["#element"]
        result = executor._inject_timeout_into_arguments(
            keyword, arguments, 5000, mock_session
        )
        # These keywords use global browser timeout, not per-keyword timeout
        assert result == arguments


class TestSeleniumLibraryTimeoutKeywords:
    """Tests for SeleniumLibrary keyword timeout support."""

    @pytest.fixture
    def executor(self):
        return KeywordExecutor(config=ExecutionConfig())

    @pytest.fixture
    def mock_session(self):
        session = MagicMock(spec=ExecutionSession)
        session.session_id = "test_session"
        session.browser_state = MagicMock()
        session.browser_state.active_library = "selenium"
        return session

    @pytest.mark.parametrize("keyword", [
        "click_element", "click_button", "click_link", "input_text",
        "input_password", "select_from_list_by_value", "select_from_list_by_label",
        "select_from_list_by_index", "select_checkbox", "unselect_checkbox",
        "mouse_over", "wait_until_element_is_visible",
        "wait_until_element_is_not_visible", "wait_until_element_is_enabled",
        "wait_until_element_contains", "wait_until_page_contains_element",
        "wait_until_page_does_not_contain_element",
    ])
    def test_selenium_library_keywords_support_timeout(self, executor, mock_session, keyword):
        """Test that SeleniumLibrary keywords that support timeout get it injected."""
        arguments = ["id:element"]
        result = executor._inject_timeout_into_arguments(
            keyword, arguments, 5000, mock_session
        )
        # SeleniumLibrary uses seconds
        assert "timeout=5.0" in result

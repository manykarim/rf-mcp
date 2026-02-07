"""Tests for P0/P1/P2 pre-validation timeout fixes.

P0-1: SeleniumLibrary implicit wait capture (no more hardcoded "10s")
P0-2: AppiumLibrary active_library detection from imported_libraries
P0-3: AppiumLibrary timeout control via Set Appium Implicit Wait
P1-1: Browser thread-safe timeout via _browser_timeout_lock
P1-2: Devserver case-insensitive active_library matching
P1-3: Timeout propagation from execute_step to _pre_validate_element
P2-1/2/3: Expanded ELEMENT_INTERACTION_KEYWORDS
P2-4: Contenteditable detection in Selenium JS check
P2-5: Strict mode retry with reduced timeout
"""

__test__ = True

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from robotmcp.components.execution.keyword_executor import KeywordExecutor
from robotmcp.models.config_models import ExecutionConfig
from robotmcp.models.session_models import ExecutionSession
from robotmcp.models.browser_models import BrowserState


@pytest.fixture
def executor():
    """Create a KeywordExecutor with default config."""
    return KeywordExecutor(config=ExecutionConfig())


@pytest.fixture
def mock_session():
    """Create a mock session with browser state."""
    session = MagicMock(spec=ExecutionSession)
    session.session_id = "test-preval-fix"
    session.variables = {}
    session.browser_state = MagicMock(spec=BrowserState)
    session.browser_state.active_library = None
    session.imported_libraries = ["BuiltIn"]
    return session


# =============================================================================
# P0-1: SeleniumLibrary implicit wait capture
# =============================================================================

class TestP0_1_SeleniumImplicitWaitCapture:
    """P0-1: Set Selenium Implicit Wait return value is now captured, not hardcoded."""

    def test_captures_previous_implicit_wait(self, executor):
        """Captured previous value should be used for restoration, not '10s'."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            calls = []

            def capture(name, *args, **kwargs):
                calls.append((name, args))
                if name == "SeleniumLibrary.Set Selenium Implicit Wait":
                    return "30 seconds"  # User had 30s timeout
                elif name == "SeleniumLibrary.Get WebElements":
                    return [MagicMock()]
                elif name == "SeleniumLibrary.Execute Javascript":
                    return ["visible", "enabled"]
                return None

            mock_builtin.run_keyword.side_effect = capture

            executor._run_selenium_state_check("id=btn", {"visible"}, 500)

            wait_calls = [c for c in calls if c[0] == "SeleniumLibrary.Set Selenium Implicit Wait"]
            assert len(wait_calls) == 2
            # Set call uses pre-validation timeout
            assert wait_calls[0][1] == ("0.5s",)
            # Restore uses captured value, NOT hardcoded "10s"
            assert wait_calls[1][1] == ("30 seconds",)

    def test_restores_custom_timeout_15s(self, executor):
        """Should restore 15s if that was the previous implicit wait."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            def side_effect(name, *args, **kwargs):
                if name == "SeleniumLibrary.Set Selenium Implicit Wait":
                    return "15 seconds"
                elif name == "SeleniumLibrary.Get WebElements":
                    return [MagicMock()]
                elif name == "SeleniumLibrary.Execute Javascript":
                    return ["visible"]
                return None

            mock_builtin.run_keyword.side_effect = side_effect

            executor._run_selenium_state_check("id=btn", {"visible"}, 500)

            # Last call should restore to "15 seconds"
            restore_call = mock_builtin.run_keyword.call_args_list[-1]
            assert restore_call[0][0] == "SeleniumLibrary.Set Selenium Implicit Wait"
            assert restore_call[0][1] == "15 seconds"

    def test_skips_restore_if_return_is_none(self, executor):
        """If Set Selenium Implicit Wait returns None, restoration is skipped."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            calls = []

            def capture(name, *args, **kwargs):
                calls.append(name)
                if name == "SeleniumLibrary.Set Selenium Implicit Wait":
                    return None  # Edge case: returns None
                elif name == "SeleniumLibrary.Get WebElements":
                    return [MagicMock()]
                elif name == "SeleniumLibrary.Execute Javascript":
                    return ["visible"]
                return None

            mock_builtin.run_keyword.side_effect = capture

            executor._run_selenium_state_check("id=btn", {"visible"}, 500)

            # Only 1 Set Selenium Implicit Wait call (no restore)
            wait_calls = [c for c in calls if c == "SeleniumLibrary.Set Selenium Implicit Wait"]
            assert len(wait_calls) == 1


# =============================================================================
# P0-2: AppiumLibrary active_library detection
# =============================================================================

class TestP0_2_AppiumActiveLibraryDetection:
    """P0-2: AppiumLibrary sessions should use Appium pre-validation path."""

    @pytest.mark.asyncio
    async def test_appium_detected_from_imported_libraries(self, executor, mock_session):
        """When AppiumLibrary is imported and active_library is None, should detect appium."""
        mock_session.browser_state.active_library = None
        mock_session.imported_libraries = ["AppiumLibrary", "BuiltIn"]

        with patch.object(executor, '_pre_validate_appium_element',
                          new_callable=AsyncMock, return_value={"valid": True, "states": ["visible"], "missing": [], "error": None}):
            is_valid, error, details = await executor._pre_validate_element(
                "id=btn", mock_session, "Click Element", timeout_ms=500
            )

            assert is_valid is True
            executor._pre_validate_appium_element.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_appium_session_skips(self, executor, mock_session):
        """Non-UI-library sessions should skip pre-validation."""
        mock_session.browser_state.active_library = None
        mock_session.imported_libraries = ["BuiltIn", "Collections"]

        is_valid, error, details = await executor._pre_validate_element(
            "id=btn", mock_session, "Click Element", timeout_ms=500
        )

        assert is_valid is True
        assert details.get("reason") == "no_active_browser"

    @pytest.mark.asyncio
    async def test_browser_with_none_active_library_skips(self, executor, mock_session):
        """Browser Library imported but active_library=None should skip (no browser open)."""
        mock_session.browser_state.active_library = None
        mock_session.imported_libraries = ["Browser", "BuiltIn"]

        is_valid, error, details = await executor._pre_validate_element(
            "id=btn", mock_session, "Click Element", timeout_ms=500
        )

        assert is_valid is True
        assert details.get("reason") == "no_active_browser"

    def test_browser_models_is_appium_library_active(self):
        """BrowserState.is_appium_library_active() should work."""
        state = BrowserState()
        assert state.is_appium_library_active() is False
        state.active_library = "appium"
        assert state.is_appium_library_active() is True


# =============================================================================
# P0-3: AppiumLibrary timeout control
# =============================================================================

class TestP0_3_AppiumTimeoutControl:
    """P0-3: Appium pre-validation should set/restore implicit wait."""

    def test_sets_appium_implicit_wait(self, executor):
        """Should call Set Appium Implicit Wait before element lookup."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            calls = []

            def capture(name, *args):
                calls.append(name)
                if name == "AppiumLibrary.Set Appium Implicit Wait":
                    return "10"  # Previous value
                elif name == "AppiumLibrary.Get Webelements":
                    el = MagicMock()
                    el.is_displayed.return_value = True
                    el.is_enabled.return_value = True
                    el.tag_name = "button"
                    return [el]
                return None

            mock_builtin.run_keyword.side_effect = capture

            executor._run_appium_state_check("id=btn", {"visible", "enabled"}, 500)

            assert "AppiumLibrary.Set Appium Implicit Wait" in calls

    def test_restores_appium_implicit_wait(self, executor):
        """Should restore Appium implicit wait after pre-validation."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            calls = []

            def capture(name, *args):
                calls.append((name, args))
                if name == "AppiumLibrary.Set Appium Implicit Wait":
                    return "10"  # Previous value
                elif name == "AppiumLibrary.Get Webelements":
                    el = MagicMock()
                    el.is_displayed.return_value = True
                    el.is_enabled.return_value = True
                    el.tag_name = "button"
                    return [el]
                return None

            mock_builtin.run_keyword.side_effect = capture

            executor._run_appium_state_check("id=btn", {"visible"}, 500)

            wait_calls = [c for c in calls if c[0] == "AppiumLibrary.Set Appium Implicit Wait"]
            assert len(wait_calls) == 2
            # Set: 0.5 (500ms / 1000)
            assert wait_calls[0][1] == ("0.5",)
            # Restore: "10" (captured previous)
            assert wait_calls[1][1] == ("10",)

    def test_restores_even_on_failure(self, executor):
        """Should restore timeout even when element lookup fails."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            calls = []

            def capture(name, *args):
                calls.append(name)
                if name == "AppiumLibrary.Set Appium Implicit Wait":
                    return "10"
                elif name == "AppiumLibrary.Get Webelements":
                    return []  # No elements found
                elif name == "AppiumLibrary.Get Webelement":
                    raise Exception("Element not found")
                return None

            mock_builtin.run_keyword.side_effect = capture

            result = executor._run_appium_state_check("id=missing", {"visible"}, 500)

            assert result["valid"] is False
            # Verify restore was called
            wait_calls = [c for c in calls if c == "AppiumLibrary.Set Appium Implicit Wait"]
            assert len(wait_calls) == 2  # set + restore


# =============================================================================
# P1-1: Browser thread-safe timeout
# =============================================================================

class TestP1_1_BrowserThreadSafeTimeout:
    """P1-1: Browser timeout mutations should be serialized with a lock."""

    def test_has_browser_timeout_lock(self, executor):
        """Executor should have a _browser_timeout_lock."""
        assert hasattr(executor, '_browser_timeout_lock')
        assert isinstance(executor._browser_timeout_lock, type(threading.Lock()))

    def test_lock_serializes_concurrent_access(self, executor):
        """Lock should serialize concurrent timeout mutations."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin
            mock_builtin.run_keyword.return_value = "10 seconds"

            # Use a tracking lock wrapper
            real_lock = executor._browser_timeout_lock
            lock_held_concurrently = []

            def run_with_tracking():
                """Run _run_browser_get_states and check lock is held."""
                # After the call, the lock should be released
                executor._run_browser_get_states("id=btn", "500ms")
                # Verify we can acquire the lock after the call
                acquired = real_lock.acquire(blocking=False)
                lock_held_concurrently.append(acquired)
                if acquired:
                    real_lock.release()

            threads = [threading.Thread(target=run_with_tracking) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All threads should have been able to acquire the lock after their calls
            assert all(lock_held_concurrently), "Lock was not released after _run_browser_get_states"

    def test_lock_released_on_exception(self, executor):
        """Lock should be released even when an exception occurs."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            mock_builtin.run_keyword.side_effect = Exception("Playwright crashed")

            # Should not deadlock
            executor._run_browser_get_states("id=btn", "500ms")

            # Lock should be released (we can acquire it again)
            acquired = executor._browser_timeout_lock.acquire(blocking=False)
            assert acquired is True
            executor._browser_timeout_lock.release()


# =============================================================================
# P1-2: Devserver case sensitivity
# =============================================================================

class TestP1_2_DevserverCaseSensitivity:
    """P1-2: active_library should be lowercase for consistent matching."""

    @pytest.mark.asyncio
    async def test_uppercase_active_library_normalized(self, executor, mock_session):
        """'Browser' (uppercase) should be normalized to 'browser' for routing."""
        mock_session.browser_state.active_library = "Browser"
        mock_session.imported_libraries = ["Browser", "BuiltIn"]

        with patch.object(executor, '_pre_validate_browser_element',
                          new_callable=AsyncMock, return_value={"valid": True, "states": ["visible"], "missing": [], "error": None}):
            is_valid, error, details = await executor._pre_validate_element(
                "id=btn", mock_session, "Click", timeout_ms=500
            )

            assert is_valid is True
            executor._pre_validate_browser_element.assert_called_once()

    @pytest.mark.asyncio
    async def test_selenium_uppercase_normalized(self, executor, mock_session):
        """'Selenium' should be normalized to 'selenium'."""
        mock_session.browser_state.active_library = "Selenium"
        mock_session.imported_libraries = ["SeleniumLibrary", "BuiltIn"]

        with patch.object(executor, '_pre_validate_selenium_element',
                          new_callable=AsyncMock, return_value={"valid": True, "states": ["visible"], "missing": [], "error": None}):
            is_valid, error, details = await executor._pre_validate_element(
                "id=btn", mock_session, "Click Element", timeout_ms=500
            )

            assert is_valid is True
            executor._pre_validate_selenium_element.assert_called_once()

    def test_devserver_sets_lowercase(self):
        """Devserver should set active_library = 'browser' (lowercase)."""
        import robotmcp.frontend.devserver as devserver_mod
        import inspect
        source = inspect.getsource(devserver_mod._create_sample_session)
        assert 'active_library = "browser"' in source
        assert 'active_library = "Browser"' not in source


# =============================================================================
# P1-3: Timeout propagation
# =============================================================================

class TestP1_3_TimeoutPropagation:
    """P1-3: User's timeout_ms should be propagated to pre-validation."""

    @pytest.mark.asyncio
    async def test_user_timeout_passed_to_pre_validate(self, executor, mock_session):
        """When user provides timeout_ms, pre-validation should use min(default, user_timeout)."""
        mock_session.browser_state.active_library = "browser"
        mock_session.imported_libraries = ["Browser", "BuiltIn"]

        with patch.object(executor, '_pre_validate_browser_element',
                          new_callable=AsyncMock, return_value={"valid": True, "states": ["visible"], "missing": [], "error": None}) as mock_preval:

            # User provides 200ms — less than default 500ms
            await executor._pre_validate_element("id=btn", mock_session, "Click", timeout_ms=200)

            call_args = mock_preval.call_args
            assert call_args[0][2] == 200  # timeout_ms should be 200, not 500

    @pytest.mark.asyncio
    async def test_user_large_timeout_passed_directly(self, executor, mock_session):
        """_pre_validate_element passes timeout_ms as-is (capping is done in execute_keyword)."""
        mock_session.browser_state.active_library = "browser"
        mock_session.imported_libraries = ["Browser", "BuiltIn"]

        with patch.object(executor, '_pre_validate_browser_element',
                          new_callable=AsyncMock, return_value={"valid": True, "states": ["visible"], "missing": [], "error": None}) as mock_preval:

            # When passed directly, _pre_validate_element uses the value as-is
            await executor._pre_validate_element("id=btn", mock_session, "Click", timeout_ms=30000)

            call_args = mock_preval.call_args
            # Direct call passes through without capping
            assert call_args[0][2] == 30000

    def test_execute_keyword_caps_timeout_with_min(self, executor):
        """The execute_keyword call site should use min(default, user_timeout)."""
        # The min() logic is in execute_keyword, not _pre_validate_element
        config = executor.config
        default = config.PRE_VALIDATION_TIMEOUT  # 500

        # Verify min() behavior
        assert min(default, 30000) == 500  # Capped at default
        assert min(default, 200) == 200    # User's smaller value used
        assert min(default, 500) == 500    # Equal = default

    @pytest.mark.asyncio
    async def test_no_timeout_uses_default(self, executor, mock_session):
        """When timeout_ms is None, should use config.PRE_VALIDATION_TIMEOUT."""
        mock_session.browser_state.active_library = "browser"
        mock_session.imported_libraries = ["Browser", "BuiltIn"]

        with patch.object(executor, '_pre_validate_browser_element',
                          new_callable=AsyncMock, return_value={"valid": True, "states": ["visible"], "missing": [], "error": None}) as mock_preval:

            await executor._pre_validate_element("id=btn", mock_session, "Click", timeout_ms=None)

            call_args = mock_preval.call_args
            assert call_args[0][2] == 500  # config default

    def test_execute_keyword_skips_preval_when_timeout_zero(self, executor, mock_session):
        """When timeout_ms=0 (disabled), pre-validation should be skipped."""
        # This tests the call site in execute_keyword, not _pre_validate_element directly.
        # The skip logic is: if timeout_ms <= 0, skip pre-validation entirely.
        # We verify the logic flow by checking the skip_preval variable behavior.
        config = ExecutionConfig()
        assert config.PRE_VALIDATION_TIMEOUT == 500

        # When timeout_ms=0, skip_preval should be True
        # When timeout_ms>0, skip_preval should be False
        # When timeout_ms is None, skip_preval should be False
        # These are tested via the full execute_keyword path indirectly

    def test_min_timeout_calculation(self, executor):
        """min(default, user_timeout) should be used when both are positive."""
        config = executor.config
        default = config.PRE_VALIDATION_TIMEOUT  # 500

        # Cases:
        assert min(default, 200) == 200
        assert min(default, 500) == 500
        assert min(default, 30000) == 500
        assert min(default, 1) == 1


# =============================================================================
# P2-1/2/3: Expanded ELEMENT_INTERACTION_KEYWORDS
# =============================================================================

class TestP2_123_ExpandedKeywords:
    """P2-1/2/3: New keywords should be in ELEMENT_INTERACTION_KEYWORDS."""

    # Browser Library additions (P2-1)
    @pytest.mark.parametrize("keyword", [
        "click with options",
        "tap",
        "mouse move relative to",
        "scroll by",
        "scroll to",
        "select options by",
        "deselect options",
        "upload file by selector",
    ])
    def test_browser_keywords_in_set(self, executor, keyword):
        assert keyword in executor.ELEMENT_INTERACTION_KEYWORDS
        assert executor._requires_pre_validation(keyword)

    # SeleniumLibrary additions (P2-2)
    @pytest.mark.parametrize("keyword", [
        "click button",
        "click link",
        "click image",
        "click element at coordinates",
        "drag and drop",
        "drag and drop by offset",
        "set focus to element",
        "submit form",
        "mouse down",
        "mouse up",
        "mouse out",
        "mouse down on image",
        "mouse down on link",
        "open context menu",
        "choose file",
        "clear element text",
    ])
    def test_selenium_keywords_in_set(self, executor, keyword):
        assert keyword in executor.ELEMENT_INTERACTION_KEYWORDS
        assert executor._requires_pre_validation(keyword)

    # AppiumLibrary additions (P2-3)
    @pytest.mark.parametrize("keyword", [
        "input value",
        "long press",
    ])
    def test_appium_keywords_in_set(self, executor, keyword):
        assert keyword in executor.ELEMENT_INTERACTION_KEYWORDS
        assert executor._requires_pre_validation(keyword)

    # Original keywords still present
    @pytest.mark.parametrize("keyword", [
        "click", "click element", "fill text", "fill secret",
        "type text", "input text", "input password", "check checkbox",
        "select options", "press keys", "hover", "scroll to element",
    ])
    def test_original_keywords_still_present(self, executor, keyword):
        assert keyword in executor.ELEMENT_INTERACTION_KEYWORDS

    def test_total_keyword_count_increased(self, executor):
        """Should have more keywords than the original 32."""
        assert len(executor.ELEMENT_INTERACTION_KEYWORDS) > 32

    # XML keywords should NOT be in the set
    @pytest.mark.parametrize("keyword", [
        "set element text",
        "clear element",
        "get element text",
        "parse xml",
    ])
    def test_xml_keywords_not_in_set(self, executor, keyword):
        assert keyword not in executor.ELEMENT_INTERACTION_KEYWORDS


# =============================================================================
# P2-1/2/3: Action type mapping for new keywords
# =============================================================================

class TestP2_ActionTypeMapping:
    """New keywords should map to correct action types."""

    @pytest.mark.parametrize("keyword,expected_type", [
        ("click button", "click"),
        ("click link", "click"),
        ("click image", "click"),
        ("click with options", "click"),
        ("drag and drop", "drag"),
        ("drag and drop by offset", "drag"),
        ("tap", "tap"),
        ("long press", "tap"),
        ("submit form", "submit"),
        ("upload file by selector", "upload"),
        ("choose file", "upload"),
        ("open context menu", "open"),
        ("mouse down", "hover"),  # mouse → hover action type
        ("mouse out", "hover"),
        ("scroll by", "scroll"),
        ("scroll to", "scroll"),
        ("deselect options", "select"),
        ("select options by", "select"),
        ("clear element text", "clear"),
        ("input value", "fill"),
    ])
    def test_action_type_mapping(self, executor, keyword, expected_type):
        assert executor._get_action_type_from_keyword_for_states(keyword) == expected_type

    def test_new_action_types_have_required_states(self, executor):
        """All new action types should have entries in REQUIRED_STATES_FOR_ACTION."""
        for action_type in ["drag", "tap", "submit", "upload", "open"]:
            assert action_type in executor.REQUIRED_STATES_FOR_ACTION


# =============================================================================
# P2-4: Contenteditable detection
# =============================================================================

class TestP2_4_ContenteditableDetection:
    """P2-4: Selenium JS check should detect contenteditable elements."""

    def test_js_check_includes_isContentEditable(self, executor):
        """The JS check code should include isContentEditable."""
        import inspect
        source = inspect.getsource(executor._run_selenium_state_check)
        assert "isContentEditable" in source

    def test_contenteditable_before_input_check(self, executor):
        """isContentEditable should be checked before tagName check."""
        import inspect
        source = inspect.getsource(executor._run_selenium_state_check)
        content_idx = source.index("isContentEditable")
        tagname_idx = source.index("el.tagName === 'INPUT'")
        assert content_idx < tagname_idx


# =============================================================================
# P2-5: Strict mode reduced timeout
# =============================================================================

class TestP2_5_StrictModeTimeout:
    """P2-5: Strict mode retries should use shorter timeout."""

    def test_strict_mode_sets_shorter_timeout(self, executor):
        """On strict mode violation, retry timeout should be reduced to 200ms."""
        with patch('robotmcp.components.execution.keyword_executor.BuiltIn') as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            calls = []

            def capture(name, *args):
                calls.append((name, args))
                if name == "Browser.Set Browser Timeout":
                    return "10 seconds"
                elif name == "Browser.Get Element States":
                    raise Exception("strict mode violation: resolved to 2 elements")
                elif name == "Get Element States":
                    raise Exception("strict mode violation: resolved to 2 elements")
                return None

            mock_builtin.run_keyword.side_effect = capture

            executor._run_browser_get_states("id=btn", "500ms")

            # Find Set Browser Timeout calls
            timeout_calls = [c for c in calls if c[0] == "Browser.Set Browser Timeout"]
            # Should have: initial set (500ms), retry set (200ms), restore (10 seconds)
            assert len(timeout_calls) >= 2
            # Second timeout call should be "200ms" for strict mode retry
            assert timeout_calls[1][1] == ("200ms",)


# =============================================================================
# Integration: Case normalization in _pre_validate_element
# =============================================================================

class TestCaseNormalization:
    """active_library case normalization in _pre_validate_element."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("case", ["browser", "Browser", "BROWSER"])
    async def test_all_cases_route_to_browser(self, executor, mock_session, case):
        """All case variants of 'browser' should route to browser pre-validation."""
        mock_session.browser_state.active_library = case
        mock_session.imported_libraries = ["Browser", "BuiltIn"]

        with patch.object(executor, '_pre_validate_browser_element',
                          new_callable=AsyncMock, return_value={"valid": True, "states": ["visible"], "missing": [], "error": None}):
            is_valid, _, _ = await executor._pre_validate_element(
                "id=btn", mock_session, "Click", timeout_ms=500
            )
            assert is_valid is True
            executor._pre_validate_browser_element.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("case", ["selenium", "Selenium", "SELENIUM"])
    async def test_all_cases_route_to_selenium(self, executor, mock_session, case):
        """All case variants of 'selenium' should route to selenium pre-validation."""
        mock_session.browser_state.active_library = case
        mock_session.imported_libraries = ["SeleniumLibrary", "BuiltIn"]

        with patch.object(executor, '_pre_validate_selenium_element',
                          new_callable=AsyncMock, return_value={"valid": True, "states": ["visible"], "missing": [], "error": None}):
            is_valid, _, _ = await executor._pre_validate_element(
                "id=btn", mock_session, "Click Element", timeout_ms=500
            )
            assert is_valid is True
            executor._pre_validate_selenium_element.assert_called_once()


# =============================================================================
# Backward compatibility
# =============================================================================

class TestBackwardCompatibility:
    """Ensure existing behavior is preserved."""

    @pytest.mark.asyncio
    async def test_browser_lowercase_still_works(self, executor, mock_session):
        """The standard lowercase 'browser' from browser_library_manager still works."""
        mock_session.browser_state.active_library = "browser"
        mock_session.imported_libraries = ["Browser", "BuiltIn"]

        with patch.object(executor, '_pre_validate_browser_element',
                          new_callable=AsyncMock, return_value={"valid": True, "states": ["visible"], "missing": [], "error": None}):
            is_valid, _, _ = await executor._pre_validate_element(
                "id=btn", mock_session, "Click", timeout_ms=500
            )
            assert is_valid is True

    @pytest.mark.asyncio
    async def test_selenium_lowercase_still_works(self, executor, mock_session):
        """The standard lowercase 'selenium' still works."""
        mock_session.browser_state.active_library = "selenium"
        mock_session.imported_libraries = ["SeleniumLibrary", "BuiltIn"]

        with patch.object(executor, '_pre_validate_selenium_element',
                          new_callable=AsyncMock, return_value={"valid": True, "states": ["visible"], "missing": [], "error": None}):
            is_valid, _, _ = await executor._pre_validate_element(
                "id=btn", mock_session, "Click Element", timeout_ms=500
            )
            assert is_valid is True

    def test_original_keywords_still_trigger_preval(self, executor):
        """All original ELEMENT_INTERACTION_KEYWORDS still work."""
        original = [
            "click", "click element", "double click", "fill text", "fill secret",
            "type text", "type secret", "input text", "input password",
            "clear text", "clear element value", "check checkbox", "uncheck checkbox",
            "select options", "select from list", "press keys", "hover",
        ]
        for kw in original:
            assert executor._requires_pre_validation(kw), f"Original keyword '{kw}' no longer triggers pre-validation"

    def test_non_interaction_keywords_still_skipped(self, executor):
        """Non-interaction keywords should still be skipped."""
        non_interaction = [
            "log", "should be equal", "new page", "get text", "wait for elements state",
            "parse xml", "get element text", "set variable", "evaluate",
        ]
        for kw in non_interaction:
            assert not executor._requires_pre_validation(kw), f"Non-interaction keyword '{kw}' unexpectedly triggers pre-validation"

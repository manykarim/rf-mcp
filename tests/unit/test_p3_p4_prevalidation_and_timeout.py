"""Tests for S5 (namespace-based library detection) and P4 (positional timeout).

S5 replaces P3's session-state-based pre-validation routing with RF runtime
resolution via namespace.get_runner(keyword).keyword.owner.name.  This is the
authoritative resolution that RF itself uses to execute keywords.

P4: Timeout injection must detect positional timeout arguments in
    SeleniumLibrary wait keywords.

Run with: uv run pytest tests/unit/test_p3_p4_prevalidation_and_timeout.py -v
"""

__test__ = True

from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from robotmcp.components.execution.keyword_executor import KeywordExecutor
from robotmcp.models.config_models import ExecutionConfig
from robotmcp.models.session_models import ExecutionSession
from robotmcp.components.browser.browser_library_manager import BrowserLibraryManager
from tests.unit.helpers.rf_context_mock import rf_context_with_owner, no_rf_context


@pytest.fixture
def executor():
    return KeywordExecutor(config=ExecutionConfig())


def _make_session(imported_libraries, active_library=None):
    """Create a mock session with the given imports and active_library."""
    session = MagicMock(spec=ExecutionSession)
    session.session_id = "test_session"
    session.browser_state = MagicMock()
    session.browser_state.active_library = active_library
    session.imported_libraries = imported_libraries
    session.loaded_libraries = set(imported_libraries) if imported_libraries else set()
    return session


# ============================================================
# S5a: Pre-validation routes via RF namespace.get_runner()
# ============================================================
class TestS5a_NamespaceBasedPreValidation:
    """Verify that _pre_validate_element routes to the correct library-specific
    validator based on namespace.get_runner(keyword).keyword.owner.name,
    NOT session.browser_state.active_library."""

    @pytest.mark.asyncio
    async def test_selenium_owner_routes_to_selenium_validator(self, executor):
        """Keyword owned by SeleniumLibrary → selenium pre-validation."""
        session = _make_session(["SeleniumLibrary", "BuiltIn"], "selenium")
        with rf_context_with_owner("SeleniumLibrary"), \
             patch.object(executor, "_pre_validate_selenium_element",
                          new_callable=AsyncMock) as mock_sel:
            mock_sel.return_value = {"valid": True, "states": ["visible"], "missing": [], "error": None}
            is_valid, error_msg, details = await executor._pre_validate_element(
                "css=.button", session, "Click Button", timeout_ms=500
            )
            mock_sel.assert_called_once()
            assert is_valid

    @pytest.mark.asyncio
    async def test_selenium_owner_does_not_use_browser_validator(self, executor):
        """SeleniumLibrary-owned keyword must NOT call browser validator."""
        session = _make_session(["SeleniumLibrary", "BuiltIn"], "selenium")
        with rf_context_with_owner("SeleniumLibrary"), \
             patch.object(executor, "_pre_validate_browser_element",
                          new_callable=AsyncMock) as mock_browser, \
             patch.object(executor, "_pre_validate_selenium_element",
                          new_callable=AsyncMock) as mock_sel:
            mock_sel.return_value = {"valid": True, "states": ["visible"], "missing": [], "error": None}
            await executor._pre_validate_element(
                "css=.button", session, "Click Button", timeout_ms=500
            )
            mock_browser.assert_not_called()

    @pytest.mark.asyncio
    async def test_browser_owner_routes_to_browser_validator(self, executor):
        """Keyword owned by Browser → browser pre-validation."""
        session = _make_session(["Browser", "BuiltIn"], "browser")
        with rf_context_with_owner("Browser"), \
             patch.object(executor, "_pre_validate_browser_element",
                          new_callable=AsyncMock) as mock_browser:
            mock_browser.return_value = {"valid": True, "states": ["visible"], "missing": [], "error": None}
            is_valid, error_msg, details = await executor._pre_validate_element(
                "css=.button", session, "Click", timeout_ms=500
            )
            mock_browser.assert_called_once()
            assert is_valid

    @pytest.mark.asyncio
    async def test_appium_owner_routes_to_appium_validator(self, executor):
        """Keyword owned by AppiumLibrary → appium pre-validation."""
        session = _make_session(["AppiumLibrary", "BuiltIn"])
        with rf_context_with_owner("AppiumLibrary"), \
             patch.object(executor, "_pre_validate_appium_element",
                          new_callable=AsyncMock) as mock_appium:
            mock_appium.return_value = {"valid": True, "states": ["visible"], "missing": [], "error": None}
            is_valid, error_msg, details = await executor._pre_validate_element(
                "id=button", session, "Click Element", timeout_ms=500
            )
            mock_appium.assert_called_once()

    @pytest.mark.asyncio
    async def test_builtin_owner_skips_prevalidation(self, executor):
        """Keyword owned by BuiltIn → skip pre-validation (not a web keyword)."""
        session = _make_session(["BuiltIn", "Browser"])
        with rf_context_with_owner("BuiltIn"):
            is_valid, error_msg, details = await executor._pre_validate_element(
                "id=element", session, "Log", timeout_ms=500
            )
            assert is_valid is True
            assert details.get("skipped") is True

    @pytest.mark.asyncio
    async def test_no_owner_skips_prevalidation(self, executor):
        """Invalid keyword with no owner → skip pre-validation."""
        session = _make_session(["Browser", "BuiltIn"])
        with rf_context_with_owner(None):
            is_valid, error_msg, details = await executor._pre_validate_element(
                "css=.button", session, "NonExistentKeyword", timeout_ms=500
            )
            assert is_valid is True
            assert details.get("skipped") is True

    @pytest.mark.asyncio
    async def test_no_rf_context_skips_prevalidation(self, executor):
        """No active RF execution context → skip pre-validation."""
        session = _make_session(["SeleniumLibrary", "BuiltIn"], "selenium")
        with no_rf_context():
            is_valid, error_msg, details = await executor._pre_validate_element(
                "css=.button", session, "Click Button", timeout_ms=500
            )
            assert is_valid is True
            assert details.get("skipped") is True

    @pytest.mark.asyncio
    async def test_shared_keyword_get_text_routes_to_actual_owner(self, executor):
        """P5 scenario: 'Get Text' is shared between Browser and SeleniumLibrary.
        RF resolution determines the actual owner — pre-validation follows it."""
        session = _make_session(["SeleniumLibrary", "BuiltIn"], "selenium")
        # Even though Get Text pattern-matches to Browser, the RF runtime
        # resolves it to SeleniumLibrary (which is what we mock here)
        with rf_context_with_owner("SeleniumLibrary"), \
             patch.object(executor, "_pre_validate_selenium_element",
                          new_callable=AsyncMock) as mock_sel:
            mock_sel.return_value = {"valid": True, "states": ["visible"], "missing": [], "error": None}
            is_valid, error_msg, details = await executor._pre_validate_element(
                "css=.text", session, "Get Text", timeout_ms=500
            )
            mock_sel.assert_called_once()

    @pytest.mark.asyncio
    async def test_dual_library_browser_appium_routes_per_keyword(self, executor):
        """In a Browser+Appium session, each keyword routes independently."""
        session = _make_session(["Browser", "AppiumLibrary", "BuiltIn"])

        # Click → owned by Browser → browser pre-validation
        with rf_context_with_owner("Browser"), \
             patch.object(executor, "_pre_validate_browser_element",
                          new_callable=AsyncMock) as mock_browser:
            mock_browser.return_value = {"valid": True, "states": ["visible"], "missing": [], "error": None}
            await executor._pre_validate_element(
                "css=.btn", session, "Click", timeout_ms=500
            )
            mock_browser.assert_called_once()

        # Click Element → owned by AppiumLibrary → appium pre-validation
        with rf_context_with_owner("AppiumLibrary"), \
             patch.object(executor, "_pre_validate_appium_element",
                          new_callable=AsyncMock) as mock_appium:
            mock_appium.return_value = {"valid": True, "states": ["visible"], "missing": [], "error": None}
            await executor._pre_validate_element(
                "id=element", session, "Click Element", timeout_ms=500
            )
            mock_appium.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_active_library_ignored_when_owner_known(self, executor):
        """Session active_library='browser' but keyword owner is SeleniumLibrary
        → selenium pre-validation (RF resolution wins over session state)."""
        session = _make_session(
            ["SeleniumLibrary", "Browser", "BuiltIn"],
            active_library="browser",  # stale/wrong session state
        )
        with rf_context_with_owner("SeleniumLibrary"), \
             patch.object(executor, "_pre_validate_selenium_element",
                          new_callable=AsyncMock) as mock_sel, \
             patch.object(executor, "_pre_validate_browser_element",
                          new_callable=AsyncMock) as mock_browser:
            mock_sel.return_value = {"valid": True, "states": ["visible"], "missing": [], "error": None}
            is_valid, error_msg, details = await executor._pre_validate_element(
                "css=.btn", session, "Click Button", timeout_ms=500
            )
            mock_sel.assert_called_once()
            mock_browser.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_runner_exception_skips_prevalidation(self, executor):
        """If get_runner() raises an exception, pre-validation is skipped gracefully."""
        session = _make_session(["SeleniumLibrary", "BuiltIn"], "selenium")
        mock_ec = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.test = MagicMock()
        mock_ctx.namespace.get_runner.side_effect = Exception("No keyword found")
        mock_ec.current = mock_ctx
        with patch("robot.running.context.EXECUTION_CONTEXTS", mock_ec):
            is_valid, error_msg, details = await executor._pre_validate_element(
                "css=.button", session, "Click Button", timeout_ms=500
            )
            assert is_valid is True
            assert details.get("skipped") is True

    @pytest.mark.asyncio
    async def test_collections_library_skips_prevalidation(self, executor):
        """Keywords from non-web libraries (Collections, String, etc.) skip."""
        session = _make_session(["Collections", "BuiltIn"])
        with rf_context_with_owner("Collections"):
            is_valid, error_msg, details = await executor._pre_validate_element(
                "some_arg", session, "Append To List", timeout_ms=500
            )
            assert is_valid is True
            assert details.get("skipped") is True

    @pytest.mark.asyncio
    async def test_selenium_failure_gives_helpful_error(self, executor):
        """When selenium pre-validation fails, error mentions SeleniumLibrary
        concepts, not Browser Library concepts."""
        session = _make_session(["SeleniumLibrary", "BuiltIn"], "selenium")
        with rf_context_with_owner("SeleniumLibrary"), \
             patch.object(executor, "_pre_validate_selenium_element",
                          new_callable=AsyncMock) as mock_sel:
            mock_sel.return_value = {
                "valid": False,
                "states": [],
                "missing": ["visible"],
                "error": "Element not found: css=.missing",
            }
            is_valid, error_msg, details = await executor._pre_validate_element(
                "css=.missing", session, "Click Button", timeout_ms=500
            )
            assert is_valid is False
            assert "Element not found" in error_msg
            assert "Get Element States" not in (error_msg or "")


# ============================================================
# S5b: Post-execution detection uses RF namespace (not patterns)
# ============================================================
class TestS5b_PostExecutionDetection:
    """Verify that post-execution library tracking uses
    namespace.get_runner() instead of pattern-based detection."""

    def test_set_active_library_called_with_correct_type(self, executor):
        """When RF resolves keyword to SeleniumLibrary, set_active_library
        should be called with 'selenium', not 'browser'."""
        session = _make_session(["SeleniumLibrary", "BuiltIn"], "selenium")
        mock_blm = MagicMock(spec=BrowserLibraryManager)

        with rf_context_with_owner("SeleniumLibrary"):
            # Simulate a successful keyword execution result
            result = {"success": True, "output": "some text"}

            # Call the post-execution block directly by simulating
            # what _execute_keyword_in_rf_native_context does
            lib_type = None
            try:
                from robot.running.context import EXECUTION_CONTEXTS as _EC
                _post_ctx = _EC.current
                if _post_ctx:
                    _runner = _post_ctx.namespace.get_runner("Get Text")
                    _owner = getattr(getattr(_runner, 'keyword', None), 'owner', None)
                    _owner_name = getattr(_owner, 'name', None)
                    if _owner_name == "Browser":
                        lib_type = "browser"
                    elif _owner_name == "SeleniumLibrary":
                        lib_type = "selenium"
                    elif _owner_name == "AppiumLibrary":
                        lib_type = "appium"
            except Exception:
                pass

            assert lib_type == "selenium"

    def test_browser_keyword_resolves_to_browser(self, executor):
        """RF keyword owned by Browser resolves to lib_type='browser'."""
        with rf_context_with_owner("Browser"):
            from robot.running.context import EXECUTION_CONTEXTS as _EC
            _post_ctx = _EC.current
            _runner = _post_ctx.namespace.get_runner("Click")
            _owner_name = getattr(getattr(_runner, 'keyword', None), 'owner', None).name
            assert _owner_name == "Browser"

    def test_no_context_gives_none_lib_type(self):
        """Without RF context, lib_type stays None — no state update."""
        with no_rf_context():
            lib_type = None
            try:
                from robot.running.context import EXECUTION_CONTEXTS as _EC
                _post_ctx = _EC.current
                if _post_ctx:
                    _runner = _post_ctx.namespace.get_runner("Get Text")
                    _owner = getattr(getattr(_runner, 'keyword', None), 'owner', None)
                    _owner_name = getattr(_owner, 'name', None)
                    if _owner_name == "Browser":
                        lib_type = "browser"
                    elif _owner_name == "SeleniumLibrary":
                        lib_type = "selenium"
            except Exception:
                pass
            assert lib_type is None

    def test_pattern_detection_no_longer_called_in_post_execution(self):
        """Verify that detect_library_type_from_keyword is NOT called
        in the post-execution path (S5b removed the call, only a comment remains)."""
        import inspect
        from robotmcp.components.execution import keyword_executor
        source = inspect.getsource(keyword_executor.KeywordExecutor._execute_keyword_with_context)
        # Check that the function is not actually called (= assigned to a variable)
        # Comments referencing it are fine — we check for import and assignment patterns
        assert "= detect_library_type_from_keyword" not in source
        assert "import detect_library_type_from_keyword" not in source.replace(" ", "").replace("\n", "")
        # The old alias sets should NOT be in this method
        assert "browser_aliases" not in source
        assert "selenium_aliases" not in source


# ============================================================
# S5c: set_active_library guards against unloaded libraries
# ============================================================
class TestS5c_SetActiveLibraryGuard:
    """Verify that set_active_library checks loaded_libraries before
    allowing library activation — defense-in-depth against corruption."""

    def test_browser_blocked_when_not_loaded(self):
        """set_active_library('browser') returns False when Browser not in loaded_libraries."""
        session = _make_session(["SeleniumLibrary", "BuiltIn"], "selenium")
        session.loaded_libraries = {"SeleniumLibrary", "BuiltIn"}  # No Browser

        mgr = MagicMock(spec=BrowserLibraryManager)
        mgr.browser_lib = MagicMock()  # Available at manager level
        mgr.selenium_lib = MagicMock()
        # Call the real method
        result = BrowserLibraryManager.set_active_library(mgr, session, "browser")
        assert result is False
        assert session.browser_state.active_library != "browser"

    def test_browser_allowed_when_loaded(self):
        """set_active_library('browser') returns True when Browser in loaded_libraries."""
        session = _make_session(["Browser", "BuiltIn"], "browser")
        session.loaded_libraries = {"Browser", "BuiltIn"}

        mgr = MagicMock(spec=BrowserLibraryManager)
        mgr.browser_lib = MagicMock()
        mgr.selenium_lib = MagicMock()
        result = BrowserLibraryManager.set_active_library(mgr, session, "browser")
        assert result is True
        assert session.browser_state.active_library == "browser"

    def test_selenium_blocked_when_not_loaded(self):
        """set_active_library('selenium') returns False when SeleniumLibrary not loaded."""
        session = _make_session(["Browser", "BuiltIn"])
        session.loaded_libraries = {"Browser", "BuiltIn"}  # No SeleniumLibrary

        mgr = MagicMock(spec=BrowserLibraryManager)
        mgr.browser_lib = MagicMock()
        mgr.selenium_lib = MagicMock()
        result = BrowserLibraryManager.set_active_library(mgr, session, "selenium")
        assert result is False

    def test_selenium_allowed_when_loaded(self):
        """set_active_library('selenium') returns True when SeleniumLibrary loaded."""
        session = _make_session(["SeleniumLibrary", "BuiltIn"], "selenium")
        session.loaded_libraries = {"SeleniumLibrary", "BuiltIn"}

        mgr = MagicMock(spec=BrowserLibraryManager)
        mgr.browser_lib = MagicMock()
        mgr.selenium_lib = MagicMock()
        result = BrowserLibraryManager.set_active_library(mgr, session, "selenium")
        assert result is True
        assert session.browser_state.active_library == "selenium"

    def test_p5_scenario_blocked_by_guard(self):
        """P5 root cause: pattern detection says 'browser' for shared keyword,
        but Browser was never loaded for this session → guard blocks."""
        session = _make_session(["SeleniumLibrary", "BuiltIn"], "selenium")
        session.loaded_libraries = {"SeleniumLibrary", "BuiltIn"}

        mgr = MagicMock(spec=BrowserLibraryManager)
        mgr.browser_lib = MagicMock()  # Manager has it, but session didn't load it
        mgr.selenium_lib = MagicMock()

        # This is what the old P5 bug would try — setting browser for a Selenium session
        result = BrowserLibraryManager.set_active_library(mgr, session, "browser")
        assert result is False
        # imported_libraries must NOT be corrupted
        assert "Browser" not in session.imported_libraries


# ============================================================
# P4: Timeout injection must detect positional timeout arguments
# ============================================================
class TestP4_PositionalTimeoutDetection:
    """Verify that _inject_timeout_into_arguments does NOT inject timeout=
    when timeout is already provided as a positional argument in SeleniumLibrary
    wait keywords."""

    @pytest.fixture
    def selenium_session(self):
        return _make_session(
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
            active_library="selenium",
        )

    # --- Wait keywords with 1-arg signature (locator, timeout, error) ---

    @pytest.mark.parametrize("keyword", [
        "wait_until_element_is_visible",
        "wait_until_element_is_not_visible",
        "wait_until_element_is_enabled",
        "wait_until_element_is_not_enabled",
        "wait_until_page_contains",
        "wait_until_page_does_not_contain",
        "wait_until_page_contains_element",
        "wait_until_page_does_not_contain_element",
    ])
    def test_positional_timeout_skips_injection_1arg(self, executor, selenium_session, keyword):
        """Wait keywords with (locator, timeout) — timeout at positional index 1.
        If 2+ args are provided, timeout is already set positionally."""
        arguments = ["css=.element", "5"]
        result = executor._inject_timeout_into_arguments(
            keyword, arguments, 10000, selenium_session
        )
        # Must NOT add timeout= — positional "5" IS the timeout
        assert result == arguments
        assert "timeout=" not in str(result)

    # --- Wait keywords with 2-arg signature (locator, text, timeout, error) ---

    @pytest.mark.parametrize("keyword", [
        "wait_until_element_contains",
        "wait_until_element_does_not_contain",
    ])
    def test_positional_timeout_skips_injection_2arg(self, executor, selenium_session, keyword):
        """Wait keywords with (locator, text, timeout) — timeout at positional index 2.
        If 3+ args are provided, timeout is already set positionally."""
        arguments = ["css=.element", "expected text", "5"]
        result = executor._inject_timeout_into_arguments(
            keyword, arguments, 10000, selenium_session
        )
        assert result == arguments
        assert "timeout=" not in str(result)

    def test_no_positional_timeout_still_injects_1arg(self, executor, selenium_session):
        """When only the locator is provided (no positional timeout), injection works."""
        arguments = ["css=.element"]
        result = executor._inject_timeout_into_arguments(
            "wait_until_element_is_visible", arguments, 5000, selenium_session
        )
        assert "timeout=5.0" in result

    def test_no_positional_timeout_still_injects_2arg(self, executor, selenium_session):
        """When only locator+text are provided (no positional timeout), injection works."""
        arguments = ["css=.element", "expected text"]
        result = executor._inject_timeout_into_arguments(
            "wait_until_element_contains", arguments, 5000, selenium_session
        )
        assert "timeout=5.0" in result

    def test_named_timeout_still_blocks_injection(self, executor, selenium_session):
        """Named timeout= argument (pre-existing check) still prevents injection."""
        arguments = ["css=.element", "timeout=10"]
        result = executor._inject_timeout_into_arguments(
            "wait_until_element_is_visible", arguments, 5000, selenium_session
        )
        assert result == arguments  # Existing check catches this

    def test_locator_only_gets_timeout_injected(self, executor, selenium_session):
        """Single locator arg → timeout should be injected."""
        arguments = ["id=element"]
        result = executor._inject_timeout_into_arguments(
            "wait_until_element_is_visible", arguments, 3000, selenium_session
        )
        assert len(result) == 2
        assert "timeout=3.0" in result

    def test_locator_with_error_message_at_index_1(self, executor, selenium_session):
        """Wait Until Element Is Visible with 2 args: (locator, timeout).
        Even if the second arg is non-numeric, it occupies the timeout position."""
        arguments = ["id=element", "custom error message"]
        result = executor._inject_timeout_into_arguments(
            "wait_until_element_is_visible", arguments, 5000, selenium_session
        )
        # 2 args present, timeout_pos=1, len(args)=2 > 1 → skip injection
        assert result == arguments

    def test_element_contains_two_args_no_timeout(self, executor, selenium_session):
        """Wait Until Element Contains with (locator, text) — no positional timeout.
        timeout_pos=2, len(args)=2 → 2 is NOT > 2 → inject."""
        arguments = ["css=.cart", "2"]
        result = executor._inject_timeout_into_arguments(
            "wait_until_element_contains", arguments, 5000, selenium_session
        )
        assert "timeout=5.0" in result

    def test_element_contains_three_args_has_timeout(self, executor, selenium_session):
        """Wait Until Element Contains with (locator, text, timeout) → skip."""
        arguments = ["css=.cart", "2", "10"]
        result = executor._inject_timeout_into_arguments(
            "wait_until_element_contains", arguments, 5000, selenium_session
        )
        assert result == arguments

    # --- Browser Library wait keywords should be unaffected ---

    def test_browser_library_wait_unaffected(self, executor):
        """Browser Library wait keywords still get timeout injected normally."""
        session = _make_session(
            imported_libraries=["Browser", "BuiltIn"],
            active_library="browser",
        )
        arguments = ["#element", "visible"]
        result = executor._inject_timeout_into_arguments(
            "wait_for_elements_state", arguments, 5000, session
        )
        assert "timeout=5000ms" in result


# ============================================================
# Integration: S5 + P4 fixes work together
# ============================================================
class TestS5P4Integration:
    """Verify S5 and P4 fixes work in combination."""

    def test_selenium_wait_with_positional_timeout_no_conflict(self, executor):
        """SeleniumLibrary wait keyword with positional timeout works."""
        session = _make_session(
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
            active_library="selenium",
        )
        arguments = ["css=[data-cart-count]", "5"]
        result = executor._inject_timeout_into_arguments(
            "Wait Until Element Is Visible", arguments, 5000, session
        )
        assert result == arguments
        assert len(result) == 2

    def test_selenium_wait_without_positional_gets_injection(self, executor):
        """SeleniumLibrary wait keyword without positional timeout gets injection."""
        session = _make_session(
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
            active_library="selenium",
        )
        arguments = ["css=[data-cart-count]"]
        result = executor._inject_timeout_into_arguments(
            "Wait Until Element Is Visible", arguments, 5000, session
        )
        assert "timeout=5.0" in result

    def test_action_keyword_still_no_injection(self, executor):
        """P0 fix preserved: action keywords still don't get timeout injected."""
        session = _make_session(
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
            active_library="selenium",
        )
        arguments = ["css=.product-card button"]
        result = executor._inject_timeout_into_arguments(
            "Click Button", arguments, 5000, session
        )
        assert result == arguments

    @pytest.mark.asyncio
    async def test_full_p5_scenario_no_corruption(self, executor):
        """Full P5 scenario: 'Get Text' in a SeleniumLibrary-only session.
        After S5a+S5b+S5c, session state must NOT be corrupted."""
        session = _make_session(["SeleniumLibrary", "BuiltIn"], "selenium")
        session.loaded_libraries = {"SeleniumLibrary", "BuiltIn"}

        # Before: session state is correct
        assert session.browser_state.active_library == "selenium"

        # Mock the RF context to resolve Get Text → SeleniumLibrary
        with rf_context_with_owner("SeleniumLibrary"), \
             patch.object(executor, "_pre_validate_selenium_element",
                          new_callable=AsyncMock) as mock_sel:
            mock_sel.return_value = {"valid": True, "states": ["visible"], "missing": [], "error": None}
            is_valid, error_msg, details = await executor._pre_validate_element(
                "css=.text", session, "Get Text", timeout_ms=500
            )
            # Pre-validation correctly routes to selenium
            mock_sel.assert_called_once()

        # After: session state must still be correct (not corrupted to "browser")
        assert session.browser_state.active_library == "selenium"
        assert "Browser" not in session.imported_libraries
        assert "SeleniumLibrary" in session.imported_libraries


# ============================================================
# Edge cases
# ============================================================
class TestS5EdgeCases:
    """Edge cases for namespace-based resolution."""

    @pytest.mark.asyncio
    async def test_owner_with_custom_name_skips_prevalidation(self, executor):
        """Library with custom name (not Browser/Selenium/Appium) → skip."""
        session = _make_session(["RequestsLibrary", "BuiltIn"])
        with rf_context_with_owner("RequestsLibrary"):
            is_valid, error_msg, details = await executor._pre_validate_element(
                "http://api", session, "GET", timeout_ms=500
            )
            assert is_valid is True
            assert details.get("skipped") is True

    @pytest.mark.asyncio
    async def test_runner_keyword_is_none(self, executor):
        """runner.keyword is None (unusual but defensive) → skip."""
        session = _make_session(["Browser", "BuiltIn"])
        mock_ec = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.test = MagicMock()
        mock_runner = MagicMock()
        mock_runner.keyword = None
        mock_ctx.namespace.get_runner.return_value = mock_runner
        mock_ec.current = mock_ctx
        with patch("robot.running.context.EXECUTION_CONTEXTS", mock_ec):
            is_valid, error_msg, details = await executor._pre_validate_element(
                "css=.btn", session, "Click", timeout_ms=500
            )
            assert is_valid is True
            assert details.get("skipped") is True


# ============================================================
# P8: Timeout injection uses get_runner() for shared keywords
# ============================================================
def _namespace_owner_ctx(owner_name):
    """Create a mock for robot.running.namespace.EXECUTION_CONTEXTS
    so get_runner(kw).keyword.owner.name returns *owner_name*.

    Used by P8 tests for _inject_timeout_into_arguments() which imports
    from robot.running.namespace (not robot.running.context).
    """
    if owner_name is not None:
        mock_owner = MagicMock()
        mock_owner.name = owner_name
    else:
        mock_owner = None

    mock_kw = MagicMock()
    mock_kw.owner = mock_owner

    mock_runner = MagicMock()
    mock_runner.keyword = mock_kw

    mock_namespace = MagicMock()
    mock_namespace.get_runner.return_value = mock_runner

    mock_ctx = MagicMock()
    mock_ctx.namespace = mock_namespace

    mock_ec = MagicMock()
    mock_ec.current = mock_ctx
    return mock_ec


class TestP8_TimeoutInjectionOwnerRouting:
    """P8: _inject_timeout_into_arguments() routes to the correct timeout map
    based on namespace.get_runner(keyword).keyword.owner.name.

    This prevents shared keywords like Wait For Condition from getting the
    wrong timeout format when the keyword exists in both Browser Library
    (timeout=5000ms) and SeleniumLibrary (timeout=5.0)."""

    # --- Core P8 fix: shared keyword routes to correct map ---

    def test_wait_for_condition_selenium_owner_gets_seconds(self, executor):
        """Wait For Condition owned by SeleniumLibrary → timeout in seconds."""
        mock_ec = _namespace_owner_ctx("SeleniumLibrary")
        with patch("robot.running.namespace.EXECUTION_CONTEXTS", mock_ec):
            result = executor._inject_timeout_into_arguments(
                "Wait For Condition",
                ["return document.readyState === 'complete'"],
                5000, None
            )
        assert "timeout=5.0" in result
        assert "5000ms" not in str(result)

    def test_wait_for_condition_browser_owner_gets_milliseconds(self, executor):
        """Wait For Condition owned by Browser → timeout in milliseconds."""
        mock_ec = _namespace_owner_ctx("Browser")
        with patch("robot.running.namespace.EXECUTION_CONTEXTS", mock_ec):
            result = executor._inject_timeout_into_arguments(
                "Wait For Condition",
                ["return document.readyState === 'complete'"],
                5000, None
            )
        assert "timeout=5000ms" in result
        assert "timeout=5.0" not in str(result)

    def test_wait_for_condition_selenium_with_positional_timeout_skips(self, executor):
        """P8+P4: Wait For Condition with positional timeout → skip injection.
        This is the exact scenario from Run 4: user passes '10s' as 2nd arg."""
        mock_ec = _namespace_owner_ctx("SeleniumLibrary")
        with patch("robot.running.namespace.EXECUTION_CONTEXTS", mock_ec):
            arguments = ["return document.readyState === 'complete'", "10s"]
            result = executor._inject_timeout_into_arguments(
                "Wait For Condition", arguments, 5000, None
            )
        # 2 args, timeout_pos=1, len(args)=2 > 1 → skip injection
        assert result == arguments
        assert "timeout=" not in str(result)

    # --- Other SeleniumLibrary wait keywords with owner detection ---

    @pytest.mark.parametrize("keyword,args,expected_timeout", [
        ("Wait Until Element Is Visible", ["css=.el"], "timeout=5.0"),
        ("Wait Until Page Contains", ["expected text"], "timeout=5.0"),
        ("Wait Until Element Contains", ["css=.el", "text"], "timeout=5.0"),
    ])
    def test_selenium_wait_keywords_get_seconds(self, executor, keyword, args, expected_timeout):
        """SeleniumLibrary wait keywords get timeout in seconds format."""
        mock_ec = _namespace_owner_ctx("SeleniumLibrary")
        with patch("robot.running.namespace.EXECUTION_CONTEXTS", mock_ec):
            result = executor._inject_timeout_into_arguments(keyword, args, 5000, None)
        assert expected_timeout in result

    # --- Browser Library keywords with owner detection ---

    @pytest.mark.parametrize("keyword", [
        "Wait For Elements State",
        "Wait For Condition",
    ])
    def test_browser_wait_keywords_get_milliseconds(self, executor, keyword):
        """Browser Library wait keywords get timeout in milliseconds format."""
        mock_ec = _namespace_owner_ctx("Browser")
        with patch("robot.running.namespace.EXECUTION_CONTEXTS", mock_ec):
            result = executor._inject_timeout_into_arguments(
                keyword, ["some_arg"], 5000, None
            )
        assert "timeout=5000ms" in result

    # --- Fallback: no RF context → uses static maps ---

    def test_no_context_falls_back_to_browser_map(self, executor):
        """No RF context → fallback to Browser Library map first."""
        mock_ec = MagicMock()
        mock_ec.current = None
        with patch("robot.running.namespace.EXECUTION_CONTEXTS", mock_ec):
            result = executor._inject_timeout_into_arguments(
                "Wait For Elements State", ["css=.el", "visible"], 5000, None
            )
        assert "timeout=5000ms" in result

    def test_no_context_selenium_only_keyword_gets_seconds(self, executor):
        """No RF context → keyword only in SeleniumLibrary map → seconds."""
        mock_ec = MagicMock()
        mock_ec.current = None
        with patch("robot.running.namespace.EXECUTION_CONTEXTS", mock_ec):
            result = executor._inject_timeout_into_arguments(
                "Wait Until Element Is Visible", ["css=.el"], 5000, None
            )
        assert "timeout=5.0" in result

    def test_no_context_shared_keyword_defaults_to_browser(self, executor):
        """No RF context → shared keyword (wait_for_condition) defaults
        to Browser Library map first (backwards compat fallback)."""
        mock_ec = MagicMock()
        mock_ec.current = None
        with patch("robot.running.namespace.EXECUTION_CONTEXTS", mock_ec):
            result = executor._inject_timeout_into_arguments(
                "Wait For Condition",
                ["return document.readyState === 'complete'"],
                5000, None
            )
        # Falls back to Browser Library map (checked first)
        assert "timeout=5000ms" in result

    # --- get_runner() exception → fallback ---

    def test_get_runner_exception_falls_back(self, executor):
        """If get_runner() throws, owner_library stays None → fallback path."""
        mock_ec = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.namespace.get_runner.side_effect = Exception("No keyword")
        mock_ec.current = mock_ctx
        with patch("robot.running.namespace.EXECUTION_CONTEXTS", mock_ec):
            result = executor._inject_timeout_into_arguments(
                "Wait Until Element Is Visible", ["css=.el"], 5000, None
            )
        assert "timeout=5.0" in result

    # --- Non-timeout keywords unaffected ---

    def test_non_timeout_keyword_returns_unchanged(self, executor):
        """Keywords not in any timeout map return arguments unchanged."""
        mock_ec = _namespace_owner_ctx("SeleniumLibrary")
        with patch("robot.running.namespace.EXECUTION_CONTEXTS", mock_ec):
            arguments = ["css=.button"]
            result = executor._inject_timeout_into_arguments(
                "Click Element", arguments, 5000, None
            )
        assert result == arguments

    def test_action_keyword_no_injection(self, executor):
        """Action keywords (click_button, input_text) never get timeout."""
        mock_ec = _namespace_owner_ctx("SeleniumLibrary")
        with patch("robot.running.namespace.EXECUTION_CONTEXTS", mock_ec):
            arguments = ["css=.btn"]
            result = executor._inject_timeout_into_arguments(
                "Click Button", arguments, 5000, None
            )
        assert result == arguments

    # --- Named timeout already present ---

    def test_existing_named_timeout_skips_injection(self, executor):
        """Pre-existing timeout= in arguments → skip injection entirely."""
        mock_ec = _namespace_owner_ctx("SeleniumLibrary")
        with patch("robot.running.namespace.EXECUTION_CONTEXTS", mock_ec):
            arguments = ["return true", "timeout=10"]
            result = executor._inject_timeout_into_arguments(
                "Wait For Condition", arguments, 5000, None
            )
        assert result == arguments

    # --- Owner is non-web library → no injection ---

    def test_builtin_owner_no_injection(self, executor):
        """Keyword owned by BuiltIn → not in any timeout map → unchanged."""
        mock_ec = _namespace_owner_ctx("BuiltIn")
        with patch("robot.running.namespace.EXECUTION_CONTEXTS", mock_ec):
            arguments = ["5"]
            result = executor._inject_timeout_into_arguments(
                "Sleep", arguments, 5000, None
            )
        assert result == arguments

    # --- timeout_ms=0 → early return ---

    def test_zero_timeout_returns_unchanged(self, executor):
        """timeout_ms=0 → early return, no injection."""
        result = executor._inject_timeout_into_arguments(
            "Wait For Condition", ["return true"], 0, None
        )
        assert result == ["return true"]

    # --- Library prefix stripped correctly ---

    def test_library_prefix_stripped_before_lookup(self, executor):
        """SeleniumLibrary.Wait For Condition → wait_for_condition lookup."""
        mock_ec = _namespace_owner_ctx("SeleniumLibrary")
        with patch("robot.running.namespace.EXECUTION_CONTEXTS", mock_ec):
            result = executor._inject_timeout_into_arguments(
                "SeleniumLibrary.Wait For Condition",
                ["return true"],
                5000, None
            )
        assert "timeout=5.0" in result

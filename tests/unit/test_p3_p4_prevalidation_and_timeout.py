"""Tests for P3/P4 fixes discovered during Demoshop SeleniumLibrary revalidation.

P3: Pre-validation uses Browser Library keywords (Get Element States) for
    SeleniumLibrary sessions because active_library detection is incomplete.
P4: Timeout injection conflicts with positional timeout arguments in
    SeleniumLibrary wait keywords ("got multiple values for argument 'timeout'").

Run with: uv run pytest tests/unit/test_p3_p4_prevalidation_and_timeout.py -v
"""

__test__ = True

from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
import pytest

from robotmcp.components.execution.keyword_executor import KeywordExecutor
from robotmcp.models.config_models import ExecutionConfig
from robotmcp.models.session_models import ExecutionSession


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
    return session


# ============================================================
# P3: Pre-validation must use correct library for SeleniumLibrary sessions
# ============================================================
class TestP3_PreValidationLibraryDetection:
    """Verify that _pre_validate_element routes to the correct library-specific
    validator based on session.imported_libraries, even when active_library is
    not set or is incorrectly set."""

    @pytest.mark.asyncio
    async def test_selenium_session_active_routes_to_selenium_validator(self, executor):
        """SeleniumLibrary session with active_library='selenium' should use selenium validator."""
        session = _make_session(
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
            active_library="selenium",
        )
        with patch.object(
            executor, "_pre_validate_selenium_element", new_callable=AsyncMock
        ) as mock_sel:
            mock_sel.return_value = {"valid": True, "states": ["visible"], "missing": [], "error": None}
            is_valid, error_msg, details = await executor._pre_validate_element(
                "css=.button", session, "Click Button", timeout_ms=500
            )
            mock_sel.assert_called_once()
            assert is_valid

    @pytest.mark.asyncio
    async def test_selenium_session_active_does_not_use_browser_validator(self, executor):
        """SeleniumLibrary session must NOT call _pre_validate_browser_element."""
        session = _make_session(
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
            active_library="selenium",
        )
        with patch.object(
            executor, "_pre_validate_browser_element", new_callable=AsyncMock
        ) as mock_browser, patch.object(
            executor, "_pre_validate_selenium_element", new_callable=AsyncMock
        ) as mock_sel:
            mock_sel.return_value = {"valid": True, "states": ["visible"], "missing": [], "error": None}
            await executor._pre_validate_element(
                "css=.button", session, "Click Button", timeout_ms=500
            )
            mock_browser.assert_not_called()

    @pytest.mark.asyncio
    async def test_selenium_no_active_library_skips_prevalidation(self, executor):
        """SeleniumLibrary imported but active_library=None means no browser open → skip."""
        session = _make_session(
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
            active_library=None,
        )
        is_valid, error_msg, details = await executor._pre_validate_element(
            "css=.button", session, "Click Button", timeout_ms=500
        )
        assert is_valid is True
        assert details.get("skipped") is True

    @pytest.mark.asyncio
    async def test_wrong_active_library_corrected_by_imports(self, executor):
        """If active_library='browser' but only SeleniumLibrary is imported,
        the validation should still route to the selenium validator."""
        session = _make_session(
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
            active_library="browser",  # Wrong! Browser not imported.
        )
        with patch.object(
            executor, "_pre_validate_selenium_element", new_callable=AsyncMock
        ) as mock_sel, patch.object(
            executor, "_pre_validate_browser_element", new_callable=AsyncMock
        ) as mock_browser:
            mock_sel.return_value = {"valid": True, "states": ["visible"], "missing": [], "error": None}
            is_valid, error_msg, details = await executor._pre_validate_element(
                "css=.button", session, "Click Button", timeout_ms=500
            )
            mock_sel.assert_called_once()
            mock_browser.assert_not_called()
            assert is_valid

    @pytest.mark.asyncio
    async def test_wrong_active_library_selenium_without_seleniumlib(self, executor):
        """If active_library='selenium' but SeleniumLibrary is NOT imported,
        the validation should correct based on actual imports."""
        session = _make_session(
            imported_libraries=["Browser", "BuiltIn"],
            active_library="selenium",  # Wrong! SeleniumLibrary not imported.
        )
        with patch.object(
            executor, "_pre_validate_browser_element", new_callable=AsyncMock
        ) as mock_browser, patch.object(
            executor, "_pre_validate_selenium_element", new_callable=AsyncMock
        ) as mock_sel:
            mock_browser.return_value = {"valid": True, "states": ["visible"], "missing": [], "error": None}
            is_valid, error_msg, details = await executor._pre_validate_element(
                "css=.button", session, "Click Button", timeout_ms=500
            )
            mock_browser.assert_called_once()
            mock_sel.assert_not_called()

    @pytest.mark.asyncio
    async def test_browser_session_routes_to_browser_validator(self, executor):
        """Browser Library session with active_library='browser' should use browser validator."""
        session = _make_session(
            imported_libraries=["Browser", "BuiltIn"],
            active_library="browser",
        )
        with patch.object(
            executor, "_pre_validate_browser_element", new_callable=AsyncMock
        ) as mock_browser:
            mock_browser.return_value = {"valid": True, "states": ["visible"], "missing": [], "error": None}
            is_valid, error_msg, details = await executor._pre_validate_element(
                "css=.button", session, "Click Button", timeout_ms=500
            )
            mock_browser.assert_called_once()
            assert is_valid

    @pytest.mark.asyncio
    async def test_browser_session_no_active_library_skips_prevalidation(self, executor):
        """Browser imported but active_library=None means no browser open → skip."""
        session = _make_session(
            imported_libraries=["Browser", "BuiltIn"],
            active_library=None,
        )
        is_valid, error_msg, details = await executor._pre_validate_element(
            "css=.button", session, "Click Button", timeout_ms=500
        )
        assert is_valid is True
        assert details.get("skipped") is True

    @pytest.mark.asyncio
    async def test_appium_session_detected_from_imports(self, executor):
        """AppiumLibrary with active_library=None should detect from imports (special case)."""
        session = _make_session(
            imported_libraries=["AppiumLibrary", "BuiltIn"],
            active_library=None,
        )
        with patch.object(
            executor, "_pre_validate_appium_element", new_callable=AsyncMock
        ) as mock_appium:
            mock_appium.return_value = {"valid": True, "states": ["visible"], "missing": [], "error": None}
            is_valid, error_msg, details = await executor._pre_validate_element(
                "id=button", session, "Click Button", timeout_ms=500
            )
            mock_appium.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_web_library_skips_prevalidation(self, executor):
        """No web library imported → pre-validation skipped."""
        session = _make_session(
            imported_libraries=["BuiltIn", "Collections"],
            active_library=None,
        )
        is_valid, error_msg, details = await executor._pre_validate_element(
            "id=button", session, "Click Button", timeout_ms=500
        )
        assert is_valid is True
        assert details.get("skipped") is True

    @pytest.mark.asyncio
    async def test_empty_imported_libraries_skips_prevalidation(self, executor):
        """Empty imported_libraries → pre-validation skipped."""
        session = _make_session(
            imported_libraries=[],
            active_library=None,
        )
        is_valid, error_msg, details = await executor._pre_validate_element(
            "id=button", session, "Click Button", timeout_ms=500
        )
        assert is_valid is True
        assert details.get("skipped") is True

    @pytest.mark.asyncio
    async def test_none_imported_libraries_skips_prevalidation(self, executor):
        """None imported_libraries → pre-validation skipped, no crash."""
        session = _make_session(
            imported_libraries=[],
            active_library=None,
        )
        session.imported_libraries = None
        is_valid, error_msg, details = await executor._pre_validate_element(
            "id=button", session, "Click Button", timeout_ms=500
        )
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_correct_active_library_preserved(self, executor):
        """When active_library matches imports, it should be preserved."""
        session = _make_session(
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
            active_library="selenium",
        )
        with patch.object(
            executor, "_pre_validate_selenium_element", new_callable=AsyncMock
        ) as mock_sel:
            mock_sel.return_value = {"valid": True, "states": ["visible"], "missing": [], "error": None}
            await executor._pre_validate_element(
                "css=.button", session, "Click Button", timeout_ms=500
            )
            mock_sel.assert_called_once()

    @pytest.mark.asyncio
    async def test_both_libraries_uses_active_library(self, executor):
        """When both libraries imported and active_library set, use active."""
        session = _make_session(
            imported_libraries=["Browser", "SeleniumLibrary", "BuiltIn"],
            active_library="selenium",
        )
        with patch.object(
            executor, "_pre_validate_selenium_element", new_callable=AsyncMock
        ) as mock_sel:
            mock_sel.return_value = {"valid": True, "states": ["visible"], "missing": [], "error": None}
            await executor._pre_validate_element(
                "css=.button", session, "Click Button", timeout_ms=500
            )
            mock_sel.assert_called_once()

    @pytest.mark.asyncio
    async def test_both_libraries_no_active_skips_prevalidation(self, executor):
        """When both libraries imported but no active_library, skip (no browser open)."""
        session = _make_session(
            imported_libraries=["Browser", "SeleniumLibrary", "BuiltIn"],
            active_library=None,
        )
        is_valid, error_msg, details = await executor._pre_validate_element(
            "css=.button", session, "Click Button", timeout_ms=500
        )
        assert is_valid is True
        assert details.get("skipped") is True

    @pytest.mark.asyncio
    async def test_case_insensitive_active_library(self, executor):
        """active_library matching should be case-insensitive."""
        session = _make_session(
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
            active_library="Selenium",  # Capital S
        )
        with patch.object(
            executor, "_pre_validate_selenium_element", new_callable=AsyncMock
        ) as mock_sel:
            mock_sel.return_value = {"valid": True, "states": ["visible"], "missing": [], "error": None}
            await executor._pre_validate_element(
                "css=.button", session, "Click Button", timeout_ms=500
            )
            mock_sel.assert_called_once()


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
        # This is a tricky case: user might pass (locator, error_message) without timeout.
        # But the signature is (locator, timeout=None, error=None), so the 2nd positional
        # arg IS the timeout param, not the error. We correctly detect this.
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
# Integration: Both P3 and P4 fixes work together
# ============================================================
class TestP3P4Integration:
    """Verify P3 and P4 fixes work in combination."""

    @pytest.mark.asyncio
    async def test_selenium_prevalidation_failure_gives_helpful_error(self, executor):
        """When selenium pre-validation fails, error should mention
        SeleniumLibrary concepts, not Browser Library concepts."""
        session = _make_session(
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
            active_library="selenium",
        )
        with patch.object(
            executor, "_pre_validate_selenium_element", new_callable=AsyncMock
        ) as mock_sel:
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
            # Must NOT mention "Get Element States" (Browser Library keyword)
            assert "Get Element States" not in (error_msg or "")

    def test_selenium_wait_with_positional_timeout_no_conflict(self, executor):
        """After P3+P4: SeleniumLibrary wait keyword with positional timeout works."""
        session = _make_session(
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
            active_library="selenium",
        )
        # Wait Until Element Is Visible    locator    5
        arguments = ["css=[data-cart-count]", "5"]
        result = executor._inject_timeout_into_arguments(
            "Wait Until Element Is Visible", arguments, 5000, session
        )
        # No timeout= injected — positional "5" is the timeout
        assert result == arguments
        assert len(result) == 2

    def test_selenium_wait_without_positional_gets_injection(self, executor):
        """After P3+P4: SeleniumLibrary wait keyword without positional timeout gets injection."""
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

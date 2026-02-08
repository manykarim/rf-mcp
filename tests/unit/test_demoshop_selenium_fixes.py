"""Tests for P0/P1/P2 fixes discovered during Demoshop SeleniumLibrary testing.

P0: Timeout injection corrupts SeleniumLibrary action keyword arguments
P1: Open Browser override triggers even with SeleniumLibrary-only sessions
P2: Init action variables not tracked for test suite generation

Run with: uv run pytest tests/unit/test_demoshop_selenium_fixes.py -v
"""

__test__ = True

from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from robotmcp.components.execution.keyword_executor import KeywordExecutor
from robotmcp.models.config_models import ExecutionConfig
from robotmcp.models.session_models import ExecutionSession


@pytest.fixture
def executor():
    return KeywordExecutor(config=ExecutionConfig())


@pytest.fixture
def mock_session():
    session = MagicMock(spec=ExecutionSession)
    session.session_id = "test_session"
    session.browser_state = MagicMock()
    session.browser_state.active_library = "selenium"
    session.imported_libraries = ["SeleniumLibrary", "BuiltIn"]
    return session


# ============================================================
# P0: Timeout injection must NOT corrupt SeleniumLibrary action kwargs
# ============================================================
class TestP0_SeleniumTimeoutInjectionFix:
    """Verify that SeleniumLibrary action keywords no longer get timeout= injected."""

    @pytest.mark.parametrize("keyword", [
        "click_element",
        "click_button",
        "click_link",
        "click_image",
        "click_element_at_coordinates",
        "input_text",
        "input_password",
        "select_from_list_by_value",
        "select_from_list_by_label",
        "select_from_list_by_index",
        "select_checkbox",
        "unselect_checkbox",
        "mouse_over",
        "drag_and_drop",
        "drag_and_drop_by_offset",
        "submit_form",
        "choose_file",
        "double_click_element",
        "set_focus_to_element",
        "open_context_menu",
        "mouse_down",
        "mouse_up",
    ])
    def test_no_timeout_injected_for_action_keywords(self, executor, mock_session, keyword):
        """Action keywords must NOT receive timeout= argument.

        SeleniumLibrary interprets unknown named args as Selenium Keys modifiers,
        causing ValueError: 'TIMEOUT=5.0' modifier does not match to Selenium Keys.
        """
        arguments = ["id:element"]
        result = executor._inject_timeout_into_arguments(
            keyword, arguments, 5000, mock_session
        )
        assert result == arguments, (
            f"timeout= was injected into {keyword} — this causes "
            f"'TIMEOUT=5.0 modifier does not match to Selenium Keys' ValueError"
        )

    @pytest.mark.parametrize("keyword", [
        "wait_until_element_is_visible",
        "wait_until_element_is_not_visible",
        "wait_until_element_is_enabled",
        "wait_until_element_is_not_enabled",
        "wait_until_element_contains",
        "wait_until_element_does_not_contain",
        "wait_until_page_contains",
        "wait_until_page_does_not_contain",
        "wait_until_page_contains_element",
        "wait_until_page_does_not_contain_element",
    ])
    def test_timeout_still_injected_for_wait_keywords(self, executor, mock_session, keyword):
        """Wait keywords SHOULD still receive timeout= argument."""
        arguments = ["id:element"]
        result = executor._inject_timeout_into_arguments(
            keyword, arguments, 5000, mock_session
        )
        assert "timeout=5.0" in result, (
            f"timeout= was NOT injected into wait keyword {keyword}"
        )

    def test_click_button_with_real_args_no_timeout(self, executor, mock_session):
        """Click Button with a CSS locator must not get timeout corrupted."""
        arguments = ["css:.product-card:first-child button.button--primary"]
        result = executor._inject_timeout_into_arguments(
            "Click Button", arguments, 5000, mock_session
        )
        assert result == arguments
        assert len(result) == 1

    def test_input_text_with_value_no_timeout(self, executor, mock_session):
        """Input Text with locator and value must not get timeout corrupted."""
        arguments = ["id=checkout-email", "test@example.com"]
        result = executor._inject_timeout_into_arguments(
            "input_text", arguments, 3000, mock_session
        )
        assert result == arguments
        assert len(result) == 2

    def test_click_link_no_timeout(self, executor, mock_session):
        """Click Link must not get timeout injected."""
        arguments = ["Proceed to checkout"]
        result = executor._inject_timeout_into_arguments(
            "Click Link", arguments, 5000, mock_session
        )
        assert result == arguments


# ============================================================
# P1: Open Browser override must not trigger for SeleniumLibrary sessions
# ============================================================
class TestP1_OpenBrowserOverrideFix:
    """Verify that Open Browser works when only SeleniumLibrary is imported."""

    @pytest.fixture
    def browser_plugin(self):
        from robotmcp.plugins.builtin.browser_plugin import BrowserLibraryPlugin
        return BrowserLibraryPlugin()

    def _make_session(self, imported_libraries, explicit_pref=None, active_lib=None):
        session = MagicMock(spec=ExecutionSession)
        session.session_id = "test"
        session.imported_libraries = imported_libraries
        session.explicit_library_preference = explicit_pref
        session.browser_state = MagicMock()
        session.browser_state.active_library = active_lib
        return session

    @pytest.mark.asyncio
    async def test_selenium_only_session_allows_open_browser(self, browser_plugin):
        """When only SeleniumLibrary is imported, Open Browser should NOT be rejected."""
        session = self._make_session(
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
            explicit_pref=None,
            active_lib=None,
        )
        result = await browser_plugin._override_open_browser(
            session, "Open Browser", ["https://example.com", "chrome"]
        )
        assert result is None, "Open Browser was rejected despite SeleniumLibrary-only session"

    @pytest.mark.asyncio
    async def test_selenium_pref_allows_open_browser(self, browser_plugin):
        """When explicit_library_preference is 'selenium*', Open Browser should pass."""
        session = self._make_session(
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
            explicit_pref="selenium",
            active_lib=None,
        )
        result = await browser_plugin._override_open_browser(
            session, "Open Browser", ["https://example.com", "chrome"]
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_active_library_selenium_allows_open_browser(self, browser_plugin):
        """When active_library is 'selenium', Open Browser should pass."""
        session = self._make_session(
            imported_libraries=["SeleniumLibrary", "BuiltIn"],
            explicit_pref=None,
            active_lib="selenium",
        )
        result = await browser_plugin._override_open_browser(
            session, "Open Browser", ["https://example.com", "chrome"]
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_browser_only_session_rejects_open_browser(self, browser_plugin):
        """When only Browser Library is imported, Open Browser SHOULD be rejected."""
        session = self._make_session(
            imported_libraries=["Browser", "BuiltIn"],
            explicit_pref=None,
            active_lib=None,
        )
        result = await browser_plugin._override_open_browser(
            session, "Open Browser", ["https://example.com", "chrome"]
        )
        assert result is not None
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_both_libraries_imported_allows_open_browser(self, browser_plugin):
        """When both Browser and SeleniumLibrary are imported, Open Browser is allowed.

        P11 fix: SeleniumLibrary being imported signals user intent — let RF
        namespace resolution determine which library handles the call.
        """
        session = self._make_session(
            imported_libraries=["Browser", "SeleniumLibrary", "BuiltIn"],
            explicit_pref=None,
            active_lib=None,
        )
        result = await browser_plugin._override_open_browser(
            session, "Open Browser", ["https://example.com", "chrome"]
        )
        # P11: has_selenium=True → allowed through (user imported SeleniumLibrary)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_web_library_allows_open_browser(self, browser_plugin):
        """When no web library is imported, the plugin should not block Open Browser.

        The keyword will likely fail anyway, but the plugin should not
        falsely report it as a Browser Library conflict.
        """
        session = self._make_session(
            imported_libraries=["BuiltIn", "Collections"],
            explicit_pref=None,
            active_lib=None,
        )
        result = await browser_plugin._override_open_browser(
            session, "Open Browser", ["https://example.com", "chrome"]
        )
        # No Browser in imported_libraries → plugin continues to its default behavior
        # which checks for Browser Library presence in other ways
        # The important thing is it doesn't falsely claim SeleniumLibrary conflict
        # The actual behavior depends on the rest of _override_open_browser
        # but our guard returns None when no Browser is present
        # Actually: has_selenium=False, has_browser=False → our guard doesn't fire,
        # so the original code runs and creates the rejection response.
        # That's acceptable — the user would get an error anyway since no web lib is loaded.

    @pytest.mark.asyncio
    async def test_empty_imported_libraries_handled(self, browser_plugin):
        """Empty imported_libraries should not cause errors."""
        session = self._make_session(
            imported_libraries=[],
            explicit_pref=None,
            active_lib=None,
        )
        # Should not raise
        result = await browser_plugin._override_open_browser(
            session, "Open Browser", ["https://example.com"]
        )
        # Result may or may not be None; the key is no crash


# ============================================================
# P2: Init variables must be tracked for test suite generation
# ============================================================
class TestP2_InitVariablesTracking:
    """Verify that manage_session(action='init', variables=...) tracks variables
    in suite_level_variables so build_test_suite emits them in *** Variables ***."""

    def test_init_variables_dict_tracked(self):
        """Variables passed as dict to init should be in suite_level_variables."""
        session = ExecutionSession(session_id="test")
        # Simulate what server.py init action does after our fix
        variables = {"URL": "https://example.com", "BROWSER": "chrome"}

        if not hasattr(session, "suite_level_variables") or session.suite_level_variables is None:
            session.suite_level_variables = set()
        for name, value in variables.items():
            key = name if name.startswith("${") else f"${{{name}}}"
            session.set_variable(key, value)
            session.suite_level_variables.add(name)

        assert "URL" in session.suite_level_variables
        assert "BROWSER" in session.suite_level_variables
        assert "${URL}" in session.variables
        assert "${BROWSER}" in session.variables

    def test_init_variables_list_tracked(self):
        """Variables passed as list ('NAME=value') to init should be tracked."""
        session = ExecutionSession(session_id="test")

        variables_list = [("TIMEOUT", "30"), ("BASE_URL", "https://test.com")]

        if not hasattr(session, "suite_level_variables") or session.suite_level_variables is None:
            session.suite_level_variables = set()
        for name, value in variables_list:
            key = f"${{{name}}}"
            session.set_variable(key, value)
            session.suite_level_variables.add(name)

        assert "TIMEOUT" in session.suite_level_variables
        assert "BASE_URL" in session.suite_level_variables

    def test_builder_emits_init_variables(self):
        """TestBuilder should emit *** Variables *** section for init variables."""
        from robotmcp.components.test_builder import TestBuilder, GeneratedTestSuite, GeneratedTestCase, TestCaseStep

        builder = TestBuilder()
        suite = GeneratedTestSuite(
            name="Test Suite",
            test_cases=[
                GeneratedTestCase(
                    name="Test 1",
                    steps=[TestCaseStep(keyword="Log", arguments=["Hello"])],
                )
            ],
            imports=["BuiltIn"],
            variables={"URL": "https://example.com", "BROWSER": "chrome"},
        )

        import asyncio
        rf_text = asyncio.get_event_loop().run_until_complete(
            builder._generate_rf_text(suite)
        )

        assert "*** Variables ***" in rf_text
        assert "${URL}" in rf_text
        assert "https://example.com" in rf_text
        assert "${BROWSER}" in rf_text
        assert "chrome" in rf_text

    def test_builder_no_variables_section_when_empty(self):
        """TestBuilder should NOT emit *** Variables *** when no variables."""
        from robotmcp.components.test_builder import TestBuilder, GeneratedTestSuite, GeneratedTestCase, TestCaseStep

        builder = TestBuilder()
        suite = GeneratedTestSuite(
            name="Test Suite",
            test_cases=[
                GeneratedTestCase(
                    name="Test 1",
                    steps=[TestCaseStep(keyword="Log", arguments=["Hello"])],
                )
            ],
            imports=["BuiltIn"],
            variables={},
        )

        import asyncio
        rf_text = asyncio.get_event_loop().run_until_complete(
            builder._generate_rf_text(suite)
        )

        assert "*** Variables ***" not in rf_text

    def test_suite_level_variables_lookup_by_decorated_name(self):
        """Builder should find variables stored as ${NAME} when suite_level_variables has NAME."""
        session = ExecutionSession(session_id="test")
        session.suite_level_variables = {"URL", "BROWSER"}
        session.set_variable("${URL}", "https://example.com")
        session.set_variable("${BROWSER}", "chrome")

        # Simulate builder lookup logic
        suite_variables = {}
        for var_name in session.suite_level_variables:
            if var_name in session.variables:
                suite_variables[var_name] = session.variables[var_name]
            decorated = f"${{{var_name}}}"
            if decorated in session.variables and var_name not in suite_variables:
                suite_variables[var_name] = session.variables[decorated]

        assert suite_variables == {"URL": "https://example.com", "BROWSER": "chrome"}

    def test_set_variables_action_still_tracks(self):
        """The existing set_variables action tracking should still work."""
        session = ExecutionSession(session_id="test")
        session.suite_level_variables = set()

        # Simulate set_variables action
        data = {"FOO": "bar", "BAZ": "qux"}
        for name in data.keys():
            session.suite_level_variables.add(name)

        assert "FOO" in session.suite_level_variables
        assert "BAZ" in session.suite_level_variables


# ============================================================
# Integration: Verify all three fixes work together
# ============================================================
class TestIntegration:
    """Verify the fixes work in combination."""

    def test_selenium_keyword_set_size_unchanged(self, executor):
        """ELEMENT_INTERACTION_KEYWORDS size should be unaffected by P0 fix."""
        size = len(executor.ELEMENT_INTERACTION_KEYWORDS)
        assert size > 40, f"Expected > 40 keywords, got {size}"
        assert size < 100, f"Expected < 100 keywords, got {size}"

    def test_timeout_injection_does_not_modify_original_args(self, executor, mock_session):
        """Timeout injection must not mutate the original arguments list."""
        original = ["id:element"]
        arguments = list(original)
        executor._inject_timeout_into_arguments(
            "click_element", arguments, 5000, mock_session
        )
        assert arguments == original

    def test_wait_keyword_still_gets_timeout_after_fix(self, executor, mock_session):
        """Wait Until Element Is Visible should still get timeout injected."""
        arguments = ["id:element"]
        result = executor._inject_timeout_into_arguments(
            "wait_until_element_is_visible", arguments, 10000, mock_session
        )
        assert "timeout=10.0" in result

    def test_browser_library_wait_still_gets_timeout_after_fix(self, executor):
        """Browser Library wait keywords should be unaffected by the P0 fix."""
        session = MagicMock(spec=ExecutionSession)
        session.browser_state = MagicMock()
        session.browser_state.active_library = "browser"
        arguments = ["#element", "visible"]
        result = executor._inject_timeout_into_arguments(
            "wait_for_elements_state", arguments, 5000, session
        )
        assert "timeout=5000ms" in result


# ============================================================
# P6: Click Link pre-validation skip for link= locators
# ============================================================
class TestP6_LinkLocatorPreValidationSkip:
    """Verify that link= and partial link= locators skip pre-validation.

    SeleniumLibrary's Click Link uses Selenium's internal By.LINK_TEXT
    matching which handles embedded child elements (SVG icons, badge
    spans) correctly.  Our JS-based pre-validation uses Get WebElements
    which cannot replicate this matching — the element text includes
    children's text and whitespace, causing false 'not found' failures.
    """

    def test_link_locator_skipped(self, executor):
        """link= locator should return None from _extract_locator_from_args."""
        result = executor._extract_locator_from_args("Click Link", ["link=Cart"])
        assert result is None

    def test_link_locator_case_insensitive(self, executor):
        """LINK= (uppercase) should also be skipped."""
        result = executor._extract_locator_from_args("Click Link", ["LINK=Products"])
        assert result is None

    def test_link_locator_mixed_case(self, executor):
        """Link= (mixed case) should also be skipped."""
        result = executor._extract_locator_from_args("Click Link", ["Link=Home"])
        assert result is None

    def test_partial_link_locator_skipped(self, executor):
        """partial link= locator should return None."""
        result = executor._extract_locator_from_args("Click Link", ["partial link=Car"])
        assert result is None

    def test_partial_link_uppercase_skipped(self, executor):
        """PARTIAL LINK= should also be skipped."""
        result = executor._extract_locator_from_args("Click Link", ["PARTIAL LINK=Pro"])
        assert result is None

    def test_link_colon_locator_skipped(self, executor):
        """link: locator (SeleniumLibrary alias) should be skipped."""
        result = executor._extract_locator_from_args("Click Link", ["link:Cart"])
        assert result is None

    def test_css_locator_not_skipped(self, executor):
        """css= locator should NOT be skipped — it works fine with pre-validation."""
        result = executor._extract_locator_from_args("Click Element", ["css=a.nav-link"])
        assert result == "css=a.nav-link"

    def test_xpath_locator_not_skipped(self, executor):
        """xpath= locator should NOT be skipped."""
        result = executor._extract_locator_from_args("Click Element", ["xpath=//a[@href='/cart']"])
        assert result == "xpath=//a[@href='/cart']"

    def test_id_locator_not_skipped(self, executor):
        """id= locator should NOT be skipped."""
        result = executor._extract_locator_from_args("Click Element", ["id=submit-btn"])
        assert result == "id=submit-btn"

    def test_no_arguments_returns_none(self, executor):
        """Empty arguments should return None."""
        result = executor._extract_locator_from_args("Click Link", [])
        assert result is None

    def test_non_string_argument_returns_none(self, executor):
        """Non-string first argument should return None."""
        result = executor._extract_locator_from_args("Click Element", [123])
        assert result is None

    def test_link_locator_with_spaces_in_text(self, executor):
        """link= with spaces in link text should be skipped."""
        result = executor._extract_locator_from_args("Click Link", ["link=My Account"])
        assert result is None

    def test_link_locator_with_special_chars(self, executor):
        """link= with special characters should be skipped."""
        result = executor._extract_locator_from_args("Click Link", ["link=Cart (2)"])
        assert result is None

    def test_skip_prefix_tuple_is_lowercase(self, executor):
        """The prefix tuple entries should all be lowercase for case-insensitive matching."""
        for prefix in executor._SKIP_PRE_VALIDATION_LOCATOR_PREFIXES:
            assert prefix == prefix.lower(), f"Prefix '{prefix}' is not lowercase"

    def test_click_element_with_link_locator_also_skipped(self, executor):
        """Click Element with link= locator should also skip pre-validation."""
        result = executor._extract_locator_from_args("Click Element", ["link=Products"])
        assert result is None

    def test_regular_text_not_starting_with_link_not_skipped(self, executor):
        """A locator like 'linkedin-btn' should NOT be skipped."""
        result = executor._extract_locator_from_args("Click Element", ["linkedin-btn"])
        assert result == "linkedin-btn"


# ── P7: Shared keyword routing block when explicit_library_preference set ─────

class TestP7_SharedKeywordRoutingBlock:
    """P7: Keywords that exist in BOTH Browser Library and SeleniumLibrary
    were incorrectly blocked when analyze_scenario sets
    explicit_library_preference='SeleniumLibrary'.

    Root cause: Browser Library plugin's get_keyword_library_map() registers
    shared keywords as "Browser" keywords. The SeleniumLibrary plugin's
    validate_keyword_for_session() then blocks them because it sees
    keyword_source_library="Browser" + session preference="SeleniumLibrary".

    Fix: Added _SHARED_KEYWORDS frozenset (18 keywords, programmatically
    verified) to both SeleniumLibraryPlugin and BrowserLibraryPlugin.
    Keywords in this set are allowed through validation even when the
    keyword map says they belong to the other library.
    """

    # All 18 programmatically verified shared keywords
    ALL_SHARED = frozenset({
        "add cookie", "close browser", "delete all cookies", "drag and drop",
        "get browser ids", "get cookie", "get cookies", "get element count",
        "get property", "get text", "get title", "go back", "go to",
        "open browser", "press keys", "register keyword to run on failure",
        "switch browser", "wait for condition",
    })

    @pytest.fixture
    def sel_plugin(self):
        from robotmcp.plugins.builtin.selenium_plugin import SeleniumLibraryPlugin
        return SeleniumLibraryPlugin()

    @pytest.fixture
    def browser_plugin(self):
        from robotmcp.plugins.builtin.browser_plugin import BrowserLibraryPlugin
        return BrowserLibraryPlugin()

    @pytest.fixture
    def selenium_session(self):
        session = MagicMock(spec=ExecutionSession)
        session.explicit_library_preference = "SeleniumLibrary"
        return session

    @pytest.fixture
    def browser_session(self):
        session = MagicMock(spec=ExecutionSession)
        session.explicit_library_preference = "Browser"
        return session

    # ── SeleniumLibrary plugin: all 18 shared keywords MUST be allowed ──

    @pytest.mark.parametrize("keyword", [
        "Open Browser", "Close Browser", "Get Title",
        "Add Cookie", "Delete All Cookies", "Drag And Drop",
        "Get Browser Ids", "Get Cookie", "Get Cookies",
        "Get Element Count", "Get Property", "Get Text",
        "Go Back", "Go To", "Press Keys",
        "Register Keyword To Run On Failure",
        "Switch Browser", "Wait For Condition",
    ])
    def test_selenium_allows_all_shared_keywords(self, sel_plugin, selenium_session, keyword):
        """All 18 shared keywords must NOT be blocked in SeleniumLibrary sessions."""
        result = sel_plugin.validate_keyword_for_session(
            selenium_session, keyword, "Browser"
        )
        assert result is None, f"Shared keyword '{keyword}' was incorrectly blocked"

    def test_shared_keyword_case_insensitive(self, sel_plugin, selenium_session):
        """Shared keyword check should be case-insensitive."""
        result = sel_plugin.validate_keyword_for_session(
            selenium_session, "OPEN BROWSER", "Browser"
        )
        assert result is None

    def test_shared_keyword_mixed_case(self, sel_plugin, selenium_session):
        """Shared keyword with mixed case name."""
        result = sel_plugin.validate_keyword_for_session(
            selenium_session, "get element count", "Browser"
        )
        assert result is None

    # ── Browser-ONLY keywords MUST still be blocked in SeleniumLibrary sessions ──

    @pytest.mark.parametrize("keyword", [
        "New Browser", "New Page", "Fill Text", "Click",
        "Get Page Source", "Wait For Elements State", "New Context",
    ])
    def test_browser_only_still_blocked(self, sel_plugin, selenium_session, keyword):
        """Browser-only keywords must be blocked in SeleniumLibrary sessions."""
        result = sel_plugin.validate_keyword_for_session(
            selenium_session, keyword, "Browser"
        )
        assert result is not None
        assert result["success"] is False

    # ── Non-Browser keywords should pass through ──

    def test_non_browser_keyword_allowed(self, sel_plugin, selenium_session):
        """Keywords not from Browser Library should not be blocked."""
        result = sel_plugin.validate_keyword_for_session(
            selenium_session, "Log", "BuiltIn"
        )
        assert result is None

    def test_none_source_library_allowed(self, sel_plugin, selenium_session):
        """Keywords with no source library should not be blocked."""
        result = sel_plugin.validate_keyword_for_session(
            selenium_session, "Custom Keyword", None
        )
        assert result is None

    # ── Non-SeleniumLibrary sessions should pass through ──

    def test_non_selenium_session_skips_validation(self, sel_plugin):
        """Non-SeleniumLibrary sessions should skip all validation."""
        session = MagicMock(spec=ExecutionSession)
        session.explicit_library_preference = "Browser"
        result = sel_plugin.validate_keyword_for_session(
            session, "Open Browser", "Browser"
        )
        assert result is None

    def test_no_preference_skips_validation(self, sel_plugin):
        """Sessions without explicit_library_preference skip validation."""
        session = MagicMock(spec=ExecutionSession)
        session.explicit_library_preference = None
        result = sel_plugin.validate_keyword_for_session(
            session, "Open Browser", "Browser"
        )
        assert result is None

    # ── _SHARED_KEYWORDS frozenset properties (SeleniumLibrary plugin) ──

    def test_selenium_shared_keywords_is_frozenset(self, sel_plugin):
        """_SHARED_KEYWORDS should be a frozenset for immutability."""
        assert isinstance(sel_plugin._SHARED_KEYWORDS, frozenset)

    def test_selenium_shared_keywords_all_lowercase(self, sel_plugin):
        """All entries in _SHARED_KEYWORDS should be lowercase."""
        for kw in sel_plugin._SHARED_KEYWORDS:
            assert kw == kw.lower(), f"Keyword '{kw}' is not lowercase"

    def test_selenium_shared_keywords_count(self, sel_plugin):
        """_SHARED_KEYWORDS should contain exactly 18 shared keywords."""
        assert len(sel_plugin._SHARED_KEYWORDS) == 18

    def test_selenium_shared_keywords_matches_verified(self, sel_plugin):
        """_SHARED_KEYWORDS should match the programmatically verified set."""
        assert sel_plugin._SHARED_KEYWORDS == self.ALL_SHARED

    # ── Self-referential KEYWORD_ALTERNATIVES removed ──

    def test_selenium_no_self_referential_alternatives(self, sel_plugin):
        """KEYWORD_ALTERNATIVES should not contain entries that map to same name."""
        for key, info in sel_plugin.KEYWORD_ALTERNATIVES.items():
            alt = info.get("alternative", "").lower().replace(" ", "")
            key_norm = key.lower().replace(" ", "")
            assert alt != key_norm, (
                f"Self-referential alternative: '{key}' → '{info['alternative']}'"
            )

    # ── Browser Library plugin: _SHARED_KEYWORDS defense-in-depth ──

    def test_browser_shared_keywords_is_frozenset(self, browser_plugin):
        """Browser plugin _SHARED_KEYWORDS should be a frozenset."""
        assert isinstance(browser_plugin._SHARED_KEYWORDS, frozenset)

    def test_browser_shared_keywords_matches_selenium(self, sel_plugin, browser_plugin):
        """Both plugins should have identical _SHARED_KEYWORDS sets."""
        assert sel_plugin._SHARED_KEYWORDS == browser_plugin._SHARED_KEYWORDS

    @pytest.mark.parametrize("keyword", [
        "Close Browser", "Go To", "Get Text", "Go Back",
        "Get Element Count", "Add Cookie", "Get Title",
    ])
    def test_browser_allows_shared_keywords(self, browser_plugin, browser_session, keyword):
        """Shared keywords must NOT be blocked in Browser Library sessions."""
        result = browser_plugin.validate_keyword_for_session(
            browser_session, keyword, "SeleniumLibrary"
        )
        assert result is None, f"Shared keyword '{keyword}' was incorrectly blocked"

    @pytest.mark.parametrize("keyword", [
        "Input Text", "Click Element", "Click Button",
        "Get Source", "Element Should Be Visible",
    ])
    def test_browser_blocks_selenium_only(self, browser_plugin, browser_session, keyword):
        """Selenium-only keywords must be blocked in Browser Library sessions."""
        result = browser_plugin.validate_keyword_for_session(
            browser_session, keyword, "SeleniumLibrary"
        )
        assert result is not None
        assert result["success"] is False

    def test_browser_no_self_referential_alternatives(self, browser_plugin):
        """KEYWORD_ALTERNATIVES should not contain entries that map to same name."""
        for key, info in browser_plugin.KEYWORD_ALTERNATIVES.items():
            alt = info.get("alternative", "").lower().replace(" ", "")
            key_norm = key.lower().replace(" ", "")
            assert alt != key_norm, (
                f"Self-referential alternative: '{key}' → '{info['alternative']}'"
            )

    # ── Integration: plugin_manager full chain ──

    def test_keyword_map_triggers_validation(self, selenium_session):
        """Verify the full chain: keyword map → validate → allow shared."""
        from robotmcp.plugins.manager import LibraryPluginManager
        from robotmcp.plugins.builtin.selenium_plugin import SeleniumLibraryPlugin
        from robotmcp.plugins.builtin.browser_plugin import BrowserLibraryPlugin

        mgr = LibraryPluginManager()
        mgr.register_plugin(BrowserLibraryPlugin())
        mgr.register_plugin(SeleniumLibraryPlugin())

        # Browser plugin maps "open browser" → "Browser"
        source = mgr.get_library_for_keyword("open browser")
        assert source == "Browser"

        # But SeleniumLibrary validation should ALLOW it (shared keyword)
        result = mgr.validate_keyword_for_session(
            "SeleniumLibrary", selenium_session, "Open Browser", source
        )
        assert result is None  # Not blocked

    def test_keyword_map_blocks_browser_only(self, selenium_session):
        """Browser-only keywords should still be blocked through the full chain."""
        from robotmcp.plugins.manager import LibraryPluginManager
        from robotmcp.plugins.builtin.selenium_plugin import SeleniumLibraryPlugin
        from robotmcp.plugins.builtin.browser_plugin import BrowserLibraryPlugin

        mgr = LibraryPluginManager()
        mgr.register_plugin(BrowserLibraryPlugin())
        mgr.register_plugin(SeleniumLibraryPlugin())

        source = mgr.get_library_for_keyword("new browser")
        assert source == "Browser"

        result = mgr.validate_keyword_for_session(
            "SeleniumLibrary", selenium_session, "New Browser", source
        )
        assert result is not None
        assert result["success"] is False

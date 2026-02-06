"""Keyword execution service."""

import asyncio
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from robotmcp.components.execution.rf_native_context_manager import (
    get_rf_native_context_manager,
)
from robotmcp.core.event_bus import FrontendEvent, event_bus
from robotmcp.components.variables.variable_resolver import VariableResolver
from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery
from robotmcp.models.config_models import ExecutionConfig
from robotmcp.models.execution_models import ExecutionStep
from robotmcp.models.session_models import ExecutionSession
from robotmcp.utils.argument_processor import ArgumentProcessor
from robotmcp.utils.response_serializer import MCPResponseSerializer
from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter
from robotmcp.plugins import get_library_plugin_manager

# Import timeout domain components for proper timeout handling
from robotmcp.domains.timeout import ActionType, TimeoutPolicy, DefaultTimeouts
from robotmcp.domains.timeout.keyword_classifier import classify_keyword
from robotmcp.container import get_container

logger = logging.getLogger(__name__)

# Import Robot Framework components
try:
    from robot.libraries.BuiltIn import BuiltIn

    ROBOT_AVAILABLE = True
except ImportError:
    BuiltIn = None
    ROBOT_AVAILABLE = False


class KeywordExecutor:
    """Handles keyword execution with proper library routing and error handling."""

    # Keywords that require element pre-validation before execution
    # These keywords interact with elements and benefit from fast visibility/state checks
    ELEMENT_INTERACTION_KEYWORDS: Set[str] = {
        # Click operations
        "click",
        "click element",
        "double click",
        "double click element",
        "right click",
        "right click element",
        # Text input operations
        "fill text",
        "fill secret",
        "type text",
        "type secret",
        "input text",
        "input password",
        "clear text",
        "clear element value",
        # Checkbox/Radio operations
        "check checkbox",
        "uncheck checkbox",
        "select checkbox",
        "unselect checkbox",
        # Select/Dropdown operations
        "select options",
        "select from list",
        "select from list by value",
        "select from list by label",
        "select from list by index",
        "deselect from list",
        # Keyboard operations
        "press keys",
        "press key",
        "keyboard key",
        # Focus/Hover operations
        "focus",
        "hover",
        "mouse over",
        "scroll to element",
        "scroll element into view",
    }

    # Required element states for different action types
    REQUIRED_STATES_FOR_ACTION: Dict[str, Set[str]] = {
        "click": {"visible", "enabled"},
        "fill": {"visible", "enabled", "editable"},
        "input": {"visible", "enabled", "editable"},
        "type": {"visible", "enabled", "editable"},
        "check": {"visible", "enabled"},
        "uncheck": {"visible", "enabled"},
        "select": {"visible", "enabled"},
        "press": {"visible", "enabled"},
        "focus": {"visible"},
        "hover": {"visible"},
        "scroll": {"attached"},
        "clear": {"visible", "enabled", "editable"},
    }

    def __init__(
        self, config: Optional[ExecutionConfig] = None, override_registry=None
    ):
        self.config = config or ExecutionConfig()
        self.keyword_discovery = get_keyword_discovery()
        self.argument_processor = ArgumentProcessor()
        self.rf_converter = RobotFrameworkNativeConverter()
        self.override_registry = override_registry
        self.variable_resolver = VariableResolver()
        self.response_serializer = MCPResponseSerializer()
        # Legacy RobotContextManager is deprecated; use RF native context only
        self.rf_native_context = get_rf_native_context_manager()
        self.plugin_manager = get_library_plugin_manager()
        # Feature flag: route RequestsLibrary session operations via RF runner
        # Default ON; set ROBOTMCP_RF_RUNNER_REQUESTS=0 to disable
        self.rf_runner_requests = os.getenv("ROBOTMCP_RF_RUNNER_REQUESTS", "1") in (
            "1",
            "true",
            "True",
        )
        # Default to context-only execution unless explicitly disabled
        self.context_only = os.getenv("ROBOTMCP_RF_CONTEXT_ONLY", "1") in (
            "1",
            "true",
            "True",
        )
        # Feature flag: enable/disable pre-validation (default ON)
        self.pre_validation_enabled = os.getenv("ROBOTMCP_PRE_VALIDATION", "1") in (
            "1",
            "true",
            "True",
        )

    def _requires_pre_validation(self, keyword: str) -> bool:
        """Check if a keyword requires element pre-validation."""
        keyword_lower = keyword.lower().strip()
        return keyword_lower in self.ELEMENT_INTERACTION_KEYWORDS

    def _get_action_type_from_keyword_for_states(self, keyword: str) -> str:
        """Extract the action type from a keyword name for state requirements."""
        keyword_lower = keyword.lower()
        if "click" in keyword_lower:
            return "click"
        elif "fill" in keyword_lower or "input" in keyword_lower or "type" in keyword_lower:
            return "fill"
        elif "check" in keyword_lower and "uncheck" not in keyword_lower:
            return "check"
        elif "uncheck" in keyword_lower:
            return "uncheck"
        elif "select" in keyword_lower:
            return "select"
        elif "press" in keyword_lower or "key" in keyword_lower:
            return "press"
        elif "focus" in keyword_lower:
            return "focus"
        elif "hover" in keyword_lower or "mouse" in keyword_lower:
            return "hover"
        elif "scroll" in keyword_lower:
            return "scroll"
        elif "clear" in keyword_lower:
            return "clear"
        return "click"

    def _extract_locator_from_args(self, keyword: str, arguments: List[Any]) -> Optional[str]:
        """Extract the element locator from keyword arguments."""
        if not arguments:
            return None
        first_arg = arguments[0]
        return first_arg if isinstance(first_arg, str) else None

    async def _pre_validate_element(
        self,
        locator: str,
        session: "ExecutionSession",
        keyword: str,
        timeout_ms: Optional[int] = None,
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """Quick pre-validation check if element is actionable before attempting action.

        This performs a fast check (default 500ms timeout) to verify element state
        before the full keyword execution, enabling early failure detection.

        Returns:
            Tuple of (is_valid, error_message, details_dict)
        """
        if timeout_ms is None:
            timeout_ms = self.config.PRE_VALIDATION_TIMEOUT

        start_time = time.time()
        action_type = self._get_action_type_from_keyword_for_states(keyword)
        required_states = self.REQUIRED_STATES_FOR_ACTION.get(action_type, {"visible"})

        details: Dict[str, Any] = {
            "locator": locator,
            "keyword": keyword,
            "action_type": action_type,
            "required_states": list(required_states),
            "timeout_ms": timeout_ms,
        }

        try:
            # Ensure ctx.test is set for BuiltIn.run_keyword() support.
            # All pre-validation paths call BuiltIn.run_keyword() which internally
            # does kw.run(result, ctx).  When ctx.test is None, RF falls back to
            # ctx.suite.setup — a running-model Keyword without .body — causing
            # AttributeError: 'Keyword' object has no attribute 'body'.
            try:
                from robot.running.context import EXECUTION_CONTEXTS as _EC
                _ctx = _EC.current
                if _ctx and not _ctx.test:
                    from robot.result.model import TestCase as _ResTest
                    _ctx.test = _ResTest(name="MCP_PreValidation")
            except Exception:
                pass

            active_library = session.browser_state.active_library
            if active_library == "browser":
                result = await self._pre_validate_browser_element(locator, required_states, timeout_ms)
            elif active_library == "selenium":
                result = await self._pre_validate_selenium_element(locator, required_states, timeout_ms)
            elif active_library == "appium":
                result = await self._pre_validate_appium_element(locator, required_states, timeout_ms)
            else:
                logger.debug(f"Pre-validation skipped: no active browser for {keyword}")
                return True, None, {"skipped": True, "reason": "no_active_browser"}

            elapsed_ms = (time.time() - start_time) * 1000
            details["elapsed_ms"] = round(elapsed_ms, 2)

            if result["valid"]:
                details["current_states"] = result.get("states", [])
                logger.debug(f"Pre-validation passed for '{locator}' in {elapsed_ms:.1f}ms")
                return True, None, details
            else:
                details["current_states"] = result.get("states", [])
                details["missing_states"] = result.get("missing", [])
                error_msg = result.get("error", f"Element not actionable: {locator}")
                logger.warning(f"Pre-validation failed for '{locator}' in {elapsed_ms:.1f}ms: {error_msg}")
                return False, error_msg, details

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            details["elapsed_ms"] = round(elapsed_ms, 2)
            details["exception"] = str(e)
            error_msg = f"Element not found or inaccessible: {locator}"
            logger.warning(f"Pre-validation exception for '{locator}': {e}")
            return False, error_msg, details

    async def _pre_validate_browser_element(
        self, locator: str, required_states: Set[str], timeout_ms: int
    ) -> Dict[str, Any]:
        """Pre-validate element using Browser Library's Get Element States."""
        try:
            timeout_str = f"{timeout_ms}ms"
            result, error_info = await asyncio.to_thread(self._run_browser_get_states, locator, timeout_str)

            if result is None:
                # Use the actual error message if available, otherwise generic message
                error_msg = error_info if error_info else f"Element not found: {locator}"
                return {"valid": False, "states": [], "missing": list(required_states),
                        "error": error_msg}

            current_states = set()
            if hasattr(result, "__iter__"):
                for state in result:
                    state_str = str(state).lower()
                    if "." in state_str:
                        state_str = state_str.split(".")[-1]
                    current_states.add(state_str)

            missing = required_states - current_states
            if missing:
                return {"valid": False, "states": list(current_states), "missing": list(missing),
                        "error": f"Element missing required states: {', '.join(sorted(missing))}"}

            return {"valid": True, "states": list(current_states), "missing": [], "error": None}

        except Exception as e:
            error_str = str(e).lower()
            if "timeout" in error_str or "timed out" in error_str:
                return {"valid": False, "states": [], "missing": list(required_states),
                        "error": f"Element not found within {timeout_ms}ms: {locator}"}
            elif "not found" in error_str or "no element" in error_str:
                return {"valid": False, "states": [], "missing": list(required_states),
                        "error": f"Element not found: {locator}"}
            return {"valid": False, "states": [], "missing": list(required_states),
                    "error": f"Pre-validation error: {str(e)}"}

    def _run_browser_get_states(self, locator: str, timeout: str) -> tuple[Optional[Any], Optional[str]]:
        """Run Browser Library's Get Element States keyword.

        Returns:
            tuple: (result, error_info) where result is the states or None,
                   and error_info contains the actual error message if failed.

        Note: Get Element States doesn't accept timeout directly.
        We set browser timeout temporarily before the call.

        IMPORTANT: The browser timeout MUST be restored after pre-validation,
        otherwise subsequent keyword executions (like Click) will use the
        pre-validation timeout (500ms) instead of the intended action timeout.
        """
        builtin = BuiltIn()

        # Set timeout temporarily (Browser Library uses global timeout for element operations)
        # Note: Set Browser Timeout returns the previous timeout value, so we use that
        original_timeout = None
        timeout_was_set = False
        try:
            # Set new timeout and capture the previous value (returned by the keyword)
            original_timeout = builtin.run_keyword("Browser.Set Browser Timeout", timeout)
            timeout_was_set = True
        except Exception as e:
            # If we can't set timeout, log but continue anyway
            logger.debug(f"Failed to set browser timeout to {timeout}: {e}")

        def try_get_states(loc: str) -> tuple[Optional[Any], Optional[str]]:
            """Try to get element states with the given locator."""
            try:
                # Get Element States doesn't take timeout - uses global browser timeout
                result = builtin.run_keyword("Browser.Get Element States", loc)
                return result, None
            except Exception as e1:
                try:
                    result = builtin.run_keyword("Get Element States", loc)
                    return result, None
                except Exception as e2:
                    return None, str(e2)

        try:
            # First attempt with original locator
            result, error = try_get_states(locator)
            if result is not None:
                return result, None

            # Check if error is due to strict mode violation (multiple elements)
            if error and ("strict mode" in error.lower() or "resolved to" in error.lower() and "elements" in error.lower()):
                logger.debug(f"Strict mode violation for '{locator}': {error}. Trying with visible filter.")

                # Try with >> visible=true filter to get only visible element
                visible_locator = f"{locator} >> visible=true"
                result, visible_error = try_get_states(visible_locator)
                if result is not None:
                    return result, None

                # If visible filter also failed, try with nth=0 as last resort
                nth_locator = f"{locator} >> nth=0"
                result, nth_error = try_get_states(nth_locator)
                if result is not None:
                    logger.debug(f"Got states using nth=0 selector for '{locator}'")
                    return result, None

                # Return informative error about multiple elements
                return None, f"Multiple elements found for '{locator}'. Tried visible filter and nth=0 but both failed. Original error: {error}"

            # Return the original error for other failure cases
            return None, error

        finally:
            # CRITICAL: Restore original timeout if we changed it
            # Failure to restore leaves browser at 500ms, causing subsequent
            # actions (like Click) to fail with timeout even if they succeed
            if timeout_was_set and original_timeout is not None:
                restore_success = False
                for attempt in range(3):  # Retry up to 3 times
                    try:
                        builtin.run_keyword("Browser.Set Browser Timeout", original_timeout)
                        restore_success = True
                        break
                    except Exception as e:
                        if attempt < 2:
                            logger.debug(f"Retry {attempt + 1}/3: Failed to restore browser timeout to {original_timeout}: {e}")
                        else:
                            logger.warning(
                                f"CRITICAL: Failed to restore browser timeout to {original_timeout} after 3 attempts. "
                                f"Browser timeout may be stuck at {timeout}. Subsequent keyword executions may fail. "
                                f"Error: {e}"
                            )
                if restore_success:
                    logger.debug(f"Browser timeout restored to {original_timeout}")

    async def _pre_validate_selenium_element(
        self, locator: str, required_states: Set[str], timeout_ms: int
    ) -> Dict[str, Any]:
        """Pre-validate element using SeleniumLibrary checks."""
        try:
            return await asyncio.to_thread(
                self._run_selenium_state_check, locator, required_states, timeout_ms
            )
        except Exception as e:
            return {"valid": False, "states": [], "missing": list(required_states),
                    "error": f"Pre-validation error: {str(e)}"}

    def _run_selenium_state_check(
        self, locator: str, required_states: Set[str], timeout_ms: int
    ) -> Dict[str, Any]:
        """Run SeleniumLibrary state check using JavaScript.

        Handles multiple elements by finding the first visible one.

        Note: We temporarily set implicit wait for element lookup, then restore
        the original value to avoid affecting subsequent keyword executions.
        """
        builtin = BuiltIn()
        original_implicit_wait = None
        implicit_wait_was_set = False

        try:
            # Save and set implicit wait temporarily
            try:
                # SeleniumLibrary doesn't have a "Get Selenium Implicit Wait", so we
                # use a reasonable default for restoration (10 seconds is Selenium default)
                original_implicit_wait = "10s"
                builtin.run_keyword("SeleniumLibrary.Set Selenium Implicit Wait", f"{timeout_ms / 1000}s")
                implicit_wait_was_set = True
            except Exception as e:
                logger.debug(f"Failed to set Selenium implicit wait: {e}")

            # Try to get all matching elements to handle duplicates
            elements = []
            try:
                elements = builtin.run_keyword("SeleniumLibrary.Get WebElements", locator)
            except Exception:
                pass

            if not elements:
                # Fallback to single element lookup
                try:
                    element = builtin.run_keyword("SeleniumLibrary.Get WebElement", locator)
                    elements = [element] if element else []
                except Exception:
                    return {"valid": False, "states": [], "missing": list(required_states),
                            "error": f"Element not found: {locator}"}

            if not elements:
                return {"valid": False, "states": [], "missing": list(required_states),
                        "error": f"Element not found: {locator}"}

            # If multiple elements, find the first visible one
            element = None
            element_count = len(elements) if hasattr(elements, '__len__') else 1

            if element_count > 1:
                logger.debug(f"Found {element_count} elements for '{locator}', checking for visible one")
                for idx, el in enumerate(elements):
                    try:
                        is_visible = builtin.run_keyword("SeleniumLibrary.Execute Javascript",
                            "var el = arguments[0]; var style = window.getComputedStyle(el); "
                            "var rect = el.getBoundingClientRect(); "
                            "return style.display !== 'none' && style.visibility !== 'hidden' && "
                            "rect.width > 0 && rect.height > 0;", "ARGUMENTS", el)
                        if is_visible:
                            element = el
                            logger.debug(f"Using visible element at index {idx} for '{locator}'")
                            break
                    except Exception:
                        continue

                if element is None:
                    # No visible element found, use first one and let it fail with proper message
                    element = elements[0]
                    logger.debug(f"No visible element found among {element_count} elements for '{locator}'")
            else:
                element = elements[0] if elements else None

            js_check = """
            var el = arguments[0];
            var states = [];
            if (document.body.contains(el)) states.push('attached');
            var style = window.getComputedStyle(el);
            var rect = el.getBoundingClientRect();
            if (style.display !== 'none' && style.visibility !== 'hidden' &&
                rect.width > 0 && rect.height > 0) states.push('visible');
            if (!el.disabled) states.push('enabled');
            if ((el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') && !el.readOnly)
                states.push('editable');
            if (el.checked !== undefined) states.push(el.checked ? 'checked' : 'unchecked');
            return states;
            """

            try:
                states_list = builtin.run_keyword("SeleniumLibrary.Execute Javascript", js_check, "ARGUMENTS", element)
                current_states = set(states_list) if states_list else set()
            except Exception:
                current_states = {"attached"}

            missing = required_states - current_states
            if missing:
                return {"valid": False, "states": list(current_states), "missing": list(missing),
                        "error": f"Element missing required states: {', '.join(sorted(missing))}"}

            return {"valid": True, "states": list(current_states), "missing": [], "error": None}

        except Exception as e:
            return {"valid": False, "states": [], "missing": list(required_states),
                    "error": f"Pre-validation error: {str(e)}"}

        finally:
            # Restore implicit wait if we changed it
            if implicit_wait_was_set and original_implicit_wait is not None:
                try:
                    builtin.run_keyword("SeleniumLibrary.Set Selenium Implicit Wait", original_implicit_wait)
                except Exception as e:
                    logger.debug(f"Failed to restore Selenium implicit wait to {original_implicit_wait}: {e}")

    async def _pre_validate_appium_element(
        self, locator: str, required_states: Set[str], timeout_ms: int
    ) -> Dict[str, Any]:
        """Pre-validate element using AppiumLibrary checks.

        Similar to SeleniumLibrary but uses AppiumLibrary keywords.
        Handles multiple elements by finding the first visible one.
        """
        try:
            return await asyncio.to_thread(
                self._run_appium_state_check, locator, required_states, timeout_ms
            )
        except Exception as e:
            return {"valid": False, "states": [], "missing": list(required_states),
                    "error": f"Pre-validation error: {str(e)}"}

    def _run_appium_state_check(
        self, locator: str, required_states: Set[str], timeout_ms: int
    ) -> Dict[str, Any]:
        """Run AppiumLibrary state check.

        Handles multiple elements by finding the first visible/enabled one.
        """
        try:
            builtin = BuiltIn()

            # Try to get all matching elements to handle duplicates
            elements = []
            try:
                elements = builtin.run_keyword("AppiumLibrary.Get Webelements", locator)
            except Exception:
                pass

            if not elements:
                # Fallback to single element lookup
                try:
                    element = builtin.run_keyword("AppiumLibrary.Get Webelement", locator)
                    elements = [element] if element else []
                except Exception:
                    return {"valid": False, "states": [], "missing": list(required_states),
                            "error": f"Element not found: {locator}"}

            if not elements:
                return {"valid": False, "states": [], "missing": list(required_states),
                        "error": f"Element not found: {locator}"}

            # If multiple elements, find the first visible one
            element = None
            element_count = len(elements) if hasattr(elements, '__len__') else 1

            if element_count > 1:
                logger.debug(f"Found {element_count} Appium elements for '{locator}', checking for visible one")
                for idx, el in enumerate(elements):
                    try:
                        # Check if element is displayed
                        is_displayed = el.is_displayed() if hasattr(el, 'is_displayed') else True
                        if is_displayed:
                            element = el
                            logger.debug(f"Using visible Appium element at index {idx} for '{locator}'")
                            break
                    except Exception:
                        continue

                if element is None:
                    element = elements[0]
                    logger.debug(f"No visible element found among {element_count} Appium elements for '{locator}'")
            else:
                element = elements[0] if elements else None

            # Check element states
            current_states = set()
            try:
                if element is not None:
                    current_states.add("attached")
                    if hasattr(element, 'is_displayed') and element.is_displayed():
                        current_states.add("visible")
                    if hasattr(element, 'is_enabled') and element.is_enabled():
                        current_states.add("enabled")
                    # For mobile, most editable elements are enabled input fields
                    tag_name = element.tag_name.lower() if hasattr(element, 'tag_name') else ""
                    if tag_name in ("edittext", "textfield", "input", "textarea"):
                        if "enabled" in current_states:
                            current_states.add("editable")
            except Exception:
                current_states = {"attached"}

            missing = required_states - current_states
            if missing:
                return {"valid": False, "states": list(current_states), "missing": list(missing),
                        "error": f"Element missing required states: {', '.join(sorted(missing))}"}

            return {"valid": True, "states": list(current_states), "missing": [], "error": None}

        except Exception as e:
            return {"valid": False, "states": [], "missing": list(required_states),
                    "error": f"Pre-validation error: {str(e)}"}

    async def execute_keyword(
        self,
        session: ExecutionSession,
        keyword: str,
        arguments: List[str],
        browser_library_manager: Any,  # BrowserLibraryManager
        detail_level: str = "minimal",
        library_prefix: str = None,
        assign_to: Union[str, List[str]] = None,
        use_context: bool = False,
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single Robot Framework keyword step with optional library prefix.

        Args:
            session: ExecutionSession to run in
            keyword: Robot Framework keyword name (supports Library.Keyword syntax)
            arguments: List of arguments for the keyword
            browser_library_manager: BrowserLibraryManager instance
            detail_level: Level of detail in response ('minimal', 'standard', 'full')
            library_prefix: Optional explicit library name to override session search order
            assign_to: Optional variable assignment
            use_context: If True, execute within full RF context
            timeout_ms: Optional timeout in milliseconds. If not provided, uses smart
                       defaults based on keyword type:
                       - Element actions (Click, Fill): 5000ms
                       - Navigation (Go To, New Page): 60000ms
                       - Read operations (Get Text): 2000ms
                       - API calls (GET, POST): 30000ms
                       Set to 0 or negative to disable timeout.

        Returns:
            Execution result with status, output, and state
        """

        try:
            # PHASE 1.2: Pre-execution Library Registration
            # Ensure required library is registered before keyword execution
            self._ensure_library_registration(keyword, session)

            # Create execution step
            step = ExecutionStep(
                step_id=str(uuid.uuid4()),
                keyword=keyword,
                arguments=arguments,
                start_time=datetime.now(),
            )
            event_bus.publish_sync(
                FrontendEvent(
                    event_type="step_started",
                    session_id=session.session_id,
                    step_id=step.step_id,
                    payload={"keyword": keyword, "arguments": arguments},
                )
            )

            # Update session activity
            session.update_activity()

            # Mark step as running
            step.status = "running"

            # NOTE: Library keyword validation is handled by plugin overrides in _execute_keyword_internal
            # The BrowserLibraryPlugin._override_open_browser handles "Open Browser" rejection with detailed guidance

            # Check if we should use context mode
            # Enable context mode for keywords that require RF execution context
            context_required_keywords = [
                "evaluate",
                "set test variable",
                "set suite variable",
                "set global variable",
                "create dictionary",
                "get variable value",
                "variable should exist",
                "call method",
                "run keyword if",
                "run keyword unless",
                "run keywords",
                # NOTE: Input Password removed - works fine in normal execution with name normalization
            ]

            # RequestsLibrary: route session-scoped operations through RF native context
            requests_library_context_keywords = [
                "create session",
                "delete session",
                "get on session",
                "post on session",
                "put on session",
                "delete on session",
                "patch on session",
                "head on session",
                "options on session",
            ]

            # Browser Library keywords should NOT use RF native context due to import issues
            # They work perfectly in regular execution mode
            browser_library_keywords = [
                "open browser",
                "close browser",
                "new browser",
                "new context",
                "new page",
                "go to",
                "click",
                "fill text",
                "take screenshot",
                "get text",
                "wait for elements state",
                "get title",
                "get url",
                "input text",
                "click element",
                "wait until element is visible",
            ]

            # KEYWORD NAME NORMALIZATION AND OVERRIDES - General solution for keyword name variations
            # NOTE: Input Password override is now handled in _execute_selenium_keyword method
            # to ensure proper execution while preserving original keyword for step recording
            keyword_name_mappings = {
                # Add other common mappings as needed (Input Password removed - handled in _execute_selenium_keyword)
                # "click element": "click_element",  # Usually handled by dynamic resolution
            }

            # Apply normalization if mapping exists (Input Password override removed from here)
            original_keyword = keyword
            if keyword in keyword_name_mappings:
                logger.info(
                    f"Keyword name normalized: '{original_keyword}' -> '{keyword_name_mappings[keyword]}'"
                )
                keyword = keyword_name_mappings[keyword]

            keyword_requires_context = keyword.lower() in context_required_keywords
            is_requests_keyword = keyword.lower() in requests_library_context_keywords
            is_browser_keyword = keyword.lower() in browser_library_keywords

            # PRE-VALIDATION: Fast check for element actionability before execution
            # This detects "element not visible/enabled" in ~500ms instead of waiting 10s
            if self.pre_validation_enabled and self._requires_pre_validation(keyword):
                locator = self._extract_locator_from_args(keyword, arguments)
                if locator:
                    is_valid, error_msg, pre_validation_details = await self._pre_validate_element(
                        locator, session, keyword
                    )
                    if not is_valid:
                        # Pre-validation failed - return early with helpful error
                        step.end_time = datetime.now()
                        step.mark_failure(error_msg)
                        hints: List[Dict[str, Any]] = [
                            {
                                "type": "pre_validation_failure",
                                "message": "Element is not in an actionable state",
                                "suggestion": "Ensure the element is visible and enabled before interaction",
                                "details": pre_validation_details,
                            }
                        ]
                        # Add specific hints based on missing states
                        if pre_validation_details and "missing_states" in pre_validation_details:
                            missing = pre_validation_details.get("missing_states", [])
                            if "visible" in missing:
                                hints.append({
                                    "type": "visibility_hint",
                                    "message": "Element is not visible",
                                    "suggestion": "Use 'Wait For Elements State' with 'visible' before clicking",
                                    "example": f"Wait For Elements State    {locator}    visible    timeout=5s",
                                })
                            if "enabled" in missing:
                                hints.append({
                                    "type": "enabled_hint",
                                    "message": "Element is not enabled",
                                    "suggestion": "Check if the element is disabled or if a previous action is required",
                                })

                        event_bus.publish_sync(
                            FrontendEvent(
                                event_type="step_failed",
                                session_id=session.session_id,
                                step_id=step.step_id,
                                payload={
                                    "status": "fail",
                                    "keyword": keyword,
                                    "arguments": arguments,
                                    "error": error_msg,
                                    "pre_validation_failed": True,
                                },
                            )
                        )
                        return {
                            "success": False,
                            "error": f"Pre-validation failed: {error_msg}",
                            "hint": "Element is not in an actionable state. Previous steps may not have completed.",
                            "pre_validation_failed": True,
                            "pre_validation_details": pre_validation_details,
                            "step_id": step.step_id,
                            "keyword": keyword,
                            "arguments": arguments,
                            "status": "fail",
                            "execution_time": step.execution_time,
                            "session_variables": dict(session.variables),
                            "hints": hints,
                        }
                    else:
                        logger.debug(
                            f"Pre-validation passed for '{keyword}' with locator '{locator}'"
                        )

            # Context-only execution: route all keywords through RF native context
            if True:
                # Determine effective timeout:
                # 1. User-provided timeout_ms takes highest precedence
                # 2. If not provided (None), use TimeoutPolicy based on keyword classification
                # 3. If timeout_ms <= 0, disable timeout entirely
                action_type = classify_keyword(keyword)
                if timeout_ms is not None:
                    effective_timeout_ms = timeout_ms if timeout_ms > 0 else None
                    timeout_source = "user-specified" if timeout_ms > 0 else "disabled"
                else:
                    container = get_container()
                    timeout_policy = container.get_timeout_policy(session.session_id)
                    effective_timeout_ms = timeout_policy.get_timeout_for(action_type).value
                    timeout_source = f"policy ({action_type.value})"

                logger.info(
                    f"Executing keyword in RF native context mode: {keyword} with args: {arguments}, "
                    f"timeout: {effective_timeout_ms}ms ({timeout_source})"
                )

                # Use RF native context mode for keywords that require it
                result = await self._execute_keyword_with_context(
                    session, keyword, arguments, assign_to, browser_library_manager,
                    timeout_ms=effective_timeout_ms
                )
                resolved_arguments = (
                    arguments  # For logging - RF handles variable resolution
                )
            else:
                # Unreachable in context-only mode
                result = {"success": False, "error": "Non-context path disabled"}

            # Update step status
            step.end_time = datetime.now()
            step.result = result.get("output")

            if result["success"]:
                step_result_value = result.get("result")
                if step_result_value is None and "output" in result:
                    step_result_value = result.get("output")
                step.mark_success(step_result_value)
                # Only append successful steps to the session for suite generation
                session.add_step(step)
                logger.debug(f"Added successful step to session: {keyword}")
            else:
                step.mark_failure(result.get("error"))
                logger.debug(
                    f"Failed step not added to session: {keyword} - {result.get('error')}"
                )

            # Update session variables if any were set
            if "variables" in result:
                session.variables.update(result["variables"])
                try:
                    step.variables.update(result["variables"])
                except Exception:
                    pass

            # Validate assignment compatibility
            if assign_to:
                self._validate_assignment_compatibility(keyword, assign_to)

            # Process variable assignment if assign_to is specified
            if assign_to and result.get("success"):
                assignment_vars = self._process_variable_assignment(
                    assign_to, result.get("result"), keyword, result.get("output")
                )
                if assignment_vars:
                    # DUAL STORAGE IMPLEMENTATION:
                    # 1. Store ORIGINAL objects in session variables for RF execution context
                    session.variables.update(assignment_vars)

                    # NEW: Store assignment info in ExecutionStep for test suite generation
                    step.assigned_variables = list(assignment_vars.keys())
                    step.assignment_type = (
                        "multiple" if isinstance(assign_to, list) else "single"
                    )

                    # DEBUG: Verify what we actually stored in session variables
                    for var_name, var_value in assignment_vars.items():
                        logger.info(
                            f"STORED IN SESSION: {var_name} = {type(var_value).__name__}"
                        )
                        logger.debug(
                            f"Session storage detail: {var_name} -> {str(var_value)[:100]}"
                        )
                        # Verify what's actually in session.variables after update
                        actual_stored = session.variables.get(var_name)
                        logger.info(
                            f"SESSION VERIFICATION: {var_name} stored as {type(actual_stored).__name__}"
                        )

                    # 2. Store raw objects for RF Variables system (needed for ${response.json()})
                    result["assigned_variables_raw"] = assignment_vars

                    # 3. Add serialized assignment info to result for MCP response compatibility
                    # This prevents serialization errors with complex objects
                    serialized_assigned_vars = (
                        self.response_serializer.serialize_assigned_variables(
                            assignment_vars
                        )
                    )
                    result["assigned_variables"] = serialized_assigned_vars
                    try:
                        step.variables.update(serialized_assigned_vars)
                    except Exception:
                        pass

                    # Log assignment for debugging
                    for var_name, var_value in assignment_vars.items():
                        logger.info(
                            f"Assigned variable {var_name} = {type(var_value).__name__} (serialized for response)"
                        )
                        logger.debug(
                            f"Assignment detail: {var_name} -> {str(var_value)[:200]}"
                        )

            # Build response based on detail level
            response = await self._build_response_by_detail_level(
                detail_level,
                result,
                step,
                keyword,
                arguments,
                session,
                resolved_arguments,
            )

            def _serialize_event_value(value: Any) -> Any:
                if isinstance(value, (str, int, float, bool)) or value is None:
                    return value
                if isinstance(value, (list, tuple)):
                    return [_serialize_event_value(item) for item in value]
                if isinstance(value, dict):
                    return {str(k): _serialize_event_value(v) for k, v in value.items()}
                return str(value)

            event_payload = {
                "status": step.status,
                "keyword": keyword,
                "arguments": arguments,
            }

            if result["success"]:
                event_payload["result"] = _serialize_event_value(step.result)
                if step.assigned_variables:
                    event_payload["assigned_variables"] = list(step.assigned_variables)
                    event_payload["assignment_type"] = step.assignment_type
                    assigned_values = {}
                    for var_name in step.assigned_variables:
                        value = step.variables.get(var_name)
                        if value is None:
                            value = session.variables.get(var_name)
                        assigned_values[var_name] = _serialize_event_value(value)
                    event_payload["assigned_values"] = assigned_values
            else:
                event_payload["error"] = result.get("error")

            event_bus.publish_sync(
                FrontendEvent(
                    event_type="step_completed" if result["success"] else "step_failed",
                    session_id=session.session_id,
                    step_id=step.step_id,
                    payload=event_payload,
                )
            )
            return response

        except Exception as e:
            logger.error(f"Error executing step {keyword}: {e}")

            # Create a failed step for error reporting
            step = ExecutionStep(
                step_id=str(uuid.uuid4()),
                keyword=keyword,
                arguments=arguments,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
            step.mark_failure(str(e))

            hints: List[Dict[str, Any]] = []
            library_name = self._get_library_for_keyword(keyword)
            plugin_hints = self.plugin_manager.generate_failure_hints(
                library_name,
                session,
                keyword,
                list(arguments or []),
                str(e),
            )
            if plugin_hints:
                hints.extend(plugin_hints)
            try:
                from robotmcp.utils.hints import HintContext, generate_hints

                if not hints:
                    hctx = HintContext(
                        session_id=session.session_id,
                        keyword=keyword,
                        arguments=list(arguments or []),
                        error_text=str(e),
                        session_search_order=getattr(session, "search_order", None),
                    )
                    hints = generate_hints(hctx)
            except Exception:
                if not hints:
                    hints = []

            return {
                "success": False,
                "error": str(e),
                "step_id": step.step_id,
                "keyword": keyword,
                "arguments": arguments,
                "status": "fail",
                "execution_time": step.execution_time,
                "session_variables": dict(session.variables),
                "hints": hints,
            }

    def _process_variable_assignment(
        self,
        assign_to: Union[str, List[str]],
        result_value: Any,
        keyword: str,
        output: str,
    ) -> Dict[str, Any]:
        """Process variable assignment from keyword execution result.

        Args:
            assign_to: Variable name(s) to assign to
            result_value: The actual return value from the keyword
            keyword: The keyword name (for logging)
            output: The output string representation

        Returns:
            Dictionary of variables to assign to session
        """
        if not assign_to:
            return {}

        # DEBUG: Log what we receive for tracing serialization issue
        logger.debug(
            f"VARIABLE_ASSIGNMENT_DEBUG: {keyword} result_value type: {type(result_value)}"
        )
        logger.debug(
            f"VARIABLE_ASSIGNMENT_DEBUG: {keyword} result_value: {str(result_value)[:200]}"
        )

        # Check if result_value is already serialized (RequestsLibrary Response issue)
        if (
            isinstance(result_value, dict)
            and result_value.get("_type") == "requests_response"
        ):
            logger.warning(
                f"SERIALIZATION_WARNING: {keyword} result_value is already serialized Response object!"
            )

        # If result_value is None but output exists, try to use output
        # This handles cases where the result is in output but not result field
        value_to_assign = result_value
        if value_to_assign is None and output:
            try:
                # Try to parse output as the actual value
                import ast

                # Handle simple cases like numbers, strings, lists
                if output.isdigit():
                    value_to_assign = int(output)
                elif output.replace(".", "").isdigit():
                    value_to_assign = float(output)
                elif output.startswith("[") and output.endswith("]"):
                    value_to_assign = ast.literal_eval(output)
                else:
                    value_to_assign = output
            except:
                value_to_assign = output

        variables = {}

        try:
            if isinstance(assign_to, str):
                # Single assignment
                var_name = self._normalize_variable_name(assign_to)
                variables[var_name] = value_to_assign
                logger.info(f"Assigned {var_name} = {value_to_assign}")

            elif isinstance(assign_to, list):
                # Multi-assignment
                if isinstance(value_to_assign, (list, tuple)):
                    for i, var_name in enumerate(assign_to):
                        normalized_name = self._normalize_variable_name(var_name)
                        if i < len(value_to_assign):
                            variables[normalized_name] = value_to_assign[i]
                        else:
                            variables[normalized_name] = None
                        logger.info(
                            f"Assigned {normalized_name} = {variables[normalized_name]}"
                        )
                else:
                    # Single value assigned to multiple variables (first gets value, rest get None)
                    for i, var_name in enumerate(assign_to):
                        normalized_name = self._normalize_variable_name(var_name)
                        variables[normalized_name] = value_to_assign if i == 0 else None
                        logger.info(
                            f"Assigned {normalized_name} = {variables[normalized_name]}"
                        )

        except Exception as e:
            logger.warning(
                f"Error processing variable assignment for keyword '{keyword}': {e}"
            )
            # Fallback: assign the raw value to first variable name
            if isinstance(assign_to, str):
                var_name = self._normalize_variable_name(assign_to)
                variables[var_name] = value_to_assign
            elif isinstance(assign_to, list) and assign_to:
                var_name = self._normalize_variable_name(assign_to[0])
                variables[var_name] = value_to_assign

        return variables

    def _ensure_library_registration(self, keyword: str, session: Any) -> None:
        """
        Ensure required library is registered in RF context before keyword execution.

        This is Phase 1.2 of the RequestsLibrary fix: Pre-execution Library Registration.
        We determine which library is needed for a keyword and ensure it's registered
        in the Robot Framework execution context.
        """
        try:
            # Determine library from keyword
            library_name = self._get_library_for_keyword(keyword)

            # Honor explicit preference for overlapping keywords
            pref = (getattr(session, "explicit_library_preference", "") or "").lower()
            if keyword.lower() == "open browser":
                if pref.startswith("selenium"):
                    library_name = "SeleniumLibrary"
                elif pref.startswith("browser"):
                    library_name = "Browser"

            # If the scenario explicitly prefers Selenium, avoid registering Browser for
            # overlapping keywords like 'Open Browser' so SeleniumLibrary stays in control.
            if library_name and library_name.lower() == "browser" and pref.startswith("selenium"):
                logger.debug(
                    "Skipping Browser registration for keyword '%s' due to Selenium preference",
                    keyword,
                )
                return

            if library_name:
                # Get the library manager from keyword discovery
                library_manager = self.keyword_discovery.library_manager

                # Ensure RequestsLibrary is loaded in our manager
                if library_name not in library_manager.libraries:
                    logger.info(
                        f"Loading {library_name} on demand for keyword: {keyword}"
                    )
                    library_manager.load_library_on_demand(
                        library_name, self.keyword_discovery
                    )

                # Ensure RequestsLibrary is properly registered in RF context
                registration_success = library_manager.ensure_library_in_rf_context(
                    library_name
                )

                if registration_success:
                    logger.debug(
                        f"Successfully ensured {library_name} registration for keyword: {keyword}"
                    )
                    self.plugin_manager.run_before_keyword_execution(
                        library_name,
                        session,
                        keyword,
                        library_manager,
                        self.keyword_discovery,
                    )

                else:
                    logger.warning(
                        f"Failed to register {library_name} in RF context for keyword: {keyword}"
                    )

        except Exception as e:
            logger.error(f"Library registration check failed for {keyword}: {e}")
            # Don't fail execution for this - let the keyword execution handle library issues

    def _get_library_for_keyword(self, keyword: str) -> Optional[str]:
        """Determine which library provides a given keyword."""

        # Handle explicit library prefixes (e.g., "RequestsLibrary.POST")
        if "." in keyword:
            parts = keyword.split(".")
            if len(parts) == 2:
                library_name, _ = parts
                return library_name

        mapped = self.plugin_manager.get_library_for_keyword(keyword)
        if mapped:
            return mapped
        return None

    def _inject_timeout_into_arguments(
        self,
        keyword: str,
        arguments: List[Any],
        timeout_ms: Optional[int],
        session: ExecutionSession,
    ) -> List[Any]:
        """Inject timeout into keyword arguments for keywords that support it.

        This method adds timeout argument to Browser Library and SeleniumLibrary
        keywords that accept a timeout parameter. The timeout is only injected if:
        1. A timeout_ms value is provided
        2. The keyword supports timeout parameter
        3. No timeout argument is already present

        Args:
            keyword: The keyword name
            arguments: The original arguments list
            timeout_ms: Timeout in milliseconds from TimeoutPolicy
            session: The execution session (for library detection)

        Returns:
            Arguments list with timeout injected if applicable
        """
        if not timeout_ms:
            return arguments

        keyword_lower = keyword.lower().replace(" ", "_").replace("-", "_")

        # Remove library prefix if present
        if "." in keyword_lower:
            keyword_lower = keyword_lower.split(".", 1)[1]

        # Check if timeout is already in arguments (as named argument)
        for arg in arguments:
            if isinstance(arg, str) and arg.lower().startswith("timeout="):
                logger.debug(f"Timeout already present in arguments for {keyword}")
                return arguments

        # Browser Library keywords that ACTUALLY accept timeout parameter
        # NOTE: Most action keywords (click, fill_text, etc.) do NOT accept timeout
        # They use global browser timeout set via "Set Browser Timeout"
        # Only explicit wait keywords accept timeout parameter
        browser_library_timeout_keywords = {
            # Wait operations - these actually accept timeout
            "wait_for_elements_state": "timeout",
            "wait_for_condition": "timeout",
            "wait_for_navigation": "timeout",
            "wait_for_request": "timeout",
            "wait_for_response": "timeout",
            "wait_for_function": "timeout",
            "wait_for_load_state": "timeout",
            "wait_until_network_is_idle": "timeout",
        }
        # NOTE: These keywords do NOT accept timeout parameter directly:
        # - click, fill_text, fill_secret, type_text, press_keys
        # - check_checkbox, uncheck_checkbox, select_options
        # - hover, focus, scroll_to_element
        # - get_text, get_attribute, get_property, get_element_count, get_element_states
        # They all use the global browser timeout

        # SeleniumLibrary keywords that accept timeout parameter
        selenium_library_timeout_keywords = {
            # Element actions
            "click_element": "timeout",
            "click_button": "timeout",
            "click_link": "timeout",
            "input_text": "timeout",
            "input_password": "timeout",
            "select_from_list_by_value": "timeout",
            "select_from_list_by_label": "timeout",
            "select_from_list_by_index": "timeout",
            "select_checkbox": "timeout",
            "unselect_checkbox": "timeout",
            "mouse_over": "timeout",
            # Wait operations
            "wait_until_element_is_visible": "timeout",
            "wait_until_element_is_not_visible": "timeout",
            "wait_until_element_is_enabled": "timeout",
            "wait_until_element_contains": "timeout",
            "wait_until_page_contains_element": "timeout",
            "wait_until_page_does_not_contain_element": "timeout",
        }

        # Convert timeout to seconds (most RF libraries use seconds)
        timeout_seconds = timeout_ms / 1000.0

        # Check Browser Library keywords
        if keyword_lower in browser_library_timeout_keywords:
            # Browser Library uses milliseconds string format like "5s" or "5000ms"
            timeout_arg = f"timeout={timeout_ms}ms"
            logger.debug(f"Injecting Browser Library timeout: {timeout_arg} for {keyword}")
            return list(arguments) + [timeout_arg]

        # Check SeleniumLibrary keywords
        if keyword_lower in selenium_library_timeout_keywords:
            # SeleniumLibrary uses seconds
            timeout_arg = f"timeout={timeout_seconds}"
            logger.debug(f"Injecting SeleniumLibrary timeout: {timeout_arg} for {keyword}")
            return list(arguments) + [timeout_arg]

        # For navigation keywords, no timeout injection needed as they have implicit timeouts
        # For other keywords, return original arguments
        return arguments

    def _normalize_variable_name(self, name: str) -> str:
        """Normalize variable name to Robot Framework format."""
        if not name.startswith("${") or not name.endswith("}"):
            return f"${{{name}}}"
        return name

    def _validate_assignment_compatibility(
        self, keyword: str, assign_to: Union[str, List[str]]
    ) -> None:
        """Validate if keyword is appropriate for variable assignment."""
        if not assign_to:
            return

        # Keywords that typically return useful values for assignment
        returnable_keywords = {
            # String operations
            "Get Length",
            "Get Substring",
            "Replace String",
            "Split String",
            "Convert To Uppercase",
            "Convert To Lowercase",
            "Strip String",
            # Web automation - element queries
            "Get Text",
            "Get Title",
            "Get Location",
            "Get Element Count",
            "Get Element Attribute",
            "Get Element Size",
            "Get Element Position",
            "Get Window Size",
            "Get Window Position",
            "Get Page Source",
            # Web automation - Browser Library
            "Get Url",
            "Get Title",
            "Get Text",
            "Get Attribute",
            "Get Property",
            "Get Element Count",
            "Get Page Source",
            "Evaluate JavaScript",
            # Conversions
            "Convert To Integer",
            "Convert To Number",
            "Convert To String",
            "Convert To Boolean",
            "Evaluate",
            # Collections
            "Get From List",
            "Get Slice From List",
            "Get Length",
            "Get Index",
            "Create List",
            "Create Dictionary",
            "Get Dictionary Keys",
            "Get Dictionary Values",
            # Built-in
            "Set Variable",
            "Get Variable Value",
            "Get Time",
            "Get Environment Variable",
            # System operations
            "Run Process",
            "Run",
            "Get Environment Variable",
        }

        keyword_lower = keyword.lower()
        found_match = False

        for returnable in returnable_keywords:
            if (
                returnable.lower() in keyword_lower
                or keyword_lower in returnable.lower()
            ):
                found_match = True
                break

        if not found_match:
            logger.warning(
                f"Keyword '{keyword}' may not return a useful value for assignment. "
                f"Typical returnable keywords include: Get Text, Get Length, Get Title, etc."
            )

        # Validate assignment count for known multi-return keywords
        multi_return_keywords = {
            "Split String": "Can return multiple parts when max_split is used",
            "Get Time": "Can return multiple time components",
            "Run Process": "Returns stdout and stderr",
            "Get Slice From List": "Can return multiple items",
        }

        for multi_keyword, description in multi_return_keywords.items():
            if multi_keyword.lower() in keyword_lower:
                if isinstance(assign_to, str):
                    logger.info(
                        f"'{keyword}' {description}. Consider using list assignment: ['part1', 'part2']"
                    )
                break

    async def _execute_keyword_internal(
        self,
        session: ExecutionSession,
        step: ExecutionStep,
        browser_library_manager: Any,
        library_prefix: str = None,
        resolved_arguments: List[str] = None,
    ) -> Dict[str, Any]:
        """Execute a specific keyword with error handling and library prefix support."""
        try:
            keyword_name = step.keyword
            # Use resolved arguments if provided, otherwise fall back to step arguments
            args = (
                resolved_arguments if resolved_arguments is not None else step.arguments
            )

            orchestrator = self.keyword_discovery
            session_libraries = self._get_session_libraries(session)
            web_automation_lib = session.get_web_automation_library()
            keyword_info = None

            if session_libraries:
                keyword_info = orchestrator.find_keyword(
                    keyword_name, session_libraries=session_libraries
                )
                logger.debug(
                    f"Session-aware keyword discovery: '{keyword_name}' in session libraries {session_libraries} → {keyword_info.library if keyword_info else None}"
                )
            elif web_automation_lib:
                active_library = (
                    web_automation_lib
                    if web_automation_lib in ["Browser", "SeleniumLibrary"]
                    else None
                )
                keyword_info = orchestrator.find_keyword(
                    keyword_name, active_library=active_library
                )
                logger.debug(
                    f"Active library keyword discovery: '{keyword_name}' with active_library='{active_library}' → {keyword_info.library if keyword_info else None}"
                )
            else:
                keyword_info = orchestrator.find_keyword(keyword_name)
                logger.debug(
                    f"Global keyword discovery: '{keyword_name}' → {keyword_info.library if keyword_info else None}"
                )

            if keyword_info is None:
                logger.debug(
                    f"Keyword '{keyword_name}' not found; ensuring session libraries are loaded"
                )
                await orchestrator._ensure_session_libraries(
                    session.session_id, keyword_name
                )
                session_libraries = self._get_session_libraries(session)
                web_automation_lib = session.get_web_automation_library()
                if session_libraries:
                    keyword_info = orchestrator.find_keyword(
                        keyword_name, session_libraries=session_libraries
                    )
                    logger.debug(
                        f"Post-loading session-aware discovery: '{keyword_name}' in session libraries {session_libraries} → {keyword_info.library if keyword_info else None}"
                    )
                elif web_automation_lib:
                    active_library = (
                        web_automation_lib
                        if web_automation_lib in ["Browser", "SeleniumLibrary"]
                        else None
                    )
                    keyword_info = orchestrator.find_keyword(
                        keyword_name, active_library=active_library
                    )
                    logger.debug(
                        f"Post-loading active library discovery: '{keyword_name}' with active_library='{active_library}' → {keyword_info.library if keyword_info else None}"
                    )
                else:
                    keyword_info = orchestrator.find_keyword(keyword_name)
                    logger.debug(
                        f"Post-loading global discovery: '{keyword_name}' → {keyword_info.library if keyword_info else None}"
                    )

            if keyword_info and keyword_info.library == "Browser":
                logger.info(
                    f"Browser Library keyword detected: {keyword_name} - forcing regular execution mode"
                )

            library_from_map = self._get_library_for_keyword(keyword_name)
            plugin_override = self.plugin_manager.get_keyword_override(
                keyword_info.library if keyword_info else library_from_map,
                keyword_name,
            )
            if plugin_override:
                override_result = await asyncio.to_thread(
                    plugin_override, session, keyword_name, args, keyword_info
                )
                if override_result is not None:
                    library_to_import = (
                        keyword_info.library if keyword_info else library_from_map
                    )
                    if library_to_import:
                        session.import_library(library_to_import, force=True)
                    return override_result

            if self.override_registry and keyword_info:
                override_handler = self.override_registry.get_override(
                    keyword_name, keyword_info.library
                )
                if override_handler:
                    logger.info(
                        f"OVERRIDE: Using override handler {type(override_handler).__name__} for {keyword_name} from {keyword_info.library}"
                    )
                    override_result = await override_handler.execute(
                        session, keyword_name, args, keyword_info
                    )
                    if override_result is not None:
                        session.import_library(keyword_info.library, force=True)
                        logger.info(
                            f"OVERRIDE: Successfully executed {keyword_name} with {keyword_info.library}, imported to session - RETURNING EARLY"
                        )
                        return {
                            "success": override_result.success,
                            "output": override_result.output
                            or f"Executed {keyword_name}",
                            "error": override_result.error,
                            "variables": {},
                            "state_updates": override_result.state_updates or {},
                        }

            # Determine library to use based on session configuration
            web_automation_lib = session.get_web_automation_library()
            current_active = session.get_active_library()
            session_type = session.get_session_type()

            # CRITICAL FIX: Respect session type boundaries
            if session_type.value in [
                "xml_processing",
                "api_testing",
                "data_processing",
                "system_testing",
            ]:
                # Typed sessions should not use web automation auto-detection
                logger.debug(
                    f"Session type '{session_type.value}' - skipping web automation auto-detection"
                )

            elif web_automation_lib:
                # Session has a specific web automation library imported - use it
                if web_automation_lib == "Browser" and (
                    not current_active or current_active == "auto"
                ):
                    browser_library_manager.set_active_library(session, "browser")
                    logger.debug("Using session's web automation library: Browser")
                elif web_automation_lib == "SeleniumLibrary" and (
                    not current_active or current_active == "auto"
                ):
                    browser_library_manager.set_active_library(session, "selenium")
                    logger.debug(
                        "Using session's web automation library: SeleniumLibrary"
                    )

            # Non-context branches removed in context-only mode

        except Exception as e:
            logger.error(f"Error in keyword execution: {e}")
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}",
                "output": "",
                "variables": {},
                "state_updates": {},
            }

    async def _execute_keyword_with_context(
        self,
        session: ExecutionSession,
        keyword: str,
        arguments: List[Any],
        assign_to: Optional[Union[str, List[str]]] = None,
        browser_library_manager: Any = None,
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute keyword within full Robot Framework native context.

        This uses RF's native execution context to enable proper execution of
        keywords like Evaluate, Set Test Variable, etc. that require RF context.

        Args:
            session: ExecutionSession to run in
            keyword: Robot Framework keyword name
            arguments: List of arguments for the keyword
            assign_to: Optional variable assignment
            browser_library_manager: BrowserLibraryManager instance
            timeout_ms: Timeout in milliseconds from TimeoutPolicy

        Returns:
            Execution result with status, output, and state
        """
        try:
            session_id = session.session_id

            # Check for plugin keyword overrides BEFORE execution
            # This allows plugins to intercept and reject incompatible keywords
            library_from_map = self._get_library_for_keyword(keyword)
            plugin_override = self.plugin_manager.get_keyword_override(
                library_from_map, keyword
            )
            if plugin_override:
                try:
                    # Call the plugin override (it's an async function)
                    override_result = await plugin_override(
                        session, keyword, arguments, None
                    )
                    if override_result is not None:
                        # Plugin returned a result - use it (may be an error)
                        logger.info(
                            f"Plugin override handled keyword '{keyword}': success={override_result.get('success', False)}"
                        )
                        return override_result
                except Exception as e:
                    logger.debug(f"Plugin override for '{keyword}' failed: {e}")

            logger.info(
                f"RF NATIVE CONTEXT: Executing {keyword} with native RF context for session {session_id}"
            )

            # Inject timeout into arguments for keywords that support it
            arguments_with_timeout = self._inject_timeout_into_arguments(
                keyword, list(arguments), timeout_ms, session
            )
            if arguments_with_timeout != arguments:
                logger.debug(f"Timeout injected into arguments: {arguments_with_timeout}")

            # Create or get RF native context for session
            context_info = self.rf_native_context.get_session_context_info(session_id)
            if not context_info["context_exists"]:
                # Create RF native context with session's library search order
                # Use search_order if available, otherwise try loaded_libraries
                if hasattr(session, "search_order") and session.search_order:
                    libraries = list(session.search_order)
                elif hasattr(session, "loaded_libraries") and session.loaded_libraries:
                    libraries = list(session.loaded_libraries)
                else:
                    libraries = []

                # If keyword has explicit library prefix (e.g., 'XML.Parse XML'), ensure it's imported
                try:
                    if "." in keyword:
                        prefix = keyword.split(".", 1)[0]
                        if prefix and prefix not in libraries:
                            libraries.append(prefix)
                except Exception:
                    pass

                logger.info(f"Creating RF native context with libraries: {libraries}")
                context_result = self.rf_native_context.create_context_for_session(
                    session_id, libraries
                )
                if not context_result.get("success"):
                    logger.error(
                        f"RF native context creation failed: {context_result.get('error')}"
                    )
                    return {
                        "success": False,
                        "error": f"Failed to create RF native context: {context_result.get('error')}",
                        "keyword": keyword,
                        "arguments": arguments,
                    }
                logger.info(
                    f"Created RF native context for session {session_id} with libraries: {libraries}"
                )

            # Execute keyword using RF native context with session variables
            result = await asyncio.to_thread(
                self.rf_native_context.execute_keyword_with_context,
                session_id=session_id,
                keyword_name=keyword,
                arguments=arguments_with_timeout,
                assign_to=assign_to,
                session_variables=dict(
                    session.variables
                ),  # Pass original objects to RF Variables
            )

            # Update session variables from RF native context
            if result.get("success") and "variables" in result:
                session.variables.update(result["variables"])
                logger.debug(
                    f"Updated session variables from RF native context: {len(result['variables'])} variables"
                )

            # Bridge RF-context browser state back to session for downstream services
            try:
                if result.get("success") and browser_library_manager is not None:
                    from robotmcp.utils.library_detector import (
                        detect_library_type_from_keyword,
                    )

                    detected = detect_library_type_from_keyword(keyword)
                    lib_type = None
                    if detected in ("browser", "selenium"):
                        lib_type = detected
                    if not lib_type and "." in keyword:
                        prefix = keyword.split(".", 1)[0].strip().lower()
                        if prefix == "browser":
                            lib_type = "browser"
                        elif prefix in ("seleniumlibrary", "selenium"):
                            lib_type = "selenium"
                    if not lib_type or lib_type == "auto":
                        keyword_lower = keyword.strip().lower()
                        browser_aliases = {
                            "new browser",
                            "new context",
                            "new page",
                            "go to",
                            "click",
                            "fill text",
                            "type text",
                            "press keys",
                            "get text",
                            "wait for elements state",
                            "get url",
                            "close browser",
                        }
                        selenium_aliases = {
                            "open browser",
                            "go to",
                            "click element",
                            "input text",
                            "press keys",
                            "get text",
                            "get location",
                        }
                        if session.browser_state.active_library == "browser" or keyword_lower in browser_aliases:
                            lib_type = "browser"
                        elif session.browser_state.active_library == "selenium" or keyword_lower in selenium_aliases:
                            lib_type = "selenium"

                    if lib_type:
                        browser_library_manager.set_active_library(session, lib_type)
                        if lib_type == "browser":
                            state_updates = self._extract_browser_state_updates(
                                keyword, arguments, result.get("output")
                            )
                            self._apply_state_updates(session, state_updates)
                            # Capture page source if applicable
                            if keyword.lower().endswith("get page source") or keyword.lower() == "get page source":
                                out = result.get("output") or result.get("result")
                                if isinstance(out, str) and out:
                                    session.browser_state.page_source = out
                        elif lib_type == "selenium":
                            state_updates = self._extract_selenium_state_updates(
                                keyword, arguments, result.get("output")
                            )
                            self._apply_state_updates(session, state_updates)
                            if keyword.lower().endswith("get source") or keyword.lower() == "get source":
                                out = result.get("output") or result.get("result")
                                if isinstance(out, str) and out:
                                    session.browser_state.page_source = out
            except Exception as _bridge_err:
                # Non-fatal; page source tool has additional fallbacks
                pass

            logger.info(
                f"RF NATIVE CONTEXT: {keyword} executed with result: {result.get('success')}"
            )
            return result

        except Exception as e:
            logger.error(f"RF native context execution failed: {e}")
            import traceback

            logger.error(f"RF native context traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"RF native context execution failed: {str(e)}",
                "keyword": keyword,
                "arguments": arguments,
            }


    async def _execute_builtin_keyword(
        self, session: ExecutionSession, keyword: str, args: List[str]
    ) -> Dict[str, Any]:
        """Execute a built-in Robot Framework keyword."""
        try:
            # First, attempt dynamic execution via orchestrator for non-built-in libraries.
            # This path supports full argument parsing (incl. **kwargs) and works for AppiumLibrary, RequestsLibrary, etc.
            try:
                from robotmcp.core.dynamic_keyword_orchestrator import (
                    get_keyword_discovery,
                )

                orchestrator = get_keyword_discovery()
                dyn_result = await orchestrator.execute_keyword(
                    keyword_name=keyword,
                    args=args,
                    session_variables=session.variables,
                    active_library=None,
                    session_id=session.session_id,
                    library_prefix=None,
                )

                # If orchestrator could resolve and execute the keyword, return immediately
                if dyn_result and dyn_result.get("success"):
                    return dyn_result
            except Exception as dyn_error:
                logger.debug(
                    f"Dynamic orchestrator path failed for '{keyword}': {dyn_error}. Falling back to BuiltIn."
                )

            if not ROBOT_AVAILABLE:
                return {
                    "success": False,
                    "error": "Robot Framework not available for built-in keywords",
                    "output": "",
                    "variables": {},
                    "state_updates": {},
                }

            builtin = BuiltIn()
            keyword_lower = keyword.lower()

            # Handle common built-in keywords
            if keyword_lower == "set variable":
                if args:
                    var_value = args[0]
                    return {
                        "success": True,
                        "result": var_value,  # Store actual return value
                        "output": var_value,
                        "variables": {"${VARIABLE}": var_value},
                        "state_updates": {},
                    }

            elif keyword_lower == "log":
                message = args[0] if args else ""
                logger.info(f"Robot Log: {message}")
                return {
                    "success": True,
                    "result": None,  # Log doesn't return a value
                    "output": message,
                    "variables": {},
                    "state_updates": {},
                }

            elif keyword_lower == "should be equal":
                if len(args) >= 2:
                    if args[0] == args[1]:
                        return {
                            "success": True,
                            "result": True,  # Assertion passed
                            "output": f"'{args[0]}' == '{args[1]}'",
                            "variables": {},
                            "state_updates": {},
                        }
                    else:
                        return {
                            "success": False,
                            "result": False,  # Assertion failed
                            "error": f"'{args[0]}' != '{args[1]}'",
                            "output": "",
                            "variables": {},
                            "state_updates": {},
                        }

            # Try to execute using BuiltIn library
            try:
                # ENHANCEMENT: Use RF native type converter for proper argument processing
                # This handles RequestsLibrary and other complex libraries with named arguments
                logger.info(
                    f"BUILTIN KEYWORD EXECUTION PATH: {keyword} with args: {args}"
                )
                print(f"🔍 BUILTIN PATH: {keyword} with args: {args}", file=sys.stderr)
                print(
                    f"🔍 BUILTIN ARGS TYPES: {[type(arg).__name__ for arg in args]}",
                    file=sys.stderr,
                )
                try:
                    processed_args = self.rf_converter.parse_and_convert_arguments(
                        keyword,
                        args,
                        library_name=None,
                        session_variables=session.variables,
                    )
                    logger.info(
                        f"RF converter processed {keyword} args: {args} → {processed_args}"
                    )
                    print(f"🔍 RF CONVERTER SUCCESS: {processed_args}", file=sys.stderr)
                except Exception as converter_error:
                    logger.warning(
                        f"RF converter failed for {keyword}: {converter_error}, falling back to basic processing"
                    )
                    print(f"🔍 RF CONVERTER FAILED: {converter_error}", file=sys.stderr)
                    processed_args = args

                # DUAL HANDLING: RequestsLibrary needs object arguments, others need string arguments
                # FINAL SOLUTION: Inject objects directly before keyword execution
                final_args = self._inject_objects_for_execution(processed_args, session)

                result = builtin.run_keyword(keyword, *final_args)
                return {
                    "success": True,
                    "result": result,  # Store the actual return value
                    "output": str(result) if result is not None else "OK",
                    "variables": {},
                    "state_updates": {},
                }
            except Exception as e:
                # Phase 4: Add comprehensive diagnostics for keyword execution failures
                diagnostics = self._get_keyword_failure_diagnostics(
                    keyword, args, str(e), session
                )
                return {
                    "success": False,
                    "error": f"Built-in keyword execution failed: {str(e)}",
                    "output": "",
                    "variables": {},
                    "state_updates": {},
                    "diagnostics": diagnostics,  # Phase 4: Enhanced diagnostics
                }

        except Exception as e:
            logger.error(f"Error executing built-in keyword {keyword}: {e}")
            # Phase 4: Add diagnostics for outer exception handler too
            diagnostics = self._get_keyword_failure_diagnostics(
                keyword, args, str(e), session
            )
            return {
                "success": False,
                "error": f"Built-in keyword execution failed: {str(e)}",
                "output": "",
                "variables": {},
                "state_updates": {},
                "diagnostics": diagnostics,  # Phase 4: Enhanced diagnostics
            }

    def _get_keyword_failure_diagnostics(
        self,
        keyword: str,
        args: List[str],
        error_message: str,
        session: ExecutionSession,
    ) -> Dict[str, Any]:
        """
        Phase 4: Get comprehensive diagnostic information for keyword execution failures.

        Args:
            keyword: The keyword that failed
            args: Arguments provided to the keyword
            error_message: The error message from the failure
            session: ExecutionSession for context

        Returns:
            Dictionary with diagnostic information
        """
        # Use the orchestrator's diagnostic capabilities
        from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery

        orchestrator = get_keyword_discovery()

        # Get comprehensive diagnostics from the orchestrator
        diagnostics = orchestrator._get_diagnostic_info(
            keyword_name=keyword,
            session_id=session.session_id,
            active_library=session.get_active_library(),
        )

        # Add keyword executor specific information
        diagnostics["execution_context"] = {
            "execution_path": "builtin_keyword_executor",
            "provided_arguments": args,
            "argument_count": len(args),
            "execution_error": error_message,
            "session_type": session.get_session_type().value,
        }

        # Add Robot Framework specific diagnostics
        try:
            from robot.running.context import EXECUTION_CONTEXTS

            rf_context_available = bool(EXECUTION_CONTEXTS.current)
            diagnostics["robot_framework_context"] = {
                "execution_context_available": rf_context_available
            }
        except:
            diagnostics["robot_framework_context"] = {
                "execution_context_available": False
            }

        return diagnostics

    def _keyword_expects_object_arguments(
        self, keyword: str, arg_index: int, arg_value: Any
    ) -> bool:
        """
        Determine if a keyword expects object arguments at a specific position.

        This is critical for RequestsLibrary which expects dict/list objects for json/data parameters,
        while most other keywords expect string arguments.
        """
        keyword_lower = keyword.lower()

        # Debug output
        print(
            f"🔍 OBJECT CHECK: keyword={keyword_lower}, arg_index={arg_index}, arg_value={arg_value}, type={type(arg_value).__name__}",
            file=sys.stderr,
        )

        # RequestsLibrary keywords that accept object parameters
        requests_keywords_with_objects = {
            "post": ["json", "data"],
            "put": ["json", "data"],
            "patch": ["json", "data"],
            "post on session": ["json", "data"],
            "put on session": ["json", "data"],
            "patch on session": ["json", "data"],
        }

        if keyword_lower in requests_keywords_with_objects:
            # Check if this is a dict or list object that should be preserved
            if isinstance(arg_value, (dict, list)):
                print(
                    f"🔍 PRESERVING OBJECT: RequestsLibrary keyword {keyword} detected with {type(arg_value).__name__} argument",
                    file=sys.stderr,
                )
                logger.debug(
                    f"RequestsLibrary keyword {keyword} detected with {type(arg_value).__name__} argument - preserving as object"
                )
                return True

        print(
            f"🔍 CONVERTING TO STRING: keyword={keyword}, arg will be converted",
            file=sys.stderr,
        )
        # For other complex argument structures that might need objects
        # Add more library-specific logic here as needed

        return False

    def _process_object_preserving_arguments(self, args: List[Any]) -> List[Any]:
        """
        Handle ObjectPreservingArgument objects for Robot Framework execution.

        Robot Framework's argument resolver expects named parameters to be handled
        differently than simple string formatting. For object values, we need to
        pass them as separate arguments or use RF's native parameter handling.
        """
        from robotmcp.components.variables.variable_resolver import (
            ObjectPreservingArgument,
        )

        processed_args = []

        for arg in args:
            if isinstance(arg, ObjectPreservingArgument):
                # CORRECT APPROACH: For RF execution, we need to preserve the object
                # and pass it in a way that RF's ArgumentResolver can handle.
                # Instead of converting to string, we store the object and use a reference
                # that will be resolved during actual keyword execution.

                # Store object in temporary session storage for later injection
                processed_args.append(arg)  # Keep the ObjectPreservingArgument object
            else:
                processed_args.append(arg)

        return processed_args

    def _store_and_reference_objects(self, args: List[Any], session: Any) -> List[str]:
        """
        FINAL SOLUTION: Store ObjectPreservingArgument objects in session and replace with references.

        This stores the actual objects in the session's temporary storage and replaces
        them with placeholder references that can be injected back later.
        """
        from robotmcp.components.variables.variable_resolver import (
            ObjectPreservingArgument,
        )

        processed_args = []

        for arg in args:
            if isinstance(arg, ObjectPreservingArgument):
                # Create a unique reference ID for this object
                import uuid

                ref_id = f"__OBJ_REF_{uuid.uuid4().hex[:8]}"

                # Store the actual object in session temporary storage
                if not hasattr(session, "_temp_objects"):
                    session._temp_objects = {}
                session._temp_objects[ref_id] = arg.value

                # Replace with a reference that includes the parameter name
                processed_args.append(f"{arg.param_name}=${{{ref_id}}}")

                # Also store the reference in session variables for RF to resolve
                session.variables[ref_id] = arg.value

            else:
                processed_args.append(arg)

        return processed_args

    def _inject_objects_for_execution(self, args: List[str], session: Any) -> List[Any]:
        """
        FINAL SOLUTION: Inject actual objects directly at execution time.

        This replaces object reference placeholders with the actual objects
        right before the keyword is executed, bypassing all the complex
        variable resolution issues.
        """
        # Inject objects for RequestsLibrary and other libraries expecting object parameters
        final_args = []

        for arg in args:
            # Handle URL parameter conversion to positional format
            if (
                isinstance(arg, str)
                and arg.startswith("url=")
                and "${__OBJ_REF_" not in arg
            ):
                # Convert URL from named to positional for RequestsLibrary
                url_value = arg[4:]  # Remove 'url=' prefix
                final_args.append(url_value)
            elif isinstance(arg, str) and "${__OBJ_REF_" in arg:
                # This argument contains an object reference - extract and inject the object
                import re

                # Find object reference patterns in the argument
                ref_pattern = r"\$\{(__OBJ_REF_[^}]+)\}"
                matches = re.findall(ref_pattern, arg)

                if matches:
                    # Replace each reference with the actual object
                    processed_arg = arg
                    for ref_id in matches:
                        if (
                            hasattr(session, "_temp_objects")
                            and ref_id in session._temp_objects
                        ):
                            actual_object = session._temp_objects[ref_id]

                            # If the entire argument is just the reference, replace with the object
                            if processed_arg == f"${{{ref_id}}}":
                                final_args.append(actual_object)
                                break
                            # If it's a named parameter, inject the object as the value
                            elif (
                                "=" in processed_arg
                                and f"${{{ref_id}}}" in processed_arg
                            ):
                                param_name = processed_arg.split("=")[0]
                                # Use tuple format for RF named args with objects
                                final_args.append((param_name, actual_object))
                                break
                    else:
                        # No replacement made, keep as string
                        final_args.append(arg)
                else:
                    final_args.append(arg)
            else:
                final_args.append(arg)

        return final_args

    def _process_arguments_with_rf_native_resolver(
        self, keyword: str, args: List[Any], session: Any
    ) -> List[Any]:
        """
        Process arguments using Robot Framework's native ArgumentResolver patterns.

        This is the general solution that handles:
        1. ObjectPreservingArgument objects from variable resolution
        2. Proper argument formatting (named vs positional parameters)
        3. Type preservation for object parameters

        This works for ANY library that expects object parameters, not just RequestsLibrary.
        """
        from robotmcp.components.variables.variable_resolver import (
            ObjectPreservingArgument,
        )

        processed_args = []

        for i, arg in enumerate(args):
            if isinstance(arg, ObjectPreservingArgument):
                # This is a named parameter with an object value
                print(
                    f"🔍 PROCESSING OBJECT ARG: {arg.param_name}={arg.value} (type: {type(arg.value).__name__})",
                    file=sys.stderr,
                )

                # For Robot Framework, we need to handle named parameters properly
                # The RF ArgumentResolver expects either:
                # 1. Positional args followed by named args like: ['value1', 'param2=value2']
                # 2. Or kwargs-style processing

                # Keep it as named parameter but preserve the object
                processed_args.append(f"{arg.param_name}={arg.value}")

            elif isinstance(arg, str) and "=" in arg and arg.count("=") == 1:
                # This is a string-based named parameter, handle URL parameter specially
                param_name, param_value = arg.split("=", 1)

                # For common first positional parameters like 'url', convert to positional
                if param_name == "url" and i == 0:  # First argument and it's URL
                    print(
                        f"🔍 CONVERTING URL TO POSITIONAL: {param_value}",
                        file=sys.stderr,
                    )
                    processed_args.append(param_value)
                else:
                    # Keep as named parameter
                    processed_args.append(arg)

            else:
                # Regular argument (positional or already processed)
                if not isinstance(arg, str):
                    # Convert non-string args to string
                    processed_args.append(str(arg))
                else:
                    processed_args.append(arg)

        return processed_args

    def _fix_stringified_objects_for_requests_library(
        self,
        keyword: str,
        original_args: List[str],
        resolved_args: List[str],
        session_variables: Dict[str, Any],
    ) -> List[Any]:
        """
        Fix stringified objects and argument format for RequestsLibrary keywords.

        This fixes two issues:
        1. Variable resolution converts objects to strings (e.g., json=${body} becomes "json={'key': 'value'}")
        2. Named parameters need proper formatting for RequestsLibrary (e.g., "url=value" → "value", "json=object" → object)
        """
        keyword_lower = keyword.lower()

        # Only apply this fix for RequestsLibrary keywords that expect object parameters
        requests_keywords_with_objects = {
            "post",
            "put",
            "patch",
            "post on session",
            "put on session",
            "patch on session",
        }

        if keyword_lower not in requests_keywords_with_objects:
            return resolved_args

        # Get the expected signature for this keyword
        from robotmcp.utils.rf_native_type_converter import REQUESTS_LIBRARY_SIGNATURES

        signature = REQUESTS_LIBRARY_SIGNATURES.get(keyword.upper(), [])

        print(
            f"🔍 REQUESTS SIGNATURE: {keyword.upper()} → {signature}", file=sys.stderr
        )

        fixed_args = []
        for i, (orig_arg, resolved_arg) in enumerate(zip(original_args, resolved_args)):
            # Check if this was a named parameter
            if (
                "=" in orig_arg
                and "=" in str(resolved_arg)
                and orig_arg.count("=") == 1
                and str(resolved_arg).count("=") == 1
            ):
                orig_param_name, orig_param_value = orig_arg.split("=", 1)
                resolved_param_name, resolved_param_value = str(resolved_arg).split(
                    "=", 1
                )

                print(
                    f"🔍 PROCESSING PARAM: {orig_param_name}={orig_param_value}",
                    file=sys.stderr,
                )

                # Handle URL parameter (first positional parameter for session-less methods)
                if orig_param_name == "url" and keyword_lower in [
                    "post",
                    "put",
                    "patch",
                    "get",
                    "delete",
                ]:
                    # URL should be positional, not named
                    print(
                        f"🔍 CONVERTING URL TO POSITIONAL: {resolved_param_value}",
                        file=sys.stderr,
                    )
                    fixed_args.append(resolved_param_value)
                    continue

                # Handle object parameters (json, data)
                if orig_param_name in ["json", "data"]:
                    # Check if original was a variable reference that should have been an object
                    if (
                        orig_param_value.startswith("${")
                        and orig_param_value.endswith("}")
                        and "[" not in orig_param_value
                    ):
                        var_name = orig_param_value[2:-1]  # Remove ${ and }
                        if var_name in session_variables:
                            original_value = session_variables[var_name]

                            # If the original value is a dict/list but got stringified, restore it
                            if isinstance(original_value, (dict, list)):
                                print(
                                    f"🔍 RESTORING OBJECT FOR {orig_param_name}: {orig_param_value} → object",
                                    file=sys.stderr,
                                )
                                # Keep it as named parameter but with restored object
                                fixed_args.append(f"{orig_param_name}={original_value}")
                                continue

                # Default: keep named parameter as-is
                fixed_args.append(resolved_arg)
            else:
                # Non-named parameter, keep as-is
                fixed_args.append(resolved_arg)

        print(f"🔍 FINAL FIXED ARGS: {fixed_args}", file=sys.stderr)
        return fixed_args

    def _extract_browser_state_updates(
        self, keyword: str, args: List[str], result: Any
    ) -> Dict[str, Any]:
        """Extract state updates from Browser Library keyword execution."""
        state_updates = {}
        keyword_lower = keyword.lower()

        # Extract state changes based on keyword
        if "new browser" in keyword_lower:
            browser_type = args[0] if args else "chromium"
            state_updates["current_browser"] = {"type": browser_type}
        elif "new context" in keyword_lower:
            state_updates["current_context"] = {
                "id": str(result) if result else "context"
            }
        elif "new page" in keyword_lower:
            url = args[0] if args else ""
            state_updates["current_page"] = {
                "id": str(result) if result else "page",
                "url": url,
            }
        elif "go to" in keyword_lower:
            url = args[0] if args else ""
            state_updates["current_page"] = {"url": url}

        return state_updates

    def _extract_selenium_state_updates(
        self, keyword: str, args: List[str], result: Any
    ) -> Dict[str, Any]:
        """Extract state updates from SeleniumLibrary keyword execution."""
        state_updates = {}
        keyword_lower = keyword.lower()

        # Extract state changes based on keyword
        if "open browser" in keyword_lower:
            state_updates["current_browser"] = {
                "type": args[1] if len(args) > 1 else "firefox"
            }
        elif "go to" in keyword_lower:
            state_updates["current_page"] = {"url": args[0] if args else ""}

        return state_updates

    def _apply_state_updates(
        self, session: ExecutionSession, state_updates: Dict[str, Any]
    ) -> None:
        """Apply state updates to session browser state."""
        if not state_updates:
            return

        browser_state = session.browser_state

        for key, value in state_updates.items():
            if key == "current_browser":
                if isinstance(value, dict):
                    browser_state.browser_type = value.get("type")
            elif key == "current_context":
                if isinstance(value, dict):
                    browser_state.context_id = value.get("id")
            elif key == "current_page":
                if isinstance(value, dict):
                    browser_state.current_url = value.get("url")
                    browser_state.page_id = value.get("id")

    async def _build_response_by_detail_level(
        self,
        detail_level: str,
        result: Dict[str, Any],
        step: ExecutionStep,
        keyword: str,
        arguments: List[str],
        session: ExecutionSession,
        resolved_arguments: List[str] = None,
    ) -> Dict[str, Any]:
        """Build execution response based on requested detail level."""
        base_response = {
            "success": result["success"],
            "step_id": step.step_id,
            "keyword": keyword,
            "arguments": arguments,  # Show original arguments in response
            "status": step.status,
            "execution_time": step.execution_time,
        }

        if not result["success"]:
            base_response["error"] = result.get("error", "Unknown error")
            # Propagate hints from lower layers or generate as fallback
            hints = result.get("hints") or []
            library_name = result.get("library_name") or self._get_library_for_keyword(
                keyword
            )
            plugin_hints = self.plugin_manager.generate_failure_hints(
                library_name,
                session,
                keyword,
                list(arguments or []),
                str(base_response["error"]),
            )
            if plugin_hints:
                hints = list(plugin_hints) + list(hints)
            if not hints:
                try:
                    from robotmcp.utils.hints import HintContext, generate_hints

                    hctx = HintContext(
                        session_id=session.session_id,
                        keyword=keyword,
                        arguments=list(arguments or []),
                        error_text=str(base_response["error"]),
                        session_search_order=getattr(session, "search_order", None),
                    )
                    hints = generate_hints(hctx)
                except Exception:
                    hints = []
            base_response["hints"] = hints

        if detail_level == "minimal":
            # Serialize output to prevent MCP serialization errors with complex objects
            raw_output = result.get("output", "")
            base_response["output"] = self.response_serializer.serialize_for_response(
                raw_output
            )
            # Include assigned variables in all detail levels for debugging
            if "assigned_variables" in result:
                base_response["assigned_variables"] = result["assigned_variables"]

        elif detail_level == "standard":
            # DUAL STORAGE: Keep ORIGINAL objects in session for RF, serialize ONLY for MCP response
            # Do NOT serialize session.variables as they need to remain original for RF execution
            session_vars_for_response = {}
            for var_name, var_value in session.variables.items():
                # Only serialize for MCP response display, but keep originals in session.variables
                session_vars_for_response[var_name] = (
                    self.response_serializer.serialize_for_response(var_value)
                )

            # Serialize output for standard detail level
            raw_output = result.get("output", "")
            serialized_output = self.response_serializer.serialize_for_response(
                raw_output
            )

            base_response.update(
                {
                    "output": serialized_output,
                    "session_variables": session_vars_for_response,  # Serialized for MCP response only
                    "active_library": session.get_active_library(),
                }
            )
            # Include assigned variables in standard detail level (serialized for MCP)
            if "assigned_variables" in result:
                base_response["assigned_variables"] = result["assigned_variables"]
            # Add resolved arguments for debugging if they differ from original (serialized)
            if resolved_arguments is not None and resolved_arguments != arguments:
                serialized_resolved_args = [
                    self.response_serializer.serialize_for_response(arg)
                    for arg in resolved_arguments
                ]
                base_response["resolved_arguments"] = serialized_resolved_args

        elif detail_level == "full":
            # DUAL STORAGE: Keep ORIGINAL objects in session for RF, serialize ONLY for MCP response
            session_vars_for_response = {}
            for var_name, var_value in session.variables.items():
                # Only serialize for MCP response display, but keep originals in session.variables
                session_vars_for_response[var_name] = (
                    self.response_serializer.serialize_for_response(var_value)
                )

            # Serialize output for full detail level
            raw_output = result.get("output", "")
            serialized_output = self.response_serializer.serialize_for_response(
                raw_output
            )

            # Serialize state_updates to prevent MCP serialization errors
            raw_state_updates = result.get("state_updates", {})
            serialized_state_updates = {}
            for key, value in raw_state_updates.items():
                serialized_state_updates[key] = (
                    self.response_serializer.serialize_for_response(value)
                )

            base_response.update(
                {
                    "output": serialized_output,
                    "session_variables": session_vars_for_response,  # Serialized for MCP response only
                    "state_updates": serialized_state_updates,
                    "active_library": session.get_active_library(),
                    "browser_state": {
                        "browser_type": session.browser_state.browser_type,
                        "current_url": session.browser_state.current_url,
                        "context_id": session.browser_state.context_id,
                        "page_id": session.browser_state.page_id,
                    },
                    "step_count": session.step_count,
                    "duration": session.duration,
                }
            )
            # Include assigned variables in full detail level
            if "assigned_variables" in result:
                base_response["assigned_variables"] = result["assigned_variables"]
            # Always include resolved arguments in full detail for debugging (serialized)
            if resolved_arguments is not None:
                serialized_resolved_args = [
                    self.response_serializer.serialize_for_response(arg)
                    for arg in resolved_arguments
                ]
                base_response["resolved_arguments"] = serialized_resolved_args

        else:
            # Unrecognized detail_level — fall back to minimal (includes output)
            logger.warning(
                f"Unrecognized detail_level '{detail_level}', "
                f"falling back to 'minimal'. Valid values: minimal, standard, full"
            )
            raw_output = result.get("output", "")
            base_response["output"] = self.response_serializer.serialize_for_response(
                raw_output
            )
            if "assigned_variables" in result:
                base_response["assigned_variables"] = result["assigned_variables"]

        return base_response

    def get_supported_detail_levels(self) -> List[str]:
        """Get list of supported detail levels."""
        return ["minimal", "standard", "full"]

    def validate_detail_level(self, detail_level: str) -> bool:
        """Validate that the detail level is supported."""
        return detail_level in self.get_supported_detail_levels()

    def _get_selenium_error_guidance(
        self, keyword: str, args: List[str], error_message: str
    ) -> Dict[str, Any]:
        """Generate SeleniumLibrary-specific error guidance for agents."""
        # Get base locator guidance
        guidance = self.rf_converter.get_selenium_locator_guidance(
            error_message, keyword
        )

        # Add keyword-specific guidance
        keyword_lower = keyword.lower()

        if any(
            term in keyword_lower
            for term in ["click", "input", "select", "clear", "wait"]
        ):
            # Element interaction keywords
            guidance["keyword_specific_tips"] = [
                f"'{keyword}' requires a valid element locator as the first argument",
                "Common locator patterns: 'id:elementId', 'name:fieldName', 'css:.className'",
                "Ensure the element is visible and interactable before interaction",
            ]

            # Analyze the locator argument if provided
            if args:
                locator = args[0]
                if not any(strategy in locator for strategy in [":", "="]):
                    guidance["locator_analysis"] = {
                        "provided_locator": locator,
                        "issue": "Locator appears to be missing strategy prefix",
                        "suggestions": [
                            f"Try 'id:{locator}' if it's an ID",
                            f"Try 'name:{locator}' if it's a name attribute",
                            f"Try 'css:{locator}' if it's a CSS selector",
                            f"Try 'xpath://*[@id=\"{locator}\"]' for XPath",
                        ],
                    }
                elif "=" in locator and ":" not in locator:
                    guidance["locator_analysis"] = {
                        "provided_locator": locator,
                        "issue": "Contains '=' but no strategy prefix - may be parsed as named argument",
                        "correct_format": f"name:{locator}"
                        if locator.startswith("name=")
                        else "Use appropriate strategy prefix",
                        "note": "SeleniumLibrary requires 'strategy:value' format, not 'strategy=value'",
                    }

        elif "open" in keyword_lower or "browser" in keyword_lower:
            guidance["keyword_specific_tips"] = [
                f"'{keyword}' manages browser/session state",
                "Ensure proper browser initialization before element interactions",
                "Check browser driver compatibility and installation",
            ]

        return guidance

    def _get_browser_error_guidance(
        self, keyword: str, args: List[str], error_message: str
    ) -> Dict[str, Any]:
        """Generate Browser Library-specific error guidance for agents."""
        # Get base locator guidance
        guidance = self.rf_converter.get_browser_locator_guidance(
            error_message, keyword
        )

        # Add keyword-specific guidance
        keyword_lower = keyword.lower()

        if any(
            term in keyword_lower
            for term in ["click", "fill", "select", "check", "type", "press", "hover"]
        ):
            # Element interaction keywords
            guidance["keyword_specific_tips"] = [
                f"'{keyword}' requires a valid element selector",
                "Browser Library uses CSS selectors by default (no prefix needed)",
                "Common patterns: '.class', '#id', 'button', 'input[type=\"submit\"]'",
                "For complex elements, use cascaded selectors: 'div.container >> .button'",
            ]

            # Analyze the selector argument if provided
            if args:
                selector = args[0]
                guidance.update(self._analyze_browser_selector(selector))

        elif any(
            term in keyword_lower
            for term in ["new browser", "new page", "new context", "go to"]
        ):
            guidance["keyword_specific_tips"] = [
                f"'{keyword}' manages browser/page state",
                "Ensure proper browser initialization sequence",
                "Check browser installation and dependencies",
                "Verify URL accessibility for navigation keywords",
            ]

        elif "wait" in keyword_lower:
            guidance["keyword_specific_tips"] = [
                f"'{keyword}' handles dynamic content and timing",
                "Adjust timeout values for slow-loading elements",
                "Use appropriate wait conditions (visible, hidden, enabled, etc.)",
                "Consider page load states for complete readiness",
            ]

        return guidance

    def _analyze_browser_selector(self, selector: str) -> Dict[str, Any]:
        """Analyze a Browser Library selector and provide specific guidance."""
        analysis = {}

        # Detect selector patterns and provide guidance (order matters - check >>> before >>)
        if ">>>" in selector:
            analysis["iframe_selector_detected"] = {
                "type": "iFrame piercing selector",
                "explanation": "Using >>> to access elements inside frames",
                "tip": "Left side selects frame, right side selects element inside frame",
            }

        elif selector.startswith("#") and not selector.startswith("\\#"):
            analysis["selector_warning"] = {
                "issue": "ID selector may need escaping in Robot Framework",
                "provided_selector": selector,
                "recommended": f"\\{selector}",
                "explanation": "# is a comment character in Robot Framework, use \\# for ID selectors",
            }

        elif ">>" in selector:
            analysis["cascaded_selector_detected"] = {
                "type": "Cascaded selector (good practice)",
                "explanation": "Using >> to chain multiple selector strategies",
                "tip": "Each part of the chain is relative to the previous match",
            }

        elif selector.startswith('"') and selector.endswith('"'):
            analysis["text_selector_detected"] = {
                "type": "Text selector (implicit)",
                "explanation": "Quoted strings are treated as text selectors",
                "equivalent_explicit": f"text={selector}",
                "tip": "Use for exact text matching",
            }

        elif selector.startswith("//") or selector.startswith(".."):
            analysis["xpath_selector_detected"] = {
                "type": "XPath selector (implicit)",
                "explanation": "Selectors starting with // or .. are treated as XPath",
                "equivalent_explicit": f"xpath={selector}",
                "tip": "XPath provides powerful element traversal capabilities",
            }

        elif "=" in selector and any(
            selector.startswith(prefix) for prefix in ["css=", "xpath=", "text=", "id="]
        ):
            strategy = selector.split("=", 1)[0]
            analysis["explicit_strategy_detected"] = {
                "type": f"Explicit {strategy} selector",
                "explanation": f"Using explicit {strategy} strategy",
                "tip": "Good practice to be explicit with selector strategies",
            }

        else:
            analysis["implicit_css_detected"] = {
                "type": "CSS selector (implicit default)",
                "explanation": "Plain selectors are treated as CSS by default",
                "equivalent_explicit": f"css={selector}",
                "tip": "Browser Library defaults to CSS selectors",
            }

        return analysis


    def _get_session_libraries(self, session: ExecutionSession) -> List[str]:
        """Get list of libraries loaded in the session for session-aware keyword resolution.

        Args:
            session: ExecutionSession to get libraries from

        Returns:
            List of library names loaded in the session
        """
        session_libraries = []

        # Try to get loaded libraries from session
        if hasattr(session, "loaded_libraries") and session.loaded_libraries:
            session_libraries = list(session.loaded_libraries)
        elif hasattr(session, "search_order") and session.search_order:
            session_libraries = list(session.search_order)
        elif hasattr(session, "imported_libraries") and session.imported_libraries:
            session_libraries = list(session.imported_libraries)

        # Always include core built-in libraries
        builtin_libraries = ["BuiltIn", "Collections", "String"]
        for lib in builtin_libraries:
            if lib not in session_libraries:
                session_libraries.append(lib)

        logger.debug(f"Session libraries for keyword resolution: {session_libraries}")
        return session_libraries

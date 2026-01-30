"""Playwright Browser Library Adapter.

This module implements the BrowserLibraryAdapter protocol for Robot Framework's
Browser Library, which uses Playwright as its backend.

The Browser Library provides modern web testing capabilities including:
- Native async support
- Network interception
- Multi-browser/context/page management
- Built-in accessibility tree (ARIA) snapshots
- Robust auto-waiting mechanisms
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set

from .browser_adapter import (
    ActionParameters,
    ActionResult,
    ActionType,
    AriaSnapshot,
    BaseBrowserAdapter,
    ElementState,
    PageInfo,
)

logger = logging.getLogger(__name__)

# Check for Browser Library availability
try:
    from Browser import Browser as BrowserLibrary

    BROWSER_LIBRARY_AVAILABLE = True
except ImportError:
    BrowserLibrary = None  # type: ignore
    BROWSER_LIBRARY_AVAILABLE = False
    logger.debug("Browser Library not available")


class PlaywrightBrowserAdapter(BaseBrowserAdapter):
    """Adapter for Robot Framework Browser Library (Playwright-based).

    This adapter translates the generic BrowserLibraryAdapter interface
    to Browser Library's Playwright-based keywords.

    Key features:
    - Native ARIA snapshot support via Get Aria Snapshots keyword
    - Element states via Get Element States keyword
    - Modern auto-waiting with configurable timeouts
    - Support for multiple browsers, contexts, and pages
    """

    # Mapping of ActionType to Browser Library keywords
    ACTION_KEYWORD_MAP: Dict[ActionType, str] = {
        ActionType.CLICK: "Click",
        ActionType.FILL: "Fill Text",
        ActionType.SELECT: "Select Options By",
        ActionType.CHECK: "Check Checkbox",
        ActionType.UNCHECK: "Uncheck Checkbox",
        ActionType.HOVER: "Hover",
        ActionType.FOCUS: "Focus",
        ActionType.BLUR: "Blur",
        ActionType.PRESS: "Keyboard Key",
        ActionType.TYPE: "Type Text",
        ActionType.CLEAR: "Clear Text",
        ActionType.SCROLL_INTO_VIEW: "Scroll To Element",
        ActionType.DOUBLE_CLICK: "Click",  # With click_count=2
        ActionType.RIGHT_CLICK: "Click",  # With button=right
        ActionType.DRAG_AND_DROP: "Drag And Drop",
        ActionType.GET_TEXT: "Get Text",
        ActionType.GET_ATTRIBUTE: "Get Attribute",
        ActionType.GET_PROPERTY: "Get Property",
        ActionType.GET_VALUE: "Get Text",  # Use Get Text for input values
        ActionType.SET_VALUE: "Fill Text",
        ActionType.UPLOAD_FILE: "Upload File By Selector",
        ActionType.SCREENSHOT: "Take Screenshot",
    }

    # Mapping of ElementState to Browser Library state names
    STATE_MAP: Dict[ElementState, str] = {
        ElementState.ATTACHED: "attached",
        ElementState.DETACHED: "detached",
        ElementState.VISIBLE: "visible",
        ElementState.HIDDEN: "hidden",
        ElementState.ENABLED: "enabled",
        ElementState.DISABLED: "disabled",
        ElementState.EDITABLE: "editable",
        ElementState.READONLY: "readonly",
        ElementState.CHECKED: "checked",
        ElementState.UNCHECKED: "unchecked",
        ElementState.FOCUSED: "focused",
        ElementState.STABLE: "stable",
    }

    # Reverse mapping for state translation
    REVERSE_STATE_MAP: Dict[str, ElementState] = {v: k for k, v in STATE_MAP.items()}

    def __init__(
        self,
        robot_instance: Any = None,
        browser_library_manager: Any = None,
    ):
        """Initialize the Playwright Browser adapter.

        Args:
            robot_instance: Robot Framework instance for keyword execution.
            browser_library_manager: BrowserLibraryManager instance for library access.
        """
        super().__init__(robot_instance)
        self._browser_manager = browser_library_manager
        self._browser_lib: Optional[Any] = None
        self._initialize_library()

    def _initialize_library(self) -> None:
        """Initialize the Browser Library instance."""
        if not BROWSER_LIBRARY_AVAILABLE:
            self._logger.warning("Browser Library not available")
            return

        try:
            if self._browser_manager and hasattr(self._browser_manager, "browser_lib"):
                self._browser_lib = self._browser_manager.browser_lib
            else:
                # Fallback: try to create a new instance
                self._browser_lib = BrowserLibrary()

            if self._browser_lib:
                # Disable interactive pause on failure
                try:
                    setattr(self._browser_lib, "pause_on_failure", False)
                except Exception as e:
                    self._logger.debug(f"Could not disable pause_on_failure: {e}")

            self._logger.info("Playwright Browser adapter initialized")

        except Exception as e:
            self._logger.error(f"Failed to initialize Browser Library: {e}")
            self._browser_lib = None

    @property
    def library_name(self) -> str:
        """Return the library name."""
        return "Browser"

    @property
    def library_type(self) -> str:
        """Return the library type identifier."""
        return "browser"

    @property
    def is_available(self) -> bool:
        """Check if Browser Library is available and initialized."""
        return BROWSER_LIBRARY_AVAILABLE and self._browser_lib is not None

    def capture_aria_snapshot(
        self,
        selector: Optional[str] = None,
        timeout_ms: int = 30000,
    ) -> AriaSnapshot:
        """Capture accessibility tree snapshot using Browser Library's Get Aria Snapshots.

        Browser Library provides native ARIA snapshot support which returns
        a YAML-formatted accessibility tree representation.

        Args:
            selector: Optional CSS selector to scope the snapshot.
            timeout_ms: Timeout in milliseconds.

        Returns:
            AriaSnapshot with the accessibility tree content.
        """
        if not self.is_available:
            return AriaSnapshot(
                content="",
                success=False,
                error="Browser Library not available",
            )

        try:
            start_time = time.time()

            # Use Browser Library's Get Aria Snapshots keyword
            if selector:
                # Ensure selector has proper prefix
                normalized_selector = self._translate_locator(selector)
                result = self._execute_keyword(
                    "Get Aria Snapshots",
                    normalized_selector,
                )
            else:
                # Get entire page snapshot
                result = self._execute_keyword("Get Aria Snapshots")

            execution_time = (time.time() - start_time) * 1000

            if result is None:
                return AriaSnapshot(
                    content="",
                    selector=selector,
                    success=False,
                    error="No ARIA snapshot returned",
                )

            # Browser Library returns YAML-formatted string
            content = str(result) if result else ""

            return AriaSnapshot(
                content=content,
                selector=selector,
                format="yaml",
                success=True,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

        except Exception as e:
            self._logger.error(f"Failed to capture ARIA snapshot: {e}")
            return AriaSnapshot(
                content="",
                selector=selector,
                success=False,
                error=str(e),
            )

    def execute_action(
        self,
        action_type: ActionType,
        locator: str,
        parameters: ActionParameters,
        timeout_ms: int = 30000,
    ) -> ActionResult:
        """Execute a browser action using Browser Library keywords.

        Args:
            action_type: The type of action to perform.
            locator: Element locator.
            parameters: Action-specific parameters.
            timeout_ms: Timeout in milliseconds.

        Returns:
            ActionResult with execution status and any returned value.
        """
        if not self.is_available:
            return ActionResult(
                success=False,
                error="Browser Library not available",
            )

        try:
            start_time = time.time()
            normalized_locator = self._translate_locator(locator)
            timeout_str = f"{timeout_ms}ms"

            result = self._execute_action_internal(
                action_type,
                normalized_locator,
                parameters,
                timeout_str,
            )

            execution_time = (time.time() - start_time) * 1000

            return ActionResult(
                success=True,
                value=result,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            return self._handle_error(e, f"Action {action_type.value}", locator)

    def _execute_action_internal(
        self,
        action_type: ActionType,
        locator: str,
        parameters: ActionParameters,
        timeout_str: str,
    ) -> Any:
        """Internal method to execute specific actions.

        Args:
            action_type: The action type.
            locator: Normalized locator.
            parameters: Action parameters.
            timeout_str: Timeout as string (e.g., "30000ms").

        Returns:
            The keyword result.
        """
        keyword = self.ACTION_KEYWORD_MAP.get(action_type)
        if not keyword:
            raise ValueError(f"Unsupported action type: {action_type}")

        # Build keyword arguments based on action type
        if action_type == ActionType.CLICK:
            kwargs: Dict[str, Any] = {}
            if parameters.click_count > 1:
                kwargs["clickCount"] = parameters.click_count
            if parameters.button != "left":
                kwargs["button"] = parameters.button
            if parameters.force:
                kwargs["force"] = True
            if parameters.position_x is not None and parameters.position_y is not None:
                kwargs["position"] = {"x": parameters.position_x, "y": parameters.position_y}
            return self._execute_keyword(keyword, locator, **kwargs)

        elif action_type == ActionType.DOUBLE_CLICK:
            return self._execute_keyword(keyword, locator, clickCount=2)

        elif action_type == ActionType.RIGHT_CLICK:
            return self._execute_keyword(keyword, locator, button="right")

        elif action_type in (ActionType.FILL, ActionType.SET_VALUE):
            text = parameters.text or ""
            kwargs = {}
            if parameters.force:
                kwargs["force"] = True
            return self._execute_keyword(keyword, locator, text, **kwargs)

        elif action_type == ActionType.TYPE:
            text = parameters.text or ""
            kwargs = {}
            if parameters.delay_ms:
                kwargs["delay"] = f"{parameters.delay_ms}ms"
            return self._execute_keyword(keyword, locator, text, **kwargs)

        elif action_type == ActionType.CLEAR:
            return self._execute_keyword(keyword, locator)

        elif action_type == ActionType.SELECT:
            # Browser Library uses Select Options By with attribute and value
            attribute = "value" if parameters.value else ("label" if parameters.label else "index")
            value = parameters.value or parameters.label or str(parameters.index or 0)
            return self._execute_keyword(keyword, locator, attribute, value)

        elif action_type in (ActionType.CHECK, ActionType.UNCHECK):
            kwargs = {}
            if parameters.force:
                kwargs["force"] = True
            return self._execute_keyword(keyword, locator, **kwargs)

        elif action_type == ActionType.HOVER:
            return self._execute_keyword(keyword, locator)

        elif action_type == ActionType.FOCUS:
            return self._execute_keyword(keyword, locator)

        elif action_type == ActionType.BLUR:
            return self._execute_keyword(keyword, locator)

        elif action_type == ActionType.PRESS:
            key = parameters.key or "Enter"
            return self._execute_keyword(keyword, "press", key)

        elif action_type == ActionType.SCROLL_INTO_VIEW:
            return self._execute_keyword(keyword, locator)

        elif action_type == ActionType.DRAG_AND_DROP:
            target = parameters.target_locator or ""
            target_normalized = self._translate_locator(target) if target else ""
            return self._execute_keyword(keyword, locator, target_normalized)

        elif action_type == ActionType.GET_TEXT:
            return self._execute_keyword(keyword, locator)

        elif action_type == ActionType.GET_ATTRIBUTE:
            attr_name = parameters.attribute_name or "value"
            return self._execute_keyword(keyword, locator, attr_name)

        elif action_type == ActionType.GET_PROPERTY:
            prop_name = parameters.property_name or "value"
            return self._execute_keyword(keyword, locator, prop_name)

        elif action_type == ActionType.GET_VALUE:
            return self._execute_keyword("Get Text", locator)

        elif action_type == ActionType.UPLOAD_FILE:
            files = parameters.file_paths or []
            if files:
                return self._execute_keyword(keyword, locator, files[0])
            raise ValueError("No file paths provided for upload")

        elif action_type == ActionType.SCREENSHOT:
            kwargs = {}
            if parameters.full_page:
                kwargs["fullPage"] = True
            if parameters.screenshot_path:
                kwargs["filename"] = parameters.screenshot_path
            return self._execute_keyword(keyword, **kwargs)

        else:
            raise ValueError(f"Unhandled action type: {action_type}")

    def check_element_states(
        self,
        locator: str,
        timeout_ms: int = 5000,
    ) -> Set[ElementState]:
        """Get element states using Browser Library's Get Element States.

        Args:
            locator: Element locator.
            timeout_ms: Timeout in milliseconds.

        Returns:
            Set of ElementState values.
        """
        if not self.is_available:
            return set()

        try:
            normalized_locator = self._translate_locator(locator)

            # Browser Library's Get Element States returns a list of state strings
            states_result = self._execute_keyword(
                "Get Element States",
                normalized_locator,
            )

            if not states_result:
                return set()

            # Convert Browser Library states to ElementState enum values
            element_states: Set[ElementState] = set()

            # Handle both list and enum-like results
            state_list = list(states_result) if hasattr(states_result, "__iter__") else [states_result]

            for state in state_list:
                state_str = str(state).lower() if state else ""
                # Handle Browser Library's AssertionOperator-style states
                if "." in state_str:
                    state_str = state_str.split(".")[-1]

                if state_str in self.REVERSE_STATE_MAP:
                    element_states.add(self.REVERSE_STATE_MAP[state_str])

            return element_states

        except Exception as e:
            self._logger.warning(f"Failed to get element states for '{locator}': {e}")
            return set()

    def wait_for_element_state(
        self,
        locator: str,
        state: ElementState,
        timeout_ms: int = 30000,
    ) -> ActionResult:
        """Wait for element state using Browser Library's Wait For Elements State.

        Args:
            locator: Element locator.
            state: The state to wait for.
            timeout_ms: Timeout in milliseconds.

        Returns:
            ActionResult indicating success or failure.
        """
        if not self.is_available:
            return ActionResult(
                success=False,
                error="Browser Library not available",
            )

        try:
            normalized_locator = self._translate_locator(locator)
            state_str = self.STATE_MAP.get(state, state.value)
            timeout_str = f"{timeout_ms}ms"

            start_time = time.time()

            self._execute_keyword(
                "Wait For Elements State",
                normalized_locator,
                state_str,
                timeout=timeout_str,
            )

            execution_time = (time.time() - start_time) * 1000

            return ActionResult(
                success=True,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            return self._handle_error(e, f"Wait for state {state.value}", locator)

    def get_page_url(self) -> str:
        """Get current page URL using Browser Library's Get Url."""
        if not self.is_available:
            return ""

        try:
            return str(self._execute_keyword("Get Url") or "")
        except Exception as e:
            self._logger.warning(f"Failed to get page URL: {e}")
            return ""

    def get_page_title(self) -> str:
        """Get current page title using Browser Library's Get Title."""
        if not self.is_available:
            return ""

        try:
            return str(self._execute_keyword("Get Title") or "")
        except Exception as e:
            self._logger.warning(f"Failed to get page title: {e}")
            return ""

    def get_page_info(self) -> PageInfo:
        """Get comprehensive page information."""
        if not self.is_available:
            return PageInfo(url="", title="")

        try:
            url = self.get_page_url()
            title = self.get_page_title()

            # Try to get viewport info via JavaScript
            viewport_info = self._get_viewport_info()

            return PageInfo(
                url=url,
                title=title,
                viewport_width=viewport_info.get("width", 0),
                viewport_height=viewport_info.get("height", 0),
                scroll_x=viewport_info.get("scrollX", 0),
                scroll_y=viewport_info.get("scrollY", 0),
                document_ready_state=viewport_info.get("readyState", "complete"),
            )

        except Exception as e:
            self._logger.warning(f"Failed to get page info: {e}")
            return PageInfo(url="", title="")

    def _get_viewport_info(self) -> Dict[str, Any]:
        """Get viewport information via JavaScript."""
        try:
            script = """
            return {
                width: window.innerWidth,
                height: window.innerHeight,
                scrollX: window.scrollX,
                scrollY: window.scrollY,
                readyState: document.readyState
            };
            """
            result = self.evaluate_javascript(script)
            return result if isinstance(result, dict) else {}
        except Exception:
            return {}

    def get_page_source(self) -> str:
        """Get page HTML source using Browser Library's Get Page Source."""
        if not self.is_available:
            return ""

        try:
            return str(self._execute_keyword("Get Page Source") or "")
        except Exception as e:
            self._logger.warning(f"Failed to get page source: {e}")
            return ""

    def evaluate_javascript(
        self,
        script: str,
        *args: Any,
    ) -> Any:
        """Execute JavaScript using Browser Library's Evaluate JavaScript."""
        if not self.is_available:
            raise RuntimeError("Browser Library not available")

        try:
            # Browser Library uses Evaluate JavaScript keyword
            if args:
                # Pass arguments as array
                return self._execute_keyword("Evaluate JavaScript", None, script, *args)
            return self._execute_keyword("Evaluate JavaScript", None, script)
        except Exception as e:
            self._logger.error(f"JavaScript evaluation failed: {e}")
            raise

    def take_screenshot(
        self,
        path: Optional[str] = None,
        full_page: bool = False,
        selector: Optional[str] = None,
    ) -> Optional[str]:
        """Take screenshot using Browser Library's Take Screenshot."""
        if not self.is_available:
            return None

        try:
            kwargs: Dict[str, Any] = {}

            if path:
                kwargs["filename"] = path

            if full_page:
                kwargs["fullPage"] = True

            if selector:
                kwargs["selector"] = self._translate_locator(selector)

            result = self._execute_keyword("Take Screenshot", **kwargs)
            return str(result) if result else path

        except Exception as e:
            self._logger.error(f"Screenshot failed: {e}")
            return None

    def _translate_locator(self, locator: str) -> str:
        """Translate locator to Browser Library format.

        Browser Library accepts various locator strategies:
        - CSS: css=selector or selector (default)
        - XPath: xpath=//path
        - Text: text=content
        - ID: id=element_id
        - Playwright-specific: >> chained selectors

        Args:
            locator: Generic locator string.

        Returns:
            Browser Library compatible locator.
        """
        if not locator:
            return locator

        # Already has a strategy prefix
        prefixes = ["css=", "xpath=", "text=", "id=", ">>", "data-testid="]
        for prefix in prefixes:
            if locator.lower().startswith(prefix.lower()):
                return locator

        # Convert common patterns
        if locator.startswith("//") or locator.startswith("(//"):
            # XPath
            return f"xpath={locator}"
        elif locator.startswith("#"):
            # ID selector - use css
            return f"css={locator}"
        elif locator.startswith("."):
            # Class selector - use css
            return f"css={locator}"
        else:
            # Default to css selector
            return f"css={locator}"

    def close(self) -> None:
        """Clean up Browser Library resources."""
        self._logger.debug("Closing Playwright Browser adapter")
        # Browser Library handles its own cleanup
        self._browser_lib = None

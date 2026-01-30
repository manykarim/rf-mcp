"""Selenium Library Adapter.

This module implements the BrowserLibraryAdapter protocol for Robot Framework's
SeleniumLibrary, which uses Selenium WebDriver as its backend.

SeleniumLibrary is the traditional/legacy browser automation library that
provides broad browser support through WebDriver. This adapter serves as a
fallback when Browser Library is not available or when Selenium-specific
features are needed.

Key differences from Browser Library:
- No native ARIA snapshot support (uses JavaScript fallback)
- Different keyword names (e.g., "Click Element" vs "Click")
- Synchronous execution model
- Uses WebDriver-based locator strategies
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

# Check for SeleniumLibrary availability
try:
    from SeleniumLibrary import SeleniumLibrary

    SELENIUM_LIBRARY_AVAILABLE = True
except ImportError:
    SeleniumLibrary = None  # type: ignore
    SELENIUM_LIBRARY_AVAILABLE = False
    logger.debug("SeleniumLibrary not available")


class SeleniumLibraryAdapter(BaseBrowserAdapter):
    """Adapter for Robot Framework SeleniumLibrary (Selenium WebDriver-based).

    This adapter translates the generic BrowserLibraryAdapter interface
    to SeleniumLibrary's WebDriver-based keywords.

    Key features:
    - JavaScript-based ARIA snapshot extraction (fallback implementation)
    - Traditional WebDriver element state checking
    - Broad browser compatibility via WebDriver
    - Explicit waits through SeleniumLibrary keywords
    """

    # Mapping of ActionType to SeleniumLibrary keywords
    ACTION_KEYWORD_MAP: Dict[ActionType, str] = {
        ActionType.CLICK: "Click Element",
        ActionType.FILL: "Input Text",
        ActionType.SELECT: "Select From List By Value",
        ActionType.CHECK: "Select Checkbox",
        ActionType.UNCHECK: "Unselect Checkbox",
        ActionType.HOVER: "Mouse Over",
        ActionType.FOCUS: "Set Focus To Element",
        ActionType.BLUR: "Execute Javascript",  # No direct blur keyword
        ActionType.PRESS: "Press Keys",
        ActionType.TYPE: "Input Text",
        ActionType.CLEAR: "Clear Element Text",
        ActionType.SCROLL_INTO_VIEW: "Scroll Element Into View",
        ActionType.DOUBLE_CLICK: "Double Click Element",
        ActionType.RIGHT_CLICK: "Open Context Menu",
        ActionType.DRAG_AND_DROP: "Drag And Drop",
        ActionType.GET_TEXT: "Get Text",
        ActionType.GET_ATTRIBUTE: "Get Element Attribute",
        ActionType.GET_PROPERTY: "Execute Javascript",  # No direct property keyword
        ActionType.GET_VALUE: "Get Value",
        ActionType.SET_VALUE: "Input Text",
        ActionType.UPLOAD_FILE: "Choose File",
        ActionType.SCREENSHOT: "Capture Page Screenshot",
    }

    # JavaScript for extracting accessibility tree (ARIA snapshot fallback)
    ARIA_EXTRACTION_JS = """
    (function extractAriaTree(root, depth) {
        if (!root || depth > 50) return '';

        const indent = '  '.repeat(depth);
        let result = '';

        // Skip hidden elements
        const style = window.getComputedStyle(root);
        if (style.display === 'none' || style.visibility === 'hidden') {
            return '';
        }

        // Get role (explicit or implicit)
        let role = root.getAttribute('role') || '';
        if (!role) {
            // Map common elements to implicit roles
            const tagRoles = {
                'A': 'link',
                'BUTTON': 'button',
                'INPUT': root.type === 'checkbox' ? 'checkbox' :
                         root.type === 'radio' ? 'radio' :
                         root.type === 'submit' || root.type === 'button' ? 'button' :
                         'textbox',
                'SELECT': 'combobox',
                'TEXTAREA': 'textbox',
                'IMG': 'img',
                'H1': 'heading',
                'H2': 'heading',
                'H3': 'heading',
                'H4': 'heading',
                'H5': 'heading',
                'H6': 'heading',
                'NAV': 'navigation',
                'MAIN': 'main',
                'HEADER': 'banner',
                'FOOTER': 'contentinfo',
                'ASIDE': 'complementary',
                'ARTICLE': 'article',
                'SECTION': 'region',
                'FORM': 'form',
                'TABLE': 'table',
                'UL': 'list',
                'OL': 'list',
                'LI': 'listitem',
                'TR': 'row',
                'TH': 'columnheader',
                'TD': 'cell',
            };
            role = tagRoles[root.tagName] || '';
        }

        // Get accessible name
        let name = root.getAttribute('aria-label') ||
                   root.getAttribute('aria-labelledby') ||
                   root.getAttribute('title') ||
                   root.getAttribute('alt') ||
                   root.getAttribute('placeholder') ||
                   '';

        // For inputs, get value
        let value = '';
        if (root.tagName === 'INPUT' || root.tagName === 'TEXTAREA') {
            value = root.value || '';
        } else if (root.tagName === 'SELECT') {
            value = root.options[root.selectedIndex]?.text || '';
        }

        // Get text content for elements without explicit name
        if (!name && role) {
            const directText = Array.from(root.childNodes)
                .filter(n => n.nodeType === Node.TEXT_NODE)
                .map(n => n.textContent.trim())
                .filter(t => t)
                .join(' ');
            if (directText) name = directText;
        }

        // Build YAML-like output
        if (role || name || value) {
            result += indent + '- ';
            if (role) result += role;
            if (name) result += ` "${name.substring(0, 100).replace(/"/g, '\\'')}"`;
            if (value) result += ` [value="${value.substring(0, 50).replace(/"/g, '\\'')}"]`;

            // Add relevant states
            const states = [];
            if (root.disabled) states.push('disabled');
            if (root.checked) states.push('checked');
            if (root.required) states.push('required');
            if (root.getAttribute('aria-expanded') === 'true') states.push('expanded');
            if (root.getAttribute('aria-selected') === 'true') states.push('selected');
            if (root.getAttribute('aria-pressed') === 'true') states.push('pressed');

            if (states.length > 0) {
                result += ' [' + states.join(', ') + ']';
            }

            result += '\\n';
        }

        // Process children
        for (const child of root.children) {
            result += extractAriaTree(child, depth + 1);
        }

        return result;
    })(arguments[0] || document.body, 0);
    """

    # JavaScript for checking element states
    ELEMENT_STATES_JS = """
    (function(element) {
        if (!element) return [];

        const states = [];
        const style = window.getComputedStyle(element);

        // Check visibility
        const isVisible = style.display !== 'none' &&
                          style.visibility !== 'hidden' &&
                          style.opacity !== '0' &&
                          element.offsetParent !== null;
        states.push(isVisible ? 'visible' : 'hidden');

        // Check if attached to DOM
        states.push(document.body.contains(element) ? 'attached' : 'detached');

        // Check enabled state
        states.push(element.disabled ? 'disabled' : 'enabled');

        // Check editable
        const isEditable = !element.disabled &&
                           !element.readOnly &&
                           (element.tagName === 'INPUT' ||
                            element.tagName === 'TEXTAREA' ||
                            element.contentEditable === 'true');
        states.push(isEditable ? 'editable' : 'readonly');

        // Check checked state (for checkboxes/radios)
        if (element.type === 'checkbox' || element.type === 'radio') {
            states.push(element.checked ? 'checked' : 'unchecked');
        }

        // Check focused
        states.push(document.activeElement === element ? 'focused' : 'unfocused');

        return states;
    })(arguments[0]);
    """

    def __init__(
        self,
        robot_instance: Any = None,
        selenium_library: Any = None,
    ):
        """Initialize the SeleniumLibrary adapter.

        Args:
            robot_instance: Robot Framework instance for keyword execution.
            selenium_library: SeleniumLibrary instance, or None to create new.
        """
        super().__init__(robot_instance)
        self._selenium_lib: Optional[Any] = None
        self._initialize_library(selenium_library)

    def _initialize_library(self, selenium_library: Any = None) -> None:
        """Initialize the SeleniumLibrary instance."""
        if not SELENIUM_LIBRARY_AVAILABLE:
            self._logger.warning("SeleniumLibrary not available")
            return

        try:
            if selenium_library:
                self._selenium_lib = selenium_library
            else:
                self._selenium_lib = SeleniumLibrary()

            self._logger.info("SeleniumLibrary adapter initialized")

        except Exception as e:
            self._logger.error(f"Failed to initialize SeleniumLibrary: {e}")
            self._selenium_lib = None

    @property
    def library_name(self) -> str:
        """Return the library name."""
        return "SeleniumLibrary"

    @property
    def library_type(self) -> str:
        """Return the library type identifier."""
        return "selenium"

    @property
    def is_available(self) -> bool:
        """Check if SeleniumLibrary is available and initialized."""
        return SELENIUM_LIBRARY_AVAILABLE and self._selenium_lib is not None

    def capture_aria_snapshot(
        self,
        selector: Optional[str] = None,
        timeout_ms: int = 30000,
    ) -> AriaSnapshot:
        """Capture accessibility tree using JavaScript extraction.

        SeleniumLibrary doesn't have native ARIA snapshot support like
        Browser Library, so this method uses JavaScript to extract a
        basic accessibility tree representation.

        Args:
            selector: Optional CSS/XPath selector to scope the snapshot.
            timeout_ms: Timeout in milliseconds.

        Returns:
            AriaSnapshot with YAML-like accessibility tree content.
        """
        if not self.is_available:
            return AriaSnapshot(
                content="",
                success=False,
                error="SeleniumLibrary not available",
            )

        try:
            start_time = time.time()

            # Get the root element for extraction
            if selector:
                normalized_selector = self._translate_locator(selector)
                # Execute JS with the element as argument
                js_code = f"""
                var el = document.querySelector('{self._css_from_locator(normalized_selector)}');
                if (!el) return '';
                {self.ARIA_EXTRACTION_JS}
                """
                # Alternative: use WebElement directly
                result = self._execute_javascript_with_element(
                    selector, self.ARIA_EXTRACTION_JS
                )
            else:
                # Get entire page
                result = self._execute_keyword(
                    "Execute Javascript", self.ARIA_EXTRACTION_JS
                )

            execution_time = (time.time() - start_time) * 1000

            content = str(result) if result else ""

            # Clean up the output
            content = content.strip()
            if not content:
                content = "# No accessible elements found"

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

    def _execute_javascript_with_element(
        self,
        locator: str,
        script: str,
    ) -> Any:
        """Execute JavaScript with an element as the first argument.

        Args:
            locator: Element locator.
            script: JavaScript code expecting element as arguments[0].

        Returns:
            JavaScript execution result.
        """
        normalized = self._translate_locator(locator)
        # SeleniumLibrary's Execute Javascript can take element locator
        full_script = f"""
        var element = arguments[0];
        {script}
        """
        return self._execute_keyword(
            "Execute Javascript",
            full_script,
            f"ARGUMENTS:{normalized}",
        )

    def _css_from_locator(self, locator: str) -> str:
        """Extract CSS selector from locator string for JS use.

        Args:
            locator: Locator string.

        Returns:
            CSS selector suitable for document.querySelector.
        """
        if locator.startswith("css="):
            return locator[4:]
        elif locator.startswith("id="):
            return f"#{locator[3:]}"
        elif locator.startswith("class="):
            return f".{locator[6:]}"
        elif locator.startswith("//") or locator.startswith("xpath="):
            # Can't convert XPath to CSS reliably
            return "*"
        else:
            return locator

    def execute_action(
        self,
        action_type: ActionType,
        locator: str,
        parameters: ActionParameters,
        timeout_ms: int = 30000,
    ) -> ActionResult:
        """Execute a browser action using SeleniumLibrary keywords.

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
                error="SeleniumLibrary not available",
            )

        try:
            start_time = time.time()
            normalized_locator = self._translate_locator(locator)

            # Set implicit wait timeout
            timeout_sec = timeout_ms / 1000
            self._execute_keyword("Set Selenium Implicit Wait", f"{timeout_sec}s")

            result = self._execute_action_internal(
                action_type,
                normalized_locator,
                parameters,
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
    ) -> Any:
        """Internal method to execute specific actions.

        Args:
            action_type: The action type.
            locator: Normalized locator.
            parameters: Action parameters.

        Returns:
            The keyword result.
        """
        keyword = self.ACTION_KEYWORD_MAP.get(action_type)
        if not keyword:
            raise ValueError(f"Unsupported action type: {action_type}")

        if action_type == ActionType.CLICK:
            return self._execute_keyword(keyword, locator)

        elif action_type == ActionType.DOUBLE_CLICK:
            return self._execute_keyword(keyword, locator)

        elif action_type == ActionType.RIGHT_CLICK:
            return self._execute_keyword(keyword, locator)

        elif action_type in (ActionType.FILL, ActionType.SET_VALUE, ActionType.TYPE):
            text = parameters.text or ""
            # Clear first if needed
            if action_type == ActionType.FILL:
                try:
                    self._execute_keyword("Clear Element Text", locator)
                except Exception:
                    pass  # Element might not be clearable
            return self._execute_keyword(keyword, locator, text)

        elif action_type == ActionType.CLEAR:
            return self._execute_keyword(keyword, locator)

        elif action_type == ActionType.SELECT:
            if parameters.value:
                return self._execute_keyword("Select From List By Value", locator, parameters.value)
            elif parameters.label:
                return self._execute_keyword("Select From List By Label", locator, parameters.label)
            elif parameters.index is not None:
                return self._execute_keyword("Select From List By Index", locator, str(parameters.index))
            raise ValueError("No selection criteria provided")

        elif action_type == ActionType.CHECK:
            return self._execute_keyword(keyword, locator)

        elif action_type == ActionType.UNCHECK:
            return self._execute_keyword(keyword, locator)

        elif action_type == ActionType.HOVER:
            return self._execute_keyword(keyword, locator)

        elif action_type == ActionType.FOCUS:
            return self._execute_keyword(keyword, locator)

        elif action_type == ActionType.BLUR:
            # No direct blur keyword - use JavaScript
            return self._execute_keyword(
                "Execute Javascript",
                "arguments[0].blur();",
                f"ARGUMENTS:{locator}",
            )

        elif action_type == ActionType.PRESS:
            key = parameters.key or "ENTER"
            return self._execute_keyword(keyword, locator, key)

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
            # Use JavaScript for property access
            prop_name = parameters.property_name or "value"
            return self._execute_keyword(
                "Execute Javascript",
                f"return arguments[0].{prop_name};",
                f"ARGUMENTS:{locator}",
            )

        elif action_type == ActionType.GET_VALUE:
            return self._execute_keyword(keyword, locator)

        elif action_type == ActionType.UPLOAD_FILE:
            files = parameters.file_paths or []
            if files:
                return self._execute_keyword(keyword, locator, files[0])
            raise ValueError("No file paths provided for upload")

        elif action_type == ActionType.SCREENSHOT:
            if parameters.screenshot_path:
                return self._execute_keyword(keyword, parameters.screenshot_path)
            return self._execute_keyword(keyword)

        else:
            raise ValueError(f"Unhandled action type: {action_type}")

    def check_element_states(
        self,
        locator: str,
        timeout_ms: int = 5000,
    ) -> Set[ElementState]:
        """Get element states using JavaScript.

        SeleniumLibrary doesn't have a direct equivalent to Browser Library's
        Get Element States, so this uses JavaScript to check element states.

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

            # Wait for element to be present
            timeout_sec = timeout_ms / 1000
            self._execute_keyword(
                "Wait Until Element Is Visible",
                normalized_locator,
                f"{timeout_sec}s",
            )

            # Execute JavaScript to get states
            states_result = self._execute_keyword(
                "Execute Javascript",
                self.ELEMENT_STATES_JS,
                f"ARGUMENTS:{normalized_locator}",
            )

            if not states_result:
                return set()

            # Convert state strings to ElementState enum
            element_states: Set[ElementState] = set()
            state_mapping = {
                "visible": ElementState.VISIBLE,
                "hidden": ElementState.HIDDEN,
                "attached": ElementState.ATTACHED,
                "detached": ElementState.DETACHED,
                "enabled": ElementState.ENABLED,
                "disabled": ElementState.DISABLED,
                "editable": ElementState.EDITABLE,
                "readonly": ElementState.READONLY,
                "checked": ElementState.CHECKED,
                "unchecked": ElementState.UNCHECKED,
                "focused": ElementState.FOCUSED,
            }

            for state_str in states_result:
                state_lower = str(state_str).lower()
                if state_lower in state_mapping:
                    element_states.add(state_mapping[state_lower])

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
        """Wait for element state using SeleniumLibrary wait keywords.

        Maps ElementState to appropriate SeleniumLibrary wait keywords.

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
                error="SeleniumLibrary not available",
            )

        try:
            normalized_locator = self._translate_locator(locator)
            timeout_str = f"{timeout_ms / 1000}s"

            start_time = time.time()

            # Map states to SeleniumLibrary wait keywords
            state_keyword_map = {
                ElementState.VISIBLE: "Wait Until Element Is Visible",
                ElementState.HIDDEN: "Wait Until Element Is Not Visible",
                ElementState.ATTACHED: "Wait Until Page Contains Element",
                ElementState.DETACHED: "Wait Until Page Does Not Contain Element",
                ElementState.ENABLED: "Wait Until Element Is Enabled",
                ElementState.DISABLED: "Wait Until Element Is Not Enabled",  # Custom JS
            }

            keyword = state_keyword_map.get(state)

            if keyword:
                self._execute_keyword(keyword, normalized_locator, timeout_str)
            else:
                # For states without direct keywords, use JavaScript polling
                self._wait_for_state_js(normalized_locator, state, timeout_ms)

            execution_time = (time.time() - start_time) * 1000

            return ActionResult(
                success=True,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            return self._handle_error(e, f"Wait for state {state.value}", locator)

    def _wait_for_state_js(
        self,
        locator: str,
        state: ElementState,
        timeout_ms: int,
    ) -> None:
        """Wait for element state using JavaScript polling.

        Args:
            locator: Normalized locator.
            state: Target state.
            timeout_ms: Timeout in milliseconds.

        Raises:
            TimeoutError: If state is not reached within timeout.
        """
        state_checks = {
            ElementState.CHECKED: "element.checked === true",
            ElementState.UNCHECKED: "element.checked === false",
            ElementState.FOCUSED: "document.activeElement === element",
            ElementState.EDITABLE: "!element.disabled && !element.readOnly",
            ElementState.READONLY: "element.readOnly === true",
        }

        check = state_checks.get(state)
        if not check:
            raise ValueError(f"Cannot wait for state: {state}")

        js_wait = f"""
        var element = arguments[0];
        var timeout = arguments[1];
        var start = Date.now();

        return new Promise((resolve, reject) => {{
            function checkState() {{
                if ({check}) {{
                    resolve(true);
                }} else if (Date.now() - start > timeout) {{
                    reject(new Error('Timeout waiting for state: {state.value}'));
                }} else {{
                    setTimeout(checkState, 100);
                }}
            }}
            checkState();
        }});
        """

        self._execute_keyword(
            "Execute Async Javascript",
            js_wait,
            f"ARGUMENTS:{locator}",
            str(timeout_ms),
        )

    def get_page_url(self) -> str:
        """Get current page URL using SeleniumLibrary's Get Location."""
        if not self.is_available:
            return ""

        try:
            return str(self._execute_keyword("Get Location") or "")
        except Exception as e:
            self._logger.warning(f"Failed to get page URL: {e}")
            return ""

    def get_page_title(self) -> str:
        """Get current page title using SeleniumLibrary's Get Title."""
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

            # Get viewport info via JavaScript
            viewport_js = """
            return {
                width: window.innerWidth,
                height: window.innerHeight,
                scrollX: window.scrollX || window.pageXOffset,
                scrollY: window.scrollY || window.pageYOffset,
                readyState: document.readyState
            };
            """
            viewport_info = self._execute_keyword("Execute Javascript", viewport_js)
            viewport_info = viewport_info if isinstance(viewport_info, dict) else {}

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

    def get_page_source(self) -> str:
        """Get page HTML source using SeleniumLibrary's Get Source."""
        if not self.is_available:
            return ""

        try:
            return str(self._execute_keyword("Get Source") or "")
        except Exception as e:
            self._logger.warning(f"Failed to get page source: {e}")
            return ""

    def evaluate_javascript(
        self,
        script: str,
        *args: Any,
    ) -> Any:
        """Execute JavaScript using SeleniumLibrary's Execute Javascript."""
        if not self.is_available:
            raise RuntimeError("SeleniumLibrary not available")

        try:
            if args:
                # Convert args to ARGUMENTS format
                args_str = ",".join(f"ARGUMENTS:{arg}" for arg in args)
                return self._execute_keyword("Execute Javascript", script, args_str)
            return self._execute_keyword("Execute Javascript", script)
        except Exception as e:
            self._logger.error(f"JavaScript evaluation failed: {e}")
            raise

    def take_screenshot(
        self,
        path: Optional[str] = None,
        full_page: bool = False,
        selector: Optional[str] = None,
    ) -> Optional[str]:
        """Take screenshot using SeleniumLibrary's Capture Page Screenshot."""
        if not self.is_available:
            return None

        try:
            if selector:
                # Capture element screenshot
                normalized = self._translate_locator(selector)
                if path:
                    return str(
                        self._execute_keyword(
                            "Capture Element Screenshot",
                            normalized,
                            path,
                        )
                    )
                return str(
                    self._execute_keyword("Capture Element Screenshot", normalized)
                )
            else:
                # Capture page screenshot
                if path:
                    return str(
                        self._execute_keyword("Capture Page Screenshot", path)
                    )
                return str(self._execute_keyword("Capture Page Screenshot"))

        except Exception as e:
            self._logger.error(f"Screenshot failed: {e}")
            return None

    def _translate_locator(self, locator: str) -> str:
        """Translate locator to SeleniumLibrary format.

        SeleniumLibrary accepts various locator strategies:
        - ID: id=element_id or id:element_id
        - Name: name=element_name
        - XPath: xpath=//path or //path
        - CSS: css=selector or css:selector
        - Class: class=classname
        - Link text: link=text
        - Tag: tag=tagname

        Args:
            locator: Generic locator string.

        Returns:
            SeleniumLibrary compatible locator.
        """
        if not locator:
            return locator

        # Already has a SeleniumLibrary strategy prefix
        prefixes = [
            "id=", "id:", "name=", "xpath=", "css=", "css:",
            "class=", "link=", "tag=", "//",
        ]
        for prefix in prefixes:
            if locator.lower().startswith(prefix.lower()):
                return locator

        # Convert common patterns
        if locator.startswith("#"):
            # ID selector
            return f"id={locator[1:]}"
        elif locator.startswith("."):
            # Class selector - use css
            return f"css={locator}"
        elif locator.startswith("["):
            # Attribute selector - use css
            return f"css={locator}"
        else:
            # Default to css selector
            return f"css={locator}"

    def close(self) -> None:
        """Clean up SeleniumLibrary resources."""
        self._logger.debug("Closing SeleniumLibrary adapter")
        # SeleniumLibrary handles its own cleanup
        self._selenium_lib = None

"""Browser Library Adapter Protocol - Anti-Corruption Layer.

This module defines the protocol (interface) for browser library adapters,
providing an abstraction layer between rf-mcp and the underlying browser
automation libraries (Browser Library/Playwright and SeleniumLibrary).

The Anti-Corruption Layer pattern ensures that:
1. rf-mcp core code doesn't depend directly on library-specific implementations
2. Library differences are encapsulated in concrete adapters
3. New browser libraries can be added without modifying core logic
4. Testing can use mock adapters for isolation
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set, runtime_checkable

logger = logging.getLogger(__name__)


class ElementState(Enum):
    """Standardized element states across browser libraries."""

    ATTACHED = "attached"
    DETACHED = "detached"
    VISIBLE = "visible"
    HIDDEN = "hidden"
    ENABLED = "enabled"
    DISABLED = "disabled"
    EDITABLE = "editable"
    READONLY = "readonly"
    CHECKED = "checked"
    UNCHECKED = "unchecked"
    FOCUSED = "focused"
    STABLE = "stable"


class ActionType(Enum):
    """Standardized browser action types."""

    CLICK = "click"
    FILL = "fill"
    SELECT = "select"
    CHECK = "check"
    UNCHECK = "uncheck"
    HOVER = "hover"
    FOCUS = "focus"
    BLUR = "blur"
    PRESS = "press"
    TYPE = "type"
    CLEAR = "clear"
    SCROLL_INTO_VIEW = "scroll_into_view"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG_AND_DROP = "drag_and_drop"
    GET_TEXT = "get_text"
    GET_ATTRIBUTE = "get_attribute"
    GET_PROPERTY = "get_property"
    GET_VALUE = "get_value"
    SET_VALUE = "set_value"
    UPLOAD_FILE = "upload_file"
    SCREENSHOT = "screenshot"


@dataclass
class ActionParameters:
    """Standardized parameters for browser actions."""

    # Text input parameters
    text: Optional[str] = None

    # Selection parameters
    value: Optional[str] = None
    label: Optional[str] = None
    index: Optional[int] = None

    # Attribute parameters
    attribute_name: Optional[str] = None
    property_name: Optional[str] = None

    # Keyboard parameters
    key: Optional[str] = None
    modifiers: List[str] = field(default_factory=list)

    # Mouse parameters
    button: str = "left"
    click_count: int = 1
    position_x: Optional[int] = None
    position_y: Optional[int] = None

    # Drag and drop parameters
    target_locator: Optional[str] = None

    # File upload parameters
    file_paths: List[str] = field(default_factory=list)

    # Screenshot parameters
    full_page: bool = False
    screenshot_path: Optional[str] = None

    # Timing parameters
    delay_ms: int = 0
    force: bool = False
    no_wait_after: bool = False


@dataclass
class ActionResult:
    """Standardized result from browser actions."""

    success: bool
    value: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    execution_time_ms: float = 0.0
    screenshot_path: Optional[str] = None
    element_info: Optional[Dict[str, Any]] = None


@dataclass
class AriaSnapshot:
    """Standardized ARIA/accessibility tree snapshot."""

    content: str
    selector: Optional[str] = None
    format: str = "yaml"
    success: bool = True
    error: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class PageInfo:
    """Standardized page information."""

    url: str
    title: str
    viewport_width: int = 0
    viewport_height: int = 0
    scroll_x: int = 0
    scroll_y: int = 0
    document_ready_state: str = "complete"


@runtime_checkable
class BrowserLibraryAdapter(Protocol):
    """Protocol defining the browser library adapter interface.

    This is the Anti-Corruption Layer interface that all browser library
    adapters must implement. It provides a unified API for browser operations
    regardless of the underlying library (Browser Library, SeleniumLibrary, etc.).

    Key responsibilities:
    - Capture accessibility snapshots (ARIA tree)
    - Execute standardized browser actions
    - Check element states
    - Retrieve page information
    - Handle library-specific error translation
    """

    @property
    def library_name(self) -> str:
        """Return the name of the underlying library."""
        ...

    @property
    def library_type(self) -> str:
        """Return the library type identifier ('browser' or 'selenium')."""
        ...

    @property
    def is_available(self) -> bool:
        """Check if the underlying library is available and initialized."""
        ...

    def capture_aria_snapshot(
        self,
        selector: Optional[str] = None,
        timeout_ms: int = 30000,
    ) -> AriaSnapshot:
        """Capture accessibility tree snapshot.

        Args:
            selector: Optional CSS/XPath selector to scope the snapshot.
                     If None, captures the entire page.
            timeout_ms: Timeout in milliseconds for the operation.

        Returns:
            AriaSnapshot containing the accessibility tree in YAML format.
        """
        ...

    def execute_action(
        self,
        action_type: ActionType,
        locator: str,
        parameters: ActionParameters,
        timeout_ms: int = 30000,
    ) -> ActionResult:
        """Execute a browser action on an element.

        Args:
            action_type: The type of action to perform.
            locator: Element locator (CSS selector, XPath, or library-specific format).
            parameters: Action-specific parameters.
            timeout_ms: Timeout in milliseconds for the operation.

        Returns:
            ActionResult with success status and any returned value.
        """
        ...

    def check_element_states(
        self,
        locator: str,
        timeout_ms: int = 5000,
    ) -> Set[ElementState]:
        """Get the current states of an element.

        Args:
            locator: Element locator.
            timeout_ms: Timeout in milliseconds for finding the element.

        Returns:
            Set of ElementState values representing the element's current states.
        """
        ...

    def wait_for_element_state(
        self,
        locator: str,
        state: ElementState,
        timeout_ms: int = 30000,
    ) -> ActionResult:
        """Wait for an element to reach a specific state.

        Args:
            locator: Element locator.
            state: The state to wait for.
            timeout_ms: Timeout in milliseconds.

        Returns:
            ActionResult indicating success or failure.
        """
        ...

    def get_page_url(self) -> str:
        """Get the current page URL."""
        ...

    def get_page_title(self) -> str:
        """Get the current page title."""
        ...

    def get_page_info(self) -> PageInfo:
        """Get comprehensive page information."""
        ...

    def get_page_source(self) -> str:
        """Get the page HTML source."""
        ...

    def evaluate_javascript(
        self,
        script: str,
        *args: Any,
    ) -> Any:
        """Execute JavaScript in the page context.

        Args:
            script: JavaScript code to execute.
            *args: Arguments to pass to the script.

        Returns:
            The result of the JavaScript execution.
        """
        ...

    def take_screenshot(
        self,
        path: Optional[str] = None,
        full_page: bool = False,
        selector: Optional[str] = None,
    ) -> Optional[str]:
        """Take a screenshot.

        Args:
            path: Path to save the screenshot. If None, returns base64.
            full_page: Whether to capture the full page or just the viewport.
            selector: Optional selector to capture only a specific element.

        Returns:
            Path to the saved screenshot or base64-encoded image data.
        """
        ...

    def close(self) -> None:
        """Clean up resources and close the adapter."""
        ...


class BaseBrowserAdapter(ABC):
    """Abstract base class providing common functionality for browser adapters.

    Concrete adapters should inherit from this class and implement the
    abstract methods. This class provides:
    - Common error handling patterns
    - Logging infrastructure
    - Default implementations where appropriate
    """

    def __init__(self, robot_instance: Any = None):
        """Initialize the base adapter.

        Args:
            robot_instance: Robot Framework instance for keyword execution.
        """
        self._robot = robot_instance
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    @abstractmethod
    def library_name(self) -> str:
        """Return the name of the underlying library."""
        pass

    @property
    @abstractmethod
    def library_type(self) -> str:
        """Return the library type identifier."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the underlying library is available."""
        pass

    @abstractmethod
    def capture_aria_snapshot(
        self,
        selector: Optional[str] = None,
        timeout_ms: int = 30000,
    ) -> AriaSnapshot:
        """Capture accessibility tree snapshot."""
        pass

    @abstractmethod
    def execute_action(
        self,
        action_type: ActionType,
        locator: str,
        parameters: ActionParameters,
        timeout_ms: int = 30000,
    ) -> ActionResult:
        """Execute a browser action."""
        pass

    @abstractmethod
    def check_element_states(
        self,
        locator: str,
        timeout_ms: int = 5000,
    ) -> Set[ElementState]:
        """Get element states."""
        pass

    @abstractmethod
    def wait_for_element_state(
        self,
        locator: str,
        state: ElementState,
        timeout_ms: int = 30000,
    ) -> ActionResult:
        """Wait for element state."""
        pass

    @abstractmethod
    def get_page_url(self) -> str:
        """Get current page URL."""
        pass

    @abstractmethod
    def get_page_title(self) -> str:
        """Get current page title."""
        pass

    def get_page_info(self) -> PageInfo:
        """Get comprehensive page information.

        Default implementation uses get_page_url and get_page_title.
        Subclasses can override for more detailed information.
        """
        return PageInfo(
            url=self.get_page_url(),
            title=self.get_page_title(),
        )

    @abstractmethod
    def get_page_source(self) -> str:
        """Get page HTML source."""
        pass

    @abstractmethod
    def evaluate_javascript(
        self,
        script: str,
        *args: Any,
    ) -> Any:
        """Execute JavaScript."""
        pass

    @abstractmethod
    def take_screenshot(
        self,
        path: Optional[str] = None,
        full_page: bool = False,
        selector: Optional[str] = None,
    ) -> Optional[str]:
        """Take a screenshot."""
        pass

    def close(self) -> None:
        """Clean up resources.

        Default implementation does nothing. Subclasses can override.
        """
        self._logger.debug(f"Closing {self.library_name} adapter")

    def _execute_keyword(
        self,
        keyword: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a Robot Framework keyword safely.

        Args:
            keyword: The keyword name to execute.
            *args: Positional arguments for the keyword.
            **kwargs: Keyword arguments for the keyword.

        Returns:
            The keyword result.

        Raises:
            RuntimeError: If Robot Framework is not available or keyword fails.
        """
        try:
            from robot.libraries.BuiltIn import BuiltIn
            builtin = BuiltIn()

            # Convert kwargs to RF format if needed
            all_args = list(args)
            for key, value in kwargs.items():
                all_args.append(f"{key}={value}")

            self._logger.debug(f"Executing keyword: {keyword} with args: {all_args}")
            result = builtin.run_keyword(keyword, *all_args)
            self._logger.debug(f"Keyword result: {result}")
            return result

        except ImportError:
            raise RuntimeError("Robot Framework is not available")
        except Exception as e:
            self._logger.error(f"Keyword execution failed: {keyword} - {e}")
            raise

    def _translate_locator(self, locator: str) -> str:
        """Translate a generic locator to library-specific format.

        Default implementation returns the locator unchanged.
        Subclasses should override for library-specific translation.

        Args:
            locator: Generic locator string.

        Returns:
            Library-specific locator string.
        """
        return locator

    def _handle_error(
        self,
        error: Exception,
        operation: str,
        locator: Optional[str] = None,
    ) -> ActionResult:
        """Create an ActionResult from an exception.

        Args:
            error: The exception that occurred.
            operation: Description of the operation that failed.
            locator: Optional locator involved in the operation.

        Returns:
            ActionResult with failure status and error information.
        """
        error_msg = f"{operation} failed"
        if locator:
            error_msg += f" for locator '{locator}'"
        error_msg += f": {str(error)}"

        self._logger.error(error_msg)

        return ActionResult(
            success=False,
            error=error_msg,
            error_type=type(error).__name__,
        )

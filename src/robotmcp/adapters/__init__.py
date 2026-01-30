"""Browser Library Adapters - Anti-Corruption Layer.

This package provides adapters that abstract the differences between
Robot Framework browser automation libraries (Browser Library and
SeleniumLibrary), enabling rf-mcp to work uniformly with either.

The Anti-Corruption Layer pattern ensures:
- Clean separation between rf-mcp core and library-specific code
- Easy testing via mock adapters
- Future extensibility for new browser libraries
- Consistent API regardless of underlying implementation

Key Components:
    BrowserLibraryAdapter: Protocol defining the adapter interface
    PlaywrightBrowserAdapter: Adapter for Browser Library (Playwright)
    SeleniumLibraryAdapter: Adapter for SeleniumLibrary (WebDriver)
    BrowserAdapterFactory: Factory for creating appropriate adapters

Usage:
    from robotmcp.adapters import (
        BrowserAdapterFactory,
        get_adapter_for_session,
        get_best_available_adapter,
    )

    # Create adapter for a session
    adapter = get_adapter_for_session(session, browser_manager)

    # Get best available adapter
    adapter = get_best_available_adapter(browser_manager)

    # Create specific adapter
    adapter = BrowserAdapterFactory.create("Browser")

    # Use adapter for operations
    aria_snapshot = adapter.capture_aria_snapshot()
    result = adapter.execute_action(ActionType.CLICK, "css=#button", ActionParameters())
    states = adapter.check_element_states("css=#element")
"""

from .browser_adapter import (
    # Protocol and base class
    BrowserLibraryAdapter,
    BaseBrowserAdapter,
    # Enums
    ElementState,
    ActionType,
    # Data classes
    ActionParameters,
    ActionResult,
    AriaSnapshot,
    PageInfo,
)

from .playwright_adapter import (
    PlaywrightBrowserAdapter,
    BROWSER_LIBRARY_AVAILABLE,
)

from .selenium_adapter import (
    SeleniumLibraryAdapter,
    SELENIUM_LIBRARY_AVAILABLE,
)

from .adapter_factory import (
    BrowserAdapterFactory,
    BrowserLibraryType,
    get_adapter_for_session,
    get_best_available_adapter,
)

__all__ = [
    # Protocol and base class
    "BrowserLibraryAdapter",
    "BaseBrowserAdapter",
    # Enums
    "ElementState",
    "ActionType",
    "BrowserLibraryType",
    # Data classes
    "ActionParameters",
    "ActionResult",
    "AriaSnapshot",
    "PageInfo",
    # Concrete adapters
    "PlaywrightBrowserAdapter",
    "SeleniumLibraryAdapter",
    # Factory
    "BrowserAdapterFactory",
    # Convenience functions
    "get_adapter_for_session",
    "get_best_available_adapter",
    # Availability flags
    "BROWSER_LIBRARY_AVAILABLE",
    "SELENIUM_LIBRARY_AVAILABLE",
]

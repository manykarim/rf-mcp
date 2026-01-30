"""Browser Adapter Factory.

This module provides a factory for creating appropriate browser library adapters
based on the session configuration, library availability, and user preferences.

The factory implements the Abstract Factory pattern to:
1. Encapsulate adapter creation logic
2. Support automatic library detection
3. Handle fallback scenarios gracefully
4. Provide consistent adapter initialization
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, Optional, TYPE_CHECKING

from .browser_adapter import BrowserLibraryAdapter
from .playwright_adapter import PlaywrightBrowserAdapter, BROWSER_LIBRARY_AVAILABLE
from .selenium_adapter import SeleniumLibraryAdapter, SELENIUM_LIBRARY_AVAILABLE

if TYPE_CHECKING:
    from robotmcp.models.session_models import ExecutionSession
    from robotmcp.components.browser.browser_library_manager import BrowserLibraryManager

logger = logging.getLogger(__name__)


class BrowserLibraryType(Enum):
    """Enumeration of supported browser library types."""

    BROWSER = "Browser"
    SELENIUM = "SeleniumLibrary"
    AUTO = "auto"
    NONE = "none"


class BrowserAdapterFactory:
    """Factory for creating browser library adapters.

    This factory creates the appropriate adapter based on:
    1. Explicit library type specification
    2. Session configuration and preferences
    3. Library availability
    4. Automatic detection from session state

    Usage:
        # Create adapter for specific library
        adapter = BrowserAdapterFactory.create("Browser")

        # Create adapter based on session
        adapter = BrowserAdapterFactory.create_for_session(session, browser_manager)

        # Auto-detect best available adapter
        adapter = BrowserAdapterFactory.create_auto(browser_manager)
    """

    # Cache for singleton adapters (optional, for performance)
    _adapter_cache: Dict[str, BrowserLibraryAdapter] = {}
    _use_cache: bool = False  # Disabled by default for safety

    @classmethod
    def create(
        cls,
        library_type: str,
        robot_instance: Any = None,
        browser_library_manager: Any = None,
        selenium_library: Any = None,
    ) -> BrowserLibraryAdapter:
        """Create an adapter for the specified library type.

        Args:
            library_type: Library type ("Browser", "SeleniumLibrary", "auto", "none").
            robot_instance: Robot Framework instance for keyword execution.
            browser_library_manager: BrowserLibraryManager for Browser Library access.
            selenium_library: SeleniumLibrary instance for SeleniumLibrary access.

        Returns:
            Appropriate BrowserLibraryAdapter implementation.

        Raises:
            ValueError: If the specified library type is not supported.
            RuntimeError: If the specified library is not available.
        """
        # Normalize library type
        lib_type = cls._normalize_library_type(library_type)

        if lib_type == BrowserLibraryType.BROWSER:
            return cls._create_playwright_adapter(
                robot_instance, browser_library_manager
            )

        elif lib_type == BrowserLibraryType.SELENIUM:
            return cls._create_selenium_adapter(robot_instance, selenium_library)

        elif lib_type == BrowserLibraryType.AUTO:
            return cls._create_auto_adapter(
                robot_instance, browser_library_manager, selenium_library
            )

        elif lib_type == BrowserLibraryType.NONE:
            raise ValueError("No browser library type specified")

        else:
            raise ValueError(f"Unsupported library type: {library_type}")

    @classmethod
    def create_for_session(
        cls,
        session: "ExecutionSession",
        browser_library_manager: Optional["BrowserLibraryManager"] = None,
        robot_instance: Any = None,
    ) -> Optional[BrowserLibraryAdapter]:
        """Create an adapter based on session configuration.

        Analyzes the session to determine the appropriate adapter:
        1. Check explicit library preference
        2. Check active library in browser state
        3. Check imported libraries
        4. Fall back to auto-detection

        Args:
            session: The execution session.
            browser_library_manager: BrowserLibraryManager instance.
            robot_instance: Robot Framework instance.

        Returns:
            Appropriate adapter, or None if no browser session.
        """
        # Check explicit library preference
        if session.explicit_library_preference:
            pref = session.explicit_library_preference
            if pref in ("Browser", "SeleniumLibrary"):
                try:
                    return cls.create(
                        pref,
                        robot_instance=robot_instance,
                        browser_library_manager=browser_library_manager,
                    )
                except (ValueError, RuntimeError) as e:
                    logger.warning(
                        f"Could not create adapter for preference '{pref}': {e}"
                    )

        # Check active library in browser state
        if session.browser_state.active_library:
            active = session.browser_state.active_library
            lib_map = {
                "browser": "Browser",
                "selenium": "SeleniumLibrary",
            }
            if active in lib_map:
                try:
                    return cls.create(
                        lib_map[active],
                        robot_instance=robot_instance,
                        browser_library_manager=browser_library_manager,
                    )
                except (ValueError, RuntimeError) as e:
                    logger.warning(
                        f"Could not create adapter for active library '{active}': {e}"
                    )

        # Check imported libraries
        imported_libs = session.imported_libraries
        if "Browser" in imported_libs:
            try:
                return cls.create(
                    "Browser",
                    robot_instance=robot_instance,
                    browser_library_manager=browser_library_manager,
                )
            except (ValueError, RuntimeError) as e:
                logger.debug(f"Browser Library adapter creation failed: {e}")

        if "SeleniumLibrary" in imported_libs:
            try:
                return cls.create(
                    "SeleniumLibrary",
                    robot_instance=robot_instance,
                    browser_library_manager=browser_library_manager,
                )
            except (ValueError, RuntimeError) as e:
                logger.debug(f"SeleniumLibrary adapter creation failed: {e}")

        # Check if this is a browser session at all
        if not session.is_browser_session():
            logger.debug("Session is not a browser session, no adapter needed")
            return None

        # Fall back to auto-detection
        return cls._create_auto_adapter(
            robot_instance, browser_library_manager, None
        )

    @classmethod
    def create_auto(
        cls,
        browser_library_manager: Optional["BrowserLibraryManager"] = None,
        robot_instance: Any = None,
    ) -> BrowserLibraryAdapter:
        """Create the best available adapter automatically.

        Preference order:
        1. Browser Library (if available) - modern, feature-rich
        2. SeleniumLibrary (if available) - traditional, broad support

        Args:
            browser_library_manager: BrowserLibraryManager instance.
            robot_instance: Robot Framework instance.

        Returns:
            The best available adapter.

        Raises:
            RuntimeError: If no browser libraries are available.
        """
        return cls._create_auto_adapter(robot_instance, browser_library_manager, None)

    @classmethod
    def get_available_libraries(cls) -> Dict[str, bool]:
        """Get availability status of browser libraries.

        Returns:
            Dictionary mapping library names to availability status.
        """
        return {
            "Browser": BROWSER_LIBRARY_AVAILABLE,
            "SeleniumLibrary": SELENIUM_LIBRARY_AVAILABLE,
        }

    @classmethod
    def get_preferred_library(
        cls,
        browser_library_manager: Optional["BrowserLibraryManager"] = None,
    ) -> Optional[str]:
        """Get the preferred/recommended library based on availability.

        Args:
            browser_library_manager: BrowserLibraryManager for configuration access.

        Returns:
            Library name ("Browser" or "SeleniumLibrary") or None if none available.
        """
        # Check manager preference first
        if browser_library_manager:
            pref = browser_library_manager.get_preferred_library()
            if pref and pref != "none":
                return "Browser" if pref == "browser" else "SeleniumLibrary"

        # Default preference: Browser > Selenium
        if BROWSER_LIBRARY_AVAILABLE:
            return "Browser"
        elif SELENIUM_LIBRARY_AVAILABLE:
            return "SeleniumLibrary"
        return None

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the adapter cache."""
        cls._adapter_cache.clear()
        logger.debug("Adapter cache cleared")

    @classmethod
    def enable_caching(cls, enable: bool = True) -> None:
        """Enable or disable adapter caching.

        Warning: Enable with caution. Cached adapters may hold stale references.

        Args:
            enable: Whether to enable caching.
        """
        cls._use_cache = enable
        if not enable:
            cls.clear_cache()
        logger.debug(f"Adapter caching {'enabled' if enable else 'disabled'}")

    # Private helper methods

    @classmethod
    def _normalize_library_type(cls, library_type: str) -> BrowserLibraryType:
        """Normalize library type string to enum.

        Args:
            library_type: Library type string.

        Returns:
            Corresponding BrowserLibraryType enum value.
        """
        if not library_type:
            return BrowserLibraryType.NONE

        type_lower = library_type.lower().strip()

        type_mapping = {
            "browser": BrowserLibraryType.BROWSER,
            "browserlibrary": BrowserLibraryType.BROWSER,
            "playwright": BrowserLibraryType.BROWSER,
            "selenium": BrowserLibraryType.SELENIUM,
            "seleniumlibrary": BrowserLibraryType.SELENIUM,
            "webdriver": BrowserLibraryType.SELENIUM,
            "auto": BrowserLibraryType.AUTO,
            "automatic": BrowserLibraryType.AUTO,
            "none": BrowserLibraryType.NONE,
            "": BrowserLibraryType.NONE,
        }

        return type_mapping.get(type_lower, BrowserLibraryType.NONE)

    @classmethod
    def _create_playwright_adapter(
        cls,
        robot_instance: Any,
        browser_library_manager: Any,
    ) -> PlaywrightBrowserAdapter:
        """Create Playwright/Browser Library adapter.

        Args:
            robot_instance: Robot Framework instance.
            browser_library_manager: BrowserLibraryManager instance.

        Returns:
            PlaywrightBrowserAdapter instance.

        Raises:
            RuntimeError: If Browser Library is not available.
        """
        if not BROWSER_LIBRARY_AVAILABLE:
            raise RuntimeError(
                "Browser Library is not available. "
                "Install with: pip install robotframework-browser && rfbrowser init"
            )

        # Check cache if enabled
        cache_key = "playwright"
        if cls._use_cache and cache_key in cls._adapter_cache:
            logger.debug("Returning cached Playwright adapter")
            return cls._adapter_cache[cache_key]  # type: ignore

        adapter = PlaywrightBrowserAdapter(
            robot_instance=robot_instance,
            browser_library_manager=browser_library_manager,
        )

        if not adapter.is_available:
            raise RuntimeError(
                "Browser Library initialization failed. "
                "Ensure Playwright is installed: rfbrowser init"
            )

        if cls._use_cache:
            cls._adapter_cache[cache_key] = adapter

        logger.info("Created Playwright/Browser Library adapter")
        return adapter

    @classmethod
    def _create_selenium_adapter(
        cls,
        robot_instance: Any,
        selenium_library: Any,
    ) -> SeleniumLibraryAdapter:
        """Create SeleniumLibrary adapter.

        Args:
            robot_instance: Robot Framework instance.
            selenium_library: SeleniumLibrary instance.

        Returns:
            SeleniumLibraryAdapter instance.

        Raises:
            RuntimeError: If SeleniumLibrary is not available.
        """
        if not SELENIUM_LIBRARY_AVAILABLE:
            raise RuntimeError(
                "SeleniumLibrary is not available. "
                "Install with: pip install robotframework-seleniumlibrary"
            )

        # Check cache if enabled
        cache_key = "selenium"
        if cls._use_cache and cache_key in cls._adapter_cache:
            logger.debug("Returning cached Selenium adapter")
            return cls._adapter_cache[cache_key]  # type: ignore

        adapter = SeleniumLibraryAdapter(
            robot_instance=robot_instance,
            selenium_library=selenium_library,
        )

        if not adapter.is_available:
            raise RuntimeError(
                "SeleniumLibrary initialization failed. "
                "Ensure WebDriver is properly configured."
            )

        if cls._use_cache:
            cls._adapter_cache[cache_key] = adapter

        logger.info("Created SeleniumLibrary adapter")
        return adapter

    @classmethod
    def _create_auto_adapter(
        cls,
        robot_instance: Any,
        browser_library_manager: Any,
        selenium_library: Any,
    ) -> BrowserLibraryAdapter:
        """Create the best available adapter automatically.

        Tries Browser Library first, then falls back to SeleniumLibrary.

        Args:
            robot_instance: Robot Framework instance.
            browser_library_manager: BrowserLibraryManager instance.
            selenium_library: SeleniumLibrary instance.

        Returns:
            Best available adapter.

        Raises:
            RuntimeError: If no browser libraries are available.
        """
        # Try Browser Library first (preferred for modern features)
        if BROWSER_LIBRARY_AVAILABLE:
            try:
                return cls._create_playwright_adapter(
                    robot_instance, browser_library_manager
                )
            except RuntimeError as e:
                logger.warning(f"Browser Library unavailable: {e}")

        # Fall back to SeleniumLibrary
        if SELENIUM_LIBRARY_AVAILABLE:
            try:
                return cls._create_selenium_adapter(robot_instance, selenium_library)
            except RuntimeError as e:
                logger.warning(f"SeleniumLibrary unavailable: {e}")

        raise RuntimeError(
            "No browser automation libraries available. "
            "Install either:\n"
            "  - Browser Library: pip install robotframework-browser && rfbrowser init\n"
            "  - SeleniumLibrary: pip install robotframework-seleniumlibrary"
        )


def get_adapter_for_session(
    session: "ExecutionSession",
    browser_library_manager: Optional["BrowserLibraryManager"] = None,
) -> Optional[BrowserLibraryAdapter]:
    """Convenience function to get adapter for a session.

    Args:
        session: The execution session.
        browser_library_manager: BrowserLibraryManager instance.

    Returns:
        Appropriate adapter or None.
    """
    return BrowserAdapterFactory.create_for_session(
        session, browser_library_manager
    )


def get_best_available_adapter(
    browser_library_manager: Optional["BrowserLibraryManager"] = None,
) -> BrowserLibraryAdapter:
    """Convenience function to get the best available adapter.

    Args:
        browser_library_manager: BrowserLibraryManager instance.

    Returns:
        Best available adapter.
    """
    return BrowserAdapterFactory.create_auto(browser_library_manager)

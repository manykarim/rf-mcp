"""Browser Library management service."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from robotmcp.config.library_registry import get_library_install_hint
from robotmcp.models.session_models import ExecutionSession
from robotmcp.models.config_models import ExecutionConfig
from robotmcp.models.browser_models import BrowserState

logger = logging.getLogger(__name__)

# Check for Browser Library availability
try:
    from Browser import Browser as BrowserLibrary
    BROWSER_LIBRARY_AVAILABLE = True
except ImportError:
    BrowserLibrary = None
    BROWSER_LIBRARY_AVAILABLE = False
    browser_hint = get_library_install_hint("Browser")
    message = "Browser Library not available"
    if browser_hint:
        message = f"{message}. {browser_hint}"
    logger.warning(message)

# Check for SeleniumLibrary availability
try:
    from SeleniumLibrary import SeleniumLibrary
    SELENIUM_LIBRARY_AVAILABLE = True
except ImportError:
    SeleniumLibrary = None
    SELENIUM_LIBRARY_AVAILABLE = False
    selenium_hint = get_library_install_hint("SeleniumLibrary")
    message = "SeleniumLibrary not available"
    if selenium_hint:
        message = f"{message}. {selenium_hint}"
    logger.warning(message)


class BrowserLibraryManager:
    """Manages browser library initialization and selection."""
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.browser_lib: Optional[Any] = None
        self.selenium_lib: Optional[Any] = None
        self._initialize_libraries()
    
    def _initialize_libraries(self) -> None:
        """Initialize available browser libraries."""
        self._initialize_browser_library()
        self._initialize_selenium_library()
    
    def _initialize_browser_library(self) -> None:
        """Initialize Browser Library instance."""
        try:
            if not BROWSER_LIBRARY_AVAILABLE:
                logger.info("Browser Library not available - using simulation mode")
                self.browser_lib = None
                return
            
            # Initialize Browser Library instance with safer defaults
            self.browser_lib = BrowserLibrary()
            try:
                # Disable interactive pause prompts on failures to prevent blocking execution
                setattr(self.browser_lib, "pause_on_failure", False)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"Browser pause_on_failure override failed: {exc}")
            logger.info("Browser Library initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Browser Library: {e}")
            self.browser_lib = None

    def _initialize_selenium_library(self) -> None:
        """Initialize SeleniumLibrary instance."""
        try:
            if not SELENIUM_LIBRARY_AVAILABLE:
                logger.info("SeleniumLibrary not available - Browser Library will be preferred")
                self.selenium_lib = None
                return
            
            # Initialize SeleniumLibrary instance
            self.selenium_lib = SeleniumLibrary()
            logger.info("SeleniumLibrary initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing SeleniumLibrary: {e}")
            self.selenium_lib = None

    def get_active_browser_library(self, session: Optional[ExecutionSession] = None) -> Tuple[Optional[Any], str]:
        """
        Determine which browser library is active for a session.
        
        Args:
            session: ExecutionSession to check, or None for default behavior
            
        Returns:
            tuple: (library_instance, library_type) where library_type is "browser", "selenium", or "none"
        """
        if session is None:
            # No session exists, prefer Browser Library if available
            if self.browser_lib:
                return self.browser_lib, "browser"
            elif self.selenium_lib:
                return self.selenium_lib, "selenium"
            else:
                return None, "none"
        
        browser_state = session.browser_state
        
        # Check session's active library preference
        if browser_state.active_library == "browser" and self.browser_lib:
            return self.browser_lib, "browser"
        elif browser_state.active_library == "selenium" and self.selenium_lib:
            return self.selenium_lib, "selenium"
        
        # Auto-detect based on session state
        if browser_state.browser_id or browser_state.context_id or browser_state.page_id:
            # Browser Library session
            if self.browser_lib:
                browser_state.active_library = "browser"
                return self.browser_lib, "browser"
        
        if browser_state.driver_instance or browser_state.selenium_session_id:
            # SeleniumLibrary session
            if self.selenium_lib:
                browser_state.active_library = "selenium"
                return self.selenium_lib, "selenium"
        
        # Default preference: Browser Library > SeleniumLibrary
        if self.browser_lib:
            return self.browser_lib, "browser"
        elif self.selenium_lib:
            return self.selenium_lib, "selenium"
        else:
            return None, "none"

    def detect_library_from_keyword(self, keyword: str, arguments: List[str]) -> str:
        """
        Detect which library a keyword belongs to based on keyword name and arguments.
        
        Args:
            keyword: The keyword name
            arguments: The keyword arguments
            
        Returns:
            str: Library type ("browser", "selenium", "builtin", or "unknown")
        """
        from robotmcp.utils.library_detector import detect_library_type_from_keyword
        return detect_library_type_from_keyword(keyword)

    def check_library_requirements(self, required_libraries: List[str]) -> Dict[str, Any]:
        """
        Check if required libraries are available and properly initialized.
        
        Args:
            required_libraries: List of required library names
            
        Returns:
            dict: Status information about library availability
        """
        status = {
            "available_libraries": [],
            "missing_libraries": [],
            "initialization_errors": []
        }
        
        for lib_name in required_libraries:
            if lib_name.lower() in ["browser", "browserlibrary"]:
                if BROWSER_LIBRARY_AVAILABLE and self.browser_lib:
                    status["available_libraries"].append("Browser")
                elif BROWSER_LIBRARY_AVAILABLE and not self.browser_lib:
                    status["initialization_errors"].append("Browser Library available but not initialized")
                else:
                    status["missing_libraries"].append("Browser Library")
                    
            elif lib_name.lower() in ["selenium", "seleniumlibrary"]:
                if SELENIUM_LIBRARY_AVAILABLE and self.selenium_lib:
                    status["available_libraries"].append("SeleniumLibrary")
                elif SELENIUM_LIBRARY_AVAILABLE and not self.selenium_lib:
                    status["initialization_errors"].append("SeleniumLibrary available but not initialized")
                else:
                    status["missing_libraries"].append("SeleniumLibrary")
        
        return status

    def get_library_capabilities(self) -> Dict[str, Any]:
        """
        Get information about available library capabilities.
        
        Returns:
            dict: Capabilities and feature support for each library
        """
        capabilities = {
            "browser": {
                "available": BROWSER_LIBRARY_AVAILABLE and self.browser_lib is not None,
                "features": {
                    "multi_browser": True,
                    "playwright_backend": True,
                    "async_support": True,
                    "context_isolation": True,
                    "network_interception": True,
                    "screenshots": True,
                    "pdf_generation": True,
                    "mobile_emulation": True
                } if BROWSER_LIBRARY_AVAILABLE else {}
            },
            "selenium": {
                "available": SELENIUM_LIBRARY_AVAILABLE and self.selenium_lib is not None,
                "features": {
                    "multi_browser": True,
                    "webdriver_backend": True,
                    "async_support": False,
                    "context_isolation": False,
                    "network_interception": False,
                    "screenshots": True,
                    "pdf_generation": False,
                    "mobile_emulation": True
                } if SELENIUM_LIBRARY_AVAILABLE else {}
            }
        }
        
        return capabilities

    def get_preferred_library(self) -> str:
        """
        Get the preferred library based on configuration and availability.
        
        Returns:
            str: Preferred library type ("browser", "selenium", or "none")
        """
        if self.config.PREFERRED_WEB_LIBRARY == "Browser" and self.browser_lib:
            return "browser"
        elif self.config.PREFERRED_WEB_LIBRARY == "SeleniumLibrary" and self.selenium_lib:
            return "selenium"
        
        # Fallback to availability
        if self.browser_lib:
            return "browser"
        elif self.selenium_lib:
            return "selenium"
        else:
            return "none"

    @staticmethod
    def normalize_library_name(name: Optional[str]) -> Optional[str]:
        """Map a library/preference name to a logical type ('browser' or 'selenium')."""
        if not name:
            return None
        n = str(name).lower()
        if "selenium" in n:
            return "selenium"
        if "browser" in n or "playwright" in n:
            return "browser"
        return None

    def set_active_library(self, session: ExecutionSession, library_type: str) -> bool:
        """
        Set the active library for a session.
        
        Args:
            session: The session to update
            library_type: Library type ("browser" or "selenium")
            
        Returns:
            bool: True if successfully set, False otherwise
        """
        pref_type = self.normalize_library_name(
            getattr(session, "explicit_library_preference", None)
        )
        if pref_type and pref_type != library_type:
            logger.debug(
                f"Skipping activation of '{library_type}' because explicit preference is '{pref_type}'"
            )
            return False

        if library_type == "browser" and self.browser_lib:
            session.browser_state.active_library = "browser"
            # Only import if not already present to avoid overwriting session configuration
            if "Browser" not in session.imported_libraries:
                session.import_library("Browser", force=True)
            return True
        elif library_type == "selenium" and self.selenium_lib:
            session.browser_state.active_library = "selenium"
            # Only import if not already present to avoid overwriting session configuration
            if "SeleniumLibrary" not in session.imported_libraries:
                session.import_library("SeleniumLibrary", force=True)
            return True
        else:
            logger.warning(f"Cannot set active library to '{library_type}' - not available or initialized")
            return False

    def reset_libraries(self) -> None:
        """Reset and reinitialize all libraries."""
        logger.info("Resetting browser libraries")
        self.browser_lib = None
        self.selenium_lib = None
        self._initialize_libraries()

    def cleanup(self) -> None:
        """Clean up library resources."""
        if self.browser_lib:
            try:
                # Browser Library cleanup if needed
                pass
            except Exception as e:
                logger.error(f"Error cleaning up Browser Library: {e}")
                
        if self.selenium_lib:
            try:
                # SeleniumLibrary cleanup if needed
                pass
            except Exception as e:
                logger.error(f"Error cleaning up SeleniumLibrary: {e}")
        
        self.browser_lib = None
        self.selenium_lib = None

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of browser library manager.
        
        Returns:
            dict: Status information
        """
        return {
            "browser_library_available": BROWSER_LIBRARY_AVAILABLE,
            "browser_library_initialized": self.browser_lib is not None,
            "selenium_library_available": SELENIUM_LIBRARY_AVAILABLE,
            "selenium_library_initialized": self.selenium_lib is not None,
            "preferred_library": self.config.PREFERRED_WEB_LIBRARY,
            "auto_selection_enabled": self.config.AUTO_LIBRARY_SELECTION
        }

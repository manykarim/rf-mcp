"""Execution Engine for running Robot Framework keywords using the API."""

import logging
import uuid
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import traceback

# Import the library availability checker
from robotmcp.utils.library_checker import LibraryAvailabilityChecker, check_and_suggest_libraries

# Import dynamic keyword discovery
from robotmcp.utils.dynamic_keywords import get_keyword_discovery
from robotmcp.utils.rf_libdoc_integration import get_rf_doc_storage

# Import hybrid execution system
from .keyword_overrides import KeywordOverrideRegistry, DynamicExecutionHandler, setup_default_overrides

# Import shared library detection utility
from robotmcp.utils.library_detector import detect_library_type_from_keyword

try:
    from robot.api import TestSuite
    from robot.running.model import TestCase, Keyword
    from robot.conf import RobotSettings
    from robot.libraries.BuiltIn import BuiltIn
    from robot.api import get_model
    from robot.result import ExecutionResult
    ROBOT_AVAILABLE = True
except ImportError:
    TestSuite = None
    TestCase = None
    Keyword = None
    RobotSettings = None
    BuiltIn = None
    get_model = None
    ExecutionResult = None
    ROBOT_AVAILABLE = False

# Browser Library imports
try:
    from Browser import Browser as BrowserLibrary
    from Browser.utils import logger as browser_logger
    from Browser.utils.data_types import SupportedBrowsers
    import datetime as dt
    BROWSER_LIBRARY_AVAILABLE = True
except ImportError:
    BrowserLibrary = None
    browser_logger = None
    SupportedBrowsers = None
    dt = None
    BROWSER_LIBRARY_AVAILABLE = False

# SeleniumLibrary imports  
try:
    from SeleniumLibrary import SeleniumLibrary
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    SELENIUM_LIBRARY_AVAILABLE = True
except ImportError:
    SeleniumLibrary = None
    webdriver = None
    By = None
    SELENIUM_LIBRARY_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ExecutionStep:
    """Represents a single execution step."""
    step_id: str
    keyword: str
    arguments: List[str]
    status: str = "pending"  # pending, running, pass, fail
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Any] = None
    variables: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BrowserState:
    """Represents Browser Library and SeleniumLibrary state."""
    # Common browser state
    browser_type: Optional[str] = None
    current_url: Optional[str] = None
    page_title: Optional[str] = None
    viewport: Dict[str, int] = field(default_factory=lambda: {"width": 1280, "height": 720})
    page_source: Optional[str] = None
    cookies: List[Dict[str, Any]] = field(default_factory=list)
    local_storage: Dict[str, str] = field(default_factory=dict)
    
    # Browser Library specific state
    browser_id: Optional[str] = None
    context_id: Optional[str] = None
    page_id: Optional[str] = None
    
    # SeleniumLibrary specific state
    driver_instance: Optional[Any] = None
    selenium_session_id: Optional[str] = None
    
    # Active library indicator ("browser" or "selenium" or None)
    active_library: Optional[str] = None
    session_storage: Dict[str, str] = field(default_factory=dict)
    page_elements: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ExecutionSession:
    """Manages execution state for a test session."""
    session_id: str
    suite: Optional[Any] = None
    steps: List[ExecutionStep] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    imported_libraries: List[str] = field(default_factory=list)
    current_browser: Optional[str] = None
    browser_state: BrowserState = field(default_factory=BrowserState)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

class ExecutionEngine:
    """Executes Robot Framework keywords and manages test sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, ExecutionSession] = {}
        self.builtin = None
        self.browser_lib = None
        self.selenium_lib = None
        
        # Initialize library checker
        self.library_checker = LibraryAvailabilityChecker()
        
        # Initialize dynamic keyword discovery
        self.keyword_discovery = get_keyword_discovery()
        self.rf_doc_storage = get_rf_doc_storage()
        
        # Initialize hybrid execution system
        self.override_registry = KeywordOverrideRegistry()
        self.dynamic_handler = DynamicExecutionHandler(self)
        setup_default_overrides(self.override_registry, self)
        
        # Initialize Robot Framework
        self._initialize_robot_framework()
        
        # Initialize Browser Library (kept for backward compatibility)
        self._initialize_browser_library()
        
        # Initialize SeleniumLibrary
        self._initialize_selenium_library()

    def __del__(self):
        """Cleanup on destruction."""
        try:
            # Attempt to clean up any remaining browser resources
            if hasattr(self, 'sessions') and self.sessions:
                logger.info("ExecutionEngine destructor: cleaning up remaining sessions")
                # Note: Can't use await in destructor, so we just log the need for cleanup
                for session_id, session in self.sessions.items():
                    if session.browser_state.browser_id:
                        logger.warning(f"Session {session_id} still has active browser {session.browser_state.browser_id}")
        except Exception as e:
            logger.error(f"Error in ExecutionEngine destructor: {e}")
    
    def _initialize_robot_framework(self) -> None:
        """Initialize Robot Framework components."""
        try:
            if not ROBOT_AVAILABLE:
                logger.warning("Robot Framework not available - using simulation mode")
                self.settings = None
                self.builtin = None
                return
            
            # Set up basic Robot Framework configuration
            self.settings = RobotSettings()
            
            # Initialize BuiltIn library for variable access
            self.builtin = BuiltIn()
            
            logger.info("Robot Framework execution engine initialized")
            
        except Exception as e:
            logger.error(f"Error initializing Robot Framework: {e}")
            self.builtin = None

    def _initialize_browser_library(self) -> None:
        """Initialize Browser Library instance."""
        try:
            if not BROWSER_LIBRARY_AVAILABLE:
                logger.info("Browser Library not available - using simulation mode")
                self.browser_lib = None
                return
            
            # Initialize Browser Library instance
            self.browser_lib = BrowserLibrary()
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

    def _get_active_browser_library(self, session_id: str = "default") -> tuple[Optional[Any], str]:
        """
        Determine which browser library is active for a session using session-scoped instances.
        
        Returns:
            tuple: (library_instance, library_type) where library_type is "browser", "selenium", or "none"
        """
        if session_id not in self.sessions:
            # No session exists, prefer Browser Library if available
            if self.browser_lib:
                return self.browser_lib, "browser"
            elif self.selenium_lib:
                return self.selenium_lib, "selenium"
            else:
                return None, "none"
        
        session = self.sessions[session_id]
        browser_state = session.browser_state
        
        # Use global instances - simpler and more reliable for MCP usage
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

    def _detect_library_from_keyword(self, keyword: str, arguments: List[str]) -> str:
        """
        Detect which library should be used based on the keyword being executed.
        
        Returns:
            str: "browser", "selenium", or "auto"
        """
        return detect_library_type_from_keyword(keyword, arguments)

    def _get_page_source_unified(self, session_id: str = "default") -> Optional[str]:
        """
        Get page source using the appropriate library (Browser Library or SeleniumLibrary).
        
        Args:
            session_id: Session ID to get page source for
            
        Returns:
            str: Page source HTML, or None if not available
        """
        try:
            active_lib, lib_type = self._get_active_browser_library(session_id)
            
            if not active_lib:
                logger.debug(f"No browser library available for session {session_id}")
                return None
            
            if lib_type == "browser":
                # Try to get page source from the dynamic keyword discovery's library instance
                try:
                    if 'Browser' in self.keyword_discovery.libraries:
                        # Use the library instance that actually executed the keywords
                        browser_lib_info = self.keyword_discovery.libraries['Browser']
                        rf_browser_lib = browser_lib_info.instance
                        if rf_browser_lib:
                            logger.debug("Using Browser Library from dynamic keyword discovery")
                            return rf_browser_lib.get_page_source()
                except Exception as context_error:
                    logger.debug(f"Could not access Browser Library from keyword discovery: {context_error}")
                
                # Fall back to our global instance
                logger.debug("Using global Browser Library instance")
                return active_lib.get_page_source()
            
            elif lib_type == "selenium":
                # Try to get page source from the dynamic keyword discovery's library instance
                try:
                    if 'SeleniumLibrary' in self.keyword_discovery.libraries:
                        # Use the library instance that actually executed the keywords
                        selenium_lib_info = self.keyword_discovery.libraries['SeleniumLibrary']
                        rf_selenium_lib = selenium_lib_info.instance
                        if rf_selenium_lib:
                            logger.debug("Using SeleniumLibrary from dynamic keyword discovery")
                            return rf_selenium_lib.get_source()
                except Exception as context_error:
                    logger.debug(f"Could not access SeleniumLibrary from keyword discovery: {context_error}")
                
                # Fall back to our global instance
                logger.debug("Using global SeleniumLibrary instance")
                return active_lib.get_source()
            
            else:
                logger.debug(f"Unknown library type: {lib_type}")
                return None
                
        except Exception as e:
            logger.debug(f"Error getting page source with unified method: {e}")
            return None

    async def _capture_page_source_after_keyword(self, session: ExecutionSession, keyword_name: str) -> Optional[str]:
        """Capture page source after a DOM-changing keyword execution."""
        try:
            # Get page source using the unified method
            page_source = self._get_page_source_unified(session.session_id)
            
            if page_source:
                # Store it in session state for future reference
                session.browser_state.page_source = page_source
                logger.debug(f"Captured page source after {keyword_name}: {len(page_source)} characters")
                return page_source
            else:
                logger.debug(f"Could not capture page source after {keyword_name}")
                return None
                
        except Exception as e:
            logger.debug(f"Error capturing page source after {keyword_name}: {e}")
            return None

    async def _execute_selenium_keyword_directly(self, session: ExecutionSession, keyword_name: str, args: List[str]) -> Dict[str, Any]:
        """Execute SeleniumLibrary keyword directly using the selenium_lib instance."""
        try:
            # Mark session as using SeleniumLibrary
            session.browser_state.active_library = "selenium"
            
            logger.info(f"Executing SeleniumLibrary keyword directly: {keyword_name}")
            
            # Get the method name from the keyword
            method_name = keyword_name.lower().replace(' ', '_')
            
            if not hasattr(self.selenium_lib, method_name):
                return {
                    "success": False,
                    "error": f"SeleniumLibrary method '{method_name}' not found",
                    "output": ""
                }
            
            method = getattr(self.selenium_lib, method_name)
            
            # Execute the method
            if args:
                result = method(*args)
            else:
                result = method()
            
            # Handle specific SeleniumLibrary keywords
            if keyword_name.lower() == "get source":
                # Store page source in session state for unified access
                if result:
                    session.browser_state.page_source = result
                    logger.info(f"Page source retrieved via SeleniumLibrary: {len(result)} characters")
            
            elif keyword_name.lower() == "open browser":
                # Track SeleniumLibrary session
                try:
                    if hasattr(self.selenium_lib, 'driver') and self.selenium_lib.driver:
                        session.browser_state.driver_instance = self.selenium_lib.driver
                        session.browser_state.selenium_session_id = self.selenium_lib.driver.session_id
                        logger.info(f"SeleniumLibrary browser session tracked: {session.browser_state.selenium_session_id}")
                except Exception as e:
                    logger.debug(f"Could not track SeleniumLibrary session: {e}")
            
            result_data = {
                "success": True,
                "output": str(result) if result is not None else f"Executed SeleniumLibrary.{keyword_name}",
                "result": result,
                "keyword_info": {
                    "name": keyword_name,
                    "library": "SeleniumLibrary",
                    "method": method_name
                }
            }
            
            # Auto-capture page source for successful DOM-changing SeleniumLibrary keywords
            if self.keyword_discovery.is_dom_changing_keyword(keyword_name):
                page_source = await self._capture_page_source_after_keyword(session, keyword_name)
                if page_source:
                    result_data["page_source"] = page_source
                    result_data["keyword_info"]["auto_captured_dom"] = True
                    logger.info(f"Auto-captured page source after SeleniumLibrary keyword: {keyword_name}")
            
            return result_data
            
        except Exception as e:
            return {
                "success": False,
                "error": f"SeleniumLibrary {keyword_name} execution error: {str(e)}",
                "output": ""
            }
    
    def check_library_requirements(self, required_libraries: List[str]) -> Dict[str, Any]:
        """
        Check if required libraries are available and suggest installations if needed.
        
        Args:
            required_libraries: List of library names to check
            
        Returns:
            Dict containing availability status and installation suggestions
        """
        available, suggestions = check_and_suggest_libraries(required_libraries)
        
        return {
            "available_libraries": available,
            "missing_libraries": [lib for lib in required_libraries if lib not in available],
            "installation_suggestions": suggestions,
            "all_available": len(suggestions) == 0
        }
    
    def get_installation_status(self, library_name: str) -> Dict[str, Any]:
        """
        Get detailed installation status for a specific library.
        
        Args:
            library_name: Name of the library to check
            
        Returns:
            Dict with detailed status information
        """
        from robotmcp.utils.library_checker import COMMON_ROBOT_LIBRARIES
        
        if library_name in COMMON_ROBOT_LIBRARIES:
            lib_info = COMMON_ROBOT_LIBRARIES[library_name]
            package_name = lib_info['package']
            import_name = lib_info['import']
            
            # Check import availability
            import_available = self.library_checker.is_library_available(package_name, import_name)
            
            # Check pip package status
            pip_installed = self.library_checker.check_pip_package_installed(package_name)
            
            result = {
                "library_name": library_name,
                "package_name": package_name,
                "import_name": import_name,
                "import_available": import_available,
                "pip_installed": pip_installed,
                "description": lib_info.get('description', ''),
                "status": "available" if import_available else "missing"
            }
            
            if not import_available:
                result["installation_command"] = ' '.join(
                    self.library_checker.get_installation_command(lib_info)
                )
                result["suggestion"] = self.library_checker.suggest_installation(lib_info)
                
                if lib_info.get('post_install'):
                    result["post_install_command"] = lib_info['post_install']
            
            return result
        else:
            # Handle unknown libraries
            import_available = self.library_checker.is_robot_library_available(library_name)
            return {
                "library_name": library_name,
                "import_available": import_available,
                "status": "available" if import_available else "unknown",
                "suggestion": f"Library '{library_name}' not found in common libraries. Manual check required."
            }

    def _convert_locator_for_library(self, locator: str, target_library: str) -> str:
        """Convert locator format between different libraries.
        
        SeleniumLibrary supports:
        - id, name, identifier (default strategies)
        - class, tag, xpath, css, dom, link, partial link, data, jquery (explicit)
        - Implicit xpath detection (starts with //)
        
        Browser Library supports:
        - CSS (default)
        - xpath (auto-detected if starts with // or ..)
        - text (finds by text content)
        - id (CSS shorthand)
        - Cascaded selectors with >>
        - Shadow DOM piercing
        """
        if not locator:
            return locator
            
        # Convert SeleniumLibrary locators to Browser Library format
        if target_library == "Browser":
            # Handle explicit strategy syntax (strategy=value or strategy:value)
            if "=" in locator and self._is_explicit_strategy(locator):
                strategy, value = locator.split("=", 1) if "=" in locator else locator.split(":", 1)
                strategy = strategy.lower().strip()
                
                if strategy == "id":
                    # id=element -> id=element (Browser supports both #element and id=element)
                    return f"id={value}"
                elif strategy == "css":
                    # css=selector -> selector (CSS is default in Browser)
                    return value
                elif strategy == "xpath":
                    # xpath=//path -> //path (Browser auto-detects xpath)
                    return value
                elif strategy == "name":
                    # name=attr -> [name="attr"]
                    return f'[name="{value}"]'
                elif strategy == "class":
                    # class=classname -> .classname
                    return f".{value}"
                elif strategy == "tag":
                    # tag=div -> div
                    return value
                elif strategy == "link" or strategy == "partial link":
                    # link=text -> text=text (Browser uses text selectors)
                    return f'text={value}'
                elif strategy == "data":
                    # data=value -> [data-*="value"] - need specific data attribute
                    return f'[data-testid="{value}"]'  # Common convention
                elif strategy == "jquery":
                    # jquery selectors -> CSS equivalent (best effort)
                    return self._convert_jquery_to_css(value)
                elif strategy == "dom":
                    # dom expressions can't be directly converted
                    logger.warning(f"DOM locator '{locator}' cannot be converted to Browser Library format")
                    return locator
                elif strategy == "identifier":
                    # identifier -> try id first, fallback to name
                    return f'id={value}, [name="{value}"]'
            
            # Handle implicit xpath (starts with // or ..)
            elif locator.startswith("//") or locator.startswith(".."):
                # XPath is auto-detected in Browser Library
                return locator
            
            # Handle CSS shortcuts that might need conversion
            elif locator.startswith("#") or locator.startswith(".") or locator.startswith("["):
                # Already in CSS format, keep as-is
                return locator
                
        # Convert Browser Library locators to SeleniumLibrary format  
        elif target_library == "SeleniumLibrary":
            # Handle text selectors
            if locator.startswith("text="):
                # text=content -> xpath=//*[contains(text(),'content')]
                text_content = locator[5:]
                return f'xpath=//*[contains(text(),"{text_content}")]'
            
            # Handle CSS shortcuts
            elif locator.startswith("#"):
                # #element -> id=element
                return f"id={locator[1:]}"
            elif locator.startswith(".") and not locator.startswith("./") and not locator.startswith(".."):
                # .classname -> class=classname  
                return f"class={locator[1:]}"
            
            # Handle cascaded selectors (>> syntax)
            elif ">>" in locator:
                # Convert cascaded to xpath (complex conversion)
                return self._convert_cascaded_to_xpath(locator)
            
            # Handle implicit xpath
            elif locator.startswith("//") or locator.startswith(".."):
                # Already xpath, add explicit strategy
                return f"xpath={locator}"
            
            # Handle CSS attribute selectors
            elif locator.startswith("[") and locator.endswith("]"):
                # [attribute="value"] -> css=[attribute="value"]
                return f"css={locator}"
                
        return locator
    
    def _is_explicit_strategy(self, locator: str) -> bool:
        """Check if locator uses explicit strategy syntax."""
        if "=" not in locator and ":" not in locator:
            return False
            
        # Get the part before = or :
        separator = "=" if "=" in locator else ":"
        strategy_part = locator.split(separator, 1)[0].lower().strip()
        
        # Known SeleniumLibrary strategies
        selenium_strategies = {
            'id', 'name', 'identifier', 'class', 'tag', 'xpath', 'css', 
            'dom', 'link', 'partial link', 'data', 'jquery'
        }
        
        return strategy_part in selenium_strategies
    
    def _convert_jquery_to_css(self, jquery_selector: str) -> str:
        """Convert jQuery selector to CSS (best effort)."""
        # This is a simplified conversion for common jQuery patterns
        # More complex jQuery expressions may not convert perfectly
        
        # Remove jQuery-specific syntax
        css_selector = jquery_selector
        
        # Convert :eq(n) to :nth-child(n+1)
        css_selector = re.sub(r':eq\((\d+)\)', r':nth-child(\1)', css_selector)
        
        # Convert :first to :first-child
        css_selector = css_selector.replace(':first', ':first-child')
        
        # Convert :last to :last-child  
        css_selector = css_selector.replace(':last', ':last-child')
        
        # Remove other jQuery-specific selectors that don't have CSS equivalents
        css_selector = re.sub(r':contains\([^)]+\)', '', css_selector)
        
        logger.debug(f"Converted jQuery '{jquery_selector}' to CSS '{css_selector}'")
        return css_selector
    
    def _convert_cascaded_to_xpath(self, cascaded_selector: str) -> str:
        """Convert Browser Library cascaded selector to XPath."""
        # This is a simplified conversion for basic cascaded selectors
        # Complex cascaded selectors may need more sophisticated handling
        
        parts = cascaded_selector.split(">>");
        xpath_parts = []
        
        for part in parts:
            part = part.strip()
            if part.startswith("text="):
                # text=content -> *[contains(text(),'content')]
                text_content = part[5:]
                xpath_parts.append(f'*[contains(text(),"{text_content}")]')
            elif part.startswith("#"):
                # #id -> *[@id='id']
                xpath_parts.append(f'*[@id="{part[1:]}"]')
            elif part.startswith("."):
                # .class -> *[contains(@class,'class')]
                xpath_parts.append(f'*[contains(@class,"{part[1:]}")]')
            else:
                # Assume CSS selector, convert to xpath
                xpath_parts.append(f'*[contains(@class,"{part}")]')
        
        xpath = "//" + "//".join(xpath_parts)
        logger.debug(f"Converted cascaded '{cascaded_selector}' to XPath '{xpath}'")
        return f"xpath={xpath}"
    
    async def _execute_hybrid_keyword(self, session: ExecutionSession, keyword_name: str, args: List[str]) -> Optional[Dict[str, Any]]:
        """Execute a keyword using hybrid approach: overrides first, then dynamic."""
        try:
            # First, check if keyword exists in dynamic discovery to get library info
            keyword_info = self.keyword_discovery.find_keyword(keyword_name)
            if not keyword_info:
                # Try to find similar keywords and provide suggestions
                suggestions = self.keyword_discovery.suggest_similar_keywords(keyword_name, 3)
                if suggestions:
                    suggestion_names = [f"{s.library}.{s.name}" for s in suggestions]
                    return {
                        "success": False,
                        "error": f"Keyword '{keyword_name}' not found. Did you mean: {', '.join(suggestion_names)}?",
                        "suggestions": suggestion_names
                    }
                # If no suggestions either, this keyword really doesn't exist
                return {
                    "success": False,
                    "error": f"Keyword '{keyword_name}' not found in any loaded libraries"
                }
            
            logger.info(f"Executing hybrid keyword: {keyword_info.library}.{keyword_info.name}")
            
            # With global instances, validation is simplified - just check availability
            if keyword_info.library == "Browser" and not self.browser_lib:
                return {
                    "success": False,
                    "error": f"Browser Library not available for keyword '{keyword_name}'"
                }
            elif keyword_info.library == "SeleniumLibrary" and not self.selenium_lib:
                return {
                    "success": False,
                    "error": f"SeleniumLibrary not available for keyword '{keyword_name}'"
                }
            
            # Convert locators if needed for consistency (before override handling)
            converted_args = args.copy()
            if keyword_info.library == "Browser" and args:
                # Convert first argument (usually a selector) for Browser Library keywords
                if any(kw in keyword_name.lower() for kw in ['click', 'fill', 'get text', 'wait', 'select']):
                    converted_args[0] = self._convert_locator_for_library(args[0], "Browser")
                    if converted_args[0] != args[0]:
                        logger.info(f"Converted locator: '{args[0]}' -> '{converted_args[0]}'")
            
            # Detect which library should handle this keyword
            detected_library = self._detect_library_from_keyword(keyword_name, converted_args)
            
            # Override library detection if we know it's a SeleniumLibrary keyword
            effective_library = keyword_info.library
            if detected_library == "selenium":
                # Force use of SeleniumLibrary for known SeleniumLibrary keywords
                effective_library = "SeleniumLibrary" 
                
                # If we have SeleniumLibrary available, use it directly instead of overrides
                if self.selenium_lib:
                    return await self._execute_selenium_keyword_directly(session, keyword_name, converted_args)
            
            # Check for override handler
            override_handler = self.override_registry.get_override(keyword_name, effective_library)
            
            if override_handler:
                # Use override handler
                logger.info(f"Using override handler for {keyword_name}")
                override_result = await override_handler.execute(session, keyword_name, converted_args, keyword_info)
                
                # Apply state updates if any
                if override_result.state_updates:
                    self._apply_state_updates(session, override_result.state_updates)
                
                # Auto-capture page source for DOM-changing keywords
                result_data = {
                    "success": override_result.success,
                    "output": override_result.output,
                    "error": override_result.error,
                    "variables": override_result.variables or {},
                    "keyword_info": {
                        **(keyword_info.__dict__ if keyword_info else {}),
                        "override_used": True,
                        "override_metadata": override_result.metadata
                    }
                }
                
                # Add auto page source capture for successful DOM-changing keywords
                if override_result.success and self.keyword_discovery.is_dom_changing_keyword(keyword_name):
                    page_source = await self._capture_page_source_after_keyword(session, keyword_name)
                    if page_source:
                        result_data["page_source"] = page_source
                        result_data["keyword_info"]["auto_captured_dom"] = True
                        logger.info(f"Auto-captured page source after DOM-changing keyword: {keyword_name}")
                
                return result_data
            else:
                # Use default dynamic handler
                logger.info(f"Using dynamic handler for {keyword_name}")
                dynamic_result = await self.dynamic_handler.execute(session, keyword_name, converted_args, keyword_info)
                
                result_data = {
                    "success": dynamic_result.success,
                    "output": dynamic_result.output,
                    "error": dynamic_result.error,
                    "variables": dynamic_result.variables or {},
                    "keyword_info": {
                        **(keyword_info.__dict__ if keyword_info else {}),
                        "override_used": False,
                        "override_metadata": dynamic_result.metadata
                    }
                }
                
                # Add auto page source capture for successful DOM-changing keywords
                if dynamic_result.success and self.keyword_discovery.is_dom_changing_keyword(keyword_name):
                    page_source = await self._capture_page_source_after_keyword(session, keyword_name)
                    if page_source:
                        result_data["page_source"] = page_source
                        result_data["keyword_info"]["auto_captured_dom"] = True
                        logger.info(f"Auto-captured page source after DOM-changing keyword: {keyword_name}")
                
                return result_data
                
        except Exception as e:
            logger.error(f"Error in hybrid keyword execution: {e}")
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": f"Hybrid keyword execution error: {str(e)}",
                "output": None
            }
            
    def _apply_state_updates(self, session: ExecutionSession, state_updates: Dict[str, Any]):
        """Apply state updates to the session."""
        for key, value in state_updates.items():
            if not hasattr(session, 'custom_state'):
                session.custom_state = {}
            session.custom_state[key] = value
            logger.debug(f"Applied state update: {key} = {value}")

    def get_available_keywords(self, library_name: str = None) -> List[Dict[str, Any]]:
        """Get list of available keywords, optionally filtered by library.
        
        Args:
            library_name: Optional library name to filter keywords (e.g., 'Browser', 'BuiltIn')
                         If None, returns all keywords from all libraries.
        
        Returns:
            List of keyword information dictionaries with short_doc from native RF libdoc
        """
        # Use libdoc-based storage if available, otherwise fall back to inspection-based
        if self.rf_doc_storage.is_available():
            if library_name:
                # Get keywords from specific library using libdoc
                keywords_from_lib = self.rf_doc_storage.get_keywords_by_library(library_name)
                return [{
                    "name": kw.name,
                    "library": kw.library,
                    "args": kw.args,
                    "short_doc": kw.short_doc,
                    "tags": kw.tags,
                    "is_deprecated": kw.is_deprecated,
                    "arg_types": kw.arg_types
                } for kw in keywords_from_lib]
            else:
                # Get all keywords using libdoc
                keywords = []
                for kw in self.rf_doc_storage.get_all_keywords():
                    keywords.append({
                        "name": kw.name,
                        "library": kw.library,
                        "args": kw.args,
                        "short_doc": kw.short_doc,
                        "tags": kw.tags,
                        "is_deprecated": kw.is_deprecated,
                        "arg_types": kw.arg_types
                    })
                return keywords
        else:
            # Fall back to inspection-based discovery
            if library_name:
                # Get keywords from specific library
                keywords_from_lib = self.keyword_discovery.get_keywords_by_library(library_name)
                return [{
                    "name": keyword_info.name,
                    "library": keyword_info.library,
                    "args": keyword_info.args,
                    "short_doc": keyword_info.short_doc,
                    "tags": keyword_info.tags,
                    "is_builtin": keyword_info.is_builtin
                } for keyword_info in keywords_from_lib]
            else:
                # Get all keywords from all libraries
                keywords = []
                
                for keyword_info in self.keyword_discovery.get_all_keywords():
                    keywords.append({
                        "name": keyword_info.name,
                        "library": keyword_info.library,
                        "args": keyword_info.args,
                        "short_doc": keyword_info.short_doc,
                        "tags": keyword_info.tags,
                        "is_builtin": keyword_info.is_builtin
                    })
                
                return keywords
    
    def search_keywords(self, pattern: str) -> List[Dict[str, Any]]:
        """Search for keywords matching a pattern using native RF libdoc when available."""
        # Use libdoc-based search if available, otherwise fall back to inspection-based
        if self.rf_doc_storage.is_available():
            matches = self.rf_doc_storage.search_keywords(pattern)
            return [{
                "name": kw.name,
                "library": kw.library,
                "args": kw.args,
                "short_doc": kw.short_doc,
                "tags": kw.tags,
                "is_deprecated": kw.is_deprecated,
                "arg_types": kw.arg_types
            } for kw in matches]
        else:
            # Fall back to inspection-based search
            matches = self.keyword_discovery.search_keywords(pattern)
            return [{
                "name": kw.name,
                "library": kw.library,
                "args": kw.args,
                "short_doc": kw.short_doc,
                "tags": kw.tags,
                "is_builtin": kw.is_builtin
            } for kw in matches]
    
    def get_library_status(self) -> Dict[str, Any]:
        """Get status of all loaded libraries including libdoc availability."""
        # Get inspection-based status
        inspection_status = self.keyword_discovery.get_library_status()
        
        # Get libdoc-based status if available
        if self.rf_doc_storage.is_available():
            libdoc_status = self.rf_doc_storage.get_library_status()
            return {
                "inspection_based": inspection_status,
                "libdoc_based": libdoc_status,
                "preferred_source": "libdoc" if libdoc_status["libdoc_available"] else "inspection"
            }
        else:
            return {
                "inspection_based": inspection_status,
                "libdoc_based": {"libdoc_available": False},
                "preferred_source": "inspection"
            }
    
    def get_keyword_documentation(self, keyword_name: str, library_name: str = None) -> Dict[str, Any]:
        """Get full documentation for a specific keyword using native RF libdoc when available.
        
        Args:
            keyword_name: Name of the keyword to get documentation for
            library_name: Optional library name to narrow search
            
        Returns:
            Dict containing full keyword documentation, args, and metadata
        """
        # Try libdoc-based lookup first
        if self.rf_doc_storage.is_available():
            keyword_info = self.rf_doc_storage.get_keyword_documentation(keyword_name, library_name)
            
            if keyword_info:
                return {
                    "success": True,
                    "keyword": {
                        "name": keyword_info.name,
                        "library": keyword_info.library,
                        "args": keyword_info.args,
                        "arg_types": keyword_info.arg_types,
                        "doc": keyword_info.doc,
                        "short_doc": keyword_info.short_doc,
                        "tags": keyword_info.tags,
                        "is_deprecated": keyword_info.is_deprecated,
                        "source": keyword_info.source,
                        "lineno": keyword_info.lineno
                    }
                }
        
        # Fall back to inspection-based discovery
        keyword_info = self.keyword_discovery.find_keyword(keyword_name)
        
        if not keyword_info:
            return {
                "success": False,
                "error": f"Keyword '{keyword_name}' not found",
                "keyword": None
            }
        
        # If library name is specified, ensure it matches
        if library_name and keyword_info.library.lower() != library_name.lower():
            return {
                "success": False,
                "error": f"Keyword '{keyword_name}' not found in library '{library_name}'",
                "keyword": None
            }
        
        return {
            "success": True,
            "keyword": {
                "name": keyword_info.name,
                "library": keyword_info.library,
                "args": keyword_info.args,
                "defaults": getattr(keyword_info, 'defaults', {}),
                "doc": keyword_info.doc,
                "short_doc": keyword_info.short_doc,
                "tags": keyword_info.tags,
                "is_builtin": getattr(keyword_info, 'is_builtin', False),
                "method_name": getattr(keyword_info, 'method_name', "")
            }
        }

    async def execute_step(
        self,
        keyword: str,
        arguments: List[str] = None,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Execute a single Robot Framework keyword step.
        
        Args:
            keyword: Robot Framework keyword name
            arguments: List of arguments for the keyword
            session_id: Session identifier
            
        Returns:
            Execution result with status, output, and state
        """
        try:
            if arguments is None:
                arguments = []
            
            # Get or create session
            session = self._get_or_create_session(session_id)
            
            # Create execution step
            step = ExecutionStep(
                step_id=str(uuid.uuid4()),
                keyword=keyword,
                arguments=arguments,
                start_time=datetime.now()
            )
            
            # Update session activity
            session.last_activity = datetime.now()
            
            # Mark step as running
            step.status = "running"
            
            logger.info(f"Executing keyword: {keyword} with args: {arguments}")
            
            # Execute the keyword
            result = await self._execute_keyword(session, step)
            
            # Update step status
            step.end_time = datetime.now()
            step.result = result.get("output")
            
            if result["success"]:
                step.status = "pass"
                # Only append successful steps to the session for suite generation
                session.steps.append(step)
                logger.debug(f"Added successful step to session: {keyword}")
            else:
                step.status = "fail"
                step.error = result.get("error")
                logger.debug(f"Failed step not added to session: {keyword} - {result.get('error')}")
            
            # Update session variables if any were set
            if "variables" in result:
                session.variables.update(result["variables"])
            
            return {
                "success": result["success"],
                "step_id": step.step_id,
                "keyword": keyword,
                "arguments": arguments,
                "status": step.status,
                "output": result.get("output"),
                "error": result.get("error"),
                "execution_time": self._calculate_execution_time(step),
                "session_variables": dict(session.variables),
                "state_snapshot": await self._capture_state_snapshot(session),
                "keyword_info": result.get("keyword_info", {}),
                "suggestions": result.get("suggestions", [])
            }
            
        except Exception as e:
            logger.error(f"Error executing step {keyword}: {e}")
            
            # For failed steps due to exceptions, create a step object for error reporting
            # but don't add it to session.steps since it failed
            step.status = "fail"
            step.error = str(e)
            step.end_time = datetime.now()
            
            session_variables = {}
            if session_id in self.sessions:
                session_variables = dict(self.sessions[session_id].variables)
            
            return {
                "success": False,
                "error": str(e),
                "step_id": step.step_id,
                "keyword": keyword,
                "arguments": arguments,
                "status": "fail",
                "execution_time": self._calculate_execution_time(step),
                "session_variables": session_variables
            }

    async def _execute_keyword(self, session: ExecutionSession, step: ExecutionStep) -> Dict[str, Any]:
        """Execute a specific keyword with error handling."""
        try:
            keyword_name = step.keyword
            args = step.arguments
            
            # Detect and set active library based on keyword
            library_preference = self._detect_library_from_keyword(keyword_name, args)
            if library_preference == "selenium":
                session.browser_state.active_library = "selenium"
            elif library_preference == "browser":
                session.browser_state.active_library = "browser"
            # If "auto", leave active_library as is (will auto-detect later)
            
            # Use hybrid execution approach
            hybrid_result = await self._execute_hybrid_keyword(session, keyword_name, args)
            if hybrid_result is not None:
                return hybrid_result
            
            # Create a test suite and case for execution
            if not session.suite:
                session.suite = self._create_test_suite(session.session_id)
            
            # Create a test case for this step
            test_case = TestCase(name=f"Step_{step.step_id}")
            
            # Create keyword call
            keyword_call = Keyword(
                name=keyword_name,
                args=args
            )
            
            test_case.body.append(keyword_call)
            session.suite.tests.append(test_case)
            
            # Execute the test case
            result = await self._run_test_case(session, test_case)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing keyword {step.keyword}: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": None
            }

    def _create_test_suite(self, session_id: str):
        """Create a new test suite for the session."""
        if not ROBOT_AVAILABLE or TestSuite is None:
            return None
        
        suite = TestSuite(name=f"Session_{session_id}")
        
        # Add default imports
        try:
            suite.resource.imports.library("BuiltIn")
        except AttributeError:
            pass  # Older Robot Framework versions may not have this structure
        
        return suite

    async def _run_test_case(self, session: ExecutionSession, test_case: TestCase) -> Dict[str, Any]:
        """Run a single test case and return results."""
        try:
            # This is a simplified execution - in a full implementation,
            # you would use Robot Framework's execution engine
            
            # For now, simulate execution based on keyword patterns
            keyword_name = test_case.body[0].name if test_case.body else ""
            args = test_case.body[0].args if test_case.body else []
            
            # Handle Browser Library keywords that may reach this fallback path
            # (Most keywords should be handled by hybrid execution system)
            result = None
            if "New Browser" in keyword_name:
                result = await self._execute_new_browser(session, args)
            elif "New Context" in keyword_name:
                result = await self._execute_new_context(session, args)
            elif "New Page" in keyword_name:
                result = await self._execute_new_page(session, args)
            elif "Fill" in keyword_name:
                result = await self._execute_fill(session, args)
            elif "Get Text" in keyword_name:
                result = await self._execute_get_text(session, args)
            elif "Get Property" in keyword_name:
                result = await self._execute_get_property(session, args)
            elif "Wait For Elements State" in keyword_name:
                result = await self._execute_wait_for_elements_state(session, args)
            elif "Close Browser" in keyword_name:
                result = await self._execute_close_browser(session, args)
            elif "Click" in keyword_name:
                result = await self._execute_click(session, args)
            else:
                # Unknown keyword - should have been handled by hybrid execution
                return {
                    "success": False,
                    "error": f"Keyword '{keyword_name}' not handled by hybrid execution system",
                    "output": None
                }
            
            # Auto-capture page source for successful DOM-changing keywords in fallback execution
            if result and result.get("success") and self.keyword_discovery.is_dom_changing_keyword(keyword_name):
                page_source = await self._capture_page_source_after_keyword(session, keyword_name)
                if page_source:
                    result["page_source"] = page_source
                    if "keyword_info" not in result:
                        result["keyword_info"] = {}
                    result["keyword_info"]["auto_captured_dom"] = True
                    logger.info(f"Auto-captured page source after DOM-changing keyword in fallback: {keyword_name}")
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": None
            }

    async def _handle_import_library(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Handle Import Library keyword."""
        if not args:
            return {
                "success": False,
                "error": "Library name required",
                "output": None
            }
        
        library_name = args[0]
        
        try:
            # Add to imported libraries
            if library_name not in session.imported_libraries:
                session.imported_libraries.append(library_name)
            
            # Add to suite imports if suite exists
            if session.suite:
                session.suite.resource.imports.library(library_name)
            
            return {
                "success": True,
                "output": f"Library '{library_name}' imported successfully",
                "variables": {}
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to import library '{library_name}': {str(e)}",
                "output": None
            }

    async def _handle_set_variable(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Handle Set Variable keyword."""
        if len(args) < 2:
            return {
                "success": False,
                "error": "Variable name and value required",
                "output": None
            }
        
        var_name = args[0]
        var_value = args[1]
        
        # Store in session variables
        session.variables[var_name] = var_value
        
        return {
            "success": True,
            "output": f"Variable '{var_name}' set to '{var_value}'",
            "variables": {var_name: var_value}
        }

    async def _handle_log(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Handle Log keyword."""
        message = args[0] if args else "No message"
        level = args[1] if len(args) > 1 else "INFO"
        
        logger.info(f"Robot Log [{level}]: {message}")
        
        return {
            "success": True,
            "output": f"Logged: {message}",
            "variables": {}
        }

    # Browser Library execution methods
    async def _execute_new_browser(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Execute real New Browser keyword."""
        try:
            # Use real Browser Library if available, otherwise simulate
            if self.browser_lib:
                return await self._real_new_browser(session, args)
            else:
                return await self._simulate_new_browser(session, args)
        except Exception as e:
            logger.error(f"Error in _execute_new_browser: {e}")
            return {"success": False, "error": str(e)}

    async def _real_new_browser(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Execute actual New Browser keyword using Browser Library."""
        try:
            browser_type_str = args[0] if args else "chromium"
            
            # Convert string to SupportedBrowsers enum
            browser_type_map = {
                "chromium": SupportedBrowsers.chromium,
                "firefox": SupportedBrowsers.firefox,
                "webkit": SupportedBrowsers.webkit,
                "chrome": SupportedBrowsers.chromium  # Map chrome to chromium
            }
            
            browser_type = browser_type_map.get(browser_type_str.lower(), SupportedBrowsers.chromium)
            
            # Parse additional arguments
            kwargs = {}
            for arg in args[1:]:
                if "headless=" in arg:
                    kwargs["headless"] = arg.split("=")[1].lower() == "true"
                elif "timeout=" in arg:
                    # Convert timeout to timedelta
                    timeout_str = arg.split("=")[1]
                    if timeout_str.endswith('s'):
                        timeout_seconds = float(timeout_str[:-1])
                    else:
                        timeout_seconds = float(timeout_str)
                    kwargs["timeout"] = dt.timedelta(seconds=timeout_seconds)
            
            # Set default headless mode to False for visibility during script creation
            if "headless" not in kwargs:
                kwargs["headless"] = False
            
            # Use session-scoped Browser Library instance
            browser_lib = self.browser_lib
            if not browser_lib:
                return {"success": False, "error": "Browser Library not available for this session"}
            
            # Call actual Browser Library method
            browser_id = browser_lib.new_browser(browser=browser_type, **kwargs)
            
            # Update session state with real browser ID
            session.browser_state.browser_type = browser_type_str  # Store string for compatibility
            session.browser_state.browser_id = str(browser_id)
            session.current_browser = browser_type_str
            
            # Update session variables
            session.variables.update({
                "browser_type": browser_type_str,
                "browser_id": str(browser_id),
                "headless": str(kwargs.get("headless", False))
            })
            
            return {
                "success": True,
                "output": f"Browser '{browser_type_str}' created with ID '{browser_id}' (headless={kwargs.get('headless', False)})",
                "variables": {"browser_type": browser_type_str, "browser_id": str(browser_id)},
                "browser_state": await self._capture_real_browser_state(session)
            }
            
        except Exception as e:
            logger.error(f"Error creating real browser: {e}")
            return {"success": False, "error": str(e)}

    async def _simulate_new_browser(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Simulate New Browser keyword."""
        browser_type = args[0] if args else "chromium"
        headless = args[1] if len(args) > 1 else "False"
        
        # Generate unique browser ID
        browser_id = f"browser_{uuid.uuid4().hex[:8]}"
        
        # Update browser state
        session.browser_state.browser_type = browser_type
        session.browser_state.browser_id = browser_id
        session.current_browser = browser_type
        
        # Update session variables
        session.variables.update({
            "browser_type": browser_type,
            "browser_id": browser_id,
            "headless": headless
        })
        
        return {
            "success": True,
            "output": f"Browser '{browser_type}' created with ID '{browser_id}' (headless={headless})",
            "variables": {"browser_type": browser_type, "browser_id": browser_id},
            "browser_state": await self._capture_browser_state(session)
        }

    async def _execute_new_context(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Execute real New Context keyword."""
        try:
            if self.browser_lib:
                return await self._real_new_context(session, args)
            else:
                return await self._simulate_new_context(session, args)
        except Exception as e:
            logger.error(f"Error in _execute_new_context: {e}")
            return {"success": False, "error": str(e)}

    async def _real_new_context(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Execute actual New Context keyword using Browser Library."""
        try:
            if not session.browser_state.browser_id:
                return {
                    "success": False,
                    "error": "No browser created. Use 'New Browser' first.",
                    "output": None
                }
            
            # Parse viewport and other context options
            kwargs = {}
            viewport = {"width": 1280, "height": 720}
            
            for arg in args:
                if "viewport=" in arg:
                    # Simple viewport parsing
                    viewport_str = arg.split("=", 1)[1].strip("{}")
                    for param in viewport_str.split(","):
                        if "width" in param:
                            viewport["width"] = int(param.split(":")[1].strip())
                        elif "height" in param:
                            viewport["height"] = int(param.split(":")[1].strip())
                elif "ignoreHTTPSErrors=" in arg:
                    kwargs["ignoreHTTPSErrors"] = arg.split("=")[1].lower() == "true"
                elif "bypassCSP=" in arg:
                    kwargs["bypassCSP"] = arg.split("=")[1].lower() == "true"
            
            kwargs["viewport"] = viewport
            
            # Use session-scoped Browser Library instance
            browser_lib = self.browser_lib
            if not browser_lib:
                return {"success": False, "error": "Browser Library not available for this session"}
            
            # Call actual Browser Library method
            context_id = browser_lib.new_context(**kwargs)
            
            # Update session state
            session.browser_state.context_id = str(context_id)
            session.browser_state.viewport = viewport
            session.variables["context_id"] = str(context_id)
            session.variables["viewport"] = viewport
            
            return {
                "success": True,
                "output": f"Context '{context_id}' created with viewport {viewport}",
                "variables": {"context_id": str(context_id), "viewport": viewport},
                "browser_state": await self._capture_real_browser_state(session)
            }
            
        except Exception as e:
            logger.error(f"Error creating real context: {e}")
            return {"success": False, "error": str(e)}

    async def _simulate_new_context(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Simulate New Context keyword."""
        if not session.browser_state.browser_id:
            return {
                "success": False,
                "error": "No browser created. Use 'New Browser' first.",
                "output": None
            }
        
        # Generate unique context ID
        context_id = f"context_{uuid.uuid4().hex[:8]}"
        session.browser_state.context_id = context_id
        
        # Parse viewport if provided
        viewport = {"width": 1280, "height": 720}
        if args:
            try:
                # Simple viewport parsing (width=1920, height=1080)
                for arg in args:
                    if "width=" in arg:
                        viewport["width"] = int(arg.split("=")[1])
                    elif "height=" in arg:
                        viewport["height"] = int(arg.split("=")[1])
            except (ValueError, IndexError):
                pass  # Use defaults
        
        session.browser_state.viewport = viewport
        session.variables["context_id"] = context_id
        session.variables["viewport"] = viewport
        
        return {
            "success": True,
            "output": f"Context '{context_id}' created with viewport {viewport}",
            "variables": {"context_id": context_id, "viewport": viewport},
            "browser_state": await self._capture_browser_state(session)
        }

    async def _execute_new_page(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Execute real New Page keyword."""
        try:
            if self.browser_lib:
                return await self._real_new_page(session, args)
            else:
                return await self._simulate_new_page(session, args)
        except Exception as e:
            logger.error(f"Error in _execute_new_page: {e}")
            return {"success": False, "error": str(e)}

    async def _real_new_page(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Execute actual New Page keyword using Browser Library."""
        try:
            # Validate URL argument FIRST, before checking browser state
            if not args:
                return {
                    "success": False,
                    "error": "New Page requires URL argument",
                    "output": None
                }
            
            url = args[0].strip()
            if not url:
                return {
                    "success": False,
                    "error": "New Page requires non-empty URL argument",
                    "output": None
                }
            
            # Basic URL validation
            if not (url.startswith(('http://', 'https://', 'file://', 'about:', 'data:'))):
                return {
                    "success": False,
                    "error": f"Invalid URL format: '{url}'. URL must start with http://, https://, file://, about:, or data:",
                    "output": None
                }
            
            # Now check browser state
            if not session.browser_state.browser_id:
                return {
                    "success": False,
                    "error": "No browser created. Use 'New Browser' first.",
                    "output": None
                }
            
            # If no context, create default one
            if not session.browser_state.context_id:
                context_result = await self._real_new_context(session, [])
                if not context_result["success"]:
                    return context_result
            
            # Use session-scoped Browser Library instance
            browser_lib = self.browser_lib
            if not browser_lib:
                return {"success": False, "error": "Browser Library not available for this session"}
            
            # Call actual Browser Library method
            page_id = browser_lib.new_page(url)
            
            # Get actual page information
            try:
                page_title = browser_lib.get_title()
                page_url = browser_lib.get_url()
            except Exception as e:
                logger.warning(f"Could not get page info: {e}")
                page_title = self._extract_title_from_url(url)
                page_url = url
            
            # Update session state with real page info
            session.browser_state.page_id = str(page_id)
            session.browser_state.current_url = page_url
            session.browser_state.page_title = page_title
            
            # Get actual page elements
            try:
                session.browser_state.page_elements = await self._get_real_page_elements()
            except Exception as e:
                logger.warning(f"Could not get page elements: {e}")
                session.browser_state.page_elements = []
            
            session.variables.update({
                "page_id": str(page_id),
                "current_url": page_url,
                "page_title": page_title
            })
            
            return {
                "success": True,
                "output": f"Page '{page_id}' opened at '{page_url}'",
                "variables": {
                    "page_id": str(page_id), 
                    "current_url": page_url,
                    "page_title": page_title
                },
                "browser_state": await self._capture_real_browser_state(session)
            }
            
        except Exception as e:
            logger.error(f"Error creating real page: {e}")
            return {"success": False, "error": str(e)}

    async def _simulate_new_page(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Simulate New Page keyword."""
        if not session.browser_state.browser_id:
            return {
                "success": False,
                "error": "No browser created. Use 'New Browser' first.",
                "output": None
            }
        
        # If no context, create default one
        if not session.browser_state.context_id:
            await self._simulate_new_context(session, [])
        
        url = args[0] if args else "about:blank"
        page_id = f"page_{uuid.uuid4().hex[:8]}"
        
        # Update browser state
        session.browser_state.page_id = page_id
        session.browser_state.current_url = url
        session.browser_state.page_title = self._extract_title_from_url(url)
        
        # Simulate page elements based on URL
        session.browser_state.page_elements = await self._simulate_page_elements(url)
        
        session.variables.update({
            "page_id": page_id,
            "current_url": url,
            "page_title": session.browser_state.page_title
        })
        
        return {
            "success": True,
            "output": f"Page '{page_id}' opened at '{url}'",
            "variables": {
                "page_id": page_id, 
                "current_url": url,
                "page_title": session.browser_state.page_title
            },
            "browser_state": await self._capture_browser_state(session)
        }

    async def _execute_fill(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Execute real Fill keyword."""
        try:
            if self.browser_lib:
                return await self._real_fill(session, args)
            else:
                return await self._simulate_fill(session, args)
        except Exception as e:
            logger.error(f"Error in _execute_fill: {e}")
            return {"success": False, "error": str(e)}

    async def _real_fill(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Execute actual Fill keyword using Browser Library."""
        try:
            if len(args) < 2:
                return {
                    "success": False,
                    "error": "Fill requires selector and text arguments",
                    "output": None
                }
            
            selector = args[0].strip()
            text = args[1]
            
            # Validate selector is not empty
            if not selector:
                return {
                    "success": False,
                    "error": "Fill requires non-empty selector argument",
                    "output": None
                }
            
            # Call actual Browser Library method
            self.browser_lib.fill_text(selector, text)
            
            # Update session variables
            session.variables["last_filled_element"] = selector
            session.variables["last_filled_text"] = text
            
            return {
                "success": True,
                "output": f"Filled element '{selector}' with text '{text}'",
                "variables": {"last_filled_element": selector, "last_filled_text": text},
                "browser_state": await self._capture_real_browser_state(session)
            }
            
        except Exception as e:
            logger.error(f"Error in real fill: {e}")
            return {"success": False, "error": str(e)}

    async def _simulate_fill(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Simulate Fill keyword."""
        if len(args) < 2:
            return {
                "success": False,
                "error": "Fill requires selector and text arguments",
                "output": None
            }
        
        selector = args[0]
        text = args[1]
        
        # Simulate element interaction
        element_info = await self._find_element_by_selector(session, selector)
        if not element_info:
            return {
                "success": False,
                "error": f"Element not found with selector: {selector}",
                "output": None
            }
        
        # Update element state
        element_info["value"] = text
        session.variables[f"last_filled_element"] = selector
        session.variables[f"last_filled_text"] = text
        
        return {
            "success": True,
            "output": f"Filled element '{selector}' with text '{text}'",
            "variables": {"last_filled_element": selector, "last_filled_text": text},
            "browser_state": await self._capture_browser_state(session)
        }

    async def _execute_get_text(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Execute real Get Text keyword."""
        try:
            if self.browser_lib:
                return await self._real_get_text(session, args)
            else:
                return await self._simulate_get_text(session, args)
        except Exception as e:
            logger.error(f"Error in _execute_get_text: {e}")
            return {"success": False, "error": str(e)}

    async def _real_get_text(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Execute actual Get Text keyword using Browser Library."""
        try:
            if not args:
                return {
                    "success": False,
                    "error": "Get Text requires selector argument",
                    "output": None
                }
            
            selector = args[0].strip()
            
            # Validate selector is not empty
            if not selector:
                return {
                    "success": False,
                    "error": "Get Text requires non-empty selector argument",
                    "output": None
                }
            
            # Call actual Browser Library method
            text_content = self.browser_lib.get_text(selector)
            
            session.variables["last_text_content"] = text_content
            
            return {
                "success": True,
                "output": f"Retrieved text '{text_content}' from element '{selector}'",
                "result": text_content,
                "variables": {"last_text_content": text_content},
                "browser_state": await self._capture_real_browser_state(session)
            }
            
        except Exception as e:
            logger.error(f"Error in real get text: {e}")
            return {"success": False, "error": str(e)}

    async def _simulate_get_text(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Simulate Get Text keyword."""
        if not args:
            return {
                "success": False,
                "error": "Get Text requires selector argument",
                "output": None
            }
        
        selector = args[0]
        element_info = await self._find_element_by_selector(session, selector)
        
        if not element_info:
            return {
                "success": False,
                "error": f"Element not found with selector: {selector}",
                "output": None
            }
        
        text_content = element_info.get("text", f"Sample text from {selector}")
        session.variables["last_text_content"] = text_content
        
        return {
            "success": True,
            "output": f"Retrieved text '{text_content}' from element '{selector}'",
            "result": text_content,
            "variables": {"last_text_content": text_content},
            "browser_state": await self._capture_browser_state(session)
        }

    async def _execute_get_property(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Execute real Get Property keyword."""
        try:
            if self.browser_lib:
                return await self._real_get_property(session, args)
            else:
                return await self._simulate_get_property(session, args)
        except Exception as e:
            logger.error(f"Error in _execute_get_property: {e}")
            return {"success": False, "error": str(e)}

    async def _real_get_property(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Execute actual Get Property keyword using Browser Library."""
        try:
            if len(args) < 2:
                return {
                    "success": False,
                    "error": "Get Property requires element reference and property name",
                    "output": None
                }
            
            selector = args[0].strip()
            property_name = args[1].strip()
            
            # Validate arguments are not empty
            if not selector:
                return {
                    "success": False,
                    "error": "Get Property requires non-empty selector argument",
                    "output": None
                }
            
            if not property_name:
                return {
                    "success": False,
                    "error": "Get Property requires non-empty property name argument",
                    "output": None
                }
            
            # Call actual Browser Library method
            property_value = self.browser_lib.get_property(selector, property_name)
            
            session.variables[f"last_property_{property_name}"] = property_value
            
            return {
                "success": True,
                "output": f"Retrieved property '{property_name}' = '{property_value}' from element '{selector}'",
                "result": property_value,
                "variables": {f"last_property_{property_name}": property_value},
                "browser_state": await self._capture_real_browser_state(session)
            }
            
        except Exception as e:
            logger.error(f"Error in real get property: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_wait_for_elements_state(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Execute real Wait For Elements State keyword."""
        try:
            if self.browser_lib:
                return await self._real_wait_for_elements_state(session, args)
            else:
                return await self._simulate_wait_for_elements_state(session, args)
        except Exception as e:
            logger.error(f"Error in _execute_wait_for_elements_state: {e}")
            return {"success": False, "error": str(e)}

    async def _real_wait_for_elements_state(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Execute actual Wait For Elements State keyword using Browser Library."""
        try:
            if len(args) < 2:
                return {
                    "success": False,
                    "error": "Wait For Elements State requires selector and state arguments",
                    "output": None
                }
            
            selector = args[0]
            state = args[1]
            timeout = args[2] if len(args) > 2 else "10s"
            
            # Call actual Browser Library method
            self.browser_lib.wait_for_elements_state(selector, state, timeout=timeout)
            
            return {
                "success": True,
                "output": f"Element '{selector}' reached state '{state}' within {timeout}",
                "variables": {"last_waited_element": selector, "last_waited_state": state},
                "browser_state": await self._capture_real_browser_state(session)
            }
            
        except Exception as e:
            logger.error(f"Error in real wait for elements state: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_click(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Execute real Click keyword."""
        try:
            if self.browser_lib:
                return await self._real_click(session, args)
            else:
                return await self._simulate_click(session, args)
        except Exception as e:
            logger.error(f"Error in _execute_click: {e}")
            return {"success": False, "error": str(e)}

    async def _real_click(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Execute actual Click keyword using Browser Library."""
        try:
            if not args:
                return {
                    "success": False,
                    "error": "Click requires selector argument",
                    "output": None
                }
            
            selector = args[0].strip()
            
            # Validate selector is not empty
            if not selector:
                return {
                    "success": False,
                    "error": "Click requires non-empty selector argument",
                    "output": None
                }
            
            # Call actual Browser Library method
            self.browser_lib.click(selector)
            
            session.variables["last_clicked_element"] = selector
            
            return {
                "success": True,
                "output": f"Clicked element '{selector}'",
                "variables": {"last_clicked_element": selector},
                "browser_state": await self._capture_real_browser_state(session)
            }
            
        except Exception as e:
            logger.error(f"Error in real click: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_close_browser(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Execute real Close Browser keyword."""
        try:
            if self.browser_lib:
                return await self._real_close_browser(session, args)
            else:
                return await self._simulate_close_browser(session, args)
        except Exception as e:
            logger.error(f"Error in _execute_close_browser: {e}")
            return {"success": False, "error": str(e)}

    async def _real_close_browser(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Execute actual Close Browser keyword using Browser Library."""
        try:
            if not session.browser_state.browser_id:
                return {
                    "success": True,
                    "output": "No browser to close",
                    "variables": {}
                }
            
            browser_id = session.browser_state.browser_id
            
            # Use session-scoped Browser Library instance
            browser_lib = self.browser_lib
            if not browser_lib:
                return {"success": False, "error": "Browser Library not available for this session"}
            
            # Call actual Browser Library method
            if args and "ALL" in args[0].upper():
                browser_lib.close_browser("ALL")
                # Clear all browser state for this session
                session.browser_state = BrowserState()
            else:
                browser_lib.close_browser()
            
            # Reset browser state
            session.browser_state = BrowserState()
            session.current_browser = None
            
            # Clear browser-related variables
            browser_vars = [k for k in session.variables.keys() 
                           if k in ["browser_type", "browser_id", "context_id", "page_id", "current_url", "page_title"]]
            for var in browser_vars:
                session.variables.pop(var, None)
            
            return {
                "success": True,
                "output": f"Browser '{browser_id}' closed",
                "variables": {},
                "browser_state": await self._capture_real_browser_state(session)
            }
            
        except Exception as e:
            logger.error(f"Error in real close browser: {e}")
            return {"success": False, "error": str(e)}

    async def _simulate_get_property(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Simulate Get Property keyword."""
        if len(args) < 2:
            return {
                "success": False,
                "error": "Get Property requires element reference and property name",
                "output": None
            }
        
        element_ref = args[0]
        property_name = args[1]
        
        # Simulate property values
        property_values = {
            "innerText": "Sample inner text",
            "innerHTML": "<span>Sample HTML</span>",
            "value": "sample_value",
            "id": "sample_id",
            "className": "sample-class",
            "tagName": "DIV"
        }
        
        property_value = property_values.get(property_name, f"sample_{property_name}")
        session.variables[f"last_property_{property_name}"] = property_value
        
        return {
            "success": True,
            "output": f"Retrieved property '{property_name}' = '{property_value}' from element '{element_ref}'",
            "result": property_value,
            "variables": {f"last_property_{property_name}": property_value},
            "browser_state": await self._capture_browser_state(session)
        }

    async def _simulate_wait_for_elements_state(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Simulate Wait For Elements State keyword."""
        if len(args) < 2:
            return {
                "success": False,
                "error": "Wait For Elements State requires selector and state arguments",
                "output": None
            }
        
        selector = args[0]
        state = args[1]
        timeout = args[2] if len(args) > 2 else "10s"
        
        # Simulate wait by sleeping briefly
        await asyncio.sleep(0.1)
        
        element_info = await self._find_element_by_selector(session, selector)
        if not element_info:
            return {
                "success": False,
                "error": f"Element not found with selector: {selector}",
                "output": None
            }
        
        # Update element state
        element_info["state"] = state
        
        return {
            "success": True,
            "output": f"Element '{selector}' reached state '{state}' within {timeout}",
            "variables": {"last_waited_element": selector, "last_waited_state": state},
            "browser_state": await self._capture_browser_state(session)
        }

    async def _simulate_close_browser(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Simulate Close Browser keyword."""
        if not session.browser_state.browser_id:
            return {
                "success": True,
                "output": "No browser to close",
                "variables": {}
            }
        
        browser_id = session.browser_state.browser_id
        
        # Reset browser state
        session.browser_state = BrowserState()
        session.current_browser = None
        
        # Clear browser-related variables
        browser_vars = [k for k in session.variables.keys() 
                       if k in ["browser_type", "browser_id", "context_id", "page_id", "current_url"]]
        for var in browser_vars:
            session.variables.pop(var, None)
        
        return {
            "success": True,
            "output": f"Browser '{browser_id}' closed",
            "variables": {},
            "browser_state": await self._capture_browser_state(session)
        }

    async def _find_element_by_selector(self, session: ExecutionSession, selector: str) -> Optional[Dict[str, Any]]:
        """Find element by selector in simulated page."""
        # Look for existing element or create simulated one
        for element in session.browser_state.page_elements:
            if (element.get("id") == selector.replace("id=", "") or
                element.get("selector") == selector):
                return element
        
        # Create simulated element
        element = {
            "selector": selector,
            "id": selector.replace("id=", "") if "id=" in selector else f"element_{uuid.uuid4().hex[:6]}",
            "tagName": "div",
            "text": f"Sample text for {selector}",
            "value": "",
            "visible": True,
            "enabled": True,
            "state": "visible"
        }
        
        session.browser_state.page_elements.append(element)
        return element

    async def _simulate_page_elements(self, url: str) -> List[Dict[str, Any]]:
        """Generate simulated page elements based on URL."""
        elements = [
            {
                "selector": "h1",
                "id": "main-heading",
                "tagName": "h1",
                "text": f"Welcome to {url}",
                "visible": True,
                "state": "visible"
            }
        ]
        
        # Add common elements based on URL patterns
        if "login" in url.lower():
            elements.extend([
                {
                    "selector": "id=username",
                    "id": "username",
                    "tagName": "input",
                    "text": "",
                    "value": "",
                    "type": "text",
                    "visible": True,
                    "state": "visible"
                },
                {
                    "selector": "id=password", 
                    "id": "password",
                    "tagName": "input",
                    "text": "",
                    "value": "",
                    "type": "password",
                    "visible": True,
                    "state": "visible"
                },
                {
                    "selector": "id=login-btn",
                    "id": "login-btn", 
                    "tagName": "button",
                    "text": "Login",
                    "visible": True,
                    "state": "visible"
                }
            ])
        
        return elements

    def _extract_title_from_url(self, url: str) -> str:
        """Extract page title from URL."""
        if url == "about:blank":
            return "Blank Page"
        
        try:
            domain = url.split("//")[1].split("/")[0] if "//" in url else url
            return f"Page - {domain}"
        except (IndexError, AttributeError):
            return "Untitled Page"

    async def _capture_real_browser_state(self, session: ExecutionSession) -> Dict[str, Any]:
        """Capture actual browser state from Browser Library."""
        try:
            if not self.browser_lib or not session.browser_state.browser_id:
                # Fall back to simulation if no real browser
                return await self._capture_browser_state(session)
            
            # Get real browser information
            state = {
                "browser_id": session.browser_state.browser_id,
                "browser_type": session.browser_state.browser_type,
                "context_id": session.browser_state.context_id,
                "page_id": session.browser_state.page_id,
            }
            
            # Try to get current page information
            try:
                state["current_url"] = self.browser_lib.get_url()
                state["page_title"] = self.browser_lib.get_title()
            except Exception as e:
                logger.debug(f"Could not get page info: {e}")
                state["current_url"] = session.browser_state.current_url
                state["page_title"] = session.browser_state.page_title
            
            # Try to get viewport info
            try:
                viewport = self.browser_lib.get_viewport_size()
                state["viewport"] = viewport
            except Exception as e:
                logger.debug(f"Could not get viewport: {e}")
                state["viewport"] = session.browser_state.viewport
            
            # Try to get page elements
            try:
                elements = await self._get_real_page_elements()
                state["elements_count"] = len(elements)
                state["page_elements"] = [
                    {
                        "selector": elem.get("selector", ""),
                        "id": elem.get("id", ""),
                        "tagName": elem.get("tagName", ""),
                        "text": elem.get("text", "")[:50],  # Truncate for response size
                        "visible": elem.get("visible", False),
                        "state": elem.get("state", "unknown")
                    } for elem in elements[:10]  # Limit to 10 elements
                ]
            except Exception as e:
                logger.debug(f"Could not get page elements: {e}")
                state["elements_count"] = len(session.browser_state.page_elements)
                state["page_elements"] = [
                    {
                        "selector": elem.get("selector", ""),
                        "id": elem.get("id", ""),
                        "tagName": elem.get("tagName", ""),
                        "text": elem.get("text", "")[:50],
                        "visible": elem.get("visible", False),
                        "state": elem.get("state", "unknown")
                    } for elem in session.browser_state.page_elements[:10]
                ]
            
            # Try to get page source when necessary
            try:
                page_source = self._get_page_source_unified(session.session_id)
                if page_source:
                    # Store full page source but return truncated version for state
                    session.browser_state.page_source = page_source
                    state["page_source_length"] = len(page_source)
                    state["page_source_preview"] = page_source[:1000] + "..." if len(page_source) > 1000 else page_source
                    state["page_source_available"] = True
                    
                    # Extract additional context from page source
                    state["page_context"] = await self._extract_page_context(page_source)
                else:
                    state["page_source_available"] = False
                    state["page_source_length"] = 0
                    state["page_context"] = {}
                
            except Exception as e:
                logger.debug(f"Could not get page source: {e}")
                state["page_source_available"] = False
                state["page_source_length"] = 0
                state["page_context"] = {}
            
            # Try to get page cookies and storage for complete context
            try:
                cookies = self.browser_lib.get_cookies()
                state["cookies"] = cookies
                state["cookie_count"] = len(cookies)
            except Exception as e:
                logger.debug(f"Could not get cookies: {e}")
                state["cookies"] = []
                state["cookie_count"] = 0
            
            return state
            
        except Exception as e:
            logger.error(f"Error capturing real browser state: {e}")
            # Fall back to simulation
            return await self._capture_browser_state(session)

    async def _get_real_page_elements(self) -> List[Dict[str, Any]]:
        """Get actual page elements from Browser Library."""
        try:
            if not self.browser_lib:
                return []
            
            # Get all interactive elements using Browser Library
            elements = []
            
            # Try to get common interactive elements
            selectors = [
                "input", "button", "a", "select", "textarea", 
                "[data-testid]", "[id]", ".btn", "[role='button']"
            ]
            
            for selector in selectors:
                try:
                    element_count = self.browser_lib.get_element_count(selector)
                    if element_count > 0:
                        # Get details for first few elements of this type
                        for i in range(min(element_count, 3)):
                            try:
                                element_selector = f"{selector} >> nth={i}"
                                element_info = {
                                    "selector": selector,
                                    "tagName": selector.split("[")[0] if "[" not in selector else "unknown",
                                    "visible": True,  # Assume visible if we can find it
                                    "enabled": True   # Assume enabled by default
                                }
                                
                                # Try to get text content
                                try:
                                    text = self.browser_lib.get_text(element_selector)
                                    element_info["text"] = text
                                except:
                                    element_info["text"] = ""
                                
                                # Try to get id attribute
                                try:
                                    id_attr = self.browser_lib.get_attribute(element_selector, "id")
                                    element_info["id"] = id_attr
                                except:
                                    element_info["id"] = ""
                                
                                elements.append(element_info)
                                
                            except Exception as e:
                                logger.debug(f"Could not get element {i} for {selector}: {e}")
                                continue
                                
                except Exception as e:
                    logger.debug(f"Could not get elements for {selector}: {e}")
                    continue
            
            return elements[:20]  # Limit total elements
            
        except Exception as e:
            logger.error(f"Error getting real page elements: {e}")
            return []

    async def _extract_page_context(self, page_source: str) -> Dict[str, Any]:
        """Extract useful context information from page source."""
        try:
            context = {
                "forms_detected": 0,
                "input_fields": 0,
                "buttons": 0,
                "links": 0,
                "images": 0,
                "scripts": 0,
                "meta_info": {},
                "has_errors": False,
                "error_indicators": [],
                "content_type": "html",
                "framework_indicators": []
            }
            
            if not page_source:
                return context
            
            # Basic HTML parsing for context extraction
            import re
            
            # Count common elements
            context["forms_detected"] = len(re.findall(r'<form[^>]*>', page_source, re.IGNORECASE))
            context["input_fields"] = len(re.findall(r'<input[^>]*>', page_source, re.IGNORECASE))
            context["buttons"] = len(re.findall(r'<button[^>]*>', page_source, re.IGNORECASE))
            context["links"] = len(re.findall(r'<a[^>]*href', page_source, re.IGNORECASE))
            context["images"] = len(re.findall(r'<img[^>]*>', page_source, re.IGNORECASE))
            context["scripts"] = len(re.findall(r'<script[^>]*>', page_source, re.IGNORECASE))
            
            # Extract meta information
            title_match = re.search(r'<title[^>]*>(.*?)</title>', page_source, re.IGNORECASE | re.DOTALL)
            if title_match:
                context["meta_info"]["title"] = title_match.group(1).strip()
            
            # Extract meta tags
            meta_matches = re.findall(r'<meta[^>]*name=["\']([^"\']*)["\'][^>]*content=["\']([^"\']*)["\']', page_source, re.IGNORECASE)
            for name, content in meta_matches:
                context["meta_info"][name.lower()] = content
            
            # Check for error indicators
            error_patterns = [
                r'error', r'exception', r'not found', r'404', r'500', r'403', 
                r'unauthorized', r'forbidden', r'bad request', r'timeout'
            ]
            
            for pattern in error_patterns:
                if re.search(pattern, page_source, re.IGNORECASE):
                    context["has_errors"] = True
                    context["error_indicators"].append(pattern)
            
            # Detect common frameworks/libraries
            framework_patterns = {
                'react': r'react',
                'angular': r'angular|ng-',
                'vue': r'vue\.js|__vue__|v-',
                'jquery': r'jquery|\$\(',
                'bootstrap': r'bootstrap',
                'tailwind': r'tailwind',
                'material': r'material-ui|mat-'
            }
            
            for framework, pattern in framework_patterns.items():
                if re.search(pattern, page_source, re.IGNORECASE):
                    context["framework_indicators"].append(framework)
            
            # Check content type
            if '<!DOCTYPE html' in page_source.upper() or '<html' in page_source.upper():
                context["content_type"] = "html"
            elif page_source.strip().startswith('{') and page_source.strip().endswith('}'):
                context["content_type"] = "json"
            elif page_source.strip().startswith('<') and not '<html' in page_source.upper():
                context["content_type"] = "xml"
            else:
                context["content_type"] = "text"
            
            return context
            
        except Exception as e:
            logger.debug(f"Error extracting page context: {e}")
            return {"error": str(e), "content_available": False}

    async def _capture_browser_state(self, session: ExecutionSession) -> Dict[str, Any]:
        """Capture current browser state for response."""
        state = {
            "browser_id": session.browser_state.browser_id,
            "browser_type": session.browser_state.browser_type,
            "context_id": session.browser_state.context_id,
            "page_id": session.browser_state.page_id,
            "current_url": session.browser_state.current_url,
            "page_title": session.browser_state.page_title,
            "viewport": session.browser_state.viewport,
            "elements_count": len(session.browser_state.page_elements),
            "page_elements": [
                {
                    "selector": el.get("selector"),
                    "id": el.get("id"),
                    "tagName": el.get("tagName"),
                    "text": el.get("text", "")[:50],  # Truncate for response size
                    "visible": el.get("visible", False),
                    "state": el.get("state", "unknown")
                } for el in session.browser_state.page_elements[:10]  # Limit to 10 elements
            ]
        }
        
        return state

    def _get_or_create_session(self, session_id: str) -> ExecutionSession:
        """Get existing session or create a new one with isolated browser instances."""
        if session_id not in self.sessions:
            session = ExecutionSession(session_id=session_id)
            session.browser_state.session_id = session_id
            
            # Initialize session-scoped library instances for proper isolation
            
            self.sessions[session_id] = session
            logger.info(f"Created new isolated session: {session_id}")
        
        return self.sessions[session_id]


    async def _cleanup_session_browsers(self, session: ExecutionSession) -> None:
        """Simplified cleanup of session browser state."""
        try:
            session_id = session.session_id
            logger.info(f"Cleaning up browser state for session {session_id}")
            
            # Just reset the browser state - global instances remain active
            # This preserves browsers for reuse while clearing session-specific state
            session.browser_state = BrowserState()
            
            logger.info(f"Browser state reset complete for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error during browser state cleanup for session {session.session_id}: {e}")

    def _validate_session_browser_access(self, session: ExecutionSession, keyword_name: str) -> bool:
        """Simplified validation for global browser instances."""
        # With global instances, validation is much simpler
        # Just check that we have browser libraries available
        return self.browser_lib is not None or self.selenium_lib is not None

    def get_session_browser_status(self, session_id: str) -> Dict[str, Any]:
        """Get detailed browser status for a specific session."""
        if session_id not in self.sessions:
            return {"error": f"Session {session_id} not found"}
        
        session = self.sessions[session_id]
        browser_state = session.browser_state
        
        return {
            "session_id": session_id,
            "active_library": browser_state.active_library,
            "browser_library": {
                "global_instance_available": self.browser_lib is not None,
                "browser_id": browser_state.browser_id,
                "context_id": browser_state.context_id,
                "page_id": browser_state.page_id,
            },
            "selenium_library": {
                "global_instance_available": self.selenium_lib is not None,
                "driver_instance": browser_state.driver_instance is not None,
                "session_id": browser_state.selenium_session_id,
            },
            "current_url": browser_state.current_url,
            "page_title": browser_state.page_title,
            "last_activity": session.last_activity.isoformat() if session.last_activity else None
        }

    def _calculate_execution_time(self, step: ExecutionStep) -> float:
        """Calculate execution time for a step."""
        if step.start_time and step.end_time:
            return (step.end_time - step.start_time).total_seconds()
        return 0.0

    async def _capture_state_snapshot(self, session: ExecutionSession) -> Dict[str, Any]:
        """Capture current state snapshot for the session."""
        base_state = {
            "session_id": session.session_id,
            "imported_libraries": session.imported_libraries,
            "variables": dict(session.variables),
            "current_browser": session.current_browser,
            "total_steps": len(session.steps),
            "successful_steps": len(session.steps),  # All steps in session.steps are successful now
            "failed_steps": 0,  # Failed steps are not stored in session.steps anymore
            "last_activity": session.last_activity.isoformat()
        }
        
        # Add browser state if available
        if session.browser_state.browser_id:
            base_state["browser_state"] = await self._capture_browser_state(session)
        
        return base_state

    async def get_session_info(self, session_id: str = "default") -> Dict[str, Any]:
        """Get information about a session."""
        if session_id not in self.sessions:
            return {
                "success": False,
                "error": f"Session '{session_id}' not found"
            }
        
        session = self.sessions[session_id]
        
        return {
            "success": True,
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "total_steps": len(session.steps),
            "successful_steps": len(session.steps),  # All steps in session.steps are successful now
            "failed_steps": 0,  # Failed steps are not stored in session.steps anymore
            "imported_libraries": session.imported_libraries,
            "variables": dict(session.variables),
            "current_browser": session.current_browser
        }

    async def clear_session(self, session_id: str = "default") -> Dict[str, Any]:
        """Clear a session and its state with proper resource cleanup."""
        try:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                
                # Clean up browser state (global instances remain active)
                await self._cleanup_session_browsers(session)
                
                del self.sessions[session_id]
                return {
                    "success": True,
                    "message": f"Session '{session_id}' cleared and resources cleaned up"
                }
            else:
                return {
                    "success": False,
                    "error": f"Session '{session_id}' not found"
                }
                
        except Exception as e:
            logger.error(f"Error clearing session {session_id}: {e}")
            return {
                "success": False,
                "error": f"Error clearing session: {str(e)}"
            }

    async def cleanup_all_sessions(self) -> Dict[str, Any]:
        """Clean up all sessions and their browser resources."""
        try:
            cleaned_sessions = []
            errors = []
            
            for session_id in list(self.sessions.keys()):
                try:
                    result = await self.clear_session(session_id)
                    if result["success"]:
                        cleaned_sessions.append(session_id)
                    else:
                        errors.append(f"Session {session_id}: {result['error']}")
                except Exception as e:
                    errors.append(f"Session {session_id}: {str(e)}")
            
            return {
                "success": len(errors) == 0,
                "cleaned_sessions": cleaned_sessions,
                "errors": errors,
                "message": f"Cleaned up {len(cleaned_sessions)} sessions"
            }
            
        except Exception as e:
            logger.error(f"Error during cleanup of all sessions: {e}")
            return {
                "success": False,
                "error": f"Cleanup failed: {str(e)}"
            }

    async def list_sessions(self) -> Dict[str, Any]:
        """List all active sessions."""
        sessions_info = []
        
        for session_id, session in self.sessions.items():
            sessions_info.append({
                "session_id": session_id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "total_steps": len(session.steps),
                "status": "active"
            })
        
        return {
            "success": True,
            "sessions": sessions_info,
            "total_sessions": len(sessions_info)
        }

    async def get_page_source(self, session_id: str = "default", full_source: bool = False) -> Dict[str, Any]:
        """Get page source for a session.
        
        Args:
            session_id: Session identifier
            full_source: If True, returns complete page source. If False, returns preview.
        """
        try:
            if session_id not in self.sessions:
                return {
                    "success": False,
                    "error": f"Session '{session_id}' not found"
                }
            
            session = self.sessions[session_id]
            
            # Try to get fresh page source using unified method
            try:
                page_source = self._get_page_source_unified(session_id)
                if page_source:
                    session.browser_state.page_source = page_source
                else:
                    page_source = session.browser_state.page_source or ""
            except Exception as e:
                logger.debug(f"Could not get fresh page source: {e}")
                page_source = session.browser_state.page_source or ""
            
            if not page_source:
                return {
                    "success": False,
                    "error": "No page source available for this session"
                }
            
            result = {
                "success": True,
                "session_id": session_id,
                "page_source_length": len(page_source),
                "current_url": session.browser_state.current_url,
                "page_title": session.browser_state.page_title,
                "context": await self._extract_page_context(page_source)
            }
            
            if full_source:
                result["page_source"] = page_source
            else:
                # Return preview for large sources
                if len(page_source) > 2000:
                    result["page_source_preview"] = page_source[:2000] + "...\n[Truncated - use full_source=True for complete source]"
                else:
                    result["page_source_preview"] = page_source
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting page source: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_session_validation_status(self, session_id: str = "default") -> Dict[str, Any]:
        """Get validation status of steps in a session.
        
        This helps AI agents understand which steps have been validated
        and are ready for inclusion in test suites.
        """
        if session_id not in self.sessions:
            return {
                "success": False,
                "error": f"Session '{session_id}' not found"
            }
        
        session = self.sessions[session_id]
        validated_steps = []
        failed_steps = []  # Will always be empty since failed steps are not stored
        
        # Since we only store successful steps now, all steps are validated
        for step in session.steps:
            step_info = {
                "step_id": step.step_id,
                "keyword": step.keyword,
                "arguments": step.arguments,
                "status": step.status,
                "execution_time": self._calculate_execution_time(step)
            }
            validated_steps.append(step_info)  # All stored steps are successful
        
        return {
            "success": True,
            "session_id": session_id,
            "validated_steps": validated_steps,
            "failed_steps": failed_steps,
            "total_steps": len(session.steps),
            "ready_for_suite": len(failed_steps) == 0 and len(validated_steps) > 0,
            "validation_summary": {
                "passed": len(validated_steps),
                "failed": len(failed_steps),
                "success_rate": len(validated_steps) / len(session.steps) if session.steps else 0
            }
        }
    
    async def validate_test_readiness(self, session_id: str = "default") -> Dict[str, Any]:
        """Check if a session is ready for test suite generation.
        
        This method helps enforce the stepwise workflow by checking that
        all steps have been validated before allowing suite generation.
        """
        validation_status = self.get_session_validation_status(session_id)
        
        if not validation_status["success"]:
            return validation_status
        
        ready = validation_status["ready_for_suite"]
        failed_count = len(validation_status["failed_steps"])
        validated_count = len(validation_status["validated_steps"])
        
        guidance = []
        if not ready:
            if validated_count == 0:
                guidance.append("❌ No validated steps found. Execute and validate steps first.")
            if failed_count > 0:
                guidance.append(f"❌ {failed_count} failed steps must be fixed before suite generation.")
        else:
            guidance.append(f"✅ {validated_count} validated steps ready for test suite generation.")
        
        return {
            "success": True,
            "session_id": session_id,
            "ready_for_suite_generation": ready,
            "guidance": guidance,
            "validation_summary": validation_status["validation_summary"],
            "next_action": "build_test_suite" if ready else "validate_step_before_suite"
        }
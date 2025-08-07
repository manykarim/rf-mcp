"""Execution Engine for running Robot Framework keywords using the API."""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import traceback

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
    """Represents Browser Library state."""
    browser_type: Optional[str] = None
    browser_id: Optional[str] = None
    context_id: Optional[str] = None
    page_id: Optional[str] = None
    current_url: Optional[str] = None
    page_title: Optional[str] = None
    viewport: Dict[str, int] = field(default_factory=lambda: {"width": 1280, "height": 720})
    page_source: Optional[str] = None
    cookies: List[Dict[str, Any]] = field(default_factory=list)
    local_storage: Dict[str, str] = field(default_factory=dict)
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
        
        # Initialize Robot Framework
        self._initialize_robot_framework()
        
        # Initialize Browser Library
        self._initialize_browser_library()

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
            session.steps.append(step)
            
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
            else:
                step.status = "fail"
                step.error = result.get("error")
            
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
                "state_snapshot": await self._capture_state_snapshot(session)
            }
            
        except Exception as e:
            logger.error(f"Error executing step {keyword}: {e}")
            return {
                "success": False,
                "error": str(e),
                "keyword": keyword,
                "arguments": arguments,
                "status": "fail"
            }

    async def _execute_keyword(self, session: ExecutionSession, step: ExecutionStep) -> Dict[str, Any]:
        """Execute a specific keyword with error handling."""
        try:
            keyword_name = step.keyword
            args = step.arguments
            
            # Handle special keywords
            if keyword_name.lower() == "import library":
                return await self._handle_import_library(session, args)
            elif keyword_name.lower() == "set variable":
                return await self._handle_set_variable(session, args)
            elif keyword_name.lower() == "log":
                return await self._handle_log(session, args)
            
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
            
            # Handle different keyword types
            # Browser Library keywords (preferred)
            if "New Browser" in keyword_name:
                return await self._execute_new_browser(session, args)
            elif "New Context" in keyword_name:
                return await self._execute_new_context(session, args)
            elif "New Page" in keyword_name:
                return await self._execute_new_page(session, args)
            elif "Fill" in keyword_name:
                return await self._execute_fill(session, args)
            elif "Get Text" in keyword_name:
                return await self._execute_get_text(session, args)
            elif "Get Property" in keyword_name:
                return await self._execute_get_property(session, args)
            elif "Wait For Elements State" in keyword_name:
                return await self._execute_wait_for_elements_state(session, args)
            elif "Close Browser" in keyword_name:
                return await self._execute_close_browser(session, args)
            elif "Click" in keyword_name:
                return await self._execute_click(session, args)
            # SeleniumLibrary keywords (legacy support)
            elif "Open Browser" in keyword_name:
                return await self._simulate_open_browser(session, args)
            elif "Go To" in keyword_name:
                return await self._simulate_go_to(session, args)
            elif "Click" in keyword_name:
                return await self._simulate_click(session, args)
            elif "Input Text" in keyword_name:
                return await self._simulate_input_text(session, args)
            elif "Page Should Contain" in keyword_name:
                return await self._simulate_page_should_contain(session, args)
            elif "Sleep" in keyword_name:
                return await self._simulate_sleep(session, args)
            else:
                # Generic keyword execution
                return await self._simulate_generic_keyword(session, keyword_name, args)
            
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

    # Simulation methods for common keywords
    async def _simulate_open_browser(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Simulate Open Browser keyword."""
        url = args[0] if args else "about:blank"
        browser = args[1] if len(args) > 1 else "chrome"
        
        # Update session state
        session.current_browser = browser
        session.variables["browser"] = browser
        session.variables["current_url"] = url
        
        return {
            "success": True,
            "output": f"Browser '{browser}' opened with URL '{url}'",
            "variables": {"browser": browser, "current_url": url}
        }

    async def _simulate_go_to(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Simulate Go To keyword."""
        if not args:
            return {
                "success": False,
                "error": "URL required",
                "output": None
            }
        
        url = args[0]
        session.variables["current_url"] = url
        
        return {
            "success": True,
            "output": f"Navigated to '{url}'",
            "variables": {"current_url": url}
        }

    async def _simulate_click(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Simulate Click Element/Button keyword."""
        if not args:
            return {
                "success": False,
                "error": "Element locator required",
                "output": None
            }
        
        locator = args[0]
        
        return {
            "success": True,
            "output": f"Clicked element '{locator}'",
            "variables": {"last_clicked_element": locator}
        }

    async def _simulate_input_text(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Simulate Input Text keyword."""
        if len(args) < 2:
            return {
                "success": False,
                "error": "Element locator and text required",
                "output": None
            }
        
        locator = args[0]
        text = args[1]
        
        return {
            "success": True,
            "output": f"Entered text '{text}' into element '{locator}'",
            "variables": {"last_input_element": locator, "last_input_text": text}
        }

    async def _simulate_page_should_contain(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Simulate Page Should Contain keyword."""
        if not args:
            return {
                "success": False,
                "error": "Text to verify required",
                "output": None
            }
        
        text = args[0]
        
        # Simulate verification - in real implementation, would check actual page content
        return {
            "success": True,
            "output": f"Verified page contains '{text}'",
            "variables": {"last_verified_text": text}
        }

    async def _simulate_sleep(self, session: ExecutionSession, args: List[str]) -> Dict[str, Any]:
        """Simulate Sleep keyword."""
        duration = args[0] if args else "1s"
        
        try:
            # Parse duration
            if duration.endswith('s'):
                sleep_time = float(duration[:-1])
            else:
                sleep_time = float(duration)
            
            # Actually sleep for the duration
            await asyncio.sleep(sleep_time)
            
            return {
                "success": True,
                "output": f"Slept for {duration}",
                "variables": {}
            }
            
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid duration format: {duration}",
                "output": None
            }

    async def _simulate_generic_keyword(
        self,
        session: ExecutionSession,
        keyword_name: str,
        args: List[str]
    ) -> Dict[str, Any]:
        """Simulate execution of a generic keyword."""
        return {
            "success": True,
            "output": f"Executed keyword '{keyword_name}' with args: {args}",
            "variables": {}
        }

    # Browser Library execution methods (real implementation)
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
            
            # Call actual Browser Library method
            browser_id = self.browser_lib.new_browser(browser=browser_type, **kwargs)
            
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
            
            # Call actual Browser Library method
            context_id = self.browser_lib.new_context(**kwargs)
            
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
            
            url = args[0] if args else "about:blank"
            
            # Call actual Browser Library method
            page_id = self.browser_lib.new_page(url)
            
            # Get actual page information
            try:
                page_title = self.browser_lib.get_title()
                page_url = self.browser_lib.get_url()
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
            
            selector = args[0]
            text = args[1]
            
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
            
            selector = args[0]
            
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
            
            selector = args[0]
            property_name = args[1]
            
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
            
            selector = args[0]
            
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
            
            # Call actual Browser Library method
            if args and "ALL" in args[0].upper():
                self.browser_lib.close_browser("ALL")
            else:
                self.browser_lib.close_browser()
            
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
                page_source = self.browser_lib.get_page_source()
                # Store full page source but return truncated version for state
                session.browser_state.page_source = page_source
                state["page_source_length"] = len(page_source)
                state["page_source_preview"] = page_source[:1000] + "..." if len(page_source) > 1000 else page_source
                state["page_source_available"] = True
                
                # Extract additional context from page source
                state["page_context"] = await self._extract_page_context(page_source)
                
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
        """Get existing session or create a new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ExecutionSession(session_id=session_id)
        
        return self.sessions[session_id]

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
            "successful_steps": len([s for s in session.steps if s.status == "pass"]),
            "failed_steps": len([s for s in session.steps if s.status == "fail"]),
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
            "successful_steps": len([s for s in session.steps if s.status == "pass"]),
            "failed_steps": len([s for s in session.steps if s.status == "fail"]),
            "imported_libraries": session.imported_libraries,
            "variables": dict(session.variables),
            "current_browser": session.current_browser
        }

    async def clear_session(self, session_id: str = "default") -> Dict[str, Any]:
        """Clear a session and its state with proper resource cleanup."""
        try:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                
                # Clean up browser resources if they exist
                if session.browser_state.browser_id and self.browser_lib:
                    try:
                        logger.info(f"Cleaning up browser resources for session {session_id}")
                        await self._real_close_browser(session, [])
                    except Exception as e:
                        logger.warning(f"Error cleaning up browser for session {session_id}: {e}")
                
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
            
            # Try to get fresh page source if browser is available
            if self.browser_lib and session.browser_state.browser_id:
                try:
                    page_source = self.browser_lib.get_page_source()
                    session.browser_state.page_source = page_source
                except Exception as e:
                    logger.debug(f"Could not get fresh page source: {e}")
                    page_source = session.browser_state.page_source or ""
            else:
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
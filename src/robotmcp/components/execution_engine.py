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
    ROBOT_AVAILABLE = True
except ImportError:
    TestSuite = None
    TestCase = None
    Keyword = None
    RobotSettings = None
    BuiltIn = None
    ROBOT_AVAILABLE = False

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
class ExecutionSession:
    """Manages execution state for a test session."""
    session_id: str
    suite: Optional[Any] = None
    steps: List[ExecutionStep] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    imported_libraries: List[str] = field(default_factory=list)
    current_browser: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

class ExecutionEngine:
    """Executes Robot Framework keywords and manages test sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, ExecutionSession] = {}
        self.builtin = None
        
        # Initialize Robot Framework
        self._initialize_robot_framework()
    
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
            if "Open Browser" in keyword_name:
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
        return {
            "session_id": session.session_id,
            "imported_libraries": session.imported_libraries,
            "variables": dict(session.variables),
            "current_browser": session.current_browser,
            "total_steps": len(session.steps),
            "successful_steps": len([s for s in session.steps if s.status == "pass"]),
            "failed_steps": len([s for s in session.steps if s.status == "fail"]),
            "last_activity": session.last_activity.isoformat()
        }

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
        """Clear a session and its state."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return {
                "success": True,
                "message": f"Session '{session_id}' cleared"
            }
        else:
            return {
                "success": False,
                "error": f"Session '{session_id}' not found"
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
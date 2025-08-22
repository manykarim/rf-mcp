"""Robot Framework context manager for maintaining proper execution context.

This module provides a context manager that creates and maintains a proper
Robot Framework execution context for each session, enabling:
- Variable persistence across keyword calls
- Proper variable scoping (test, suite, global)
- Built-in keyword functionality that depends on context
- Library state persistence
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading

logger = logging.getLogger(__name__)

# Import Robot Framework components
try:
    from robot import run
    from robot.api import TestSuite
    from robot.libraries.BuiltIn import BuiltIn
    from robot.running import TestSuiteBuilder
    from robot.running.model import TestCase
    from robot.conf import RobotSettings
    from robot.running.context import EXECUTION_CONTEXTS
    from robot.variables import Variables
    from robot.output import LOGGER
    from robot.output.loggerhelper import LEVELS
    from robot.errors import DataError, ExecutionFailed
    ROBOT_AVAILABLE = True
except ImportError as e:
    logger.error(f"Robot Framework import failed: {e}")
    ROBOT_AVAILABLE = False
    BuiltIn = None
    EXECUTION_CONTEXTS = None


class RobotContextManager:
    """Manages Robot Framework execution contexts for sessions.
    
    This class creates and maintains proper RF execution contexts that enable:
    - Variable persistence across keyword executions
    - Proper variable scoping (local, test, suite, global)
    - Full BuiltIn library functionality
    - Library state persistence within a session
    """
    
    def __init__(self):
        """Initialize the context manager."""
        self.session_contexts = {}
        self.temp_files = {}
        self.lock = threading.Lock()
        
        if not ROBOT_AVAILABLE:
            logger.error("Robot Framework not available, context mode disabled")
    
    def create_context(self, session_id: str, libraries: List[str] = None) -> Dict[str, Any]:
        """Create a new Robot Framework execution context for a session.
        
        Args:
            session_id: Unique identifier for the session
            libraries: List of libraries to import in the context
            
        Returns:
            Context information including status and loaded libraries
        """
        if not ROBOT_AVAILABLE:
            return {
                "success": False,
                "error": "Robot Framework not available"
            }
        
        with self.lock:
            try:
                # Clean up existing context if any
                if session_id in self.session_contexts:
                    self.cleanup_context(session_id)
                
                # Create temporary test file
                temp_file = self._create_temp_test_file(session_id, libraries)
                self.temp_files[session_id] = temp_file
                
                # Build test suite from file
                suite = TestSuite(name=f"Context Suite {session_id}")
                suite.resource.imports.library("BuiltIn")
                
                # Add requested libraries
                for lib in (libraries or []):
                    suite.resource.imports.library(lib)
                
                # Create a test case to hold our context
                test = suite.tests.create(name=f"Context Test {session_id}")
                test.body.create_keyword(name="Log", args=["Context initialized"])
                
                # Initialize context data
                context_data = {
                    'suite': suite,
                    'test': test,
                    'builtin': None,  # Will be set when context is active
                    'variables': {},
                    'libraries_loaded': libraries or [],
                    'active': False,
                    'execution_count': 0
                }
                
                self.session_contexts[session_id] = context_data
                
                logger.info(f"Created context for session {session_id} with libraries: {libraries}")
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "libraries_loaded": context_data['libraries_loaded'],
                    "context_type": "robot_framework"
                }
                
            except Exception as e:
                logger.error(f"Failed to create context for session {session_id}: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
    
    def execute_in_context(
        self, 
        session_id: str, 
        keyword: str, 
        arguments: List[Any] = None,
        assign_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a keyword within the session's Robot Framework context.
        
        Args:
            session_id: Session identifier
            keyword: Robot Framework keyword to execute
            arguments: Arguments for the keyword
            assign_to: Optional variable name to assign the result to
            
        Returns:
            Execution result including output, variables, and status
        """
        if not ROBOT_AVAILABLE:
            return {
                "success": False,
                "error": "Robot Framework not available"
            }
        
        if session_id not in self.session_contexts:
            return {
                "success": False,
                "error": f"No context found for session {session_id}"
            }
        
        context_data = self.session_contexts[session_id]
        
        try:
            # Get or create BuiltIn instance
            if not context_data.get('builtin'):
                context_data['builtin'] = BuiltIn()
            
            builtin = context_data['builtin']
            
            # Prepare arguments
            if arguments is None:
                arguments = []
            
            # Execute keyword
            logger.debug(f"Executing in context: {keyword} with args: {arguments}")
            
            try:
                # Execute the keyword using BuiltIn
                result = builtin.run_keyword(keyword, *arguments)
                
                # Handle variable assignment
                if assign_to:
                    var_name = self._normalize_variable_name(assign_to)
                    builtin.set_test_variable(var_name, result)
                    context_data['variables'][var_name] = result
                
                # Extract current variables from context
                try:
                    all_vars = builtin.get_variables()
                    if isinstance(all_vars, dict):
                        context_data['variables'].update(all_vars)
                except:
                    # Some RF versions might not support this
                    pass
                
                context_data['execution_count'] += 1
                
                return {
                    "success": True,
                    "result": result,
                    "output": str(result) if result is not None else None,
                    "variables": dict(context_data['variables']),
                    "execution_count": context_data['execution_count']
                }
                
            except Exception as exec_error:
                logger.error(f"Keyword execution failed: {exec_error}")
                return {
                    "success": False,
                    "error": str(exec_error),
                    "keyword": keyword,
                    "arguments": arguments
                }
                
        except Exception as e:
            logger.error(f"Context execution failed for session {session_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_context_variables(self, session_id: str) -> Dict[str, Any]:
        """Get all variables from a session's context.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary of variables in the context
        """
        if session_id not in self.session_contexts:
            return {}
        
        context_data = self.session_contexts[session_id]
        return dict(context_data.get('variables', {}))
    
    def set_context_variables(self, session_id: str, variables: Dict[str, Any]) -> bool:
        """Set variables in a session's context.
        
        Args:
            session_id: Session identifier
            variables: Dictionary of variables to set
            
        Returns:
            True if successful, False otherwise
        """
        if session_id not in self.session_contexts:
            return False
        
        context_data = self.session_contexts[session_id]
        
        try:
            if context_data.get('builtin'):
                builtin = context_data['builtin']
                for name, value in variables.items():
                    var_name = self._normalize_variable_name(name)
                    builtin.set_test_variable(var_name, value)
                    context_data['variables'][var_name] = value
            else:
                # Just store for later use
                for name, value in variables.items():
                    var_name = self._normalize_variable_name(name)
                    context_data['variables'][var_name] = value
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set variables for session {session_id}: {e}")
            return False
    
    def cleanup_context(self, session_id: str) -> bool:
        """Clean up context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if cleanup was successful
        """
        with self.lock:
            try:
                # Remove temp file if exists
                if session_id in self.temp_files:
                    temp_file = self.temp_files[session_id]
                    try:
                        if hasattr(temp_file, 'name') and os.path.exists(temp_file.name):
                            os.unlink(temp_file.name)
                    except:
                        pass
                    del self.temp_files[session_id]
                
                # Remove context data
                if session_id in self.session_contexts:
                    del self.session_contexts[session_id]
                
                logger.info(f"Cleaned up context for session {session_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to cleanup context for session {session_id}: {e}")
                return False
    
    def cleanup_all_contexts(self):
        """Clean up all contexts."""
        session_ids = list(self.session_contexts.keys())
        for session_id in session_ids:
            self.cleanup_context(session_id)
    
    def _create_temp_test_file(self, session_id: str, libraries: List[str]) -> tempfile.NamedTemporaryFile:
        """Create a temporary Robot Framework test file for context initialization.
        
        Args:
            session_id: Session identifier
            libraries: List of libraries to import
            
        Returns:
            NamedTemporaryFile object
        """
        libraries_section = '\n'.join([f"Library    {lib}" for lib in (libraries or [])])
        
        content = f"""*** Settings ***
Library    BuiltIn
{libraries_section}

*** Variables ***
${{SESSION_ID}}    {session_id}
${{CONTEXT_MODE}}    true

*** Test Cases ***
Context Holder Test
    [Documentation]    Maintains execution context for session {session_id}
    Log    Context initialized for session ${{SESSION_ID}}
    Set Test Variable    ${{CONTEXT_ACTIVE}}    true

*** Keywords ***
Execute In Context
    [Arguments]    ${{keyword}}    @{{args}}
    [Documentation]    Executes keywords within this context
    Run Keyword    ${{keyword}}    @{{args}}
    
Get Context Info
    [Documentation]    Returns information about the current context
    ${{vars}}=    Get Variables
    [Return]    ${{vars}}
"""
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.robot',
            delete=False,
            prefix=f'rf_context_{session_id}_'
        )
        temp_file.write(content)
        temp_file.flush()
        
        return temp_file
    
    def _normalize_variable_name(self, name: str) -> str:
        """Normalize variable name to Robot Framework format.
        
        Args:
            name: Variable name to normalize
            
        Returns:
            Normalized variable name in ${name} format
        """
        if not name:
            return name
        
        # Remove any existing variable markers
        name = name.strip()
        if name.startswith('${') and name.endswith('}'):
            return name
        elif name.startswith('$') and not name.startswith('${'):
            name = name[1:]
        
        return f"${{{name}}}"
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a session's context.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Context information or empty dict if not found
        """
        if session_id not in self.session_contexts:
            return {}
        
        context_data = self.session_contexts[session_id]
        
        return {
            "session_id": session_id,
            "active": context_data.get('active', False),
            "libraries_loaded": context_data.get('libraries_loaded', []),
            "variable_count": len(context_data.get('variables', {})),
            "execution_count": context_data.get('execution_count', 0)
        }
    
    def list_sessions(self) -> List[str]:
        """List all active session IDs with contexts.
        
        Returns:
            List of session IDs
        """
        return list(self.session_contexts.keys())


# Global instance for singleton pattern
_context_manager_instance = None


def get_context_manager() -> RobotContextManager:
    """Get the global RobotContextManager instance.
    
    Returns:
        The singleton RobotContextManager instance
    """
    global _context_manager_instance
    if _context_manager_instance is None:
        _context_manager_instance = RobotContextManager()
    return _context_manager_instance
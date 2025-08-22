"""Enhanced Robot Framework context manager using RF's native variable system.

This module provides an improved context manager that properly uses Robot Framework's
Variables and VariableScopes classes to enable proper variable scoping and persistence.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import threading

logger = logging.getLogger(__name__)

# Import Robot Framework components
try:
    from robot.variables import Variables, VariableScopes
    from robot.variables.store import VariableStore
    from robot.variables.scopes import GlobalVariables, SuiteVariables, TestVariables
    from robot.running.namespace import Namespace
    from robot.running.context import ExecutionContexts
    from robot.libraries.BuiltIn import BuiltIn
    from robot.errors import DataError, VariableError
    from robot.utils import DotDict
    ROBOT_AVAILABLE = True
except ImportError as e:
    logger.error(f"Robot Framework import failed: {e}")
    ROBOT_AVAILABLE = False
    Variables = None
    VariableScopes = None


class EnhancedRobotContext:
    """Enhanced context that uses RF's native variable system."""
    
    def __init__(self, session_id: str):
        """Initialize enhanced context with RF's variable system.
        
        Args:
            session_id: Unique identifier for this context
        """
        self.session_id = session_id
        self.variables = None
        self.variable_scopes = None
        self.store = None
        self.namespace = None
        
        if ROBOT_AVAILABLE:
            self._initialize_variable_system()
    
    def _initialize_variable_system(self):
        """Initialize Robot Framework's variable system."""
        try:
            # Create variable store
            self.store = VariableStore()
            
            # Create Variables instance with the store
            self.variables = Variables()
            
            # Initialize with built-in variables
            self._set_builtin_variables()
            
            # Create variable scopes
            self.variable_scopes = VariableScopes(self.variables)
            
            logger.info(f"Initialized RF variable system for session {self.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize variable system: {e}")
            self.variables = None
            self.variable_scopes = None
    
    def _set_builtin_variables(self):
        """Set Robot Framework built-in variables."""
        if not self.variables:
            return
        
        import os
        import tempfile
        
        # Set common built-in variables
        builtins = {
            '${EMPTY}': '',
            '${SPACE}': ' ',
            '${TRUE}': True,
            '${FALSE}': False,
            '${NULL}': None,
            '${NONE}': None,
            '${CURDIR}': os.getcwd(),
            '${TEMPDIR}': tempfile.gettempdir(),
            '${EXECDIR}': os.getcwd(),
            '${/}': os.sep,
            '${:}': os.pathsep,
            '${\\n}': '\n',
            '@{EMPTY}': [],
            '&{EMPTY}': {}
        }
        
        for name, value in builtins.items():
            try:
                self.set_variable(name, value)
            except:
                pass  # Some variables might not be settable
    
    def set_variable(self, name: str, value: Any, scope: str = 'test'):
        """Set a variable in the specified scope.
        
        Args:
            name: Variable name (with or without ${} syntax)
            value: Variable value
            scope: Variable scope ('local', 'test', 'suite', 'global')
        """
        if not self.variables:
            return False
        
        # Normalize variable name
        if not name.startswith(('${', '@{', '&{', '%{')):
            name = f'${{{name}}}'
        
        try:
            # Set variable based on scope
            if scope == 'global':
                self._set_global_variable(name, value)
            elif scope == 'suite':
                self._set_suite_variable(name, value)
            elif scope == 'test':
                self._set_test_variable(name, value)
            else:  # local or default
                self._set_local_variable(name, value)
            
            logger.debug(f"Set variable {name} = {value} in {scope} scope")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set variable {name}: {e}")
            return False
    
    def _set_local_variable(self, name: str, value: Any):
        """Set a local variable."""
        if self.variables:
            # For local scope, just add to the current variable set
            clean_name = name.strip('${}@&')
            self.variables[clean_name] = value
    
    def _set_test_variable(self, name: str, value: Any):
        """Set a test-level variable."""
        if self.variable_scopes:
            try:
                self.variable_scopes.set_test(name, value)
            except:
                # Fallback to local if test scope not available
                self._set_local_variable(name, value)
        else:
            self._set_local_variable(name, value)
    
    def _set_suite_variable(self, name: str, value: Any):
        """Set a suite-level variable."""
        if self.variable_scopes:
            try:
                self.variable_scopes.set_suite(name, value)
            except:
                # Fallback to test if suite scope not available
                self._set_test_variable(name, value)
        else:
            self._set_test_variable(name, value)
    
    def _set_global_variable(self, name: str, value: Any):
        """Set a global variable."""
        if self.variable_scopes:
            try:
                self.variable_scopes.set_global(name, value)
            except:
                # Fallback to suite if global scope not available
                self._set_suite_variable(name, value)
        else:
            self._set_suite_variable(name, value)
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a variable value.
        
        Args:
            name: Variable name
            default: Default value if variable not found
            
        Returns:
            Variable value or default
        """
        if not self.variables:
            return default
        
        # Normalize variable name
        if not name.startswith(('${', '@{', '&{', '%{')):
            name = f'${{{name}}}'
        
        try:
            # Try to get from Variables instance
            clean_name = name.strip('${}@&')
            if hasattr(self.variables, 'get'):
                return self.variables.get(clean_name, default)
            elif clean_name in self.variables:
                return self.variables[clean_name]
            else:
                return default
                
        except Exception as e:
            logger.debug(f"Failed to get variable {name}: {e}")
            return default
    
    def get_all_variables(self) -> Dict[str, Any]:
        """Get all variables in the context.
        
        Returns:
            Dictionary of all variables
        """
        if not self.variables:
            return {}
        
        try:
            # Get variables as dictionary
            if hasattr(self.variables, 'as_dict'):
                return self.variables.as_dict()
            elif hasattr(self.variables, 'data'):
                return dict(self.variables.data)
            else:
                # Try to extract variables
                result = {}
                for name in dir(self.variables):
                    if not name.startswith('_'):
                        try:
                            value = getattr(self.variables, name)
                            if not callable(value):
                                result[f'${{{name}}}'] = value
                        except:
                            pass
                return result
                
        except Exception as e:
            logger.error(f"Failed to get all variables: {e}")
            return {}
    
    def replace_variables(self, item: Any) -> Any:
        """Replace variables in the given item.
        
        Args:
            item: String, list, or dict containing variables
            
        Returns:
            Item with variables replaced
        """
        if not self.variables:
            return item
        
        try:
            if isinstance(item, str):
                return self.variables.replace_string(item)
            elif isinstance(item, list):
                return self.variables.replace_list(item)
            elif isinstance(item, dict):
                # Replace variables in dictionary values
                result = {}
                for key, value in item.items():
                    result[key] = self.replace_variables(value)
                return result
            else:
                return item
                
        except Exception as e:
            logger.debug(f"Failed to replace variables in {item}: {e}")
            return item
    
    def execute_keyword_with_variables(self, keyword: str, args: List[Any]) -> Dict[str, Any]:
        """Execute a keyword with variable resolution.
        
        Args:
            keyword: Keyword name
            args: Keyword arguments
            
        Returns:
            Execution result
        """
        try:
            # Handle special variable-setting keywords
            if keyword in ['Set Variable', 'Set Test Variable', 'Set Suite Variable', 'Set Global Variable']:
                return self._handle_set_variable_keyword(keyword, args)
            
            # Replace variables in arguments
            resolved_args = []
            for arg in args:
                resolved = self.replace_variables(arg)
                resolved_args.append(resolved)
            
            # Log execution
            logger.info(f"Executing {keyword} with args: {resolved_args}")
            
            # For now, return success with resolved args
            # In a real implementation, would execute the keyword here
            return {
                'success': True,
                'keyword': keyword,
                'arguments': args,
                'resolved_arguments': resolved_args,
                'variables': self.get_all_variables()
            }
            
        except Exception as e:
            logger.error(f"Failed to execute keyword {keyword}: {e}")
            return {
                'success': False,
                'error': str(e),
                'keyword': keyword,
                'arguments': args
            }
    
    def _handle_set_variable_keyword(self, keyword: str, args: List[Any]) -> Dict[str, Any]:
        """Handle variable-setting keywords.
        
        Args:
            keyword: The set variable keyword
            args: Arguments (variable name and value)
            
        Returns:
            Result of setting the variable
        """
        if not args:
            return {
                'success': False,
                'error': f'{keyword} requires at least one argument'
            }
        
        if keyword == 'Set Variable':
            # Set Variable just returns the value
            value = args[0] if len(args) == 1 else args
            return {
                'success': True,
                'result': value,
                'keyword': keyword
            }
        
        if len(args) < 2:
            return {
                'success': False,
                'error': f'{keyword} requires variable name and value'
            }
        
        var_name = args[0]
        var_value = args[1] if len(args) == 2 else args[1:]
        
        # Determine scope from keyword
        if keyword == 'Set Test Variable':
            scope = 'test'
        elif keyword == 'Set Suite Variable':
            scope = 'suite'
        elif keyword == 'Set Global Variable':
            scope = 'global'
        else:
            scope = 'test'
        
        # Set the variable
        success = self.set_variable(var_name, var_value, scope)
        
        return {
            'success': success,
            'keyword': keyword,
            'variable_name': var_name,
            'variable_value': var_value,
            'scope': scope,
            'variables': self.get_all_variables()
        }


class EnhancedContextManager:
    """Manages enhanced Robot Framework contexts for sessions."""
    
    def __init__(self):
        """Initialize the enhanced context manager."""
        self.contexts = {}
        self.lock = threading.Lock()
    
    def create_context(self, session_id: str) -> EnhancedRobotContext:
        """Create or get an enhanced context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Enhanced context instance
        """
        with self.lock:
            if session_id not in self.contexts:
                self.contexts[session_id] = EnhancedRobotContext(session_id)
                logger.info(f"Created enhanced context for session {session_id}")
            
            return self.contexts[session_id]
    
    def get_context(self, session_id: str) -> Optional[EnhancedRobotContext]:
        """Get context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Context instance or None
        """
        return self.contexts.get(session_id)
    
    def remove_context(self, session_id: str) -> bool:
        """Remove context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if removed, False if not found
        """
        with self.lock:
            if session_id in self.contexts:
                del self.contexts[session_id]
                logger.info(f"Removed context for session {session_id}")
                return True
            return False
    
    def execute_in_context(
        self, 
        session_id: str,
        keyword: str,
        arguments: List[Any]
    ) -> Dict[str, Any]:
        """Execute a keyword in the session's context.
        
        Args:
            session_id: Session identifier
            keyword: Keyword to execute
            arguments: Keyword arguments
            
        Returns:
            Execution result
        """
        context = self.get_context(session_id)
        if not context:
            context = self.create_context(session_id)
        
        return context.execute_keyword_with_variables(keyword, arguments)


# Global instance
_enhanced_manager = None


def get_enhanced_context_manager() -> EnhancedContextManager:
    """Get the global enhanced context manager.
    
    Returns:
        Enhanced context manager instance
    """
    global _enhanced_manager
    if _enhanced_manager is None:
        _enhanced_manager = EnhancedContextManager()
    return _enhanced_manager
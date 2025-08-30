"""Robot Framework native execution context manager for MCP keywords."""

import logging
import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

# Import Robot Framework native components
try:
    from robot.running.context import EXECUTION_CONTEXTS, _ExecutionContext
    from robot.running.namespace import Namespace
    from robot.variables import Variables
    from robot.output import Output
    from robot.running.model import TestSuite, TestCase
    from robot.libraries import STDLIBS
    from robot.running.arguments import ArgumentSpec
    from robot.running.arguments.argumentresolver import ArgumentResolver
    from robot.running.arguments.typeconverters import TypeConverter
    from robot.running.arguments.typeinfo import TypeInfo
    from robot.libdoc import LibraryDocumentation
    from robot.running.importer import Importer
    from robot.running import Keyword
    from robot.conf import Languages
    RF_NATIVE_AVAILABLE = True
    logger.info("Robot Framework native components imported successfully")
except ImportError as e:
    RF_NATIVE_AVAILABLE = False
    logger.error(f"Robot Framework native components not available: {e}")


class RobotFrameworkNativeContextManager:
    """
    Manages Robot Framework execution context using native RF APIs.
    
    This provides the proper execution environment for keywords that require
    RF execution context like Evaluate, Set Test Variable, etc.
    """
    
    def __init__(self):
        self._session_contexts = {}  # session_id -> context info
        self._active_context = None
        
        if not RF_NATIVE_AVAILABLE:
            logger.warning("RF native context manager initialized without RF components")
    
    def create_context_for_session(self, session_id: str, libraries: List[str] = None) -> Dict[str, Any]:
        """
        Create proper Robot Framework execution context for a session.
        
        This takes a much simpler approach - just ensure EXECUTION_CONTEXTS.current
        exists and use BuiltIn.run_keyword which should now work.
        """
        if not RF_NATIVE_AVAILABLE:
            return {
                "success": False,
                "error": "Robot Framework native components not available"
            }
        
        try:
            logger.info(f"Creating minimal RF context for session {session_id}")
            
            # Simple approach: Create minimal context that enables BuiltIn keywords
            from robot.libraries.BuiltIn import BuiltIn
            from robot.running.testlibraries import TestLibrary
            
            # Check if we already have a context
            if EXECUTION_CONTEXTS.current is None:
                # Create minimal variables with proper structure for BuiltIn.evaluate
                variables = Variables()
                
                # Add the 'current' attribute that BuiltIn.evaluate expects
                # This should return an object with replace_scalar method and store attribute
                if not hasattr(variables, 'current'):
                    # Variables.current should return the Variables object itself
                    # because evaluate_expression expects variables.replace_scalar and variables.store
                    variables.current = variables
                    logger.info("Added 'current' attribute to Variables pointing to self for BuiltIn compatibility")
                
                # Try the simplest possible approach
                # Create a basic test suite (this is closer to how RF actually works)
                suite = TestSuite(name=f"MCP_Session_{session_id}")
                
                # Set a minimal source path to avoid full_name issues
                from pathlib import Path
                suite.source = Path(f"MCP_Session_{session_id}.robot")
                
                # Ensure suite has a resource with required attributes
                from robot.running.resourcemodel import ResourceFile
                suite.resource = ResourceFile(source=suite.source)
                
                # Create minimal namespace with correct parameter order: variables, suite, resource, languages
                namespace = Namespace(variables, suite, suite.resource, Languages())
                
                # Create simple output (try without settings first)
                try:
                    from robot.conf import RobotSettings
                    settings = RobotSettings(output=None)  # Disable actual output
                    output = Output(settings)
                except Exception:
                    # If Output still fails, try a different approach
                    logger.warning("Could not create Output, using minimal logging")
                    output = None
                
                # Start execution context
                if output:
                    ctx = EXECUTION_CONTEXTS.start_suite(suite, namespace, output, dry_run=True)  # dry_run to avoid file I/O
                else:
                    # Even simpler - just set a current context manually
                    from robot.running.context import _ExecutionContext
                    ctx = _ExecutionContext(suite, namespace, output, dry_run=True)
                    EXECUTION_CONTEXTS._contexts.append(ctx)
                    EXECUTION_CONTEXTS._context = ctx
                
                logger.info(f"Minimal RF context created for session {session_id}")
                
            else:
                logger.info(f"RF context already exists, reusing for session {session_id}")
                ctx = EXECUTION_CONTEXTS.current
                variables = ctx.variables
                namespace = ctx.namespace
                output = getattr(ctx, 'output', None)
                suite = ctx.suite
            
            # Store context info
            self._session_contexts[session_id] = {
                "context": ctx,
                "variables": variables,
                "namespace": namespace,
                "output": output,
                "suite": suite,
                "created_at": datetime.now(),
                "libraries": libraries or []
            }
            
            # Set as active context
            self._active_context = session_id
            
            logger.info(f"RF context ready for session {session_id}")
            
            return {
                "success": True,
                "session_id": session_id,
                "context_active": True,
                "libraries_loaded": libraries or []
            }
            
        except Exception as e:
            logger.error(f"Failed to create RF context for session {session_id}: {e}")
            import traceback
            logger.error(f"Context creation traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Context creation failed: {str(e)}"
            }
    
    def execute_keyword_with_context(
        self, 
        session_id: str, 
        keyword_name: str, 
        arguments: List[str],
        assign_to: Optional[Union[str, List[str]]] = None,
        session_variables: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute keyword within proper Robot Framework context.
        
        Args:
            session_id: Session identifier
            keyword_name: RF keyword name
            arguments: List of argument strings
            assign_to: Optional variable assignment
            session_variables: Session variables to sync to RF Variables (for ${response.json()})
            
        Returns:
            Execution result
        """
        if not RF_NATIVE_AVAILABLE:
            return {
                "success": False,
                "error": "Robot Framework native components not available"
            }
        
        if session_id not in self._session_contexts:
            # Try to create context automatically
            result = self.create_context_for_session(session_id)
            if not result["success"]:
                return result
        
        try:
            context_info = self._session_contexts[session_id]
            ctx = context_info["context"]
            namespace = context_info["namespace"]
            variables = context_info["variables"]
            
            logger.info(f"Executing {keyword_name} in RF native context for session {session_id}")
            
            # SYNC SESSION VARIABLES TO RF VARIABLES (critical for ${response.json()})
            if session_variables:
                logger.info(f"Syncing {len(session_variables)} session variables to RF Variables before execution")
                for var_name, var_value in session_variables.items():
                    try:
                        # Normalize and set in RF Variables
                        normalized_name = self._normalize_variable_name(var_name)
                        variables[normalized_name] = var_value
                        logger.debug(f"Synced {normalized_name} = {type(var_value).__name__} to RF Variables")
                    except Exception as e:
                        logger.warning(f"Failed to sync variable {var_name}: {e}")
            
            # Ensure this context is active
            if EXECUTION_CONTEXTS.current != ctx:
                logger.warning(f"Context mismatch for session {session_id}, fixing...")
                # Note: We may need to handle context switching differently
            
            # Use RF's native argument resolution
            result = self._execute_with_native_resolution(
                keyword_name, arguments, namespace, variables, assign_to
            )
            
            # Update session variables from RF variables
            context_info["variables"] = variables
            
            return result
            
        except Exception as e:
            logger.error(f"Context execution failed for session {session_id}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                "success": False,
                "error": f"Context execution failed: {str(e)}",
                "keyword": keyword_name,
                "arguments": arguments
            }
    
    def _execute_with_native_resolution(
        self,
        keyword_name: str,
        arguments: List[str], 
        namespace: Namespace,
        variables: Variables,
        assign_to: Optional[Union[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Execute keyword using RF's native argument resolution and execution.
        
        This uses a simplified approach - just use BuiltIn.run_keyword
        which should now work because we have proper RF execution context.
        """
        try:
            logger.info(f"RF NATIVE: Executing {keyword_name} with args: {arguments}")
            
            # Direct approach: Call BuiltIn methods directly
            # This avoids the run_keyword complexity and uses RF's execution context
            from robot.libraries.BuiltIn import BuiltIn
            
            builtin = BuiltIn()
            
            # Handle specific keywords directly to avoid run_keyword issues
            if keyword_name.lower() == "evaluate":
                if not arguments:
                    raise ValueError("Evaluate keyword requires an expression argument")
                result = builtin.evaluate(arguments[0])
            elif keyword_name.lower() == "set variable":
                if not arguments:
                    raise ValueError("Set Variable keyword requires a value argument")
                result = builtin.set_variable(*arguments)
            elif keyword_name.lower() == "create dictionary":
                # Handle dictionary creation properly
                result = builtin.create_dictionary(*arguments)
            else:
                # For other keywords, try BuiltIn.run_keyword
                if arguments:
                    result = builtin.run_keyword(keyword_name, *arguments)
                else:
                    result = builtin.run_keyword(keyword_name)
            
            # Handle variable assignment using RF's native variable system
            assigned_vars = {}
            if assign_to and result is not None:
                assigned_vars = self._handle_variable_assignment(
                    assign_to, result, variables
                )
            
            # Get variables in a way that works with Variables object
            current_vars = {}
            try:
                if hasattr(variables, 'store'):
                    # Try to get variables from the store
                    current_vars = dict(variables.store.data)
                elif hasattr(variables, 'current') and hasattr(variables.current, 'store'):
                    current_vars = dict(variables.current.store.data)
            except Exception as e:
                logger.debug(f"Could not extract variables: {e}")
                current_vars = {}
            
            return {
                "success": True,
                "result": result,
                "output": str(result) if result is not None else "OK",
                "variables": current_vars,
                "assigned_variables": assigned_vars
            }
            
        except Exception as e:
            logger.error(f"RF native execution failed for {keyword_name}: {e}")
            import traceback
            logger.error(f"RF native execution traceback: {traceback.format_exc()}")
            
            return {
                "success": False,
                "error": f"Keyword execution failed: {str(e)}",
                "keyword": keyword_name,
                "arguments": arguments
            }
    
# Fallback method removed - using simplified approach
    
    def _handle_variable_assignment(
        self,
        assign_to: Union[str, List[str]],
        result: Any,
        variables: Variables
    ) -> Dict[str, Any]:
        """Handle variable assignment using RF's native variable system."""
        assigned_vars = {}
        
        try:
            if isinstance(assign_to, str):
                # Single assignment using RF's native Variables methods
                var_name = self._normalize_variable_name(assign_to)
                # Use Variables.__setitem__ which is the correct RF way
                variables[var_name] = result
                assigned_vars[var_name] = result
                logger.info(f"Assigned {var_name} = {result}")
                
            elif isinstance(assign_to, list):
                # Multiple assignment
                if isinstance(result, (list, tuple)):
                    for i, name in enumerate(assign_to):
                        var_name = self._normalize_variable_name(name)
                        value = result[i] if i < len(result) else None
                        variables[var_name] = value
                        assigned_vars[var_name] = value
                        logger.info(f"Assigned {var_name} = {value}")
                else:
                    # Single value to first variable
                    var_name = self._normalize_variable_name(assign_to[0])
                    variables[var_name] = result
                    assigned_vars[var_name] = result
                    logger.info(f"Assigned {var_name} = {result}")
                    
        except Exception as e:
            logger.warning(f"Variable assignment failed: {e}")
        
        return assigned_vars
    
    def _normalize_variable_name(self, name: str) -> str:
        """Normalize variable name to Robot Framework format."""
        if not name.startswith('${') or not name.endswith('}'):
            return f"${{{name}}}"
        return name
    
    def cleanup_context(self, session_id: str) -> Dict[str, Any]:
        """Clean up Robot Framework context for a session."""
        try:
            if session_id in self._session_contexts:
                # End RF execution context
                EXECUTION_CONTEXTS.end_suite()
                
                # Remove from our tracking
                del self._session_contexts[session_id]
                
                if self._active_context == session_id:
                    self._active_context = None
                
                logger.info(f"Cleaned up RF context for session {session_id}")
                
                return {"success": True, "session_id": session_id}
            else:
                return {
                    "success": False, 
                    "error": f"No context found for session {session_id}"
                }
                
        except Exception as e:
            logger.error(f"Context cleanup failed for session {session_id}: {e}")
            return {
                "success": False,
                "error": f"Context cleanup failed: {str(e)}"
            }
    
    def get_session_context_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a session's RF context."""
        if session_id not in self._session_contexts:
            return {
                "session_id": session_id,
                "context_exists": False
            }
        
        context_info = self._session_contexts[session_id]
        return {
            "session_id": session_id,
            "context_exists": True,
            "created_at": context_info["created_at"].isoformat(),
            "libraries_loaded": context_info["libraries"],
            "variable_count": len(context_info["variables"].store.data) if hasattr(context_info["variables"], 'store') else 0,
            "is_active": self._active_context == session_id
        }
    
    def list_session_contexts(self) -> Dict[str, Any]:
        """List all active RF contexts."""
        contexts = []
        for session_id in self._session_contexts:
            contexts.append(self.get_session_context_info(session_id))
        
        return {
            "total_contexts": len(contexts),
            "active_context": self._active_context,
            "contexts": contexts
        }


# Global instance for use throughout the application
_rf_native_context_manager = None

def get_rf_native_context_manager() -> RobotFrameworkNativeContextManager:
    """Get the global RF native context manager instance."""
    global _rf_native_context_manager
    if _rf_native_context_manager is None:
        _rf_native_context_manager = RobotFrameworkNativeContextManager()
    return _rf_native_context_manager
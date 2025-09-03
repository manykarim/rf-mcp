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

# Import our compatibility utilities
from robotmcp.utils.rf_variables_compatibility import (
    create_compatible_variables, 
    create_compatible_namespace
)


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
                original_variables = Variables()
                
                # Create compatible variables with set_global method for BuiltIn
                variables = create_compatible_variables(original_variables)
                
                # Add the 'current' attribute that BuiltIn.evaluate expects
                if not hasattr(variables, 'current'):
                    variables.current = variables
                    logger.info("Added 'current' attribute to compatible Variables for BuiltIn compatibility")
                
                # Create a basic test suite with proper output directory setup
                suite = TestSuite(name=f"MCP_Session_{session_id}")
                
                # Set a minimal source path to avoid full_name issues  
                from pathlib import Path
                suite.source = Path(f"MCP_Session_{session_id}.robot")
                
                # Ensure suite has a resource with required attributes
                from robot.running.resourcemodel import ResourceFile
                suite.resource = ResourceFile(source=suite.source)
                
                # Create minimal namespace with correct parameter order: variables, suite, resource, languages
                original_namespace = Namespace(original_variables, suite, suite.resource, Languages())
                
                # Create compatible namespace with our compatible variables
                namespace = create_compatible_namespace(original_namespace, variables)
                
                # Create simple output with proper output directory for Browser Library
                try:
                    from robot.conf import RobotSettings
                    import tempfile
                    import os
                    
                    # Create temporary output directory for Browser Library
                    temp_output_dir = tempfile.mkdtemp(prefix="rf_mcp_")
                    
                    # Create settings with output directory - this fixes Browser Library initialization
                    settings = RobotSettings(outputdir=temp_output_dir, output=None)
                    output = Output(settings)
                    
                    # Set OUTPUTDIR variable for Browser Library compatibility
                    # Browser Library uses BuiltIn().get_variable_value("${OUTPUTDIR}")
                    variables["${OUTPUTDIR}"] = temp_output_dir
                    
                    # Set LOGFILE variable for SeleniumLibrary compatibility
                    # SeleniumLibrary needs os.path.dirname(logfile) in log_dir property
                    log_file_path = os.path.join(temp_output_dir, "log.html")
                    variables["${LOGFILE}"] = log_file_path
                    
                    logger.info(f"Created RF context with output directory: {temp_output_dir}")
                    logger.info(f"Set ${{OUTPUTDIR}} variable to: {temp_output_dir}")
                    logger.info(f"Set ${{LOGFILE}} variable to: {log_file_path}")
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
                
            # CRITICAL: Set up BuiltIn library with proper context access
            # This enables Input Password and other context-dependent keywords
            self._setup_builtin_context_access(ctx, namespace)
            
            # Import libraries into the RF namespace
            imported_libraries = []
            if libraries:
                logger.info(f"Importing libraries into RF context: {libraries}")
                for lib_name in libraries:
                    try:
                        # Use correct Robot Framework namespace.import_library API
                        # Signature: import_library(self, name, args=(), alias=None, notify=True)
                        namespace.import_library(lib_name, args=(), alias=None)
                        imported_libraries.append(lib_name)
                        logger.info(f"Successfully imported {lib_name} into RF context using correct API")
                            
                    except Exception as e:
                        logger.warning(f"Failed to import library {lib_name} into RF context: {e}")
                        logger.warning(f"Import error type: {type(e).__name__}")
                        import traceback
                        logger.warning(f"Import traceback: {traceback.format_exc()}")
                        
                        # For Browser Library specifically, try to avoid the problematic import
                        if lib_name == "Browser" and ("list index out of range" in str(e) or "index out of range" in str(e)):
                            logger.info(f"Skipping Browser Library import due to index error - will try alternative approach")
                            continue
                        
                        # For SeleniumLibrary, try with proper arguments for RF context
                        if lib_name == "SeleniumLibrary":
                            logger.info(f"Retrying SeleniumLibrary import with proper RF context configuration")
                            try:
                                # Import with empty arguments - RF context will handle initialization
                                namespace.import_library("SeleniumLibrary", args=())
                                imported_libraries.append(lib_name)
                                logger.info(f"Successfully imported SeleniumLibrary into RF context on retry")
                                continue
                            except Exception as retry_error:
                                logger.warning(f"SeleniumLibrary retry also failed: {retry_error}")
                                # Continue to alternative approach
                        
                        # Try alternative approach with library arguments from library manager
                        try:
                            from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery
                            orchestrator = get_keyword_discovery()
                            if lib_name in orchestrator.library_manager.libraries:
                                lib_info = orchestrator.library_manager.libraries[lib_name]
                                # Try with any library-specific arguments if available
                                lib_args = getattr(lib_info, 'args', ()) or ()
                                namespace.import_library(lib_name, args=lib_args, alias=None)
                                imported_libraries.append(lib_name)
                                logger.info(f"Imported {lib_name} from library manager with args {lib_args}")
                            else:
                                # Library not in manager, try basic import one more time
                                namespace.import_library(lib_name)
                                imported_libraries.append(lib_name)
                                logger.info(f"Imported {lib_name} with basic call")
                        except Exception as fallback_error:
                            logger.warning(f"Fallback import also failed for {lib_name}: {fallback_error}")
                            logger.warning(f"Fallback error type: {type(fallback_error).__name__}")

            # Store context info
            self._session_contexts[session_id] = {
                "context": ctx,
                "variables": variables,
                "namespace": namespace,
                "output": output,
                "suite": suite,
                "created_at": datetime.now(),
                "libraries": libraries or [],
                "imported_libraries": imported_libraries
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
    
    def _execute_any_keyword_generic(self, keyword_name: str, arguments: List[str], namespace) -> Any:
        """
        Execute any keyword using Robot Framework's native keyword resolution.
        
        This is a generic approach that works with any library and avoids the
        'Keyword object has no attribute body' issue in RF 7.x by using direct
        keyword execution instead of run_keyword.
        """
        try:
            # Use RF's native keyword resolution through the namespace
            # This is the most generic approach that works with any library
            
            # Try to resolve the keyword using RF's namespace
            try:
                # Debug: Check available libraries and keywords
                if hasattr(namespace, 'libraries'):
                    lib_names = list(namespace.libraries.keys()) if hasattr(namespace.libraries, 'keys') else ['(unknown format)']
                    logger.info(f"Available libraries in RF namespace: {lib_names}")
                
                keyword = namespace.get_keyword(keyword_name)
                if keyword:
                    logger.info(f"Found keyword '{keyword_name}' via namespace resolution")
                    
                    # Execute the keyword directly using its method
                    if hasattr(keyword, 'method') and callable(keyword.method):
                        return keyword.method(*arguments)
                    elif hasattr(keyword, 'run'):
                        # Some keywords have a run method
                        return keyword.run(*arguments)
                    else:
                        # Fallback: try to get the actual callable
                        if hasattr(keyword, '_handler') and callable(keyword._handler):
                            return keyword._handler(*arguments)
                        elif hasattr(keyword, 'keyword') and callable(keyword.keyword):
                            return keyword.keyword(*arguments)
                            
            except Exception as e:
                logger.debug(f"Namespace resolution failed for {keyword_name}: {e}")
            
            # HYBRID APPROACH: For Input Password specifically, use library manager instance with RF context
            if keyword_name.lower() == "input password":
                logger.info(f"Using hybrid approach for Input Password with RF context support")
                try:
                    from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery
                    orchestrator = get_keyword_discovery()
                    if "SeleniumLibrary" in orchestrator.library_manager.libraries:
                        lib_instance = orchestrator.library_manager.libraries["SeleniumLibrary"]
                        
                        # Check if Input Password method exists directly
                        if hasattr(lib_instance, 'input_password'):
                            logger.info(f"Found input_password method in SeleniumLibrary instance")
                            
                            # Execute with RF context available for BuiltIn calls
                            # The key is that we already have RF context set up, so BuiltIn calls should work
                            return lib_instance.input_password(*arguments)
                        else:
                            logger.warning(f"input_password method not found in SeleniumLibrary instance")
                            # List available methods for debugging
                            methods = [attr for attr in dir(lib_instance) if not attr.startswith('_') and callable(getattr(lib_instance, attr))]
                            logger.info(f"Available methods in SeleniumLibrary: {methods[:10]}...")  # Show first 10 methods
                except Exception as e:
                    logger.warning(f"Hybrid approach failed for Input Password: {e}")
            
            # Fallback: Manual library search
            from robot.running import EXECUTION_CONTEXTS
            ctx = EXECUTION_CONTEXTS.current
            
            if ctx and hasattr(ctx, 'namespace') and hasattr(ctx.namespace, 'libraries'):
                # Handle different RF versions - libraries might be dict or other collection
                libraries = ctx.namespace.libraries
                if hasattr(libraries, 'items'):
                    # It's a dict-like object
                    for lib_name, lib_instance in libraries.items():
                        try:
                            self._try_execute_from_library(keyword_name, arguments, lib_name, lib_instance)
                        except Exception as e:
                            logger.debug(f"Failed to execute {keyword_name} from {lib_name}: {e}")
                            continue
                elif hasattr(libraries, '__iter__'):
                    # It's an iterable (like odict_values)
                    for lib_instance in libraries:
                        if hasattr(lib_instance, '__class__'):
                            lib_name = lib_instance.__class__.__name__
                            try:
                                result = self._try_execute_from_library(keyword_name, arguments, lib_name, lib_instance)
                                if result is not None:
                                    return result
                            except Exception as e:
                                logger.debug(f"Failed to execute {keyword_name} from {lib_name}: {e}")
                                continue
            
            # If we get here, try the final fallback approach
            return self._final_fallback_execution(keyword_name, arguments)
            
        except Exception as e:
            logger.error(f"Generic keyword execution failed for {keyword_name}: {e}")
            raise
    
    def _try_execute_from_library(self, keyword_name: str, arguments: List[str], lib_name: str, lib_instance) -> Any:
        """Try to execute keyword from a specific library instance."""
        # Check if this library has the keyword
        if hasattr(lib_instance, keyword_name.replace(' ', '_').lower()):
            method = getattr(lib_instance, keyword_name.replace(' ', '_').lower())
            if callable(method):
                logger.info(f"Executing {keyword_name} from {lib_name} via direct method")
                return method(*arguments)
        
        # Try different naming conventions
        for method_name in [
            keyword_name.replace(' ', '_'),
            keyword_name.replace(' ', '').lower(),
            keyword_name.lower().replace(' ', '_')
        ]:
            if hasattr(lib_instance, method_name):
                method = getattr(lib_instance, method_name)
                if callable(method):
                    logger.info(f"Executing {keyword_name} as {method_name} from {lib_name}")
                    return method(*arguments)
        
        return None
    
    def _final_fallback_execution(self, keyword_name: str, arguments: List[str]) -> Any:
        """Final fallback execution using BuiltIn library."""
        from robot.libraries.BuiltIn import BuiltIn
        builtin = BuiltIn()
        
        # Check if it's a BuiltIn method
        method_name = keyword_name.replace(' ', '_').lower()
        if hasattr(builtin, method_name):
            method = getattr(builtin, method_name)
            if callable(method):
                logger.info(f"Executing {keyword_name} as BuiltIn method")
                return method(*arguments)
        
        # If nothing worked, raise an error
        raise RuntimeError(f"Keyword '{keyword_name}' could not be resolved or executed")
    
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
            
            # No special handling needed - Input Password will work through normal RF context now
            
            # Direct approach: Call BuiltIn methods directly
            # This avoids the run_keyword complexity and uses RF's execution context
            from robot.libraries.BuiltIn import BuiltIn
            
            builtin = BuiltIn()
            
            # Use generic keyword execution approach that works with RF 7.x
            # Instead of run_keyword, use direct keyword resolution and execution
            result = self._execute_any_keyword_generic(keyword_name, arguments, namespace)
            
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
    
    def _setup_builtin_context_access(self, context, namespace):
        """Set up BuiltIn library with proper context access for Input Password and similar keywords.
        
        This is a general solution that enables any keyword requiring RF context access.
        """
        try:
            from robot.libraries.BuiltIn import BuiltIn
            
            # Get or create BuiltIn instance 
            builtin_instance = BuiltIn()
            
            # Set up BuiltIn with proper context access
            # BuiltIn.set_log_level needs: self._context.output and self._namespace.variables.set_global
            builtin_instance._context = context
            builtin_instance._namespace = namespace
            
            logger.info("✅ BuiltIn library configured with RF context access for Input Password support")
            
        except Exception as e:
            logger.warning(f"Failed to setup BuiltIn context access: {e}")
    
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
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
    # Avoid importing robot.running.Keyword to prevent shadowing model classes in some RF versions
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
                
                # Create compatible variables with set_global/start_keyword methods for BuiltIn/RF
                variables = create_compatible_variables(original_variables)
                # Inject Robot built-in variables for runtime scalar replacement (e.g., ${True})
                try:
                    from robotmcp.components.variables.variable_resolver import (
                        VariableResolver,
                    )
                    _vr = VariableResolver()
                    for k, v in _vr.builtin_variables.items():
                        variables[k] = v
                    # Common-case synonyms
                    for k, v in {
                        "${True}": True,
                        "${true}": True,
                        "${False}": False,
                        "${false}": False,
                        "${None}": None,
                        "${none}": None,
                    }.items():
                        variables[k] = v
                except Exception:
                    pass
                
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
                
                # Create minimal namespace; assign compatible variables so RF internals use it
                original_namespace = Namespace(original_variables, suite, suite.resource, Languages())
                # Replace Variables with our compatible wrapper
                original_namespace.variables = variables
                namespace = original_namespace
                
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
                
                # Ensure BuiltIn is available for user keywords and variable ops
                try:
                    original_namespace.import_library("BuiltIn")
                except Exception:
                    pass

                # Start execution context (must not be dry_run to actually execute keywords)
                if output:
                    ctx = EXECUTION_CONTEXTS.start_suite(suite, namespace, output, dry_run=False)
                else:
                    # Fallback: create a minimal context without output
                    from robot.running.context import _ExecutionContext
                    ctx = _ExecutionContext(suite, namespace, output, dry_run=False)
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
                
            # BuiltIn context access: avoid setting internal attributes in RF7+.
            # BuiltIn will operate correctly when an execution context is active.
            
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
                                namespace.import_library(lib_name, args=tuple(lib_args), alias=None)
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

            # Apply RF search order and initialize suite/test scopes in Namespace and Context
            try:
                if imported_libraries:
                    namespace.set_search_order(imported_libraries)
                    logger.info(f"Set RF Namespace search order: {imported_libraries}")
                # Initialize variable and library scopes for suite and test
                namespace.start_suite()
                namespace.start_test()
                # Also start a real running+result test in ExecutionContext for StatusReporter/BuiltIn
                try:
                    from robot.running.model import TestCase as RunTest
                    from robot.result.model import TestCase as ResTest
                    run_test = RunTest(name=f"MCP_Test_{session_id}")
                    res_test = ResTest(name=f"MCP_Test_{session_id}")
                    ctx.start_test(run_test, res_test)
                    logger.info("Started ExecutionContext test for session")
                except Exception as e:
                    logger.debug(f"Could not start ExecutionContext test: {e}")
                logger.info("Initialized Namespace + Context suite/test scopes for session context")
            except Exception as e:
                logger.debug(f"Namespace initialization failed: {e}")

            # Store context info
            # Prepare result model holders (for future RF runner/BuiltIn status reporting integration)
            try:
                from robot.result.model import TestSuite as ResultSuite, TestCase as ResultTest
                result_suite = ResultSuite(name=suite.name)
                result_test = ResultTest(name=f"MCP_Test_{session_id}")
            except Exception:
                result_suite = None
                result_test = None

            self._session_contexts[session_id] = {
                "context": ctx,
                "variables": variables,
                "namespace": namespace,
                "output": output,
                "suite": suite,
                "result_suite": result_suite,
                "result_test": result_test,
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

            # Ensure the RF context is healthy after any prior suite execution.
            # Rebuild missing output/result parent as needed to avoid NoneType.is_logged errors.
            try:
                from robot.conf import RobotSettings
                from robot.output import Output
                from robot.output import logger as rf_logger
                import tempfile, os

                # Validate current execution context
                from robot.running.context import EXECUTION_CONTEXTS as _CTX
                active_ctx = _CTX.current or ctx

                # Ensure output is available and has is_logged
                needs_output = not getattr(active_ctx, 'output', None) or not hasattr(getattr(active_ctx, 'output', None), 'message')
                if needs_output:
                    temp_output_dir = tempfile.mkdtemp(prefix="rf_mcp_")
                    settings = RobotSettings(outputdir=temp_output_dir, output=None)
                    active_ctx.output = Output(settings)
                    # Update ${OUTPUTDIR} and ${LOGFILE} for library compatibility
                    variables["${OUTPUTDIR}"] = temp_output_dir
                    variables["${LOGFILE}"] = os.path.join(temp_output_dir, "log.html")

                # Ensure global LOGGER has an output file registered
                try:
                    if getattr(rf_logger.LOGGER, "_output_file", None) is None:
                        temp_output_dir = variables.get("${OUTPUTDIR}")
                        if not temp_output_dir:
                            temp_output_dir = tempfile.mkdtemp(prefix="rf_mcp_")
                            variables["${OUTPUTDIR}"] = temp_output_dir
                            variables["${LOGFILE}"] = os.path.join(temp_output_dir, "log.html")
                        settings = RobotSettings(outputdir=temp_output_dir, output=None)
                        # Creating Output registers output file with LOGGER
                        new_output = Output(settings)
                        # Prefer to use the most recent output on the context too
                        active_ctx.output = new_output
                except Exception:
                    pass

                # Ensure a parent result test exists for StatusReporter
                try:
                    from robot.result.model import TestCase as ResultTest
                    parent_test = context_info.get("result_test")
                    if parent_test is None:
                        parent_test = ResultTest(name=f"MCP_Test_{session_id}")
                        context_info["result_test"] = parent_test
                except Exception:
                    pass
            except Exception:
                # Best-effort self-heal; continue
                pass

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
                session_id, keyword_name, arguments, namespace, variables, assign_to
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

        # Try generic RF-level resolution with run_keyword (handles decorated names)
        try:
            logger.info(f"FALLBACK: BuiltIn.run_keyword('{keyword_name}', args={len(arguments)})")
            return builtin.run_keyword(keyword_name, *arguments)
        except Exception as e:
            logger.debug(f"BuiltIn.run_keyword failed for {keyword_name}: {e}")

        # Check if it's a BuiltIn method (direct call)
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
        session_id: str,
        keyword_name: str,
        arguments: List[str], 
        namespace: Namespace,
        variables: Variables,
        assign_to: Optional[Union[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Execute keyword using RF's native runner for discovery + argument resolution.
        """
        try:
            logger.info(f"RF NATIVE: Executing {keyword_name} with args: {arguments}")
            # Generic path: try direct resolution/dispatch (often works for BuiltIn)
            try:
                generic_result = self._execute_any_keyword_generic(
                    keyword_name, arguments, namespace
                )
                # If generic path succeeded, proceed with assignment/variables and return
                assigned_vars = {}
                if assign_to and generic_result is not None:
                    assigned_vars = self._handle_variable_assignment(
                        assign_to, generic_result, variables
                    )
                current_vars = {}
                try:
                    if hasattr(variables, 'store'):
                        current_vars = dict(variables.store.data)
                    elif hasattr(variables, 'current') and hasattr(variables.current, 'store'):
                        current_vars = dict(variables.current.store.data)
                except Exception as e:
                    logger.debug(f"Could not extract variables: {e}")
                    current_vars = {}
                return {
                    "success": True,
                    "result": generic_result,
                    "output": str(generic_result) if generic_result is not None else "OK",
                    "variables": current_vars,
                    "assigned_variables": assigned_vars,
                }
            except Exception:
                pass
            # Evaluate expression normalization: support ${var} and bare variable names
            try:
                if keyword_name.strip().lower() == "evaluate" and arguments:
                    import re
                    expr = arguments[0]
                    if isinstance(expr, str):
                        # Convert ${var.suffix} -> $var.suffix for RF Evaluate semantics
                        expr2 = re.sub(r"\$\{([A-Za-z_]\w*)([^}]*)\}", r"$\1\2", expr)
                        # If expression starts with a bare variable name that exists in session vars, prefix with $.
                        try:
                            bare = re.match(r"^\s*([A-Za-z_]\w+)\b(.*)$", expr2)
                            if bare:
                                name, rest = bare.group(1), bare.group(2)
                                # Build a set of known names without ${}
                                known = set()
                                try:
                                    for k in (session_variables or {}).keys():
                                        kstr = str(k)
                                        if kstr.startswith("${") and kstr.endswith("}"):
                                            known.add(kstr[2:-1])
                                        else:
                                            known.add(kstr)
                                except Exception:
                                    pass
                                # Fallback: probe RF Variables by normalized name
                                in_variables = False
                                if name not in known:
                                    try:
                                        _ = variables[f"${{{name}}}"]
                                        in_variables = True
                                    except Exception:
                                        in_variables = False
                                if (name in known or in_variables) and not expr2.lstrip().startswith("$"):
                                    expr2 = f"${name}{rest}"
                        except Exception:
                            pass

                        # Also normalize bracketed indexing like [item] or [item[0]] to use $item
                        try:
                            # Rebuild known set if needed
                            kn = set()
                            for k in (session_variables or {}).keys():
                                ks = str(k)
                                if ks.startswith("${") and ks.endswith("}"):
                                    kn.add(ks[2:-1])
                                else:
                                    kn.add(ks)
                            # Probe variables store for additional names
                            for candidate in list(kn):
                                pass
                            # Apply substitutions for each known name
                            import re as _re
                            for nm in kn or []:
                                pat_simple = _re.compile(r"\[" + _re.escape(nm) + r"\]")
                                expr2 = pat_simple.sub("[${}".format(nm) + "]", expr2)
                                pat_index = _re.compile(r"\[" + _re.escape(nm) + r"\[(.*?)\]\]")
                                expr2 = pat_index.sub(r"[$" + nm + r"[\1]]", expr2)
                        except Exception:
                            pass
                        arguments = [expr2] + list(arguments[1:])
            except Exception:
                pass

            # Resolve via Namespace → get_runner → run with proper models
            try:
                from robot.running.model import Keyword as RunKeyword
                from robot.result.model import Keyword as ResultKeyword
                from robot.running.context import EXECUTION_CONTEXTS

                runner = namespace.get_runner(keyword_name)
                ctx = EXECUTION_CONTEXTS.current
                # IMPORTANT: Do not pre-split name=value here. Let RF resolve named args
                # based on the keyword's real signature to avoid passing unexpected kwargs
                # (e.g., BuiltIn.Set Variable should treat 'token=${auth}' as positional text).
                # Normalize Windows absolute paths to use forward slashes to avoid RF de-escaping
                # of sequences like \t, \r, \b in file paths.
                def _normalize_arg(a: Any) -> Any:
                    try:
                        import os, re
                        if isinstance(a, str) and os.name == 'nt':
                            # Match Windows absolute paths like C:\folder\file
                            if re.match(r'^[A-Za-z]:\\', a):
                                from pathlib import Path
                                try:
                                    return Path(a).as_posix()
                                except Exception:
                                    return a.replace('\\\\', '/')
                    except Exception:
                        pass
                    return a

                pos_args: list[Any] = [_normalize_arg(arg) for arg in list(arguments)]
                named_args: dict[str, object] = {}
                # Build running/data and result keyword models
                data_kw = RunKeyword(
                    name=keyword_name,
                    args=tuple(pos_args),
                    named_args=named_args,  # pass empty dict when no named args
                    assign=tuple(assign_to) if isinstance(assign_to, list) else (() if not assign_to else (assign_to,)),
                )
                # Build result keyword and bind to a parent test result to satisfy StatusReporter
                res_kw = ResultKeyword(
                    name=keyword_name,
                    args=tuple(pos_args),
                    assign=tuple(assign_to) if isinstance(assign_to, list) else (() if not assign_to else (assign_to,)),
                )
                try:
                    ctx_info = self._session_contexts.get(session_id)
                    parent_test = ctx_info.get("result_test") if ctx_info else None
                    if parent_test is None:
                        # Self-heal: create a minimal ResultTest to satisfy StatusReporter
                        from robot.result.model import TestCase as ResultTest
                        parent_test = ResultTest(name=f"MCP_Test_{session_id}")
                        ctx_info["result_test"] = parent_test
                    res_kw.parent = parent_test
                except Exception:
                    pass
                result = runner.run(data_kw, res_kw, ctx)
            except Exception as runner_error:
                logger.error(
                    f"RF runner failed for '{keyword_name}' with args {arguments}: {runner_error}"
                )
                # Fallback: BuiltIn.run_keyword under active context
                try:
                    from robot.libraries.BuiltIn import BuiltIn
                    # Ensure BuiltIn uses test-level result, not previous step
                    saved_steps = []
                    try:
                        saved_steps = list(ctx.steps)
                        ctx.steps.clear()
                    except Exception:
                        pass
                    try:
                        # Force context.test to a valid result test
                        try:
                            ctx_info = self._session_contexts.get(session_id)
                            from robot.result.model import TestCase as ResultTest
                            if ctx_info:
                                parent_test = ctx_info.get("result_test")
                                if parent_test is None:
                                    parent_test = ResultTest(name=f"MCP_Test_{session_id}")
                                    ctx_info["result_test"] = parent_test
                                ctx.test = parent_test
                        except Exception:
                            pass
                        # Apply the same Windows path normalization for BuiltIn fallback
                        norm_args = [_normalize_arg(a) for a in list(arguments)]
                        result = BuiltIn().run_keyword(keyword_name, *norm_args)
                    finally:
                        try:
                            ctx.steps[:] = saved_steps
                        except Exception:
                            pass
                except Exception as e:
                    logger.error(
                        f"BuiltIn.run_keyword fallback failed for '{keyword_name}': {e}"
                    )
                    raise
            
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
            
            # Attach contextual hints
            try:
                from robotmcp.utils.hints import HintContext, generate_hints
                ctx = HintContext(
                    session_id=session_id,
                    keyword=keyword_name,
                    arguments=list(arguments or []),
                    error_text=str(e),
                )
                hints = generate_hints(ctx)
            except Exception:
                hints = []

            return {
                "success": False,
                "error": f"Keyword execution failed: {str(e)}",
                "keyword": keyword_name,
                "arguments": arguments,
                "hints": hints,
            }
    
# Fallback method removed - using simplified approach
    
    def _setup_builtin_context_access(self, context, namespace):
        """No-op: BuiltIn picks up active context automatically; avoid touching internals in RF7+."""
        return
    
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

    # --- New: Resource and custom library management + discovery ---

    def import_resource_for_session(self, session_id: str, resource_path: str) -> Dict[str, Any]:
        """Import a Robot Framework resource file into the session Namespace."""
        try:
            if session_id not in self._session_contexts:
                create = self.create_context_for_session(session_id)
                if not create.get("success"):
                    return create
            namespace = self._session_contexts[session_id]["namespace"]
            from pathlib import Path
            # Resolve relative and platform-specific paths robustly and pass POSIX-style to RF
            p = Path(resource_path)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            path_str = p.as_posix()
            namespace.import_resource(path_str)
            ctx_info = self._session_contexts[session_id]
            resources = ctx_info.setdefault("resources", [])
            if resource_path not in resources:
                resources.append(resource_path)
            return {"success": True, "session_id": session_id, "resource": resource_path}
        except Exception as e:
            logger.error(f"Failed to import resource '{resource_path}': {e}")
            return {"success": False, "error": str(e), "session_id": session_id}

    def import_library_for_session(
        self,
        session_id: str,
        library_name_or_path: str,
        args: tuple = (),
        alias: str | None = None,
    ) -> Dict[str, Any]:
        """Import a custom Robot Framework library (by module name or file path) into the session Namespace."""
        try:
            if session_id not in self._session_contexts:
                create = self.create_context_for_session(session_id)
                if not create.get("success"):
                    return create
            namespace = self._session_contexts[session_id]["namespace"]
            from pathlib import Path
            name = library_name_or_path
            try:
                p = Path(library_name_or_path)
                if not p.is_absolute():
                    p = (Path.cwd() / p).resolve()
                if p.exists():
                    name = p.as_posix()
            except Exception:
                pass
            namespace.import_library(name, args=tuple(args or ()), alias=alias)
            ctx_info = self._session_contexts[session_id]
            libs = ctx_info.setdefault("imported_libraries", [])
            if library_name_or_path not in libs:
                libs.append(library_name_or_path)
            return {
                "success": True,
                "session_id": session_id,
                "library": library_name_or_path,
                "alias": alias,
            }
        except Exception as e:
            logger.error(f"Failed to import library '{library_name_or_path}': {e}")
            return {"success": False, "error": str(e), "session_id": session_id}

    def list_available_keywords(self, session_id: str) -> Dict[str, Any]:
        """List available keyword names from imported libraries and resources in this session."""
        try:
            if session_id not in self._session_contexts:
                return {"success": False, "error": "No RF context for session", "session_id": session_id}
            namespace = self._session_contexts[session_id]["namespace"]

            library_keywords = []
            for lib in list(namespace.libraries):
                try:
                    for kw in getattr(lib, "keywords", []) or []:
                        library_keywords.append({
                            "name": kw.name,
                            "full_name": getattr(kw, "full_name", kw.name),
                            "library": getattr(lib, "name", None) or getattr(lib, "__class__", type(lib)).__name__,
                        })
                except Exception:
                    continue

            # Resource keywords: use LibDoc for resources tracked in context
            resource_keywords = []
            try:
                from robot.libdoc import LibraryDocumentation
                for res in self._session_contexts[session_id].get("resources", []) or []:
                    try:
                        doc = LibraryDocumentation(res)
                        for kw in doc.keywords:
                            resource_keywords.append({
                                "name": kw.name,
                                "full_name": kw.name,
                                "resource": res,
                            })
                    except Exception:
                        continue
            except Exception:
                pass

            return {
                "success": True,
                "session_id": session_id,
                "libraries_count": len(list(namespace.libraries)),
                "library_keywords": library_keywords,
                "resource_keywords": resource_keywords,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "session_id": session_id}

    def get_keyword_documentation(self, session_id: str, keyword_name: str) -> Dict[str, Any]:
        """Get documentation and signature for a keyword available in this session.

        Attempts library handlers first, then resource keywords via LibDoc.
        """
        try:
            if session_id not in self._session_contexts:
                return {"success": False, "error": "No RF context for session", "session_id": session_id}
            namespace = self._session_contexts[session_id]["namespace"]

            # Search library handlers
            for lib in list(namespace.libraries):
                for kw in getattr(lib, "keywords", []) or []:
                    if kw.name.lower() == keyword_name.lower():
                        info = {
                            "success": True,
                            "session_id": session_id,
                            "name": kw.name,
                            "source": getattr(lib, "name", None) or getattr(lib, "__class__", type(lib)).__name__,
                            "doc": getattr(kw, "doc", ""),
                            "args": [str(a) for a in (getattr(kw, "args", []) or [])],
                            "type": "library",
                        }
                        return info

            # Search resources via LibDoc
            try:
                from robot.libdoc import LibraryDocumentation
                for res in self._session_contexts[session_id].get("resources", []) or []:
                    try:
                        doc = LibraryDocumentation(res)
                        for kw in doc.keywords:
                            if kw.name.lower() == keyword_name.lower():
                                return {
                                    "success": True,
                                    "session_id": session_id,
                                    "name": kw.name,
                                    "source": res,
                                    "doc": kw.doc,
                                    "args": [str(a) for a in kw.args],
                                    "type": "resource",
                                }
                    except Exception:
                        continue
            except Exception:
                pass

            return {
                "success": False,
                "error": f"Keyword '{keyword_name}' not found in session",
                "session_id": session_id,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "session_id": session_id}


# Global instance for use throughout the application
_rf_native_context_manager = None

def get_rf_native_context_manager() -> RobotFrameworkNativeContextManager:
    """Get the global RF native context manager instance."""
    global _rf_native_context_manager
    if _rf_native_context_manager is None:
        _rf_native_context_manager = RobotFrameworkNativeContextManager()
    return _rf_native_context_manager

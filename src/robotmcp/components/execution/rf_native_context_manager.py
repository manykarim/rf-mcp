"""Robot Framework native execution context manager for MCP keywords."""

import logging
import os as _os
import sys as _sys
import threading as _threading
import uuid
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Import Robot Framework native components
try:
    from robot.running.context import EXECUTION_CONTEXTS, _ExecutionContext
    from robot.running.namespace import Namespace
    from robot.variables import Variables
    from robot.variables.scopes import VariableScopes
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
    from robot.conf import Languages, RobotSettings
    RF_NATIVE_AVAILABLE = True
    logger.info("Robot Framework native components imported successfully")
except ImportError as e:
    RF_NATIVE_AVAILABLE = False
    logger.error(f"Robot Framework native components not available: {e}")


# ── P2: Thread-safe fd 1 redirect with reference counting ────────────
# A process-global lock + counter ensures that concurrent RF executions
# (dispatched via asyncio.to_thread) keep fd 1 redirected until the LAST
# thread exits.  Without this, Thread A restoring fd 1 while Thread B is
# still inside RF would expose the MCP transport to contamination.
_stdout_redirect_lock = _threading.Lock()
_stdout_redirect_count = 0
_stdout_saved_fd = None


def _suppress_stdout():
    """Context manager that redirects fd 1 → fd 2 (stderr) to prevent RF console
    output from corrupting the MCP stdio transport.

    RF's Output class writes test progress (suite/test names, dots) directly
    to file descriptor 1 via a C-level buffered file object.  In MCP stdio
    mode fd 1 IS the JSON-RPC transport, so any non-JSON bytes corrupt the
    protocol and cause the client to hang waiting for a valid response.

    Uses reference counting so concurrent threads all keep fd 1 redirected
    until the last thread exits.  The lock is only held during fd operations,
    never during the (potentially slow) RF keyword execution.
    """
    import contextlib

    @contextlib.contextmanager
    def _redirect():
        global _stdout_redirect_count, _stdout_saved_fd

        redirect_ok = False
        with _stdout_redirect_lock:
            _stdout_redirect_count += 1
            if _stdout_redirect_count == 1:
                # First redirector — save original fd 1 and redirect
                try:
                    _stdout_saved_fd = _os.dup(1)
                    _os.dup2(2, 1)  # fd 1 → stderr
                    redirect_ok = True
                except OSError:
                    _stdout_redirect_count -= 1
            else:
                # Another thread already redirected — just piggyback
                redirect_ok = True

        if not redirect_ok:
            # Could not redirect (OSError on dup) — execute unprotected
            yield
            return

        try:
            yield
        finally:
            with _stdout_redirect_lock:
                _stdout_redirect_count -= 1
                if _stdout_redirect_count == 0 and _stdout_saved_fd is not None:
                    # Last redirector — restore fd 1 to MCP transport
                    try:
                        _os.dup2(_stdout_saved_fd, 1)
                    except OSError:
                        pass
                    try:
                        _os.close(_stdout_saved_fd)
                    except OSError:
                        pass
                    _stdout_saved_fd = None

    return _redirect()


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
                # Create VariableScopes for proper test/suite scope isolation (ADR-005)
                variables = VariableScopes(RobotSettings())
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
                
                # Create a basic test suite with proper output directory setup
                suite = TestSuite(name=f"MCP_Session_{session_id}")
                
                # Set a minimal source path to avoid full_name issues  
                from pathlib import Path
                suite.source = Path(f"MCP_Session_{session_id}.robot")
                
                # Ensure suite has a resource with required attributes
                from robot.running.resourcemodel import ResourceFile
                suite.resource = ResourceFile(source=suite.source)
                
                # Create namespace with VariableScopes for proper scope management
                namespace = Namespace(variables, suite, suite.resource, Languages())
                
                # Create simple output with proper output directory for Browser Library
                try:
                    import tempfile
                    import os
                    
                    # Create temporary output directory for Browser Library
                    temp_output_dir = tempfile.mkdtemp(prefix="rf_mcp_")
                    
                    # Create settings with output directory - this fixes Browser Library initialization
                    # console='none' prevents RF from writing test progress to stdout,
                    # which would corrupt the MCP stdio transport (fd 1 = JSON-RPC).
                    settings = RobotSettings(outputdir=temp_output_dir, output=None, console='none')
                    output = Output(settings)

                    # Library listeners expect a suite scope before registering any
                    try:
                        output.library_listeners.new_suite_scope()
                    except Exception as listener_err:
                        logger.debug(
                            f"Unable to initialize library listener scope: {listener_err}"
                        )
                    
                    # Set OUTPUTDIR variable for Browser Library compatibility
                    # Browser Library uses BuiltIn().get_variable_value("${OUTPUTDIR}")
                    variables["${OUTPUTDIR}"] = temp_output_dir
                    
                    # Set LOGFILE variable for SeleniumLibrary compatibility
                    # SeleniumLibrary needs os.path.dirname(logfile) in log_dir property
                    log_file_path = os.path.join(temp_output_dir, "log.html")
                    variables["${LOGFILE}"] = log_file_path

                    # Provide Robot's standard ${OUTPUT} variable (output.xml path)
                    output_file_path = os.path.join(temp_output_dir, "output.xml")
                    variables["${OUTPUT}"] = output_file_path
                    
                    logger.info(f"Created RF context with output directory: {temp_output_dir}")
                    logger.info(f"Set ${{OUTPUTDIR}} variable to: {temp_output_dir}")
                    logger.info(f"Set ${{LOGFILE}} variable to: {log_file_path}")
                    logger.info(f"Set ${{OUTPUT}} variable to: {output_file_path}")
                except Exception:
                    # If Output still fails, try a different approach
                    logger.warning("Could not create Output, using minimal logging")
                    output = None
                
                # Ensure BuiltIn is available for user keywords and variable ops
                try:
                    namespace.import_library("BuiltIn")
                except Exception:
                    pass

                # Start execution context (must not be dry_run to actually execute keywords)
                # Wrap in _suppress_stdout() to prevent RF console output from
                # corrupting the MCP stdio transport (fd 1 = JSON-RPC channel).
                with _suppress_stdout():
                    if output:
                        ctx = EXECUTION_CONTEXTS.start_suite(suite, namespace, output, dry_run=False)
                    else:
                        # Fallback: create a minimal context without output
                        from robot.running.context import _ExecutionContext
                        ctx = _ExecutionContext(suite, namespace, output, dry_run=False)
                        EXECUTION_CONTEXTS._contexts.append(ctx)
                        EXECUTION_CONTEXTS._context = ctx

                # Tag the execution context so tests can clean up deterministically
                try:
                    setattr(ctx, "_mcp_session", session_id)
                except Exception:
                    pass
                
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
            failed_imports: Dict[str, str] = {}  # H2: track failed imports for diagnostics
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
                            logger.warning(f"Skipping Browser Library import due to index error — keywords from Browser will not resolve until a prefixed call (e.g., Browser.New Browser) triggers late import")
                            failed_imports[lib_name] = f"index error during import: {e}"
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
                            failed_imports[lib_name] = str(fallback_error)

            # Apply RF search order and initialize suite/test scopes in Namespace and Context.
            # Wrap in _suppress_stdout() because ctx.start_test() writes the test name
            # to fd 1 via the Output console writer, which would corrupt MCP stdio.
            with _suppress_stdout():
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
                "imported_libraries": imported_libraries,
                "failed_imports": failed_imports,
                # ADR-005: multi-test cycling tracking
                "current_run_test": None,
                "current_res_test": None,
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
                from robot.output import logger as rf_logger
                import tempfile, os

                # Validate current execution context
                from robot.running.context import EXECUTION_CONTEXTS as _CTX
                active_ctx = _CTX.current or ctx

                # Ensure output is available and has is_logged
                needs_output = not getattr(active_ctx, 'output', None) or not hasattr(getattr(active_ctx, 'output', None), 'message')
                if needs_output:
                    temp_output_dir = tempfile.mkdtemp(prefix="rf_mcp_")
                    settings = RobotSettings(outputdir=temp_output_dir, output=None, console='none')
                    active_ctx.output = Output(settings)
                    try:
                        active_ctx.output.library_listeners.new_suite_scope()
                    except Exception as listener_err:
                        logger.debug(
                            f"Unable to initialize library listener scope during refresh: {listener_err}"
                        )
                    # Update ${OUTPUTDIR}, ${OUTPUT}, and ${LOGFILE} for library compatibility
                    variables["${OUTPUTDIR}"] = temp_output_dir
                    variables["${OUTPUT}"] = os.path.join(temp_output_dir, "output.xml")
                    variables["${LOGFILE}"] = os.path.join(temp_output_dir, "log.html")

                # Ensure global LOGGER has an output file registered
                try:
                    if getattr(rf_logger.LOGGER, "_output_file", None) is None:
                        try:
                            temp_output_dir = variables["${OUTPUTDIR}"]
                        except (KeyError, Exception):
                            temp_output_dir = None
                        if not temp_output_dir:
                            temp_output_dir = tempfile.mkdtemp(prefix="rf_mcp_")
                            variables["${OUTPUTDIR}"] = temp_output_dir
                            variables["${OUTPUT}"] = os.path.join(temp_output_dir, "output.xml")
                            variables["${LOGFILE}"] = os.path.join(temp_output_dir, "log.html")
                        settings = RobotSettings(outputdir=temp_output_dir, output=None, console='none')
                        # Creating Output registers output file with LOGGER
                        new_output = Output(settings)
                        try:
                            new_output.library_listeners.new_suite_scope()
                        except Exception as listener_err:
                            logger.debug(
                                f"Unable to initialize library listener scope during logger setup: {listener_err}"
                            )
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
            
            # Use RF's native argument resolution.
            # Suppress stdout to prevent RF console output from corrupting
            # the MCP stdio transport (runner.run writes test progress to fd 1).
            with _suppress_stdout():
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
        Execute any keyword using Robot Framework 7.x native keyword resolution.

        Resolution order:
        1. namespace.get_runner() — the canonical RF 7 API
        2. Direct method lookup on the actual library Python instances
        3. BuiltIn.run_keyword() — ultimate fallback
        """
        try:
            # ── 1. RF-native resolution via Namespace.get_runner() ──────────
            # get_runner() is the correct API in RF 7.x (get_keyword() does not exist)
            if hasattr(namespace, 'get_runner'):
                try:
                    runner = namespace.get_runner(keyword_name)
                    if runner:
                        logger.info(f"Found keyword '{keyword_name}' via namespace.get_runner()")

                        # Build lightweight RF keyword models for the runner
                        from robot.running.model import Keyword as RunKeyword
                        from robot.result.model import Keyword as ResultKeyword

                        data_kw = RunKeyword(name=keyword_name, args=tuple(arguments))
                        res_kw = ResultKeyword(name=keyword_name, args=tuple(arguments))
                        ctx = EXECUTION_CONTEXTS.current
                        return runner.run(data_kw, res_kw, ctx)
                except Exception as e:
                    # P1: If RF resolved the keyword and it entered StatusReporter
                    # but the execution itself failed (assertion error, wrong args,
                    # keyword-not-found via InvalidKeywordRunner, etc.), the error
                    # comes out as ExecutionStatus.  Re-raise immediately — retrying
                    # through alternative paths will just repeat the same failure
                    # with extra StatusReporter cycles (console output to fd 1).
                    from robot.errors import ExecutionStatus
                    if isinstance(e, ExecutionStatus):
                        raise
                    logger.debug(f"namespace.get_runner() failed for '{keyword_name}': {e}")

            # ── 2. Direct method lookup on actual library instances ──────────
            # namespace.libraries yields TestLibrary wrappers in RF 7.x;
            # access .instance to reach the real Python library object.
            ctx = EXECUTION_CONTEXTS.current

            if ctx and hasattr(ctx, 'namespace') and hasattr(ctx.namespace, 'libraries'):
                for test_library in ctx.namespace.libraries:
                    lib_name = getattr(test_library, 'name', type(test_library).__name__)
                    # Unwrap TestLibrary → actual Python library instance
                    raw_instance = getattr(test_library, 'instance', test_library)
                    try:
                        result = self._try_execute_from_library(
                            keyword_name, arguments, lib_name, raw_instance
                        )
                        if result is not None:
                            return result
                    except Exception as e:
                        logger.debug(f"Direct method lookup failed for '{keyword_name}' in {lib_name}: {e}")
                        continue

            # ── 3. BuiltIn.run_keyword() fallback ───────────────────────────
            return self._final_fallback_execution(keyword_name, arguments)

        except Exception as e:
            logger.error(f"Generic keyword execution failed for '{keyword_name}': {e}")
            raise

    def _try_execute_from_library(self, keyword_name: str, arguments: List[str], lib_name: str, lib_instance) -> Any:
        """Try to execute keyword from an actual Python library instance.

        Args:
            keyword_name: RF keyword name (e.g. "New Browser")
            arguments: positional arguments
            lib_name: library display name for logging
            lib_instance: the raw Python library object (NOT a TestLibrary wrapper)
        """
        # Canonical RF method name: "New Browser" → "new_browser"
        method_name = keyword_name.replace(' ', '_').lower()
        if hasattr(lib_instance, method_name):
            method = getattr(lib_instance, method_name)
            if callable(method):
                logger.info(f"Executing '{keyword_name}' as {lib_name}.{method_name}()")
                return method(*arguments)

        # Try alternative naming conventions
        for alt_name in [
            keyword_name.replace(' ', '_'),           # "New Browser" → "New_Browser"
            keyword_name.replace(' ', '').lower(),     # "New Browser" → "newbrowser"
        ]:
            if alt_name != method_name and hasattr(lib_instance, alt_name):
                method = getattr(lib_instance, alt_name)
                if callable(method):
                    logger.info(f"Executing '{keyword_name}' as {lib_name}.{alt_name}()")
                    return method(*arguments)

        return None

    def _final_fallback_execution(self, keyword_name: str, arguments: List[str]) -> Any:
        """Final fallback execution using BuiltIn library."""
        from robot.libraries.BuiltIn import BuiltIn
        builtin = BuiltIn()

        # BuiltIn.run_keyword delegates to the current RF namespace
        try:
            logger.info(f"FALLBACK: BuiltIn.run_keyword('{keyword_name}', args={len(arguments)})")
            return builtin.run_keyword(keyword_name, *arguments)
        except Exception as e:
            logger.debug(f"BuiltIn.run_keyword failed for '{keyword_name}': {e}")

        # Direct BuiltIn method call (e.g. "Log" → builtin.log)
        method_name = keyword_name.replace(' ', '_').lower()
        if hasattr(builtin, method_name):
            method = getattr(builtin, method_name)
            if callable(method):
                logger.info(f"Executing '{keyword_name}' as BuiltIn.{method_name}()")
                return method(*arguments)

        raise RuntimeError(f"Keyword '{keyword_name}' could not be resolved or executed")
    
    # ── ADR-005: Multi-test context cycling ────────────────────────────

    def start_test_in_context(
        self, session_id: str, test_name: str,
        documentation: str = "", tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Start a new test in the RF execution context.

        Pushes a new test scope onto the variable stack and starts a running/result
        test pair in the ExecutionContext. If a test is already active it is
        auto-ended first.
        """
        ctx_info = self._session_contexts.get(session_id)
        if not ctx_info:
            return {"success": False, "error": f"No context for session {session_id}"}

        ctx = ctx_info.get("context")
        namespace = ctx_info.get("namespace")

        # Auto-end current test if still active
        if ctx_info.get("current_run_test"):
            self.end_test_in_context(session_id)

        try:
            from robot.running.model import TestCase as RunTest
            from robot.result.model import TestCase as ResTest

            run_test = RunTest(name=test_name)
            if documentation:
                run_test.doc = documentation
            if tags:
                run_test.tags = tags

            res_test = ResTest(name=test_name)

            # Push test scope in Namespace (variable scope) and ExecutionContext.
            # Suppress stdout: ctx.start_test writes test name via console writer.
            with _suppress_stdout():
                namespace.start_test()
                ctx.start_test(run_test, res_test)

            ctx_info["current_run_test"] = run_test
            ctx_info["current_res_test"] = res_test

            logger.info(f"Started RF test '{test_name}' for session {session_id}")
            return {"success": True, "test_name": test_name}

        except Exception as e:
            logger.error(f"Failed to start test '{test_name}' in context: {e}")
            return {"success": False, "error": str(e)}

    def end_test_in_context(
        self, session_id: str, status: str = "PASS", message: str = "",
    ) -> Dict[str, Any]:
        """End the current test in the RF execution context.

        Pops the test-scoped variables and sets ``PREV_TEST_*`` automatic vars.
        """
        ctx_info = self._session_contexts.get(session_id)
        if not ctx_info:
            return {"success": False, "error": f"No context for session {session_id}"}

        ctx = ctx_info.get("context")
        namespace = ctx_info.get("namespace")
        res_test = ctx_info.get("current_res_test")

        if not res_test:
            return {"success": False, "error": "No active test to end"}

        try:
            res_test.status = status
            if message:
                res_test.message = message

            # Pop test scope — cleans test-scoped variables, sets PREV_TEST_*
            with _suppress_stdout():
                ctx.end_test(res_test)
                namespace.end_test()

            test_name = res_test.name
            ctx_info["current_run_test"] = None
            ctx_info["current_res_test"] = None

            logger.info(f"Ended RF test '{test_name}' ({status}) for session {session_id}")
            return {"success": True, "test_name": test_name, "status": status}

        except Exception as e:
            logger.error(f"Failed to end test in context: {e}")
            return {"success": False, "error": str(e)}

    def ensure_test_active(self, session_id: str, default_name: str = None) -> None:
        """Ensure a test is active. Auto-starts default test if needed (legacy compat)."""
        ctx_info = self._session_contexts.get(session_id)
        if not ctx_info or ctx_info.get("current_run_test"):
            return  # already active or no context
        name = default_name or f"MCP_Test_{session_id}"
        self.start_test_in_context(session_id, name)

    # ── End ADR-005 test cycling ─────────────────────────────────────

    def _execute_with_native_resolution(
        self,
        session_id: str,
        keyword_name: str,
        arguments: List[str],
        namespace: Namespace,
        variables,
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
            except Exception as generic_err:
                # P1: If the generic path raised ExecutionStatus, the keyword was
                # resolved by RF and executed through StatusReporter — retrying
                # via the outer runner.run / BuiltIn fallback will just repeat
                # the same failure with additional StatusReporter cycles.
                from robot.errors import ExecutionStatus
                if isinstance(generic_err, ExecutionStatus):
                    raise
                # M1: Log the error instead of silently swallowing it.
                # This path covers namespace.get_runner(), library method lookup,
                # and BuiltIn.run_keyword() fallback.  Falling through to the
                # outer get_runner() + BuiltIn fallback below is intentional, but
                # the error should be visible for debugging.
                logger.debug(
                    f"Generic keyword execution path failed for '{keyword_name}': "
                    f"{type(generic_err).__name__}: {generic_err}"
                )
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

            # H2: Surface failed library imports when keyword resolution fails.
            # If the keyword's library failed to import during context creation,
            # include that in the error so the user sees the real cause instead
            # of a cryptic "No keyword with name 'X' found" message.
            error_msg = f"Keyword execution failed: {str(e)}"
            ctx_info = self._session_contexts.get(session_id)
            if ctx_info:
                failed_imports = ctx_info.get("failed_imports") or {}
                if failed_imports:
                    failed_list = ", ".join(
                        f"{lib} ({reason})" for lib, reason in failed_imports.items()
                    )
                    error_msg += (
                        f". Note: the following libraries failed to import into "
                        f"the RF namespace: {failed_list}"
                    )

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
                "error": error_msg,
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

    # --- Variable sync helpers ---

    def _snapshot_variables(self, variables: Variables) -> Dict[str, Any]:
        """Best-effort snapshot of RF Variables without decoration."""
        try:
            return dict(variables.as_dict(decoration=False))
        except Exception:
            try:
                # Fallback to VariableStore data
                return dict(getattr(variables, "store").data)
            except Exception:
                return {}

    def _filter_serializable_value(self, value: Any) -> Any:
        """
        Helper to filter individual values for JSON serializability.

        Recursively processes values to ensure they can be serialized by
        Pydantic/FastMCP, converting or rejecting RF internal objects.

        Args:
            value: The value to filter/convert

        Returns:
            Serializable version of the value

        Raises:
            ValueError: If value cannot be made serializable
        """
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._filter_serializable_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._filter_serializable_value(v) for k, v in value.items()}
        else:
            # Try string conversion for non-basic types (e.g., RF internal objects)
            str_value = str(value)
            # Only accept if it's not a Python repr (doesn't start with '<')
            if not str_value.startswith('<'):
                return str_value
            raise ValueError(f"Non-serializable type: {type(value)}")

    def _filter_serializable_variables(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter variables dict to include only JSON-serializable values.

        Converts or excludes Robot Framework internal objects that cannot be
        serialized by Pydantic/FastMCP (e.g., ScalarVariableResolver, other
        RF internal resolver/setter objects).

        This is critical for MCP tool responses that must be JSON-serializable.

        Args:
            variables: Raw variables dict from RF Variables object

        Returns:
            Dict with only serializable values (str, int, float, bool, list, dict, None)
        """
        serializable = {}

        for key, value in variables.items():
            try:
                serializable[key] = self._filter_serializable_value(value)
            except (ValueError, Exception) as e:
                # Skip non-serializable variables with debug logging
                logger.debug(f"Skipping non-serializable variable {key}: {type(value).__name__} - {e}")
                continue

        return serializable

    def _snapshot_variables_with_decoration(self, variables: Variables) -> Dict[str, Any]:
        """Snapshot of RF Variables with proper decoration (${}, @{}, &{}) preserved.

        Handles Robot Framework's variable naming conventions:
        - LIST__name in Python files becomes @{name} (list variable)
        - DICT__name in Python files becomes &{name} (dict variable)
        - Regular names become ${name} (scalar variable)
        """
        try:
            return dict(variables.as_dict(decoration=True))
        except Exception:
            try:
                # Fallback: get raw data and add decoration manually
                raw_vars = dict(getattr(variables, "store").data)
                decorated_vars = {}
                for name, value in raw_vars.items():
                    decorated_name = self._add_variable_decoration(name, value)
                    decorated_vars[decorated_name] = value
                return decorated_vars
            except Exception:
                return {}

    def _add_variable_decoration(self, name: str, value: Any) -> str:
        """Add proper RF variable decoration based on name prefix and value type.

        Robot Framework uses special prefixes in Python variable files:
        - LIST__varname -> @{varname} (list variable)
        - DICT__varname -> &{varname} (dict variable)
        - Regular names -> ${varname} (scalar variable)

        Args:
            name: Variable name (may have LIST__ or DICT__ prefix)
            value: Variable value (used to infer type if no prefix)

        Returns:
            Properly decorated variable name
        """
        # Already decorated - return as-is
        if name.startswith('${') or name.startswith('@{') or name.startswith('&{'):
            return name

        # Handle LIST__ prefix (RF convention for list variables in Python files)
        if name.startswith('LIST__'):
            bare_name = name[6:]  # Remove 'LIST__' prefix
            return f"@{{{bare_name}}}"

        # Handle DICT__ prefix (RF convention for dict variables in Python files)
        if name.startswith('DICT__'):
            bare_name = name[6:]  # Remove 'DICT__' prefix
            return f"&{{{bare_name}}}"

        # Infer type from value if no prefix
        if isinstance(value, (list, tuple)):
            return f"@{{{name}}}"
        elif isinstance(value, dict):
            return f"&{{{name}}}"
        else:
            # Default: scalar variable
            return f"${{{name}}}"

    def _normalize_session_keys(self, name: str) -> Tuple[str, str]:
        """Return bare and decorated variants of a variable name."""
        base = name
        if base.startswith("${") and base.endswith("}"):
            base = base[2:-1]
        decorated = f"${{{base}}}"
        return base, decorated

    def _record_imported_variable_files(self, namespace) -> List[Dict[str, Any]]:
        """Return imported variable file metadata from the namespace."""
        imported = getattr(namespace, "_imported_variable_files", None)
        if not imported:
            return []

        results: List[Dict[str, Any]] = []
        try:
            iterable = imported.values() if hasattr(imported, "values") else imported
        except Exception:
            return []

        for item in iterable or []:
            if not item:
                continue
            try:
                if isinstance(item, tuple) and len(item) >= 1:
                    path, *rest = item
                    args = rest[0] if rest else []
                else:
                    path = getattr(item, "name", None) or getattr(item, "path", None)
                    args = getattr(item, "args", []) if hasattr(item, "args") else []
                if path:
                    results.append({"path": path, "args": list(args) if args else []})
            except Exception:
                continue

        return results

    # --- New: Resource and custom library management + discovery ---

    def import_resource_for_session(self, session_id: str, resource_path: str) -> Dict[str, Any]:
        """Import a Robot Framework resource file into the session Namespace."""
        try:
            if session_id not in self._session_contexts:
                create = self.create_context_for_session(session_id)
                if not create.get("success"):
                    return create

            namespace = self._session_contexts[session_id]["namespace"]
            variables_obj = getattr(namespace, "variables", None)

            # Snapshot variables before import to compute diff later
            before_snapshot: Dict[str, Any] = (
                self._snapshot_variables(variables_obj) if variables_obj else {}
            )

            from pathlib import Path

            # Resolve relative and platform-specific paths robustly and pass POSIX-style to RF
            p = Path(resource_path)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            path_str = p.as_posix()

            namespace.import_resource(path_str)

            # Track imported resources on the context
            ctx_info = self._session_contexts[session_id]
            resources = ctx_info.setdefault("resources", [])
            if resource_path not in resources:
                resources.append(resource_path)

            # Diff and sync variable file outputs back into session storage
            loaded_variables: Dict[str, Any] = {}
            if variables_obj is not None:
                after_snapshot = self._snapshot_variables(variables_obj)
                diff = {
                    k: v
                    for k, v in after_snapshot.items()
                    if k not in before_snapshot or before_snapshot[k] != v
                }
                loaded_variables = diff

            # Persist variable file metadata for observability/idempotence
            variable_files = self._record_imported_variable_files(namespace)

            # Filter out non-serializable RF internal objects before returning
            # This prevents FastMCP serialization errors (e.g., ScalarVariableResolver)
            serializable_variables = self._filter_serializable_variables(loaded_variables)

            return {
                "success": True,
                "session_id": session_id,
                "resource": resource_path,
                "variables_loaded": list(loaded_variables.keys()),
                "variable_files": variable_files,
                "variables_map": serializable_variables,
            }
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

    def import_variables_for_session(
        self,
        session_id: str,
        variable_file_path: str,
        args: List[str] = None
    ) -> Dict[str, Any]:
        """
        Import variables from a file into the session using RF's native APIs.

        Supports .py, .yaml/.yml, and .json variable files with Robot Framework's
        native behavior including:
        - Python static variables and get_variables() functions
        - YAML/JSON structured data
        - Arguments for Python variable files
        - RF's native path resolution and error handling

        Args:
            session_id: Target session for variable import
            variable_file_path: Path to variable file (.py, .yaml/.yml, .json)
            args: Optional arguments for Python variable files with get_variables()

        Returns:
            Dict with success status, loaded variables, and metadata
        """
        if args is None:
            args = []

        try:
            if session_id not in self._session_contexts:
                create_result = self.create_context_for_session(session_id)
                if not create_result.get("success"):
                    return create_result

            namespace = self._session_contexts[session_id]["namespace"]
            variables_obj = self._session_contexts[session_id]["variables"]

            # Normalize path for cross-platform compatibility (especially Windows)
            # Convert backslashes to forward slashes to avoid escape sequence issues
            from pathlib import Path
            normalized_path = variable_file_path
            try:
                p = Path(variable_file_path)
                if not p.is_absolute():
                    p = (Path.cwd() / p).resolve()
                # Use forward slashes for RF compatibility on all platforms
                normalized_path = p.as_posix()
            except Exception:
                # If path normalization fails, use original path
                pass
            
            # Capture variables before import to detect what was added
            before_vars = self._snapshot_variables_with_decoration(variables_obj)

            # Use Robot Framework's native variable import through namespace
            # This handles all file types (.py, .yaml, .json) and RF's path resolution
            # Use overwrite=True to ensure variables from file overwrite existing ones
            try:
                namespace.import_variables(normalized_path, args, overwrite=True)
            except Exception as import_error:
                # Provide descriptive error message as requested
                error_msg = f"Failed to import variable file '{variable_file_path}'"
                if args:
                    error_msg += f" with arguments {args}"
                error_msg += f": {str(import_error)}"

                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "session_id": session_id,
                    "variable_file": variable_file_path,
                    "args": args
                }
            
            # Capture variables after import to get diff
            after_vars = self._snapshot_variables_with_decoration(variables_obj)
            loaded_vars = {
                k: v for k, v in after_vars.items()
                if k not in before_vars or before_vars[k] != v
            }
            
            # Record variable file metadata for tracking (similar to resource import)
            variable_files = self._record_imported_variable_files(namespace)
            
            # Update session metadata to track imported variable files
            ctx_info = self._session_contexts[session_id]
            session_var_files = ctx_info.setdefault("imported_variable_files", [])

            # Check if this file was already imported in THIS session (not globally)
            existing_key = (variable_file_path, tuple(args))
            existing_keys = {
                (item.get("path"), tuple(item.get("args", [])))
                for item in session_var_files
            }
            is_first_import_for_session = existing_key not in existing_keys

            # If this is the first import of this file in this session, but loaded_vars is empty,
            # it means the file was already imported in a different session (global RF context).
            # In this case, we should still return the variables that are now available.
            if is_first_import_for_session and len(loaded_vars) == 0:
                logger.info(f"File '{normalized_path}' was imported in a different session, "
                           f"including all relevant variables for session {session_id}")
                # Get all non-builtin variables from after_vars
                # Since this is cross-session import, the file's variables are already in the shared
                # context, so we just need to return them for this session's first import
                builtin_prefixes = ('${/', '${\\', '${:}', '${TEMPDIR}', '${EXECDIR}',
                                   '${CURDIR}', '${SPACE}', '${EMPTY}', '${TRUE}',
                                   '${FALSE}', '${NULL}', '${NONE}', '${0}', '${1}', '${-1}',
                                   '${OUTPUTDIR}', '${LOGFILE}', '${OUTPUT}', '${OUTPUT_FILE}',
                                   '${LOG_FILE}', '${REPORT_FILE}', '${DEBUG_FILE}',
                                   '${LOG_LEVEL}', '${PREV_TEST}', '${SUITE_NAME}')
                # For cross-session imports, include all non-builtin variables
                # Don't filter by before_vars since they're shared across sessions
                loaded_vars = {
                    k: v for k, v in after_vars.items()
                    if not any(k.upper().startswith(prefix.upper()) for prefix in builtin_prefixes)
                }

            # Extract variable names without ${} decoration for the variables_loaded list
            variable_names_loaded = []
            for var_name in loaded_vars.keys():
                if var_name.startswith('${') and var_name.endswith('}'):
                    # Strip ${} decoration for the list
                    variable_names_loaded.append(var_name[2:-1])
                elif var_name.startswith('@{') and var_name.endswith('}'):
                    # List variable
                    variable_names_loaded.append(var_name)
                elif var_name.startswith('&{') and var_name.endswith('}'):
                    # Dict variable
                    variable_names_loaded.append(var_name)
                else:
                    variable_names_loaded.append(var_name)

            # Add this import to session tracking if not already present
            var_file_record = {
                "path": variable_file_path,
                "args": args,
                "variables_loaded": variable_names_loaded
            }

            if is_first_import_for_session:
                session_var_files.append(var_file_record)

            logger.info(f"Successfully imported variable file '{normalized_path}' "
                       f"for session {session_id}, loaded {len(loaded_vars)} variables")

            # Filter out non-serializable RF internal objects before returning
            # This prevents FastMCP serialization errors (e.g., ScalarVariableResolver)
            serializable_variables = self._filter_serializable_variables(loaded_vars)

            return {
                "success": True,
                "session_id": session_id,
                "variable_file": variable_file_path,
                "args": args,
                "variables_loaded": variable_names_loaded,
                "variables_map": serializable_variables,
                "variable_files": variable_files,
                "total_variables_in_session": len(after_vars)
            }
            
        except Exception as e:
            error_msg = f"Failed to import variable file '{variable_file_path}': {str(e)}"
            logger.error(error_msg)
            return {
                "success": False, 
                "error": error_msg,
                "session_id": session_id,
                "variable_file": variable_file_path,
                "args": args or []
            }

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

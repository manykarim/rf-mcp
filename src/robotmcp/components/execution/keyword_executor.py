"""Keyword execution service."""

import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from robotmcp.components.execution.rf_native_context_manager import (
    get_rf_native_context_manager,
)
from robotmcp.components.variables.variable_resolver import VariableResolver
from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery
from robotmcp.models.config_models import ExecutionConfig
from robotmcp.models.execution_models import ExecutionStep
from robotmcp.models.session_models import ExecutionSession
from robotmcp.utils.argument_processor import ArgumentProcessor
from robotmcp.utils.response_serializer import MCPResponseSerializer
from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter

logger = logging.getLogger(__name__)

# Import Robot Framework components
try:
    from robot.libraries.BuiltIn import BuiltIn

    ROBOT_AVAILABLE = True
except ImportError:
    BuiltIn = None
    ROBOT_AVAILABLE = False


class KeywordExecutor:
    """Handles keyword execution with proper library routing and error handling."""

    def __init__(
        self, config: Optional[ExecutionConfig] = None, override_registry=None
    ):
        self.config = config or ExecutionConfig()
        self.keyword_discovery = get_keyword_discovery()
        self.argument_processor = ArgumentProcessor()
        self.rf_converter = RobotFrameworkNativeConverter()
        self.override_registry = override_registry
        self.variable_resolver = VariableResolver()
        self.response_serializer = MCPResponseSerializer()
        # Legacy RobotContextManager is deprecated; use RF native context only
        self.rf_native_context = get_rf_native_context_manager()
        # Feature flag: route RequestsLibrary session operations via RF runner
        # Default ON; set ROBOTMCP_RF_RUNNER_REQUESTS=0 to disable
        self.rf_runner_requests = os.getenv("ROBOTMCP_RF_RUNNER_REQUESTS", "1") in (
            "1",
            "true",
            "True",
        )

    async def execute_keyword(
        self,
        session: ExecutionSession,
        keyword: str,
        arguments: List[str],
        browser_library_manager: Any,  # BrowserLibraryManager
        detail_level: str = "minimal",
        library_prefix: str = None,
        assign_to: Union[str, List[str]] = None,
        use_context: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a single Robot Framework keyword step with optional library prefix.

        Args:
            session: ExecutionSession to run in
            keyword: Robot Framework keyword name (supports Library.Keyword syntax)
            arguments: List of arguments for the keyword
            browser_library_manager: BrowserLibraryManager instance
            detail_level: Level of detail in response ('minimal', 'standard', 'full')
            library_prefix: Optional explicit library name to override session search order
            assign_to: Optional variable assignment
            use_context: If True, execute within full RF context

        Returns:
            Execution result with status, output, and state
        """

        try:
            # PHASE 1.2: Pre-execution Library Registration
            # Ensure required library is registered before keyword execution
            self._ensure_library_registration(keyword, session)

            # Create execution step
            step = ExecutionStep(
                step_id=str(uuid.uuid4()),
                keyword=keyword,
                arguments=arguments,
                start_time=datetime.now(),
            )

            # Update session activity
            session.update_activity()

            # Mark step as running
            step.status = "running"

            # Check if we should use context mode
            # Enable context mode for keywords that require RF execution context
            context_required_keywords = [
                "evaluate",
                "set test variable",
                "set suite variable",
                "set global variable",
                "create dictionary",
                "get variable value",
                "variable should exist",
                "call method",
                "run keyword if",
                "run keyword unless",
                "run keywords",
                # NOTE: Input Password removed - works fine in normal execution with name normalization
            ]

            # RequestsLibrary: route session-scoped operations through RF native context
            requests_library_context_keywords = [
                "create session",
                "delete session",
                "get on session",
                "post on session",
                "put on session",
                "delete on session",
                "patch on session",
                "head on session",
                "options on session",
            ]

            # Browser Library keywords should NOT use RF native context due to import issues
            # They work perfectly in regular execution mode
            browser_library_keywords = [
                "open browser",
                "close browser",
                "new browser",
                "new context",
                "new page",
                "go to",
                "click",
                "fill text",
                "take screenshot",
                "get text",
                "wait for elements state",
                "get title",
                "get url",
                "input text",
                "click element",
                "wait until element is visible",
            ]

            # KEYWORD NAME NORMALIZATION AND OVERRIDES - General solution for keyword name variations
            # NOTE: Input Password override is now handled in _execute_selenium_keyword method
            # to ensure proper execution while preserving original keyword for step recording
            keyword_name_mappings = {
                # Add other common mappings as needed (Input Password removed - handled in _execute_selenium_keyword)
                # "click element": "click_element",  # Usually handled by dynamic resolution
            }

            # Apply normalization if mapping exists (Input Password override removed from here)
            original_keyword = keyword
            if keyword in keyword_name_mappings:
                logger.info(
                    f"Keyword name normalized: '{original_keyword}' -> '{keyword_name_mappings[keyword]}'"
                )
                keyword = keyword_name_mappings[keyword]

            keyword_requires_context = keyword.lower() in context_required_keywords
            is_requests_keyword = keyword.lower() in requests_library_context_keywords
            is_browser_keyword = keyword.lower() in browser_library_keywords

            # Feature-flagged path for RequestsLibrary session ops
            if is_requests_keyword:
                if self.rf_runner_requests:
                    use_context = True
                    logger.info(
                        f"Feature flag enabled: routing RequestsLibrary keyword '{keyword}' to RF native runner"
                    )
                else:
                    use_context = False
                    logger.info(
                        f"Routing RequestsLibrary keyword '{keyword}' via non-context handler (temporary exception)"
                    )

            if use_context or session.is_context_mode() or keyword_requires_context:
                # Use RF native context mode for keywords that require it
                logger.info(
                    f"Executing keyword in RF native context mode: {keyword} with args: {arguments}"
                )
                result = await self._execute_keyword_with_context(
                    session, keyword, arguments, assign_to
                )
                resolved_arguments = (
                    arguments  # For logging - RF handles variable resolution
                )
            else:
                # Resolve variables in arguments before execution for non-context mode
                try:
                    # Merge session variables with RF context variables (if available)
                    merged_vars = dict(session.variables)
                    try:
                        from robotmcp.components.execution.rf_native_context_manager import (
                            get_rf_native_context_manager,
                        )

                        rf_mgr = get_rf_native_context_manager()
                        ctx_info = rf_mgr.get_session_context_info(session.session_id)
                        if ctx_info.get("context_exists"):
                            ctx = rf_mgr._session_contexts.get(session.session_id)
                            rf_vars_obj = ctx.get("variables") if ctx else None
                            if rf_vars_obj is not None:
                                rf_data = None
                                if hasattr(rf_vars_obj, "store"):
                                    rf_data = rf_vars_obj.store.data
                                elif hasattr(rf_vars_obj, "current") and hasattr(
                                    rf_vars_obj.current, "store"
                                ):
                                    rf_data = rf_vars_obj.current.store.data
                                if rf_data:
                                    for k, v in rf_data.items():
                                        key = k if isinstance(k, str) else str(k)
                                        if not key.startswith("${"):
                                            key = f"${{{key}}}"
                                        merged_vars[key] = v
                    except Exception:
                        # If RF context not available, continue with session vars only
                        pass

                    resolved_arguments = self.variable_resolver.resolve_arguments(
                        arguments, merged_vars
                    )
                    logger.debug(
                        f"Variable resolution: {arguments} → {resolved_arguments}"
                    )

                    # General solution: preserve object values for named params by converting
                    # ObjectPreservingArgument to (name, value) tuples. Orchestrator preserves tuples.
                    try:
                        from robotmcp.components.variables.variable_resolver import (
                            ObjectPreservingArgument,
                        )

                        converted_args = []
                        for arg in resolved_arguments:
                            if isinstance(arg, ObjectPreservingArgument):
                                converted_args.append((arg.param_name, arg.value))
                            else:
                                converted_args.append(arg)
                        resolved_arguments = converted_args
                    except Exception as _:
                        # If conversion helpers are unavailable, keep resolved arguments as-is
                        pass

                except Exception as var_error:
                    logger.error(
                        f"Variable resolution failed for {arguments}: {var_error}"
                    )
                    import traceback

                    logger.error(
                        f"Variable resolution traceback: {traceback.format_exc()}"
                    )
                    # Return error result for variable resolution failure
                    step.mark_failure(f"Variable resolution failed: {str(var_error)}")
                    return {
                        "success": False,
                        "error": f"Variable resolution failed: {str(var_error)}",
                        "keyword": keyword,
                        "arguments": arguments,
                        "step_id": step.step_id,
                    }

                # Log keyword execution with resolved arguments
                # Note: Input Password secure logging is now handled in _execute_selenium_keyword
                logger.info(
                    f"Executing keyword: {keyword} with resolved args: {resolved_arguments}"
                )

                # Temporary fix: pre-process RequestsLibrary Create Session named args like headers={...}
                try:
                    if keyword.lower() == "create session":
                        import ast
                        fixed_args: List[Any] = []
                        for arg in resolved_arguments:
                            if isinstance(arg, str) and "=" in arg and arg.count("=") == 1:
                                name, val = arg.split("=", 1)
                                name = name.strip().lower()
                                if name in {"headers", "cookies", "proxies", "retry_status_list", "retry_method_list"}:
                                    try:
                                        parsed = ast.literal_eval(val.strip())
                                        fixed_args.append((name, parsed))
                                        continue
                                    except Exception:
                                        pass
                            fixed_args.append(arg)
                        resolved_arguments = fixed_args
                except Exception:
                    pass

                # Execute the keyword with library prefix support using resolved arguments
                result = await self._execute_keyword_internal(
                    session,
                    step,
                    browser_library_manager,
                    library_prefix,
                    resolved_arguments,
                )

            # Update step status
            step.end_time = datetime.now()
            step.result = result.get("output")

            if result["success"]:
                step.mark_success(result.get("output"))
                # Only append successful steps to the session for suite generation
                session.add_step(step)
                logger.debug(f"Added successful step to session: {keyword}")
            else:
                step.mark_failure(result.get("error"))
                logger.debug(
                    f"Failed step not added to session: {keyword} - {result.get('error')}"
                )

            # Update session variables if any were set
            if "variables" in result:
                session.variables.update(result["variables"])

            # Validate assignment compatibility
            if assign_to:
                self._validate_assignment_compatibility(keyword, assign_to)

            # Process variable assignment if assign_to is specified
            if assign_to and result.get("success"):
                assignment_vars = self._process_variable_assignment(
                    assign_to, result.get("result"), keyword, result.get("output")
                )
                if assignment_vars:
                    # DUAL STORAGE IMPLEMENTATION:
                    # 1. Store ORIGINAL objects in session variables for RF execution context
                    session.variables.update(assignment_vars)

                    # NEW: Store assignment info in ExecutionStep for test suite generation
                    step.assigned_variables = list(assignment_vars.keys())
                    step.assignment_type = (
                        "multiple" if isinstance(assign_to, list) else "single"
                    )

                    # DEBUG: Verify what we actually stored in session variables
                    for var_name, var_value in assignment_vars.items():
                        logger.info(
                            f"STORED IN SESSION: {var_name} = {type(var_value).__name__}"
                        )
                        logger.debug(
                            f"Session storage detail: {var_name} -> {str(var_value)[:100]}"
                        )
                        # Verify what's actually in session.variables after update
                        actual_stored = session.variables.get(var_name)
                        logger.info(
                            f"SESSION VERIFICATION: {var_name} stored as {type(actual_stored).__name__}"
                        )

                    # 2. Store raw objects for RF Variables system (needed for ${response.json()})
                    result["assigned_variables_raw"] = assignment_vars

                    # 3. Add serialized assignment info to result for MCP response compatibility
                    # This prevents serialization errors with complex objects
                    serialized_assigned_vars = (
                        self.response_serializer.serialize_assigned_variables(
                            assignment_vars
                        )
                    )
                    result["assigned_variables"] = serialized_assigned_vars

                    # Log assignment for debugging
                    for var_name, var_value in assignment_vars.items():
                        logger.info(
                            f"Assigned variable {var_name} = {type(var_value).__name__} (serialized for response)"
                        )
                        logger.debug(
                            f"Assignment detail: {var_name} -> {str(var_value)[:200]}"
                        )

            # Build response based on detail level
            response = await self._build_response_by_detail_level(
                detail_level,
                result,
                step,
                keyword,
                arguments,
                session,
                resolved_arguments,
            )
            return response

        except Exception as e:
            logger.error(f"Error executing step {keyword}: {e}")

            # Create a failed step for error reporting
            step = ExecutionStep(
                step_id=str(uuid.uuid4()),
                keyword=keyword,
                arguments=arguments,
                start_time=datetime.now(),
                end_time=datetime.now(),
            )
            step.mark_failure(str(e))

            return {
                "success": False,
                "error": str(e),
                "step_id": step.step_id,
                "keyword": keyword,
                "arguments": arguments,
                "status": "fail",
                "execution_time": step.execution_time,
                "session_variables": dict(session.variables),
            }

    def _process_variable_assignment(
        self,
        assign_to: Union[str, List[str]],
        result_value: Any,
        keyword: str,
        output: str,
    ) -> Dict[str, Any]:
        """Process variable assignment from keyword execution result.

        Args:
            assign_to: Variable name(s) to assign to
            result_value: The actual return value from the keyword
            keyword: The keyword name (for logging)
            output: The output string representation

        Returns:
            Dictionary of variables to assign to session
        """
        if not assign_to:
            return {}

        # DEBUG: Log what we receive for tracing serialization issue
        logger.debug(
            f"VARIABLE_ASSIGNMENT_DEBUG: {keyword} result_value type: {type(result_value)}"
        )
        logger.debug(
            f"VARIABLE_ASSIGNMENT_DEBUG: {keyword} result_value: {str(result_value)[:200]}"
        )

        # Check if result_value is already serialized (RequestsLibrary Response issue)
        if (
            isinstance(result_value, dict)
            and result_value.get("_type") == "requests_response"
        ):
            logger.warning(
                f"SERIALIZATION_WARNING: {keyword} result_value is already serialized Response object!"
            )

        # If result_value is None but output exists, try to use output
        # This handles cases where the result is in output but not result field
        value_to_assign = result_value
        if value_to_assign is None and output:
            try:
                # Try to parse output as the actual value
                import ast

                # Handle simple cases like numbers, strings, lists
                if output.isdigit():
                    value_to_assign = int(output)
                elif output.replace(".", "").isdigit():
                    value_to_assign = float(output)
                elif output.startswith("[") and output.endswith("]"):
                    value_to_assign = ast.literal_eval(output)
                else:
                    value_to_assign = output
            except:
                value_to_assign = output

        variables = {}

        try:
            if isinstance(assign_to, str):
                # Single assignment
                var_name = self._normalize_variable_name(assign_to)
                variables[var_name] = value_to_assign
                logger.info(f"Assigned {var_name} = {value_to_assign}")

            elif isinstance(assign_to, list):
                # Multi-assignment
                if isinstance(value_to_assign, (list, tuple)):
                    for i, var_name in enumerate(assign_to):
                        normalized_name = self._normalize_variable_name(var_name)
                        if i < len(value_to_assign):
                            variables[normalized_name] = value_to_assign[i]
                        else:
                            variables[normalized_name] = None
                        logger.info(
                            f"Assigned {normalized_name} = {variables[normalized_name]}"
                        )
                else:
                    # Single value assigned to multiple variables (first gets value, rest get None)
                    for i, var_name in enumerate(assign_to):
                        normalized_name = self._normalize_variable_name(var_name)
                        variables[normalized_name] = value_to_assign if i == 0 else None
                        logger.info(
                            f"Assigned {normalized_name} = {variables[normalized_name]}"
                        )

        except Exception as e:
            logger.warning(
                f"Error processing variable assignment for keyword '{keyword}': {e}"
            )
            # Fallback: assign the raw value to first variable name
            if isinstance(assign_to, str):
                var_name = self._normalize_variable_name(assign_to)
                variables[var_name] = value_to_assign
            elif isinstance(assign_to, list) and assign_to:
                var_name = self._normalize_variable_name(assign_to[0])
                variables[var_name] = value_to_assign

        return variables

    def _ensure_library_registration(self, keyword: str, session: Any) -> None:
        """
        Ensure required library is registered in RF context before keyword execution.

        This is Phase 1.2 of the RequestsLibrary fix: Pre-execution Library Registration.
        We determine which library is needed for a keyword and ensure it's registered
        in the Robot Framework execution context.
        """
        try:
            # Determine library from keyword
            library_name = self._get_library_for_keyword(keyword)

            if library_name and library_name == "RequestsLibrary":
                # Get the library manager from keyword discovery
                library_manager = self.keyword_discovery.library_manager

                # Ensure RequestsLibrary is loaded in our manager
                if library_name not in library_manager.libraries:
                    logger.info(
                        f"Loading {library_name} on demand for keyword: {keyword}"
                    )
                    library_manager.load_library_on_demand(
                        library_name, self.keyword_discovery
                    )

                # Ensure RequestsLibrary is properly registered in RF context
                registration_success = library_manager.ensure_library_in_rf_context(
                    library_name
                )

                if registration_success:
                    logger.debug(
                        f"Successfully ensured {library_name} registration for keyword: {keyword}"
                    )

                    # PHASE 2: Session State Synchronization
                    # After successful registration, synchronize session state
                    session_manager = getattr(session, "_session_manager", None)
                    if session_manager:
                        sync_success = (
                            session_manager.synchronize_requests_library_state(session)
                        )
                        if sync_success:
                            logger.debug(
                                f"RequestsLibrary session state synchronized for keyword: {keyword}"
                            )
                        else:
                            logger.debug(
                                f"RequestsLibrary session synchronization skipped for keyword: {keyword}"
                            )

                else:
                    logger.warning(
                        f"Failed to register {library_name} in RF context for keyword: {keyword}"
                    )

        except Exception as e:
            logger.error(f"Library registration check failed for {keyword}: {e}")
            # Don't fail execution for this - let the keyword execution handle library issues

    def _get_library_for_keyword(self, keyword: str) -> Optional[str]:
        """Determine which library provides a given keyword."""

        # Handle explicit library prefixes (e.g., "RequestsLibrary.POST")
        if "." in keyword:
            parts = keyword.split(".")
            if len(parts) == 2:
                library_name, _ = parts
                return library_name

        # Map common HTTP keywords to RequestsLibrary
        requests_keywords = {
            "POST",
            "GET",
            "PUT",
            "DELETE",
            "PATCH",
            "HEAD",
            "OPTIONS",
            "POST On Session",
            "GET On Session",
            "PUT On Session",
            "DELETE On Session",
            "Create Session",
            "Delete All Sessions",
        }

        if keyword in requests_keywords:
            return "RequestsLibrary"

        # For other keywords, we could extend this mapping
        # For now, return None for non-RequestsLibrary keywords
        return None

    def _normalize_variable_name(self, name: str) -> str:
        """Normalize variable name to Robot Framework format."""
        if not name.startswith("${") or not name.endswith("}"):
            return f"${{{name}}}"
        return name

    def _validate_assignment_compatibility(
        self, keyword: str, assign_to: Union[str, List[str]]
    ) -> None:
        """Validate if keyword is appropriate for variable assignment."""
        if not assign_to:
            return

        # Keywords that typically return useful values for assignment
        returnable_keywords = {
            # String operations
            "Get Length",
            "Get Substring",
            "Replace String",
            "Split String",
            "Convert To Uppercase",
            "Convert To Lowercase",
            "Strip String",
            # Web automation - element queries
            "Get Text",
            "Get Title",
            "Get Location",
            "Get Element Count",
            "Get Element Attribute",
            "Get Element Size",
            "Get Element Position",
            "Get Window Size",
            "Get Window Position",
            "Get Page Source",
            # Web automation - Browser Library
            "Get Url",
            "Get Title",
            "Get Text",
            "Get Attribute",
            "Get Property",
            "Get Element Count",
            "Get Page Source",
            "Evaluate JavaScript",
            # Conversions
            "Convert To Integer",
            "Convert To Number",
            "Convert To String",
            "Convert To Boolean",
            "Evaluate",
            # Collections
            "Get From List",
            "Get Slice From List",
            "Get Length",
            "Get Index",
            "Create List",
            "Create Dictionary",
            "Get Dictionary Keys",
            "Get Dictionary Values",
            # Built-in
            "Set Variable",
            "Get Variable Value",
            "Get Time",
            "Get Environment Variable",
            # System operations
            "Run Process",
            "Run",
            "Get Environment Variable",
        }

        keyword_lower = keyword.lower()
        found_match = False

        for returnable in returnable_keywords:
            if (
                returnable.lower() in keyword_lower
                or keyword_lower in returnable.lower()
            ):
                found_match = True
                break

        if not found_match:
            logger.warning(
                f"Keyword '{keyword}' may not return a useful value for assignment. "
                f"Typical returnable keywords include: Get Text, Get Length, Get Title, etc."
            )

        # Validate assignment count for known multi-return keywords
        multi_return_keywords = {
            "Split String": "Can return multiple parts when max_split is used",
            "Get Time": "Can return multiple time components",
            "Run Process": "Returns stdout and stderr",
            "Get Slice From List": "Can return multiple items",
        }

        for multi_keyword, description in multi_return_keywords.items():
            if multi_keyword.lower() in keyword_lower:
                if isinstance(assign_to, str):
                    logger.info(
                        f"'{keyword}' {description}. Consider using list assignment: ['part1', 'part2']"
                    )
                break

    async def _execute_keyword_internal(
        self,
        session: ExecutionSession,
        step: ExecutionStep,
        browser_library_manager: Any,
        library_prefix: str = None,
        resolved_arguments: List[str] = None,
    ) -> Dict[str, Any]:
        """Execute a specific keyword with error handling and library prefix support."""
        try:
            keyword_name = step.keyword
            # Use resolved arguments if provided, otherwise fall back to step arguments
            args = (
                resolved_arguments if resolved_arguments is not None else step.arguments
            )

            # Check for keyword overrides first (before library detection)
            if self.override_registry:
                # Find keyword in discovery to determine library (session-aware)
                from robotmcp.core.dynamic_keyword_orchestrator import (
                    get_keyword_discovery,
                )

                orchestrator = get_keyword_discovery()
                # PHASE 1 INTEGRATION: Session-aware keyword discovery with session libraries
                session_libraries = self._get_session_libraries(session)
                web_automation_lib = session.get_web_automation_library()

                if session_libraries:
                    # Use session libraries for keyword discovery (highest priority)
                    keyword_info = orchestrator.find_keyword(
                        keyword_name, session_libraries=session_libraries
                    )
                    logger.debug(
                        f"Session-aware keyword discovery: '{keyword_name}' in session libraries {session_libraries} → {keyword_info.library if keyword_info else None}"
                    )
                elif web_automation_lib:
                    # Fallback to active library approach
                    active_library = (
                        web_automation_lib
                        if web_automation_lib in ["Browser", "SeleniumLibrary"]
                        else None
                    )
                    keyword_info = orchestrator.find_keyword(
                        keyword_name, active_library=active_library
                    )
                    logger.debug(
                        f"Active library keyword discovery: '{keyword_name}' with active_library='{active_library}' → {keyword_info.library if keyword_info else None}"
                    )
                else:
                    # Final fallback to global discovery
                    keyword_info = orchestrator.find_keyword(keyword_name)
                    logger.debug(
                        f"Global keyword discovery: '{keyword_name}' → {keyword_info.library if keyword_info else None}"
                    )

                # BROWSER LIBRARY FALLBACK: Force regular execution due to RF context import issues
                if keyword_info and keyword_info.library == "Browser":
                    logger.info(
                        f"Browser Library keyword detected: {keyword_name} - forcing regular execution mode"
                    )
                    # This ensures Browser keywords use the regular execution path which works
                    # instead of RF context mode which has library import issues

                if keyword_info:
                    override_handler = self.override_registry.get_override(
                        keyword_name, keyword_info.library
                    )
                    if override_handler:
                        logger.info(
                            f"OVERRIDE: Using override handler {type(override_handler).__name__} for {keyword_name} from {keyword_info.library}"
                        )
                        override_result = await override_handler.execute(
                            session, keyword_name, args, keyword_info
                        )
                        if override_result is not None:
                            # Ensure proper library is imported in session
                            session.import_library(keyword_info.library, force=True)
                            logger.info(
                                f"OVERRIDE: Successfully executed {keyword_name} with {keyword_info.library}, imported to session - RETURNING EARLY"
                            )
                            return {
                                "success": override_result.success,
                                "output": override_result.output
                                or f"Executed {keyword_name}",
                                "error": override_result.error,
                                "variables": {},
                                "state_updates": override_result.state_updates or {},
                            }
                        else:
                            logger.warning(
                                f"OVERRIDE: Override handler returned None for {keyword_name}"
                            )
                else:
                    logger.debug(
                        f"OVERRIDE: No keyword info found for {keyword_name}, trying session-based loading"
                    )
                    # Try to ensure libraries are loaded for this session
                    await orchestrator._ensure_session_libraries(
                        session.session_id, keyword_name
                    )
                    # Try finding the keyword again after loading (session-aware)
                    session_libraries = self._get_session_libraries(session)
                    web_automation_lib = session.get_web_automation_library()

                    if session_libraries:
                        keyword_info = orchestrator.find_keyword(
                            keyword_name, session_libraries=session_libraries
                        )
                        logger.debug(
                            f"Post-loading session-aware discovery: '{keyword_name}' in session libraries {session_libraries} → {keyword_info.library if keyword_info else None}"
                        )
                    elif web_automation_lib:
                        active_library = (
                            web_automation_lib
                            if web_automation_lib in ["Browser", "SeleniumLibrary"]
                            else None
                        )
                        keyword_info = orchestrator.find_keyword(
                            keyword_name, active_library=active_library
                        )
                        logger.debug(
                            f"Post-loading active library discovery: '{keyword_name}' with active_library='{active_library}' → {keyword_info.library if keyword_info else None}"
                        )
                    else:
                        keyword_info = orchestrator.find_keyword(keyword_name)
                        logger.debug(
                            f"Post-loading global discovery: '{keyword_name}' → {keyword_info.library if keyword_info else None}"
                        )

                    # BROWSER LIBRARY FALLBACK: Force regular execution due to RF context import issues (post-loading)
                    if keyword_info and keyword_info.library == "Browser":
                        logger.info(
                            f"Browser Library keyword detected (post-loading): {keyword_name} - forcing regular execution mode"
                        )
                    if keyword_info:
                        override_handler = self.override_registry.get_override(
                            keyword_name, keyword_info.library
                        )
                        if override_handler:
                            logger.info(
                                f"OVERRIDE: Using override handler {type(override_handler).__name__} for {keyword_name} from {keyword_info.library} (after loading)"
                            )
                            override_result = await override_handler.execute(
                                session, keyword_name, args, keyword_info
                            )
                            if override_result is not None:
                                session.import_library(keyword_info.library, force=True)
                                logger.info(
                                    f"OVERRIDE: Successfully executed {keyword_name} with {keyword_info.library} (after loading) - RETURNING EARLY"
                                )
                                return {
                                    "success": override_result.success,
                                    "output": override_result.output
                                    or f"Executed {keyword_name}",
                                    "error": override_result.error,
                                    "variables": {},
                                    "state_updates": override_result.state_updates
                                    or {},
                                }

            # Determine library to use based on session configuration
            web_automation_lib = session.get_web_automation_library()
            current_active = session.get_active_library()
            session_type = session.get_session_type()

            # CRITICAL FIX: Respect session type boundaries
            if session_type.value in [
                "xml_processing",
                "api_testing",
                "data_processing",
                "system_testing",
            ]:
                # Typed sessions should not use web automation auto-detection
                logger.debug(
                    f"Session type '{session_type.value}' - skipping web automation auto-detection"
                )

            elif web_automation_lib:
                # Session has a specific web automation library imported - use it
                if web_automation_lib == "Browser" and (
                    not current_active or current_active == "auto"
                ):
                    browser_library_manager.set_active_library(session, "browser")
                    logger.debug("Using session's web automation library: Browser")
                elif web_automation_lib == "SeleniumLibrary" and (
                    not current_active or current_active == "auto"
                ):
                    browser_library_manager.set_active_library(session, "selenium")
                    logger.debug(
                        "Using session's web automation library: SeleniumLibrary"
                    )

            elif session_type.value in ["web_automation", "unknown"]:
                # Only auto-detect for web/unknown sessions (REMOVED mobile_testing)
                if not current_active or current_active == "auto":
                    detected_library = (
                        browser_library_manager.detect_library_from_keyword(
                            keyword_name, args
                        )
                    )
                    if detected_library in ["browser", "selenium"]:
                        browser_library_manager.set_active_library(
                            session, detected_library
                        )
                        logger.debug(
                            f"Auto-detected library for '{keyword_name}': {detected_library}"
                        )

            elif session_type.value == "mobile_testing":
                # CRITICAL FIX: Mobile sessions should NEVER auto-detect web libraries
                # They should only use their configured session libraries
                logger.debug(
                    "Mobile session - using only session libraries, no web auto-detection"
                )
                # Let execution flow through session-aware keyword resolution only
            else:
                logger.debug(
                    f"Session type '{session_type.value}' - no web automation library handling needed"
                )

            # Handle special built-in keywords first
            if keyword_name.lower() in ["set variable", "log", "should be equal"]:
                return await self._execute_builtin_keyword(session, keyword_name, args)

            # Route ALL keywords through AppiumLibrary for mobile sessions
            if session.is_mobile_session():
                logger.debug(
                    f"Mobile session detected; routing '{keyword_name}' to AppiumLibrary execution path"
                )
                return await self._execute_mobile_keyword(session, keyword_name, args)

            # If library prefix is specified, use direct execution
            if library_prefix:
                return await self._execute_with_library_prefix(
                    session, keyword_name, args, library_prefix
                )

            # Get active browser library and execute
            library, library_type = browser_library_manager.get_active_browser_library(
                session
            )

            if library_type == "browser":
                return await self._execute_browser_keyword(
                    session, keyword_name, args, library
                )
            elif library_type == "selenium":
                logger.debug(f"INPUT PASSWORD DEBUG: Routing '{keyword_name}' to SeleniumLibrary execution path")
                return await self._execute_selenium_keyword(
                    session, keyword_name, args, library
                )
            else:
                # Try built-in execution as fallback
                return await self._execute_builtin_keyword(session, keyword_name, args)

        except Exception as e:
            logger.error(f"Error in keyword execution: {e}")
            return {
                "success": False,
                "error": f"Execution failed: {str(e)}",
                "output": "",
                "variables": {},
                "state_updates": {},
            }

    async def _execute_keyword_with_context(
        self,
        session: ExecutionSession,
        keyword: str,
        arguments: List[Any],
        assign_to: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Execute keyword within full Robot Framework native context.

        This uses RF's native execution context to enable proper execution of
        keywords like Evaluate, Set Test Variable, etc. that require RF context.

        Args:
            session: ExecutionSession to run in
            keyword: Robot Framework keyword name
            arguments: List of arguments for the keyword
            assign_to: Optional variable assignment

        Returns:
            Execution result with status, output, and state
        """
        try:
            session_id = session.session_id

            logger.info(
                f"RF NATIVE CONTEXT: Executing {keyword} with native RF context for session {session_id}"
            )

            # Create or get RF native context for session
            context_info = self.rf_native_context.get_session_context_info(session_id)
            if not context_info["context_exists"]:
                # Create RF native context with session's library search order
                # Use search_order if available, otherwise try loaded_libraries
                if hasattr(session, "search_order") and session.search_order:
                    libraries = list(session.search_order)
                elif hasattr(session, "loaded_libraries") and session.loaded_libraries:
                    libraries = list(session.loaded_libraries)
                else:
                    libraries = []

                logger.info(f"Creating RF native context with libraries: {libraries}")
                context_result = self.rf_native_context.create_context_for_session(
                    session_id, libraries
                )
                if not context_result.get("success"):
                    logger.error(
                        f"RF native context creation failed: {context_result.get('error')}"
                    )
                    return {
                        "success": False,
                        "error": f"Failed to create RF native context: {context_result.get('error')}",
                        "keyword": keyword,
                        "arguments": arguments,
                    }
                logger.info(
                    f"Created RF native context for session {session_id} with libraries: {libraries}"
                )

            # Execute keyword using RF native context with session variables
            result = self.rf_native_context.execute_keyword_with_context(
                session_id=session_id,
                keyword_name=keyword,
                arguments=arguments,
                assign_to=assign_to,
                session_variables=dict(
                    session.variables
                ),  # Pass original objects to RF Variables
            )

            # Update session variables from RF native context
            if result.get("success") and "variables" in result:
                session.variables.update(result["variables"])
                logger.debug(
                    f"Updated session variables from RF native context: {len(result['variables'])} variables"
                )

            logger.info(
                f"RF NATIVE CONTEXT: {keyword} executed with result: {result.get('success')}"
            )
            return result

        except Exception as e:
            logger.error(f"RF native context execution failed: {e}")
            import traceback

            logger.error(f"RF native context traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"RF native context execution failed: {str(e)}",
                "keyword": keyword,
                "arguments": arguments,
            }

    async def _execute_with_library_prefix(
        self,
        session: ExecutionSession,
        keyword: str,
        args: List[str],
        library_prefix: str,
    ) -> Dict[str, Any]:
        """Execute keyword with explicit library prefix."""
        try:
            # Use keyword discovery with library prefix
            result = await self.keyword_discovery.execute_keyword(
                keyword_name=keyword,
                args=args,
                session_variables=session.variables,
                active_library=None,  # Don't use active library when prefix is explicit
                session_id=session.session_id,
                library_prefix=library_prefix,
            )

            # Update session state if successful
            if result.get("success"):
                # Extract any state updates based on the library and keyword
                if library_prefix.lower() == "browser":
                    state_updates = self._extract_browser_state_updates(
                        keyword, args, result.get("output")
                    )
                    self._apply_state_updates(session, state_updates)

                # Add library prefix information to result
                result["library_prefix_used"] = library_prefix

            return result

        except Exception as e:
            logger.error(f"Error executing {library_prefix}.{keyword}: {e}")
            return {
                "success": False,
                "error": f"Library prefix execution failed: {str(e)}",
                "output": "",
                "variables": {},
                "state_updates": {},
                "library_prefix_used": library_prefix,
            }

    async def _execute_browser_keyword(
        self, session: ExecutionSession, keyword: str, args: List[str], library: Any
    ) -> Dict[str, Any]:
        """Execute a Browser Library keyword using the dynamic execution handler."""
        try:
            # Use the keyword discovery's execute_keyword method with Browser Library filter
            result = await self.keyword_discovery.execute_keyword(
                keyword_name=keyword,
                args=args,
                session_variables=session.variables,
                active_library="Browser",
                session_id=session.session_id,
            )

            # Update session browser state based on keyword if successful
            if result.get("success"):
                state_updates = self._extract_browser_state_updates(
                    keyword, args, result.get("output")
                )
                self._apply_state_updates(session, state_updates)
                result["state_updates"] = state_updates
            else:
                # Add Browser Library-specific error guidance for failed keywords
                result["browser_guidance"] = self._get_browser_error_guidance(
                    keyword, args, result.get("error", "")
                )

            return result

        except Exception as e:
            logger.error(f"Error executing Browser Library keyword {keyword}: {e}")
            error_msg = f"Browser keyword execution failed: {str(e)}"

            # Include locator guidance for Browser Library errors
            guidance = self._get_browser_error_guidance(keyword, args, str(e))

            return {
                "success": False,
                "error": error_msg,
                "output": "",
                "variables": {},
                "state_updates": {},
                "browser_guidance": guidance,
            }

    async def _execute_selenium_keyword(
        self, session: ExecutionSession, keyword: str, args: List[str], library: Any
    ) -> Dict[str, Any]:
        """Execute a SeleniumLibrary keyword using the dynamic execution handler."""
        try:
            # Use the keyword discovery's execute_keyword method with SeleniumLibrary filter
            result = await self.keyword_discovery.execute_keyword(
                keyword_name=keyword,
                args=args,
                session_variables=session.variables,
                active_library="SeleniumLibrary", 
                session_id=session.session_id,
            )

            # Update session browser state based on keyword if successful
            if result.get("success"):
                state_updates = self._extract_selenium_state_updates(
                    keyword, args, result.get("output")
                )
                self._apply_state_updates(session, state_updates)
                result["state_updates"] = state_updates
            else:
                # Add SeleniumLibrary-specific error guidance for failed keywords
                result["selenium_guidance"] = self._get_selenium_error_guidance(
                    keyword, args, result.get("error", "")
                )

            return result

        except Exception as e:
            logger.error(f"Error executing SeleniumLibrary keyword {keyword}: {e}")
            error_msg = f"Selenium keyword execution failed: {str(e)}"

            # Include locator guidance for SeleniumLibrary errors
            guidance = self._get_selenium_error_guidance(keyword, args, str(e))

            return {
                "success": False,
                "error": error_msg,
                "output": "",
                "variables": {},
                "state_updates": {},
                "selenium_guidance": guidance,
            }

    async def _execute_builtin_keyword(
        self, session: ExecutionSession, keyword: str, args: List[str]
    ) -> Dict[str, Any]:
        """Execute a built-in Robot Framework keyword."""
        try:
            # First, attempt dynamic execution via orchestrator for non-built-in libraries.
            # This path supports full argument parsing (incl. **kwargs) and works for AppiumLibrary, RequestsLibrary, etc.
            try:
                from robotmcp.core.dynamic_keyword_orchestrator import (
                    get_keyword_discovery,
                )

                orchestrator = get_keyword_discovery()
                dyn_result = await orchestrator.execute_keyword(
                    keyword_name=keyword,
                    args=args,
                    session_variables=session.variables,
                    active_library=None,
                    session_id=session.session_id,
                    library_prefix=None,
                )

                # If orchestrator could resolve and execute the keyword, return immediately
                if dyn_result and dyn_result.get("success"):
                    return dyn_result
            except Exception as dyn_error:
                logger.debug(
                    f"Dynamic orchestrator path failed for '{keyword}': {dyn_error}. Falling back to BuiltIn."
                )

            if not ROBOT_AVAILABLE:
                return {
                    "success": False,
                    "error": "Robot Framework not available for built-in keywords",
                    "output": "",
                    "variables": {},
                    "state_updates": {},
                }

            builtin = BuiltIn()
            keyword_lower = keyword.lower()

            # Handle common built-in keywords
            if keyword_lower == "set variable":
                if args:
                    var_value = args[0]
                    return {
                        "success": True,
                        "result": var_value,  # Store actual return value
                        "output": var_value,
                        "variables": {"${VARIABLE}": var_value},
                        "state_updates": {},
                    }

            elif keyword_lower == "log":
                message = args[0] if args else ""
                logger.info(f"Robot Log: {message}")
                return {
                    "success": True,
                    "result": None,  # Log doesn't return a value
                    "output": message,
                    "variables": {},
                    "state_updates": {},
                }

            elif keyword_lower == "should be equal":
                if len(args) >= 2:
                    if args[0] == args[1]:
                        return {
                            "success": True,
                            "result": True,  # Assertion passed
                            "output": f"'{args[0]}' == '{args[1]}'",
                            "variables": {},
                            "state_updates": {},
                        }
                    else:
                        return {
                            "success": False,
                            "result": False,  # Assertion failed
                            "error": f"'{args[0]}' != '{args[1]}'",
                            "output": "",
                            "variables": {},
                            "state_updates": {},
                        }

            # Try to execute using BuiltIn library
            try:
                # ENHANCEMENT: Use RF native type converter for proper argument processing
                # This handles RequestsLibrary and other complex libraries with named arguments
                logger.info(
                    f"BUILTIN KEYWORD EXECUTION PATH: {keyword} with args: {args}"
                )
                print(f"🔍 BUILTIN PATH: {keyword} with args: {args}", file=sys.stderr)
                print(
                    f"🔍 BUILTIN ARGS TYPES: {[type(arg).__name__ for arg in args]}",
                    file=sys.stderr,
                )
                try:
                    processed_args = self.rf_converter.parse_and_convert_arguments(
                        keyword,
                        args,
                        library_name=None,
                        session_variables=session.variables,
                    )
                    logger.info(
                        f"RF converter processed {keyword} args: {args} → {processed_args}"
                    )
                    print(f"🔍 RF CONVERTER SUCCESS: {processed_args}", file=sys.stderr)
                except Exception as converter_error:
                    logger.warning(
                        f"RF converter failed for {keyword}: {converter_error}, falling back to basic processing"
                    )
                    print(f"🔍 RF CONVERTER FAILED: {converter_error}", file=sys.stderr)
                    processed_args = args

                # DUAL HANDLING: RequestsLibrary needs object arguments, others need string arguments
                # FINAL SOLUTION: Inject objects directly before keyword execution
                final_args = self._inject_objects_for_execution(processed_args, session)

                result = builtin.run_keyword(keyword, *final_args)
                return {
                    "success": True,
                    "result": result,  # Store the actual return value
                    "output": str(result) if result is not None else "OK",
                    "variables": {},
                    "state_updates": {},
                }
            except Exception as e:
                # Phase 4: Add comprehensive diagnostics for keyword execution failures
                diagnostics = self._get_keyword_failure_diagnostics(
                    keyword, args, str(e), session
                )
                return {
                    "success": False,
                    "error": f"Built-in keyword execution failed: {str(e)}",
                    "output": "",
                    "variables": {},
                    "state_updates": {},
                    "diagnostics": diagnostics,  # Phase 4: Enhanced diagnostics
                }

        except Exception as e:
            logger.error(f"Error executing built-in keyword {keyword}: {e}")
            # Phase 4: Add diagnostics for outer exception handler too
            diagnostics = self._get_keyword_failure_diagnostics(
                keyword, args, str(e), session
            )
            return {
                "success": False,
                "error": f"Built-in keyword execution failed: {str(e)}",
                "output": "",
                "variables": {},
                "state_updates": {},
                "diagnostics": diagnostics,  # Phase 4: Enhanced diagnostics
            }

    def _get_keyword_failure_diagnostics(
        self,
        keyword: str,
        args: List[str],
        error_message: str,
        session: ExecutionSession,
    ) -> Dict[str, Any]:
        """
        Phase 4: Get comprehensive diagnostic information for keyword execution failures.

        Args:
            keyword: The keyword that failed
            args: Arguments provided to the keyword
            error_message: The error message from the failure
            session: ExecutionSession for context

        Returns:
            Dictionary with diagnostic information
        """
        # Use the orchestrator's diagnostic capabilities
        from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery

        orchestrator = get_keyword_discovery()

        # Get comprehensive diagnostics from the orchestrator
        diagnostics = orchestrator._get_diagnostic_info(
            keyword_name=keyword,
            session_id=session.session_id,
            active_library=session.get_active_library(),
        )

        # Add keyword executor specific information
        diagnostics["execution_context"] = {
            "execution_path": "builtin_keyword_executor",
            "provided_arguments": args,
            "argument_count": len(args),
            "execution_error": error_message,
            "session_type": session.get_session_type().value,
        }

        # Add Robot Framework specific diagnostics
        try:
            from robot.running.context import EXECUTION_CONTEXTS

            rf_context_available = bool(EXECUTION_CONTEXTS.current)
            diagnostics["robot_framework_context"] = {
                "execution_context_available": rf_context_available
            }
        except:
            diagnostics["robot_framework_context"] = {
                "execution_context_available": False
            }

        return diagnostics

    def _keyword_expects_object_arguments(
        self, keyword: str, arg_index: int, arg_value: Any
    ) -> bool:
        """
        Determine if a keyword expects object arguments at a specific position.

        This is critical for RequestsLibrary which expects dict/list objects for json/data parameters,
        while most other keywords expect string arguments.
        """
        keyword_lower = keyword.lower()

        # Debug output
        print(
            f"🔍 OBJECT CHECK: keyword={keyword_lower}, arg_index={arg_index}, arg_value={arg_value}, type={type(arg_value).__name__}",
            file=sys.stderr,
        )

        # RequestsLibrary keywords that accept object parameters
        requests_keywords_with_objects = {
            "post": ["json", "data"],
            "put": ["json", "data"],
            "patch": ["json", "data"],
            "post on session": ["json", "data"],
            "put on session": ["json", "data"],
            "patch on session": ["json", "data"],
        }

        if keyword_lower in requests_keywords_with_objects:
            # Check if this is a dict or list object that should be preserved
            if isinstance(arg_value, (dict, list)):
                print(
                    f"🔍 PRESERVING OBJECT: RequestsLibrary keyword {keyword} detected with {type(arg_value).__name__} argument",
                    file=sys.stderr,
                )
                logger.debug(
                    f"RequestsLibrary keyword {keyword} detected with {type(arg_value).__name__} argument - preserving as object"
                )
                return True

        print(
            f"🔍 CONVERTING TO STRING: keyword={keyword}, arg will be converted",
            file=sys.stderr,
        )
        # For other complex argument structures that might need objects
        # Add more library-specific logic here as needed

        return False

    def _process_object_preserving_arguments(self, args: List[Any]) -> List[Any]:
        """
        Handle ObjectPreservingArgument objects for Robot Framework execution.

        Robot Framework's argument resolver expects named parameters to be handled
        differently than simple string formatting. For object values, we need to
        pass them as separate arguments or use RF's native parameter handling.
        """
        from robotmcp.components.variables.variable_resolver import (
            ObjectPreservingArgument,
        )

        processed_args = []

        for arg in args:
            if isinstance(arg, ObjectPreservingArgument):
                # CORRECT APPROACH: For RF execution, we need to preserve the object
                # and pass it in a way that RF's ArgumentResolver can handle.
                # Instead of converting to string, we store the object and use a reference
                # that will be resolved during actual keyword execution.

                # Store object in temporary session storage for later injection
                processed_args.append(arg)  # Keep the ObjectPreservingArgument object
            else:
                processed_args.append(arg)

        return processed_args

    def _store_and_reference_objects(self, args: List[Any], session: Any) -> List[str]:
        """
        FINAL SOLUTION: Store ObjectPreservingArgument objects in session and replace with references.

        This stores the actual objects in the session's temporary storage and replaces
        them with placeholder references that can be injected back later.
        """
        from robotmcp.components.variables.variable_resolver import (
            ObjectPreservingArgument,
        )

        processed_args = []

        for arg in args:
            if isinstance(arg, ObjectPreservingArgument):
                # Create a unique reference ID for this object
                import uuid

                ref_id = f"__OBJ_REF_{uuid.uuid4().hex[:8]}"

                # Store the actual object in session temporary storage
                if not hasattr(session, "_temp_objects"):
                    session._temp_objects = {}
                session._temp_objects[ref_id] = arg.value

                # Replace with a reference that includes the parameter name
                processed_args.append(f"{arg.param_name}=${{{ref_id}}}")

                # Also store the reference in session variables for RF to resolve
                session.variables[ref_id] = arg.value

            else:
                processed_args.append(arg)

        return processed_args

    def _inject_objects_for_execution(self, args: List[str], session: Any) -> List[Any]:
        """
        FINAL SOLUTION: Inject actual objects directly at execution time.

        This replaces object reference placeholders with the actual objects
        right before the keyword is executed, bypassing all the complex
        variable resolution issues.
        """
        # Inject objects for RequestsLibrary and other libraries expecting object parameters
        final_args = []

        for arg in args:
            # Handle URL parameter conversion to positional format
            if (
                isinstance(arg, str)
                and arg.startswith("url=")
                and "${__OBJ_REF_" not in arg
            ):
                # Convert URL from named to positional for RequestsLibrary
                url_value = arg[4:]  # Remove 'url=' prefix
                final_args.append(url_value)
            elif isinstance(arg, str) and "${__OBJ_REF_" in arg:
                # This argument contains an object reference - extract and inject the object
                import re

                # Find object reference patterns in the argument
                ref_pattern = r"\$\{(__OBJ_REF_[^}]+)\}"
                matches = re.findall(ref_pattern, arg)

                if matches:
                    # Replace each reference with the actual object
                    processed_arg = arg
                    for ref_id in matches:
                        if (
                            hasattr(session, "_temp_objects")
                            and ref_id in session._temp_objects
                        ):
                            actual_object = session._temp_objects[ref_id]

                            # If the entire argument is just the reference, replace with the object
                            if processed_arg == f"${{{ref_id}}}":
                                final_args.append(actual_object)
                                break
                            # If it's a named parameter, inject the object as the value
                            elif (
                                "=" in processed_arg
                                and f"${{{ref_id}}}" in processed_arg
                            ):
                                param_name = processed_arg.split("=")[0]
                                # Use tuple format for RF named args with objects
                                final_args.append((param_name, actual_object))
                                break
                    else:
                        # No replacement made, keep as string
                        final_args.append(arg)
                else:
                    final_args.append(arg)
            else:
                final_args.append(arg)

        return final_args

    def _process_arguments_with_rf_native_resolver(
        self, keyword: str, args: List[Any], session: Any
    ) -> List[Any]:
        """
        Process arguments using Robot Framework's native ArgumentResolver patterns.

        This is the general solution that handles:
        1. ObjectPreservingArgument objects from variable resolution
        2. Proper argument formatting (named vs positional parameters)
        3. Type preservation for object parameters

        This works for ANY library that expects object parameters, not just RequestsLibrary.
        """
        from robotmcp.components.variables.variable_resolver import (
            ObjectPreservingArgument,
        )

        processed_args = []

        for i, arg in enumerate(args):
            if isinstance(arg, ObjectPreservingArgument):
                # This is a named parameter with an object value
                print(
                    f"🔍 PROCESSING OBJECT ARG: {arg.param_name}={arg.value} (type: {type(arg.value).__name__})",
                    file=sys.stderr,
                )

                # For Robot Framework, we need to handle named parameters properly
                # The RF ArgumentResolver expects either:
                # 1. Positional args followed by named args like: ['value1', 'param2=value2']
                # 2. Or kwargs-style processing

                # Keep it as named parameter but preserve the object
                processed_args.append(f"{arg.param_name}={arg.value}")

            elif isinstance(arg, str) and "=" in arg and arg.count("=") == 1:
                # This is a string-based named parameter, handle URL parameter specially
                param_name, param_value = arg.split("=", 1)

                # For common first positional parameters like 'url', convert to positional
                if param_name == "url" and i == 0:  # First argument and it's URL
                    print(
                        f"🔍 CONVERTING URL TO POSITIONAL: {param_value}",
                        file=sys.stderr,
                    )
                    processed_args.append(param_value)
                else:
                    # Keep as named parameter
                    processed_args.append(arg)

            else:
                # Regular argument (positional or already processed)
                if not isinstance(arg, str):
                    # Convert non-string args to string
                    processed_args.append(str(arg))
                else:
                    processed_args.append(arg)

        return processed_args

    def _fix_stringified_objects_for_requests_library(
        self,
        keyword: str,
        original_args: List[str],
        resolved_args: List[str],
        session_variables: Dict[str, Any],
    ) -> List[Any]:
        """
        Fix stringified objects and argument format for RequestsLibrary keywords.

        This fixes two issues:
        1. Variable resolution converts objects to strings (e.g., json=${body} becomes "json={'key': 'value'}")
        2. Named parameters need proper formatting for RequestsLibrary (e.g., "url=value" → "value", "json=object" → object)
        """
        keyword_lower = keyword.lower()

        # Only apply this fix for RequestsLibrary keywords that expect object parameters
        requests_keywords_with_objects = {
            "post",
            "put",
            "patch",
            "post on session",
            "put on session",
            "patch on session",
        }

        if keyword_lower not in requests_keywords_with_objects:
            return resolved_args

        # Get the expected signature for this keyword
        from robotmcp.utils.rf_native_type_converter import REQUESTS_LIBRARY_SIGNATURES

        signature = REQUESTS_LIBRARY_SIGNATURES.get(keyword.upper(), [])

        print(
            f"🔍 REQUESTS SIGNATURE: {keyword.upper()} → {signature}", file=sys.stderr
        )

        fixed_args = []
        for i, (orig_arg, resolved_arg) in enumerate(zip(original_args, resolved_args)):
            # Check if this was a named parameter
            if (
                "=" in orig_arg
                and "=" in str(resolved_arg)
                and orig_arg.count("=") == 1
                and str(resolved_arg).count("=") == 1
            ):
                orig_param_name, orig_param_value = orig_arg.split("=", 1)
                resolved_param_name, resolved_param_value = str(resolved_arg).split(
                    "=", 1
                )

                print(
                    f"🔍 PROCESSING PARAM: {orig_param_name}={orig_param_value}",
                    file=sys.stderr,
                )

                # Handle URL parameter (first positional parameter for session-less methods)
                if orig_param_name == "url" and keyword_lower in [
                    "post",
                    "put",
                    "patch",
                    "get",
                    "delete",
                ]:
                    # URL should be positional, not named
                    print(
                        f"🔍 CONVERTING URL TO POSITIONAL: {resolved_param_value}",
                        file=sys.stderr,
                    )
                    fixed_args.append(resolved_param_value)
                    continue

                # Handle object parameters (json, data)
                if orig_param_name in ["json", "data"]:
                    # Check if original was a variable reference that should have been an object
                    if (
                        orig_param_value.startswith("${")
                        and orig_param_value.endswith("}")
                        and "[" not in orig_param_value
                    ):
                        var_name = orig_param_value[2:-1]  # Remove ${ and }
                        if var_name in session_variables:
                            original_value = session_variables[var_name]

                            # If the original value is a dict/list but got stringified, restore it
                            if isinstance(original_value, (dict, list)):
                                print(
                                    f"🔍 RESTORING OBJECT FOR {orig_param_name}: {orig_param_value} → object",
                                    file=sys.stderr,
                                )
                                # Keep it as named parameter but with restored object
                                fixed_args.append(f"{orig_param_name}={original_value}")
                                continue

                # Default: keep named parameter as-is
                fixed_args.append(resolved_arg)
            else:
                # Non-named parameter, keep as-is
                fixed_args.append(resolved_arg)

        print(f"🔍 FINAL FIXED ARGS: {fixed_args}", file=sys.stderr)
        return fixed_args

    def _extract_browser_state_updates(
        self, keyword: str, args: List[str], result: Any
    ) -> Dict[str, Any]:
        """Extract state updates from Browser Library keyword execution."""
        state_updates = {}
        keyword_lower = keyword.lower()

        # Extract state changes based on keyword
        if "new browser" in keyword_lower:
            browser_type = args[0] if args else "chromium"
            state_updates["current_browser"] = {"type": browser_type}
        elif "new context" in keyword_lower:
            state_updates["current_context"] = {
                "id": str(result) if result else "context"
            }
        elif "new page" in keyword_lower:
            url = args[0] if args else ""
            state_updates["current_page"] = {
                "id": str(result) if result else "page",
                "url": url,
            }
        elif "go to" in keyword_lower:
            url = args[0] if args else ""
            state_updates["current_page"] = {"url": url}

        return state_updates

    def _extract_selenium_state_updates(
        self, keyword: str, args: List[str], result: Any
    ) -> Dict[str, Any]:
        """Extract state updates from SeleniumLibrary keyword execution."""
        state_updates = {}
        keyword_lower = keyword.lower()

        # Extract state changes based on keyword
        if "open browser" in keyword_lower:
            state_updates["current_browser"] = {
                "type": args[1] if len(args) > 1 else "firefox"
            }
        elif "go to" in keyword_lower:
            state_updates["current_page"] = {"url": args[0] if args else ""}

        return state_updates

    def _apply_state_updates(
        self, session: ExecutionSession, state_updates: Dict[str, Any]
    ) -> None:
        """Apply state updates to session browser state."""
        if not state_updates:
            return

        browser_state = session.browser_state

        for key, value in state_updates.items():
            if key == "current_browser":
                if isinstance(value, dict):
                    browser_state.browser_type = value.get("type")
            elif key == "current_context":
                if isinstance(value, dict):
                    browser_state.context_id = value.get("id")
            elif key == "current_page":
                if isinstance(value, dict):
                    browser_state.current_url = value.get("url")
                    browser_state.page_id = value.get("id")

    async def _build_response_by_detail_level(
        self,
        detail_level: str,
        result: Dict[str, Any],
        step: ExecutionStep,
        keyword: str,
        arguments: List[str],
        session: ExecutionSession,
        resolved_arguments: List[str] = None,
    ) -> Dict[str, Any]:
        """Build execution response based on requested detail level."""
        base_response = {
            "success": result["success"],
            "step_id": step.step_id,
            "keyword": keyword,
            "arguments": arguments,  # Show original arguments in response
            "status": step.status,
            "execution_time": step.execution_time,
        }

        if not result["success"]:
            base_response["error"] = result.get("error", "Unknown error")

        if detail_level == "minimal":
            # Serialize output to prevent MCP serialization errors with complex objects
            raw_output = result.get("output", "")
            base_response["output"] = self.response_serializer.serialize_for_response(
                raw_output
            )
            # Include assigned variables in all detail levels for debugging
            if "assigned_variables" in result:
                base_response["assigned_variables"] = result["assigned_variables"]

        elif detail_level == "standard":
            # DUAL STORAGE: Keep ORIGINAL objects in session for RF, serialize ONLY for MCP response
            # Do NOT serialize session.variables as they need to remain original for RF execution
            session_vars_for_response = {}
            for var_name, var_value in session.variables.items():
                # Only serialize for MCP response display, but keep originals in session.variables
                session_vars_for_response[var_name] = (
                    self.response_serializer.serialize_for_response(var_value)
                )

            # Serialize output for standard detail level
            raw_output = result.get("output", "")
            serialized_output = self.response_serializer.serialize_for_response(
                raw_output
            )

            base_response.update(
                {
                    "output": serialized_output,
                    "session_variables": session_vars_for_response,  # Serialized for MCP response only
                    "active_library": session.get_active_library(),
                }
            )
            # Include assigned variables in standard detail level (serialized for MCP)
            if "assigned_variables" in result:
                base_response["assigned_variables"] = result["assigned_variables"]
            # Add resolved arguments for debugging if they differ from original (serialized)
            if resolved_arguments is not None and resolved_arguments != arguments:
                serialized_resolved_args = [
                    self.response_serializer.serialize_for_response(arg)
                    for arg in resolved_arguments
                ]
                base_response["resolved_arguments"] = serialized_resolved_args

        elif detail_level == "full":
            # DUAL STORAGE: Keep ORIGINAL objects in session for RF, serialize ONLY for MCP response
            session_vars_for_response = {}
            for var_name, var_value in session.variables.items():
                # Only serialize for MCP response display, but keep originals in session.variables
                session_vars_for_response[var_name] = (
                    self.response_serializer.serialize_for_response(var_value)
                )

            # Serialize output for full detail level
            raw_output = result.get("output", "")
            serialized_output = self.response_serializer.serialize_for_response(
                raw_output
            )

            # Serialize state_updates to prevent MCP serialization errors
            raw_state_updates = result.get("state_updates", {})
            serialized_state_updates = {}
            for key, value in raw_state_updates.items():
                serialized_state_updates[key] = (
                    self.response_serializer.serialize_for_response(value)
                )

            base_response.update(
                {
                    "output": serialized_output,
                    "session_variables": session_vars_for_response,  # Serialized for MCP response only
                    "state_updates": serialized_state_updates,
                    "active_library": session.get_active_library(),
                    "browser_state": {
                        "browser_type": session.browser_state.browser_type,
                        "current_url": session.browser_state.current_url,
                        "context_id": session.browser_state.context_id,
                        "page_id": session.browser_state.page_id,
                    },
                    "step_count": session.step_count,
                    "duration": session.duration,
                }
            )
            # Include assigned variables in full detail level
            if "assigned_variables" in result:
                base_response["assigned_variables"] = result["assigned_variables"]
            # Always include resolved arguments in full detail for debugging (serialized)
            if resolved_arguments is not None:
                serialized_resolved_args = [
                    self.response_serializer.serialize_for_response(arg)
                    for arg in resolved_arguments
                ]
                base_response["resolved_arguments"] = serialized_resolved_args

        return base_response

    def get_supported_detail_levels(self) -> List[str]:
        """Get list of supported detail levels."""
        return ["minimal", "standard", "full"]

    def validate_detail_level(self, detail_level: str) -> bool:
        """Validate that the detail level is supported."""
        return detail_level in self.get_supported_detail_levels()

    def _get_selenium_error_guidance(
        self, keyword: str, args: List[str], error_message: str
    ) -> Dict[str, Any]:
        """Generate SeleniumLibrary-specific error guidance for agents."""
        # Get base locator guidance
        guidance = self.rf_converter.get_selenium_locator_guidance(
            error_message, keyword
        )

        # Add keyword-specific guidance
        keyword_lower = keyword.lower()

        if any(
            term in keyword_lower
            for term in ["click", "input", "select", "clear", "wait"]
        ):
            # Element interaction keywords
            guidance["keyword_specific_tips"] = [
                f"'{keyword}' requires a valid element locator as the first argument",
                "Common locator patterns: 'id:elementId', 'name:fieldName', 'css:.className'",
                "Ensure the element is visible and interactable before interaction",
            ]

            # Analyze the locator argument if provided
            if args:
                locator = args[0]
                if not any(strategy in locator for strategy in [":", "="]):
                    guidance["locator_analysis"] = {
                        "provided_locator": locator,
                        "issue": "Locator appears to be missing strategy prefix",
                        "suggestions": [
                            f"Try 'id:{locator}' if it's an ID",
                            f"Try 'name:{locator}' if it's a name attribute",
                            f"Try 'css:{locator}' if it's a CSS selector",
                            f"Try 'xpath://*[@id=\"{locator}\"]' for XPath",
                        ],
                    }
                elif "=" in locator and ":" not in locator:
                    guidance["locator_analysis"] = {
                        "provided_locator": locator,
                        "issue": "Contains '=' but no strategy prefix - may be parsed as named argument",
                        "correct_format": f"name:{locator}"
                        if locator.startswith("name=")
                        else "Use appropriate strategy prefix",
                        "note": "SeleniumLibrary requires 'strategy:value' format, not 'strategy=value'",
                    }

        elif "open" in keyword_lower or "browser" in keyword_lower:
            guidance["keyword_specific_tips"] = [
                f"'{keyword}' manages browser/session state",
                "Ensure proper browser initialization before element interactions",
                "Check browser driver compatibility and installation",
            ]

        return guidance

    def _get_browser_error_guidance(
        self, keyword: str, args: List[str], error_message: str
    ) -> Dict[str, Any]:
        """Generate Browser Library-specific error guidance for agents."""
        # Get base locator guidance
        guidance = self.rf_converter.get_browser_locator_guidance(
            error_message, keyword
        )

        # Add keyword-specific guidance
        keyword_lower = keyword.lower()

        if any(
            term in keyword_lower
            for term in ["click", "fill", "select", "check", "type", "press", "hover"]
        ):
            # Element interaction keywords
            guidance["keyword_specific_tips"] = [
                f"'{keyword}' requires a valid element selector",
                "Browser Library uses CSS selectors by default (no prefix needed)",
                "Common patterns: '.class', '#id', 'button', 'input[type=\"submit\"]'",
                "For complex elements, use cascaded selectors: 'div.container >> .button'",
            ]

            # Analyze the selector argument if provided
            if args:
                selector = args[0]
                guidance.update(self._analyze_browser_selector(selector))

        elif any(
            term in keyword_lower
            for term in ["new browser", "new page", "new context", "go to"]
        ):
            guidance["keyword_specific_tips"] = [
                f"'{keyword}' manages browser/page state",
                "Ensure proper browser initialization sequence",
                "Check browser installation and dependencies",
                "Verify URL accessibility for navigation keywords",
            ]

        elif "wait" in keyword_lower:
            guidance["keyword_specific_tips"] = [
                f"'{keyword}' handles dynamic content and timing",
                "Adjust timeout values for slow-loading elements",
                "Use appropriate wait conditions (visible, hidden, enabled, etc.)",
                "Consider page load states for complete readiness",
            ]

        return guidance

    def _analyze_browser_selector(self, selector: str) -> Dict[str, Any]:
        """Analyze a Browser Library selector and provide specific guidance."""
        analysis = {}

        # Detect selector patterns and provide guidance (order matters - check >>> before >>)
        if ">>>" in selector:
            analysis["iframe_selector_detected"] = {
                "type": "iFrame piercing selector",
                "explanation": "Using >>> to access elements inside frames",
                "tip": "Left side selects frame, right side selects element inside frame",
            }

        elif selector.startswith("#") and not selector.startswith("\\#"):
            analysis["selector_warning"] = {
                "issue": "ID selector may need escaping in Robot Framework",
                "provided_selector": selector,
                "recommended": f"\\{selector}",
                "explanation": "# is a comment character in Robot Framework, use \\# for ID selectors",
            }

        elif ">>" in selector:
            analysis["cascaded_selector_detected"] = {
                "type": "Cascaded selector (good practice)",
                "explanation": "Using >> to chain multiple selector strategies",
                "tip": "Each part of the chain is relative to the previous match",
            }

        elif selector.startswith('"') and selector.endswith('"'):
            analysis["text_selector_detected"] = {
                "type": "Text selector (implicit)",
                "explanation": "Quoted strings are treated as text selectors",
                "equivalent_explicit": f"text={selector}",
                "tip": "Use for exact text matching",
            }

        elif selector.startswith("//") or selector.startswith(".."):
            analysis["xpath_selector_detected"] = {
                "type": "XPath selector (implicit)",
                "explanation": "Selectors starting with // or .. are treated as XPath",
                "equivalent_explicit": f"xpath={selector}",
                "tip": "XPath provides powerful element traversal capabilities",
            }

        elif "=" in selector and any(
            selector.startswith(prefix) for prefix in ["css=", "xpath=", "text=", "id="]
        ):
            strategy = selector.split("=", 1)[0]
            analysis["explicit_strategy_detected"] = {
                "type": f"Explicit {strategy} selector",
                "explanation": f"Using explicit {strategy} strategy",
                "tip": "Good practice to be explicit with selector strategies",
            }

        else:
            analysis["implicit_css_detected"] = {
                "type": "CSS selector (implicit default)",
                "explanation": "Plain selectors are treated as CSS by default",
                "equivalent_explicit": f"css={selector}",
                "tip": "Browser Library defaults to CSS selectors",
            }

        return analysis

    async def _execute_mobile_keyword(
        self, session: ExecutionSession, keyword_name: str, args: List[Any]
    ) -> Dict[str, Any]:
        """
        Execute mobile-specific keywords through AppiumLibrary.

        This method handles mobile keywords that would otherwise fail with
        "Cannot access execution context" when executed through BuiltIn.run_keyword.

        Args:
            session: ExecutionSession containing mobile configuration
            keyword_name: Mobile keyword name (e.g., "Get Source")
            args: Keyword arguments

        Returns:
            Execution result with status, output, and state
        """
        try:
            logger.info(f"Executing mobile keyword: '{keyword_name}' with args: {args}")

            # Use keyword discovery system for mobile keywords
            from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery

            keyword_discovery = get_keyword_discovery()
            result = await keyword_discovery.execute_keyword(
                keyword_name=keyword_name,
                args=args,
                session_variables=session.variables,
                active_library="AppiumLibrary",
                session_id=session.session_id,
            )

            if result and result.get("success"):
                # Process successful result
                output = result.get("output", "")
                result_value = result.get("result")

                # For Get Source, the result might be in either output or result field
                if (
                    keyword_name.lower() in ["get source", "get page source"]
                    and not output
                    and result_value
                ):
                    output = str(result_value)

                response = {
                    "success": True,
                    "output": output,
                    "result": result_value,
                    "variables": result.get("variables", {}),
                    "state_updates": result.get("state_updates", {}),
                }

                logger.info(f"Mobile keyword '{keyword_name}' executed successfully")
                return response
            else:
                # Keyword discovery failed, try fallback
                error_msg = (
                    result.get("error", "Unknown error")
                    if result
                    else "No result from keyword discovery"
                )
                logger.debug(
                    f"Keyword discovery failed for mobile keyword '{keyword_name}': {error_msg}"
                )

                # For Get Source, try direct library method call as fallback
                if keyword_name.lower() == "get source":
                    return await self._execute_get_source_fallback(session)

                return {
                    "success": False,
                    "error": f"Mobile keyword execution failed: {error_msg}",
                    "output": "",
                    "variables": {},
                    "state_updates": {},
                }

        except Exception as e:
            logger.error(f"Error executing mobile keyword '{keyword_name}': {e}")

            # Try fallback for Get Source
            if keyword_name.lower() == "get source":
                try:
                    return await self._execute_get_source_fallback(session)
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback also failed for Get Source: {fallback_error}"
                    )

            return {
                "success": False,
                "error": f"Mobile keyword execution error: {str(e)}",
                "output": "",
                "variables": {},
                "state_updates": {},
            }

    async def _execute_get_source_fallback(
        self, session: ExecutionSession
    ) -> Dict[str, Any]:
        """
        Fallback method for Get Source using direct AppiumLibrary method call.

        Args:
            session: ExecutionSession containing mobile configuration

        Returns:
            Execution result
        """
        try:
            from robotmcp.core.library_manager import LibraryManager

            library_manager = LibraryManager()

            # Try to get AppiumLibrary instance
            appium_lib = None
            if hasattr(library_manager, "_libraries"):
                for lib_name, lib_instance in library_manager._libraries.items():
                    if lib_name == "AppiumLibrary" or "appium" in lib_name.lower():
                        appium_lib = lib_instance
                        break

            if appium_lib and hasattr(appium_lib, "get_source"):
                source = appium_lib.get_source()
                logger.info(
                    f"Got mobile source via fallback: {len(source) if source else 0} chars"
                )

                return {
                    "success": True,
                    "output": source,
                    "result": source,
                    "variables": {},
                    "state_updates": {},
                }
            else:
                return {
                    "success": False,
                    "error": "No active mobile app session: AppiumLibrary not properly loaded",
                    "output": "",
                    "variables": {},
                    "state_updates": {},
                }

        except Exception as e:
            logger.error(f"Get Source fallback failed: {e}")
            return {
                "success": False,
                "error": f"No active mobile app session: {str(e)}",
                "output": "",
                "variables": {},
                "state_updates": {},
            }

    def _get_session_libraries(self, session: ExecutionSession) -> List[str]:
        """Get list of libraries loaded in the session for session-aware keyword resolution.

        Args:
            session: ExecutionSession to get libraries from

        Returns:
            List of library names loaded in the session
        """
        session_libraries = []

        # Try to get loaded libraries from session
        if hasattr(session, "loaded_libraries") and session.loaded_libraries:
            session_libraries = list(session.loaded_libraries)
        elif hasattr(session, "search_order") and session.search_order:
            session_libraries = list(session.search_order)
        elif hasattr(session, "imported_libraries") and session.imported_libraries:
            session_libraries = list(session.imported_libraries)

        # Always include core built-in libraries
        builtin_libraries = ["BuiltIn", "Collections", "String"]
        for lib in builtin_libraries:
            if lib not in session_libraries:
                session_libraries.append(lib)

        logger.debug(f"Session libraries for keyword resolution: {session_libraries}")
        return session_libraries

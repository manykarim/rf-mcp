"""Main MCP Server implementation for Robot Framework integration."""

import logging
from typing import Any, Dict, List, Union

from fastmcp import FastMCP

from robotmcp.components.execution import ExecutionCoordinator
from robotmcp.components.keyword_matcher import KeywordMatcher
from robotmcp.components.library_recommender import LibraryRecommender
from robotmcp.components.nlp_processor import NaturalLanguageProcessor
from robotmcp.components.state_manager import StateManager
from robotmcp.components.test_builder import TestBuilder

logger = logging.getLogger(__name__)


# Initialize FastMCP server
mcp = FastMCP("Robot Framework MCP Server")

# Initialize components
nlp_processor = NaturalLanguageProcessor()
keyword_matcher = KeywordMatcher()
library_recommender = LibraryRecommender()
execution_engine = ExecutionCoordinator()
state_manager = StateManager()
test_builder = TestBuilder(execution_engine)


@mcp.tool
async def analyze_scenario(
    scenario: str, context: str = "web", session_id: str = None
) -> Dict[str, Any]:
    """Process natural language test description into structured test intent.

    CRITICAL: This tool ALWAYS creates a session for your test execution.
    Use the returned session_id in ALL subsequent tool calls (execute_step, build_test_suite, etc.)
    
    RECOMMENDED WORKFLOW - STEP 1 OF 4:
    This tool should be used as the FIRST step in the Robot Framework automation workflow:
    1. ✅ analyze_scenario (THIS TOOL) - Creates session and understands requirements
    2. ➡️ recommend_libraries - Get targeted library suggestions  
    3. ➡️ execute_step - Execute steps using the SAME session_id
    4. ➡️ build_test_suite - Build suite using the SAME session_id

    Using this order prevents unnecessary library checks and pip installations by ensuring
    you only verify libraries that are actually relevant to the user's scenario.

    NEW FEATURE: Automatic Session Creation
    - If session_id not provided: Creates new unique session ID
    - If session_id provided: Uses existing session or creates new one
    - Session is auto-configured based on scenario analysis and explicit library preferences
    - Returns session_id that MUST be used in all subsequent calls

    Args:
        scenario: Human language scenario description
        context: Optional context about the application (web, mobile, API, etc.)
        session_id: Optional session ID to create and auto-configure for this scenario

    Returns:
        Structured test intent with session_info containing session_id for subsequent calls.
        Session is automatically configured with optimal library choices for the scenario.
    """
    # Analyze the scenario first
    result = await nlp_processor.analyze_scenario(scenario, context)

    # ALWAYS create a session - either use provided ID or generate one
    if not session_id:
        session_id = execution_engine.session_manager.create_session_id()
        logger.info(f"Auto-generated session ID: {session_id}")
    else:
        logger.info(f"Using provided session ID: {session_id}")

    logger.info(
        f"Creating and auto-configuring session '{session_id}' based on scenario analysis"
    )

    # Get or create session using execution coordinator
    session = execution_engine.session_manager.get_or_create_session(session_id)

    # Auto-configure session based on scenario
    session.configure_from_scenario(scenario)

    # Enhanced session info with guidance
    result["session_info"] = {
        "session_id": session_id,
        "auto_configured": session.auto_configured,
        "session_type": session.session_type.value,
        "explicit_library_preference": session.explicit_library_preference,
        "recommended_libraries": session.get_libraries_to_load(),
        "search_order": session.get_search_order(),
        "libraries_loaded": list(session.loaded_libraries),
        "next_step_guidance": f"Use session_id='{session_id}' in all subsequent tool calls",
        "status": "active",
        "ready_for_execution": True
    }

    logger.info(
        f"Session '{session_id}' configured: type={session.session_type.value}, preference={session.explicit_library_preference}"
    )

    return result


@mcp.tool
async def discover_keywords(
    action_description: str, context: str = "web", current_state: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Find matching Robot Framework keywords for an action.

    Args:
        action_description: Description of the action to perform
        context: Current context (web, mobile, API, etc.)
        current_state: Current application state
    """
    if current_state is None:
        current_state = {}
    return await keyword_matcher.discover_keywords(
        action_description, context, current_state
    )


@mcp.tool
async def execute_step(
    keyword: str,
    arguments: List[str] = None,
    session_id: str = "default",
    raise_on_failure: bool = True,
    detail_level: str = "minimal",
    scenario_hint: str = None,
    assign_to: Union[str, List[str]] = None,
) -> Dict[str, Any]:
    """Execute a single test step using Robot Framework API.

    STEPWISE TEST DEVELOPMENT GUIDANCE:
    - ALWAYS execute and verify each keyword individually BEFORE building test suites
    - Test each step to confirm it works as expected
    - Only add verified keywords to .robot files
    - Use this method to validate arguments and behavior step-by-step
    - Build incrementally: execute_step() → verify → add to suite

    Args:
        keyword: Robot Framework keyword name
        arguments: Arguments for the keyword (supports both positional and named: ["arg1", "param=value"])
        session_id: Session identifier for maintaining context
        raise_on_failure: If True, raises exception for failed steps (proper MCP failure reporting).
                         If False, returns failure details in response (for debugging/analysis).
        detail_level: Level of detail in response ('minimal', 'standard', 'full').
                     'minimal' reduces response size for AI agents by ~80-90%.
        scenario_hint: Optional scenario text for intelligent library auto-configuration.
                      When provided on first call, automatically configures the session
                      based on detected scenario type and explicit library preferences.
        assign_to: Variable name(s) to assign the keyword's return value to.
                  Single string for single assignment: "result" creates ${result}
                  List of strings for multi-assignment: ["first", "rest"] creates ${first}, ${rest}
    """
    if arguments is None:
        arguments = []

    result = await execution_engine.execute_step(
        keyword,
        arguments,
        session_id,
        detail_level,
        scenario_hint=scenario_hint,
        assign_to=assign_to,
    )

    # For proper MCP protocol compliance, failed steps should raise exceptions
    # This ensures AI agents see failures as red/failed instead of green/successful
    if not result.get("success", False) and raise_on_failure:
        error_msg = result.get("error", f"Step '{keyword}' failed")

        # Create detailed error message including suggestions if available
        detailed_error = f"Step execution failed: {error_msg}"
        if "suggestions" in result:
            detailed_error += f"\nSuggestions: {', '.join(result['suggestions'])}"
        if "step_id" in result:
            detailed_error += f"\nStep ID: {result['step_id']}"

        raise Exception(detailed_error)

    return result


@mcp.tool
async def get_application_state(
    state_type: str = "all",
    elements_of_interest: List[str] = None,
    session_id: str = "default",
) -> Dict[str, Any]:
    """Retrieve current application state.

    Args:
        state_type: Type of state to retrieve (dom, api, database, all)
        elements_of_interest: Specific elements to focus on
        session_id: Session identifier
    """
    if elements_of_interest is None:
        elements_of_interest = []
    return await state_manager.get_state(
        state_type, elements_of_interest, session_id, execution_engine
    )


@mcp.tool
async def suggest_next_step(
    current_state: Dict[str, Any],
    test_objective: str,
    executed_steps: List[Dict[str, Any]] = None,
    session_id: str = "default",
) -> Dict[str, Any]:
    """AI-driven suggestion for next test step.

    Args:
        current_state: Current application state
        test_objective: Overall test objective
        executed_steps: Previously executed steps
        session_id: Session identifier
    """
    if executed_steps is None:
        executed_steps = []
    return await nlp_processor.suggest_next_step(
        current_state, test_objective, executed_steps, session_id
    )


@mcp.tool
async def build_test_suite(
    test_name: str,
    session_id: str = "",
    tags: List[str] = None,
    documentation: str = "",
    remove_library_prefixes: bool = True,
) -> Dict[str, Any]:
    """Generate Robot Framework test suite from successful steps with intelligent session resolution.

    IMPORTANT: Only use AFTER validating all steps individually with execute_step().
    This tool generates .robot files from previously executed and verified steps.
    Do NOT write test suites before confirming each keyword works correctly.

    Enhanced Session Resolution:
    - If session_id provided and valid: Uses that session
    - If session_id empty/invalid: Automatically finds most suitable session with steps
    - Provides clear guidance on session issues and recovery options

    Recommended workflow:
    1. Use analyze_scenario() to create configured session
    2. Use execute_step() to test each keyword with the SAME session_id
    3. Use build_test_suite() with the SAME session_id to create .robot files

    Args:
        test_name: Name for the test case
        session_id: Session with executed steps (auto-resolves if empty/invalid)
        tags: Test tags
        documentation: Test documentation
        remove_library_prefixes: Remove library prefixes from keywords (default: True)
    """
    if tags is None:
        tags = []
    
    # Import session resolver here to avoid circular imports
    from robotmcp.utils.session_resolution import SessionResolver
    session_resolver = SessionResolver(execution_engine.session_manager)
    
    # Resolve session with intelligent fallback
    resolution_result = session_resolver.resolve_session_with_fallback(session_id)
    
    if not resolution_result["success"]:
        # Return enhanced error with guidance
        return {
            "success": False,
            "error": "Session not ready for test suite generation",
            "error_details": resolution_result["error_guidance"],
            "guidance": [
                "Create a session and execute some steps first",
                "Use the session_id returned by analyze_scenario",
                "Check session status with get_session_validation_status"
            ],
            "validation_summary": {"passed": 0, "failed": 0},
            "recommendation": "Start with analyze_scenario() to create a properly configured session"
        }
    
    # Use resolved session ID
    resolved_session_id = resolution_result["session_id"]
    
    # Build the test suite with resolved session
    result = await test_builder.build_suite(
        resolved_session_id, test_name, tags, documentation, remove_library_prefixes
    )
    
    # Add session resolution info to result
    if resolution_result.get("fallback_used", False):
        result["session_resolution"] = {
            "fallback_used": True,
            "original_session_id": session_id,
            "resolved_session_id": resolved_session_id,
            "message": f"Automatically used session '{resolved_session_id}' with {resolution_result['session_info']['step_count']} executed steps"
        }
    else:
        result["session_resolution"] = {
            "fallback_used": False,
            "session_id": resolved_session_id
        }
    
    return result


@mcp.tool
async def validate_scenario(
    parsed_scenario: Dict[str, Any], available_libraries: List[str] = None
) -> Dict[str, Any]:
    """Pre-execution validation of scenario feasibility.

    Args:
        parsed_scenario: Parsed scenario from analyze_scenario
        available_libraries: List of available RF libraries
    """
    if available_libraries is None:
        available_libraries = []
    return await nlp_processor.validate_scenario(parsed_scenario, available_libraries)


@mcp.tool
async def recommend_libraries(
    scenario: str,
    context: str = "web",
    max_recommendations: int = 5,
    session_id: str = None,
) -> Dict[str, Any]:
    """Recommend Robot Framework libraries based on test scenario.

    RECOMMENDED WORKFLOW - STEP 2 OF 3:
    This tool should be used as the SECOND step in the Robot Framework automation workflow:
    1. ✅ analyze_scenario - Understand what the user wants to accomplish
    2. ✅ recommend_libraries (THIS TOOL) - Get targeted library suggestions for the scenario
    3. ➡️ check_library_availability - Verify only the recommended libraries

    IMPORTANT: Use the scenario output from analyze_scenario as input to this tool for
    the most accurate library recommendations. This prevents checking irrelevant libraries
    in the next step.

    NEW: Session Management Integration
    If session_id is provided, this tool will setup the session with recommended libraries
    and configure library search order for optimal keyword resolution.

    Args:
        scenario: Natural language description of the test scenario (ideally from analyze_scenario output)
        context: Testing context (web, mobile, api, database, desktop, system, visual)
        max_recommendations: Maximum number of library recommendations to return
        session_id: Optional session ID to setup with recommended libraries

    Returns:
        Targeted library recommendations that should be passed to check_library_availability.
        If session_id provided, also includes session setup details.
    """
    # Get library recommendations
    result = library_recommender.recommend_libraries(
        scenario, context, max_recommendations
    )

    # If session_id provided, setup session with recommended libraries
    if session_id:
        logger.info(f"Setting up session '{session_id}' with recommended libraries")

        # Get or create session
        session = execution_engine.session_manager.get_or_create_session(session_id)

        # If not already auto-configured, configure from scenario
        if not session.auto_configured:
            session.configure_from_scenario(scenario)

        # Get recommended libraries from result
        recommended_libs = result.get("recommended_libraries", [])

        # Setup library search order based on recommendations
        if recommended_libs:
            # Update session search order to prioritize recommended libraries
            session.search_order = recommended_libs + [
                lib for lib in session.search_order if lib not in recommended_libs
            ]
            logger.info(
                f"Updated session '{session_id}' search order: {session.search_order[:3]}..."
            )

        # Add session setup info to result
        result["session_setup"] = {
            "session_id": session_id,
            "configured": True,
            "search_order": session.search_order,
            "session_type": session.session_type.value,
            "explicit_preference": session.explicit_library_preference,
            "recommended_libraries_applied": recommended_libs,
        }

    return result


@mcp.tool
async def get_page_source(
    session_id: str = "default",
    full_source: bool = False,
    filtered: bool = False,
    filtering_level: str = "standard",
) -> Dict[str, Any]:
    """Get page source and context for a browser session with optional DOM filtering.
    Call this tool after opening a web page or when changes are done to the page.

    Args:
        session_id: Session identifier
        full_source: If True, returns complete page source. If False, returns preview only.
        filtered: If True, returns filtered page source with only automation-relevant content.
        filtering_level: Filtering intensity when filtered=True:
                        - 'minimal': Remove only scripts and styles
                        - 'standard': Remove scripts, styles, metadata, SVG, embeds (default)
                        - 'aggressive': Remove all non-interactive elements and media

    Returns:
        Dict with page source, metadata, and filtering information. When filtered=True,
        includes both original and filtered page source lengths for comparison.
    """
    return await execution_engine.get_page_source(
        session_id, full_source, filtered, filtering_level
    )


@mcp.tool
async def check_library_availability(libraries: List[str]) -> Dict[str, Any]:
    """Check if Robot Framework libraries are available before installation.

    RECOMMENDED WORKFLOW - STEP 3 OF 3:
    This tool should be used as the THIRD step in the Robot Framework automation workflow:
    1. ✅ analyze_scenario - Understand what the user wants to accomplish
    2. ✅ recommend_libraries - Get targeted library suggestions for the scenario
    3. ✅ check_library_availability (THIS TOOL) - Verify only the recommended libraries

    CRITICAL: Do NOT call this tool first! It may return empty results if called before
    the Robot Framework environment is initialized, leading to unnecessary pip installations.

    PREFERRED INPUT: Use the library recommendations from recommend_libraries as the
    'libraries' parameter to avoid checking irrelevant libraries.

    FALLBACK INITIALIZATION: If you must call this tool without the recommended workflow,
    first call 'get_available_keywords' or 'execute_step' to initialize library discovery,
    then re-run this check.

    Args:
        libraries: List of library names to check (preferably from recommend_libraries output)

    Returns:
        Dict with availability status, installation suggestions, and workflow guidance.
        Includes smart hints if called in wrong order or without initialization.

    Example workflow:
        scenario_result = await analyze_scenario("I want to test a web form")
        recommendations = await recommend_libraries(scenario_result["scenario"])
        availability = await check_library_availability(recommendations["recommended_libraries"])
    """
    return execution_engine.check_library_requirements(libraries)


@mcp.tool
async def get_library_status(library_name: str) -> Dict[str, Any]:
    """Get detailed installation status for a specific library.

    Args:
        library_name: Name of the library to check (e.g., 'Browser', 'SeleniumLibrary')

    Returns:
        Dict with detailed status and installation information
    """
    return execution_engine.get_installation_status(library_name)


@mcp.tool
async def get_available_keywords(library_name: str = None) -> List[Dict[str, Any]]:
    """Get available Robot Framework keywords with native RF libdoc short documentation, optionally filtered by library.

    Uses Robot Framework's native libdoc API to provide accurate short_doc, argument types, and metadata.
    Falls back to inspection-based discovery if libdoc is not available.

    NOTE: This tool initializes library discovery and can be used as a fallback if you need to call
    check_library_availability without following the recommended 3-step workflow.

    Args:
        library_name: Optional library name to filter keywords (e.g., 'Browser', 'BuiltIn', 'Collections').
                     If not provided, returns all keywords from all loaded libraries.

    Returns:
        List of keyword information including:
        - name: Keyword name
        - library: Library name
        - args: List of argument names
        - arg_types: List of argument types (when available from libdoc)
        - short_doc: Short documentation from Robot Framework's native short_doc
        - tags: Keyword tags
        - is_deprecated: Whether keyword is deprecated (libdoc only)

    Related tools: Use analyze_scenario → recommend_libraries → check_library_availability for optimal workflow.
    """
    return execution_engine.get_available_keywords(library_name)


@mcp.tool
async def search_keywords(pattern: str) -> List[Dict[str, Any]]:
    """Search for Robot Framework keywords matching a pattern using native RF libdoc.

    Uses Robot Framework's native libdoc API for accurate search results and documentation.
    Searches through keyword names, documentation, short_doc, and tags.

    Args:
        pattern: Search pattern to match against keyword names, documentation, or tags

    Returns:
        List of matching keywords with native RF libdoc metadata including short_doc,
        argument types, deprecation status, and enhanced tag information.
    """
    return execution_engine.search_keywords(pattern)


@mcp.tool
async def get_keyword_documentation(
    keyword_name: str, library_name: str = None
) -> Dict[str, Any]:
    """Get full documentation for a specific Robot Framework keyword using native RF libdoc.

    Uses Robot Framework's native LibraryDocumentation and KeywordDoc objects to provide
    comprehensive keyword information including source location, argument types, and
    deprecation status when available.

    Args:
        keyword_name: Name of the keyword to get documentation for
        library_name: Optional library name to narrow search

    Returns:
        Dict containing comprehensive keyword information:
        - success: Boolean indicating if keyword was found
        - keyword: Dict with keyword details including:
          - name, library, args: Basic keyword information
          - arg_types: Argument types from libdoc (when available)
          - doc: Full documentation text
          - short_doc: Native Robot Framework short_doc
          - tags: Keyword tags
          - is_deprecated: Deprecation status (libdoc only)
          - source: Source file path (libdoc only)
          - lineno: Line number in source (libdoc only)
    """
    return execution_engine.get_keyword_documentation(keyword_name, library_name)


# TOOL DISABLED: validate_step_before_suite
#
# Reason for removal: This tool is functionally redundant with execute_step().
# Analysis shows that it duplicates execution (performance impact) and adds
# minimal unique value beyond what execute_step() already provides.
#
# Key issues:
# 1. Functional redundancy - re-executes the same step as execute_step()
# 2. Performance overhead - double execution of steps
# 3. Agent confusion - two similar tools with overlapping purposes
# 4. Limited additional value - only adds guidance text and redundant metadata
#
# The validation workflow can be achieved with:
# execute_step() → validate_test_readiness() → build_test_suite()
#
# @mcp.tool
# async def validate_step_before_suite(
#     keyword: str,
#     arguments: List[str] = None,
#     session_id: str = "default",
#     expected_outcome: str = None,
# ) -> Dict[str, Any]:
#     """Validate a single step before adding it to a test suite.
#
#     This method enforces stepwise test development by requiring step validation
#     before suite generation. Use this to verify each keyword works as expected.
#
#     Workflow:
#     1. Call this method for each test step
#     2. Verify the step succeeds and produces expected results
#     3. Only after all steps are validated, use build_test_suite()
#
#     Args:
#         keyword: Robot Framework keyword to validate
#         arguments: Arguments for the keyword
#         session_id: Session identifier
#         expected_outcome: Optional description of expected result for validation
#
#     Returns:
#         Validation result with success status, output, and recommendations
#     """
#     if arguments is None:
#         arguments = []
#
#     # Execute the step with detailed error reporting
#     result = await execution_engine.execute_step(
#         keyword, arguments, session_id, detail_level="full"
#     )
#
#     # Add validation metadata
#     result["validated"] = result.get("success", False)
#     result["validation_time"] = result.get("execution_time")
#
#     if expected_outcome:
#         result["expected_outcome"] = expected_outcome
#         result["meets_expectation"] = "unknown"  # AI agent should evaluate this
#
#     # Add guidance for next steps
#     if result.get("success"):
#         result["next_step_guidance"] = (
#             "✅ Step validated successfully. Safe to include in test suite."
#         )
#     else:
#         result["next_step_guidance"] = (
#             "❌ Step failed validation. Fix issues before adding to test suite."
#         )
#         result["debug_suggestions"] = [
#             "Check keyword spelling and library availability",
#             "Verify argument types and values",
#             "Ensure required browser/context is open",
#             "Review error message for specific issues",
#         ]
#
#     return result


@mcp.tool
async def get_session_validation_status(session_id: str = "") -> Dict[str, Any]:
    """Get validation status of all steps in a session with intelligent session resolution.

    Use this to check which steps have been validated and are ready for test suite generation.
    Helps ensure stepwise test development by showing validation progress.

    Enhanced Session Resolution:
    - If session_id provided and valid: Uses that session
    - If session_id empty/invalid: Automatically finds most suitable session with steps

    Args:
        session_id: Session identifier to check (auto-resolves if empty/invalid)

    Returns:
        Validation status with passed/failed step counts and readiness assessment
    """
    # Import session resolver here to avoid circular imports
    from robotmcp.utils.session_resolution import SessionResolver
    session_resolver = SessionResolver(execution_engine.session_manager)
    
    # Resolve session with intelligent fallback
    resolution_result = session_resolver.resolve_session_with_fallback(session_id)
    
    if not resolution_result["success"]:
        # Return enhanced error with guidance
        return {
            "success": False,
            "error": f"Session '{session_id}' not found",
            "error_details": resolution_result["error_guidance"],
            "available_sessions": resolution_result["error_guidance"]["available_sessions"],
            "sessions_with_steps": resolution_result["error_guidance"]["sessions_with_steps"],
            "recommendation": "Use analyze_scenario() to create a session first"
        }
    
    # Use resolved session ID
    resolved_session_id = resolution_result["session_id"]
    
    # Get validation status for resolved session
    result = execution_engine.get_session_validation_status(resolved_session_id)
    
    # Add session resolution info to result
    if resolution_result.get("fallback_used", False):
        result["session_resolution"] = {
            "fallback_used": True,
            "original_session_id": session_id,
            "resolved_session_id": resolved_session_id,
            "message": f"Automatically checked session '{resolved_session_id}'"
        }
    else:
        result["session_resolution"] = {
            "fallback_used": False,
            "session_id": resolved_session_id
        }
    
    return result


@mcp.tool
async def validate_test_readiness(session_id: str = "default") -> Dict[str, Any]:
    """Check if session is ready for test suite generation.

    Enforces stepwise workflow by verifying all steps have been validated.
    Use this before calling build_test_suite() to ensure quality.

    Args:
        session_id: Session identifier to validate

    Returns:
        Readiness status with guidance on next actions
    """
    return await execution_engine.validate_test_readiness(session_id)


@mcp.tool
async def set_library_search_order(
    libraries: List[str], session_id: str = "default"
) -> Dict[str, Any]:
    """Set explicit library search order for keyword resolution (like RF Set Library Search Order).

    This tool implements Robot Framework's Set Library Search Order concept, allowing explicit
    control over which library's keywords take precedence when multiple libraries have the
    same keyword name.

    Args:
        libraries: List of library names in priority order (highest priority first)
        session_id: Session identifier to configure

    Returns:
        Dict with success status, applied search order, and any warnings about invalid libraries

    Example:
        # Prioritize SeleniumLibrary over Browser Library for web automation
        await set_library_search_order(["SeleniumLibrary", "BuiltIn", "Collections"], "web_session")

        # Prioritize RequestsLibrary for API testing
        await set_library_search_order(["RequestsLibrary", "BuiltIn", "String"], "api_session")
    """
    try:
        # Get or create session
        session = execution_engine.session_manager.get_or_create_session(session_id)

        # Set library search order
        old_order = session.get_search_order()
        session.set_library_search_order(libraries)
        new_order = session.get_search_order()

        return {
            "success": True,
            "session_id": session_id,
            "old_search_order": old_order,
            "new_search_order": new_order,
            "libraries_requested": libraries,
            "libraries_applied": new_order,
            "message": f"Library search order updated for session '{session_id}'",
        }

    except Exception as e:
        logger.error(f"Error setting library search order: {e}")
        return {"success": False, "error": str(e), "session_id": session_id}


@mcp.tool
async def get_session_info(session_id: str = "default") -> Dict[str, Any]:
    """Get comprehensive information about a session's configuration and state.

    Args:
        session_id: Session identifier to get information for

    Returns:
        Dict with session configuration, library status, and execution history
    """
    try:
        session = execution_engine.session_manager.get_session(session_id)

        if not session:
            return {
                "success": False,
                "error": f"Session '{session_id}' not found",
                "available_sessions": execution_engine.session_manager.get_all_session_ids(),
            }

        return {"success": True, "session_info": session.get_session_info()}

    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        return {"success": False, "error": str(e), "session_id": session_id}


@mcp.tool
async def get_selenium_locator_guidance(
    error_message: str = None, keyword_name: str = None
) -> Dict[str, Any]:
    """Get comprehensive SeleniumLibrary locator strategy guidance for AI agents.

    This tool helps AI agents understand SeleniumLibrary's locator strategies and
    provides context-aware suggestions for element location and error resolution.

    SeleniumLibrary supports these locator strategies:
    - id: Element id (e.g., 'id:example')
    - name: name attribute (e.g., 'name:example')
    - identifier: Either id or name (e.g., 'identifier:example')
    - class: Element class (e.g., 'class:example')
    - tag: Tag name (e.g., 'tag:div')
    - xpath: XPath expression (e.g., 'xpath://div[@id="example"]')
    - css: CSS selector (e.g., 'css:div#example')
    - dom: DOM expression (e.g., 'dom:document.images[5]')
    - link: Exact link text (e.g., 'link:Click Here')
    - partial link: Partial link text (e.g., 'partial link:Click')
    - data: Element data-* attribute (e.g., 'data:id:my_id')
    - jquery: jQuery expression (e.g., 'jquery:div.example')
    - default: Keyword-specific default (e.g., 'default:example')

    Args:
        error_message: Optional error message to analyze for specific guidance
        keyword_name: Optional keyword name that failed for context-specific tips

    Returns:
        Comprehensive locator strategy guidance with examples, tips, and error-specific advice
    """
    from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter

    converter = RobotFrameworkNativeConverter()
    return converter.get_selenium_locator_guidance(error_message, keyword_name)


@mcp.tool
async def get_browser_locator_guidance(
    error_message: str = None, keyword_name: str = None
) -> Dict[str, Any]:
    """Get comprehensive Browser Library (Playwright) locator strategy guidance for AI agents.

    This tool helps AI agents understand Browser Library's selector strategies and
    provides context-aware suggestions for element location and error resolution.

    Browser Library uses Playwright's locator strategies with these key features:

    **Selector Strategies:**
    - css: CSS selector (default) - e.g., '.button' or 'css=.button'
    - xpath: XPath expression - e.g., '//button' or 'xpath=//button'
    - text: Text content matching - e.g., '"Login"' or 'text=Login'
    - id: Element ID - e.g., 'id=submit-btn'
    - data-testid: Test ID attribute - e.g., 'data-testid=login-button'

    **Advanced Features:**
    - Cascaded selectors: 'text=Hello >> ../.. >> .select_button'
    - iFrame piercing: 'id=myframe >>> .inner-button'
    - Shadow DOM: Automatic piercing with CSS and text engines
    - Strict mode: Controls behavior with multiple element matches
    - Element references: '${ref} >> .child' for chained operations

    **Implicit Detection Rules:**
    - Plain selectors → CSS (default): '.button' becomes 'css=.button'
    - Starting with // or .. → XPath: '//button' becomes 'xpath=//button'
    - Quoted text → Text selector: '"Login"' becomes 'text=Login'
    - Explicit format: 'strategy=value' for any strategy

    Args:
        error_message: Optional error message to analyze for specific guidance
        keyword_name: Optional keyword name that failed for context-specific tips

    Returns:
        Comprehensive Browser Library locator guidance with examples, patterns, and error-specific advice
    """
    from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter

    converter = RobotFrameworkNativeConverter()
    return converter.get_browser_locator_guidance(error_message, keyword_name)


@mcp.tool
async def get_loaded_libraries() -> Dict[str, Any]:
    """Get status of all loaded Robot Framework libraries using both libdoc and inspection methods.

    Returns comprehensive library status including:
    - Native Robot Framework libdoc information (when available)
    - Inspection-based discovery fallback
    - Preferred data source (libdoc vs inspection)
    - Library versions, scopes, types, and keyword counts

    Returns:
        Dict with detailed library information:
        - preferred_source: 'libdoc' or 'inspection'
        - libdoc_based: Native RF libdoc library information (if available)
        - inspection_based: Inspection-based library discovery information
    """
    return execution_engine.get_library_status()


@mcp.tool
async def run_test_suite_dry(
    session_id: str = "",
    suite_file_path: str = None,
    validation_level: str = "standard",
    include_warnings: bool = True
) -> Dict[str, Any]:
    """Validate test suite using Robot Framework dry run mode.
    
    RECOMMENDED WORKFLOW - SUITE VALIDATION:
    This tool should be used AFTER build_test_suite to validate the generated suite:
    1. ✅ build_test_suite - Generate .robot file from session steps
    2. ✅ run_test_suite_dry (THIS TOOL) - Validate syntax and structure 
    3. ➡️ run_test_suite - Execute if validation passes
    
    Enhanced Session Resolution:
    - If session_id provided and valid: Uses that session's generated suite
    - If session_id empty/invalid: Automatically finds most suitable session
    - If suite_file_path provided: Validates specified file directly
    
    Validation Levels:
    - minimal: Basic syntax checking only
    - standard: Syntax + keyword verification + imports (default)
    - strict: All checks + argument validation + structure analysis
    
    Args:
        session_id: Session with executed steps (auto-resolves if empty/invalid)
        suite_file_path: Direct path to .robot file (optional, overrides session)
        validation_level: Validation depth ('minimal', 'standard', 'strict')
        include_warnings: Include warnings in validation report
        
    Returns:
        Structured validation results with issues, warnings, and suggestions
    """
    
    # Session resolution with same logic as build_test_suite
    from robotmcp.utils.session_resolution import SessionResolver
    session_resolver = SessionResolver(execution_engine.session_manager)
    
    if suite_file_path:
        # Direct file validation mode
        logger.info(f"Running dry run validation on file: {suite_file_path}")
        return await execution_engine.run_suite_dry_run_from_file(
            suite_file_path, validation_level, include_warnings
        )
    else:
        # Session-based validation mode  
        resolution_result = session_resolver.resolve_session_with_fallback(session_id)
        
        if not resolution_result["success"]:
            return {
                "success": False,
                "tool": "run_test_suite_dry",
                "error": "No valid session or suite file for validation",
                "error_details": resolution_result["error_guidance"],
                "guidance": [
                    "Create a session and execute some steps first",
                    "Use build_test_suite to generate a test suite",
                    "Or provide suite_file_path to validate an existing file"
                ],
                "recommendation": "Use build_test_suite first or provide suite_file_path"
            }
        
        resolved_session_id = resolution_result["session_id"]
        logger.info(f"Running dry run validation for session: {resolved_session_id}")
        
        result = await execution_engine.run_suite_dry_run(
            resolved_session_id, validation_level, include_warnings
        )
        
        # Add session resolution info to result
        if resolution_result.get("fallback_used", False):
            result["session_resolution"] = {
                "fallback_used": True,
                "original_session_id": session_id,
                "resolved_session_id": resolved_session_id,
                "message": f"Automatically used session '{resolved_session_id}' with {resolution_result['session_info']['step_count']} executed steps"
            }
        else:
            result["session_resolution"] = {
                "fallback_used": False,
                "session_id": resolved_session_id
            }
        
        return result


@mcp.tool  
async def run_test_suite(
    session_id: str = "",
    suite_file_path: str = None,
    execution_options: Dict[str, Any] = None,
    output_level: str = "standard",
    capture_screenshots: bool = False
) -> Dict[str, Any]:
    """Execute test suite using Robot Framework normal execution.
    
    RECOMMENDED WORKFLOW - SUITE EXECUTION:
    This tool should be used AFTER validation for full test execution:
    1. ✅ build_test_suite - Generate .robot file from session steps
    2. ✅ run_test_suite_dry - Validate syntax and structure
    3. ✅ run_test_suite (THIS TOOL) - Execute validated test suite
    
    Enhanced Session Resolution:
    - If session_id provided and valid: Uses that session's generated suite
    - If session_id empty/invalid: Automatically finds most suitable session  
    - If suite_file_path provided: Executes specified file directly
    
    Output Levels:
    - minimal: Basic execution statistics only
    - standard: Statistics + failed tests + output files (default)
    - detailed: All information + execution details + timing
    
    Args:
        session_id: Session with executed steps (auto-resolves if empty/invalid)
        suite_file_path: Direct path to .robot file (optional, overrides session)
        execution_options: Dict with RF options (variables, tags, loglevel, etc.)
        output_level: Response verbosity ('minimal', 'standard', 'detailed')
        capture_screenshots: Enable screenshot capture on failures
        
    Returns:
        Comprehensive execution results with statistics and output files
    """
    
    if execution_options is None:
        execution_options = {}
    
    # Session resolution with same logic as build_test_suite
    from robotmcp.utils.session_resolution import SessionResolver
    session_resolver = SessionResolver(execution_engine.session_manager)
    
    if suite_file_path:
        # Direct file execution mode
        logger.info(f"Running suite execution on file: {suite_file_path}")
        return await execution_engine.run_suite_execution_from_file(
            suite_file_path, execution_options, output_level, capture_screenshots
        )
    else:
        # Session-based execution mode
        resolution_result = session_resolver.resolve_session_with_fallback(session_id)
        
        if not resolution_result["success"]:
            return {
                "success": False,
                "tool": "run_test_suite",
                "error": "No valid session or suite file for execution", 
                "error_details": resolution_result["error_guidance"],
                "guidance": [
                    "Create a session and execute some steps first",
                    "Use build_test_suite to generate a test suite",
                    "Or provide suite_file_path to execute an existing file"
                ],
                "recommendation": "Use build_test_suite first or provide suite_file_path"
            }
        
        resolved_session_id = resolution_result["session_id"]
        logger.info(f"Running suite execution for session: {resolved_session_id}")
        
        result = await execution_engine.run_suite_execution(
            resolved_session_id, execution_options, output_level, capture_screenshots
        )
        
        # Add session resolution info to result
        if resolution_result.get("fallback_used", False):
            result["session_resolution"] = {
                "fallback_used": True,
                "original_session_id": session_id,
                "resolved_session_id": resolved_session_id,
                "message": f"Automatically used session '{resolved_session_id}' with {resolution_result['session_info']['step_count']} executed steps"
            }
        else:
            result["session_resolution"] = {
                "fallback_used": False,
                "session_id": resolved_session_id
            }
        
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()

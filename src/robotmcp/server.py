"""Main MCP Server implementation for Robot Framework integration."""

import logging
from typing import Any, Dict, List, Union

from fastmcp import FastMCP

from robotmcp.components.execution import ExecutionCoordinator
from robotmcp.components.execution.mobile_capability_service import (
    MobileCapabilityService,
)
from robotmcp.components.execution.rf_native_context_manager import (
    get_rf_native_context_manager,
)
from robotmcp.components.keyword_matcher import KeywordMatcher
from robotmcp.components.library_recommender import LibraryRecommender
from robotmcp.components.nlp_processor import NaturalLanguageProcessor
from robotmcp.components.state_manager import StateManager
from robotmcp.components.test_builder import TestBuilder
from robotmcp.models.session_models import PlatformType
from robotmcp.utils.server_integration import initialize_enhanced_serialization

logger = logging.getLogger(__name__)


# Initialize FastMCP server
mcp = FastMCP("Robot Framework MCP Server")


# Internal helpers to build prompt texts (used by both @mcp.prompt and wrapper tools)
def _build_recommend_libraries_sampling_prompt(
    scenario: str,
    k: int = 4,
    available_libraries: List[Dict[str, Any]] = None,
) -> str:
    try:
        import json

        libs_section = (
            json.dumps(available_libraries, ensure_ascii=False, indent=2)
            if available_libraries
            else "[]"
        )
    except Exception:
        libs_section = "[]"

    return (
        "# Task\n"
        "You are 1 of {k} samplers. Recommend the best Robot Framework libraries for this scenario.\n"
        "- Consider ONLY the libraries listed below as available in this environment.\n"
        "- Resolve conflicts (e.g., prefer one of Browser/SeleniumLibrary).\n"
        "- Output strictly the JSON schema in the Output Format section.\n\n"
        "# Scenario\n"
        f"{scenario}\n\n"
        "# Available Libraries (from environment)\n"
        f"{libs_section}\n\n"
        "# Guidance\n"
        "- Choose 2–5 libraries maximum.\n"
        "- Justify each choice concisely, referencing capabilities from 'available_libraries'.\n"
        "- If multiple web libs exist, pick one with a short rationale.\n"
        "- For API use, mention RequestsLibrary and how sessions are created.\n"
        "- For XML/data flows, consider XML/Collections/String.\n"
        "- If specialized libs are not needed, do not recommend them.\n\n"
        "# Output Format (JSON)\n"
        "{\n"
        '  "recommendations": [\n'
        '    { "name": "<LibraryName>", "reason": "<1-2 lines>", "score": 0.0 },\n'
        "    ... up to 5 total ...\n"
        "  ],\n"
        '  "conflicts": [\n'
        '    { "conflict_set": ["Browser", "SeleniumLibrary"], "chosen": "Browser", "reason": "<1 line>" }\n'
        "  ]\n"
        "}\n"
    )


def _build_choose_recommendations_prompt(
    candidates: List[Dict[str, Any]] = None,
) -> str:
    import json

    cand_section = (
        json.dumps(candidates, ensure_ascii=False, indent=2) if candidates else "[]"
    )

    return (
        "# Task\n"
        "Select or merge the following sampled recommendations into a final JSON.\n"
        "- Deduplicate libraries by name.\n"
        "- Resolve conflicts (e.g., Browser vs SeleniumLibrary) by choosing the higher total score; state a 1-line reason.\n"
        "- Normalize scores to 0..1, and keep at most 5 libraries.\n"
        "- Output strictly the JSON under 'Output Format'.\n\n"
        "# Candidates (JSON)\n"
        f"{cand_section}\n\n"
        "# Output Format (JSON)\n"
        "{\n"
        '  "recommendations": [\n'
        '    { "name": "<LibraryName>", "reason": "<1-2 lines>", "score": 0.0 }\n'
        "  ],\n"
        '  "conflicts": [\n'
        '    { "conflict_set": ["Browser", "SeleniumLibrary"], "chosen": "<name>", "reason": "<1 line>" }\n'
        "  ]\n"
        "}\n"
    )


# Initialize components
nlp_processor = NaturalLanguageProcessor()
keyword_matcher = KeywordMatcher()
library_recommender = LibraryRecommender()
execution_engine = ExecutionCoordinator()
state_manager = StateManager()
test_builder = TestBuilder(execution_engine)
mobile_capability_service = MobileCapabilityService()

# Initialize enhanced serialization system
initialize_enhanced_serialization(execution_engine)


# Helper functions
async def _ensure_all_session_libraries_loaded():
    """
    Ensure all imported session libraries are loaded in LibraryManager.

    Enhanced validation to prevent keyword filtering issues and provide better error reporting.
    """
    try:
        session_manager = execution_engine.session_manager
        all_sessions = session_manager.sessions.values()

        for session in all_sessions:
            for library_name in session.imported_libraries:
                # Check if library is loaded in the orchestrator
                if library_name not in execution_engine.keyword_discovery.libraries:
                    logger.warning(
                        f"Session library '{library_name}' not loaded in orchestrator, attempting to load"
                    )
                    session._ensure_library_loaded_immediately(library_name)

                    # Verify loading succeeded
                    if library_name not in execution_engine.keyword_discovery.libraries:
                        logger.error(
                            f"Failed to load session library '{library_name}' - may cause keyword filtering issues"
                        )
                else:
                    logger.debug(
                        f"Session library '{library_name}' already loaded in orchestrator"
                    )

        logger.debug(
            "Validated all session libraries are loaded for discovery operations"
        )

    except Exception as e:
        logger.error(f"Error ensuring session libraries loaded: {e}")
        # Don't fail the discovery operation, but log the issue for debugging


@mcp.prompt
def automate(scenario: str) -> str:
    """Uses RobotMCP to create a test suite from a scenario description"""
    return (
        "# Task\n"
        "Use RobotMCP to create a TestSuite and execute it step wise.\n"
        "1. Use analyze_scenario to understand the requirements and create a session.\n"
        "2. Use recommend_libraries to get library suggestions based on the scenario.\n"
        "3. Use execute_step to run the test steps in the created session.\n"
        "4. Use build_test_suite to compile the test steps into a complete suite.\n"
        "5. Use run_test_suite_dry to execute a staged dry run of the test suite.\n"
        "6. Use run_test_suite to execute the test suite with all libraries loaded.\n"
        "General hints:\n"
        "- in case of UI testing, use get_page_source to retrieve the current state of the UI.\n"
        "- in case of UI testing, ensure the Browser is running in non-headless mode.\n"
        "- in case of problems with keyword calls, use get_keyword_documentation and get_library_documentation to get more information.\n"
        "# Scenario:\n"
        f"{scenario}\n"
    )


# Note: Prompt endpoints removed per Option B. Use tools below that return plain prompt text.


@mcp.tool(
    name="recommend_libraries_sampling_tool",
    description="STEP 2 (planning): Recommend the best libraries for a scenario using sampling; returns prompt text and suggested sampling config.",
)
async def recommend_libraries_sampling_tool(
    scenario: str,
    k: int = 4,
    available_libraries: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Recommend the best Robot Framework libraries for a scenario (sampling-enabled).

    Purpose:
    - Produce a prioritized list of libraries with reasons and scores, chosen strictly
      from the provided environment and tailored to the scenario. Use sampling to
      generate diverse candidates, then select/merge with the chooser tool.

    How to use:
    - Call this tool to get the plain prompt text and a suggested sampling config.
    - Use your model to generate k sampled recommendations from that prompt.
    - Then call `choose_recommendations_tool` to merge/score into a final set.
    - Follow with `check_library_availability`, `set_library_search_order`, then `execute_step`.

    Arguments:
    - scenario: Plain-language scenario to analyze.
    - k: Number of samples to generate (e.g., 3–5).
    - available_libraries: Array of objects with: name, description, categories,
      requires_setup, setup_commands, use_cases, conflicts (optional).

    Returns:
    - success: True
    - prompt: Prompt text for your model to sample against
    - recommended_sampling: Suggested sampling config, e.g., {count: k, temperature: 0.4}
    """
    prompt_text = _build_recommend_libraries_sampling_prompt(
        scenario, k, available_libraries
    )
    return {
        "success": True,
        "prompt": prompt_text,
        "recommended_sampling": {"count": k, "temperature": 0.4},
    }


@mcp.tool(
    name="choose_recommendations_tool",
    description="STEP 2b (selection): Merge/score sampled recommendation payloads into a final prioritized set before availability checks.",
)
async def choose_recommendations_tool(
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Choose/merge sampled recommender outputs into a final recommendation set.

    Purpose:
    - Turn multiple sampled recommendation payloads into a single, deduplicated,
      conflict‑resolved, prioritized list with normalized scores.

    How to use:
    - Call this tool to get the chooser prompt text, then use your model to produce
      the final set from the provided candidates. Proceed to `check_library_availability`
      and `set_library_search_order`.

    Arguments:
    - candidates: List of sampled recommendation payloads (objects with a `recommendations`
      array of {name, reason, score} and optional `conflicts`).

    Returns:
    - success: True
    - prompt: Chooser prompt text for your model
    """
    prompt_text = _build_choose_recommendations_prompt(candidates)
    return {
        "success": True,
        "prompt": prompt_text,
    }


@mcp.tool(
    name="list_available_libraries_for_prompt",
    description="Emit the available_libraries payload shaped for the sampling recommender (names, descriptions, categories, setup flags).",
)
async def list_available_libraries_for_prompt() -> Dict[str, Any]:
    """List available libraries formatted for recommend_libraries_sampling.

    Produces an array of library objects containing fields referenced by the
    sampling prompt: name, description, categories, requires_setup, setup_commands,
    use_cases, conflicts (optional), platform_requirements, dependencies, is_builtin.

    Returns:
    - success: True
    - available_libraries: Array of library objects suitable for the sampling prompt
    """
    from robotmcp.config.library_registry import get_recommendation_info

    libs = get_recommendation_info()
    # Add an explicit empty conflicts field for consistency in prompts
    for lib in libs:
        lib.setdefault("conflicts", [])
    return {"success": True, "available_libraries": libs}


@mcp.tool
async def recommend_libraries(
    scenario: str,
    context: str = "web",
    max_recommendations: int = 5,
    session_id: str = None,
    check_availability: bool = True,
    apply_search_order: bool = True,
) -> Dict[str, Any]:
    """Recommend Robot Framework libraries for a scenario and optionally apply them.

    Returns a prioritized list of recommended Robot Framework libraries based on
    the scenario description and context. Can also check availability and, when a
    session_id is provided, apply the recommended search order to that session.

    Args:
        scenario: Scenario description.
        context: Testing context (web, mobile, api, data, etc.).
        max_recommendations: Max number of libraries to return.
        session_id: If provided, apply the search order to this session.
        check_availability: When True, includes availability and install suggestions.
        apply_search_order: When True and session_id provided, sets session search order.

    Returns:
        - success: True on success
        - recommended_libraries: Ordered list of library names (primary output)
        - recommendations: Detailed entries (name, rationale, confidence, etc.)
        - availability: When requested, available/missing lists and installation suggestions
        - session_setup: When applied, session_id and resulting search order
    """
    # 1) Compute recommendations deterministically
    rec = library_recommender.recommend_libraries(
        scenario, context=context, max_recommendations=max_recommendations
    )
    if not rec.get("success"):
        return {"success": False, "error": rec.get("error", "Recommendation failed")}

    recommendations = rec.get("recommendations", [])
    recommended_names = [
        r.get("library_name") for r in recommendations if r.get("library_name")
    ]

    result: Dict[str, Any] = {
        "success": True,
        "scenario": scenario,
        "context": context,
        "recommended_libraries": recommended_names,
        "recommendations": recommendations,
    }

    # 2) Optionally check availability
    availability_info = None
    if check_availability and recommended_names:
        availability_info = execution_engine.check_library_requirements(
            recommended_names
        )
        result["availability"] = availability_info

    # 3) Optionally apply search order to session
    if session_id and apply_search_order and recommended_names:
        session = execution_engine.session_manager.get_or_create_session(session_id)
        old_order = session.get_search_order()

        # Respect explicit library preference detected during analyze_scenario
        explicit = getattr(session, "explicit_library_preference", None)
        if explicit:
            # Put explicit preference first
            recommended_names = [explicit] + [
                n for n in recommended_names if n != explicit
            ]
            # Resolve web lib conflicts by removing the opposite when explicit is set
            if explicit == "SeleniumLibrary" and "Browser" in recommended_names:
                recommended_names = [n for n in recommended_names if n != "Browser"]
            if explicit == "Browser" and "SeleniumLibrary" in recommended_names:
                recommended_names = [
                    n for n in recommended_names if n != "SeleniumLibrary"
                ]

            # Also align the detailed recommendations list with the adjusted names/order
            name_to_rec = {r.get("library_name"): r for r in recommendations}
            recommendations = [
                name_to_rec[n] for n in recommended_names if n in name_to_rec
            ]
            result["recommendations"] = recommendations
            result["recommended_libraries"] = recommended_names

        # Ensure recommended libraries are imported/loaded into the session so they can be applied to search order
        for lib in recommended_names:
            try:
                # Force switch if conflicting web libraries
                session.import_library(lib, force=True)
            except Exception as e:
                logger.debug(f"Could not import {lib} into session {session_id}: {e}")

        # Prefer available libraries first if we have that info, else use all recommendations
        preferred = (
            availability_info.get("available_libraries", [])
            if availability_info
            else recommended_names
        )
        # If explicit preference exists, ensure it leads and resolve web conflicts in preferred list as well
        if explicit:
            # Put explicit first
            preferred = [explicit] + [n for n in preferred if n != explicit]
            # Resolve Browser/Selenium conflict
            if explicit == "SeleniumLibrary":
                preferred = [n for n in preferred if n != "Browser"]
            if explicit == "Browser":
                preferred = [n for n in preferred if n != "SeleniumLibrary"]
        # Merge with existing order while preserving priority
        new_order = list(
            dict.fromkeys(
                preferred + [lib for lib in old_order if lib not in preferred]
            )
        )
        session.set_library_search_order(new_order)

        result["session_setup"] = {
            "session_id": session_id,
            "old_search_order": old_order,
            "new_search_order": new_order,
            "applied": True,
        }

    return result


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

    # Detect platform type from scenario
    platform_type = execution_engine.session_manager.detect_platform_from_scenario(
        scenario
    )

    # Initialize mobile session if detected
    if platform_type == PlatformType.MOBILE:
        execution_engine.session_manager.initialize_mobile_session(session, scenario)
        logger.info(
            f"Initialized mobile session for platform: {session.mobile_config.platform_name if session.mobile_config else 'Unknown'}"
        )
    else:
        # Auto-configure session based on scenario (existing web flow)
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
        "ready_for_execution": True,
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
    use_context: bool = False,
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
        use_context: If True, executes within full RF context (maintains variables, state across calls).
                    This enables proper variable scoping, built-in keyword functionality, and
                    library state persistence.
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
        use_context=use_context,
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


@mcp.tool(enabled=False)
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
                "Check session status with get_session_validation_status",
            ],
            "validation_summary": {"passed": 0, "failed": 0},
            "recommendation": "Start with analyze_scenario() to create a properly configured session",
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
            "message": f"Automatically used session '{resolved_session_id}' with {resolution_result['session_info']['step_count']} executed steps",
        }
    else:
        result["session_resolution"] = {
            "fallback_used": False,
            "session_id": resolved_session_id,
        }

    return result


@mcp.tool(enabled=False)
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


# Note: Removed legacy disabled recommend_libraries_ tool to avoid confusion.


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


@mcp.tool(enabled=False)
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
    """List available RF keywords with minimal metadata.

    Returns one entry per keyword with fields:
    - name: keyword name
    - library: library name
    - args: list of argument names
    - arg_types: list of argument types if available (empty when unknown)
    - short_doc: short documentation summary (no full docstrings)

    If `library_name` is provided, results are filtered to that library, loading it on demand if needed.
    """
    # CRITICAL FIX: Ensure all session libraries are loaded before discovery
    await _ensure_all_session_libraries_loaded()

    return execution_engine.get_available_keywords(library_name)


@mcp.tool
async def search_keywords(pattern: str) -> List[Dict[str, Any]]:
    """Search for Robot Framework keywords matching a pattern using native RF libdoc.

    Uses Robot Framework's native libdoc API for accurate search results and documentation.
    Searches through keyword names, documentation, short_doc, and tags.

    CRITICAL FIX: Now ensures all session libraries are loaded before search.

    Args:
        pattern: Search pattern to match against keyword names, documentation, or tags

    Returns:
        List of matching keywords with native RF libdoc metadata including short_doc,
        argument types, deprecation status, and enhanced tag information.
    """
    # CRITICAL FIX: Ensure all session libraries are loaded before search
    await _ensure_all_session_libraries_loaded()

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


@mcp.tool
async def get_library_documentation(library_name: str) -> Dict[str, Any]:
    """Get full documentation for a Robot Framework library using native RF libdoc.

    Uses Robot Framework's native LibraryDocumentation API to provide comprehensive
    library information including library metadata and all keywords with their
    documentation, arguments, and metadata.

    Args:
        library_name: Name of the library to get documentation for

    Returns:
        Dict containing comprehensive library information:
        - success: Boolean indicating if library was found
        - library: Dict with library details including:
          - name: Library name
          - doc: Library documentation
          - version: Library version
          - type: Library type
          - scope: Library scope
          - source: Source file path
          - keywords: List of all library keywords with full details including:
            - name: Keyword name
            - args: List of argument names
            - arg_types: List of argument types (when available from libdoc)
            - doc: Full keyword documentation text
            - short_doc: Native Robot Framework short_doc
            - tags: Keyword tags
            - is_deprecated: Deprecation status (libdoc only)
            - source: Source file path (libdoc only)
            - lineno: Line number in source (libdoc only)
          - keyword_count: Total number of keywords in library
          - data_source: 'libdoc' or 'inspection' indicating data source
    """
    return execution_engine.get_library_documentation(library_name)


@mcp.tool(enabled=True)
async def debug_parse_keyword_arguments(
    keyword_name: str,
    arguments: List[str],
    library_name: str = None,
    session_id: str = None,
) -> Dict[str, Any]:
    """Debug helper: Parse arguments into positional and named using RF-native logic.

    Uses the same parsing path as execution to verify how name=value pairs are handled
    for a given keyword and (optionally) library.

    Args:
        keyword_name: Keyword to parse for (e.g., 'Open Application').
        arguments: List of argument strings as they would be passed to execute_step.
        library_name: Optional library name to disambiguate (e.g., 'AppiumLibrary').
        session_id: Optional session to pull variables from for resolution.

    Returns:
        - success: True
        - parsed: { positional: [...], named: {k: v} }
        - notes: brief info on library and session impact
    """
    try:
        session_vars = {}
        if session_id:
            sess = execution_engine.get_session(session_id)
            if sess:
                session_vars = sess.variables

        parsed = execution_engine.keyword_executor.argument_processor.parse_arguments_for_keyword(
            keyword_name, arguments, library_name, session_vars
        )
        return {
            "success": True,
            "parsed": {"positional": parsed.positional, "named": parsed.named},
            "notes": {
                "library_name": library_name,
                "session_id": session_id,
                "positional_count": len(parsed.positional),
                "named_count": len(parsed.named or {}),
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


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
            "available_sessions": resolution_result["error_guidance"][
                "available_sessions"
            ],
            "sessions_with_steps": resolution_result["error_guidance"][
                "sessions_with_steps"
            ],
            "recommendation": "Use analyze_scenario() to create a session first",
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
            "message": f"Automatically checked session '{resolved_session_id}'",
        }
    else:
        result["session_resolution"] = {
            "fallback_used": False,
            "session_id": resolved_session_id,
        }

    return result


@mcp.tool(enabled=False)
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
async def initialize_context(
    session_id: str, libraries: List[str] = None, variables: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Initialize a session with libraries and variables.

    NOTE: Full RF context mode is not yet implemented. This tool currently
    initializes a session with the specified libraries and variables using
    the existing session-based variable system.

    Args:
        session_id: Session identifier
        libraries: List of libraries to import in the session
        variables: Initial variables to set in the session

    Returns:
        Session initialization status with information
    """
    try:
        # Get or create session
        session = execution_engine.session_manager.get_or_create_session(session_id)

        # Import libraries into session
        if libraries:
            for library in libraries:
                try:
                    session.import_library(library)
                    # Also add to loaded_libraries for tracking
                    session.loaded_libraries.add(library)
                    logger.info(f"Imported {library} into session {session_id}")
                except Exception as lib_error:
                    logger.warning(f"Could not import {library}: {lib_error}")

        # Set initial variables in session
        if variables:
            for name, value in variables.items():
                # Normalize variable name to RF format
                if not name.startswith("$"):
                    var_name = f"${{{name}}}"
                else:
                    var_name = name
                session.set_variable(var_name, value)
                logger.info(
                    f"Set variable {var_name} = {value} in session {session_id}"
                )

        return {
            "success": True,
            "session_id": session_id,
            "context_enabled": False,  # Context mode not fully implemented
            "libraries_loaded": list(session.loaded_libraries),
            "variables_set": list(variables.keys()) if variables else [],
            "message": f"Session '{session_id}' initialized with libraries and variables",
            "note": "Using session-based variable system (context mode not available)",
        }

    except Exception as e:
        logger.error(f"Error initializing session {session_id}: {e}")
        return {"success": False, "error": str(e), "session_id": session_id}


@mcp.tool
async def get_context_variables(session_id: str) -> Dict[str, Any]:
    """Get all variables from a session.

    Args:
        session_id: Session identifier

    Returns:
        Dictionary containing all session variables
    """
    try:
        # Helper to sanitize values: return scalars as-is; for complex objects, return their type name.
        def _sanitize(val: Any) -> Any:
            if isinstance(val, (str, int, float, bool)) or val is None:
                return val
            # Avoid serializing complex/large objects
            return f"<{type(val).__name__}>"

        # Prefer RF Namespace/Variables if an RF context exists for the session
        try:
            from robotmcp.components.execution.rf_native_context_manager import (
                get_rf_native_context_manager,
            )

            mgr = get_rf_native_context_manager()
            ctx_info = mgr.get_session_context_info(session_id)
            if ctx_info.get("context_exists"):
                # Extract variables from RF Variables object
                ctx = mgr._session_contexts.get(session_id)  # internal read-only access
                rf_vars_obj = ctx.get("variables") if ctx else None
                rf_vars: Dict[str, Any] = {}
                if rf_vars_obj is not None:
                    try:
                        if hasattr(rf_vars_obj, "store"):
                            rf_vars = dict(rf_vars_obj.store.data)
                        elif hasattr(rf_vars_obj, "current") and hasattr(
                            rf_vars_obj.current, "store"
                        ):
                            rf_vars = dict(rf_vars_obj.current.store.data)
                    except Exception:
                        rf_vars = {}

                # Attempt to resolve variable resolvers to concrete values via Variables API
                resolved: Dict[str, Any] = {}
                for k, v in rf_vars.items():
                    key = k if isinstance(k, str) else str(k)
                    try:
                        norm = key if key.startswith("${") else f"${{{key}}}"
                        concrete = rf_vars_obj[norm]
                    except Exception:
                        concrete = v
                    resolved[key if not key.startswith("${") else key.strip("${}")] = (
                        concrete
                    )

                sanitized = {str(k): _sanitize(v) for k, v in resolved.items()}
                return {
                    "success": True,
                    "session_id": session_id,
                    "variables": sanitized,
                    "variable_count": len(sanitized),
                    "source": "rf_context",
                }
        except Exception:
            # Fall back to session store below
            pass

        # Fallback: session-based variable store
        session = execution_engine.session_manager.get_session(session_id)
        if not session:
            return {
                "success": False,
                "error": f"Session '{session_id}' not found",
                "session_id": session_id,
            }
        sess_vars_raw = dict(session.variables)
        sess_vars = {str(k): _sanitize(v) for k, v in sess_vars_raw.items()}
        return {
            "success": True,
            "session_id": session_id,
            "variables": sess_vars,
            "variable_count": len(sess_vars),
            "source": "session_store",
        }

    except Exception as e:
        logger.error(f"Error getting variables for session {session_id}: {e}")
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
async def get_appium_locator_guidance(
    error_message: str = None, keyword_name: str = None
) -> Dict[str, Any]:
    """Get comprehensive AppiumLibrary locator strategy guidance for AI agents.

    This tool helps AI agents understand AppiumLibrary's locator strategies and
    provides context-aware suggestions for mobile element location and error resolution.

    AppiumLibrary supports these locator strategies:

    **Basic Locators:**
    - id: Element ID (e.g., 'id=my_element' or just 'my_element')
    - xpath: XPath expression (e.g., '//*[@type="android.widget.EditText"]')
    - identifier: Matches by @id attribute (e.g., 'identifier=my_element')
    - accessibility_id: Accessibility options utilize (e.g., 'accessibility_id=button3')
    - class: Matches by class (e.g., 'class=UIAPickerWheel')
    - name: Matches by @name attribute (e.g., 'name=my_element') - Only valid for Selendroid

    **Platform-Specific Locators:**
    - android: Android UI Automator (e.g., 'android=UiSelector().description("Apps")')
    - ios: iOS UI Automation (e.g., 'ios=.buttons().withName("Apps")')
    - predicate: iOS Predicate (e.g., 'predicate=name=="login"')
    - chain: iOS Class Chain (e.g., 'chain=XCUIElementTypeWindow[1]/*')

    **WebView Locators:**
    - css: CSS selector in webview (e.g., 'css=.green_button')

    **Default Behavior:**
    - By default, locators match against key attributes (id for all elements)
    - Plain text (e.g., 'my_element') is treated as ID lookup
    - XPath should start with // or use explicit 'xpath=' prefix

    **WebElement Support:**
    Starting with AppiumLibrary v1.4, you can pass WebElement objects:
    - Get elements with: Get WebElements or Get WebElement
    - Use directly: Click Element ${element}

    Args:
        error_message: Optional error message to analyze for specific guidance
        keyword_name: Optional keyword name that failed for context-specific tips

    Returns:
        Comprehensive locator strategy guidance with examples, tips, and error-specific advice
    """
    from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter

    converter = RobotFrameworkNativeConverter()
    return converter.get_appium_locator_guidance(error_message, keyword_name)


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
    include_warnings: bool = True,
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
                    "Or provide suite_file_path to validate an existing file",
                ],
                "recommendation": "Use build_test_suite first or provide suite_file_path",
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
                "message": f"Automatically used session '{resolved_session_id}' with {resolution_result['session_info']['step_count']} executed steps",
            }
        else:
            result["session_resolution"] = {
                "fallback_used": False,
                "session_id": resolved_session_id,
            }

        return result


@mcp.tool
async def run_test_suite(
    session_id: str = "",
    suite_file_path: str = None,
    execution_options: Dict[str, Any] = None,
    output_level: str = "standard",
    capture_screenshots: bool = False,
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
                    "Or provide suite_file_path to execute an existing file",
                ],
                "recommendation": "Use build_test_suite first or provide suite_file_path",
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
                "message": f"Automatically used session '{resolved_session_id}' with {resolution_result['session_info']['step_count']} executed steps",
            }
        else:
            result["session_resolution"] = {
                "fallback_used": False,
                "session_id": resolved_session_id,
            }

        return result


@mcp.tool(
    name="diagnose_rf_context",
    description="Inspect RF context state for a session: libraries, search order, and variables count.",
)
async def diagnose_rf_context(session_id: str) -> Dict[str, Any]:
    """Return diagnostic information about the current RF execution context for a session.

    Includes: whether context exists, created_at, imported libraries, variables count,
    and where possible, the current RF library search order.
    """
    try:
        mgr = get_rf_native_context_manager()
        info = mgr.get_session_context_info(session_id)
        # Try to enrich with Namespace search order and imported libraries
        if info.get("context_exists"):
            ctx = mgr._session_contexts.get(session_id)  # internal, read-only
            extra = {}
            try:
                namespace = ctx.get("namespace")
                # Namespace has no direct getter for search order; infer from libraries list
                lib_names = []
                if hasattr(namespace, "libraries"):
                    libs = namespace.libraries
                    if hasattr(libs, "keys"):
                        lib_names = list(libs.keys())
                extra["namespace_libraries"] = lib_names
            except Exception:
                pass
            info["extra"] = extra
        return info
    except Exception as e:
        logger.error(f"diagnose_rf_context failed: {e}")
        return {"context_exists": False, "error": str(e), "session_id": session_id}


@mcp.tool(
    name="import_resource",
    description="Import a Robot Framework resource file into the session RF Namespace.",
)
async def import_resource(session_id: str, path: str) -> Dict[str, Any]:
    mgr = get_rf_native_context_manager()
    return mgr.import_resource_for_session(session_id, path)


@mcp.tool(
    name="import_custom_library",
    description="Import a custom Robot Framework library (module name or file path) into the session RF Namespace.",
)
async def import_custom_library(
    session_id: str,
    name_or_path: str,
    args: List[str] | None = None,
    alias: str | None = None,
) -> Dict[str, Any]:
    mgr = get_rf_native_context_manager()
    return mgr.import_library_for_session(
        session_id, name_or_path, tuple(args or ()), alias
    )


@mcp.tool(
    name="list_available_keywords",
    description="List available keywords from imported libraries and resources in the session RF Namespace.",
)
async def list_available_keywords(session_id: str) -> Dict[str, Any]:
    mgr = get_rf_native_context_manager()
    return mgr.list_available_keywords(session_id)


@mcp.tool(
    name="get_session_keyword_documentation",
    description="Get documentation for a keyword (library or resource) available in the session RF Namespace.",
)
async def get_session_keyword_documentation(
    session_id: str, keyword_name: str
) -> Dict[str, Any]:
    mgr = get_rf_native_context_manager()
    return mgr.get_keyword_documentation(session_id, keyword_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()

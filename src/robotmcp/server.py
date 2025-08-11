"""Main MCP Server implementation for Robot Framework integration."""

import asyncio
import json
import logging
from typing import Any, Dict, List

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
async def analyze_scenario(scenario: str, context: str = "web") -> Dict[str, Any]:
    """Process natural language test description into structured test intent.
    
    Args:
        scenario: Human language scenario description
        context: Optional context about the application (web, mobile, API, etc.)
    """
    return await nlp_processor.analyze_scenario(scenario, context)
@mcp.tool
async def discover_keywords(
    action_description: str, 
    context: str = "web", 
    current_state: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Find matching Robot Framework keywords for an action.
    
    Args:
        action_description: Description of the action to perform
        context: Current context (web, mobile, API, etc.)
        current_state: Current application state
    """
    if current_state is None:
        current_state = {}
    return await keyword_matcher.discover_keywords(action_description, context, current_state)
@mcp.tool
async def execute_step(
    keyword: str, 
    arguments: List[str] = None, 
    session_id: str = "default",
    raise_on_failure: bool = True,
    detail_level: str = "minimal"
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
    """
    if arguments is None:
        arguments = []
    
    result = await execution_engine.execute_step(keyword, arguments, session_id, detail_level)
    
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
    session_id: str = "default"
) -> Dict[str, Any]:
    """Retrieve current application state.
    
    Args:
        state_type: Type of state to retrieve (dom, api, database, all)
        elements_of_interest: Specific elements to focus on
        session_id: Session identifier
    """
    if elements_of_interest is None:
        elements_of_interest = []
    return await state_manager.get_state(state_type, elements_of_interest, session_id, execution_engine)
@mcp.tool
async def suggest_next_step(
    current_state: Dict[str, Any],
    test_objective: str,
    executed_steps: List[Dict[str, Any]] = None,
    session_id: str = "default"
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
    return await nlp_processor.suggest_next_step(current_state, test_objective, executed_steps, session_id)
@mcp.tool
async def build_test_suite(
    test_name: str,
    session_id: str = "default",
    tags: List[str] = None,
    documentation: str = "",
    remove_library_prefixes: bool = True
) -> Dict[str, Any]:
    """Generate Robot Framework test suite from successful steps.
    
    IMPORTANT: Only use AFTER validating all steps individually with execute_step().
    This tool generates .robot files from previously executed and verified steps.
    Do NOT write test suites before confirming each keyword works correctly.
    
    Recommended workflow:
    1. Use execute_step() to test each keyword
    2. Verify expected outputs and behavior  
    3. Only then use build_test_suite() to create .robot files
    
    Args:
        test_name: Name for the test case
        session_id: Session with executed steps (must contain verified steps)
        tags: Test tags
        documentation: Test documentation
        remove_library_prefixes: Remove library prefixes from keywords (default: True)
    """
    if tags is None:
        tags = []
    return await test_builder.build_suite(session_id, test_name, tags, documentation, remove_library_prefixes)
@mcp.tool
async def validate_scenario(
    parsed_scenario: Dict[str, Any], 
    available_libraries: List[str] = None
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
    max_recommendations: int = 5
) -> Dict[str, Any]:
    """Recommend Robot Framework libraries based on test scenario.
    
    Args:
        scenario: Natural language description of the test scenario
        context: Testing context (web, mobile, api, database, desktop, system, visual)
        max_recommendations: Maximum number of library recommendations to return
    """
    return library_recommender.recommend_libraries(scenario, context, max_recommendations)

@mcp.tool
async def get_page_source(
    session_id: str = "default",
    full_source: bool = False,
    filtered: bool = False,
    filtering_level: str = "standard"
) -> Dict[str, Any]:
    """Get page source and context for a browser session with optional DOM filtering.
    
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
    return await execution_engine.get_page_source(session_id, full_source, filtered, filtering_level)

@mcp.tool
async def check_library_availability(
    libraries: List[str]
) -> Dict[str, Any]:
    """Check if Robot Framework libraries are available before installation.
    
    Args:
        libraries: List of library names to check (e.g., ['Browser', 'SeleniumLibrary', 'RequestsLibrary'])
    
    Returns:
        Dict with availability status and installation suggestions
    """
    return execution_engine.check_library_requirements(libraries)

@mcp.tool
async def get_library_status(
    library_name: str
) -> Dict[str, Any]:
    """Get detailed installation status for a specific library.
    
    Args:
        library_name: Name of the library to check (e.g., 'Browser', 'SeleniumLibrary')
    
    Returns:
        Dict with detailed status and installation information
    """
    return execution_engine.get_installation_status(library_name)

@mcp.tool
async def get_available_keywords(
    library_name: str = None
) -> List[Dict[str, Any]]:
    """Get available Robot Framework keywords with native RF libdoc short documentation, optionally filtered by library.
    
    Uses Robot Framework's native libdoc API to provide accurate short_doc, argument types, and metadata.
    Falls back to inspection-based discovery if libdoc is not available.
    
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
    """
    return execution_engine.get_available_keywords(library_name)

@mcp.tool
async def search_keywords(
    pattern: str
) -> List[Dict[str, Any]]:
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
    keyword_name: str,
    library_name: str = None
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
async def validate_step_before_suite(
    keyword: str,
    arguments: List[str] = None,
    session_id: str = "default",
    expected_outcome: str = None
) -> Dict[str, Any]:
    """Validate a single step before adding it to a test suite.
    
    This method enforces stepwise test development by requiring step validation
    before suite generation. Use this to verify each keyword works as expected.
    
    Workflow:
    1. Call this method for each test step
    2. Verify the step succeeds and produces expected results
    3. Only after all steps are validated, use build_test_suite()
    
    Args:
        keyword: Robot Framework keyword to validate
        arguments: Arguments for the keyword
        session_id: Session identifier
        expected_outcome: Optional description of expected result for validation
        
    Returns:
        Validation result with success status, output, and recommendations
    """
    if arguments is None:
        arguments = []
        
    # Execute the step with detailed error reporting
    result = await execution_engine.execute_step(keyword, arguments, session_id, detail_level="full")
    
    # Add validation metadata
    result["validated"] = result.get("success", False)
    result["validation_time"] = result.get("execution_time")
    
    if expected_outcome:
        result["expected_outcome"] = expected_outcome
        result["meets_expectation"] = "unknown"  # AI agent should evaluate this
    
    # Add guidance for next steps
    if result.get("success"):
        result["next_step_guidance"] = "✅ Step validated successfully. Safe to include in test suite."
    else:
        result["next_step_guidance"] = "❌ Step failed validation. Fix issues before adding to test suite."
        result["debug_suggestions"] = [
            "Check keyword spelling and library availability",
            "Verify argument types and values", 
            "Ensure required browser/context is open",
            "Review error message for specific issues"
        ]
    
    return result

@mcp.tool
async def get_session_validation_status(
    session_id: str = "default"
) -> Dict[str, Any]:
    """Get validation status of all steps in a session.
    
    Use this to check which steps have been validated and are ready for test suite generation.
    Helps ensure stepwise test development by showing validation progress.
    
    Args:
        session_id: Session identifier to check
        
    Returns:
        Validation status with passed/failed step counts and readiness assessment
    """
    return execution_engine.get_session_validation_status(session_id)

@mcp.tool
async def validate_test_readiness(
    session_id: str = "default"  
) -> Dict[str, Any]:
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



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()

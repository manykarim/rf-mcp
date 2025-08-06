"""Main MCP Server implementation for Robot Framework integration."""

import asyncio
import json
import logging
from typing import Any, Dict, List

from fastmcp import FastMCP

from robotmcp.components.execution_engine import ExecutionEngine
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
execution_engine = ExecutionEngine()
state_manager = StateManager()
test_builder = TestBuilder()

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
    session_id: str = "default"
) -> Dict[str, Any]:
    """Execute a single test step using Robot Framework API.
    
    Args:
        keyword: Robot Framework keyword name
        arguments: Arguments for the keyword
        session_id: Session identifier for maintaining context
    """
    if arguments is None:
        arguments = []
    return await execution_engine.execute_step(keyword, arguments, session_id)
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
    return await state_manager.get_state(state_type, elements_of_interest, session_id)
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
    documentation: str = ""
) -> Dict[str, Any]:
    """Generate Robot Framework test suite from successful steps.
    
    Args:
        test_name: Name for the test case
        session_id: Session with executed steps
        tags: Test tags
        documentation: Test documentation
    """
    if tags is None:
        tags = []
    return await test_builder.build_suite(session_id, test_name, tags, documentation)
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
    full_source: bool = False
) -> Dict[str, Any]:
    """Get page source and context for a browser session.
    
    Args:
        session_id: Session identifier
        full_source: If True, returns complete page source. If False, returns preview only.
    """
    return await execution_engine.get_page_source(session_id, full_source)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run()

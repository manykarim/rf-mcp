"""Test Builder component for generating Robot Framework test suites from executed steps."""

import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

try:
    from robot.api import TestSuite
except ImportError:
    TestSuite = None

try:
    from robot.running.model import Keyword as RunningKeyword
except ImportError:
    RunningKeyword = None

# Import shared library detection utility
from robotmcp.utils.library_detector import detect_library_from_keyword

logger = logging.getLogger(__name__)

@dataclass
class TestCaseStep:
    """Represents a test case step."""
    keyword: str
    arguments: List[str]
    comment: Optional[str] = None

@dataclass
class GeneratedTestCase:
    """Represents a generated test case."""
    name: str
    steps: List[TestCaseStep]
    documentation: str = ""
    tags: List[str] = None
    setup: Optional[TestCaseStep] = None
    teardown: Optional[TestCaseStep] = None

@dataclass
class GeneratedTestSuite:
    """Represents a generated test suite."""
    name: str
    test_cases: List[GeneratedTestCase]
    documentation: str = ""
    tags: List[str] = None
    setup: Optional[TestCaseStep] = None
    teardown: Optional[TestCaseStep] = None
    imports: List[str] = None

class TestBuilder:
    """Builds Robot Framework test suites from successful execution steps."""
    
    def __init__(self, execution_engine=None):
        self.execution_engine = execution_engine
        self.optimization_rules = {
            'combine_waits': True,
            'remove_redundant_verifications': True,
            'group_similar_actions': True,
            'add_meaningful_comments': True,
            'generate_variables': True
        }

    async def build_suite(
        self,
        session_id: str = "default",
        test_name: str = "",
        tags: List[str] = None,
        documentation: str = "",
        remove_library_prefixes: bool = True
    ) -> Dict[str, Any]:
        """
        Generate Robot Framework test suite from successful execution steps.
        
        Args:
            session_id: Session with executed steps
            test_name: Name for the test case
            tags: Test tags
            documentation: Test documentation
            remove_library_prefixes: Remove library prefixes from keywords (e.g., "Browser.Click" -> "Click")
            
        Returns:
            Generated test suite with RF API objects and text representation
        """
        try:
            if tags is None:
                tags = []
            
            # First, check if session is ready for test suite generation
            if self.execution_engine:
                readiness_check = await self.execution_engine.validate_test_readiness(session_id)
                if not readiness_check.get("ready_for_suite_generation", False):
                    return {
                        "success": False,
                        "error": "Session not ready for test suite generation",
                        "guidance": readiness_check.get("guidance", []),
                        "validation_summary": readiness_check.get("validation_summary", {}),
                        "recommendation": "Use validate_step_before_suite() to validate individual steps first"
                    }
            
            # Get session steps from execution engine
            steps = await self._get_session_steps(session_id)
            
            if not steps:
                return {
                    "success": False,
                    "error": f"No steps found for session '{session_id}'",
                    "suite": None
                }
            
            # Filter successful steps
            successful_steps = [step for step in steps if step.get("status") == "pass"]
            
            if not successful_steps:
                return {
                    "success": False,
                    "error": "No successful steps to build suite from",
                    "suite": None
                }
            
            # Build test case from steps
            test_case = await self._build_test_case(
                successful_steps, test_name or f"Test_{session_id}", tags, documentation, session_id
            )
            
            # Create test suite
            suite = await self._build_test_suite([test_case], session_id)
            
            # Apply library prefix removal if requested
            if remove_library_prefixes:
                suite = self._apply_prefix_removal(suite)
            
            # Generate Robot Framework API objects
            rf_suite = await self._create_rf_suite(suite)
            
            # Generate text representation
            rf_text = await self._generate_rf_text(suite)
            
            # Generate execution statistics
            stats = await self._generate_statistics(successful_steps, suite)
            
            return {
                "success": True,
                "session_id": session_id,
                "suite": {
                    "name": suite.name,
                    "documentation": suite.documentation,
                    "tags": suite.tags or [],
                    "test_cases": [
                        {
                            "name": tc.name,
                            "documentation": tc.documentation,
                            "tags": tc.tags or [],
                            "steps": [
                                {
                                    "keyword": step.keyword,
                                    "arguments": [self._escape_robot_argument(arg) for arg in (step.arguments or [])],
                                    "comment": step.comment
                                } for step in tc.steps
                            ],
                            "setup": {
                                "keyword": tc.setup.keyword,
                                "arguments": [self._escape_robot_argument(arg) for arg in (tc.setup.arguments or [])]
                            } if tc.setup else None,
                            "teardown": {
                                "keyword": tc.teardown.keyword, 
                                "arguments": [self._escape_robot_argument(arg) for arg in (tc.teardown.arguments or [])]
                            } if tc.teardown else None
                        } for tc in suite.test_cases
                    ],
                    "imports": suite.imports or [],
                    "setup": {
                        "keyword": suite.setup.keyword,
                        "arguments": [self._escape_robot_argument(arg) for arg in (suite.setup.arguments or [])]
                    } if suite.setup else None,
                    "teardown": {
                        "keyword": suite.teardown.keyword,
                        "arguments": [self._escape_robot_argument(arg) for arg in (suite.teardown.arguments or [])]
                    } if suite.teardown else None
                },
                "rf_text": rf_text,
                "statistics": stats,
                "optimization_applied": list(self.optimization_rules.keys())
            }
            
        except Exception as e:
            logger.error(f"Error building test suite: {e}")
            return {
                "success": False,
                "error": str(e),
                "suite": None
            }

    async def _get_session_steps(self, session_id: str) -> List[Dict[str, Any]]:
        """Get executed steps from session."""
        if not self.execution_engine:
            logger.warning("No execution engine provided, returning empty steps list")
            return []
        
        try:
            # Get session from execution engine
            session = self.execution_engine.sessions.get(session_id)
            if not session:
                logger.warning(f"Session '{session_id}' not found")
                return []
            
            # Update session activity to prevent cleanup during suite building
            from datetime import datetime
            session.last_activity = datetime.now()
            logger.debug(f"Updated session {session_id} activity during suite building")
            
            # Convert ExecutionStep objects to dictionary format
            steps = []
            for step in session.steps:
                step_dict = {
                    "keyword": step.keyword,
                    "arguments": step.arguments,
                    "status": step.status,
                    "step_id": step.step_id
                }
                
                # Add optional fields if available
                if step.error:
                    step_dict["error"] = step.error
                if step.result:
                    step_dict["result"] = step.result
                if step.start_time and step.end_time:
                    step_dict["duration"] = (step.end_time - step.start_time).total_seconds()
                
                steps.append(step_dict)
            
            logger.info(f"Retrieved {len(steps)} steps from session '{session_id}'")
            return steps
            
        except Exception as e:
            logger.error(f"Error retrieving session steps: {e}")
            return []

    async def _build_test_case(
        self,
        steps: List[Dict[str, Any]],
        test_name: str,
        tags: List[str],
        documentation: str,
        session_id: str = None
    ) -> GeneratedTestCase:
        """Build a test case from execution steps."""
        
        # Convert steps to test case steps
        test_steps = []
        imports = set()
        
        for step in steps:
            keyword = step.get("keyword", "")
            arguments = step.get("arguments", [])
            
            # Handle import statements separately
            if keyword.lower() == "import library":
                if arguments:
                    imports.add(arguments[0])
                continue
            
            # Apply optimizations
            optimized_step = await self._optimize_step(keyword, arguments, test_steps, session_id)
            
            if optimized_step:  # Only add if not filtered out by optimization
                test_steps.append(optimized_step)
        
        # Generate meaningful documentation if not provided
        if not documentation:
            documentation = await self._generate_documentation(test_steps, test_name)
        
        # Add setup and teardown if needed
        setup, teardown = await self._generate_setup_teardown(test_steps)
        
        return GeneratedTestCase(
            name=test_name,
            steps=test_steps,
            documentation=documentation,
            tags=tags or [],
            setup=setup,
            teardown=teardown
        )

    async def _build_test_suite(
        self,
        test_cases: List[GeneratedTestCase],
        session_id: str
    ) -> GeneratedTestSuite:
        """Build a test suite from test cases."""
        
        # Get libraries from session execution history (more reliable than keyword pattern matching)
        all_imports = self._get_session_libraries(session_id)
        
        # If no session libraries found, fall back to keyword detection
        if not all_imports:
            for test_case in test_cases:
                for step in test_case.steps:
                    library = await self._detect_library_from_keyword(step.keyword, session_id)
                    if library and library != "BuiltIn":  # Exclude BuiltIn as it's automatically available
                        all_imports.add(library)
        
        # Validate library exclusion rules for test suite generation
        # Only validate if we don't have a session with execution history
        if not self._session_has_execution_history(session_id):
            self._validate_suite_library_exclusions(all_imports, session_id)
        
        # BuiltIn is automatically available in Robot Framework, so we don't import it explicitly
        
        # Generate suite documentation
        suite_docs = await self._generate_suite_documentation(test_cases, session_id)
        
        # Generate common tags
        common_tags = await self._extract_common_tags(test_cases)
        
        return GeneratedTestSuite(
            name=f"Generated_Suite_{session_id}",
            test_cases=test_cases,
            documentation=suite_docs,
            tags=common_tags,
            imports=list(all_imports)
        )

    async def _optimize_step(
        self,
        keyword: str,
        arguments: List[str],
        existing_steps: List[TestCaseStep],
        session_id: str = None
    ) -> Optional[TestCaseStep]:
        """Apply optimization rules to a step."""
        
        # Rule: Combine consecutive waits
        if self.optimization_rules.get('combine_waits') and keyword.lower() in ['sleep', 'wait']:
            if existing_steps and existing_steps[-1].keyword.lower() in ['sleep', 'wait']:
                # Skip this wait step as it's redundant
                return None
        
        # Rule: Remove redundant verifications
        if self.optimization_rules.get('remove_redundant_verifications'):
            if keyword.lower().startswith('page should contain'):
                # Check if we already have the same verification
                for step in existing_steps:
                    if (step.keyword.lower().startswith('page should contain') and 
                        step.arguments == arguments):
                        return None  # Skip redundant verification
        
        # Use original arguments - they already worked during execution
        processed_arguments = arguments
        
        # Rule: Add meaningful comments
        comment = None
        if self.optimization_rules.get('add_meaningful_comments'):
            comment = await self._generate_step_comment(keyword, processed_arguments)
        
        return TestCaseStep(
            keyword=keyword,
            arguments=processed_arguments,
            comment=comment
        )


    async def _generate_step_comment(self, keyword: str, arguments: List[str]) -> Optional[str]:
        """Generate a meaningful comment for a step."""
        
        keyword_lower = keyword.lower()
        
        if "open browser" in keyword_lower:
            url = arguments[0] if arguments else "default"
            browser = arguments[1] if len(arguments) > 1 else "default browser"
            return f"# Open {browser} and navigate to {url}"
        
        elif "input text" in keyword_lower:
            element = arguments[0] if arguments else "element"
            value = arguments[1] if len(arguments) > 1 else "value"
            return f"# Enter '{value}' into {element}"
        
        elif "click" in keyword_lower:
            element = arguments[0] if arguments else "element"
            return f"# Click on {element}"
        
        elif "should contain" in keyword_lower:
            text = arguments[0] if arguments else "text"
            return f"# Verify page contains '{text}'"
        
        return None

    async def _generate_documentation(self, steps: List[TestCaseStep], test_name: str) -> str:
        """Generate documentation for a test case."""
        
        # Analyze steps to understand the test flow
        flow_description = []
        
        for step in steps:
            keyword_lower = step.keyword.lower()
            
            if "open browser" in keyword_lower:
                flow_description.append("Opens browser")
            elif "go to" in keyword_lower or "navigate" in keyword_lower:
                flow_description.append("Navigates to page")
            elif "input" in keyword_lower:
                flow_description.append("Enters data")
            elif "click" in keyword_lower:
                flow_description.append("Performs click action")
            elif "should" in keyword_lower or "verify" in keyword_lower:
                flow_description.append("Verifies result")
            elif "close" in keyword_lower:
                flow_description.append("Cleans up")
        
        if flow_description:
            description = ", ".join(flow_description)
            return f"Test case that {description.lower()}."
        
        return f"Automated test case: {test_name}"

    async def _generate_setup_teardown(
        self,
        steps: List[TestCaseStep]
    ) -> Tuple[Optional[TestCaseStep], Optional[TestCaseStep]]:
        """Generate setup and teardown steps if needed."""
        
        setup = None
        teardown = None
        
        # Check if we need browser cleanup
        has_browser_actions = any(
            "browser" in step.keyword.lower() or 
            "click" in step.keyword.lower() or
            "fill" in step.keyword.lower() or
            "get text" in step.keyword.lower() or
            "input" in step.keyword.lower()
            for step in steps
        )
        
        # Determine if using Browser Library or SeleniumLibrary
        has_browser_lib = any("new browser" in step.keyword.lower() or 
                             "new page" in step.keyword.lower() or
                             "fill" in step.keyword.lower() 
                             for step in steps)
        
        if has_browser_actions:
            # Check if we already have close browser
            has_close = any("close browser" in step.keyword.lower() for step in steps)
            
            if not has_close:
                teardown = TestCaseStep(
                    keyword="Close Browser",
                    arguments=[],
                    comment="# Cleanup: Close browser"
                )
        
        return setup, teardown

    async def _detect_library_from_keyword(self, keyword: str, session_id: str = None) -> Optional[str]:
        """
        Detect which library a keyword belongs to, respecting session library choice.
        
        Args:
            keyword: Keyword name to detect library for
            session_id: Session ID to check for library preference
            
        Returns:
            Library name or None
        """
        # First check if we have a session with a specific web automation library
        if session_id and self.execution_engine and hasattr(self.execution_engine, 'sessions'):
            session = self.execution_engine.sessions.get(session_id)
            if session:
                session_web_lib = self._get_session_web_library(session)
                if session_web_lib:
                    # Check if this keyword could belong to the session's library
                    keyword_lower = keyword.lower().strip()
                    
                    # For Browser Library keywords
                    if (session_web_lib == "Browser" and 
                        any(kw in keyword_lower for kw in [
                            'click', 'fill text', 'get text', 'wait for elements state',
                            'check checkbox', 'select options by', 'hover', 'new browser', 'new page'
                        ])):
                        return "Browser"
                    
                    # For SeleniumLibrary keywords  
                    elif (session_web_lib == "SeleniumLibrary" and
                          any(kw in keyword_lower for kw in [
                              'click element', 'input text', 'select from list', 'wait until element',
                              'open browser', 'close browser', 'select checkbox', 'get text'
                          ])):
                        return "SeleniumLibrary"
        
        # Fallback to shared library detection utility with dynamic keyword discovery
        keyword_discovery = None
        if self.execution_engine and hasattr(self.execution_engine, 'keyword_discovery'):
            keyword_discovery = self.execution_engine.keyword_discovery
        
        return detect_library_from_keyword(keyword, keyword_discovery)
    
    def _get_session_libraries(self, session_id: str) -> set:
        """
        Get libraries that were actually used in session execution.
        
        This method prioritizes the session's imported_libraries over keyword pattern matching
        because the session has already proven these libraries work together successfully.
        
        Args:
            session_id: Session ID to get libraries from
            
        Returns:
            Set of library names from session execution history
        """
        if not self.execution_engine or not hasattr(self.execution_engine, 'sessions'):
            return set()
        
        session = self.execution_engine.sessions.get(session_id)
        if not session:
            return set()
        
        # Use imported_libraries from session (these are known to work together)
        session_libraries = set(session.imported_libraries)
        
        # Filter out BuiltIn as it's automatically available
        session_libraries.discard("BuiltIn")
        
        logger.debug(f"Session {session_id} libraries from execution history: {session_libraries}")
        return session_libraries
    
    def _session_has_execution_history(self, session_id: str) -> bool:
        """
        Check if session has execution history (successful steps).
        
        Args:
            session_id: Session ID to check
            
        Returns:
            True if session has executed steps, False otherwise
        """
        if not self.execution_engine or not hasattr(self.execution_engine, 'sessions'):
            return False
        
        session = self.execution_engine.sessions.get(session_id)
        if not session:
            return False
        
        # Check if session has any successful steps
        has_history = len(session.steps) > 0
        logger.debug(f"Session {session_id} has execution history: {has_history} ({len(session.steps)} steps)")
        return has_history
    
    def _get_session_web_library(self, session) -> Optional[str]:
        """
        Safely get the web automation library from a session.
        
        Handles both new sessions (with get_web_automation_library method)
        and older sessions (without the method).
        
        Args:
            session: ExecutionSession object
            
        Returns:
            Web automation library name or None
        """
        # Try the new method first
        if hasattr(session, 'get_web_automation_library'):
            return session.get_web_automation_library()
        
        # Fallback for older sessions - check imported_libraries directly
        if hasattr(session, 'imported_libraries'):
            web_automation_libs = ['Browser', 'SeleniumLibrary']
            for lib in session.imported_libraries:
                if lib in web_automation_libs:
                    return lib
        
        # Final fallback - check browser_state.active_library
        if hasattr(session, 'browser_state') and hasattr(session.browser_state, 'active_library'):
            active_lib = session.browser_state.active_library
            if active_lib == 'browser':
                return 'Browser'
            elif active_lib == 'selenium':
                return 'SeleniumLibrary'
        
        return None

    async def _generate_suite_documentation(
        self,
        test_cases: List[GeneratedTestCase],
        session_id: str
    ) -> str:
        """Generate documentation for the test suite."""
        
        case_count = len(test_cases)
        
        # Analyze test types
        test_types = set()
        for test_case in test_cases:
            for step in test_case.steps:
                if "browser" in step.keyword.lower():
                    test_types.add("web automation")
                elif "request" in step.keyword.lower():
                    test_types.add("API testing")
                elif "database" in step.keyword.lower():
                    test_types.add("database testing")
        
        type_description = ", ".join(test_types) if test_types else "automation"
        
        # Create a single-line documentation that won't break Robot Framework syntax
        doc = f"Test suite generated from session {session_id} containing {case_count} test case{'s' if case_count != 1 else ''} for {type_description}."
        
        return doc

    def _format_rf_documentation(self, documentation: str) -> List[str]:
        """Format documentation for Robot Framework syntax.
        
        Handles both single-line and multi-line documentation properly.
        Multi-line documentation uses '...' continuation markers.
        
        Args:
            documentation: Documentation string (can contain newlines)
            
        Returns:
            List of formatted documentation lines
        """
        if not documentation:
            return []
        
        # Split documentation into lines and clean them
        doc_lines = [line.strip() for line in documentation.strip().split('\n') if line.strip()]
        
        if not doc_lines:
            return []
        
        formatted_lines = []
        
        if len(doc_lines) == 1:
            # Single line documentation
            formatted_lines.append(f"Documentation    {doc_lines[0]}")
        else:
            # Multi-line documentation with continuation markers
            formatted_lines.append(f"Documentation    {doc_lines[0]}")
            for line in doc_lines[1:]:
                formatted_lines.append(f"...              {line}")
        
        return formatted_lines

    def _format_rf_test_case_documentation(self, documentation: str) -> List[str]:
        """Format test case documentation for Robot Framework syntax.
        
        Similar to suite documentation but uses [Documentation] format with proper indentation.
        
        Args:
            documentation: Documentation string (can contain newlines)
            
        Returns:
            List of formatted test case documentation lines
        """
        if not documentation:
            return []
        
        # Split documentation into lines and clean them
        doc_lines = [line.strip() for line in documentation.strip().split('\n') if line.strip()]
        
        if not doc_lines:
            return []
        
        formatted_lines = []
        
        if len(doc_lines) == 1:
            # Single line test case documentation
            formatted_lines.append(f"    [Documentation]    {doc_lines[0]}")
        else:
            # Multi-line test case documentation with continuation markers
            formatted_lines.append(f"    [Documentation]    {doc_lines[0]}")
            for line in doc_lines[1:]:
                formatted_lines.append(f"    ...                {line}")
        
        return formatted_lines

    async def _extract_common_tags(self, test_cases: List[GeneratedTestCase]) -> List[str]:
        """Extract common tags across test cases."""
        
        if not test_cases:
            return []
        
        # Find tags that appear in all test cases
        common_tags = set(test_cases[0].tags or [])
        
        for test_case in test_cases[1:]:
            case_tags = set(test_case.tags or [])
            common_tags = common_tags.intersection(case_tags)
        
        # Add generated tags based on content analysis
        generated_tags = ["automated", "generated"]
        
        # Analyze test content for additional tags
        has_web = any(
            any("browser" in step.keyword.lower() for step in tc.steps)
            for tc in test_cases
        )
        if has_web:
            generated_tags.append("web")
        
        has_api = any(
            any("request" in step.keyword.lower() for step in tc.steps)
            for tc in test_cases
        )
        if has_api:
            generated_tags.append("api")
        
        return list(common_tags) + generated_tags

    async def _create_rf_suite(self, suite: GeneratedTestSuite) -> TestSuite:
        """Create Robot Framework API suite object."""
        
        rf_suite = TestSuite(name=suite.name)
        rf_suite.doc = suite.documentation
        
        # Add imports
        for library in suite.imports or []:
            rf_suite.resource.imports.library(library)
        
        # Add test cases
        for test_case in suite.test_cases:
            rf_test = rf_suite.tests.create(
                name=test_case.name,
                doc=test_case.documentation
            )
            
            # Add tags
            if test_case.tags:
                rf_test.tags.add(test_case.tags)
            
            # Add setup
            if test_case.setup:
                rf_test.setup.config(
                    name=test_case.setup.keyword,
                    args=test_case.setup.arguments
                )
            
            # Add steps
            for step in test_case.steps:
                rf_test.body.create_keyword(
                    name=step.keyword,
                    args=step.arguments
                )
            
            # Add teardown
            if test_case.teardown:
                rf_test.teardown.config(
                    name=test_case.teardown.keyword,
                    args=test_case.teardown.arguments
                )
        
        return rf_suite

    async def _generate_rf_text(self, suite: GeneratedTestSuite) -> str:
        """Generate Robot Framework text representation."""
        
        lines = []
        
        # Suite header
        lines.append(f"*** Settings ***")
        
        # Format documentation properly for Robot Framework
        if suite.documentation:
            doc_lines = self._format_rf_documentation(suite.documentation)
            lines.extend(doc_lines)
        
        # Imports
        if suite.imports:
            for library in suite.imports:
                lines.append(f"Library          {library}")
        
        if suite.tags:
            lines.append(f"Force Tags       {' '.join(suite.tags)}")
        
        lines.append("")
        
        # Test cases
        lines.append("*** Test Cases ***")
        
        for test_case in suite.test_cases:
            lines.append(f"{test_case.name}")
            
            if test_case.documentation:
                # Format test case documentation properly
                test_doc_lines = self._format_rf_test_case_documentation(test_case.documentation)
                lines.extend(test_doc_lines)
            
            if test_case.tags:
                lines.append(f"    [Tags]    {' '.join(test_case.tags)}")
            
            if test_case.setup:
                escaped_setup_args = [self._escape_robot_argument(arg) for arg in test_case.setup.arguments]
                lines.append(f"    [Setup]    {test_case.setup.keyword}    {' '.join(escaped_setup_args)}")
            
            # Test steps
            for step in test_case.steps:
                step_line = f"    {step.keyword}"
                if step.arguments:
                    # Convert locators for consistency if using execution engine
                    processed_args = step.arguments.copy()
                    if (self.execution_engine and hasattr(self.execution_engine, '_convert_locator_for_library') 
                        and step.arguments):
                        # Detect library from keyword
                        library = await self._detect_library_from_keyword(step.keyword)
                        if library and any(kw in step.keyword.lower() for kw in ['click', 'fill', 'get text', 'wait', 'select']):
                            converted_locator = self.execution_engine._convert_locator_for_library(step.arguments[0], library)
                            if converted_locator != step.arguments[0]:
                                processed_args[0] = converted_locator
                    
                    # Escape arguments that start with special characters
                    escaped_args = [self._escape_robot_argument(arg) for arg in processed_args]
                    # Use proper 4-space separation between keyword and arguments
                    args_str = "    ".join(escaped_args)
                    step_line += f"    {args_str}"
                
                if step.comment:
                    step_line += f"    {step.comment}"
                
                lines.append(step_line)
            
            if test_case.teardown:
                escaped_teardown_args = [self._escape_robot_argument(arg) for arg in test_case.teardown.arguments]
                lines.append(f"    [Teardown]    {test_case.teardown.keyword}    {' '.join(escaped_teardown_args)}")
            
            lines.append("")
        
        return "\n".join(lines)

    async def _generate_statistics(
        self,
        steps: List[Dict[str, Any]],
        suite: GeneratedTestSuite
    ) -> Dict[str, Any]:
        """Generate execution statistics."""
        
        total_original_steps = len(steps)
        total_optimized_steps = sum(len(tc.steps) for tc in suite.test_cases)
        
        # Count step types
        step_types = {}
        for test_case in suite.test_cases:
            for step in test_case.steps:
                step_type = self._categorize_step(step.keyword)
                step_types[step_type] = step_types.get(step_type, 0) + 1
        
        optimization_ratio = (total_original_steps - total_optimized_steps) / total_original_steps if total_original_steps > 0 else 0
        
        return {
            "original_steps": total_original_steps,
            "optimized_steps": total_optimized_steps,
            "optimization_ratio": optimization_ratio,
            "test_cases_generated": len(suite.test_cases),
            "libraries_required": len(suite.imports or []),
            "step_types": step_types,
            "estimated_execution_time": total_optimized_steps * 2  # 2 seconds per step estimate
        }

    def _categorize_step(self, keyword: str) -> str:
        """Categorize a step by its type."""
        keyword_lower = keyword.lower()
        
        if any(kw in keyword_lower for kw in ['open', 'go to', 'navigate']):
            return "navigation"
        elif any(kw in keyword_lower for kw in ['click', 'press', 'select']):
            return "interaction"
        elif any(kw in keyword_lower for kw in ['input', 'type', 'enter', 'fill']):
            return "input"
        elif any(kw in keyword_lower for kw in ['should', 'verify', 'assert', 'check']):
            return "verification"
        elif any(kw in keyword_lower for kw in ['wait', 'sleep', 'pause']):
            return "synchronization"
        elif any(kw in keyword_lower for kw in ['close', 'quit', 'cleanup']):
            return "cleanup"
        else:
            return "other"

    def _escape_robot_argument(self, arg: str) -> str:
        """Escape Robot Framework arguments that start with special characters."""
        if not arg:
            return arg
        
        # Escape arguments starting with # (treated as comments in RF)
        if arg.startswith('#'):
            return f"\\{arg}"
        
        # Future escaping rules can be added here:
        # - Arguments starting with $ or & (variables)
        # - Arguments with spaces that need quoting
        # - Arguments with special RF syntax
        
        return arg

    def _remove_library_prefix(self, keyword: str) -> str:
        """Remove library prefix from keyword name for cleaner test suites.
        
        Converts "LibraryName.KeywordName" -> "KeywordName"
        Leaves keywords without prefixes unchanged.
        
        Args:
            keyword: Keyword name potentially with library prefix
            
        Returns:
            Keyword name without library prefix
        """
        if '.' in keyword:
            return keyword.split('.', 1)[1]  # Return everything after first dot
        return keyword

    def _apply_prefix_removal(self, suite: GeneratedTestSuite) -> GeneratedTestSuite:
        """Apply library prefix removal to all keywords in the test suite.
        
        Args:
            suite: Test suite with potentially prefixed keywords
            
        Returns:
            Test suite with library prefixes removed from keywords
        """
        # Process test cases
        for test_case in suite.test_cases:
            # Process test steps
            for step in test_case.steps:
                step.keyword = self._remove_library_prefix(step.keyword)
            
            # Process setup
            if test_case.setup:
                test_case.setup.keyword = self._remove_library_prefix(test_case.setup.keyword)
            
            # Process teardown  
            if test_case.teardown:
                test_case.teardown.keyword = self._remove_library_prefix(test_case.teardown.keyword)
        
        # Process suite-level setup and teardown
        if suite.setup:
            suite.setup.keyword = self._remove_library_prefix(suite.setup.keyword)
        
        if suite.teardown:
            suite.teardown.keyword = self._remove_library_prefix(suite.teardown.keyword)
        
        return suite
    
    def _get_session_libraries(self, session_id: str) -> set:
        """
        Get libraries that were actually used in session execution.
        
        This method prioritizes the session's imported_libraries over keyword pattern matching
        because the session has already proven these libraries work together successfully.
        
        Args:
            session_id: Session ID to get libraries from
            
        Returns:
            Set of library names from session execution history
        """
        if not self.execution_engine or not hasattr(self.execution_engine, 'sessions'):
            return set()
        
        session = self.execution_engine.sessions.get(session_id)
        if not session:
            return set()
        
        # Use imported_libraries from session (these are known to work together)
        session_libraries = set(session.imported_libraries)
        
        # Filter out BuiltIn as it's automatically available
        session_libraries.discard("BuiltIn")
        
        logger.debug(f"Session {session_id} libraries from execution history: {session_libraries}")
        return session_libraries
    
    def _session_has_execution_history(self, session_id: str) -> bool:
        """
        Check if session has execution history (successful steps).
        
        Args:
            session_id: Session ID to check
            
        Returns:
            True if session has executed steps, False otherwise
        """
        if not self.execution_engine or not hasattr(self.execution_engine, 'sessions'):
            return False
        
        session = self.execution_engine.sessions.get(session_id)
        if not session:
            return False
        
        # Check if session has any successful steps
        has_history = len(session.steps) > 0
        logger.debug(f"Session {session_id} has execution history: {has_history} ({len(session.steps)} steps)")
        return has_history
    
    def _validate_suite_library_exclusions(self, imports: set, session_id: str) -> None:
        """
        Validate that the test suite doesn't violate library exclusion rules.
        
        Only applies strict validation for new/empty sessions. Sessions with execution
        history have already proven their library combinations work successfully.
        
        Args:
            imports: Set of library names to be imported
            session_id: Session ID for error reporting
            
        Raises:
            ValueError: If conflicting libraries are detected for new sessions
        """
        # If session has execution history, trust its library choices
        if self._session_has_execution_history(session_id):
            logger.info(f"Session {session_id} has execution history - skipping library exclusion validation")
            return
        
        # Only apply strict validation for new/empty sessions
        web_automation_libs = ['Browser', 'SeleniumLibrary']
        detected_web_libs = [lib for lib in imports if lib in web_automation_libs]
        
        if len(detected_web_libs) > 1:
            raise ValueError(
                f"Test suite for session '{session_id}' contains conflicting web automation libraries: "
                f"{detected_web_libs}. Browser Library and SeleniumLibrary are mutually exclusive. "
                f"Please use separate sessions for different libraries."
            )
        
        # For new sessions, also check session consistency if execution engine is available
        if self.execution_engine and hasattr(self.execution_engine, 'sessions'):
            session = self.execution_engine.sessions.get(session_id)
            if session:
                # Use safe method to get web automation library (handles older sessions)
                session_web_lib = self._get_session_web_library(session)
                if session_web_lib and detected_web_libs:
                    suite_web_lib = detected_web_libs[0]
                    if session_web_lib != suite_web_lib:
                        logger.warning(
                            f"Session '{session_id}' uses '{session_web_lib}' but suite "
                            f"detected '{suite_web_lib}' from keywords. Using session library."
                        )
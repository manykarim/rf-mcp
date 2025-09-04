"""Main execution coordinator orchestrating all execution services."""

import logging
import os
from typing import Any, Dict, List, Optional, Union

from robotmcp.components.browser import BrowserLibraryManager
from robotmcp.components.execution import (
    KeywordExecutor,
    LocatorConverter,
    PageSourceService,
    SessionManager,
)
from robotmcp.components.execution.suite_execution_service import SuiteExecutionService
from robotmcp.models.config_models import ExecutionConfig
from robotmcp.models.session_models import ExecutionSession
from robotmcp.utils.library_checker import LibraryAvailabilityChecker
from robotmcp.utils.rf_libdoc_integration import get_rf_doc_storage

logger = logging.getLogger(__name__)


class ExecutionCoordinator:
    """
    Main coordinator that orchestrates all execution services.

    This class replaces the original monolithic execution engine with a clean,
    service-oriented architecture that separates concerns and improves maintainability.
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()

        # Initialize keyword override system first
        from robotmcp.components.keyword_overrides import (
            KeywordOverrideRegistry,
            setup_default_overrides,
        )

        self.override_registry = KeywordOverrideRegistry()

        # Initialize all service components
        self.session_manager = SessionManager(self.config)
        self.browser_library_manager = BrowserLibraryManager(self.config)
        self.page_source_service = PageSourceService(self.config)
        self.keyword_executor = KeywordExecutor(self.config, self.override_registry)
        self.locator_converter = LocatorConverter(self.config)
        self.suite_execution_service = SuiteExecutionService(self.config)

        # Initialize additional components for backward compatibility
        self.library_checker = LibraryAvailabilityChecker()
        self.rf_doc_storage = get_rf_doc_storage()

        # Set up default keyword overrides
        setup_default_overrides(self.override_registry, self)

        # Set session manager for keyword discovery
        from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery

        keyword_discovery = get_keyword_discovery()
        keyword_discovery.set_session_manager(self.session_manager)

        logger.info(
            "ExecutionCoordinator initialized with service-oriented architecture and keyword overrides"
        )

    # Add properties for override handlers to access browser libraries
    @property
    def browser_lib(self):
        """Access to Browser library instance for override handlers."""
        return self.browser_library_manager.browser_lib

    @property
    def selenium_lib(self):
        """Access to SeleniumLibrary instance for override handlers."""
        return self.browser_library_manager.selenium_lib

    async def execute_step(
        self,
        keyword: str,
        arguments: List[str] = None,
        session_id: str = "default",
        detail_level: str = "minimal",
        library_prefix: str = None,
        scenario_hint: str = None,
        assign_to: Union[str, List[str]] = None,
        use_context: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a single Robot Framework keyword step with intelligent library auto-configuration.

        Args:
            keyword: Robot Framework keyword name (supports Library.Keyword syntax)
            arguments: List of arguments for the keyword
            session_id: Session identifier
            detail_level: Level of detail in response ('minimal', 'standard', 'full')
            library_prefix: Optional explicit library name to override session search order
            scenario_hint: Optional scenario text for auto-configuration (used on first call)
            assign_to: Variable name(s) to assign the keyword's return value to

        Returns:
            Execution result with status, output, and state
        """
        try:
            if arguments is None:
                arguments = []

            # Get or create session and ensure libraries are loaded
            session = self.session_manager.get_or_create_session(session_id)

            # Ensure session libraries are loaded
            if not session.libraries_loaded:
                self._load_session_libraries(session)

            # Auto-configure session based on scenario hint (if provided and not already configured)
            if scenario_hint and not session.auto_configured:
                logger.info(
                    f"Auto-configuring session {session_id} based on scenario hint"
                )
                session.configure_from_scenario(scenario_hint)

            # Convert locators if needed
            converted_arguments = self._convert_locators_in_arguments(
                arguments, session
            )

            # Execute the keyword using the keyword executor with library prefix support
            result = await self.keyword_executor.execute_keyword(
                session=session,
                keyword=keyword,
                arguments=converted_arguments,
                browser_library_manager=self.browser_library_manager,
                detail_level=detail_level,
                library_prefix=library_prefix,
                assign_to=assign_to,
                use_context=use_context,
            )

            return result

        except Exception as e:
            logger.error(f"Error in ExecutionCoordinator.execute_step: {e}")
            return {
                "success": False,
                "error": f"Execution coordinator error: {str(e)}",
                "keyword": keyword,
                "arguments": arguments or [],
                "session_id": session_id,
            }

    async def get_page_source(
        self,
        session_id: str = "default",
        full_source: bool = False,
        filtered: bool = False,
        filtering_level: str = "standard",
    ) -> Dict[str, Any]:
        """
        Get page source for a session.

        Args:
            session_id: Session identifier
            full_source: If True, returns complete page source
            filtered: If True, returns filtered page source
            filtering_level: Filtering intensity ('minimal', 'standard', 'aggressive')

        Returns:
            Page source data and metadata
        """
        try:
            session = self.session_manager.get_session(session_id)
            if not session:
                return {"success": False, "error": f"Session '{session_id}' not found"}

            return await self.page_source_service.get_page_source(
                session=session,
                browser_library_manager=self.browser_library_manager,
                full_source=full_source,
                filtered=filtered,
                filtering_level=filtering_level,
            )

        except Exception as e:
            logger.error(f"Error getting page source for session {session_id}: {e}")
            return {"success": False, "error": f"Failed to get page source: {str(e)}"}

    def create_session(self, session_id: str) -> ExecutionSession:
        """Create a new execution session."""
        return self.session_manager.create_session(session_id)

    def get_session(self, session_id: str) -> Optional[ExecutionSession]:
        """Get an existing session by ID and ensure libraries are loaded."""
        session = self.session_manager.get_session(session_id)
        if session and not session.libraries_loaded:
            self._load_session_libraries(session)
        return session

    def remove_session(self, session_id: str) -> bool:
        """Remove a session."""
        return self.session_manager.remove_session(session_id)

    def _load_session_libraries(self, session: ExecutionSession) -> None:
        """
        Load libraries specified in session search order.

        This ensures that libraries configured for the session (like RequestsLibrary)
        are actually loaded into the library manager and available for keyword execution.

        Args:
            session: ExecutionSession with library search order to load
        """
        try:
            # Get the session's library search order
            search_order = getattr(session, "search_order", [])
            if not search_order:
                # Fallback to a reasonable default for API sessions
                if session.session_type.value == "api_testing":
                    search_order = [
                        "RequestsLibrary",
                        "BuiltIn",
                        "Collections",
                        "String",
                    ]
                else:
                    search_order = ["BuiltIn", "Collections", "String"]
                logger.debug(
                    f"Using fallback search order for session {session.session_id}: {search_order}"
                )

            logger.info(
                f"Loading libraries for session {session.session_id}: {search_order}"
            )

            # Load libraries into the dynamic keyword orchestrator's library manager
            library_manager = self.keyword_executor.keyword_discovery.library_manager
            keyword_extractor = (
                self.keyword_executor.keyword_discovery.keyword_discovery
            )

            # Load the session libraries
            library_manager.load_session_libraries(search_order, keyword_extractor)

            # Update session to reflect loaded libraries and mark as loaded
            loaded_count = 0
            for lib_name in search_order:
                if lib_name in library_manager.libraries:
                    try:
                        session.import_library(lib_name, force=False)
                        loaded_count += 1
                        logger.debug(
                            f"Imported {lib_name} into session {session.session_id}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to import {lib_name} into session: {e}")

            session.libraries_loaded = True
            logger.info(
                f"Successfully loaded {loaded_count} libraries for session {session.session_id}"
            )

        except Exception as e:
            logger.error(
                f"Error loading libraries for session {session.session_id}: {e}"
            )
            # Mark as loaded even if there were errors to prevent retry loops
            session.libraries_loaded = True

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary information about a session."""
        return self.session_manager.get_session_info(session_id)

    def get_all_sessions_info(self) -> Dict[str, Dict[str, Any]]:
        """Get summary information about all sessions."""
        return self.session_manager.get_all_sessions_info()

    @property
    def sessions(self) -> Dict[str, Any]:
        """Provide access to sessions for compatibility with TestBuilder."""
        return self.session_manager.sessions

    def cleanup_expired_sessions(self) -> int:
        """Clean up sessions that have been inactive for too long."""
        return self.session_manager.cleanup_expired_sessions()

    def check_library_requirements(
        self, required_libraries: List[str]
    ) -> Dict[str, Any]:
        """Check if required libraries are available and properly initialized with intelligent guidance."""
        # Import here to avoid circular imports
        from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery
        from robotmcp.utils.library_checker import check_and_suggest_libraries

        if not required_libraries:
            return {
                "available_libraries": [],
                "missing_libraries": [],
                "initialization_errors": [],
                "hint": "No libraries specified to check. Provide a list of library names (e.g., ['Browser', 'XML', 'SeleniumLibrary']).",
            }

        try:
            # Check if the dynamic keyword orchestrator has been initialized
            orchestrator = get_keyword_discovery()
            libraries_initialized = (
                len(orchestrator.libraries) > 3
            )  # More than just the basic core libraries
            total_libraries_loaded = len(orchestrator.libraries)

            # Use the comprehensive library checker
            available, suggestions = check_and_suggest_libraries(required_libraries)
            missing = [lib for lib in required_libraries if lib not in available]

            # Build the response with guidance
            result = {
                "available_libraries": available,
                "missing_libraries": missing,
                "installation_suggestions": suggestions,
                "initialization_errors": [],
                "libraries_initialized": libraries_initialized,
                "total_libraries_discovered": total_libraries_loaded,
            }

            # Add intelligent guidance based on the situation
            if not libraries_initialized and not available:
                result["hint"] = (
                    "Library discovery appears not to be initialized yet. "
                    "RECOMMENDED: Follow the 3-step workflow instead of calling this tool first."
                )
                result["recommended_workflow"] = [
                    "PREFERRED: 1. analyze_scenario → 2. recommend_libraries → 3. check_library_availability",
                    "FALLBACK: 1. Call 'get_available_keywords' to initialize library discovery",
                    "FALLBACK: 2. Call 'check_library_availability' again to get accurate results",
                    "FALLBACK: 3. Only install packages that are confirmed missing after initialization",
                ]
            elif not libraries_initialized and len(missing) == len(required_libraries):
                result["hint"] = (
                    "All requested libraries appear missing, but library discovery may not be complete. "
                    "RECOMMENDED: Use the proper 3-step workflow for better results."
                )
                result["recommended_workflow"] = [
                    "PREFERRED: 1. analyze_scenario → 2. recommend_libraries → 3. check_library_availability",
                    "ALTERNATIVE: 1. Run 'get_available_keywords' or 'execute_step' to initialize libraries",
                    "ALTERNATIVE: 2. Re-check library availability to get accurate results",
                    "ALTERNATIVE: 3. Install only genuinely missing packages",
                ]
            elif missing:
                result["hint"] = (
                    f"Found {len(available)} available, {len(missing)} missing. Only install the missing ones."
                )
                result["recommended_workflow"] = [
                    f"Install missing libraries: {', '.join(missing)}",
                    "Re-run this check after installation to verify",
                ]
            else:
                result["hint"] = (
                    "All requested libraries are available. No installation needed."
                )
                result["recommended_workflow"] = [
                    "All libraries ready to use",
                    "Proceed with your Robot Framework automation",
                ]

            # Add fallback check for browser libraries (maintain compatibility)
            browser_status = self.browser_library_manager.check_library_requirements(
                [
                    lib
                    for lib in required_libraries
                    if lib.lower() in ["browser", "selenium", "seleniumlibrary"]
                ]
            )
            if browser_status.get("initialization_errors"):
                result["initialization_errors"].extend(
                    browser_status["initialization_errors"]
                )

            return result

        except Exception as e:
            # Fallback to original implementation if there are issues
            logger.warning(f"Enhanced library checking failed, using fallback: {e}")
            return self.browser_library_manager.check_library_requirements(
                required_libraries
            )

    def get_library_capabilities(self) -> Dict[str, Any]:
        """Get information about available library capabilities."""
        return self.browser_library_manager.get_library_capabilities()

    def convert_locator(self, locator: str, target_library: str) -> str:
        """Convert locator format for a specific library."""
        return self.locator_converter.convert_locator_for_library(
            locator, target_library
        )

    def validate_locator(self, locator: str, library_type: str) -> Dict[str, bool]:
        """Validate locator syntax for a specific library."""
        return self.locator_converter.validate_locator(locator, library_type)

    def filter_page_source(self, html: str, filtering_level: str = "standard") -> str:
        """Filter HTML page source to keep only automation-relevant content."""
        return self.page_source_service.filter_page_source(html, filtering_level)

    def get_browser_library_status(self) -> Dict[str, Any]:
        """Get current status of browser library manager."""
        return self.browser_library_manager.get_status()

    def set_active_library(self, session_id: str, library_type: str) -> bool:
        """Set the active library for a session."""
        session = self.session_manager.get_session(session_id)
        if session:
            return self.browser_library_manager.set_active_library(
                session, library_type
            )
        return False

    def update_config(self, **config_updates) -> None:
        """Update configuration for all services."""
        try:
            self.config.update(**config_updates)

            # Update service configurations
            self.session_manager.config = self.config
            self.browser_library_manager.config = self.config
            self.page_source_service.config = self.config
            self.keyword_executor.config = self.config
            self.locator_converter.config = self.config

            logger.info(f"Configuration updated: {list(config_updates.keys())}")

        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            raise

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.to_dict()

    def _convert_locators_in_arguments(
        self, arguments: List[str], session: ExecutionSession
    ) -> List[str]:
        """Convert locators in arguments based on active library."""
        if not self.config.ENABLE_LOCATOR_CONVERSION:
            return arguments

        # Get active library for the session
        active_library = session.get_active_library()

        # Convert locators in arguments
        converted_args = []
        for arg in arguments:
            # Simple heuristic: if argument looks like a locator, convert it
            if self._looks_like_locator(arg):
                # For execution, don't add strategy prefixes (causes parsing issues)
                # Strategy prefixes will be added during test suite generation instead
                processed_arg = arg

                # Then apply library-specific conversion if needed
                if active_library:
                    converted_arg = self.locator_converter.convert_locator_for_library(
                        processed_arg, active_library.capitalize()
                    )
                    if converted_arg != processed_arg:
                        logger.debug(
                            f"Converted locator '{processed_arg}' -> '{converted_arg}' for {active_library}"
                        )
                    converted_args.append(converted_arg)
                else:
                    converted_args.append(processed_arg)
            else:
                converted_args.append(arg)

        return converted_args

    def _looks_like_locator(self, argument: str) -> bool:
        """
        Simple heuristic to determine if an argument might be a locator.

        Args:
            argument: Argument string to check

        Returns:
            bool: True if argument looks like a locator
        """
        if not argument or len(argument) < 2:
            return False

        locator_indicators = [
            argument.startswith("//"),  # XPath
            argument.startswith("#"),  # CSS ID
            argument.startswith("."),  # CSS class
            argument.startswith("["),  # CSS attribute
            "=" in argument,  # Explicit strategy
            "text=" in argument,  # Text selector
            "id=" in argument,  # ID selector
            "css=" in argument,  # CSS selector
            "xpath=" in argument,  # XPath selector
            " > " in argument,  # CSS child combinator
            ">>" in argument,  # Browser Library cascaded
        ]

        return any(locator_indicators)

    async def validate_test_readiness(self, session_id: str) -> Dict[str, Any]:
        """Check if a session is ready for test suite generation."""
        session = self.session_manager.get_session(session_id)
        if not session:
            return {
                "ready_for_suite_generation": False,
                "error": f"Session '{session_id}' not found",
                "guidance": ["Create a session and execute some steps first"],
                "validation_summary": {"passed": 0, "failed": 0},
            }

        step_count = session.step_count
        if step_count == 0:
            return {
                "ready_for_suite_generation": False,
                "error": "No steps executed in session",
                "guidance": [
                    "Execute some automation steps before building a test suite"
                ],
                "validation_summary": {"passed": 0, "failed": 0},
            }

        return {
            "ready_for_suite_generation": True,
            "validation_summary": {
                "passed": step_count,
                "failed": 0,
                "success_rate": 1.0,
            },
            "guidance": [
                f"Session has {step_count} successful steps ready for suite generation"
            ],
        }

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics across all services."""
        stats = {
            "sessions": {
                "total_sessions": self.session_manager.get_session_count(),
                "active_session_ids": self.session_manager.get_all_session_ids(),
            },
            "browser_libraries": self.browser_library_manager.get_status(),
            "locator_conversions": self.locator_converter.get_conversion_stats(),
            "configuration": {
                "locator_conversion_enabled": self.config.ENABLE_LOCATOR_CONVERSION,
                "preferred_web_library": self.config.PREFERRED_WEB_LIBRARY,
                "default_filtering_level": self.config.DEFAULT_FILTERING_LEVEL,
            },
        }

        return stats

    def reset_all_services(self) -> None:
        """Reset all services to initial state."""
        logger.info("Resetting all execution services")

        # Clean up all sessions
        self.session_manager.cleanup_all_sessions()

        # Reset browser libraries
        self.browser_library_manager.reset_libraries()

        logger.info("All execution services reset")

    def cleanup(self) -> None:
        """Clean up all resources."""
        logger.info("Cleaning up ExecutionCoordinator")

        # Clean up all sessions
        self.session_manager.cleanup_all_sessions()

        # Clean up browser libraries
        self.browser_library_manager.cleanup()

        # Clean up suite execution service
        self.suite_execution_service.cleanup_all()

        logger.info("ExecutionCoordinator cleanup completed")

    # ============================================
    # SUITE EXECUTION METHODS
    # ============================================

    async def run_suite_dry_run(
        self,
        session_id: str,
        validation_level: str = "standard",
        include_warnings: bool = True,
    ) -> Dict[str, Any]:
        """
        Run test suite in dry run mode for validation.

        Args:
            session_id: Session with generated test suite
            validation_level: Validation depth ('minimal', 'standard', 'strict')
            include_warnings: Include warnings in validation report

        Returns:
            Structured validation results
        """
        try:
            # Get session
            session = self.session_manager.get_session(session_id)
            if not session:
                return {
                    "success": False,
                    "error": f"Session '{session_id}' not found",
                    "tool": "run_test_suite_dry",
                    "session_id": session_id,
                }

            # Generate suite content using TestBuilder
            from robotmcp.components.test_builder import TestBuilder

            test_builder = TestBuilder(self)

            suite_result = await test_builder.build_suite(
                session_id=session_id,
                test_name=f"DryRun_Test_{session_id}",
                remove_library_prefixes=True,
            )

            if not suite_result.get("success", False):
                return {
                    "success": False,
                    "error": f"Failed to generate test suite: {suite_result.get('error', 'Unknown error')}",
                    "tool": "run_test_suite_dry",
                    "session_id": session_id,
                }

            # Get suite content
            suite_content = suite_result["rf_text"]

            # Execute dry run using suite execution service
            options = {
                "validation_level": validation_level,
                "include_warnings": include_warnings,
            }

            result = await self.suite_execution_service.execute_dry_run(
                suite_content, session_id, options
            )

            return result

        except Exception as e:
            logger.error(f"Error in run_suite_dry_run for session {session_id}: {e}")
            return {
                "success": False,
                "error": f"Dry run execution failed: {str(e)}",
                "tool": "run_test_suite_dry",
                "session_id": session_id,
            }

    async def run_suite_dry_run_from_file(
        self,
        suite_file_path: str,
        validation_level: str = "standard",
        include_warnings: bool = True,
    ) -> Dict[str, Any]:
        """
        Run test suite from file in dry run mode for validation.

        Args:
            suite_file_path: Path to .robot file
            validation_level: Validation depth ('minimal', 'standard', 'strict')
            include_warnings: Include warnings in validation report

        Returns:
            Structured validation results
        """
        try:
            # Read suite content from file
            with open(suite_file_path, "r", encoding="utf-8") as f:
                suite_content = f.read()

            # Execute dry run using suite execution service
            options = {
                "validation_level": validation_level,
                "include_warnings": include_warnings,
            }

            result = await self.suite_execution_service.execute_dry_run(
                suite_content, f"file_{os.path.basename(suite_file_path)}", options
            )

            # Update result with file information
            result["suite_file_path"] = suite_file_path
            result["tool"] = "run_test_suite_dry"

            return result

        except Exception as e:
            logger.error(
                f"Error in run_suite_dry_run_from_file for {suite_file_path}: {e}"
            )
            return {
                "success": False,
                "error": f"Dry run execution failed: {str(e)}",
                "tool": "run_test_suite_dry",
                "suite_file_path": suite_file_path,
            }

    async def run_suite_execution(
        self,
        session_id: str,
        execution_options: Dict[str, Any] = None,
        output_level: str = "standard",
        capture_screenshots: bool = False,
    ) -> Dict[str, Any]:
        """
        Run test suite in normal execution mode.

        Args:
            session_id: Session with generated test suite
            execution_options: Dict with RF options (tags, variables, etc.)
            output_level: Response verbosity ('minimal', 'standard', 'detailed')
            capture_screenshots: Enable screenshot capture on failures

        Returns:
            Comprehensive execution results
        """
        try:
            if execution_options is None:
                execution_options = {}

            # Get session
            session = self.session_manager.get_session(session_id)
            if not session:
                return {
                    "success": False,
                    "error": f"Session '{session_id}' not found",
                    "tool": "run_test_suite",
                    "session_id": session_id,
                }

            # Generate suite content using TestBuilder
            from robotmcp.components.test_builder import TestBuilder

            test_builder = TestBuilder(self)

            suite_result = await test_builder.build_suite(
                session_id=session_id,
                test_name=f"Execution_Test_{session_id}",
                remove_library_prefixes=True,
            )

            if not suite_result.get("success", False):
                return {
                    "success": False,
                    "error": f"Failed to generate test suite: {suite_result.get('error', 'Unknown error')}",
                    "tool": "run_test_suite",
                    "session_id": session_id,
                }

            # Get suite content
            suite_content = suite_result["rf_text"]

            # Prepare execution options
            options = execution_options.copy()
            options.update(
                {
                    "output_level": output_level,
                    "capture_screenshots": capture_screenshots,
                }
            )

            # Execute suite using suite execution service
            result = await self.suite_execution_service.execute_normal(
                suite_content, session_id, options
            )

            return result

        except Exception as e:
            logger.error(f"Error in run_suite_execution for session {session_id}: {e}")
            return {
                "success": False,
                "error": f"Suite execution failed: {str(e)}",
                "tool": "run_test_suite",
                "session_id": session_id,
            }

    async def run_suite_execution_from_file(
        self,
        suite_file_path: str,
        execution_options: Dict[str, Any] = None,
        output_level: str = "standard",
        capture_screenshots: bool = False,
    ) -> Dict[str, Any]:
        """
        Run test suite from file in normal execution mode.

        Args:
            suite_file_path: Path to .robot file
            execution_options: Dict with RF options (tags, variables, etc.)
            output_level: Response verbosity ('minimal', 'standard', 'detailed')
            capture_screenshots: Enable screenshot capture on failures

        Returns:
            Comprehensive execution results
        """
        try:
            if execution_options is None:
                execution_options = {}

            # Read suite content from file
            with open(suite_file_path, "r", encoding="utf-8") as f:
                suite_content = f.read()

            # Prepare execution options
            options = execution_options.copy()
            options.update(
                {
                    "output_level": output_level,
                    "capture_screenshots": capture_screenshots,
                }
            )

            # Execute suite using suite execution service
            result = await self.suite_execution_service.execute_normal(
                suite_content, f"file_{os.path.basename(suite_file_path)}", options
            )

            # Update result with file information
            result["suite_file_path"] = suite_file_path
            result["tool"] = "run_test_suite"

            return result

        except Exception as e:
            logger.error(
                f"Error in run_suite_execution_from_file for {suite_file_path}: {e}"
            )
            return {
                "success": False,
                "error": f"Suite execution failed: {str(e)}",
                "tool": "run_test_suite",
                "suite_file_path": suite_file_path,
            }

    # ============================================
    # MISSING METHODS FOR BACKWARD COMPATIBILITY
    # ============================================

    def get_available_keywords(self, library_name: str = None) -> List[Dict[str, Any]]:
        """Get list of available keywords, optionally filtered by library.

        CRITICAL FIX: Now uses LibraryManager as source of truth, with LibDoc for enrichment.
        This fixes the synchronization issue between static LibDoc and dynamic LibraryManager loading.
        """
        # CRITICAL FIX: Use LibraryManager as primary source since it reflects current loaded state
        keyword_discovery = self.keyword_executor.keyword_discovery

        if library_name:
            # Validate library is loaded before filtering
            if library_name not in keyword_discovery.libraries:
                logger.warning(
                    f"Library '{library_name}' not loaded, attempting on-demand loading"
                )

                # Try to load the library
                if not keyword_discovery.library_manager.load_library_on_demand(
                    library_name, keyword_discovery.keyword_discovery
                ):
                    logger.error(f"Failed to load library '{library_name}' on demand")
                    return []  # Return empty list if library can't be loaded

            # Get keywords from specific library using LibraryManager
            keywords_from_lib = keyword_discovery.get_keywords_by_library(library_name)
            result = []

            for keyword_info in keywords_from_lib:
                # CRITICAL FIX: Enhanced MCP serialization with null-safety and validation
                keyword_dict = {
                    "name": str(keyword_info.name if keyword_info.name else ""),
                    "library": str(
                        keyword_info.library if keyword_info.library else ""
                    ),
                    "args": [
                        str(arg)
                        for arg in (keyword_info.args if keyword_info.args else [])
                    ],
                    "short_doc": str(
                        keyword_info.short_doc if keyword_info.short_doc else ""
                    ),
                    "tags": [
                        str(tag)
                        for tag in (keyword_info.tags if keyword_info.tags else [])
                    ],
                    "is_builtin": bool(getattr(keyword_info, "is_builtin", False)),
                }

                # CRITICAL FIX: Validate no complex objects remain
                for key, value in keyword_dict.items():
                    if not isinstance(value, (str, int, bool, list)):
                        logger.warning(
                            f"Non-serializable value in keyword_dict[{key}]: {type(value)}"
                        )
                        keyword_dict[key] = str(value)  # Force string conversion
                    elif isinstance(value, list):
                        # Ensure list elements are also serializable
                        keyword_dict[key] = [str(item) for item in value]

                # Enrich with LibDoc data if available
                if self.rf_doc_storage.is_available():
                    try:
                        libdoc_keywords = self.rf_doc_storage.get_keywords_by_library(
                            library_name
                        )
                        for libdoc_kw in libdoc_keywords:
                            if libdoc_kw.name == keyword_info.name:
                                keyword_dict.update(
                                    {
                                        "is_deprecated": libdoc_kw.is_deprecated,
                                        "arg_types": libdoc_kw.arg_types,
                                    }
                                )
                                break
                    except Exception:
                        pass  # LibDoc enrichment is optional

                result.append(keyword_dict)

            logger.info(
                f"Retrieved {len(result)} keywords for library '{library_name}'"
            )
            return result
        else:
            # Get all keywords from all loaded libraries using LibraryManager
            keywords = []
            for keyword_info in keyword_discovery.get_all_keywords():
                # CRITICAL FIX: Enhanced MCP serialization with null-safety and validation
                keyword_dict = {
                    "name": str(keyword_info.name if keyword_info.name else ""),
                    "library": str(
                        keyword_info.library if keyword_info.library else ""
                    ),
                    "args": [
                        str(arg)
                        for arg in (keyword_info.args if keyword_info.args else [])
                    ],
                    "short_doc": str(
                        keyword_info.short_doc if keyword_info.short_doc else ""
                    ),
                    "tags": [
                        str(tag)
                        for tag in (keyword_info.tags if keyword_info.tags else [])
                    ],
                    "is_builtin": bool(getattr(keyword_info, "is_builtin", False)),
                }

                # CRITICAL FIX: Validate no complex objects remain
                for key, value in keyword_dict.items():
                    if not isinstance(value, (str, int, bool, list)):
                        logger.warning(
                            f"Non-serializable value in keyword_dict[{key}]: {type(value)}"
                        )
                        keyword_dict[key] = str(value)  # Force string conversion
                    elif isinstance(value, list):
                        # Ensure list elements are also serializable
                        keyword_dict[key] = [str(item) for item in value]

                # ENHANCEMENT: Try to enrich with LibDoc data if available
                if self.rf_doc_storage.is_available():
                    try:
                        libdoc_keywords = self.rf_doc_storage.get_keywords_by_library(
                            keyword_info.library
                        )
                        for libdoc_kw in libdoc_keywords:
                            if libdoc_kw.name == keyword_info.name:
                                keyword_dict.update(
                                    {
                                        "is_deprecated": libdoc_kw.is_deprecated,
                                        "arg_types": libdoc_kw.arg_types,
                                    }
                                )
                                break
                    except Exception:
                        pass  # LibDoc enrichment is optional

                keywords.append(keyword_dict)

            return keywords

    def search_keywords(self, pattern: str) -> List[Dict[str, Any]]:
        """Search for keywords matching a pattern using native RF libdoc when available."""
        # Use libdoc-based search if available, otherwise fall back to inspection-based
        if self.rf_doc_storage.is_available():
            matches = self.rf_doc_storage.search_keywords(pattern)
            return [
                {
                    "name": kw.name,
                    "library": kw.library,
                    "args": kw.args,
                    "short_doc": kw.short_doc,
                    "tags": kw.tags,
                    "is_deprecated": kw.is_deprecated,
                    "arg_types": kw.arg_types,
                }
                for kw in matches
            ]
        else:
            # Fall back to inspection-based search via keyword_executor
            matches = self.keyword_executor.keyword_discovery.search_keywords(pattern)
            return [
                {
                    "name": kw.name,
                    "library": kw.library,
                    "args": kw.args,
                    "short_doc": kw.short_doc,
                    "tags": kw.tags,
                    "is_builtin": getattr(kw, "is_builtin", False),
                }
                for kw in matches
            ]

    def get_keyword_documentation(
        self, keyword_name: str, library_name: str = None
    ) -> Dict[str, Any]:
        """Get keyword documentation using RF libdoc with strict library filtering.

        - If library_name is provided: restrict search to that library only. No cross-library fallback.
        - If library_name is None: return all matches across libraries in a 'matches' array.
        """
        # LibDoc preferred path
        if self.rf_doc_storage.is_available():
            if library_name:
                kw = self.rf_doc_storage.get_keyword_documentation(keyword_name, library_name)
                if kw:
                    return {
                        "success": True,
                        "keyword": {
                            "name": kw.name,
                            "library": kw.library,
                            "args": kw.args,
                            "arg_types": kw.arg_types,
                            "doc": kw.doc,
                            "short_doc": kw.short_doc,
                            "tags": kw.tags,
                            "is_deprecated": kw.is_deprecated,
                            "source": kw.source,
                            "lineno": kw.lineno,
                        },
                    }
                # Not found in the requested library: no cross-library fallback
                # Provide suggestions from that library via fuzzy match
                suggestions = []
                try:
                    lib_keywords = self.rf_doc_storage.get_keywords_by_library(library_name)
                    # simple fuzzy: match words ignoring case/underscores/spaces
                    target = keyword_name.lower().replace('_', ' ').strip()
                    for k in lib_keywords:
                        name_norm = k.name.lower().replace('_', ' ').strip()
                        if target in name_norm or name_norm in target or k.name.lower().startswith(keyword_name.lower()[0:3]):
                            suggestions.append(k.name)
                except Exception:
                    pass
                return {
                    "success": False,
                    "error": f"Keyword '{keyword_name}' not found in library '{library_name}'",
                    "suggestions": sorted(list(set(suggestions)))[:5],
                }
            else:
                # Return all matches across libraries
                matches = self.rf_doc_storage.get_keywords_documentation_all(keyword_name)
                if matches:
                    return {
                        "success": True,
                        "matches": [
                            {
                                "name": m.name,
                                "library": m.library,
                                "args": m.args,
                                "arg_types": m.arg_types,
                                "doc": m.doc,
                                "short_doc": m.short_doc,
                                "tags": m.tags,
                                "is_deprecated": m.is_deprecated,
                                "source": m.source,
                                "lineno": m.lineno,
                            }
                            for m in matches
                        ],
                    }
                # No matches anywhere
                return {"success": False, "error": f"Keyword '{keyword_name}' not found in any loaded library"}

        # Inspection-based fallback (no LibDoc available)
        keyword_discovery = self.keyword_executor.keyword_discovery
        if library_name:
            # Strict filter: use library_prefix to restrict search
            ki = keyword_discovery.find_keyword(keyword_name, active_library=library_name)
            if ki and ki.library == library_name:
                return {
                    "success": True,
                    "keyword": {
                        "name": ki.name,
                        "library": ki.library,
                        "args": ki.args,
                        "arg_types": getattr(ki, "arg_types", []),
                        "doc": getattr(ki, "doc", ""),
                        "short_doc": ki.short_doc,
                        "tags": ki.tags,
                        "is_deprecated": False,
                        "source": getattr(ki, "source", ""),
                        "lineno": getattr(ki, "lineno", 0),
                    },
                }
            return {"success": False, "error": f"Keyword '{keyword_name}' not found in library '{library_name}'"}
        else:
            # No library specified: return first match for backward compatibility
            ki = keyword_discovery.find_keyword(keyword_name)
            if ki:
                return {
                    "success": True,
                    "keyword": {
                        "name": ki.name,
                        "library": ki.library,
                        "args": ki.args,
                        "arg_types": getattr(ki, "arg_types", []),
                        "doc": getattr(ki, "doc", ""),
                        "short_doc": ki.short_doc,
                        "tags": ki.tags,
                        "is_deprecated": False,
                        "source": getattr(ki, "source", ""),
                        "lineno": getattr(ki, "lineno", 0),
                    },
                }
            return {"success": False, "error": f"Keyword '{keyword_name}' not found"}

    def get_library_documentation(self, library_name: str) -> Dict[str, Any]:
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
              - keywords: List of all library keywords with full details
        """
        # Try libdoc-based lookup first
        if self.rf_doc_storage.is_available():
            library_info = self.rf_doc_storage.get_library_documentation(library_name)
            
            if library_info:
                # Convert keyword dict to list for API consistency
                keywords_list = []
                for keyword_info in library_info.keywords.values():
                    keywords_list.append({
                        "name": keyword_info.name,
                        "library": keyword_info.library,
                        "args": keyword_info.args,
                        "arg_types": keyword_info.arg_types,
                        "doc": keyword_info.doc,
                        "short_doc": keyword_info.short_doc,
                        "tags": keyword_info.tags,
                        "is_deprecated": keyword_info.is_deprecated,
                        "source": keyword_info.source,
                        "lineno": keyword_info.lineno,
                    })
                
                return {
                    "success": True,
                    "library": {
                        "name": library_info.name,
                        "doc": library_info.doc,
                        "version": library_info.version,
                        "type": library_info.type,
                        "scope": library_info.scope,
                        "source": library_info.source,
                        "keywords": keywords_list,
                        "keyword_count": len(keywords_list),
                        "data_source": "libdoc"
                    }
                }
        
        # Fall back to inspection-based search via keyword_executor
        keyword_discovery = self.keyword_executor.keyword_discovery
        if library_name in keyword_discovery.libraries:
            library_obj = keyword_discovery.libraries[library_name]
            
            # Get all keywords for this library from inspection-based discovery
            keywords_list = []
            for keyword_info in keyword_discovery.get_keywords_by_library(library_name):
                keywords_list.append({
                    "name": keyword_info.name,
                    "library": keyword_info.library,
                    "args": keyword_info.args,
                    "arg_types": getattr(keyword_info, "arg_types", []),
                    "doc": getattr(keyword_info, "doc", ""),
                    "short_doc": keyword_info.short_doc,
                    "tags": keyword_info.tags,
                    "is_deprecated": False,
                    "source": getattr(keyword_info, "source", ""),
                    "lineno": getattr(keyword_info, "lineno", 0),
                })
            
            # Extract library metadata from the library object
            library_doc = getattr(library_obj, '__doc__', '') or ''
            library_version = getattr(library_obj, 'ROBOT_LIBRARY_VERSION', 'Unknown')
            library_scope = getattr(library_obj, 'ROBOT_LIBRARY_SCOPE', 'TEST')
            
            return {
                "success": True,
                "library": {
                    "name": library_name,
                    "doc": library_doc,
                    "version": library_version,
                    "type": "LIBRARY",
                    "scope": library_scope,
                    "source": getattr(library_obj, '__file__', ''),
                    "keywords": keywords_list,
                    "keyword_count": len(keywords_list),
                    "data_source": "inspection"
                }
            }
        else:
            return {"success": False, "error": f"Library '{library_name}' not found or not loaded"}

    def get_installation_status(self, library_name: str) -> Dict[str, Any]:
        """Get detailed installation status for a specific library."""
        from robotmcp.utils.library_checker import COMMON_ROBOT_LIBRARIES

        if library_name in COMMON_ROBOT_LIBRARIES:
            lib_info = COMMON_ROBOT_LIBRARIES[library_name]
            package_name = lib_info["package"]
            import_name = lib_info["import"]
            is_builtin = lib_info.get("is_builtin", False)

            # Check import availability
            import_available = self.library_checker.is_library_available(
                package_name, import_name
            )

            # For built-in libraries, check if they're actually loaded in our system
            is_loaded = (
                library_name in self.keyword_executor.keyword_discovery.libraries
            )

            if is_builtin:
                # Built-in libraries are always "pip installed" with robotframework
                pip_installed = True

                status = {
                    "library_name": library_name,
                    "package_name": package_name,
                    "import_name": import_name,
                    "is_builtin": True,
                    "import_available": import_available,
                    "pip_installed": True,
                    "is_loaded": is_loaded,
                    "status": "available"
                    if (import_available and is_loaded)
                    else "not_available",
                    "installation_command": "Built-in with Robot Framework",
                    "description": lib_info.get("description", ""),
                }

                # Add specific guidance for built-in libraries
                if not import_available:
                    status["message"] = (
                        f"Built-in library {library_name} cannot be imported. Check Robot Framework installation."
                    )
                elif not is_loaded:
                    status["message"] = (
                        f"Built-in library {library_name} is available but not loaded in the current session."
                    )
                else:
                    keyword_count = len(
                        self.keyword_executor.keyword_discovery.get_keywords_by_library(
                            library_name
                        )
                    )
                    status["message"] = (
                        f"Built-in library {library_name} is loaded and ready to use ({keyword_count} keywords available)."
                    )
                    status["keyword_count"] = keyword_count

            else:
                # External libraries need pip installation
                pip_installed = self.library_checker.check_pip_package_installed(
                    package_name
                )

                status = {
                    "library_name": library_name,
                    "package_name": package_name,
                    "import_name": import_name,
                    "is_builtin": False,
                    "import_available": import_available,
                    "pip_installed": pip_installed,
                    "is_loaded": is_loaded,
                    "status": "available" if import_available else "not_available",
                    "installation_command": f"pip install {package_name}",
                    "description": lib_info.get("description", ""),
                }

                # Add keyword count if library is loaded (regardless of pip status)
                if is_loaded:
                    keyword_count = len(
                        self.keyword_executor.keyword_discovery.get_keywords_by_library(
                            library_name
                        )
                    )
                    status["keyword_count"] = keyword_count

                # Add specific guidance for external libraries
                if not import_available:
                    status["message"] = (
                        f"Package {package_name} is not available. Install with: pip install {package_name}"
                    )
                elif not is_loaded:
                    status["message"] = (
                        f"Library {library_name} is available but not loaded in the current session."
                    )
                else:
                    # Library is loaded and working
                    keyword_count = status["keyword_count"]
                    if pip_installed:
                        status["message"] = (
                            f"Library {library_name} is installed and loaded ({keyword_count} keywords available)."
                        )
                    else:
                        # Library is working despite pip check failing (common in virtual environments)
                        status["message"] = (
                            f"Library {library_name} is loaded and working ({keyword_count} keywords available). Pip status unclear."
                        )

                # Add post-install commands if specified
                if lib_info.get("post_install") and import_available and is_loaded:
                    status["post_install"] = lib_info["post_install"]
                    if "Note:" not in status["message"]:
                        status["message"] += (
                            f" Note: Run '{lib_info['post_install']}' for full functionality."
                        )

            return status
        else:
            # Check if it's a loaded library we don't recognize
            is_loaded = (
                library_name in self.keyword_executor.keyword_discovery.libraries
            )
            if is_loaded:
                keyword_count = len(
                    self.keyword_executor.keyword_discovery.get_keywords_by_library(
                        library_name
                    )
                )
                return {
                    "library_name": library_name,
                    "status": "available",
                    "is_loaded": True,
                    "keyword_count": keyword_count,
                    "message": f"Library {library_name} is loaded and available ({keyword_count} keywords). Not in standard registry.",
                }
            else:
                return {
                    "library_name": library_name,
                    "status": "unknown",
                    "message": f"Unknown library '{library_name}'. Cannot determine installation status.",
                }

    def get_library_status(self) -> Dict[str, Any]:
        """Get status of all loaded libraries including libdoc availability."""
        # Get inspection-based status via keyword_executor
        inspection_status = self.keyword_executor.keyword_discovery.get_library_status()

        # Get libdoc-based status if available
        if self.rf_doc_storage.is_available():
            libdoc_status = self.rf_doc_storage.get_library_status()
            return {
                "inspection_based": inspection_status,
                "libdoc_based": libdoc_status,
                "preferred_source": "libdoc"
                if libdoc_status["libdoc_available"]
                else "inspection",
            }
        else:
            return {
                "inspection_based": inspection_status,
                "libdoc_based": {"libdoc_available": False},
                "preferred_source": "inspection",
            }

    def get_session_validation_status(
        self, session_id: str = "default"
    ) -> Dict[str, Any]:
        """Get validation status of steps in a session."""
        session = self.session_manager.get_session(session_id)
        if not session:
            return {"success": False, "error": f"Session '{session_id}' not found"}

        validated_steps = []
        failed_steps = []  # Will always be empty since failed steps are not stored

        # Since we only store successful steps now, all steps are validated
        for step in session.steps:
            step_info = {
                "step_id": step.step_id,
                "keyword": step.keyword,
                "arguments": step.arguments,
                "status": step.status,
                "validated": True,  # All stored steps are considered validated
                "execution_time": step.execution_time,
            }
            validated_steps.append(step_info)

        total_steps = len(validated_steps)
        passed_steps = len([s for s in validated_steps if s["status"] == "pass"])

        return {
            "success": True,
            "session_id": session_id,
            "total_steps": total_steps,
            "validated_steps": passed_steps,
            "failed_steps": 0,  # Failed steps are not stored
            "validation_rate": 100.0
            if total_steps > 0
            else 0.0,  # All stored steps are validated
            "steps": validated_steps,
            "ready_for_test_suite": total_steps > 0,
            "message": f"Session has {total_steps} validated steps ready for test suite generation"
            if total_steps > 0
            else "Session has no validated steps yet",
        }

    def get_session_browser_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get browser status for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary containing browser status information
        """
        try:
            session = self.session_manager.get_session(session_id)
            if not session:
                return {"error": f"Session {session_id} not found"}

            browser_state = session.browser_state

            return {
                "success": True,
                "session_id": session_id,
                "current_url": browser_state.current_url,
                "page_title": browser_state.page_title,
                "browser_library": {
                    "browser_id": browser_state.browser_id,
                    "context_id": browser_state.context_id,
                    "page_id": browser_state.page_id,
                    "library_type": browser_state.active_library,
                },
                "viewport": {
                    "width": getattr(browser_state, "viewport_width", 1280),
                    "height": getattr(browser_state, "viewport_height", 720),
                },
            }

        except Exception as e:
            logger.error(f"Error getting browser status for session {session_id}: {e}")
            return {"error": str(e)}

    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """
        Get complete session state including browser and variables.

        Args:
            session_id: Session identifier

        Returns:
            Complete session state information
        """
        try:
            session = self.session_manager.get_session(session_id)
            if not session:
                return {"error": f"Session {session_id} not found"}

            return {
                "success": True,
                "session_id": session_id,
                "browser_state": {
                    "current_url": session.browser_state.current_url,
                    "page_title": session.browser_state.page_title,
                    "browser_id": session.browser_state.browser_id,
                    "context_id": session.browser_state.context_id,
                    "page_id": session.browser_state.page_id,
                    "active_library": session.browser_state.active_library,
                    "page_source": session.browser_state.page_source,
                },
                "variables": dict(session.variables),
                "execution_history": [
                    {
                        "step_number": i + 1,
                        "keyword": step.keyword,
                        "status": step.status,
                        "execution_time": step.execution_time,
                    }
                    for i, step in enumerate(session.steps)
                ],
            }

        except Exception as e:
            logger.error(f"Error getting session state for {session_id}: {e}")
            return {"error": str(e)}

    # ============================================
    # BACKWARD COMPATIBILITY PROPERTIES
    # ============================================

    @property
    def sessions(self) -> Dict[str, ExecutionSession]:
        """Access to sessions for backward compatibility."""
        return self.session_manager.sessions

    @property
    def keyword_discovery(self):
        """Access to keyword discovery for backward compatibility."""
        return self.keyword_executor.keyword_discovery

    def _convert_locator_for_library(self, locator: str, target_library: str) -> str:
        """Backward compatibility method for TestBuilder."""
        return self.locator_converter.convert_locator_for_library(
            locator, target_library
        )

    # ============================================
    # CONTEXT MANAGER
    # ============================================

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

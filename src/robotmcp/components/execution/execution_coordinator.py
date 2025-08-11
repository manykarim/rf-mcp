"""Main execution coordinator orchestrating all execution services."""

import logging
from typing import Any, Dict, List, Optional

from robotmcp.models.session_models import ExecutionSession
from robotmcp.models.config_models import ExecutionConfig
from robotmcp.components.browser import BrowserLibraryManager
from robotmcp.components.execution import SessionManager, PageSourceService, KeywordExecutor, LocatorConverter

logger = logging.getLogger(__name__)


class ExecutionCoordinator:
    """
    Main coordinator that orchestrates all execution services.
    
    This class replaces the original monolithic execution engine with a clean,
    service-oriented architecture that separates concerns and improves maintainability.
    """
    
    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        
        # Initialize all service components
        self.session_manager = SessionManager(self.config)
        self.browser_library_manager = BrowserLibraryManager(self.config)
        self.page_source_service = PageSourceService(self.config)
        self.keyword_executor = KeywordExecutor(self.config)
        self.locator_converter = LocatorConverter(self.config)
        
        logger.info("ExecutionCoordinator initialized with service-oriented architecture")
    
    async def execute_step(
        self,
        keyword: str,
        arguments: List[str] = None,
        session_id: str = "default",
        detail_level: str = "minimal"
    ) -> Dict[str, Any]:
        """
        Execute a single Robot Framework keyword step.
        
        Args:
            keyword: Robot Framework keyword name
            arguments: List of arguments for the keyword
            session_id: Session identifier
            detail_level: Level of detail in response ('minimal', 'standard', 'full')
            
        Returns:
            Execution result with status, output, and state
        """
        try:
            if arguments is None:
                arguments = []
            
            # Get or create session
            session = self.session_manager.get_or_create_session(session_id)
            
            # Convert locators if needed
            converted_arguments = self._convert_locators_in_arguments(arguments, session)
            
            # Execute the keyword using the keyword executor
            result = await self.keyword_executor.execute_keyword(
                session=session,
                keyword=keyword,
                arguments=converted_arguments,
                browser_library_manager=self.browser_library_manager,
                detail_level=detail_level
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ExecutionCoordinator.execute_step: {e}")
            return {
                "success": False,
                "error": f"Execution coordinator error: {str(e)}",
                "keyword": keyword,
                "arguments": arguments or [],
                "session_id": session_id
            }
    
    async def get_page_source(
        self, 
        session_id: str = "default", 
        full_source: bool = False, 
        filtered: bool = False, 
        filtering_level: str = "standard"
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
                return {
                    "success": False,
                    "error": f"Session '{session_id}' not found"
                }
            
            return await self.page_source_service.get_page_source(
                session=session,
                browser_library_manager=self.browser_library_manager,
                full_source=full_source,
                filtered=filtered,
                filtering_level=filtering_level
            )
            
        except Exception as e:
            logger.error(f"Error getting page source for session {session_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to get page source: {str(e)}"
            }
    
    def create_session(self, session_id: str) -> ExecutionSession:
        """Create a new execution session."""
        return self.session_manager.create_session(session_id)
    
    def get_session(self, session_id: str) -> Optional[ExecutionSession]:
        """Get an existing session by ID."""
        return self.session_manager.get_session(session_id)
    
    def remove_session(self, session_id: str) -> bool:
        """Remove a session."""
        return self.session_manager.remove_session(session_id)
    
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
    
    def check_library_requirements(self, required_libraries: List[str]) -> Dict[str, Any]:
        """Check if required libraries are available and properly initialized."""
        return self.browser_library_manager.check_library_requirements(required_libraries)
    
    def get_library_capabilities(self) -> Dict[str, Any]:
        """Get information about available library capabilities."""
        return self.browser_library_manager.get_library_capabilities()
    
    def convert_locator(self, locator: str, target_library: str) -> str:
        """Convert locator format for a specific library."""
        return self.locator_converter.convert_locator_for_library(locator, target_library)
    
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
            return self.browser_library_manager.set_active_library(session, library_type)
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
    
    def _convert_locators_in_arguments(self, arguments: List[str], session: ExecutionSession) -> List[str]:
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
                        logger.debug(f"Converted locator '{processed_arg}' -> '{converted_arg}' for {active_library}")
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
            argument.startswith("//"),        # XPath
            argument.startswith("#"),         # CSS ID
            argument.startswith("."),         # CSS class
            argument.startswith("["),         # CSS attribute
            "=" in argument,                  # Explicit strategy
            "text=" in argument,              # Text selector
            "id=" in argument,                # ID selector
            "css=" in argument,               # CSS selector
            "xpath=" in argument,             # XPath selector
            " > " in argument,                # CSS child combinator
            ">>" in argument,                 # Browser Library cascaded
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
                "validation_summary": {"passed": 0, "failed": 0}
            }
        
        step_count = session.step_count
        if step_count == 0:
            return {
                "ready_for_suite_generation": False,
                "error": "No steps executed in session",
                "guidance": ["Execute some automation steps before building a test suite"],
                "validation_summary": {"passed": 0, "failed": 0}
            }
        
        return {
            "ready_for_suite_generation": True,
            "validation_summary": {
                "passed": step_count,
                "failed": 0,
                "success_rate": 1.0
            },
            "guidance": [f"Session has {step_count} successful steps ready for suite generation"]
        }

    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics across all services."""
        stats = {
            "sessions": {
                "total_sessions": self.session_manager.get_session_count(),
                "active_session_ids": self.session_manager.get_all_session_ids()
            },
            "browser_libraries": self.browser_library_manager.get_status(),
            "locator_conversions": self.locator_converter.get_conversion_stats(),
            "configuration": {
                "locator_conversion_enabled": self.config.ENABLE_LOCATOR_CONVERSION,
                "preferred_web_library": self.config.PREFERRED_WEB_LIBRARY,
                "default_filtering_level": self.config.DEFAULT_FILTERING_LEVEL
            }
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
        
        logger.info("ExecutionCoordinator cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
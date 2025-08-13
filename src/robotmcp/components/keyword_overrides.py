"""Keyword execution override system for custom handling of specific keywords."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable, Protocol
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# Import ExecutionSession from execution_engine (will be resolved at runtime)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .execution_engine import ExecutionSession

@dataclass
class OverrideResult:
    """Result from a keyword override execution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    state_updates: Optional[Dict[str, Any]] = None
    variables: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class KeywordOverrideHandler(Protocol):
    """Protocol for keyword override handlers."""
    
    async def execute(
        self, 
        session: 'ExecutionSession', 
        keyword: str, 
        args: List[str],
        keyword_info: Optional[Any] = None
    ) -> OverrideResult:
        """Execute the keyword override."""
        ...

class KeywordOverrideRegistry:
    """Registry for keyword override handlers."""
    
    def __init__(self):
        self.overrides: Dict[str, KeywordOverrideHandler] = {}
        self.pattern_overrides: Dict[str, KeywordOverrideHandler] = {}
        self.library_overrides: Dict[str, KeywordOverrideHandler] = {}
        
    def register_keyword(self, keyword_name: str, handler: KeywordOverrideHandler):
        """Register an override for a specific keyword name."""
        normalized_name = keyword_name.lower().strip()
        self.overrides[normalized_name] = handler
        logger.info(f"Registered keyword override: {keyword_name}")
        
    def register_pattern(self, pattern: str, handler: KeywordOverrideHandler):
        """Register an override for a keyword pattern (e.g., 'Browser.*')."""
        self.pattern_overrides[pattern.lower()] = handler
        logger.info(f"Registered pattern override: {pattern}")
        
    def register_library(self, library_name: str, handler: KeywordOverrideHandler):
        """Register an override for all keywords from a library."""
        self.library_overrides[library_name.lower()] = handler
        logger.info(f"Registered library override: {library_name}")
        
    def get_override(self, keyword_name: str, library_name: str = None) -> Optional[KeywordOverrideHandler]:
        """Get the appropriate override handler for a keyword."""
        normalized_keyword = keyword_name.lower().strip()
        
        # 1. Check exact keyword match first
        if normalized_keyword in self.overrides:
            return self.overrides[normalized_keyword]
            
        # 2. Check library-specific overrides (only if library matches exactly)
        if library_name:
            library_key = library_name.lower()
            if library_key in self.library_overrides:
                handler = self.library_overrides[library_key]
                # Verify the handler is appropriate for this specific library
                if self._handler_supports_library(handler, library_key):
                    return handler
                
        # 3. Check pattern matches
        for pattern, handler in self.pattern_overrides.items():
            if self._matches_pattern(normalized_keyword, pattern, library_name):
                return handler
                
        return None
    
    def _handler_supports_library(self, handler: KeywordOverrideHandler, library_key: str) -> bool:
        """Check if a handler is appropriate for a specific library."""
        # Check handler type based on class name
        handler_class_name = handler.__class__.__name__.lower()
        
        # Browser library handler should only handle Browser library keywords
        if 'browser' in handler_class_name and library_key == 'browser':
            return True
        
        # Selenium library handler should only handle SeleniumLibrary keywords    
        if 'selenium' in handler_class_name and library_key in ['seleniumlibrary', 'selenium']:
            return True
            
        return False
        
    def _matches_pattern(self, keyword: str, pattern: str, library: str = None) -> bool:
        """Check if a keyword matches a pattern."""
        # Simple pattern matching - can be enhanced later
        if pattern.endswith('.*'):
            prefix = pattern[:-2]
            if library and prefix == library.lower():
                return True
            if keyword.startswith(prefix):
                return True
        return pattern in keyword

# Unified Base Handler
class UnifiedLibraryHandler:
    """Base class for consistent library override handling using keyword discovery."""
    
    def __init__(self, execution_engine, supported_library: str):
        self.execution_engine = execution_engine
        self.supported_library = supported_library.lower()
    
    # Implement the protocol
    def __call__(self):
        return self
    
    async def execute(self, session, keyword, args, keyword_info):
        """Unified execution approach using keyword discovery with library prefix."""
        try:
            # Get keyword discovery orchestrator
            from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery
            orchestrator = get_keyword_discovery()
            
            # Execute with explicit library specification to prevent wrong routing
            # Using library_prefix ensures we bypass override system recursion
            result = await orchestrator.execute_keyword(
                keyword_name=keyword,
                args=args,
                session_variables=session.variables,
                active_library=keyword_info.library if keyword_info else None,
                session_id=session.session_id,
                library_prefix=keyword_info.library if keyword_info else self.supported_library
            )
            
            # Apply library-specific state updates
            if result.get("success"):
                await self._handle_success_state_updates(session, keyword, result, keyword_info)
            
            return OverrideResult(
                success=result.get("success", False),
                output=result.get("output"),
                error=result.get("error"),
                state_updates=self._get_library_state_updates(session, keyword, result)
            )
            
        except Exception as e:
            logger.error(f"{self.supported_library} override error: {e}")
            return OverrideResult(
                success=False,
                error=f"{self.supported_library} override error: {str(e)}"
            )
    
    async def _handle_success_state_updates(self, session, keyword, result, keyword_info):
        """Override in subclasses for library-specific state handling."""
        pass
    
    def _get_library_state_updates(self, session, keyword, result) -> Dict[str, Any]:
        """Override in subclasses for library-specific state updates."""
        from datetime import datetime
        return {'last_activity': datetime.now().isoformat()}

# Browser Library Override Handler
class BrowserLibraryHandler(UnifiedLibraryHandler):
    """Specialized handler for Browser Library keywords using unified approach."""
    
    def __init__(self, execution_engine):
        super().__init__(execution_engine, "browser")
        
    async def _handle_success_state_updates(self, session, keyword, result, keyword_info):
        """Handle Browser Library-specific state updates."""
        keyword_lower = keyword.lower()
        
        # Browser-specific state management
        if 'new browser' in keyword_lower:
            session.browser_state.browser_type = 'chromium'  # Default, could be extracted from args
            session.browser_state.browser_id = f"browser_{session.session_id}"
            session.browser_state.active_library = "browser"
            
        elif 'new page' in keyword_lower:
            # Extract URL from args if provided
            url = result.get('args', ['about:blank'])[0] if result.get('args') else 'about:blank'
            session.browser_state.current_url = url
            session.browser_state.page_id = f"page_{session.session_id}"
            session.browser_state.active_library = "browser"
            
    def _get_library_state_updates(self, session, keyword, result) -> Dict[str, Any]:
        """Get Browser Library-specific state updates."""
        keyword_lower = keyword.lower()
        state_updates = super()._get_library_state_updates(session, keyword, result)
        
        # Add Browser-specific state tracking
        if 'new browser' in keyword_lower:
            state_updates['current_browser'] = {
                'type': 'chromium',  # Could extract from args
                'created_at': datetime.now().isoformat()
            }
        elif 'new context' in keyword_lower:
            state_updates['current_context'] = {
                'created_at': datetime.now().isoformat()
            }
        elif 'new page' in keyword_lower:
            state_updates['current_page'] = {
                'url': 'about:blank',  # Could extract from args
                'loaded_at': datetime.now().isoformat()
            }
        elif 'close browser' in keyword_lower:
            state_updates.update({
                'current_browser': None,
                'current_context': None,
                'current_page': None
            })
        else:
            state_updates['last_browser_activity'] = datetime.now().isoformat()
        
        return state_updates

# Generic Dynamic Handler
class DynamicExecutionHandler:
    """Default handler that uses pure dynamic keyword discovery."""
    
    def __init__(self, execution_engine):
        self.execution_engine = execution_engine
        
    async def execute(
        self, 
        session: 'ExecutionSession', 
        keyword: str, 
        args: List[str],
        keyword_info: Optional[Any] = None
    ) -> OverrideResult:
        """Execute keyword using dynamic discovery."""
        try:
            result = await self.execution_engine.keyword_discovery.execute_keyword(
                keyword,
                args,
                session.variables
            )
            
            return OverrideResult(
                success=result.get("success", False),
                output=result.get("output"),
                error=result.get("error"),
                metadata={'override': 'dynamic'}
            )
            
        except Exception as e:
            return OverrideResult(
                success=False,
                error=f"Dynamic execution error: {str(e)}",
                metadata={'override': 'dynamic'}
            )

# SeleniumLibrary Override Handler
class SeleniumLibraryHandler(UnifiedLibraryHandler):
    """Specialized handler for SeleniumLibrary keywords using unified approach."""
    
    def __init__(self, execution_engine):
        super().__init__(execution_engine, "seleniumlibrary")
        
    async def _handle_success_state_updates(self, session, keyword, result, keyword_info):
        """Handle SeleniumLibrary-specific state updates."""
        keyword_lower = keyword.lower()
        
        # SeleniumLibrary-specific state management
        session.browser_state.active_library = "selenium"
        
        if 'open browser' in keyword_lower:
            # Track SeleniumLibrary session using global instance
            try:
                selenium_lib = self.execution_engine.selenium_lib
                if selenium_lib and hasattr(selenium_lib, 'driver'):
                    driver = selenium_lib.driver
                    session.browser_state.driver_instance = driver
                    session.browser_state.selenium_session_id = driver.session_id if driver else None
                    logger.info(f"SeleniumLibrary browser opened for session {session.session_id}, driver session: {session.browser_state.selenium_session_id}")
            except Exception as e:
                logger.debug(f"Could not track SeleniumLibrary session: {e}")
                
        elif 'get source' in keyword_lower:
            # Store page source in session state
            page_source = result.get("output")
            if page_source:
                session.browser_state.page_source = page_source
                logger.debug(f"Page source retrieved via SeleniumLibrary: {len(page_source)} characters")
                
    def _get_library_state_updates(self, session, keyword, result) -> Dict[str, Any]:
        """Get SeleniumLibrary-specific state updates."""
        keyword_lower = keyword.lower()
        state_updates = super()._get_library_state_updates(session, keyword, result)
        
        # Add SeleniumLibrary-specific state tracking
        if 'open browser' in keyword_lower:
            browser_type = 'firefox'  # SeleniumLibrary default
            state_updates['current_browser'] = {
                'type': browser_type,
                'created_at': datetime.now().isoformat()
            }
        elif 'go to' in keyword_lower:
            state_updates['current_page'] = {
                'url': result.get('args', [''])[0] if result.get('args') else '',
                'loaded_at': datetime.now().isoformat()
            }
        else:
            state_updates['last_browser_activity'] = datetime.now().isoformat()
        
        return state_updates

def setup_default_overrides(registry: KeywordOverrideRegistry, execution_engine):
    """Set up the default keyword overrides."""
    
    # Browser Library - gets custom handling for state management and defaults
    browser_handler = BrowserLibraryHandler(execution_engine)
    registry.register_library('Browser', browser_handler)
    
    # SeleniumLibrary - gets custom handling for session tracking
    selenium_handler = SeleniumLibraryHandler(execution_engine)
    registry.register_library('SeleniumLibrary', selenium_handler)
    
    # Could add more specific overrides here:
    # registry.register_keyword('Get Text', custom_get_text_handler)
    # registry.register_pattern('SeleniumLibrary.*', selenium_handler)
    
    logger.info("Default keyword overrides configured")
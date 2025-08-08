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
            
        # 2. Check library-specific overrides
        if library_name:
            library_key = library_name.lower()
            if library_key in self.library_overrides:
                return self.library_overrides[library_key]
                
        # 3. Check pattern matches
        for pattern, handler in self.pattern_overrides.items():
            if self._matches_pattern(normalized_keyword, pattern, library_name):
                return handler
                
        return None
        
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

# Browser Library Override Handler
class BrowserLibraryHandler:
    """Specialized handler for Browser Library keywords with custom logic."""
    
    def __init__(self, execution_engine):
        self.execution_engine = execution_engine
        
    # Implement the protocol
    def __call__(self):
        return self
        
    async def execute(
        self, 
        session: 'ExecutionSession', 
        keyword: str, 
        args: List[str],
        keyword_info: Optional[Any] = None
    ) -> OverrideResult:
        """Execute Browser Library keyword with custom logic."""
        
        keyword_lower = keyword.lower()
        
        # Custom handling for specific Browser Library keywords
        if 'new browser' in keyword_lower:
            return await self._handle_new_browser(session, args, keyword_info)
        elif 'new context' in keyword_lower:
            return await self._handle_new_context(session, args, keyword_info)  
        elif 'new page' in keyword_lower:
            return await self._handle_new_page(session, args, keyword_info)
        elif 'close browser' in keyword_lower:
            return await self._handle_close_browser(session, args, keyword_info)
        # Handle SeleniumLibrary keywords
        elif 'open browser' in keyword_lower:
            return await self._handle_selenium_open_browser(session, args, keyword_info)
        elif 'get source' in keyword_lower:
            return await self._handle_selenium_get_source(session, args, keyword_info)
        else:
            # For other Browser keywords, use dynamic execution with state updates
            return await self._handle_generic_browser_keyword(session, keyword, args, keyword_info)
            
    async def _handle_new_browser(self, session, args, keyword_info) -> OverrideResult:
        """Handle New Browser with custom defaults."""
        try:
            # Apply custom defaults for headless
            processed_args = args.copy()
            
            # Check if headless is specified
            has_headless = any('headless=' in str(arg) for arg in args)
            if not has_headless:
                # Add default headless=False for better visibility
                processed_args.append('headless=False')
                logger.info("Applied default headless=False for New Browser")
            
            # Execute via dynamic discovery with processed args
            result = await self.execution_engine.keyword_discovery.execute_keyword(
                'New Browser',
                processed_args,
                session.variables
            )
            
            # Update browser state
            state_updates = {}
            if result.get("success"):
                state_updates['current_browser'] = {
                    'type': processed_args[0] if processed_args else 'chromium',
                    'headless': 'headless=True' in processed_args,
                    'created_at': datetime.now().isoformat()
                }
                
            return OverrideResult(
                success=result.get("success", False),
                output=result.get("output"),
                error=result.get("error"),
                state_updates=state_updates,
                metadata={'override': 'browser_new_browser', 'args_modified': len(processed_args) != len(args)}
            )
            
        except Exception as e:
            return OverrideResult(
                success=False,
                error=f"Browser override error: {str(e)}",
                metadata={'override': 'browser_new_browser'}
            )
            
    async def _handle_new_context(self, session, args, keyword_info) -> OverrideResult:
        """Handle New Context with state tracking."""
        try:
            result = await self.execution_engine.keyword_discovery.execute_keyword(
                'New Context',
                args,
                session.variables
            )
            
            state_updates = {}
            if result.get("success"):
                state_updates['current_context'] = {
                    'created_at': datetime.now().isoformat(),
                    'args': args
                }
                
            return OverrideResult(
                success=result.get("success", False),
                output=result.get("output"),
                error=result.get("error"),
                state_updates=state_updates,
                metadata={'override': 'browser_new_context'}
            )
            
        except Exception as e:
            return OverrideResult(
                success=False,
                error=f"Context override error: {str(e)}",
                metadata={'override': 'browser_new_context'}
            )
            
    async def _handle_new_page(self, session, args, keyword_info) -> OverrideResult:
        """Handle New Page with URL tracking."""
        try:
            result = await self.execution_engine.keyword_discovery.execute_keyword(
                'New Page',
                args,
                session.variables
            )
            
            state_updates = {}
            if result.get("success"):
                state_updates['current_page'] = {
                    'url': args[0] if args else 'about:blank',
                    'loaded_at': datetime.now().isoformat()
                }
                
            return OverrideResult(
                success=result.get("success", False),
                output=result.get("output"),
                error=result.get("error"),
                state_updates=state_updates,
                metadata={'override': 'browser_new_page'}
            )
            
        except Exception as e:
            return OverrideResult(
                success=False,
                error=f"Page override error: {str(e)}",
                metadata={'override': 'browser_new_page'}
            )
            
    async def _handle_close_browser(self, session, args, keyword_info) -> OverrideResult:
        """Handle Close Browser with state cleanup."""
        try:
            result = await self.execution_engine.keyword_discovery.execute_keyword(
                'Close Browser',
                args,
                session.variables
            )
            
            state_updates = {}
            if result.get("success"):
                # Clear browser state
                state_updates['current_browser'] = None
                state_updates['current_context'] = None  
                state_updates['current_page'] = None
                
            return OverrideResult(
                success=result.get("success", False),
                output=result.get("output"),
                error=result.get("error"),
                state_updates=state_updates,
                metadata={'override': 'browser_close_browser'}
            )
            
        except Exception as e:
            return OverrideResult(
                success=False,
                error=f"Close browser override error: {str(e)}",
                metadata={'override': 'browser_close_browser'}
            )
            
    async def _handle_generic_browser_keyword(self, session, keyword, args, keyword_info) -> OverrideResult:
        """Handle other Browser keywords with argument conversion and state awareness."""
        try:
            # Use the argument conversion logic from dynamic keywords
            discovery = self.execution_engine.keyword_discovery
            
            # Get keyword info if not provided
            if not keyword_info:
                keyword_info = discovery.find_keyword(keyword)
            
            # Convert arguments using the same logic as dynamic keywords
            if keyword_info and keyword_info.library == "Browser":
                converted_args = discovery._convert_browser_arguments(keyword_info, args)
                
                # Get the library instance and execute with converted args
                library = discovery.libraries.get("Browser")
                if library:
                    method = getattr(library.instance, keyword_info.method_name)
                    try:
                        result = method(*converted_args)
                        
                        # Update last activity timestamp
                        state_updates = {
                            'last_browser_activity': datetime.now().isoformat()
                        }
                        
                        return OverrideResult(
                            success=True,
                            output=str(result) if result is not None else f"Executed {keyword}",
                            state_updates=state_updates,
                            metadata={'override': 'browser_generic_with_conversion'}
                        )
                    except Exception as method_error:
                        # If direct method call fails, try the original approach
                        logger.debug(f"Direct method call failed: {method_error}, trying original approach")
                        result = await discovery.execute_keyword(keyword, args, session.variables)
                        
                        state_updates = {
                            'last_browser_activity': datetime.now().isoformat()
                        }
                        
                        return OverrideResult(
                            success=result.get("success", False),
                            output=result.get("output"),
                            error=result.get("error"),
                            state_updates=state_updates,
                            metadata={'override': 'browser_generic_fallback'}
                        )
            
            # Fall back to original approach if no conversion available
            result = await discovery.execute_keyword(keyword, args, session.variables)
            
            # Update last activity timestamp
            state_updates = {
                'last_browser_activity': datetime.now().isoformat()
            }
            
            return OverrideResult(
                success=result.get("success", False),
                output=result.get("output"),
                error=result.get("error"),
                state_updates=state_updates,
                metadata={'override': 'browser_generic'}
            )
            
        except Exception as e:
            return OverrideResult(
                success=False,
                error=f"Browser keyword error: {str(e)}",
                metadata={'override': 'browser_generic'}
            )

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

    async def _handle_selenium_open_browser(self, session, args, keyword_info) -> OverrideResult:
        """Handle SeleniumLibrary Open Browser keyword with session tracking."""
        try:
            # Mark session as using SeleniumLibrary
            session.browser_state.active_library = "selenium"
            
            # Execute via dynamic discovery
            result = await self.execution_engine.keyword_discovery.execute_keyword(
                'Open Browser',
                args,
                session.variables
            )
            
            if result.get("success"):
                # Track SeleniumLibrary session
                try:
                    # Get the WebDriver instance from SeleniumLibrary
                    if self.execution_engine.selenium_lib:
                        driver = self.execution_engine.selenium_lib.driver
                        session.browser_state.driver_instance = driver
                        session.browser_state.selenium_session_id = driver.session_id if driver else None
                        logger.info(f"SeleniumLibrary browser opened, session: {session.browser_state.selenium_session_id}")
                except Exception as e:
                    logger.debug(f"Could not track SeleniumLibrary session: {e}")
            
            return OverrideResult(
                success=result.get("success", False),
                output=result.get("output"),
                error=result.get("error"),
                metadata={'override': 'selenium_open_browser'}
            )
            
        except Exception as e:
            return OverrideResult(
                success=False,
                error=f"SeleniumLibrary Open Browser error: {str(e)}",
                metadata={'override': 'selenium_open_browser'}
            )

    async def _handle_selenium_get_source(self, session, args, keyword_info) -> OverrideResult:
        """Handle SeleniumLibrary Get Source keyword."""
        try:
            # Mark session as using SeleniumLibrary  
            session.browser_state.active_library = "selenium"
            
            # Execute via dynamic discovery
            result = await self.execution_engine.keyword_discovery.execute_keyword(
                'Get Source',
                args,
                session.variables
            )
            
            if result.get("success"):
                # Store page source in session state
                page_source = result.get("result")
                if page_source:
                    session.browser_state.page_source = page_source
                    logger.debug(f"Page source retrieved via SeleniumLibrary: {len(page_source)} characters")
            
            return OverrideResult(
                success=result.get("success", False),
                output=result.get("output"),
                error=result.get("error"),
                metadata={'override': 'selenium_get_source'}
            )
            
        except Exception as e:
            return OverrideResult(
                success=False,
                error=f"SeleniumLibrary Get Source error: {str(e)}",
                metadata={'override': 'selenium_get_source'}
            )

def setup_default_overrides(registry: KeywordOverrideRegistry, execution_engine):
    """Set up the default keyword overrides."""
    
    # Browser Library - gets custom handling for state management and defaults
    browser_handler = BrowserLibraryHandler(execution_engine)
    registry.register_library('Browser', browser_handler)
    
    # Could add more specific overrides here:
    # registry.register_keyword('Get Text', custom_get_text_handler)
    # registry.register_pattern('SeleniumLibrary.*', selenium_handler)
    
    logger.info("Default keyword overrides configured")
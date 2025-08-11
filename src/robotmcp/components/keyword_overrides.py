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
        else:
            # For other Browser keywords, use dynamic execution with state updates
            return await self._handle_generic_browser_keyword(session, keyword, args, keyword_info)
            
    async def _handle_new_browser(self, session, args, keyword_info) -> OverrideResult:
        """Handle New Browser with custom defaults."""
        try:
            # Parse arguments using Robot Framework's native approach if available
            discovery = self.execution_engine.keyword_discovery
            # Get keyword info for better parsing
            keyword_info = discovery.find_keyword('New Browser')
            if keyword_info:
                parsed_args = discovery._parse_arguments_with_rf_spec(keyword_info, args)
            else:
                parsed_args = discovery._parse_arguments(args)
            
            # Apply custom defaults for headless
            if 'headless' not in parsed_args.named:
                # Add default headless=False for better visibility
                parsed_args.named['headless'] = 'False'
                logger.info("Applied default headless=False for New Browser")
            
            # Reconstruct args from parsed arguments
            processed_args = parsed_args.positional.copy()
            for key, value in parsed_args.named.items():
                processed_args.append(f"{key}={value}")
            
            # Execute via dynamic discovery with processed args
            result = await self.execution_engine.keyword_discovery.execute_keyword(
                'New Browser',
                processed_args,
                session.variables
            )
            
            # Update browser state
            state_updates = {}
            if result.get("success"):
                # Determine browser type from positional args or named args
                browser_type = 'chromium'  # default
                if parsed_args.positional:
                    browser_type = parsed_args.positional[0]
                elif 'browser' in parsed_args.named:
                    browser_type = parsed_args.named['browser']
                    
                # Determine headless setting
                headless = False  # default
                if 'headless' in parsed_args.named:
                    headless = parsed_args.named['headless'].lower() in ['true', '1', 'yes']
                    
                # Set browser state fields that the detection logic relies on
                session.browser_state.browser_type = browser_type
                session.browser_state.browser_id = f"browser_{session.session_id}"  # Set browser_id for detection
                session.browser_state.active_library = "browser"  # Explicitly set active library
                
                state_updates['current_browser'] = {
                    'type': browser_type,
                    'headless': headless,
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
                # Set page state fields for proper detection
                url = args[0] if args else 'about:blank'
                session.browser_state.current_url = url
                session.browser_state.page_id = f"page_{session.session_id}"  # Set page_id for detection
                session.browser_state.active_library = "browser"  # Ensure Browser Library is active
                
                state_updates['current_page'] = {
                    'url': url,
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
            
            # Parse arguments using Robot Framework's native approach if available
            if keyword_info and keyword_info.library == "Browser":
                # Use the LibDoc-based conversion approach directly from argument processor
                libdoc_converted = discovery.argument_processor.convert_browser_arguments(keyword_info.name, args, keyword_info.library)
                
                # Extract LibDoc type info metadata
                libdoc_type_info = libdoc_converted.pop('_libdoc_type_info', {})
                
                # Apply smart conversion only for patterns that LibDoc didn't handle
                smart_converted = {}
                for key, value in libdoc_converted.items():
                    if isinstance(value, str) and not libdoc_type_info.get(key, False):
                        # Only apply smart conversion if LibDoc didn't provide type info for this argument
                        if value.startswith('{') and value.endswith('}'):
                            smart_converted[key] = discovery.argument_processor.convert_string_value(value, "dict")
                        elif value.startswith('[') and value.endswith(']'):
                            smart_converted[key] = discovery.argument_processor.convert_string_value(value, "list") 
                        elif value.lower() in ['true', 'false']:
                            smart_converted[key] = discovery.argument_processor.convert_string_value(value, "bool")
                        elif value.isdigit():
                            smart_converted[key] = discovery.argument_processor.convert_string_value(value, "int")
                        else:
                            smart_converted[key] = value
                    else:
                        # Keep original value if LibDoc provided type info or value is not a string
                        smart_converted[key] = value
                
                # Extract positional and keyword arguments
                converted_args = [v for k, v in smart_converted.items() if k.startswith('arg_')]
                converted_kwargs = {k: v for k, v in smart_converted.items() if not k.startswith('arg_')}
                
                # Use global browser library instance
                browser_lib = self.execution_engine.browser_lib
                if browser_lib and keyword_info.library == "Browser":
                    method = getattr(browser_lib, keyword_info.method_name)
                    try:
                        if converted_kwargs:
                            result = method(*converted_args, **converted_kwargs)
                        else:
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

# SeleniumLibrary Override Handler
class SeleniumLibraryHandler:
    """Specialized handler for SeleniumLibrary keywords with custom logic."""
    
    def __init__(self, execution_engine):
        self.execution_engine = execution_engine
        
    async def execute(
        self, 
        session: 'ExecutionSession', 
        keyword: str, 
        args: List[str],
        keyword_info: Optional[Any] = None
    ) -> OverrideResult:
        """Execute SeleniumLibrary keyword with custom logic."""
        
        keyword_lower = keyword.lower()
        
        # Custom handling for specific SeleniumLibrary keywords
        if 'open browser' in keyword_lower:
            return await self._handle_selenium_open_browser(session, args, keyword_info)
        elif 'get source' in keyword_lower:
            return await self._handle_selenium_get_source(session, args, keyword_info)
        else:
            # For other SeleniumLibrary keywords, use dynamic execution
            return await self._handle_generic_selenium_keyword(session, keyword, args, keyword_info)

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
            
            state_updates = {}
            if result.get("success"):
                # Track SeleniumLibrary session using global instance
                try:
                    # Get the WebDriver instance from global SeleniumLibrary
                    selenium_lib = self.execution_engine.selenium_lib
                    if selenium_lib and hasattr(selenium_lib, 'driver'):
                        driver = selenium_lib.driver
                        session.browser_state.driver_instance = driver
                        session.browser_state.selenium_session_id = driver.session_id if driver else None
                        logger.info(f"SeleniumLibrary browser opened for session {session.session_id}, driver session: {session.browser_state.selenium_session_id}")
                    else:
                        logger.debug(f"No SeleniumLibrary driver available for session {session.session_id}")
                except Exception as e:
                    logger.debug(f"Could not track SeleniumLibrary session: {e}")
                
                state_updates['last_browser_activity'] = datetime.now().isoformat()
            
            return OverrideResult(
                success=result.get("success", False),
                output=result.get("output"),
                error=result.get("error"),
                state_updates=state_updates,
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

    async def _handle_generic_selenium_keyword(self, session, keyword, args, keyword_info) -> OverrideResult:
        """Handle other SeleniumLibrary keywords with argument conversion and state awareness."""
        try:
            # Mark session as using SeleniumLibrary
            session.browser_state.active_library = "selenium"
            
            # Execute via dynamic discovery
            result = await self.execution_engine.keyword_discovery.execute_keyword(
                keyword,
                args,
                session.variables
            )
            
            # Update last activity timestamp
            state_updates = {
                'last_browser_activity': datetime.now().isoformat()
            }
            
            return OverrideResult(
                success=result.get("success", False),
                output=result.get("output"),
                error=result.get("error"),
                state_updates=state_updates,
                metadata={'override': 'selenium_generic'}
            )
            
        except Exception as e:
            return OverrideResult(
                success=False,
                error=f"SeleniumLibrary keyword error: {str(e)}",
                metadata={'override': 'selenium_generic'}
            )

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
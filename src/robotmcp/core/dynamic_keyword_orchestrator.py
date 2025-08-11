"""Main orchestrator for dynamic keyword discovery."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from robotmcp.models.library_models import KeywordInfo, ParsedArguments
from robotmcp.core.library_manager import LibraryManager
from robotmcp.core.keyword_discovery import KeywordDiscovery
from robotmcp.utils.argument_processor import ArgumentProcessor

logger = logging.getLogger(__name__)


class DynamicKeywordDiscovery:
    """Main orchestrator for dynamic Robot Framework keyword discovery and management."""
    
    def __init__(self):
        self.library_manager = LibraryManager()
        self.keyword_discovery = KeywordDiscovery()
        self.argument_processor = ArgumentProcessor()
        
        # Initialize all components
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize all libraries and set up keyword discovery."""
        # Load all libraries through the library manager
        self.library_manager.load_all_libraries(self.keyword_discovery)
        
        # Add all keywords to cache
        for lib_info in self.library_manager.libraries.values():
            self.keyword_discovery.add_keywords_to_cache(lib_info)
        
        logger.info(f"Initialized {len(self.library_manager.libraries)} libraries with {len(self.keyword_discovery.keyword_cache)} keywords")
    
    # Public API methods
    def find_keyword(self, keyword_name: str) -> Optional[KeywordInfo]:
        """Find a keyword by name with fuzzy matching."""
        return self.keyword_discovery.find_keyword(keyword_name)
    
    def get_keyword_suggestions(self, keyword_name: str, limit: int = 5) -> List[str]:
        """Get keyword suggestions based on partial match."""
        return self.keyword_discovery.get_keyword_suggestions(keyword_name, limit)
    
    def suggest_similar_keywords(self, keyword_name: str, max_suggestions: int = 5) -> List[KeywordInfo]:
        """Suggest similar keywords based on name similarity."""
        # This is a more sophisticated version that returns KeywordInfo objects
        suggestions = []
        keyword_lower = keyword_name.lower()
        
        for cached_name, keyword_info in self.keyword_discovery.keyword_cache.items():
            score = self._similarity_score(keyword_lower, cached_name)
            if score > 0.3:  # Minimum similarity threshold
                suggestions.append((score, keyword_info))
        
        # Sort by similarity score and return top suggestions
        suggestions.sort(key=lambda x: x[0], reverse=True)
        return [info for _, info in suggestions[:max_suggestions]]
    
    def search_keywords(self, pattern: str) -> List[KeywordInfo]:
        """Search for keywords matching a pattern."""
        import re
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            matches = []
            for keyword_info in self.keyword_discovery.keyword_cache.values():
                if (regex.search(keyword_info.name) or 
                    regex.search(keyword_info.doc) or 
                    regex.search(keyword_info.library)):
                    matches.append(keyword_info)
            return matches
        except re.error:
            # If pattern is not valid regex, do simple string matching
            pattern_lower = pattern.lower()
            return [info for info in self.keyword_discovery.keyword_cache.values() 
                   if pattern_lower in info.name.lower() or 
                      pattern_lower in info.doc.lower() or 
                      pattern_lower in info.library.lower()]
    
    def get_keywords_by_library(self, library_name: str) -> List[KeywordInfo]:
        """Get all keywords from a specific library."""
        return self.keyword_discovery.get_keywords_by_library(library_name)
    
    def get_all_keywords(self) -> List[KeywordInfo]:
        """Get all cached keywords."""
        return self.keyword_discovery.get_all_keywords()
    
    def get_keyword_count(self) -> int:
        """Get total number of cached keywords."""
        return self.keyword_discovery.get_keyword_count()
    
    def is_dom_changing_keyword(self, keyword_name: str) -> bool:
        """Check if a keyword likely changes the DOM."""
        return self.keyword_discovery.is_dom_changing_keyword(keyword_name)
    
    # Argument processing methods
    def parse_arguments(self, args: List[str]) -> ParsedArguments:
        """Parse a list of arguments into positional and named arguments."""
        return self.argument_processor.parse_arguments(args)
    
    def _parse_arguments(self, args: List[str]) -> ParsedArguments:
        """Parse Robot Framework-style arguments (internal method for compatibility)."""
        return self.argument_processor.parse_arguments(args)
    
    def _parse_arguments_with_rf_spec(self, keyword_info: KeywordInfo, args: List[str]) -> ParsedArguments:
        """Parse arguments using Robot Framework's native ArgumentSpec if available."""
        try:
            from robot.running.arguments import ArgumentSpec
            from robot.running.arguments.argumentresolver import ArgumentResolver
            
            # Try to create ArgumentSpec from keyword info
            if hasattr(keyword_info, 'args') and keyword_info.args:
                spec = ArgumentSpec(
                    positional_or_named=keyword_info.args,
                    defaults=keyword_info.defaults if hasattr(keyword_info, 'defaults') else {}
                )
                
                # Use Robot Framework's ArgumentResolver to split arguments
                resolver = ArgumentResolver(spec, resolve_named=True)
                positional, named = resolver.resolve(args, named_args=None)
                
                # Convert to our ParsedArguments format
                parsed = ParsedArguments()
                parsed.positional = positional
                parsed.named = {k: v for k, v in named.items()} if named else {}
                
                return parsed
                
        except (ImportError, Exception) as e:
            logger.debug(f"RF ArgumentSpec parsing failed: {e}, using fallback parsing")
            
        # Fall back to our custom parsing logic
        return self._parse_arguments(args)
    
    
    
    # Library management methods
    def get_library_exclusion_info(self) -> Dict[str, Any]:
        """Get information about library exclusions."""
        return self.library_manager.get_library_exclusion_info()
    
    def get_library_status(self) -> Dict[str, Any]:
        """Get status of all libraries."""
        return {
            "loaded_libraries": {
                name: {
                    "keywords": len(lib.keywords),
                    "doc": lib.doc,
                    "version": lib.version,
                    "scope": lib.scope
                }
                for name, lib in self.library_manager.libraries.items()
            },
            "failed_imports": self.library_manager.failed_imports,
            "total_keywords": len(self.keyword_discovery.keyword_cache)
        }
    
    # Properties for backward compatibility and access to internal components
    @property
    def libraries(self) -> Dict[str, Any]:
        """Access to loaded libraries."""
        return self.library_manager.libraries
    
    @property
    def keyword_cache(self) -> Dict[str, KeywordInfo]:
        """Access to keyword cache."""
        return self.keyword_discovery.keyword_cache
    
    @property
    def failed_imports(self) -> Dict[str, str]:
        """Access to failed imports."""
        return self.library_manager.failed_imports
    
    @property
    def excluded_libraries(self) -> set:
        """Access to excluded libraries."""
        return self.library_manager.excluded_libraries
    
    # Utility methods
    def _similarity_score(self, a: str, b: str) -> float:
        """Calculate similarity score between two strings."""
        if not a or not b:
            return 0.0
        
        # Simple similarity based on common substring length
        a, b = a.lower(), b.lower()
        if a == b:
            return 1.0
        
        # Check for substring matches
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        if shorter in longer:
            return len(shorter) / len(longer)
        
        # Calculate based on common characters
        common = sum(1 for char in shorter if char in longer)
        return common / max(len(a), len(b))
    
    # Execution methods (delegated from the original implementation)
    def _execute_direct_method_call(self, keyword_info: KeywordInfo, parsed_args: ParsedArguments, original_args: List[str], session_variables: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a keyword by calling its method directly."""
        try:
            # Get library instance
            library = self.libraries[keyword_info.library]
            method = getattr(library.instance, keyword_info.method_name)
            
            # Handle different types of method calls
            if keyword_info.is_builtin and hasattr(library.instance, '_context'):
                # BuiltIn library methods might need context
                result = method(*original_args)
            else:
                # Regular library methods
                if keyword_info.library == "Browser":
                    try:
                        # Use the LibDoc-based approach, with fallback to smart conversion
                        libdoc_converted = self.argument_processor.convert_browser_arguments(keyword_info.name, original_args, keyword_info.library)
                        
                        # Extract LibDoc type info metadata
                        libdoc_type_info = libdoc_converted.pop('_libdoc_type_info', {})
                        
                        # Apply smart conversion only when LibDoc didn't provide explicit type information
                        smart_converted = {}
                        for key, value in libdoc_converted.items():
                            if isinstance(value, str) and not libdoc_type_info.get(key, False):
                                # Only apply smart type conversion if LibDoc didn't provide type info for this argument
                                if value.startswith('{') and value.endswith('}'):
                                    # Dictionary pattern
                                    smart_converted[key] = self.argument_processor.convert_string_value(value, "dict")
                                elif value.startswith('[') and value.endswith(']'):
                                    # List pattern
                                    smart_converted[key] = self.argument_processor.convert_string_value(value, "list") 
                                elif value.lower() in ['true', 'false']:
                                    # Boolean pattern
                                    smart_converted[key] = self.argument_processor.convert_string_value(value, "bool")
                                elif value.isdigit():
                                    # Integer pattern - only if LibDoc didn't specify the type
                                    smart_converted[key] = self.argument_processor.convert_string_value(value, "int")
                                else:
                                    # Keep as string
                                    smart_converted[key] = value
                            else:
                                # Keep original value if LibDoc provided type info or value is not a string
                                smart_converted[key] = value
                        
                        # Extract positional and keyword arguments
                        pos_args = [v for k, v in smart_converted.items() if k.startswith('arg_')]
                        kwargs = {k: v for k, v in smart_converted.items() if not k.startswith('arg_')}
                        
                        if kwargs:
                            result = method(*pos_args, **kwargs)
                        else:
                            result = method(*pos_args)
                    except Exception as browser_error:
                        logger.debug(f"Browser Library execution failed: {browser_error}")
                        # Don't fall back to unconverted args as this can cause enum type errors
                        # Re-raise the original error
                        raise browser_error
                elif keyword_info.name == "Create List":
                    # Collections.Create List takes variable arguments
                    result = method(*parsed_args.positional)
                elif keyword_info.name == "Set Variable":
                    # Set Variable takes one argument
                    value = parsed_args.positional[0] if parsed_args.positional else None
                    result = method(value)
                else:
                    # For other libraries, use positional args
                    result = method(*parsed_args.positional)
            
            return {
                "success": True,
                "output": str(result) if result is not None else f"Executed {keyword_info.name}",
                "result": result,
                "keyword_info": {
                    "name": keyword_info.name,
                    "library": keyword_info.library,
                    "doc": keyword_info.doc
                }
            }
            
        except Exception as e:
            import traceback
            logger.debug(f"Full traceback for {keyword_info.library}.{keyword_info.name}: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Error executing {keyword_info.library}.{keyword_info.name}: {str(e)}",
                "keyword_info": {
                    "name": keyword_info.name,
                    "library": keyword_info.library,
                    "args": keyword_info.args,
                    "doc": keyword_info.doc
                }
            }
    
    async def execute_keyword(self, keyword_name: str, args: List[str], session_variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a keyword dynamically."""
        keyword_info = self.find_keyword(keyword_name)
        
        if not keyword_info:
            return {
                "success": False,
                "error": f"Keyword '{keyword_name}' not found in any loaded library",
                "suggestions": self.get_keyword_suggestions(keyword_name, 3)
            }
        
        try:
            # Parse arguments
            parsed_args = self.parse_arguments(args)
            
            # Execute the keyword
            return self._execute_direct_method_call(keyword_info, parsed_args, args, session_variables or {})
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error executing {keyword_name}: {str(e)}",
                "keyword_info": {
                    "name": keyword_info.name,
                    "library": keyword_info.library,
                    "doc": keyword_info.doc
                }
            }


# Global instance management
_keyword_discovery = None


def get_keyword_discovery() -> DynamicKeywordDiscovery:
    """Get the global keyword discovery instance."""
    global _keyword_discovery
    if _keyword_discovery is None:
        _keyword_discovery = DynamicKeywordDiscovery()
    return _keyword_discovery
"""Dynamic keyword discovery and execution for Robot Framework libraries."""

import importlib
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import re

logger = logging.getLogger(__name__)

@dataclass
class KeywordInfo:
    """Information about a Robot Framework keyword."""
    name: str
    library: str
    method_name: str
    doc: str = ""
    args: List[str] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    is_builtin: bool = False

@dataclass 
class LibraryInfo:
    """Information about a Robot Framework library."""
    name: str
    instance: Any
    keywords: Dict[str, KeywordInfo] = field(default_factory=dict)
    doc: str = ""
    version: str = ""
    scope: str = "SUITE"

class DynamicKeywordDiscovery:
    """Discovers and manages Robot Framework keywords dynamically."""
    
    def __init__(self):
        self.libraries: Dict[str, LibraryInfo] = {}
        self.keyword_cache: Dict[str, KeywordInfo] = {}
        self.failed_imports: Dict[str, str] = {}
        
        # Common Robot Framework libraries to try
        self.common_libraries = [
            'BuiltIn',
            'Browser', 
            'SeleniumLibrary',
            'RequestsLibrary',
            'Collections',
            'String',
            'DateTime',
            'OperatingSystem',
            'Process',
            'DatabaseLibrary',
            'SSHLibrary',
            'AppiumLibrary'
        ]
        
        self._initialize_libraries()
    
    def _initialize_libraries(self):
        """Initialize available Robot Framework libraries."""
        logger.info("Initializing Robot Framework libraries...")
        
        # Always try to initialize BuiltIn library first
        self._try_import_library('BuiltIn')
        
        # Try to import other common libraries
        for library_name in self.common_libraries:
            if library_name != 'BuiltIn':
                self._try_import_library(library_name)
        
        logger.info(f"Initialized {len(self.libraries)} libraries with {len(self.keyword_cache)} keywords")
    
    def _try_import_library(self, library_name: str) -> bool:
        """Try to import and initialize a Robot Framework library."""
        try:
            # Handle special cases
            if library_name == 'BuiltIn':
                from robot.libraries.BuiltIn import BuiltIn
                instance = BuiltIn()
            elif library_name == 'Browser':
                try:
                    from Browser import Browser
                    instance = Browser()
                except ImportError:
                    logger.debug(f"Browser library not available")
                    self.failed_imports[library_name] = "Not installed"
                    return False
            elif library_name == 'SeleniumLibrary':
                try:
                    from SeleniumLibrary import SeleniumLibrary
                    instance = SeleniumLibrary()
                except ImportError:
                    logger.debug(f"SeleniumLibrary not available")
                    self.failed_imports[library_name] = "Not installed"
                    return False
            elif library_name == 'RequestsLibrary':
                try:
                    from RequestsLibrary import RequestsLibrary
                    instance = RequestsLibrary()
                except ImportError:
                    logger.debug(f"RequestsLibrary not available")
                    self.failed_imports[library_name] = "Not installed"
                    return False
            elif library_name == 'Collections':
                from robot.libraries.Collections import Collections
                instance = Collections()
            elif library_name == 'String':
                from robot.libraries.String import String
                instance = String()
            elif library_name == 'DateTime':
                from robot.libraries.DateTime import DateTime
                instance = DateTime()
            elif library_name == 'OperatingSystem':
                from robot.libraries.OperatingSystem import OperatingSystem
                instance = OperatingSystem()
            elif library_name == 'Process':
                from robot.libraries.Process import Process
                instance = Process()
            else:
                # Try generic import
                module = importlib.import_module(library_name)
                if hasattr(module, library_name):
                    instance = getattr(module, library_name)()
                else:
                    instance = module
            
            # Extract keywords from the library instance
            lib_info = self._extract_library_info(library_name, instance)
            self.libraries[library_name] = lib_info
            
            # Add keywords to cache
            for keyword_name, keyword_info in lib_info.keywords.items():
                self.keyword_cache[keyword_name.lower()] = keyword_info
            
            logger.info(f"Successfully loaded library '{library_name}' with {len(lib_info.keywords)} keywords")
            return True
            
        except Exception as e:
            logger.debug(f"Failed to import library '{library_name}': {e}")
            self.failed_imports[library_name] = str(e)
            return False
    
    def _extract_library_info(self, library_name: str, instance: Any) -> LibraryInfo:
        """Extract keyword information from a library instance."""
        lib_info = LibraryInfo(
            name=library_name,
            instance=instance,
            doc=getattr(instance, '__doc__', ''),
            version=getattr(instance, '__version__', getattr(instance, 'ROBOT_LIBRARY_VERSION', '')),
            scope=getattr(instance, 'ROBOT_LIBRARY_SCOPE', 'SUITE')
        )
        
        # Get all public methods that could be keywords
        for attr_name in dir(instance):
            if attr_name.startswith('_'):
                continue
                
            attr = getattr(instance, attr_name)
            if not callable(attr):
                continue
            
            # Convert method name to Robot Framework keyword format
            keyword_name = self._method_to_keyword_name(attr_name)
            
            # Extract keyword information
            keyword_info = self._extract_keyword_info(library_name, keyword_name, attr_name, attr)
            lib_info.keywords[keyword_name] = keyword_info
        
        return lib_info
    
    def _method_to_keyword_name(self, method_name: str) -> str:
        """Convert Python method name to Robot Framework keyword name."""
        # Convert snake_case to Title Case
        words = method_name.split('_')
        return ' '.join(word.capitalize() for word in words)
    
    def _extract_keyword_info(self, library_name: str, keyword_name: str, method_name: str, method: Callable) -> KeywordInfo:
        """Extract information about a specific keyword."""
        try:
            # Get method signature
            sig = inspect.signature(method)
            args = []
            defaults = {}
            
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                    
                args.append(param_name)
                if param.default != inspect.Parameter.empty:
                    defaults[param_name] = param.default
            
            # Get documentation
            doc = inspect.getdoc(method) or ""
            
            # Extract tags from docstring (Robot Framework convention)
            tags = []
            if doc:
                tag_match = re.search(r'Tags:\s*(.+)', doc)
                if tag_match:
                    tags = [tag.strip() for tag in tag_match.group(1).split(',')]
            
            return KeywordInfo(
                name=keyword_name,
                library=library_name,
                method_name=method_name,
                doc=doc,
                args=args,
                defaults=defaults,
                tags=tags,
                is_builtin=(library_name in ['BuiltIn', 'Collections', 'String', 'DateTime', 'OperatingSystem', 'Process'])
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract info for {library_name}.{method_name}: {e}")
            return KeywordInfo(
                name=keyword_name,
                library=library_name,
                method_name=method_name
            )
    
    def find_keyword(self, keyword_name: str) -> Optional[KeywordInfo]:
        """Find a keyword by name (case-insensitive)."""
        # Normalize keyword name
        normalized = keyword_name.lower().strip()
        
        # Direct lookup
        if normalized in self.keyword_cache:
            return self.keyword_cache[normalized]
        
        # Try variations
        variations = [
            normalized.replace(' ', '_'),
            normalized.replace('_', ' '),
            normalized.replace(' ', ''),
            normalized
        ]
        
        for variation in variations:
            if variation in self.keyword_cache:
                return self.keyword_cache[variation]
        
        # Partial matching
        for cached_name, keyword_info in self.keyword_cache.items():
            if normalized in cached_name or cached_name in normalized:
                return keyword_info
        
        return None
    
    def get_keywords_by_library(self, library_name: str) -> List[KeywordInfo]:
        """Get all keywords from a specific library."""
        if library_name not in self.libraries:
            return []
        
        return list(self.libraries[library_name].keywords.values())
    
    def get_all_keywords(self) -> List[KeywordInfo]:
        """Get all available keywords."""
        return list(self.keyword_cache.values())
    
    def search_keywords(self, pattern: str) -> List[KeywordInfo]:
        """Search for keywords matching a pattern."""
        pattern = pattern.lower()
        matches = []
        
        for keyword_info in self.keyword_cache.values():
            if (pattern in keyword_info.name.lower() or 
                pattern in keyword_info.doc.lower() or
                any(pattern in tag.lower() for tag in keyword_info.tags)):
                matches.append(keyword_info)
        
        return matches
    
    async def execute_keyword(self, keyword_name: str, args: List[str], session_variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a keyword dynamically."""
        keyword_info = self.find_keyword(keyword_name)
        
        if not keyword_info:
            return {
                "success": False,
                "error": f"Keyword '{keyword_name}' not found in any loaded library",
                "suggestions": self._get_keyword_suggestions(keyword_name)
            }
        
        try:
            # Get library instance
            library = self.libraries[keyword_info.library]
            method = getattr(library.instance, keyword_info.method_name)
            
            # Validate arguments
            validation_result = self._validate_arguments(keyword_info, args)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "keyword_info": {
                        "name": keyword_info.name,
                        "library": keyword_info.library,
                        "args": keyword_info.args,
                        "doc": keyword_info.doc
                    }
                }
            
            # Execute the keyword
            if session_variables is None:
                session_variables = {}
            
            # Handle different types of method calls
            try:
                if keyword_info.is_builtin and hasattr(library.instance, '_context'):
                    # BuiltIn library methods might need context
                    result = method(*args)
                else:
                    # Regular library methods - handle variable arguments
                    if keyword_info.name == "Create List":
                        # Collections.Create List takes variable arguments
                        result = method(*args)
                    elif keyword_info.name == "Set Variable":
                        # Set Variable only takes one argument (the value)
                        result = method(args[0] if args else None)
                    elif keyword_info.library == "Browser" and keyword_info.name == "New Browser":
                        # Handle Browser Library's enum requirement
                        try:
                            from Browser.utils.data_types import SupportedBrowsers
                            browser_type = args[0] if args else "chromium"
                            browser_enum = getattr(SupportedBrowsers, browser_type.lower(), SupportedBrowsers.chromium)
                            result = method(browser_enum)
                        except Exception:
                            # Fall back to regular call
                            result = method(*args)
                    else:
                        result = method(*args)
            except TypeError as e:
                # Try calling with different argument patterns
                if "takes" in str(e) and "positional argument" in str(e):
                    # Try calling with args as a single list argument
                    try:
                        result = method(args)
                    except:
                        # Fall back to original error
                        raise e
                else:
                    raise e
            
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
    
    def _validate_arguments(self, keyword_info: KeywordInfo, args: List[str]) -> Dict[str, Any]:
        """Validate arguments for a keyword."""
        required_args = [arg for arg in keyword_info.args if arg not in keyword_info.defaults]
        
        if len(args) < len(required_args):
            missing = required_args[len(args):]
            return {
                "valid": False,
                "error": f"{keyword_info.name} requires {len(required_args)} arguments: {', '.join(required_args)}. Missing: {', '.join(missing)}"
            }
        
        if len(args) > len(keyword_info.args):
            return {
                "valid": False,
                "error": f"{keyword_info.name} accepts at most {len(keyword_info.args)} arguments: {', '.join(keyword_info.args)}. Got {len(args)}"
            }
        
        return {"valid": True}
    
    def _get_keyword_suggestions(self, keyword_name: str) -> List[str]:
        """Get suggestions for similar keyword names."""
        suggestions = []
        keyword_lower = keyword_name.lower()
        
        # Find similar keywords
        for cached_name, keyword_info in self.keyword_cache.items():
            if (self._similarity_score(keyword_lower, cached_name) > 0.6 or
                any(word in cached_name for word in keyword_lower.split())):
                suggestions.append(keyword_info.name)
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _similarity_score(self, a: str, b: str) -> float:
        """Calculate similarity score between two strings."""
        if not a or not b:
            return 0.0
        
        # Simple Jaccard similarity
        set_a = set(a.split())
        set_b = set(b.split())
        
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        return intersection / union if union > 0 else 0.0
    
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
                for name, lib in self.libraries.items()
            },
            "failed_imports": self.failed_imports,
            "total_keywords": len(self.keyword_cache)
        }

# Global instance
_keyword_discovery = None

def get_keyword_discovery() -> DynamicKeywordDiscovery:
    """Get the global keyword discovery instance."""
    global _keyword_discovery
    if _keyword_discovery is None:
        _keyword_discovery = DynamicKeywordDiscovery()
    return _keyword_discovery
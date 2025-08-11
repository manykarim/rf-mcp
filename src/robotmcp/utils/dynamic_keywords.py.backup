"""Dynamic keyword discovery and execution for Robot Framework libraries."""

import importlib
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import re

# Import LibDoc integration
from robotmcp.utils.rf_libdoc_integration import get_rf_doc_storage

logger = logging.getLogger(__name__)

@dataclass
class KeywordInfo:
    """Information about a Robot Framework keyword."""
    name: str
    library: str
    method_name: str
    doc: str = ""
    short_doc: str = ""
    args: List[str] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    is_builtin: bool = False

@dataclass
class ParsedArguments:
    """Parsed positional and named arguments."""
    positional: List[str] = field(default_factory=list)
    named: Dict[str, str] = field(default_factory=dict)

@dataclass
class ArgumentInfo:
    """Information about a single argument from LibDoc signature parsing."""
    name: str
    type_hint: str
    default_value: Optional[str] = None
    is_varargs: bool = False
    is_kwargs: bool = False

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
        
        # Library exclusion rules - only one from each group can be loaded
        self.exclusion_groups = {
            'web_automation': ['Browser', 'SeleniumLibrary'],  # Browser OR Selenium, not both
        }
        self.excluded_libraries: set = set()  # Track which libraries are excluded
        
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
        
        # Keywords that modify the DOM or navigate pages
        self.dom_changing_patterns = [
            # Navigation patterns
            'go to', 'go back', 'go forward', 'open browser', 'reload', 'refresh',
            'new page', 'new tab', 'navigate',
            
            # Click interactions
            'click', 'double click', 'right click',
            
            # Input patterns
            'fill', 'input', 'type', 'clear', 'enter',
            
            # Selection patterns
            'select', 'check', 'uncheck', 'choose',
            
            # Form actions
            'submit', 'reset form', 'reset',
            
            # Drag and drop
            'drag', 'drop',
            
            # Keyboard actions
            'press keys', 'keyboard',
            
            # File operations
            'upload', 'choose file',
            
            # Alert/Dialog interactions
            'handle alert', 'accept alert', 'dismiss alert', 'input text into alert',
            
            # JavaScript that might modify DOM
            'execute javascript', 'execute async javascript',
            
            # Element state changes
            'focus', 'blur', 'hover', 'mouse over', 'mouse out', 'mouse down', 'mouse up'
        ]
        
        # Keywords that should NEVER trigger page source capture
        self.read_only_patterns = [
            # All getter keywords (word boundary)
            'get text', 'get value', 'get title', 'get source', 'get element', 'get cookie', 'get location',
            'get window', 'get selenium', 'get all', 'get current', 'get vertical', 'get horizontal',
            
            # All assertion keywords
            'should be', 'should contain', 'should not', 'should equal',
            
            # All configuration keywords (word boundary)
            'set selenium', 'set window', 'set timeout', 'set implicit', 'set browser',
            
            # Logging and capture keywords
            'log source', 'log title', 'log location', 'capture page', 'capture element', 'screenshot',
            
            # Wait keywords (passive)
            'wait for', 'wait until',
            
            # Browser management (non-DOM)
            'close browser', 'close window', 'maximize', 'minimize'
        ]
        
        self._initialize_libraries()
    
    def _initialize_libraries(self):
        """Initialize available Robot Framework libraries."""
        logger.info("Initializing Robot Framework libraries...")
        
        # Always try to initialize BuiltIn library first
        self._try_import_library('BuiltIn')
        
        # Try to import other common libraries first
        for library_name in self.common_libraries:
            if library_name != 'BuiltIn':
                self._try_import_library(library_name)
        
        # Apply exclusion logic after all libraries have been attempted
        self._resolve_library_conflicts()
        
        logger.info(f"Initialized {len(self.libraries)} libraries with {len(self.keyword_cache)} keywords")
    
    def _check_library_exclusion(self, library_name: str) -> bool:
        """
        Check if a library should be excluded due to exclusion rules.
        
        Args:
            library_name: Name of library to check
            
        Returns:
            bool: True if library should be excluded, False otherwise
        """
        # Check if library is already explicitly excluded
        if library_name in self.excluded_libraries:
            logger.debug(f"Library '{library_name}' is explicitly excluded")
            return True
        
        # Check exclusion groups
        for group_name, group_libraries in self.exclusion_groups.items():
            if library_name in group_libraries:
                # Check if any other library from this group is already loaded
                for other_lib in group_libraries:
                    if other_lib != library_name and other_lib in self.libraries:
                        logger.info(f"Excluding '{library_name}' because '{other_lib}' is already loaded (group: {group_name})")
                        self.excluded_libraries.add(library_name)
                        return True
        
        return False
    
    def _resolve_library_conflicts(self) -> None:
        """
        Resolve conflicts between loaded libraries by removing less preferred ones.
        Browser Library is preferred over SeleniumLibrary for web automation.
        """
        web_automation_libs = self.exclusion_groups.get('web_automation', [])
        
        # Check which web automation libraries were actually loaded
        loaded_web_libs = [lib for lib in web_automation_libs if lib in self.libraries]
        
        if len(loaded_web_libs) > 1:
            logger.info(f"Multiple web automation libraries loaded: {loaded_web_libs}")
            
            # Prefer Browser Library over SeleniumLibrary
            if 'Browser' in loaded_web_libs and 'SeleniumLibrary' in loaded_web_libs:
                logger.info("Both Browser Library and SeleniumLibrary loaded - removing SeleniumLibrary")
                self._remove_library('SeleniumLibrary')
                self.excluded_libraries.add('SeleniumLibrary')
            
        elif len(loaded_web_libs) == 1:
            logger.info(f"Single web automation library loaded: {loaded_web_libs[0]}")
        else:
            logger.debug("No web automation libraries loaded")
    
    def _remove_library(self, library_name: str) -> None:
        """
        Remove a library and its keywords from the discovery system.
        
        Args:
            library_name: Name of library to remove
        """
        if library_name not in self.libraries:
            return
        
        # Get library info
        lib_info = self.libraries[library_name]
        
        # Remove keywords from cache
        keywords_removed = 0
        for keyword_name in list(lib_info.keywords.keys()):
            if keyword_name.lower() in self.keyword_cache:
                # Only remove if this keyword belongs to the library being removed
                if self.keyword_cache[keyword_name.lower()].library == library_name:
                    del self.keyword_cache[keyword_name.lower()]
                    keywords_removed += 1
        
        # Remove library
        del self.libraries[library_name]
        
        logger.info(f"Removed library '{library_name}' and {keywords_removed} keywords")
    
    def _is_library_importable(self, library_name: str) -> bool:
        """Check if a library can be imported without actually importing it."""
        try:
            if library_name == 'Browser':
                import Browser
                return True
            elif library_name == 'SeleniumLibrary':
                import SeleniumLibrary
                return True
            else:
                __import__(library_name)
                return True
        except ImportError:
            return False
        except Exception:
            # Other errors still mean the module exists
            return True
    
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
                    # Create instance in a way that avoids browser dependency during keyword discovery
                    instance = SeleniumLibrary()
                    
                    # SeleniumLibrary sometimes throws errors during initialization
                    # when no browser is open. We'll catch this and still proceed
                    # with keyword extraction since the class methods exist.
                except ImportError:
                    logger.debug(f"SeleniumLibrary not available")
                    self.failed_imports[library_name] = "Not installed"
                    return False
                except Exception as e:
                    logger.debug(f"SeleniumLibrary initialization error (continuing anyway): {e}")
                    # Still try to create the instance for keyword discovery
                    try:
                        from SeleniumLibrary import SeleniumLibrary
                        # Create a "bare" instance that might not be fully functional
                        # but still allows us to discover its methods/keywords
                        instance = object.__new__(SeleniumLibrary)
                        # Initialize basic attributes without calling __init__
                        instance.__class__ = SeleniumLibrary
                    except Exception as e2:
                        logger.debug(f"SeleniumLibrary fallback failed: {e2}")
                        self.failed_imports[library_name] = f"Initialization failed: {e}"
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
            
            try:
                attr = getattr(instance, attr_name)
                if not callable(attr):
                    continue
                
                # Convert method name to Robot Framework keyword format
                keyword_name = self._method_to_keyword_name(attr_name)
                
                # Extract keyword information
                keyword_info = self._extract_keyword_info(library_name, keyword_name, attr_name, attr)
                lib_info.keywords[keyword_name] = keyword_info
                
            except Exception as e:
                # Some library methods may throw errors during inspection (e.g., SeleniumLibrary when no browser is open)
                # Skip these methods but continue with others
                logger.debug(f"Skipped method '{attr_name}' from {library_name}: {e}")
                continue
        
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
            
            # Create short documentation
            short_doc = self._create_short_doc(doc)
            
            return KeywordInfo(
                name=keyword_name,
                library=library_name,
                method_name=method_name,
                doc=doc,
                short_doc=short_doc,
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
                method_name=method_name,
                doc="Documentation not available",
                short_doc="Documentation not available"
            )
    
    def _create_short_doc(self, doc: str, max_length: int = 120) -> str:
        """Create a short documentation string from full documentation.
        
        Args:
            doc: Full documentation string
            max_length: Maximum length of short documentation
            
        Returns:
            str: Shortened documentation
        """
        if not doc:
            return ""
        
        # Clean and normalize whitespace
        doc = doc.strip()
        if not doc:
            return ""
        
        # Split into lines and get first meaningful line
        lines = doc.split('\n')
        first_line = ""
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('Tags:') and not line.startswith('Arguments:'):
                first_line = line
                break
        
        if not first_line:
            return ""
        
        # Remove common Robot Framework prefixes
        prefixes_to_remove = [
            "Keyword ", "The keyword ", "This keyword ", "Method ", "Function "
        ]
        for prefix in prefixes_to_remove:
            if first_line.startswith(prefix):
                first_line = first_line[len(prefix):]
                break
        
        # Ensure it ends with a period for consistency
        if first_line and not first_line.endswith(('.', '!', '?', ':')):
            if len(first_line) < max_length - 1:
                first_line += "."
        
        # Truncate if too long
        if len(first_line) > max_length:
            # Try to truncate at word boundary
            if ' ' in first_line[:max_length-3]:
                truncate_pos = first_line[:max_length-3].rfind(' ')
                first_line = first_line[:truncate_pos] + "..."
            else:
                first_line = first_line[:max_length-3] + "..."
        
        return first_line
    
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
        
        # More careful partial matching - only if no exact matches found
        partial_matches = []
        for cached_name, keyword_info in self.keyword_cache.items():
            # Only match if the search term is a significant part of the keyword
            # and not just a small substring that could be misleading
            search_words = normalized.split()
            cached_words = cached_name.split()
            
            # Check if most search words appear in the cached keyword
            word_matches = sum(1 for search_word in search_words
                             if any(search_word in cached_word for cached_word in cached_words))
            
            # Only consider it a match if at least half the words match
            # and avoid misleading matches like "select" matching "deselect"
            if word_matches >= len(search_words) / 2 and word_matches > 0:
                # Additional check: avoid opposite meanings
                if not (('select' in normalized and 'deselect' in cached_name.lower()) or
                        ('deselect' in normalized and 'select' in cached_name.lower() and 'deselect' not in cached_name.lower())):
                    partial_matches.append((word_matches, keyword_info))
        
        # Return the best partial match if found
        if partial_matches:
            # Sort by number of word matches (descending)
            partial_matches.sort(key=lambda x: x[0], reverse=True)
            best_match = partial_matches[0][1]
            logger.warning(f"Partial match for '{keyword_name}': using '{best_match.name}' from {best_match.library}")
            return best_match
        
        return None
    
    def suggest_similar_keywords(self, keyword_name: str, max_suggestions: int = 5) -> List[KeywordInfo]:
        """Find similar keywords for incorrect keyword names."""
        normalized = keyword_name.lower().strip()
        suggestions = []
        
        # Look for partial matches in keyword names
        for cached_name, keyword_info in self.keyword_cache.items():
            # Check if any word in the search term appears in the keyword
            search_words = normalized.split()
            keyword_words = cached_name.split()
            
            matches = sum(1 for search_word in search_words 
                         if any(search_word in kw_word for kw_word in keyword_words))
            
            if matches > 0:
                suggestions.append((matches, keyword_info))
        
        # Sort by number of matching words (descending) and return top suggestions
        suggestions.sort(key=lambda x: x[0], reverse=True)
        return [kw_info for _, kw_info in suggestions[:max_suggestions]]
    
    def get_keyword_count(self) -> int:
        """Get total number of discovered keywords."""
        return len(self.keyword_cache)
    
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
    
    def is_dom_changing_keyword(self, keyword_name: str) -> bool:
        """Determine if a keyword potentially changes the DOM or page state."""
        keyword_lower = keyword_name.lower().strip()
        
        # First check if it's explicitly marked as read-only
        for read_only_pattern in self.read_only_patterns:
            if read_only_pattern in keyword_lower:
                return False
        
        # Then check if it matches DOM-changing patterns
        for dom_pattern in self.dom_changing_patterns:
            if dom_pattern in keyword_lower:
                return True
        
        # Special cases for JavaScript keywords - only if they might modify
        if 'execute' in keyword_lower and 'javascript' in keyword_lower:
            return True
            
        # Default to False for unknown keywords (safer approach)
        return False
    
    async def execute_keyword(self, keyword_name: str, args: List[str], session_variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a keyword dynamically using Robot Framework's run_keyword for proper argument conversion."""
        keyword_info = self.find_keyword(keyword_name)
        
        if not keyword_info:
            return {
                "success": False,
                "error": f"Keyword '{keyword_name}' not found in any loaded library",
                "suggestions": self._get_keyword_suggestions(keyword_name)
            }
        
        try:
            # Parse arguments using Robot Framework's native ArgumentSpec if available
            parsed_args = self._parse_arguments_with_rf_spec(keyword_info, args)
            
            # Validate arguments (use original args for backward compatibility)
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
            
            # Try Robot Framework's run_keyword first for proper argument conversion
            try:
                result = self._execute_via_robot_framework(keyword_info, parsed_args, args)
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
            except Exception as rf_error:
                logger.debug(f"Robot Framework execution failed for {keyword_info.name}: {rf_error}")
                # Fall back to direct method call
                return self._execute_direct_method_call(keyword_info, parsed_args, args, session_variables)
            
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
    
    def _execute_via_robot_framework(self, keyword_info: KeywordInfo, parsed_args: ParsedArguments, original_args: List[str]) -> Any:
        """Execute keyword with Robot Framework-style argument conversion."""
        try:
            # For Browser Library keywords, convert arguments appropriately
            if keyword_info.library == "Browser":
                return self._execute_browser_keyword_with_conversion(keyword_info, parsed_args, original_args)
            else:
                # For other libraries, try the RF context approach
                return self._execute_with_rf_context(keyword_info, parsed_args, original_args)
            
        except Exception as e:
            logger.debug(f"RF execution failed: {e}")
            raise e
    
    def _execute_browser_keyword_with_conversion(self, keyword_info: KeywordInfo, parsed_args: ParsedArguments, original_args: List[str]) -> Any:
        """Execute Browser Library keyword with proper argument conversion."""
        try:
            # Get the library instance
            library = self.libraries[keyword_info.library]
            method = getattr(library.instance, keyword_info.method_name)
            
            # Convert arguments based on keyword-specific requirements
            converted_args, converted_kwargs = self._convert_browser_arguments(keyword_info, parsed_args, original_args)
            
            # Execute the method with converted arguments and keyword arguments
            if converted_kwargs:
                result = method(*converted_args, **converted_kwargs)
            else:
                result = method(*converted_args)
            return result
            
        except Exception as e:
            logger.debug(f"Browser keyword execution failed: {e}")
            raise e
    
    def _convert_browser_arguments(self, keyword_info: KeywordInfo, parsed_args: ParsedArguments, original_args: List[str]) -> Tuple[List[Any], Dict[str, Any]]:
        """Convert parsed arguments to appropriate types for Browser Library keywords.
        
        Returns:
            Tuple of (positional_args, keyword_args)
        """
        converted_args = []
        converted_kwargs = {}
        
        # Process positional arguments
        converted_args = parsed_args.positional.copy()
        
        # Handle specific keyword argument conversions
        keyword_name = keyword_info.name.lower()
        
        # Try to import Browser data types for advanced conversions
        try:
            from Browser.utils.data_types import (
                SelectAttribute, SupportedBrowsers, ElementState, 
                MouseButton, KeyAction, ViewportDimensions
            )
            browser_library_available = True
        except ImportError:
            browser_library_available = False
            logger.debug("Browser library data types not available for advanced conversion")
        
        # Handle Browser-specific conversions if library is available
        if browser_library_available:
            
            if "new browser" in keyword_name:
                # new_browser(browser: SupportedBrowsers, **kwargs)
                
                # Handle browser type (first positional arg or 'browser' named arg)
                browser_type = None
                browser_is_named = False
                
                if converted_args:
                    # Browser provided as positional argument
                    browser_type = converted_args[0].lower()
                elif 'browser' in parsed_args.named:
                    # Browser provided as named argument
                    browser_type = parsed_args.named['browser'].lower()
                    browser_is_named = True
                    
                if browser_type:
                    # Convert browser string to enum
                    if browser_type in ['chromium', 'chrome']:
                        browser_enum = SupportedBrowsers.chromium
                    elif browser_type == 'firefox':
                        browser_enum = SupportedBrowsers.firefox
                    elif browser_type == 'webkit':
                        browser_enum = SupportedBrowsers.webkit
                    else:
                        browser_enum = getattr(SupportedBrowsers, browser_type, SupportedBrowsers.chromium)
                    
                    if browser_is_named:
                        # Add as named argument, don't add to positional
                        converted_kwargs['browser'] = browser_enum
                    else:
                        # Replace the positional argument
                        converted_args[0] = browser_enum
                else:
                    # Default browser if none specified - add as positional to maintain compatibility
                    converted_args.insert(0, SupportedBrowsers.chromium)
                
                # Convert other named arguments to appropriate types
                for key, value in parsed_args.named.items():
                    if key == 'browser':
                        continue  # Already handled above
                    elif key == 'headless':
                        converted_kwargs[key] = self._convert_boolean_string(value)
                    elif key in ['timeout', 'slowmo']:
                        try:
                            converted_kwargs[key] = float(value) if '.' in value else int(value)
                        except ValueError:
                            converted_kwargs[key] = value  # Keep as string if conversion fails
                    else:
                        # Keep other named arguments as strings
                        converted_kwargs[key] = value
                        
            elif "select options by" in keyword_name:
                # select_options_by(selector: str, attribute: SelectAttribute, *values)
                if len(converted_args) >= 2:
                    # Second arg: attribute (convert to SelectAttribute enum)
                    attr_str = converted_args[1].lower()
                    if attr_str in ['value', 'val']:
                        converted_args[1] = SelectAttribute.value
                    elif attr_str in ['label', 'text']:
                        converted_args[1] = SelectAttribute.label
                    elif attr_str in ['index', 'idx']:
                        converted_args[1] = SelectAttribute.index
                    else:
                        converted_args[1] = getattr(SelectAttribute, attr_str, SelectAttribute.value)
                        
            elif "wait for elements state" in keyword_name:
                # wait_for_elements_state(selector: str, state: ElementState, timeout: str = "10s")
                if len(converted_args) >= 2:
                    # Second arg: state (convert to ElementState)
                    state_str = converted_args[1].lower()
                    try:
                        converted_args[1] = getattr(ElementState, state_str)
                    except AttributeError:
                        pass  # Keep as string if enum value not found
                        
                # Handle timeout as named argument
                if 'timeout' in parsed_args.named:
                    converted_kwargs['timeout'] = parsed_args.named['timeout']
            
        # Handle common named arguments for all Browser keywords using LibDoc type detection
        for key, value in parsed_args.named.items():
            if key not in converted_kwargs:  # Don't override specific conversions above
                # Get parameter type from LibDoc
                param_type = self._get_parameter_type_from_libdoc(key, keyword_info.name, keyword_info.library)
                
                if param_type:
                    # Use LibDoc-detected type for conversion
                    converted_kwargs[key] = self._convert_string_value(value, param_type)
                else:
                    # No LibDoc type found - keep as string
                    converted_kwargs[key] = value
            
        return (converted_args, converted_kwargs)
    
    def _execute_with_rf_context(self, keyword_info: KeywordInfo, parsed_args: ParsedArguments, original_args: List[str]) -> Any:
        """Execute keyword in a minimal Robot Framework context."""
        try:
            from robot.api import TestSuite
            from robot.running.model import TestCase, Keyword
            from robot.result import ExecutionResult
            from robot.conf import RobotSettings
            import tempfile
            import os
            
            # Create a minimal test suite
            suite = TestSuite(name="TempSuite")
            
            # Import the required library
            suite.resource.imports.library(keyword_info.library)
            
            # Create a test case with the keyword call
            test = TestCase(name="TempTest")
            
            # Reconstruct arguments for Robot Framework - combine positional and named
            rf_args = parsed_args.positional.copy()
            for key, value in parsed_args.named.items():
                rf_args.append(f"{key}={value}")
            
            keyword_call = Keyword(name=keyword_info.name, args=rf_args)
            test.body.append(keyword_call)
            suite.tests.append(test)
            
            # Execute the suite in memory
            with tempfile.TemporaryDirectory() as temp_dir:
                output_file = os.path.join(temp_dir, "output.xml")
                
                # Run the suite
                result = suite.run(outputdir=temp_dir, output=output_file)
                
                # Parse the result to get the keyword result
                execution_result = ExecutionResult(output_file)
                test_result = execution_result.suite.tests[0]
                keyword_result = test_result.keywords[0]
                
                if keyword_result.passed:
                    # Extract return value if available
                    # This is a simplified approach - RF's return values are complex
                    return keyword_result.messages[-1].message if keyword_result.messages else "OK"
                else:
                    raise Exception(f"Keyword execution failed: {keyword_result.message}")
                
        except Exception as e:
            logger.debug(f"RF context execution failed: {e}")
            raise e
    
    def _ensure_library_imported(self, library_name: str, builtin: Any) -> None:
        """Ensure library is imported in Robot Framework context."""
        try:
            # Try to import the library using RF's import mechanism
            builtin.import_library(library_name)
            logger.debug(f"Imported library {library_name} in RF context")
        except Exception as e:
            # Library might already be imported or import might fail
            # This is not critical - RF will handle missing libraries appropriately
            logger.debug(f"Library import for {library_name} in RF context: {e}")
    
    def _execute_direct_method_call(self, keyword_info: KeywordInfo, parsed_args: ParsedArguments, original_args: List[str], session_variables: Dict[str, Any]) -> Dict[str, Any]:
        """Fall back to direct method call execution (original implementation)."""
        try:
            # Get library instance
            library = self.libraries[keyword_info.library]
            method = getattr(library.instance, keyword_info.method_name)
            
            # Handle different types of method calls with parsed arguments
            try:
                if keyword_info.is_builtin and hasattr(library.instance, '_context'):
                    # BuiltIn library methods might need context - use original args for BuiltIn
                    result = method(*original_args)
                else:
                    # Regular library methods - handle Browser Library specially
                    if keyword_info.library == "Browser":
                        # Use the converted Browser arguments
                        converted_args, converted_kwargs = self._convert_browser_arguments(keyword_info, parsed_args, original_args)
                        if converted_kwargs:
                            result = method(*converted_args, **converted_kwargs)
                        else:
                            result = method(*converted_args)
                    elif keyword_info.name == "Create List":
                        # Collections.Create List takes variable arguments (use positional only)
                        result = method(*parsed_args.positional)
                    elif keyword_info.name == "Set Variable":
                        # Set Variable only takes one argument (the value)
                        value = parsed_args.positional[0] if parsed_args.positional else None
                        result = method(value)
                    else:
                        # For other libraries, use positional args only (most don't support **kwargs)
                        result = method(*parsed_args.positional)
            except TypeError as e:
                # Try calling with different argument patterns
                if "takes" in str(e) and "positional argument" in str(e):
                    # Try calling with original args as a single list argument
                    try:
                        result = method(original_args)
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
                "error": f"Error in direct method call for {keyword_info.library}.{keyword_info.name}: {str(e)}",
                "keyword_info": {
                    "name": keyword_info.name,
                    "library": keyword_info.library,
                    "args": keyword_info.args,
                    "doc": keyword_info.doc
                }
            }
    
    def _validate_arguments(self, keyword_info: KeywordInfo, args: List[str]) -> Dict[str, Any]:
        """Validate arguments for a keyword."""
        
        # Special handling for keywords that use *args and **kwargs (like RequestsLibrary)
        if self._uses_varargs_kwargs(keyword_info):
            # For *args/**kwargs keywords, we allow any number of arguments
            # since they handle flexible argument parsing internally
            return {"valid": True}
        
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
    
    def _uses_varargs_kwargs(self, keyword_info: KeywordInfo) -> bool:
        """Check if a keyword uses *args and **kwargs for flexible argument handling."""
        # Check if the keyword signature indicates varargs/kwargs usage
        args_list = keyword_info.args
        
        # Common patterns for *args/**kwargs in Robot Framework
        if len(args_list) == 2 and args_list == ['args', 'kwargs']:
            return True
        
        # Also check for RequestsLibrary specific patterns
        if keyword_info.library == "RequestsLibrary":
            # Most RequestsLibrary keywords use flexible argument handling
            flexible_keywords = [
                "post on session", "get on session", "put on session", 
                "delete on session", "patch on session", "head on session",
                "options on session", "post request", "get request"
            ]
            if any(pattern in keyword_info.name.lower() for pattern in flexible_keywords):
                return True
        
        return False
    
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
    
    def _parse_arguments(self, args: List[str]) -> ParsedArguments:
        """Parse Robot Framework-style arguments into positional and named arguments.
        
        Args:
            args: List of string arguments, potentially containing 'key=value' named arguments
            
        Returns:
            ParsedArguments with separated positional and named arguments
        """
        parsed = ParsedArguments()
        
        for arg in args:
            # Check if this is a named argument (contains = and is not clearly a value)
            if '=' in arg and self._is_named_argument(arg):
                # Split on first = to separate key and value
                key, value = arg.split('=', 1)
                parsed.named[key.strip()] = value.strip()
            else:
                # This is a positional argument
                parsed.positional.append(arg)
        
        return parsed
    
    def _is_named_argument(self, arg: str) -> bool:
        """Determine if an argument string is a named argument vs a value containing =.
        
        Named arguments typically follow patterns like:
        - headless=true
        - timeout=5000
        - browser=chromium
        
        NOT named arguments:
        - xpath=//div[@class='test']  (XPath expression)
        - url=https://example.com?param=value  (URL with parameters)
        - css=[data-testid="button"]  (CSS selector)
        """
        if '=' not in arg:
            return False
            
        key, value = arg.split('=', 1)
        key = key.strip().lower()
        value = value.strip()
        
        # Common Robot Framework named argument patterns
        common_named_args = {
            'headless', 'timeout', 'browser', 'width', 'height', 'viewport', 
            'slowmo', 'devtools', 'args', 'env', 'proxy', 'downloads',
            'permissions', 'geolocation', 'locale', 'timezone', 'colorscheme',
            'reducedmotion', 'forcedcolors', 'acceptdownloads', 'bypasscsp',
            'strictselectors', 'serviceworkers', 'recordhar', 'recordvideo',
            'state', 'method', 'data', 'json', 'headers', 'cookies'
        }
        
        # Check if the key matches common named argument patterns
        if key in common_named_args:
            return True
            
        # Check for selector prefixes that indicate values, not named args
        selector_prefixes = ['xpath', 'css', 'id', 'name', 'class', 'tag', 'text', 'url', 'href']
        if any(key.startswith(prefix) for prefix in selector_prefixes):
            return False
            
        # Check if value looks like a typical selector or URL
        if (value.startswith(('http://', 'https://', '//', '//')) or 
            value.startswith(('[', '#', '.')) or
            '@' in value):  # Likely XPath, CSS, or URL
            return False
            
        # Default to named argument if key is alphanumeric
        return key.replace('_', '').replace('-', '').isalnum()
    
    def _parse_arguments_with_rf_spec(self, keyword_info: KeywordInfo, args: List[str]) -> ParsedArguments:
        """Parse arguments using Robot Framework's native ArgumentSpec if available.
        
        This is a more accurate approach that uses RF's actual argument parsing logic
        to determine which arguments are named vs positional based on the keyword's
        parameter specification.
        """
        try:
            from robot.running.arguments import ArgumentSpec
            from robot.running.arguments.argumentresolver import ArgumentResolver
            
            # Try to create ArgumentSpec from keyword info
            if hasattr(keyword_info, 'args') and keyword_info.args:
                # Create a basic ArgumentSpec - this may need refinement
                # based on how keyword_info.args is structured
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
                
                logger.debug(f"RF ArgumentSpec parsing - Positional: {positional}, Named: {named}")
                return parsed
                
        except ImportError:
            logger.debug("Robot Framework ArgumentSpec not available, using fallback parsing")
        except Exception as e:
            logger.debug(f"RF ArgumentSpec parsing failed: {e}, using fallback parsing")
            
        # Fall back to our custom parsing logic
        return self._parse_arguments(args)
    
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

    def _parse_argument_signature(self, signature: str) -> List[ArgumentInfo]:
        """Parse Robot Framework argument signature to extract type information."""
        
        if not signature or not signature.strip():
            return []
        
        # Split by comma, but respect nested brackets/parentheses
        args = self._split_signature_args(signature)
        parsed_args = []
        
        for arg in args:
            arg = arg.strip()
            if not arg:
                continue
                
            # Parse individual argument
            arg_info = self._parse_single_argument(arg)
            if arg_info:
                parsed_args.append(arg_info)
        
        return parsed_args

    def _split_signature_args(self, signature: str) -> List[str]:
        """Split signature by comma, respecting nested structures."""
        args = []
        current_arg = ""
        bracket_depth = 0
        paren_depth = 0
        
        for char in signature:
            if char in '[{':
                bracket_depth += 1
            elif char in ']}':
                bracket_depth -= 1
            elif char == ')':
                paren_depth -= 1
            elif char == '(':
                paren_depth += 1
            elif char == ',' and bracket_depth == 0 and paren_depth == 0:
                args.append(current_arg)
                current_arg = ""
                continue
                
            current_arg += char
        
        if current_arg:
            args.append(current_arg)
        
        return args

    def _parse_single_argument(self, arg: str) -> Optional[ArgumentInfo]:
        """Parse a single argument like 'force: bool = False'."""
        
        # Handle *args and **kwargs
        if arg.startswith('**'):
            name = arg[2:].strip()
            return ArgumentInfo(name=name, type_hint='dict', is_kwargs=True)
        elif arg.startswith('*') and not arg.startswith('**'):
            name = arg[1:].strip()
            # Handle bare * separator
            if not name or name == ',':
                return None
            return ArgumentInfo(name=name, type_hint='list', is_varargs=True)
        
        # Regular argument: name: type = default
        # Match pattern: name : type = default (all parts optional except name)
        match = re.match(r'^([^:=]+)(?:\s*:\s*([^=]+))?(?:\s*=\s*(.+))?$', arg)
        
        if not match:
            return None
        
        name = match.group(1).strip()
        type_hint = match.group(2).strip() if match.group(2) else 'str'  # Default to str
        default_value = match.group(3).strip() if match.group(3) else None
        
        return ArgumentInfo(
            name=name,
            type_hint=type_hint,
            default_value=default_value
        )

    def _detect_argument_type(self, type_hint: str) -> str:
        """Detect the basic type from a type hint."""
        
        type_hint_lower = type_hint.lower().strip()
        
        # Boolean types
        if any(indicator in type_hint_lower for indicator in ['bool', 'boolean']):
            return 'bool'
        
        # Dictionary types (check before int due to dict[str, int] patterns)
        if any(indicator in type_hint_lower for indicator in ['dict', 'mapping']) or 'viewportdimensions' in type_hint_lower:
            return 'dict'
        
        # List types (check before int due to list[int] patterns)  
        if any(indicator in type_hint_lower for indicator in ['list', 'sequence']):
            return 'list'
        
        # Integer types
        if any(indicator in type_hint_lower for indicator in ['int', 'integer']) and 'point' not in type_hint_lower:
            return 'int'
        
        # Float types  
        if any(indicator in type_hint_lower for indicator in ['float', 'double']):
            return 'float'
        
        # Enum types (e.g., SupportedBrowsers, MouseButton)
        if any(indicator in type_hint_lower for indicator in ['enum', 'supportedbrowsers', 'mousebutton', 'keyboardmodifier', 'colorscheme', 'forcedcolors']):
            return 'enum'
        
        # Timedelta
        if 'timedelta' in type_hint_lower:
            return 'timedelta'
        
        # Tuple types
        if 'tuple' in type_hint_lower:
            return 'tuple'
        
        # Default to string
        return 'str'

    def _convert_string_value(self, value: str, target_type: str):
        """Convert string value to target type."""
        
        if target_type == 'bool':
            return value.lower().strip() in ['true', '1', 'yes', 'on']
        
        elif target_type == 'int':
            try:
                return int(value)
            except ValueError:
                return value
        
        elif target_type == 'float':
            try:
                return float(value)
            except ValueError:
                return value
        
        elif target_type in ['dict', 'list', 'tuple']:
            try:
                # First try ast.literal_eval for safe evaluation of Python literals
                import ast
                result = ast.literal_eval(value)
                
                # Convert to target type if needed
                if target_type == 'tuple' and isinstance(result, (list, dict)):
                    return tuple(result) if isinstance(result, list) else tuple(result.items())
                elif target_type == 'list' and isinstance(result, (tuple, dict)):
                    return list(result) if isinstance(result, tuple) else list(result.items())
                elif target_type == 'dict' and isinstance(result, (list, tuple)):
                    # Try to convert list/tuple of pairs to dict
                    return dict(result) if len(result) > 0 and all(len(item) == 2 for item in result if isinstance(item, (list, tuple))) else result
                
                return result
                
            except (ValueError, SyntaxError):
                # If ast.literal_eval fails, try JSON parsing
                try:
                    import json
                    result = json.loads(value)
                    
                    # Convert to target type if needed
                    if target_type == 'tuple':
                        return tuple(result) if isinstance(result, list) else result
                    elif target_type == 'list':
                        return list(result) if isinstance(result, dict) else result
                    
                    return result
                    
                except json.JSONDecodeError:
                    # If both fail, return original string
                    return value
        
        # For enum, timedelta, etc., return as string for now
        return value

    def _get_libdoc_argument_info(self, keyword_name: str, library_name: str = None) -> List[ArgumentInfo]:
        """Get argument information from LibDoc for a keyword."""
        
        # Get LibDoc storage
        rf_storage = get_rf_doc_storage()
        
        if not rf_storage.is_available():
            return []
        
        # Find keyword in LibDoc
        keyword_info = rf_storage.find_keyword(keyword_name)
        
        if not keyword_info:
            return []
        
        # If library specified, ensure it matches
        if library_name and keyword_info.library.lower() != library_name.lower():
            return []
        
        # Parse argument signature from LibDoc args
        if keyword_info.args:
            # LibDoc provides a list of argument strings, join them
            signature = ", ".join(keyword_info.args) if isinstance(keyword_info.args, list) else str(keyword_info.args)
            return self._parse_argument_signature(signature)
        
        return []

    def _get_parameter_type_from_libdoc(self, key: str, keyword_name: str, library_name: str = None) -> Optional[str]:
        """Get parameter type from LibDoc for a specific keyword argument."""
        
        # Get argument information from LibDoc
        arg_infos = self._get_libdoc_argument_info(keyword_name, library_name)
        
        # Find the matching argument
        for arg_info in arg_infos:
            if arg_info.name.lower() == key.lower():
                return self._detect_argument_type(arg_info.type_hint)
        
        return None

    def _is_boolean_parameter(self, key: str, value: str, keyword_name: str = None, library_name: str = None) -> bool:
        """Determine if a parameter should be converted to boolean using LibDoc."""
        
        # Use LibDoc to get the actual parameter type
        if keyword_name:
            param_type = self._get_parameter_type_from_libdoc(key, keyword_name, library_name)
            return param_type == 'bool'
        
        return False
    
    def _convert_boolean_string(self, value: str) -> bool:
        """Convert string to boolean value."""
        return value.lower().strip() in ['true', '1', 'yes', 'on']

    def get_library_exclusion_info(self) -> Dict[str, Any]:
        """
        Get information about library exclusions.
        
        Returns:
            dict: Information about exclusion groups and excluded libraries
        """
        return {
            "exclusion_groups": self.exclusion_groups,
            "excluded_libraries": list(self.excluded_libraries),
            "loaded_libraries": list(self.libraries.keys()),
            "failed_imports": dict(self.failed_imports),
            "preference_applied": {
                "browser_available": self._is_library_importable('Browser'),
                "selenium_available": self._is_library_importable('SeleniumLibrary'),
                "active_web_library": next((lib for lib in ['Browser', 'SeleniumLibrary'] if lib in self.libraries), None)
            }
        }

# Global instance
_keyword_discovery = None

def get_keyword_discovery() -> DynamicKeywordDiscovery:
    """Get the global keyword discovery instance."""
    global _keyword_discovery
    if _keyword_discovery is None:
        _keyword_discovery = DynamicKeywordDiscovery()
    return _keyword_discovery
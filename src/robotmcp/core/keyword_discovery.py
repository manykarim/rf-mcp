"""Keyword discovery and caching functionality."""

import inspect
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from robotmcp.models.library_models import LibraryInfo, KeywordInfo

logger = logging.getLogger(__name__)


class KeywordDiscovery:
    """Handles keyword extraction from library instances and keyword caching."""
    
    def __init__(self):
        # Cache of fully-qualified library.keyword entries
        self.keyword_cache: Dict[str, KeywordInfo] = {}
        # Map of simple keyword name -> list of KeywordInfo objects from all libraries
        self.simple_keyword_map: Dict[str, List[KeywordInfo]] = {}

        # Keywords that modify the DOM or navigate pages
        self.dom_changing_patterns = [
            'click', 'fill', 'type', 'select', 'check', 'uncheck',
            'navigate', 'go to', 'reload', 'back', 'forward',
            'submit', 'clear', 'upload', 'download',
            'new page', 'close page', 'switch', 'open browser', 'close browser'
        ]
    
    def extract_library_info(self, library_name: str, instance: Any) -> LibraryInfo:
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
                keyword_name = self.method_to_keyword_name(attr_name)
                
                # Extract keyword information
                keyword_info = self.extract_keyword_info(library_name, keyword_name, attr_name, attr)
                lib_info.keywords[keyword_name] = keyword_info
                
            except Exception as e:
                # Some library methods may throw errors during inspection (e.g., SeleniumLibrary when no browser is open)
                # Skip these methods but continue with others
                logger.debug(f"Skipped method '{attr_name}' from {library_name}: {e}")
                continue
        
        return lib_info
    
    def method_to_keyword_name(self, method_name: str) -> str:
        """Convert Python method name to Robot Framework keyword name."""
        # Convert snake_case to Title Case
        words = method_name.split('_')
        return ' '.join(word.capitalize() for word in words)
    
    def extract_keyword_info(self, library_name: str, keyword_name: str, method_name: str, method: Callable) -> KeywordInfo:
        """Extract information about a specific keyword."""
        try:
            # Get method signature
            sig = inspect.signature(method)
            args: List[str] = []
            defaults = {}

            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue

                # Build argument representation preserving kind and default
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    arg_repr = f"*{param_name}"
                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                    arg_repr = f"**{param_name}"
                else:
                    arg_repr = param_name

                if param.default != inspect.Parameter.empty:
                    defaults[param_name] = param.default
                    default_str = repr(param.default) if param.default is not None else 'None'
                    arg_repr = f"{arg_repr}={default_str}"

                args.append(arg_repr)
            
            # Get documentation
            doc = inspect.getdoc(method) or ""
            
            # Extract tags from docstring (Robot Framework convention)
            tags = []
            if doc:
                tag_match = re.search(r'Tags:\\s*(.+)', doc)
                if tag_match:
                    tags = [tag.strip() for tag in tag_match.group(1).split(',')]
            
            # Create short documentation
            short_doc = self.create_short_doc(doc)
            
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
            logger.debug(f"Failed to extract keyword info for {method_name}: {e}")
            return KeywordInfo(
                name=keyword_name,
                library=library_name,
                method_name=method_name
            )
    
    def create_short_doc(self, doc: str) -> str:
        """Create a short version of the documentation."""
        if not doc:
            return ""
        
        # Take first sentence or first line
        lines = doc.strip().split('\\n')
        first_line = lines[0].strip()
        
        # If first line ends with a period, use it as short doc
        if first_line.endswith('.'):
            return first_line
        
        # Otherwise, find first sentence
        sentences = first_line.split('. ')
        return sentences[0] + ('.' if not sentences[0].endswith('.') else '')
    
    def add_keywords_to_cache(self, lib_info: LibraryInfo) -> None:
        """Add keywords from library to the cache."""
        for keyword_name, keyword_info in lib_info.keywords.items():
            # Use library.keyword format as key to avoid overwriting between libraries
            cache_key = f"{lib_info.name.lower()}.{keyword_name.lower()}"
            self.keyword_cache[cache_key] = keyword_info

            # Track all libraries providing this keyword for ambiguity detection
            simple_key = keyword_name.lower()
            existing = self.simple_keyword_map.setdefault(simple_key, [])
            if all(info.library != keyword_info.library for info in existing):
                existing.append(keyword_info)
    
    def remove_keywords_from_cache(self, lib_info: LibraryInfo) -> int:
        """Remove keywords from a specific library from the cache."""
        keywords_removed = 0
        library_prefix = f"{lib_info.name.lower()}."
        
        # Remove library-specific keys
        keys_to_remove = [key for key in list(self.keyword_cache.keys()) if key.startswith(library_prefix)]
        for key in keys_to_remove:
            del self.keyword_cache[key]
            keywords_removed += 1

        # Remove from simple map
        for keyword_name in list(lib_info.keywords.keys()):
            simple_key = keyword_name.lower()
            if simple_key in self.simple_keyword_map:
                self.simple_keyword_map[simple_key] = [
                    info for info in self.simple_keyword_map[simple_key]
                    if info.library != lib_info.name
                ]
                if not self.simple_keyword_map[simple_key]:
                    del self.simple_keyword_map[simple_key]
                keywords_removed += 1
        
        return keywords_removed
    
    def find_keyword(self, keyword_name: str, active_library: str = None) -> Optional[KeywordInfo]:
        """Find a keyword by name with fuzzy matching, optionally filtering by active library."""
        if not keyword_name:
            return None

        # Support fully qualified names Library.Keyword
        if '.' in keyword_name:
            lib, kw = keyword_name.split('.', 1)
            return self.find_keyword(kw, active_library=lib)

        normalized = keyword_name.lower().strip()

        # If active_library is specified, try library-specific key first
        if active_library:
            library_specific_key = f"{active_library.lower()}.{normalized}"
            if library_specific_key in self.keyword_cache:
                logger.debug(f"Found exact library match: {keyword_name} in {active_library}")
                return self.keyword_cache[library_specific_key]

        builtin_libraries = ['BuiltIn', 'Collections', 'String', 'DateTime', 'OperatingSystem', 'Process']

        # Search using simple keyword map
        if normalized in self.simple_keyword_map:
            candidates = self.simple_keyword_map[normalized]
            if active_library:
                candidates = [c for c in candidates if c.library == active_library or c.library in builtin_libraries]
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                raise ValueError(f"Keyword '{keyword_name}' is defined in multiple libraries: {[c.library for c in candidates]}")

        # Try common variations against library-specific cache
        variations = [
            normalized.replace(' ', ''),  # Remove spaces
            normalized.replace('_', ' '),  # Replace underscores
            normalized.replace('-', ' '),  # Replace hyphens
        ]

        for variation in variations:
            if variation in self.simple_keyword_map:
                candidates = self.simple_keyword_map[variation]
                if active_library:
                    candidates = [c for c in candidates if c.library == active_library or c.library in builtin_libraries]
                if len(candidates) == 1:
                    return candidates[0]
                if len(candidates) > 1:
                    raise ValueError(
                        f"Keyword '{keyword_name}' is defined in multiple libraries: {[c.library for c in candidates]}"
                    )

        # Fuzzy matching across library-specific keys
        best_match = None
        best_score = 0

        for cache_key, keyword_info in self.keyword_cache.items():
            cached_name = cache_key.split('.', 1)[1] if '.' in cache_key else cache_key
            if normalized in cached_name:
                score = len(normalized) / len(cached_name)
            elif cached_name in normalized:
                score = len(cached_name) / len(normalized)
            else:
                score = 0

            if score > best_score:
                best_score = score
                best_match = keyword_info

        if best_match and best_score >= 0.6:
            simple_key = best_match.name.lower()
            candidates = self.simple_keyword_map.get(simple_key, [])
            if active_library:
                candidates = [c for c in candidates if c.library == active_library or c.library in builtin_libraries]
            if len(candidates) == 1:
                logger.debug(
                    f"Fuzzy matched '{keyword_name}' to '{best_match.name}' from {best_match.library} (score: {best_score:.2f})"
                )
                return best_match
            if len(candidates) > 1:
                raise ValueError(
                    f"Keyword '{keyword_name}' is defined in multiple libraries: {[c.library for c in candidates]}"
                )

        return None
    
    def get_keyword_suggestions(self, keyword_name: str, limit: int = 5) -> List[str]:
        """Get keyword suggestions based on partial match."""
        if not keyword_name:
            return []
        
        normalized = keyword_name.lower().strip()
        suggestions = []
        
        for cached_name, keyword_info in self.keyword_cache.items():
            if normalized in cached_name or any(word in cached_name for word in normalized.split()):
                suggestions.append(keyword_info.name)
        
        return suggestions[:limit]
    
    def get_keywords_by_library(self, library_name: str) -> List[KeywordInfo]:
        """Get all keywords from a specific library."""
        library_prefix = f"{library_name.lower()}."
        return [info for key, info in self.keyword_cache.items() 
                if key.startswith(library_prefix) or info.library == library_name]
    
    def get_all_keywords(self) -> List[KeywordInfo]:
        """Get all cached keywords."""
        return list(self.keyword_cache.values())
    
    def get_keyword_count(self) -> int:
        """Get total number of cached keywords."""
        return len(self.keyword_cache)
    
    def is_dom_changing_keyword(self, keyword_name: str) -> bool:
        """Check if a keyword likely changes the DOM."""
        keyword_lower = keyword_name.lower()
        return any(pattern in keyword_lower for pattern in self.dom_changing_patterns)
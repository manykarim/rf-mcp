"""Keyword discovery and caching functionality."""

import inspect
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from robotmcp.models.library_models import LibraryInfo, KeywordInfo

logger = logging.getLogger(__name__)


class KeywordDiscovery:
    """Handles keyword extraction from library instances and keyword caching."""
    
    def __init__(self):
        self.keyword_cache: Dict[str, KeywordInfo] = {}
        self.shadowed_keywords: Dict[str, List[Tuple[str, KeywordInfo]]] = {}  # keyword -> [(library, info), ...]
        self.ambiguous_keywords: Set[str] = set()  # Keywords that exist in multiple libraries

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

        auto_keywords = getattr(instance, 'ROBOT_AUTO_KEYWORDS', True)
        
        # Get all public methods that could be keywords
        for attr_name in dir(instance):
            if attr_name.startswith('_'):
                continue
            
            try:
                attr = getattr(instance, attr_name)
                if not inspect.isroutine(attr):
                    continue

                decorated = any(
                    hasattr(attr, marker)
                    for marker in ('robot_name', 'robot_tags', 'robot_types')
                )
                robot_name_attr = getattr(attr, 'robot_name', None)

                # Respect ROBOT_AUTO_KEYWORDS flag: when disabled, only include decorated keywords
                if decorated:
                    keyword_name = robot_name_attr or self.method_to_keyword_name(attr_name)
                elif auto_keywords:
                    keyword_name = self.method_to_keyword_name(attr_name)
                else:
                    continue
                
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
            
            # Also maintain a simple lookup for backward compatibility
            simple_key = keyword_name.lower()

            if simple_key in self.keyword_cache:
                existing = self.keyword_cache[simple_key]
                existing_lib = getattr(existing, 'library', 'unknown')
                if existing_lib != lib_info.name:
                    # Track shadowed keyword
                    if simple_key not in self.shadowed_keywords:
                        self.shadowed_keywords[simple_key] = [(existing_lib, existing)]
                    self.shadowed_keywords[simple_key].append((lib_info.name, keyword_info))
                    self.ambiguous_keywords.add(simple_key)
                    logger.warning(
                        f"Keyword '{keyword_name}' from {lib_info.name} shadows "
                        f"existing keyword from {existing_lib}. Use 'Library.{keyword_name}' for disambiguation."
                    )
            else:
                self.keyword_cache[simple_key] = keyword_info
    
    def remove_keywords_from_cache(self, lib_info: LibraryInfo) -> int:
        """Remove keywords from a specific library from the cache."""
        keywords_removed = 0
        library_prefix = f"{lib_info.name.lower()}."
        
        # Remove library-specific keys
        keys_to_remove = [key for key in self.keyword_cache.keys() if key.startswith(library_prefix)]
        for key in keys_to_remove:
            del self.keyword_cache[key]
            keywords_removed += 1
        
        # Remove simple keys that belong to this library
        for keyword_name in list(lib_info.keywords.keys()):
            simple_key = keyword_name.lower()
            if simple_key in self.keyword_cache:
                # Only remove if this keyword belongs to the library being removed
                if self.keyword_cache[simple_key].library == lib_info.name:
                    del self.keyword_cache[simple_key]
                    keywords_removed += 1
        
        return keywords_removed
    
    def find_keyword(self, keyword_name: str, active_library: str = None, session_libraries: List[str] = None) -> Optional[KeywordInfo]:
        """Find a keyword by name with fuzzy matching, optionally filtering by active library and session libraries."""
        if not keyword_name:
            return None
        
        normalized = keyword_name.lower().strip()
        
        # If active_library is specified, try library-specific key first
        if active_library:
            library_specific_key = f"{active_library.lower()}.{normalized}"
            if library_specific_key in self.keyword_cache:
                logger.debug(f"Found exact library match: {keyword_name} in {active_library}")
                return self.keyword_cache[library_specific_key]
        
        # PHASE 1: Session-aware keyword filtering
        search_cache = self.keyword_cache
        
        # Priority 1: Filter by session libraries (if provided)
        if session_libraries:
            logger.debug(f"Session-aware keyword filtering for '{keyword_name}' in libraries: {session_libraries}")
            
            # Get library priorities for proper ordering
            from robotmcp.config.library_registry import get_library_config
            
            # Sort session libraries by priority (lower number = higher priority)
            prioritized_libraries = []
            for lib_name in session_libraries:
                lib_config = get_library_config(lib_name)
                priority = lib_config.load_priority if lib_config else 999
                prioritized_libraries.append((priority, lib_name))
            
            sorted_session_libs = [lib for priority, lib in sorted(prioritized_libraries, key=lambda x: x[0])]
            logger.debug(f"Priority-ordered session libraries: {sorted_session_libs}")
            
            # Filter cache to only include session libraries
            session_filtered_cache = {}
            for cache_key, keyword_info in self.keyword_cache.items():
                if keyword_info.library in session_libraries:
                    session_filtered_cache[cache_key] = keyword_info
            
            search_cache = session_filtered_cache
            logger.debug(f"Session filtering: {len(search_cache)} keywords from session libraries")
            
            # Try exact match in session libraries with priority order
            for lib_name in sorted_session_libs:
                for cache_key, keyword_info in search_cache.items():
                    if (keyword_info.library == lib_name and 
                        keyword_info.name.lower() == normalized):
                        logger.debug(f"Session-aware exact match: {keyword_name} from {lib_name} (priority {get_library_config(lib_name).load_priority if get_library_config(lib_name) else 999})")
                        return keyword_info
        
        # Priority 2: Filter by active library (if provided and no session libraries)
        elif active_library:
            # Filter keywords to only include those from the active library or built-in libraries
            builtin_libraries = ['BuiltIn', 'Collections', 'String', 'DateTime', 'OperatingSystem', 'Process']
            filtered_cache = {}
            
            for cache_key, keyword_info in self.keyword_cache.items():
                if (keyword_info.library == active_library or keyword_info.library in builtin_libraries):
                    filtered_cache[cache_key] = keyword_info
            
            search_cache = filtered_cache
            logger.debug(f"Active library filtering to '{active_library}' - {len(search_cache)} keywords available")
        
        # Try exact match first (simple key for backward compatibility)
        if normalized in search_cache:
            keyword_info = search_cache[normalized]
            # When active_library is specified, ensure the found keyword is from the correct library or built-in
            if active_library:
                if (keyword_info.library == active_library or 
                    keyword_info.library in ['BuiltIn', 'Collections', 'String', 'DateTime', 'OperatingSystem', 'Process']):
                    logger.debug(f"Found exact match: {keyword_name} from {keyword_info.library} (filtered for {active_library})")
                    return keyword_info
                else:
                    logger.debug(f"Rejecting {keyword_name} from {keyword_info.library} due to active_library filter ({active_library})")
                    # Continue searching - don't return this result
            else:
                return keyword_info
        
        # Try common variations
        variations = [
            normalized.replace(' ', ''),  # Remove spaces
            normalized.replace('_', ' '),  # Replace underscores
            normalized.replace('-', ' '),  # Replace hyphens
        ]
        
        for variation in variations:
            if variation in search_cache:
                return search_cache[variation]
        
        # Try fuzzy matching - find best partial match
        best_match = None
        best_score = 0
        
        for cached_name, keyword_info in search_cache.items():
            # Score based on how much of the search term matches
            if normalized in cached_name:
                score = len(normalized) / len(cached_name)
                if score > best_score:
                    best_score = score
                    best_match = keyword_info
            elif cached_name in normalized:
                score = len(cached_name) / len(normalized)
                if score > best_score:
                    best_score = score
                    best_match = keyword_info
        
        # Only return matches with reasonable confidence
        if best_score >= 0.6:
            library_info = f" from {best_match.library}" if active_library else ""
            logger.debug(f"Fuzzy matched '{keyword_name}' to '{best_match.name}'{library_info} (score: {best_score:.2f})")
            return best_match
        
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
        results: Dict[str, KeywordInfo] = {}
        for info in self.keyword_cache.values():
            if info.library == library_name:
                results[info.name.lower()] = info
        return list(results.values())
    
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

    def get_keyword_shadow_info(self) -> Dict[str, Any]:
        """Get information about shadowed keywords.

        Returns:
            Dictionary with shadow information
        """
        return {
            "ambiguous_count": len(self.ambiguous_keywords),
            "ambiguous_keywords": list(self.ambiguous_keywords),
            "shadows": {
                kw: [(lib, info.source_method if hasattr(info, 'source_method') else info.method_name) for lib, info in libs]
                for kw, libs in self.shadowed_keywords.items()
            }
        }

    def is_keyword_ambiguous(self, keyword_name: str) -> bool:
        """Check if a keyword name is ambiguous (exists in multiple libraries)."""
        return keyword_name.lower() in self.ambiguous_keywords

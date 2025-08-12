"""Native Robot Framework libdoc integration for keyword discovery."""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

try:
    from robot.libdoc import LibraryDocumentation
    from robot.libdocpkg.model import LibraryDoc, KeywordDoc
    HAS_LIBDOC = True
except ImportError:
    HAS_LIBDOC = False
    LibraryDocumentation = None
    LibraryDoc = None
    KeywordDoc = None

logger = logging.getLogger(__name__)

@dataclass
class RFKeywordInfo:
    """Information about a Robot Framework keyword using native RF libdoc."""
    name: str
    library: str
    doc: str = ""
    short_doc: str = ""
    args: List[str] = field(default_factory=list)
    arg_types: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    is_deprecated: bool = False
    source: str = ""
    lineno: int = 0

@dataclass 
class RFLibraryInfo:
    """Information about a Robot Framework library using native RF libdoc."""
    name: str
    doc: str = ""
    version: str = ""
    type: str = ""
    scope: str = ""
    source: str = ""
    keywords: Dict[str, RFKeywordInfo] = field(default_factory=dict)

class RobotFrameworkDocStorage:
    """Storage and retrieval of Robot Framework library documentation using native libdoc."""
    
    def __init__(self):
        self.libraries: Dict[str, RFLibraryInfo] = {}
        self.keyword_cache: Dict[str, RFKeywordInfo] = {}
        self.failed_imports: Dict[str, str] = {}
        
        # Load library list from centralized registry
        from robotmcp.config.library_registry import get_library_names_for_loading
        self.common_libraries = get_library_names_for_loading()
        
        if not HAS_LIBDOC:
            logger.warning("Robot Framework libdoc not available. Falling back to inspection-based discovery.")
            return
            
        self._initialize_libraries()
    
    def _initialize_libraries(self):
        """Initialize library documentation using native Robot Framework libdoc."""
        if not HAS_LIBDOC:
            return
            
        logger.info("Initializing Robot Framework libraries using native libdoc...")
        
        for library_name in self.common_libraries:
            self._load_library_documentation(library_name)
        
        logger.info(f"Initialized {len(self.libraries)} libraries with {len(self.keyword_cache)} keywords using libdoc")
    
    def _load_library_documentation(self, library_name: str) -> bool:
        """Load library documentation using Robot Framework's LibraryDocumentation."""
        try:
            # Use Robot Framework's native libdoc to get library documentation
            lib_doc = LibraryDocumentation(library_name)
            
            # Create our library info from the libdoc data
            source = lib_doc.source or ""
            if source and hasattr(source, '__fspath__'):  # Path-like object
                source = str(source)
            
            lib_info = RFLibraryInfo(
                name=library_name,
                doc=lib_doc.doc,
                version=lib_doc.version,
                type=lib_doc.type,
                scope=lib_doc.scope,
                source=source
            )
            
            # Extract keywords using native libdoc KeywordDoc objects
            for kw_doc in lib_doc.keywords:
                keyword_info = self._extract_keyword_from_libdoc(library_name, kw_doc)
                lib_info.keywords[keyword_info.name] = keyword_info
                
                # Add to cache with normalized name
                cache_key = keyword_info.name.lower().strip()
                self.keyword_cache[cache_key] = keyword_info
            
            self.libraries[library_name] = lib_info
            
            logger.info(f"Successfully loaded library '{library_name}' with {len(lib_info.keywords)} keywords using libdoc")
            return True
            
        except Exception as e:
            logger.debug(f"Failed to load library documentation for '{library_name}': {e}")
            self.failed_imports[library_name] = str(e)
            return False
    
    def _extract_keyword_from_libdoc(self, library_name: str, kw_doc: 'KeywordDoc') -> RFKeywordInfo:
        """Extract keyword information from Robot Framework's KeywordDoc object."""
        # Get arguments as strings
        args = []
        arg_types = []
        
        if hasattr(kw_doc, 'args') and kw_doc.args:
            args = [str(arg) for arg in kw_doc.args]
            
        if hasattr(kw_doc, 'arg_types') and kw_doc.arg_types:
            arg_types = [str(arg_type) for arg_type in kw_doc.arg_types]
        
        # Get tags
        tags = []
        if hasattr(kw_doc, 'tags') and kw_doc.tags:
            tags = list(kw_doc.tags)
        
        # Get deprecation status
        is_deprecated = getattr(kw_doc, 'deprecated', False)
        
        # Get source information (convert Path objects to strings)
        source = getattr(kw_doc, 'source', "")
        if source and hasattr(source, '__fspath__'):  # Path-like object
            source = str(source)
        lineno = getattr(kw_doc, 'lineno', 0)
        
        # Use native short_doc from Robot Framework
        short_doc = getattr(kw_doc, 'short_doc', "")
        if not short_doc:
            # Fallback to creating short doc from full doc
            short_doc = self._create_short_doc(kw_doc.doc)
        
        return RFKeywordInfo(
            name=kw_doc.name,
            library=library_name,
            doc=kw_doc.doc,
            short_doc=short_doc,
            args=args,
            arg_types=arg_types,
            tags=tags,
            is_deprecated=is_deprecated,
            source=source,
            lineno=lineno
        )
    
    def _create_short_doc(self, doc: str, max_length: int = 120) -> str:
        """Create a short documentation string from full documentation.
        
        This is a fallback for when short_doc is not available from libdoc.
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
            if line and not line.startswith(('Tags:', 'Arguments:', '.. note::', '.. warning::')):
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
    
    def find_keyword(self, keyword_name: str) -> Optional[RFKeywordInfo]:
        """Find a keyword by name (case-insensitive)."""
        if not HAS_LIBDOC:
            return None
            
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
        
        return None
    
    def get_keywords_by_library(self, library_name: str) -> List[RFKeywordInfo]:
        """Get all keywords from a specific library."""
        if not HAS_LIBDOC or library_name not in self.libraries:
            return []
        
        return list(self.libraries[library_name].keywords.values())
    
    def get_all_keywords(self) -> List[RFKeywordInfo]:
        """Get all available keywords."""
        if not HAS_LIBDOC:
            return []
            
        return list(self.keyword_cache.values())
    
    def search_keywords(self, pattern: str) -> List[RFKeywordInfo]:
        """Search for keywords matching a pattern."""
        if not HAS_LIBDOC:
            return []
            
        pattern = pattern.lower()
        matches = []
        
        for keyword_info in self.keyword_cache.values():
            if (pattern in keyword_info.name.lower() or 
                pattern in keyword_info.doc.lower() or
                pattern in keyword_info.short_doc.lower() or
                any(pattern in tag.lower() for tag in keyword_info.tags)):
                matches.append(keyword_info)
        
        return matches
    
    def get_library_documentation(self, library_name: str) -> Optional[RFLibraryInfo]:
        """Get full documentation for a library."""
        if not HAS_LIBDOC:
            return None
            
        return self.libraries.get(library_name)
    
    def get_keyword_documentation(self, keyword_name: str, library_name: str = None) -> Optional[RFKeywordInfo]:
        """Get full documentation for a specific keyword."""
        if not HAS_LIBDOC:
            return None
            
        keyword_info = self.find_keyword(keyword_name)
        
        if keyword_info and library_name:
            # If library specified, ensure it matches
            if keyword_info.library.lower() == library_name.lower():
                return keyword_info
            else:
                return None
        
        return keyword_info
    
    def refresh_library(self, library_name: str) -> bool:
        """Refresh documentation for a specific library."""
        if not HAS_LIBDOC:
            return False
            
        # Remove old data
        if library_name in self.libraries:
            old_keywords = self.libraries[library_name].keywords
            for kw_name in old_keywords:
                cache_key = kw_name.lower().strip()
                if cache_key in self.keyword_cache:
                    del self.keyword_cache[cache_key]
            del self.libraries[library_name]
        
        # Reload
        return self._load_library_documentation(library_name)
    
    def get_library_status(self) -> Dict[str, Any]:
        """Get status of all libraries."""
        if not HAS_LIBDOC:
            return {
                "libdoc_available": False,
                "loaded_libraries": {},
                "failed_imports": self.failed_imports,
                "total_keywords": 0
            }
        
        return {
            "libdoc_available": True,
            "loaded_libraries": {
                name: {
                    "keywords": len(lib.keywords),
                    "doc": lib.doc,
                    "version": lib.version,
                    "type": lib.type,
                    "scope": lib.scope
                }
                for name, lib in self.libraries.items()
            },
            "failed_imports": self.failed_imports,
            "total_keywords": len(self.keyword_cache)
        }
    
    def is_available(self) -> bool:
        """Check if libdoc functionality is available."""
        return HAS_LIBDOC

# Global instance
_rf_doc_storage = None

def get_rf_doc_storage() -> RobotFrameworkDocStorage:
    """Get the global Robot Framework documentation storage instance."""
    global _rf_doc_storage
    if _rf_doc_storage is None:
        _rf_doc_storage = RobotFrameworkDocStorage()
    return _rf_doc_storage
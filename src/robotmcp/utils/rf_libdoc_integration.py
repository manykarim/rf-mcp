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
        # Map normalized keyword name -> { library_name -> RFKeywordInfo }
        self.keyword_index_by_name: Dict[str, Dict[str, RFKeywordInfo]] = {}
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
        
        # Count total keywords across all libraries
        total_keywords = sum(len(lib.keywords) for lib in self.libraries.values())
        logger.info(f"Initialized {len(self.libraries)} libraries with {total_keywords} keywords using libdoc")
    
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
                
                # Index by normalized name and library to avoid collisions
                norm = self._normalize_name(keyword_info.name)
                if norm not in self.keyword_index_by_name:
                    self.keyword_index_by_name[norm] = {}
                self.keyword_index_by_name[norm][library_name] = keyword_info
            
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
            
            # Check if signature is decorator-masked (e.g., *args, **kwargs)
            is_decorator_masked = (
                len(args) <= 2 and 
                any('*args' in arg or '**kwargs' in arg for arg in args)
            )
            
            # If decorator-masked, try hybrid signature extraction
            if is_decorator_masked:
                try:
                    hybrid_args = self._extract_hybrid_signature(kw_doc.name, library_name)
                    if hybrid_args and len(hybrid_args) > 2:
                        args = hybrid_args
                        logger.debug(f"Used hybrid extraction for {kw_doc.name}: {args}")
                except Exception as e:
                    logger.debug(f"Hybrid extraction failed for {kw_doc.name}: {e}")
            
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
        norm = self._normalize_name(keyword_name)
        by_lib = self.keyword_index_by_name.get(norm)
        if not by_lib:
            return None
        # Return deterministically the first library’s entry
        for lib in sorted(by_lib.keys()):
            return by_lib[lib]
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
        # Flatten index across libraries
        result: List[RFKeywordInfo] = []
        for by_lib in self.keyword_index_by_name.values():
            result.extend(by_lib.values())
        return result
    
    def get_keywords_from_libraries(self, library_names: List[str]) -> List[RFKeywordInfo]:
        """Get keywords from specific libraries in the order provided.
        
        Args:
            library_names: List of library names to get keywords from
            
        Returns:
            List of RFKeywordInfo objects from the specified libraries
        """
        if not HAS_LIBDOC:
            return []
            
        keywords = []
        for library_name in library_names:
            if library_name in self.libraries:
                keywords.extend(self.libraries[library_name].keywords.values())
        
        return keywords
    
    def search_keywords(self, pattern: str) -> List[RFKeywordInfo]:
        """Search for keywords matching a pattern."""
        if not HAS_LIBDOC:
            return []
            
        pattern = pattern.lower()
        matches = []
        
        for keyword_info in self.get_all_keywords():
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
        """Get full documentation for a specific keyword.

        If library_name is provided, performs a strict lookup within that library only.
        If not provided, returns the first match (for backward compatibility). Use
        get_keywords_documentation_all() to fetch all matches across libraries.
        """
        if not HAS_LIBDOC:
            return None

        norm = self._normalize_name(keyword_name)
        if library_name:
            by_lib = self.keyword_index_by_name.get(norm, {})
            # Strict per-library search
            return by_lib.get(library_name)
        # Backward-compat: return first available match across libraries
        by_lib = self.keyword_index_by_name.get(norm)
        if not by_lib:
            return None
        # Return the entry from the first library deterministically (sorted keys)
        for lib in sorted(by_lib.keys()):
            return by_lib[lib]
        return None

    def get_keywords_documentation_all(self, keyword_name: str) -> List[RFKeywordInfo]:
        """Return all matches for a keyword across libraries (exact/normalized)."""
        if not HAS_LIBDOC:
            return []
        norm = self._normalize_name(keyword_name)
        by_lib = self.keyword_index_by_name.get(norm, {})
        return [by_lib[k] for k in sorted(by_lib.keys())]

    def _normalize_name(self, name: str) -> str:
        return name.lower().replace('_', ' ').strip()
    
    def _extract_hybrid_signature(self, keyword_name: str, library_name: str) -> Optional[List[str]]:
        """
        Extract keyword signature using hybrid RF native + closure inspection approach.
        
        For decorated keywords showing *args, **kwargs, tries to extract the original
        signature using Robot Framework's TestLibrary and closure inspection.
        """
        try:
            from robot.running.testlibraries import TestLibrary
            
            # Load library using Robot Framework's native TestLibrary
            lib = TestLibrary.from_name(library_name)
            
            # Find the keyword
            keyword_obj = None
            for kw in lib.keywords:
                if kw.name == keyword_name:
                    keyword_obj = kw
                    break
            
            if not keyword_obj or not hasattr(keyword_obj, 'args'):
                return None
            
            args_spec = keyword_obj.args
            
            # Check if this is a decorated keyword (generic *args, **kwargs)
            if self._is_decorated_keyword(args_spec):
                # Use closure inspection for decorated keywords
                return self._extract_from_closure(keyword_obj, keyword_name, library_name)
            else:
                # Use native RF ArgumentSpec for non-decorated keywords
                return self._extract_from_argumentspec(args_spec)
        
        except Exception as e:
            logger.debug(f"Hybrid signature extraction failed for {keyword_name}: {e}")
            return None
    
    def _is_decorated_keyword(self, args_spec) -> bool:
        """
        Detect if keyword is decorated by checking for generic signature pattern.
        
        Decorated keywords show:
        - var_positional = 'args' 
        - var_named = 'kwargs'
        - positional_or_named = () (empty)
        """
        return (
            args_spec.var_positional == 'args' and
            args_spec.var_named == 'kwargs' and 
            len(args_spec.positional_or_named) == 0
        )
    
    def _extract_from_argumentspec(self, args_spec) -> List[str]:
        """Extract signature from RF ArgumentSpec (for non-decorated keywords)."""
        signature = []
        
        # Required positional args (no defaults)
        required_positional = []
        for name in args_spec.positional:
            if name not in args_spec.defaults:
                required_positional.append(name)
        
        # Args with defaults (can be positional or named)
        optional_args = []
        for name in args_spec.positional_or_named:
            if name in args_spec.defaults:
                default = args_spec.defaults[name]
                optional_args.append(f"{name}={default}")
            elif name not in required_positional:  # Not already added as required
                optional_args.append(name)
        
        # Combine in order: required positional, optional, *args, **kwargs
        signature.extend(required_positional)
        signature.extend(optional_args)
        
        # Add *args if present
        if args_spec.var_positional:
            signature.append(f"*{args_spec.var_positional}")
        
        # Add **kwargs if present
        if args_spec.var_named:
            signature.append(f"**{args_spec.var_named}")
        
        return signature
    
    def _extract_from_closure(self, keyword_obj, keyword_name: str, library_name: str) -> Optional[List[str]]:
        """Extract signature from method closure (for decorated keywords)."""
        try:
            # Check different possible attributes for the handler function
            handler = None
            if hasattr(keyword_obj, 'method'):
                handler = keyword_obj.method
            elif hasattr(keyword_obj, '_handler'):
                handler = keyword_obj._handler
            elif hasattr(keyword_obj, 'handler'):
                handler = keyword_obj.handler
            
            if not handler:
                return None
            
            # For RequestsLibrary, try to find the original function in closure
            if library_name == "RequestsLibrary" and hasattr(handler, '__closure__') and handler.__closure__:
                
                # Map keyword names to method names
                keyword_to_method = {
                    "POST On Session": "post_on_session",
                    "POST": "session_less_post",
                    "GET On Session": "get_on_session", 
                    "GET": "session_less_get",
                    "PUT On Session": "put_on_session",
                    "PUT": "session_less_put",
                    "PATCH On Session": "patch_on_session",
                    "PATCH": "session_less_patch",
                    "DELETE On Session": "delete_on_session",
                    "DELETE": "session_less_delete",
                }
                
                method_name = keyword_to_method.get(keyword_name)
                if method_name:
                    
                    for cell in handler.__closure__:
                        try:
                            content = cell.cell_contents
                            if (callable(content) and 
                                hasattr(content, '__name__') and 
                                content.__name__ == method_name):
                                
                                import inspect
                                original_sig = inspect.signature(content)
                                
                                # Convert to signature args format
                                signature_args = []
                                for name, param in original_sig.parameters.items():
                                    if name != 'self':
                                        if param.kind == inspect.Parameter.VAR_KEYWORD:
                                            signature_args.append(f"**{name}")
                                        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                                            signature_args.append(f"*{name}")
                                        elif param.default is param.empty:
                                            signature_args.append(name)
                                        else:
                                            signature_args.append(f"{name}={param.default}")
                                
                                return signature_args
                                
                        except Exception:
                            continue
        
        except Exception as e:
            logger.debug(f"Closure inspection failed for {keyword_name}: {e}")
        
        return None
    
    def refresh_library(self, library_name: str) -> bool:
        """Refresh documentation for a specific library."""
        if not HAS_LIBDOC:
            return False
            
        # Remove old data
        if library_name in self.libraries:
            old_keywords = self.libraries[library_name].keywords
            # Remove entries from index for this library
            for kw_name in old_keywords:
                norm = self._normalize_name(kw_name)
                by_lib = self.keyword_index_by_name.get(norm)
                if by_lib and library_name in by_lib:
                    del by_lib[library_name]
                    if not by_lib:
                        del self.keyword_index_by_name[norm]
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
            "total_keywords": sum(len(lib.keywords) for lib in self.libraries.values())
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

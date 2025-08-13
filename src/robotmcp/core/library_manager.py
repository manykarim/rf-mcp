"""Library loading and management functionality."""

import importlib
import logging
from typing import Any, Dict, Set

from robotmcp.models.library_models import LibraryInfo, KeywordInfo

logger = logging.getLogger(__name__)


class LibraryManager:
    """Manages Robot Framework library loading, exclusion, and conflict resolution."""
    
    def __init__(self):
        self.libraries: Dict[str, LibraryInfo] = {}
        self.failed_imports: Dict[str, str] = {}
        
        # Library exclusion rules - only one from each group can be loaded
        self.exclusion_groups = {
            'web_automation': ['Browser', 'SeleniumLibrary'],  # Browser OR Selenium, not both
        }
        self.excluded_libraries: Set[str] = set()
        
        # Load library list from centralized registry
        from robotmcp.config.library_registry import get_library_names_for_loading
        self.common_libraries = get_library_names_for_loading()
    
    def load_all_libraries(self, keyword_extractor) -> None:
        """
        Load all common libraries (legacy method for backward compatibility).
        
        Args:
            keyword_extractor: KeywordDiscovery instance for extracting library info
        """
        # Always try to initialize BuiltIn library first
        self.try_import_library('BuiltIn', keyword_extractor)
        
        # Try to import other common libraries first
        for library_name in self.common_libraries:
            if library_name != 'BuiltIn':
                self.try_import_library(library_name, keyword_extractor)
        
        # Apply exclusion logic after all libraries have been attempted
        self.resolve_library_conflicts()
        
        logger.info(f"Initialized {len(self.libraries)} libraries")
    
    def load_session_libraries(self, library_names: list, keyword_extractor) -> None:
        """
        Load specific libraries for a session.
        
        Args:
            library_names: List of library names to load
            keyword_extractor: KeywordDiscovery instance for extracting library info
        """
        # Always ensure BuiltIn is loaded first
        if 'BuiltIn' not in self.libraries:
            self.try_import_library('BuiltIn', keyword_extractor)
        
        # Load requested libraries
        loaded_count = 0
        for library_name in library_names:
            if library_name not in self.libraries and library_name != 'BuiltIn':
                if self.try_import_library(library_name, keyword_extractor):
                    loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} new libraries for session: {library_names}")
    
    def load_library_on_demand(self, library_name: str, keyword_extractor) -> bool:
        """
        Load a single library on demand.
        
        Args:
            library_name: Name of library to load
            keyword_extractor: KeywordDiscovery instance for extracting library info
            
        Returns:
            True if library was loaded successfully, False otherwise
        """
        if library_name in self.libraries:
            return True  # Already loaded
        
        success = self.try_import_library(library_name, keyword_extractor)
        if success:
            # Add keywords to cache after successful loading
            lib_info = self.libraries[library_name]
            keyword_extractor.add_keywords_to_cache(lib_info)
            logger.info(f"Loaded library on demand: {library_name} with {len(lib_info.keywords)} keywords")
        
        return success
    
    def try_import_library(self, library_name: str, keyword_extractor) -> bool:
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
                import robot.libraries.DateTime as DateTime
                instance = DateTime
            elif library_name == 'OperatingSystem':
                from robot.libraries.OperatingSystem import OperatingSystem
                instance = OperatingSystem()
            elif library_name == 'Process':
                from robot.libraries.Process import Process
                instance = Process()
            elif library_name == 'XML':
                from robot.libraries.XML import XML
                instance = XML()
            elif library_name == 'Telnet':
                from robot.libraries.Telnet import Telnet
                instance = Telnet()
            elif library_name == 'Screenshot':
                from robot.libraries.Screenshot import Screenshot
                instance = Screenshot()
            elif library_name == 'Dialogs':
                import robot.libraries.Dialogs as Dialogs
                instance = Dialogs
            else:
                # Try generic import
                module = importlib.import_module(library_name)
                if hasattr(module, library_name):
                    instance = getattr(module, library_name)()
                else:
                    instance = module
            
            # Extract keywords from the library instance
            lib_info = keyword_extractor.extract_library_info(library_name, instance)
            self.libraries[library_name] = lib_info
            
            logger.info(f"Successfully loaded library '{library_name}' with {len(lib_info.keywords)} keywords")
            return True
            
        except Exception as e:
            logger.debug(f"Failed to import library '{library_name}': {e}")
            self.failed_imports[library_name] = str(e)
            return False
    
    def resolve_library_conflicts(self) -> None:
        """
        Resolve conflicts between loaded libraries by removing less preferred ones.
        
        UPDATED: With session-based loading, conflicts are minimized since only
        relevant libraries are loaded per session.
        """
        web_automation_libs = self.exclusion_groups.get('web_automation', [])
        
        # Check which web automation libraries were actually loaded
        loaded_web_libs = [lib for lib in web_automation_libs if lib in self.libraries]
        
        if len(loaded_web_libs) > 1:
            logger.info(f"Multiple web automation libraries loaded: {loaded_web_libs}")
            logger.info("Session-based search order will resolve keyword conflicts")
            
        elif len(loaded_web_libs) == 1:
            logger.info(f"Single web automation library loaded: {loaded_web_libs[0]}")
        else:
            logger.debug("No web automation libraries loaded")
    
    def remove_library(self, library_name: str) -> None:
        """
        Remove a library from the loaded libraries.
        
        Args:
            library_name: Name of library to remove
        """
        if library_name not in self.libraries:
            return
        
        # Get library info
        lib_info = self.libraries[library_name]
        
        # Remove library
        del self.libraries[library_name]
        
        logger.info(f"Removed library '{library_name}' with {len(lib_info.keywords)} keywords")
    
    def is_library_importable(self, library_name: str) -> bool:
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
                "browser_available": self.is_library_importable('Browser'),
                "selenium_available": self.is_library_importable('SeleniumLibrary'),
                "active_web_library": next((lib for lib in ['Browser', 'SeleniumLibrary'] if lib in self.libraries), None)
            }
        }
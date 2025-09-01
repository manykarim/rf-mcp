"""Library loading and management functionality."""

import importlib
import logging
from typing import Any, Dict, Set

from robotmcp.models.library_models import LibraryInfo

logger = logging.getLogger(__name__)


class LibraryManager:
    """Manages Robot Framework library loading, exclusion, and conflict resolution."""

    def __init__(self):
        self.libraries: Dict[str, LibraryInfo] = {}
        self.failed_imports: Dict[str, str] = {}

        # Library exclusion rules - only one from each group can be loaded
        self.exclusion_groups = {
            "web_automation": [
                "Browser",
                "SeleniumLibrary",
            ],  # Browser OR Selenium, not both
            "mobile_automation": ["AppiumLibrary"],  # Mobile testing library
            # Note: AppiumLibrary can conflict with SeleniumLibrary for some keywords
            # but they're in separate groups to allow advanced users to use both if needed
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
        self.try_import_library("BuiltIn", keyword_extractor)

        # Try to import other common libraries first
        for library_name in self.common_libraries:
            if library_name != "BuiltIn":
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
        if "BuiltIn" not in self.libraries:
            if self.try_import_library("BuiltIn", keyword_extractor):
                # Add BuiltIn keywords to cache
                lib_info = self.libraries["BuiltIn"]
                keyword_extractor.add_keywords_to_cache(lib_info)
                logger.debug(
                    f"Added {len(lib_info.keywords)} BuiltIn keywords to cache"
                )

        # Load requested libraries
        loaded_count = 0
        for library_name in library_names:
            if library_name not in self.libraries and library_name != "BuiltIn":
                if self.try_import_library(library_name, keyword_extractor):
                    loaded_count += 1
                    # Add keywords to cache after successful loading
                    lib_info = self.libraries[library_name]
                    keyword_extractor.add_keywords_to_cache(lib_info)
                    logger.debug(
                        f"Added {len(lib_info.keywords)} keywords from {library_name} to cache"
                    )

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
            logger.info(
                f"Loaded library on demand: {library_name} with {len(lib_info.keywords)} keywords"
            )

        return success

    def try_import_library(self, library_name: str, keyword_extractor) -> bool:
        """Try to import and initialize a Robot Framework library."""
        try:
            # Handle special cases
            if library_name == "BuiltIn":
                from robot.libraries.BuiltIn import BuiltIn

                instance = BuiltIn()
            elif library_name == "Browser":
                try:
                    from Browser import Browser

                    instance = Browser()
                except ImportError:
                    logger.debug("Browser library not available")
                    self.failed_imports[library_name] = "Not installed"
                    return False
            elif library_name == "SeleniumLibrary":
                try:
                    from SeleniumLibrary import SeleniumLibrary

                    # Create instance in a way that avoids browser dependency during keyword discovery
                    instance = SeleniumLibrary()

                    # SeleniumLibrary sometimes throws errors during initialization
                    # when no browser is open. We'll catch this and still proceed
                    # with keyword extraction since the class methods exist.
                except ImportError:
                    logger.debug("SeleniumLibrary not available")
                    self.failed_imports[library_name] = "Not installed"
                    return False
                except Exception as e:
                    logger.debug(
                        f"SeleniumLibrary initialization error (continuing anyway): {e}"
                    )
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
                        self.failed_imports[library_name] = (
                            f"Initialization failed: {e}"
                        )
                        return False
            elif library_name == "RequestsLibrary":
                try:
                    from RequestsLibrary import RequestsLibrary

                    instance = RequestsLibrary()
                except ImportError:
                    logger.debug("RequestsLibrary not available")
                    self.failed_imports[library_name] = "Not installed"
                    return False
            elif library_name == "Collections":
                from robot.libraries.Collections import Collections

                instance = Collections()
            elif library_name == "String":
                from robot.libraries.String import String

                instance = String()
            elif library_name == "DateTime":
                import robot.libraries.DateTime as DateTime

                instance = DateTime
            elif library_name == "OperatingSystem":
                from robot.libraries.OperatingSystem import OperatingSystem

                instance = OperatingSystem()
            elif library_name == "Process":
                from robot.libraries.Process import Process

                instance = Process()
            elif library_name == "XML":
                from robot.libraries.XML import XML

                instance = XML()
            elif library_name == "Telnet":
                from robot.libraries.Telnet import Telnet

                instance = Telnet()
            elif library_name == "Screenshot":
                from robot.libraries.Screenshot import Screenshot

                instance = Screenshot()
            elif library_name == "Dialogs":
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

            logger.info(
                f"Successfully loaded library '{library_name}' with {len(lib_info.keywords)} keywords"
            )
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
        web_automation_libs = self.exclusion_groups.get("web_automation", [])

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

        logger.info(
            f"Removed library '{library_name}' with {len(lib_info.keywords)} keywords"
        )

    def is_library_importable(self, library_name: str) -> bool:
        """Check if a library can be imported without actually importing it."""
        try:
            if library_name == "Browser":
                import Browser

                return True
            elif library_name == "SeleniumLibrary":
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

    def ensure_library_in_rf_context(self, library_name: str) -> bool:
        """
        Ensure library is properly registered in Robot Framework execution context.

        This is Phase 1 of the RequestsLibrary fix: Library Registration Fix.
        The issue was that RequestsLibrary was loaded at Python module level but
        not registered in Robot Framework's execution context.

        Args:
            library_name: Name of library to register in RF context

        Returns:
            True if library is registered successfully, False otherwise
        """
        try:
            from robot.running.context import EXECUTION_CONTEXTS

            # Check if we have an active execution context
            if not EXECUTION_CONTEXTS.current:
                logger.warning(f"No active RF execution context for {library_name}")
                return False

            current_context = EXECUTION_CONTEXTS.current

            # Check if library is already registered in RF context
            try:
                # Try to get the library instance from RF context
                lib_instance = current_context.namespace.get_library_instance(
                    library_name
                )
                if lib_instance:
                    logger.debug(f"{library_name} already registered in RF context")
                    return True
            except Exception:
                # Library not found in RF context, need to register it
                pass

            # Get the library instance from our manager
            if library_name not in self.libraries:
                logger.warning(f"{library_name} not loaded in library manager")
                return False

            lib_info = self.libraries[library_name]
            lib_instance = lib_info.instance

            if not lib_instance:
                logger.warning(f"{library_name} has no instance in library manager")
                return False

            # Register the library in RF execution context
            # This is the critical fix - ensure the library is available to RF keyword resolution
            # Use Robot Framework's native library import mechanism
            current_context.namespace.import_library(
                library_name, args=[], alias=None, notify=True
            )

            logger.info(
                f"Successfully registered {library_name} in RF execution context"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to register {library_name} in RF context: {e}")
            import traceback

            logger.debug(f"RF context registration traceback: {traceback.format_exc()}")
            return False

    def get_library_instance(self, library_name: str) -> Any:
        """Get the actual library instance for a given library name."""
        if library_name not in self.libraries:
            return None
        return self.libraries[library_name].instance

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
                "browser_available": self.is_library_importable("Browser"),
                "selenium_available": self.is_library_importable("SeleniumLibrary"),
                "active_web_library": next(
                    (
                        lib
                        for lib in ["Browser", "SeleniumLibrary"]
                        if lib in self.libraries
                    ),
                    None,
                ),
            },
        }

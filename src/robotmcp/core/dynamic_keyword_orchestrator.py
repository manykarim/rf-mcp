"""Main orchestrator for dynamic keyword discovery."""

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from robotmcp.core.keyword_discovery import KeywordDiscovery
from robotmcp.core.library_manager import LibraryManager
from robotmcp.models.library_models import KeywordInfo, ParsedArguments
from robotmcp.utils.argument_processor import ArgumentProcessor

if TYPE_CHECKING:
    from robotmcp.models.session_models import ExecutionSession

logger = logging.getLogger(__name__)

# Define builtin libraries constant for keyword classification
BUILTIN_LIBRARIES = {
    "BuiltIn",
    "Collections",
    "DateTime",
    "Dialogs",
    "OperatingSystem",
    "Process",
    "Screenshot",
    "String",
    "Telnet",
    "XML",
}


class DynamicKeywordDiscovery:
    """Main orchestrator for dynamic Robot Framework keyword discovery and management."""

    def __init__(self):
        self.library_manager = LibraryManager()
        self.keyword_discovery = KeywordDiscovery()
        self.argument_processor = ArgumentProcessor()

        # Initialize session manager (use the execution session manager instead)
        self.session_manager = None  # Will be set by ExecutionCoordinator when needed

        # Initialize with minimal libraries
        self._initialize_minimal()

    def set_session_manager(self, session_manager):
        """Set the session manager from the execution coordinator."""
        self.session_manager = session_manager

    def _initialize_minimal(self) -> None:
        """Initialize with minimal core libraries only."""
        # Load minimal core libraries
        core_libraries = ["BuiltIn", "Collections", "String"]
        self.library_manager.load_session_libraries(
            core_libraries, self.keyword_discovery
        )

        # Add keywords to cache
        for lib_info in self.library_manager.libraries.values():
            self.keyword_discovery.add_keywords_to_cache(lib_info)

        logger.info(
            f"Initialized with minimal libraries: {len(self.library_manager.libraries)} libraries with {len(self.keyword_discovery.keyword_cache)} keywords"
        )

    def _initialize_legacy(self) -> None:
        """Legacy initialization method - loads all libraries."""
        # Load all libraries through the library manager
        self.library_manager.load_all_libraries(self.keyword_discovery)

        # Add all keywords to cache
        for lib_info in self.library_manager.libraries.values():
            self.keyword_discovery.add_keywords_to_cache(lib_info)

        logger.info(
            f"Initialized {len(self.library_manager.libraries)} libraries with {len(self.keyword_discovery.keyword_cache)} keywords"
        )

    # Public API methods
    def find_keyword(
        self,
        keyword_name: str,
        active_library: str = None,
        session_libraries: List[str] = None,
    ) -> Optional[KeywordInfo]:
        """Find a keyword by name with fuzzy matching, optionally filtering by active library and session libraries."""
        # Try LibDoc-based storage first if available (more accurate)
        try:
            from robotmcp.utils.rf_libdoc_integration import get_rf_doc_storage

            rf_doc_storage = get_rf_doc_storage()

            if rf_doc_storage.is_available():
                libdoc_result = self._find_keyword_libdoc(
                    keyword_name, active_library, rf_doc_storage, session_libraries
                )
                if libdoc_result:
                    return libdoc_result
        except Exception as e:
            logger.debug(
                f"LibDoc keyword search failed, falling back to inspection: {e}"
            )

        # Fall back to inspection-based discovery with session libraries
        return self.keyword_discovery.find_keyword(
            keyword_name, active_library, session_libraries
        )

    def _find_keyword_libdoc(
        self,
        keyword_name: str,
        active_library: str,
        rf_doc_storage,
        session_libraries: List[str] = None,
    ) -> Optional[KeywordInfo]:
        """Find keyword using LibDoc storage with session library filtering and priority ordering."""
        if not keyword_name:
            return None

        # PHASE 1 IMPLEMENTATION: Session-aware keyword resolution
        # Priority 1: Search in session libraries first (if provided)
        if session_libraries:
            logger.debug(
                f"LibDoc: Session-aware search for '{keyword_name}' in libraries: {session_libraries}"
            )

            # Get library priorities for proper ordering
            from robotmcp.config.library_registry import get_library_config

            # Sort session libraries by priority (lower number = higher priority)
            prioritized_libraries = []
            for lib_name in session_libraries:
                lib_config = get_library_config(lib_name)
                priority = lib_config.load_priority if lib_config else 999
                prioritized_libraries.append((priority, lib_name))

            # Sort by priority and extract library names
            sorted_session_libs = [
                lib
                for priority, lib in sorted(prioritized_libraries, key=lambda x: x[0])
            ]
            logger.debug(
                f"LibDoc: Priority-ordered session libraries: {sorted_session_libs}"
            )

            # Search in session libraries first, respecting priority order
            session_keywords = rf_doc_storage.get_keywords_from_libraries(
                sorted_session_libs
            )

            # Find exact match in session libraries first (highest priority first)
            for lib_name in sorted_session_libs:
                for kw in session_keywords:
                    if (
                        kw.library == lib_name
                        and kw.name.lower() == keyword_name.lower()
                    ):
                        logger.debug(
                            f"LibDoc: Found '{keyword_name}' in session library '{lib_name}' (priority {get_library_config(lib_name).load_priority if get_library_config(lib_name) else 999})"
                        )
                        return self._convert_libdoc_to_keyword_info(kw)

            # Try fuzzy matching in session libraries if no exact match
            for lib_name in sorted_session_libs:
                fuzzy_result = self._fuzzy_match_in_library(
                    keyword_name, session_keywords, lib_name
                )
                if fuzzy_result:
                    logger.debug(
                        f"LibDoc: Fuzzy match '{keyword_name}' -> '{fuzzy_result.name}' in session library '{lib_name}'"
                    )
                    return fuzzy_result

            logger.debug(
                f"LibDoc: No match for '{keyword_name}' in session libraries, checking if fallback to global search is needed"
            )

        # CRITICAL FIX: Session Library Isolation
        # If session libraries are specified, NEVER fall back to global search
        # This ensures session boundaries are respected for ALL session types
        if session_libraries:
            logger.debug(
                f"LibDoc: Session libraries specified {session_libraries} - no global fallback allowed"
            )
            logger.debug(
                f"LibDoc: Keyword '{keyword_name}' not found in session libraries - returning None for strict session isolation"
            )
            return None

        # Priority 2: Global search only allowed when no session libraries are specified
        # This maintains compatibility for non-session-aware operations
        logger.debug("LibDoc: No session libraries specified - allowing global search")

        if active_library:
            logger.debug(
                f"LibDoc: Active library filter '{active_library}' specified, including in global search"
            )

        # Get all keywords for global fallback (only when no session libraries)
        try:
            keywords = rf_doc_storage.get_all_keywords()
            logger.debug(
                f"LibDoc: Global fallback search - found {len(keywords)} total keywords"
            )
        except Exception as e:
            logger.debug(f"LibDoc: Failed to get keywords for global fallback: {e}")
            return None

        # Global search for exact match (fallback)
        for kw in keywords:
            if kw.name.lower() == keyword_name.lower():
                logger.debug(
                    f"LibDoc: Global fallback found '{keyword_name}' in '{kw.library}'"
                )
                return self._convert_libdoc_to_keyword_info(kw)

        # Global fuzzy matching (fallback)
        fuzzy_result = self._fuzzy_match_global(keyword_name, keywords)
        if fuzzy_result:
            logger.debug(
                f"LibDoc: Global fuzzy match '{keyword_name}' -> '{fuzzy_result.name}' in '{fuzzy_result.library}'"
            )
            return fuzzy_result

        logger.debug(f"LibDoc: No match found for '{keyword_name}' in any library")
        return None

    def _convert_libdoc_to_keyword_info(self, kw) -> KeywordInfo:
        """Convert RF LibDoc keyword to our KeywordInfo format."""
        return KeywordInfo(
            name=kw.name,
            library=kw.library,
            method_name=kw.name.replace(" ", "_").lower(),
            doc=kw.doc,
            short_doc=kw.short_doc,
            args=kw.args,
            defaults={},  # LibDoc doesn't provide defaults in same format
            tags=kw.tags,
            is_builtin=(kw.library in BUILTIN_LIBRARIES),
        )

    def _fuzzy_match_in_library(
        self, keyword_name: str, keywords: List, target_library: str
    ) -> Optional[KeywordInfo]:
        """Perform fuzzy matching for a keyword within a specific library."""
        normalized = keyword_name.lower().strip()
        variations = [
            normalized.replace(" ", ""),  # Remove spaces
            normalized.replace("_", " "),  # Replace underscores
            normalized.replace("-", " "),  # Replace hyphens
        ]

        for variation in variations:
            for kw in keywords:
                if (
                    kw.library == target_library
                    and kw.name.lower().replace(" ", "") == variation
                ):
                    return self._convert_libdoc_to_keyword_info(kw)

        return None

    def _fuzzy_match_global(
        self, keyword_name: str, keywords: List
    ) -> Optional[KeywordInfo]:
        """Perform fuzzy matching globally across all keywords."""
        normalized = keyword_name.lower().strip()
        variations = [
            normalized.replace(" ", ""),  # Remove spaces
            normalized.replace("_", " "),  # Replace underscores
            normalized.replace("-", " "),  # Replace hyphens
        ]

        for variation in variations:
            for kw in keywords:
                if kw.name.lower().replace(" ", "") == variation:
                    return self._convert_libdoc_to_keyword_info(kw)

        return None

    def _execute_with_rf_type_conversion(self, method, keyword_info, corrected_args):
        """Execute method using Robot Framework's native type conversion system."""
        try:
            import inspect

            from robot.running.arguments import ArgumentSpec
            from robot.running.arguments.argumentresolver import ArgumentResolver
            from robot.running.arguments.typeconverters import TypeConverter
            from robot.running.arguments.typeinfo import TypeInfo

            # Get method signature
            sig = inspect.signature(method)

            # Smart detection: only process named arguments if we actually find valid ones
            potential_named_args = any("=" in arg for arg in corrected_args)

            if potential_named_args:
                # Try to use the actual method signature for parameter validation (more reliable than KeywordInfo.args)
                try:
                    param_names = list(sig.parameters.keys())
                    logger.debug(
                        f"Method signature parameters for {keyword_info.name}: {param_names}"
                    )

                    # Parse arguments to see if we actually have valid named arguments
                    positional_args, named_args = (
                        self._split_args_using_method_signature(corrected_args, sig)
                    )
                except (AttributeError, TypeError):
                    # Fallback to the original KeywordInfo-based approach (for tests and edge cases)
                    logger.debug(
                        "Unable to inspect method signature, falling back to KeywordInfo.args approach"
                    )
                    if hasattr(keyword_info, "args") and keyword_info.args:
                        positional_args, named_args = (
                            self._split_args_into_positional_and_named(
                                corrected_args, keyword_info.args
                            )
                        )
                    else:
                        # No signature info available, fall back to positional-only
                        raise Exception("fallback_to_positional")

                # Only use named argument processing if we found actual named arguments
                if named_args:
                    logger.debug(
                        f"Found valid named arguments for {keyword_info.name}: {named_args}"
                    )

                    # Create ArgumentSpec from LibDoc signature
                    spec = self._create_argument_spec_from_libdoc(keyword_info.args)

                    # Use Robot Framework's ArgumentResolver to properly handle arguments
                    resolver = ArgumentResolver(spec)
                    resolved_positional, resolved_named = resolver.resolve(
                        positional_args, named_args
                    )

                    # ArgumentResolver might return different types, handle them properly
                    if not isinstance(resolved_named, dict):
                        logger.debug(
                            f"ArgumentResolver returned non-dict for named args: {type(resolved_named)} = {resolved_named}"
                        )
                        if hasattr(resolved_named, "__iter__") and not isinstance(
                            resolved_named, str
                        ):
                            # Convert list/tuple to dict if possible
                            resolved_named = (
                                dict(resolved_named) if resolved_named else {}
                            )
                        else:
                            resolved_named = {}

                    # Apply type conversion to resolved arguments
                    converted_positional = self._convert_positional_with_rf(
                        resolved_positional, sig
                    )
                    converted_named = self._convert_named_with_rf(resolved_named, sig)

                    # Execute with both positional and named arguments
                    if converted_named:
                        result = method(*converted_positional, **converted_named)
                    else:
                        result = method(*converted_positional)

                    logger.debug(
                        f"RF native type conversion succeeded for {keyword_info.name} with named args: {list(converted_named.keys()) if converted_named else 'none'}"
                    )
                    return (
                        "executed",
                        result,
                    )  # Return tuple to indicate execution happened
                else:
                    # Arguments contain '=' but none are valid named arguments (e.g., locator strings)
                    logger.debug(
                        f"Arguments contain '=' but no valid named arguments found for {keyword_info.name}, using positional processing"
                    )
                    raise Exception("fallback_to_positional")  # Trigger fallback

            else:
                # No '=' signs detected, use original positional-only logic
                logger.debug(
                    f"No potential named arguments detected for {keyword_info.name}, using positional processing"
                )
                raise Exception("fallback_to_positional")  # Trigger fallback

        except Exception as e:
            if str(e) == "fallback_to_positional":
                # Use the original positional-only logic
                logger.debug(
                    f"Falling back to positional-only processing for {keyword_info.name}"
                )

                converted_args = []
                param_list = list(sig.parameters.values())

                for i, (arg_value, param) in enumerate(zip(corrected_args, param_list)):
                    if param.annotation != inspect.Parameter.empty:
                        # Create TypeInfo from annotation
                        type_info = TypeInfo.from_type_hint(param.annotation)

                        # Get converter for this type
                        converter = TypeConverter.converter_for(type_info)

                        # BUGFIX: Convert non-string arguments to strings before passing to RF TypeConverter
                        # Robot Framework's TypeConverter expects string input that it converts to other types
                        if not isinstance(arg_value, str):
                            string_arg_value = str(arg_value)
                            logger.debug(
                                f"Converting non-string argument {arg_value} (type: {type(arg_value).__name__}) to string '{string_arg_value}' for RF TypeConverter"
                            )
                        else:
                            string_arg_value = arg_value

                        # Convert the argument - handle both parameter objects and strings
                        param_name = (
                            param.name if hasattr(param, "name") else str(param)
                        )

                        try:
                            converted_value = converter.convert(
                                string_arg_value, param_name
                            )
                            converted_args.append(converted_value)
                            logger.debug(
                                f"RF converted arg {i} '{param_name}': {arg_value} -> {converted_value} (type: {type(converted_value).__name__})"
                            )
                        except Exception as conversion_error:
                            # Enhanced error handling for type conversion failures
                            error_msg = str(conversion_error)

                            # Check if this looks like a misplaced named argument
                            if (
                                "=" in string_arg_value
                                and "does not have member" in error_msg
                            ):
                                # This looks like a named argument that was treated as positional
                                logger.debug(
                                    f"Type conversion failed for '{param_name}' with value '{string_arg_value}' - appears to be misplaced named argument"
                                )
                                raise ValueError(
                                    f"Argument '{param_name}' got value '{string_arg_value}' which appears to be a misplaced named argument. "
                                    f"Original error: {error_msg}"
                                )
                            elif "'str' object has no attribute 'name'" in error_msg:
                                # This is the specific error we're trying to fix
                                logger.debug(
                                    f"Caught 'str' object error for parameter '{param_name}' with value '{string_arg_value}'"
                                )
                                raise ValueError(
                                    f"Parameter '{param_name}' received invalid value '{string_arg_value}'. "
                                    f"This may be a misplaced named argument or invalid type conversion. "
                                    f"Original error: {error_msg}"
                                )
                            else:
                                # Re-raise other conversion errors with better context
                                raise ValueError(
                                    f"Type conversion failed for parameter '{param_name}' with value '{string_arg_value}': {error_msg}"
                                )

                    else:
                        # No type annotation, use as-is
                        converted_args.append(arg_value)

                # Execute with converted arguments
                result = method(*converted_args)

                logger.debug(
                    f"RF native type conversion succeeded for {keyword_info.name}"
                )
                return (
                    "executed",
                    result,
                )  # Return tuple to indicate execution happened
            else:
                # Re-raise other exceptions
                raise e

        except ImportError as ie:
            logger.debug(f"Robot Framework type conversion not available: {ie}")
            return ("not_available", None)  # Indicate type conversion wasn't available
        except Exception as e:
            logger.debug(
                f"RF native type conversion failed for {keyword_info.name}: {e}"
            )
            # Log more details for debugging
            logger.debug(
                f"Method signature: {inspect.signature(method) if 'inspect' in locals() else 'N/A'}"
            )
            logger.debug(f"Corrected args: {corrected_args}")
            import traceback

            logger.debug(f"Full traceback: {traceback.format_exc()}")
            # CRITICAL: Even though execution failed, it DID execute - don't try again
            raise e  # Re-raise the exception instead of returning None

    def _split_args_into_positional_and_named(
        self, args: List[str], signature_args: List[str] = None
    ) -> tuple[List[str], Dict[str, str]]:
        """
        Split user arguments into positional and named arguments using LibDoc signature information.

        This method reuses the logic from rf_native_type_converter to ensure consistency.
        """
        positional = []
        named = {}

        # Build list of valid parameter names from signature
        valid_param_names = set()
        if signature_args:
            for arg_str in signature_args:
                if ":" in arg_str:
                    param_name = arg_str.split(":", 1)[0].strip()
                    if param_name.startswith("*"):
                        param_name = param_name[1:]  # Remove * for varargs
                    if param_name.startswith("*"):
                        param_name = param_name[1:]  # Remove ** for kwargs
                    if param_name and not param_name.startswith("*"):
                        valid_param_names.add(param_name)

        for arg in args:
            if "=" in arg and self._looks_like_named_arg(arg, valid_param_names):
                key, value = arg.split("=", 1)
                named[key.strip()] = value
            else:
                positional.append(arg)

        return positional, named

    def _convert_rf_keyword_to_keyword_info(self, rf_kw_info) -> KeywordInfo:
        """Convert RFKeywordInfo to our KeywordInfo format."""
        from robotmcp.models.library_models import KeywordInfo

        # Find the actual method name by reverse lookup
        method_name = self._find_method_name_for_keyword(
            rf_kw_info.name, rf_kw_info.library
        )

        return KeywordInfo(
            name=rf_kw_info.name,
            library=rf_kw_info.library,
            method_name=method_name,
            args=rf_kw_info.args,
            defaults={},  # Could extract from args if needed
            doc=rf_kw_info.doc,
            short_doc=rf_kw_info.short_doc,
            tags=rf_kw_info.tags,
            is_builtin=False,  # RF LibDoc keywords are typically not built-in
        )

    def _find_method_name_for_keyword(
        self, keyword_name: str, library_name: str
    ) -> str:
        """Find the actual Python method name for a keyword (handles decorators)."""
        try:
            if library_name in self.library_manager.libraries:
                library = self.library_manager.libraries[library_name]
                if library.instance:
                    # Check all methods for robot_name matching the keyword
                    for attr_name in dir(library.instance):
                        if attr_name.startswith("_"):
                            continue
                        try:
                            method = getattr(library.instance, attr_name)
                            if callable(method) and hasattr(method, "robot_name"):
                                if method.robot_name == keyword_name:
                                    return attr_name
                            # Also check converted method names
                            converted_name = " ".join(
                                word.capitalize() for word in attr_name.split("_")
                            )
                            if converted_name == keyword_name:
                                return attr_name
                        except:
                            continue
        except Exception as e:
            logger.debug(f"Method name lookup failed for {keyword_name}: {e}")

        # Fallback to simple conversion
        return keyword_name.lower().replace(" ", "_")

    def _split_args_using_method_signature(
        self, corrected_args: List[str], method_signature
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Split arguments into positional and named using the actual method signature.
        This is more reliable than using KeywordInfo.args which may be outdated or incorrect.
        """
        positional_args = []
        named_args = {}

        # Get parameter names from the actual method signature
        param_names = list(method_signature.parameters.keys())

        for arg in corrected_args:
            if "=" in arg:
                # Potential named argument
                param_name, param_value = arg.split("=", 1)

                # Check if this is a valid parameter name for the method
                if param_name in param_names:
                    named_args[param_name] = param_value
                    logger.debug(f"Valid named argument: {param_name}={param_value}")
                else:
                    # Not a valid parameter name - treat as positional (e.g., locator string)
                    positional_args.append(arg)
                    logger.debug(
                        f"Invalid parameter name '{param_name}' - treating '{arg}' as positional"
                    )
            else:
                # Regular positional argument
                positional_args.append(arg)

        return positional_args, named_args

    def _looks_like_named_arg(self, arg: str, valid_param_names: set = None) -> bool:
        """
        Check if an argument looks like a named argument.

        Uses valid parameter names from LibDoc signature to distinguish between
        actual named parameters and locator strings containing '=' characters.
        """
        if "=" not in arg:
            return False

        key_part = arg.split("=", 1)[0].strip()

        # Must be valid Python identifier
        if not key_part.isidentifier():
            return False

        # If we have valid parameter names from signature, only treat as named arg
        # if the key matches an actual parameter name
        if valid_param_names:
            return key_part in valid_param_names

        # Fallback: assume it's a named argument if it's a valid identifier
        return True

    def _create_argument_spec_from_libdoc(self, libdoc_args: List[str]):
        """Create Robot Framework ArgumentSpec from LibDoc signature."""
        from robot.running.arguments import ArgumentSpec

        positional_or_named = []
        kw_only = []
        defaults = {}
        var_positional = None
        var_named = None
        keyword_only_separator_found = False

        for arg_str in libdoc_args:
            # Handle the special "*" separator for keyword-only arguments
            if arg_str.strip() == "*":
                keyword_only_separator_found = True
                continue

            if ":" in arg_str:
                # Extract parameter name and default value
                name, default_part = arg_str.split(":", 1)
                name = name.strip()

                # Handle varargs and kwargs
                if name.startswith("**"):
                    var_named = name[2:] if len(name) > 2 else "kwargs"
                elif name.startswith("*"):
                    var_positional = name[1:] if len(name) > 1 else "args"
                elif name not in ["*", "**"]:  # Only add regular parameters
                    if keyword_only_separator_found:
                        kw_only.append(name)
                    else:
                        positional_or_named.append(name)

                    # Extract default value if present
                    if "=" in default_part:
                        defaults[name] = default_part.split("=", 1)[1].strip()

        return ArgumentSpec(
            positional_or_named=positional_or_named,
            named_only=kw_only,
            defaults=defaults,
            var_positional=var_positional,
            var_named=var_named,
        )

    def _convert_positional_with_rf(self, args: List[str], signature) -> List[Any]:
        """Convert positional arguments using RF type conversion."""
        from robot.running.arguments.typeconverters import TypeConverter
        from robot.running.arguments.typeinfo import TypeInfo

        converted_args = []
        param_list = list(signature.parameters.values())

        for i, arg_value in enumerate(args):
            if i < len(param_list):
                param = param_list[i]
                if param.annotation != param.empty:
                    # Create TypeInfo from annotation
                    type_info = TypeInfo.from_type_hint(param.annotation)

                    # Get converter for this type
                    converter = TypeConverter.converter_for(type_info)

                    # Convert the argument - handle both parameter objects and strings
                    param_name = param.name if hasattr(param, "name") else str(param)
                    converted_value = converter.convert(arg_value, param_name)
                    converted_args.append(converted_value)

                    logger.debug(
                        f"RF converted positional arg {i} '{param_name}': {arg_value} -> {converted_value} (type: {type(converted_value).__name__})"
                    )
                else:
                    # No type annotation, use as-is
                    converted_args.append(arg_value)
            else:
                # More arguments than parameters, use as-is
                converted_args.append(arg_value)

        return converted_args

    def _convert_named_with_rf(
        self, named_args: Dict[str, str], signature
    ) -> Dict[str, Any]:
        """Convert named arguments using RF type conversion."""
        from robot.running.arguments.typeconverters import TypeConverter
        from robot.running.arguments.typeinfo import TypeInfo

        converted_named = {}
        param_dict = {param.name: param for param in signature.parameters.values()}

        for name, value in named_args.items():
            if name in param_dict:
                param = param_dict[name]
                if param.annotation != param.empty:
                    # Create TypeInfo from annotation
                    type_info = TypeInfo.from_type_hint(param.annotation)

                    # Get converter for this type
                    converter = TypeConverter.converter_for(type_info)

                    # Convert the argument
                    converted_value = converter.convert(value, name)
                    converted_named[name] = converted_value

                    logger.debug(
                        f"RF converted named arg '{name}': {value} -> {converted_value} (type: {type(converted_value).__name__})"
                    )
                else:
                    # No type annotation, use as-is
                    converted_named[name] = value
            else:
                # Unknown parameter, use as-is
                converted_named[name] = value

        return converted_named

    def get_keyword_suggestions(self, keyword_name: str, limit: int = 5) -> List[str]:
        """Get keyword suggestions based on partial match."""
        return self.keyword_discovery.get_keyword_suggestions(keyword_name, limit)

    def suggest_similar_keywords(
        self, keyword_name: str, max_suggestions: int = 5
    ) -> List[KeywordInfo]:
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
                if (
                    regex.search(keyword_info.name)
                    or regex.search(keyword_info.doc)
                    or regex.search(keyword_info.library)
                ):
                    matches.append(keyword_info)
            return matches
        except re.error:
            # If pattern is not valid regex, do simple string matching
            pattern_lower = pattern.lower()
            return [
                info
                for info in self.keyword_discovery.keyword_cache.values()
                if pattern_lower in info.name.lower()
                or pattern_lower in info.doc.lower()
                or pattern_lower in info.library.lower()
            ]

    def get_keywords_by_library(self, library_name: str) -> List[KeywordInfo]:
        """Get all keywords from a specific library with enhanced validation."""

        # First, ensure library is loaded
        if library_name not in self.libraries:
            logger.warning(f"Library '{library_name}' not loaded for keyword discovery")

            # Try to load on demand
            if not self.library_manager.load_library_on_demand(
                library_name, self.keyword_discovery
            ):
                logger.error(f"Failed to load library '{library_name}' on demand")
                return []

            # Add keywords to cache after loading
            if library_name in self.libraries:
                lib_info = self.libraries[library_name]
                self.keyword_discovery.add_keywords_to_cache(lib_info)

        # Use enhanced keyword discovery
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

    def _parse_arguments_for_keyword(
        self,
        keyword_name: str,
        args: List[str],
        library_name: str = None,
        session_variables: Dict[str, Any] = None,
    ) -> ParsedArguments:
        """Parse arguments using LibDoc information for a specific keyword."""
        return self.argument_processor.parse_arguments_for_keyword(
            keyword_name, args, library_name, session_variables
        )

    def _parse_arguments_with_rf_spec(
        self, keyword_info: KeywordInfo, args: List[str]
    ) -> ParsedArguments:
        """Parse arguments using Robot Framework's native ArgumentSpec if available."""
        try:
            from robot.running.arguments import ArgumentSpec
            from robot.running.arguments.argumentresolver import ArgumentResolver

            # Try to create ArgumentSpec from keyword info
            if hasattr(keyword_info, "args") and keyword_info.args:
                spec = ArgumentSpec(
                    positional_or_named=keyword_info.args,
                    defaults=keyword_info.defaults
                    if hasattr(keyword_info, "defaults")
                    else {},
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

    def _get_session_library_loading_info(
        self, session_id: str = None, active_library: str = None
    ) -> Dict[str, Any]:
        """
        Get information about session library loading status for improved error messages.

        IMPROVEMENT: This helps distinguish between 'keyword not found' vs 'library not loaded' errors.
        """
        info = {
            "has_unloaded_session_libraries": False,
            "session_libraries": [],
            "loaded_libraries": [],
            "active_library": active_library,
            "session_id": session_id,
        }

        try:
            if session_id and self.session_manager:
                session = self.session_manager.get_session(session_id)
                if session:
                    info["session_libraries"] = list(session.imported_libraries)
                    info["loaded_libraries"] = list(
                        self.library_manager.libraries.keys()
                    )

                    # Check if session has imported libraries that aren't loaded
                    unloaded_libs = set(session.imported_libraries) - set(
                        self.library_manager.libraries.keys()
                    )
                    info["has_unloaded_session_libraries"] = len(unloaded_libs) > 0
                    info["unloaded_libraries"] = list(unloaded_libs)

        except Exception as e:
            logger.debug(f"Error getting session library loading info: {e}")

        return info

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
                    "scope": lib.scope,
                }
                for name, lib in self.library_manager.libraries.items()
            },
            "failed_imports": self.library_manager.failed_imports,
            "total_keywords": len(self.keyword_discovery.keyword_cache),
        }

    async def _ensure_session_libraries(
        self, session_id: str, keyword_name: str
    ) -> None:
        """
        Ensure required libraries are loaded for the session.

        Phase 3 Enhancement: Improved RequestsLibrary loading consistency.
        """
        if not self.session_manager:
            logger.debug(
                "No session manager available, skipping session library loading"
            )
            return

        session = self.session_manager.get_session(session_id)
        if not session:
            session = self.session_manager.create_session(session_id)

        # CRITICAL FIX: Check if session libraries need to be reloaded
        # This happens when session search order changes (e.g., auto-configuration)
        logger.debug(
            f"LIBRARY_RELOAD_DEBUG: Session {session_id} libraries_loaded={session.libraries_loaded}"
        )
        if not session.libraries_loaded:
            logger.info(
                f"LIBRARY_RELOAD_DEBUG: Session {session_id} libraries need reloading, triggering immediate reload"
            )

            # Get all libraries that should be loaded for this session
            all_required_libraries = session.get_libraries_to_load()

            if all_required_libraries:
                logger.info(
                    f"LIBRARY_RELOAD_DEBUG: Reloading session libraries: {all_required_libraries}"
                )
                # Force reload all session libraries
                self.library_manager.load_session_libraries(
                    all_required_libraries, self.keyword_discovery
                )

                # Add keywords to cache for all loaded libraries
                for lib_name in all_required_libraries:
                    if lib_name in self.library_manager.libraries:
                        lib_info = self.library_manager.libraries[lib_name]
                        self.keyword_discovery.add_keywords_to_cache(lib_info)
                        session.mark_library_loaded(lib_name)

                # Mark session libraries as loaded
                logger.debug(
                    f"LIBRARIES_LOADED_DEBUG: Setting libraries_loaded = True for session {session_id} after reload"
                )
                session.libraries_loaded = True
                logger.info(
                    f"LIBRARY_RELOAD_DEBUG: Session {session_id} libraries reloaded successfully"
                )
            else:
                logger.warning(
                    f"LIBRARY_RELOAD_DEBUG: No libraries to reload for session {session_id}"
                )
        else:
            logger.debug(
                f"LIBRARY_RELOAD_DEBUG: Session {session_id} libraries already loaded, skipping reload"
            )

        # Get libraries that should be loaded for this session
        required_libraries = session.get_libraries_to_load()
        optional_libraries = session.get_optional_libraries()

        # Load any missing required libraries
        libraries_to_load = []
        for lib_name in required_libraries:
            if lib_name not in self.library_manager.libraries:
                libraries_to_load.append(lib_name)

        if libraries_to_load:
            self.library_manager.load_session_libraries(
                libraries_to_load, self.keyword_discovery
            )

            # Add new keywords to cache
            for lib_name in libraries_to_load:
                if lib_name in self.library_manager.libraries:
                    lib_info = self.library_manager.libraries[lib_name]
                    self.keyword_discovery.add_keywords_to_cache(lib_info)
                    session.mark_library_loaded(lib_name)

        # Phase 3 Fix: Proactive RequestsLibrary loading for HTTP keywords
        # Check if this is an HTTP-related keyword BEFORE trying to find it
        if self._is_http_keyword(keyword_name):
            logger.info(
                f"Phase 3: '{keyword_name}' is HTTP keyword, checking if RequestsLibrary is loaded"
            )
            if "RequestsLibrary" not in self.library_manager.libraries:
                logger.warning(
                    f"Phase 3: RequestsLibrary NOT in library manager, attempting to load for '{keyword_name}'"
                )
                if self.library_manager.load_library_on_demand(
                    "RequestsLibrary", self.keyword_discovery
                ):
                    lib_info = self.library_manager.libraries["RequestsLibrary"]
                    self.keyword_discovery.add_keywords_to_cache(lib_info)
                    session.mark_library_loaded("RequestsLibrary")
                    logger.info(
                        f"Phase 3: Successfully loaded RequestsLibrary for '{keyword_name}'"
                    )
                else:
                    logger.error(
                        f"Phase 3: Failed to load RequestsLibrary for '{keyword_name}' - library loading failed"
                    )
            else:
                logger.info(
                    f"Phase 3: RequestsLibrary already loaded for '{keyword_name}'"
                )

        # Try to load library on-demand if keyword is not found
        if not self.find_keyword(keyword_name):
            # Try to determine which library might have this keyword, respecting session context
            potential_library = self._guess_library_for_keyword(keyword_name, session)
            if (
                potential_library
                and potential_library not in self.library_manager.libraries
            ):
                if self.library_manager.load_library_on_demand(
                    potential_library, self.keyword_discovery
                ):
                    lib_info = self.library_manager.libraries[potential_library]
                    self.keyword_discovery.add_keywords_to_cache(lib_info)
                    session.mark_library_loaded(potential_library)

        # Mark session libraries as loaded after all loading operations complete
        if not session.libraries_loaded:
            session.libraries_loaded = True

    def _is_http_keyword(self, keyword_name: str) -> bool:
        """
        Check if a keyword is HTTP-related and likely needs RequestsLibrary.

        Phase 3 Addition: More precise HTTP keyword detection.
        """
        keyword_lower = keyword_name.lower()

        # Exact HTTP keyword matches (case-insensitive)
        exact_http_keywords = {
            "post",
            "get",
            "put",
            "delete",
            "patch",
            "head",
            "options",
            "create session",
            "delete all sessions",
            "get request",
            "post request",
            "put request",
            "delete request",
            "patch request",
            "head request",
            "options request",
            "post on session",
            "get on session",
            "put on session",
            "delete on session",
            "patch on session",
            "head on session",
            "options on session",
        }

        if keyword_lower in exact_http_keywords:
            return True

        # HTTP-related patterns
        http_patterns = [
            r"\b(get|post|put|delete|patch|head|options)\s*(request|on\s*session)?\b",
            r"\bcreate\s*session\b",
            r"\bdelete\s*(all\s*)?sessions?\b",
        ]

        for pattern in http_patterns:
            if re.search(pattern, keyword_lower):
                return True

        return False

    def _guess_library_for_keyword(
        self, keyword_name: str, session: "ExecutionSession" = None
    ) -> Optional[str]:
        """Guess which library might contain a keyword based on name patterns, respecting session context."""
        keyword_lower = keyword_name.lower()

        # If session has a specific web automation library, respect it for browser keywords
        if session:
            web_lib = session.get_web_automation_library()
            if web_lib and any(
                term in keyword_lower
                for term in [
                    "browser",
                    "click",
                    "fill",
                    "navigate",
                    "page",
                    "screenshot",
                ]
            ):
                # Session has explicit web automation library - use it instead of guessing
                logger.debug(
                    f"Session has {web_lib}, using it for '{keyword_name}' instead of auto-detection"
                )
                return web_lib

        # Phase 3 Enhancement: Use the more precise HTTP keyword detection
        if self._is_http_keyword(keyword_name):
            return "RequestsLibrary"

        # Common keyword patterns to library mappings
        patterns = {
            r"\b(click|fill|navigate|browser|page|screenshot)\b": "Browser",
            r"\b(parse xml|get element|xpath)\b": "XML",
            r"\b(run process|start process|terminate)\b": "Process",
            r"\b(create file|remove file|directory)\b": "OperatingSystem",
            r"\b(get current date|convert date)\b": "DateTime",
        }

        for pattern, library in patterns.items():
            if re.search(pattern, keyword_lower):
                return library

        return None

    def _find_keyword_with_session(
        self, keyword_name: str, active_library: str = None, session_id: str = None
    ) -> Optional[KeywordInfo]:
        """Find keyword respecting session search order with strict boundary enforcement."""
        # If no session or session manager, use normal search
        if not session_id or not self.session_manager:
            return self.find_keyword(keyword_name, active_library)

        session = self.session_manager.get_session(session_id)
        if not session:
            return self.find_keyword(keyword_name, active_library)

        # Get session configuration
        search_order = session.get_search_order()
        session_type = session.get_session_type()

        logger.debug(
            f"Session '{session_id}' search order: {search_order}, type: {session_type.value}"
        )

        # Phase 3 Enhancement: Handle active_library conflicts with session search order
        if active_library and active_library not in search_order:
            logger.warning(
                f"Active library '{active_library}' not in session '{session_id}' search order {search_order}"
            )
            # Phase 3: Instead of ignoring, search session libraries first, then active library as fallback
            logger.info(
                f"Phase 3: Will prioritize session search order over active_library '{active_library}'"
            )
            # Don't nullify active_library - use it as fallback after session search

        # Search in session search order with exact matches first using LibDoc (decorator-aware)
        for lib_name in search_order:
            if lib_name in self.library_manager.libraries:
                # CRITICAL FIX: Use LibDoc-based search which handles decorators correctly
                try:
                    from robotmcp.utils.rf_libdoc_integration import get_rf_doc_storage

                    rf_doc_storage = get_rf_doc_storage()

                    logger.info(
                        f"SESSION SEARCH DEBUG: Looking for '{keyword_name}' in library '{lib_name}'"
                    )
                    if rf_doc_storage.is_available():
                        lib_keywords = rf_doc_storage.get_keywords_by_library(lib_name)
                        logger.info(
                            f"SESSION SEARCH DEBUG: Found {len(lib_keywords)} keywords in '{lib_name}' via LibDoc"
                        )
                        for kw in lib_keywords:
                            if kw.name.lower() == keyword_name.lower():
                                logger.info(
                                    f"Found '{keyword_name}' in '{lib_name}' via session search order using LibDoc (type: {session_type.value})"
                                )
                                # Convert to our KeywordInfo format
                                return self._convert_rf_keyword_to_keyword_info(kw)
                        logger.info(
                            f"SESSION SEARCH DEBUG: '{keyword_name}' not found in {lib_name} LibDoc keywords"
                        )
                    else:
                        logger.info("SESSION SEARCH DEBUG: LibDoc not available")
                except Exception as e:
                    logger.warning(f"LibDoc search failed for {lib_name}: {e}")
                    import traceback

                    logger.debug(f"LibDoc traceback: {traceback.format_exc()}")

                # Fallback to inspection-based search
                lib_keywords = self.get_keywords_by_library(lib_name)
                for kw in lib_keywords:
                    if kw.name.lower() == keyword_name.lower():
                        logger.info(
                            f"Found '{keyword_name}' in '{lib_name}' via session search order (inspection fallback)"
                        )
                        return kw

        # Try fuzzy matching within session boundaries
        normalized = keyword_name.lower().strip()
        variations = [
            normalized.replace(" ", ""),  # Remove spaces
            normalized.replace("_", " "),  # Replace underscores
            normalized.replace("-", " "),  # Replace hyphens
        ]

        for lib_name in search_order:
            if lib_name in self.library_manager.libraries:
                lib_keywords = self.get_keywords_by_library(lib_name)
                for variation in variations:
                    for kw in lib_keywords:
                        if kw.name.lower().replace(" ", "") == variation:
                            logger.info(
                                f"Found '{keyword_name}' in '{lib_name}' via fuzzy search order match (type: {session_type.value})"
                            )
                            return kw

        # Phase 3 Enhancement: If keyword not found in session search order, try active_library as fallback
        if active_library and active_library not in search_order:
            logger.info(
                f"Phase 3: Trying active_library '{active_library}' as fallback for '{keyword_name}'"
            )
            # Try the active library that wasn't in session search order
            fallback_keyword = self.find_keyword(keyword_name, active_library)
            if fallback_keyword:
                logger.info(
                    f"Phase 3: Found '{keyword_name}' in fallback active_library '{active_library}'"
                )
                return fallback_keyword

        # Phase 3 Enhancement: If still not found, try searching ALL session libraries directly
        for lib_name in search_order:
            if lib_name in self.library_manager.libraries:
                fallback_keyword = self.find_keyword(keyword_name, lib_name)
                if fallback_keyword:
                    logger.info(
                        f"Phase 3: Found '{keyword_name}' in session library '{lib_name}' via direct search"
                    )
                    return fallback_keyword

        logger.info(
            f"Keyword '{keyword_name}' not found within session '{session_id}' boundaries or fallback (type: {session_type.value}, search_order: {search_order})"
        )
        return None

    def _get_diagnostic_info(
        self, keyword_name: str, session_id: str = None, active_library: str = None
    ) -> Dict[str, Any]:
        """
        Phase 4: Gather comprehensive diagnostic information for keyword execution failures.

        Args:
            keyword_name: The keyword that failed to execute
            session_id: Session ID for context
            active_library: Active library for context

        Returns:
            Dictionary with diagnostic information
        """
        diagnostics = {
            "keyword_analysis": {
                "is_http_keyword": self._is_http_keyword(keyword_name),
                "keyword_variations": [
                    keyword_name,
                    keyword_name.lower(),
                    keyword_name.replace(" ", ""),
                    keyword_name.replace("_", " "),
                ],
            },
            "library_status": {
                "loaded_libraries": list(self.library_manager.libraries.keys()),
                "failed_imports": self.library_manager.failed_imports,
                "total_libraries": len(self.library_manager.libraries),
                "requests_library_loaded": "RequestsLibrary"
                in self.library_manager.libraries,
            },
            "session_info": {},
            "search_analysis": {
                "active_library": active_library,
                "active_library_available": active_library
                in self.library_manager.libraries
                if active_library
                else None,
            },
        }

        # Phase 4: Session-specific diagnostics
        if session_id and self.session_manager:
            session = self.session_manager.get_session(session_id)
            if session:
                search_order = session.get_search_order()
                session_type = session.get_session_type()

                diagnostics["session_info"] = {
                    "session_id": session_id,
                    "session_type": session_type.value,
                    "search_order": search_order,
                    "active_library_in_search_order": active_library in search_order
                    if active_library
                    else None,
                    "imported_libraries": session.imported_libraries,
                    "loaded_libraries": list(session.loaded_libraries),
                    "step_count": session.step_count,
                }

                # Check if keyword exists in any session library
                diagnostics["search_analysis"]["keyword_found_in_libraries"] = {}
                for lib_name in search_order:
                    if lib_name in self.library_manager.libraries:
                        found = bool(self.find_keyword(keyword_name, lib_name))
                        diagnostics["search_analysis"]["keyword_found_in_libraries"][
                            lib_name
                        ] = found

        # Phase 4: RequestsLibrary specific diagnostics for HTTP keywords
        if self._is_http_keyword(keyword_name):
            diagnostics["http_keyword_diagnostics"] = {
                "requests_library_status": self._get_requests_library_status(),
                "phase_analysis": {
                    "phase1_rf_registration": self._check_requests_library_rf_registration(),
                    "phase2_session_state": self._check_requests_library_session_state(
                        session_id
                    ),
                    "phase3_library_loading": "RequestsLibrary"
                    in self.library_manager.libraries,
                },
            }

        return diagnostics

    def _get_requests_library_status(self) -> Dict[str, Any]:
        """Get detailed RequestsLibrary status for diagnostics."""
        if "RequestsLibrary" not in self.library_manager.libraries:
            return {"loaded": False, "reason": "Not loaded in library manager"}

        lib_info = self.library_manager.libraries["RequestsLibrary"]
        return {
            "loaded": True,
            "instance_available": lib_info.instance is not None,
            "keywords_count": len(lib_info.keywords)
            if hasattr(lib_info, "keywords")
            else "unknown",
            "import_time": lib_info.import_time
            if hasattr(lib_info, "import_time")
            else None,
        }

    def _check_requests_library_rf_registration(self) -> Dict[str, Any]:
        """Check Phase 1: RequestsLibrary RF context registration."""
        try:
            from robot.running.context import EXECUTION_CONTEXTS

            if not EXECUTION_CONTEXTS.current:
                return {"status": "no_rf_context", "registered": False}

            rf_context = EXECUTION_CONTEXTS.current
            try:
                requests_lib = rf_context.namespace.get_library_instance(
                    "RequestsLibrary"
                )
                return {
                    "status": "success" if requests_lib else "not_registered",
                    "registered": bool(requests_lib),
                    "rf_context_available": True,
                }
            except Exception as e:
                return {
                    "status": "error",
                    "registered": False,
                    "rf_context_available": True,
                    "error": str(e),
                }
        except Exception as e:
            return {
                "status": "no_rf_context",
                "registered": False,
                "rf_context_available": False,
                "error": str(e),
            }

    def _check_requests_library_session_state(
        self, session_id: str = None
    ) -> Dict[str, Any]:
        """Check Phase 2: RequestsLibrary session state synchronization."""
        if not session_id or not self.session_manager:
            return {"status": "no_session", "synchronized": False}

        session = self.session_manager.get_session(session_id)
        if not session:
            return {"status": "session_not_found", "synchronized": False}

        # Check if session manager has synchronization capability
        has_sync_method = hasattr(
            self.session_manager, "synchronize_requests_library_state"
        )

        if not has_sync_method:
            return {"status": "no_sync_method", "synchronized": False}

        try:
            # Attempt synchronization check
            sync_result = self.session_manager.synchronize_requests_library_state(
                session
            )
            return {
                "status": "success" if sync_result else "sync_failed",
                "synchronized": sync_result,
                "has_sync_method": True,
            }
        except Exception as e:
            return {
                "status": "error",
                "synchronized": False,
                "has_sync_method": True,
                "error": str(e),
            }

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session."""
        if not self.session_manager:
            return None
        session = self.session_manager.get_session(session_id)
        if not session:
            return None
        return session.get_session_info()

    async def _update_rf_search_order(self, session) -> None:
        """Update Robot Framework's native library search order."""
        try:
            # Get BuiltIn library instance to call Set Library Search Order
            if "BuiltIn" in self.library_manager.libraries:
                builtin_lib = self.library_manager.libraries["BuiltIn"]
                builtin_instance = builtin_lib.instance

                # Get current search order from session
                search_order = session.get_search_order()

                # Filter to only include loaded libraries
                loaded_search_order = [
                    lib for lib in search_order if lib in self.library_manager.libraries
                ]

                if loaded_search_order and hasattr(
                    builtin_instance, "set_library_search_order"
                ):
                    # Use Robot Framework's native Set Library Search Order
                    builtin_instance.set_library_search_order(*loaded_search_order)
                    logger.debug(f"Updated RF search order: {loaded_search_order}")
        except Exception as e:
            logger.debug(f"Could not update RF search order: {e}")

    def create_session(self, session_id: str) -> Dict[str, Any]:
        """Create a new session."""
        if not self.session_manager:
            return {"error": "Session manager not available"}
        session = self.session_manager.create_session(session_id)
        return session.get_session_info()

    def _parse_library_prefix(
        self, keyword_name: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Parse library prefix from keyword name (e.g., 'XML.Get Element Count' -> ('XML', 'Get Element Count'))."""
        if "." not in keyword_name:
            return None, None

        parts = keyword_name.split(".", 1)
        if len(parts) == 2:
            library_name, keyword_part = parts
            # Validate that library_name looks like a valid library name
            if (
                library_name
                and keyword_part
                and library_name.replace("_", "").replace(" ", "").isalnum()
            ):
                return library_name, keyword_part

        return None, None

    async def _ensure_library_loaded(self, library_name: str) -> bool:
        """Ensure a specific library is loaded."""
        if library_name in self.library_manager.libraries:
            return True

        # Try to load the library on demand
        success = self.library_manager.load_library_on_demand(
            library_name, self.keyword_discovery
        )
        if success:
            # Add keywords to cache
            lib_info = self.library_manager.libraries[library_name]
            self.keyword_discovery.add_keywords_to_cache(lib_info)
            logger.info(f"Loaded library '{library_name}' for explicit prefix")
            return True

        logger.warning(f"Could not load library '{library_name}' for prefix")
        return False

    def _find_keyword_with_library_prefix(
        self, keyword_name: str, library_name: str
    ) -> Optional[KeywordInfo]:
        """Find keyword in a specific library only."""
        if library_name not in self.library_manager.libraries:
            logger.debug(f"Library '{library_name}' not loaded for prefix search")
            return None

        # Search only in the specified library
        lib_keywords = self.get_keywords_by_library(library_name)
        for kw in lib_keywords:
            if kw.name.lower() == keyword_name.lower():
                logger.debug(
                    f"Found '{keyword_name}' in '{library_name}' via explicit prefix"
                )
                return kw

        # Try fuzzy matching within the library
        normalized = keyword_name.lower().strip()
        variations = [
            normalized.replace(" ", ""),  # Remove spaces
            normalized.replace("_", " "),  # Replace underscores
            normalized.replace("-", " "),  # Replace hyphens
        ]

        for variation in variations:
            for kw in lib_keywords:
                if kw.name.lower().replace(" ", "") == variation:
                    logger.debug(
                        f"Found '{keyword_name}' in '{library_name}' via fuzzy prefix match"
                    )
                    return kw

        logger.debug(f"Keyword '{keyword_name}' not found in library '{library_name}'")
        return None

    def set_session_search_order(
        self, session_id: str, search_order: List[str]
    ) -> bool:
        """Manually set search order for a session."""
        if not self.session_manager:
            return False
        session = self.session_manager.get_session(session_id)
        if not session:
            return False

        session.search_order = search_order.copy()

        # Update Robot Framework's native search order
        try:
            import asyncio

            asyncio.create_task(self._update_rf_search_order(session))
        except Exception as e:
            logger.debug(f"Could not update search order: {e}")

        return True

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
    def _execute_direct_method_call(
        self,
        keyword_info: KeywordInfo,
        parsed_args: ParsedArguments,
        corrected_args: List[str],
        session_variables: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a keyword by calling its method directly."""
        try:
            # Get library instance
            if keyword_info.library not in self.libraries:
                return {
                    "success": False,
                    "error": f"Library '{keyword_info.library}' is not loaded. Available libraries: {list(self.libraries.keys())}",
                    "keyword_info": {
                        "name": keyword_info.name,
                        "library": keyword_info.library,
                        "args": keyword_info.args,
                        "doc": keyword_info.doc,
                    },
                }

            library = self.libraries[keyword_info.library]

            if library.instance is None:
                return {
                    "success": False,
                    "error": f"Library '{keyword_info.library}' instance is not initialized",
                    "keyword_info": {
                        "name": keyword_info.name,
                        "library": keyword_info.library,
                        "args": keyword_info.args,
                        "doc": keyword_info.doc,
                    },
                }

            method = getattr(library.instance, keyword_info.method_name)

            # Handle different types of method calls
            if keyword_info.is_builtin and hasattr(library.instance, "_context"):
                # BuiltIn library methods might need context
                # BUGFIX: Convert non-string arguments to strings for Robot Framework execution
                # Robot Framework expects string arguments that it converts to appropriate types
                string_args = []
                for arg in corrected_args:
                    if not isinstance(arg, str):
                        string_arg = str(arg)
                        logger.debug(
                            f"Converting non-string argument {arg} (type: {type(arg).__name__}) to string '{string_arg}' for BuiltIn method with context"
                        )
                        string_args.append(string_arg)
                    else:
                        string_args.append(arg)

                result = method(*string_args)
            else:
                # Regular library methods - use centralized type conversion configuration
                from robotmcp.config.library_registry import (
                    get_libraries_requiring_type_conversion,
                )

                type_conversion_libraries = get_libraries_requiring_type_conversion()

                if keyword_info.library in type_conversion_libraries:
                    # Enhanced logging for named arguments debugging
                    has_potential_named_args = any("=" in arg for arg in corrected_args)
                    logger.debug(
                        f"NAMED_ARGS_DEBUG: {keyword_info.name} from {keyword_info.library} - args={corrected_args}, has_potential_named={has_potential_named_args}"
                    )

                    try:
                        # Use Robot Framework's native type conversion system for libraries that need it
                        conversion_result = self._execute_with_rf_type_conversion(
                            method, keyword_info, corrected_args
                        )
                        logger.debug(
                            f"NAMED_ARGS_DEBUG: Type conversion result for {keyword_info.name}: {conversion_result[0] if conversion_result else 'None'}"
                        )
                        if (
                            conversion_result[0] == "executed"
                        ):  # Successfully converted and executed
                            result = conversion_result[1]  # Use the result as-is
                        elif conversion_result[0] == "not_available":
                            # Type conversion not available, use pre-parsed corrected arguments
                            parsed = parsed_args  # Use the corrected arguments from the caller
                            pos_args = parsed.positional
                            kwargs = parsed.named

                            if kwargs:
                                result = method(*pos_args, **kwargs)
                            else:
                                result = method(*pos_args)
                    except Exception as lib_error:
                        logger.debug(
                            f"{keyword_info.library} execution failed: {lib_error}"
                        )
                        # NAMED ARGUMENTS FIX: Use the already parsed and corrected arguments
                        # This ensures named arguments are preserved and argument ordering is fixed
                        logger.info(
                            f"Type conversion failed for {keyword_info.name}, using pre-parsed corrected arguments"
                        )
                        parsed = (
                            parsed_args  # Use the corrected arguments from the caller
                        )
                        pos_args = parsed.positional
                        kwargs = parsed.named

                        logger.debug(
                            f"Fallback execution: positional={pos_args}, named={list(kwargs.keys()) if kwargs else 'none'}"
                        )

                        if kwargs:
                            result = method(*pos_args, **kwargs)
                            logger.debug(
                                f"Successfully executed {keyword_info.name} with fallback named arguments"
                            )
                        else:
                            result = method(*pos_args)
                            logger.debug(
                                f"Successfully executed {keyword_info.name} with fallback positional arguments"
                            )
                elif keyword_info.name == "Create List":
                    # Collections.Create List takes variable arguments
                    result = method(*parsed_args.positional)
                elif keyword_info.name == "Set Variable":
                    # Set Variable takes one argument
                    value = (
                        parsed_args.positional[0] if parsed_args.positional else None
                    )
                    result = method(value)
                else:
                    # OBJECT ARGUMENTS FIX: Libraries like XML, RequestsLibrary, Collections need original object arguments, not string conversions
                    libraries_needing_objects = [
                        "XML",
                        "RequestsLibrary",
                        "Collections",
                    ]

                    if keyword_info.library in libraries_needing_objects:
                        # Use original arguments to preserve object types (e.g., XML Elements)
                        logger.debug(
                            f"Object-preserving execution for {keyword_info.name} in {keyword_info.library}: using original args"
                        )
                        if parsed_args.named:
                            result = method(*corrected_args, **parsed_args.named)
                            logger.debug(
                                f"Successfully executed {keyword_info.name} with original args + named arguments"
                            )
                        else:
                            result = method(*corrected_args)
                            logger.debug(
                                f"Successfully executed {keyword_info.name} with original args only"
                            )
                    else:
                        # NAMED ARGUMENTS FIX: For other libraries, check for named arguments
                        # This was the critical bug - line was ignoring named arguments completely
                        logger.debug(
                            f"Standard execution path for {keyword_info.name}: positional={parsed_args.positional}, named={list(parsed_args.named.keys()) if parsed_args.named else 'none'}"
                        )

                        if parsed_args.named:
                            result = method(
                                *parsed_args.positional, **parsed_args.named
                            )
                            logger.debug(
                                f"Successfully executed {keyword_info.name} with named arguments: {list(parsed_args.named.keys())}"
                            )
                        else:
                            result = method(*parsed_args.positional)
                            logger.debug(
                                f"Successfully executed {keyword_info.name} with positional arguments only"
                            )

            return {
                "success": True,
                "output": str(result)
                if result is not None
                else f"Executed {keyword_info.name}",
                "result": result,
                "keyword_info": {
                    "name": keyword_info.name,
                    "library": keyword_info.library,
                    "doc": keyword_info.doc,
                },
            }

        except Exception as e:
            import traceback

            logger.debug(
                f"Full traceback for {keyword_info.library}.{keyword_info.name}: {traceback.format_exc()}"
            )
            return {
                "success": False,
                "error": f"Error executing {keyword_info.library}.{keyword_info.name}: {str(e)}",
                "keyword_info": {
                    "name": keyword_info.name,
                    "library": keyword_info.library,
                    "args": keyword_info.args,
                    "doc": keyword_info.doc,
                },
            }

    async def execute_keyword(
        self,
        keyword_name: str,
        args: List[str],
        session_variables: Dict[str, Any] = None,
        active_library: str = None,
        session_id: str = None,
        library_prefix: str = None,
    ) -> Dict[str, Any]:
        """Execute a keyword dynamically with session-based library management and optional library prefix support."""
        # Parse library prefix from keyword name if present (e.g., "XML.Get Element Count")
        parsed_library, parsed_keyword = self._parse_library_prefix(keyword_name)

        # Determine effective library prefix (parameter overrides parsed)
        effective_library_prefix = library_prefix or parsed_library
        effective_keyword_name = parsed_keyword or keyword_name
        # Handle session-based library loading
        if session_id:
            await self._ensure_session_libraries(session_id, effective_keyword_name)

        # Handle library prefix loading if specified
        if effective_library_prefix:
            await self._ensure_library_loaded(effective_library_prefix)

        # Find keyword with library prefix, session search order, or active library filtering
        if effective_library_prefix:
            keyword_info = self._find_keyword_with_library_prefix(
                effective_keyword_name, effective_library_prefix
            )
        else:
            keyword_info = self._find_keyword_with_session(
                effective_keyword_name, active_library, session_id
            )

        # CRITICAL FIX: Check if session libraries need to be reloaded AFTER keyword resolution
        # This is needed because session auto-configuration happens during _find_keyword_with_session
        if session_id and self.session_manager:
            session = self.session_manager.get_session(session_id)
            if session and not session.libraries_loaded:
                logger.info(
                    f"LIBRARY_RELOAD_DEBUG: Session {session_id} libraries need post-resolution reload"
                )
                await self._ensure_session_libraries(session_id, effective_keyword_name)
                # Re-search for keyword after reloading libraries
                if effective_library_prefix:
                    keyword_info = self._find_keyword_with_library_prefix(
                        effective_keyword_name, effective_library_prefix
                    )
                else:
                    keyword_info = self._find_keyword_with_session(
                        effective_keyword_name, active_library, session_id
                    )
                logger.info(
                    f"LIBRARY_RELOAD_DEBUG: Post-reload keyword search result: {keyword_info is not None}"
                )

        if not keyword_info:
            # IMPROVEMENT: Provide more helpful error messages
            if effective_library_prefix:
                error_msg = f"Keyword '{effective_keyword_name}' not found in library '{effective_library_prefix}'"
                error_msg += f". Check if {effective_library_prefix} is properly loaded or use execute_step which triggers library loading automatically."
            else:
                # Check if this might be a library loading issue
                session_libraries_info = self._get_session_library_loading_info(
                    session_id, active_library
                )

                if session_libraries_info.get("has_unloaded_session_libraries"):
                    error_msg = f"Keyword '{effective_keyword_name}' not found - library loading issue detected"
                    error_msg += f". Session has imported libraries {session_libraries_info.get('session_libraries', [])} "
                    error_msg += f"but only {session_libraries_info.get('loaded_libraries', [])} are loaded in LibraryManager. "
                    error_msg += "Try: 1) Use initialize_context to load libraries, or 2) Libraries should auto-load on import."
                else:
                    error_msg = f"Keyword '{effective_keyword_name}' not found"
                    if active_library:
                        error_msg += f" in active library '{active_library}' or built-in libraries"
                    else:
                        error_msg += " in any loaded library"

            # Phase 4: Enhanced error handling and diagnostics
            try:
                diagnostic_info = self._get_diagnostic_info(
                    effective_keyword_name, session_id, active_library
                )
                print(
                    f"PHASE4_DEBUG: Generated diagnostics for '{effective_keyword_name}': {list(diagnostic_info.keys())}"
                )
            except Exception as diag_error:
                print(
                    f"PHASE4_DEBUG: Diagnostics generation failed for '{effective_keyword_name}': {diag_error}"
                )
                diagnostic_info = {"diagnostic_error": str(diag_error)}

            result = {
                "success": False,
                "error": error_msg,
                "suggestions": self.get_keyword_suggestions(effective_keyword_name, 3),
                "library_prefix": effective_library_prefix,
                "active_library_filter": active_library,
                "session_id": session_id,
                "diagnostics": diagnostic_info,  # Phase 4: Enhanced diagnostics
                "source": "orchestrator",  # Phase 4: Debug - identify source
            }
            print(
                f"PHASE4_DEBUG: Orchestrator returning error response with diagnostics: {list(result.keys())}"
            )
            return result

        # Record keyword usage for session management and update search order
        if session_id:
            session = self.session_manager.get_session(session_id)
            if session:
                # Record the base keyword name (without library prefix) for session detection
                configuration_changed = session.record_keyword_usage(
                    effective_keyword_name
                )

                # CRITICAL FIX: If session auto-configuration changed, immediately reload libraries
                if configuration_changed:
                    logger.info(
                        f"LIBRARY_RELOAD_DEBUG: Session {session_id} auto-configuration triggered, reloading libraries immediately"
                    )
                    await self._ensure_session_libraries(
                        session_id, effective_keyword_name
                    )
                    # Re-find keyword with updated libraries
                    if effective_library_prefix:
                        keyword_info = self._find_keyword_with_library_prefix(
                            effective_keyword_name, effective_library_prefix
                        )
                    else:
                        keyword_info = self._find_keyword_with_session(
                            effective_keyword_name, active_library, session_id
                        )
                    logger.info(
                        f"LIBRARY_RELOAD_DEBUG: Post-auto-config keyword search result: {keyword_info is not None}"
                    )

                # Update Robot Framework's native search order
                await self._update_rf_search_order(session)

        try:
            # Log which library the keyword was found in for debugging
            if active_library and keyword_info.library != active_library:
                logger.debug(
                    f"Using built-in keyword '{keyword_info.name}' from {keyword_info.library} (active library: {active_library})"
                )
            else:
                logger.debug(
                    f"Executing keyword '{keyword_info.name}' from {keyword_info.library}"
                )

            # Parse arguments using LibDoc information for accuracy
            # BUGFIX: Convert all arguments to strings before parsing since Robot Framework expects string inputs
            logger.debug(
                f"ORCHESTRATOR: Before string conversion, args={args}, types={[type(arg).__name__ for arg in args]}"
            )
            string_args = []
            for arg in args:
                if not isinstance(arg, str):
                    # Special handling for tuples from variable resolution (like ('json', {'test': 'value'}))
                    if isinstance(arg, tuple) and len(arg) == 2:
                        # This is a named parameter tuple - convert back to name=value format
                        param_name, param_value = arg
                        if isinstance(param_name, str) and param_name.isidentifier():
                            # For named parameters, we need to create a special format that preserves the object
                            # Use ObjectPreservingArgument format that RF converter can understand
                            string_args.append(
                                arg
                            )  # Keep the tuple as-is for RF converter
                            logger.debug(
                                f"Preserving named parameter tuple {arg} for RF converter"
                            )
                            continue

                    # Default conversion for other non-string types
                    string_arg = str(arg)
                    logger.debug(
                        f"Converting non-string argument {arg} (type: {type(arg).__name__}) to string '{string_arg}' for RF argument parsing"
                    )
                    string_args.append(string_arg)
                else:
                    string_args.append(arg)

            parsed_args = self._parse_arguments_for_keyword(
                effective_keyword_name,
                string_args,
                keyword_info.library,
                session_variables,
            )

            # Enhanced debug logging for named arguments
            logger.debug(
                f"NAMED_ARGS_DEBUG: Parsed arguments for {effective_keyword_name}: positional={parsed_args.positional}, named={list(parsed_args.named.keys()) if parsed_args.named else 'none'}"
            )

            # Reconstruct corrected arguments from parsed_args for execution
            corrected_args = parsed_args.positional + [
                f"{k}={v}" for k, v in parsed_args.named.items()
            ]
            logger.debug(
                f"Using corrected arguments for execution: {args} → {corrected_args}"
            )

            # Execute the keyword with corrected arguments
            result = self._execute_direct_method_call(
                keyword_info, parsed_args, corrected_args, session_variables or {}
            )
            result["session_id"] = session_id
            return result

        except Exception as e:
            return {
                "success": False,
                "error": f"Error executing {effective_keyword_name}: {str(e)}",
                "library_prefix": effective_library_prefix,
                "keyword_info": {
                    "name": keyword_info.name,
                    "library": keyword_info.library,
                    "doc": keyword_info.doc,
                },
                "active_library_filter": active_library,
                "session_id": session_id,
            }


# Global instance management
_keyword_discovery = None


def get_keyword_discovery() -> DynamicKeywordDiscovery:
    """Get the global keyword discovery instance."""
    global _keyword_discovery
    if _keyword_discovery is None:
        _keyword_discovery = DynamicKeywordDiscovery()
    return _keyword_discovery

"""Centralized validation gateway for library loading operations."""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class LibraryLoadingError(Exception):
    """Exception raised when library loading validation fails."""

    def __init__(self, library_name: str, reason: str, suggestions: List[str] = None):
        self.library_name = library_name
        self.reason = reason
        self.suggestions = suggestions or []
        super().__init__(f"Cannot load '{library_name}': {reason}")


class LibraryLoadingValidator:
    """Centralized validation gateway for all library loading.

    This class consolidates validation logic that was previously scattered
    across LibraryManager, SessionManager, and ExecutionCoordinator.

    Usage:
        validator = LibraryLoadingValidator(library_manager, session_manager)
        can_load, reason = validator.can_load("Browser", session_id)
        if can_load:
            validator.load_with_validation("Browser", session_id)
    """

    # Exclusion groups - only one from each group can be loaded
    EXCLUSION_GROUPS = {
        'web_automation': ['Browser', 'SeleniumLibrary'],
        'mobile_automation': ['AppiumLibrary'],
    }

    # Library preferences within groups (first = preferred)
    GROUP_PREFERENCES = {
        'web_automation': ['Browser', 'SeleniumLibrary'],  # Prefer Browser
    }

    def __init__(self, library_manager=None, session_manager=None):
        """Initialize the validator.

        Args:
            library_manager: Optional LibraryManager instance
            session_manager: Optional SessionManager instance
        """
        self._library_manager = library_manager
        self._session_manager = session_manager

    @property
    def library_manager(self):
        """Get library manager, lazily importing if needed."""
        if self._library_manager is None:
            from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery
            self._library_manager = get_keyword_discovery().library_manager
        return self._library_manager

    @property
    def session_manager(self):
        """Get session manager, lazily importing if needed."""
        if self._session_manager is None:
            from robotmcp.components.execution.execution_coordinator import get_execution_coordinator
            self._session_manager = get_execution_coordinator().session_manager
        return self._session_manager

    def can_load(self, library_name: str, session_id: str = None) -> Tuple[bool, str]:
        """Check if a library can be loaded.

        Args:
            library_name: Name of the library to check
            session_id: Optional session ID for session-specific validation

        Returns:
            Tuple of (can_load, reason)
        """
        # Check 1: Library availability
        if not self._is_library_available(library_name):
            return False, f"Library '{library_name}' is not installed"

        # Check 2: Exclusion group conflicts
        conflict_check = self._check_exclusion_conflicts(library_name)
        if not conflict_check[0]:
            return conflict_check

        # Check 3: Session type compatibility (if session provided)
        if session_id:
            session_check = self._check_session_compatibility(library_name, session_id)
            if not session_check[0]:
                return session_check

        return True, "OK"

    def _is_library_available(self, library_name: str) -> bool:
        """Check if library is installed and importable."""
        return self.library_manager.is_library_importable(library_name)

    def _check_exclusion_conflicts(self, library_name: str) -> Tuple[bool, str]:
        """Check if library conflicts with already loaded libraries."""
        for group_name, group_libs in self.EXCLUSION_GROUPS.items():
            if library_name in group_libs:
                # Check if another library from this group is already loaded
                loaded = [lib for lib in group_libs
                         if lib in self.library_manager.libraries and lib != library_name]
                if loaded:
                    return False, (
                        f"Conflicts with already loaded '{loaded[0]}' "
                        f"(exclusion group: {group_name})"
                    )
        return True, "OK"

    def _check_session_compatibility(self, library_name: str, session_id: str) -> Tuple[bool, str]:
        """Check if library is compatible with session type."""
        try:
            session = self.session_manager.get_session(session_id)
            if session is None:
                return True, "OK"  # No session = allow any

            if hasattr(session, 'validate_library_for_session'):
                if not session.validate_library_for_session(library_name):
                    return False, (
                        f"Library not valid for session type '{session.session_type.value}'"
                    )
        except Exception as e:
            logger.debug(f"Session compatibility check failed: {e}")
            # Don't block on session check failures

        return True, "OK"

    def load_with_validation(
        self,
        library_name: str,
        session_id: str = None,
        keyword_extractor=None
    ) -> bool:
        """Load a library with full validation.

        Args:
            library_name: Name of the library to load
            session_id: Optional session ID
            keyword_extractor: Optional KeywordDiscovery instance

        Returns:
            True if library was loaded successfully

        Raises:
            LibraryLoadingError: If validation fails
        """
        can_load, reason = self.can_load(library_name, session_id)

        if not can_load:
            suggestions = self._get_suggestions(library_name, reason)
            raise LibraryLoadingError(library_name, reason, suggestions)

        # Get keyword extractor if not provided
        if keyword_extractor is None:
            from robotmcp.core.dynamic_keyword_orchestrator import get_keyword_discovery
            keyword_extractor = get_keyword_discovery().keyword_discovery

        return self.library_manager.load_library_on_demand(library_name, keyword_extractor)

    def _get_suggestions(self, library_name: str, reason: str) -> List[str]:
        """Get suggestions for resolving loading issues."""
        suggestions = []

        if "not installed" in reason.lower():
            from robotmcp.config.library_registry import get_library_install_hint
            hint = get_library_install_hint(library_name)
            if hint:
                suggestions.append(hint)

        if "conflicts with" in reason.lower():
            # Suggest using the already loaded library
            for group_name, group_libs in self.EXCLUSION_GROUPS.items():
                if library_name in group_libs:
                    loaded = [lib for lib in group_libs if lib in self.library_manager.libraries]
                    if loaded:
                        suggestions.append(f"Use '{loaded[0]}' instead (already loaded)")
                    break

        if "not valid for session type" in reason.lower():
            suggestions.append("Create a new session with appropriate type")
            suggestions.append("Use force=True to bypass session validation (not recommended)")

        return suggestions

    def get_preferred_library(self, group_name: str) -> Optional[str]:
        """Get the preferred library from an exclusion group.

        Args:
            group_name: Name of the exclusion group

        Returns:
            Preferred library name, or None if group not found
        """
        preferences = self.GROUP_PREFERENCES.get(group_name)
        if preferences:
            # Return first available preference
            for lib in preferences:
                if self._is_library_available(lib):
                    return lib
        return None

    def resolve_conflicts(self, library_names: List[str]) -> List[str]:
        """Resolve conflicts in a list of library names.

        Args:
            library_names: List of library names to check

        Returns:
            Filtered list with conflicts resolved (keeps first from each group)
        """
        result = []
        used_groups: Set[str] = set()

        for lib in library_names:
            group = self._get_exclusion_group(lib)
            if group:
                if group in used_groups:
                    logger.debug(f"Filtering '{lib}': {group} group already has library")
                    continue
                used_groups.add(group)
            result.append(lib)

        return result

    def _get_exclusion_group(self, library_name: str) -> Optional[str]:
        """Get the exclusion group a library belongs to."""
        for group_name, group_libs in self.EXCLUSION_GROUPS.items():
            if library_name in group_libs:
                return group_name
        return None

    def get_validation_report(self, library_names: List[str], session_id: str = None) -> Dict[str, Any]:
        """Get a validation report for multiple libraries.

        Args:
            library_names: List of library names to validate
            session_id: Optional session ID

        Returns:
            Validation report dictionary
        """
        report = {
            'valid': [],
            'invalid': [],
            'conflicts': [],
            'suggestions': [],
        }

        seen_groups: Set[str] = set()

        for lib in library_names:
            can_load, reason = self.can_load(lib, session_id)

            if can_load:
                group = self._get_exclusion_group(lib)
                if group and group in seen_groups:
                    report['conflicts'].append({
                        'library': lib,
                        'group': group,
                        'reason': f"Another library from '{group}' already included"
                    })
                else:
                    report['valid'].append(lib)
                    if group:
                        seen_groups.add(group)
            else:
                report['invalid'].append({
                    'library': lib,
                    'reason': reason,
                    'suggestions': self._get_suggestions(lib, reason)
                })

        return report


# Module-level singleton
_validator: Optional[LibraryLoadingValidator] = None


def get_library_loading_validator() -> LibraryLoadingValidator:
    """Get the global LibraryLoadingValidator instance."""
    global _validator
    if _validator is None:
        _validator = LibraryLoadingValidator()
    return _validator

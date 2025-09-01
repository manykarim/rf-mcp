"""Session-based library activation and search order management."""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class SessionType(Enum):
    """Types of test automation sessions."""

    XML_PROCESSING = "xml_processing"
    WEB_AUTOMATION = "web_automation"
    API_TESTING = "api_testing"
    DATA_PROCESSING = "data_processing"
    SYSTEM_TESTING = "system_testing"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class SessionProfile:
    """Configuration for a session type."""

    session_type: SessionType
    core_libraries: List[str] = field(default_factory=list)
    optional_libraries: List[str] = field(default_factory=list)
    search_order: List[str] = field(default_factory=list)
    keywords_patterns: List[str] = field(default_factory=list)
    description: str = ""


class SessionManager:
    """Manages session-based library activation and search order."""

    def __init__(self):
        self.sessions: Dict[str, "ActiveSession"] = {}
        self.session_profiles = self._initialize_session_profiles()

    def _initialize_session_profiles(self) -> Dict[SessionType, SessionProfile]:
        """Initialize predefined session profiles."""
        return {
            SessionType.XML_PROCESSING: SessionProfile(
                session_type=SessionType.XML_PROCESSING,
                core_libraries=[
                    "BuiltIn",
                    "XML",
                    "Collections",
                    "String",
                    "OperatingSystem",
                ],
                optional_libraries=["DateTime", "Process"],
                search_order=[
                    "XML",
                    "BuiltIn",
                    "Collections",
                    "String",
                    "OperatingSystem",
                ],
                keywords_patterns=[
                    r"\b(parse|xml|xpath|element|attribute)\b",
                    r"\b(get element|set element|xml)\b",
                ],
                description="XML file processing and manipulation",
            ),
            SessionType.WEB_AUTOMATION: SessionProfile(
                session_type=SessionType.WEB_AUTOMATION,
                core_libraries=["BuiltIn", "Browser", "Collections", "String"],
                optional_libraries=["XML", "DateTime", "Screenshot"],
                search_order=["Browser", "BuiltIn", "Collections", "String", "XML"],
                keywords_patterns=[
                    r"\b(click|fill|navigate|browser|page|element|locator)\b",
                    r"\b(new page|go to|wait for|screenshot)\b",
                    r"\b(get text|get attribute|should contain)\b",
                ],
                description="Web browser automation testing",
            ),
            SessionType.API_TESTING: SessionProfile(
                session_type=SessionType.API_TESTING,
                core_libraries=["BuiltIn", "RequestsLibrary", "Collections", "String"],
                optional_libraries=["XML", "DateTime"],
                search_order=[
                    "RequestsLibrary",
                    "BuiltIn",
                    "Collections",
                    "String",
                    "XML",
                ],
                keywords_patterns=[
                    r"\b(get request|post|put|delete|api|http)\b",
                    r"\b(create session|request|response|status)\b",
                    r"\b(json|rest|endpoint)\b",
                ],
                description="API and HTTP testing",
            ),
            SessionType.DATA_PROCESSING: SessionProfile(
                session_type=SessionType.DATA_PROCESSING,
                core_libraries=["BuiltIn", "Collections", "String", "DateTime", "XML"],
                optional_libraries=["OperatingSystem", "Process"],
                search_order=["Collections", "String", "DateTime", "XML", "BuiltIn"],
                keywords_patterns=[
                    r"\b(create list|append|remove|sort|filter)\b",
                    r"\b(convert to|get from|set to)\b",
                    r"\b(data|process|transform)\b",
                ],
                description="Data processing and manipulation",
            ),
            SessionType.SYSTEM_TESTING: SessionProfile(
                session_type=SessionType.SYSTEM_TESTING,
                core_libraries=["BuiltIn", "OperatingSystem", "Process", "Collections"],
                optional_libraries=["String", "DateTime", "SSHLibrary"],
                search_order=["OperatingSystem", "Process", "BuiltIn", "Collections"],
                keywords_patterns=[
                    r"\b(run|execute|command|file|directory)\b",
                    r"\b(create file|remove|copy|move)\b",
                    r"\b(start process|terminate|ssh)\b",
                ],
                description="System and process testing",
            ),
        }

    def create_session(self, session_id: str) -> "ActiveSession":
        """Create a new session with unknown type initially."""
        session = ActiveSession(session_id, self)
        self.sessions[session_id] = session
        logger.info(f"Created session {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional["ActiveSession"]:
        """Get an existing session."""
        return self.sessions.get(session_id)

    def remove_session(self, session_id: str) -> None:
        """Remove a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Removed session {session_id}")

    def detect_session_type(self, keywords_used: List[str]) -> SessionType:
        """Detect session type based on keywords used."""
        keyword_text = " ".join(keywords_used).lower()

        # Score each session type based on pattern matches
        scores = {}
        for session_type, profile in self.session_profiles.items():
            score = 0
            for pattern in profile.keywords_patterns:
                matches = len(re.findall(pattern, keyword_text, re.IGNORECASE))
                score += matches
            scores[session_type] = score

        # Find the session type with highest score
        if not scores or max(scores.values()) == 0:
            return SessionType.UNKNOWN

        best_type = max(scores, key=scores.get)

        # If multiple session types have similar high scores, it's mixed
        max_score = max(scores.values())
        high_scores = [k for k, v in scores.items() if v >= max_score * 0.6 and v > 0]
        if (
            len(high_scores) > 1 and max_score > 1
        ):  # Need at least some matches for mixed
            return SessionType.MIXED

        return best_type

    def get_session_profile(
        self, session_type: SessionType
    ) -> Optional[SessionProfile]:
        """Get session profile for a session type."""
        return self.session_profiles.get(session_type)


@dataclass
class ActiveSession:
    """Represents an active automation session."""

    session_id: str
    session_manager: SessionManager
    session_type: SessionType = SessionType.UNKNOWN
    loaded_libraries: Set[str] = field(default_factory=set)
    keywords_used: List[str] = field(default_factory=list)
    search_order: List[str] = field(default_factory=list)
    request_count: int = 0

    def record_keyword_usage(self, keyword_name: str) -> bool:
        """Record that a keyword was used. Returns True if session configuration changed."""
        self.keywords_used.append(keyword_name.lower())
        self.request_count += 1

        # Re-evaluate session type every few requests
        if self.request_count <= 5 or self.request_count % 10 == 0:
            old_type = self.session_type
            self._update_session_type()
            # Return True if session type changed
            return old_type != self.session_type
        return False

    def _update_session_type(self) -> None:
        """Update session type based on keyword usage."""
        old_type = self.session_type
        new_type = self.session_manager.detect_session_type(self.keywords_used)

        if new_type != old_type:
            logger.info(
                f"Session {self.session_id} type changed from {old_type.value} to {new_type.value}"
            )
            self.session_type = new_type
            self._update_libraries_and_search_order()

    def _update_libraries_and_search_order(self) -> None:
        """Update loaded libraries and search order based on session type."""
        profile = self.session_manager.get_session_profile(self.session_type)
        if not profile:
            return

        # Update search order
        old_search_order = self.search_order.copy()
        self.search_order = profile.search_order.copy()

        # Add any currently loaded libraries that aren't in the new search order
        for lib in self.loaded_libraries:
            if lib not in self.search_order:
                self.search_order.append(lib)

        if self.search_order != old_search_order:
            logger.info(
                f"Session {self.session_id} search order updated: {self.search_order}"
            )

    def get_libraries_to_load(self) -> List[str]:
        """Get list of libraries that should be loaded for this session."""
        if self.session_type == SessionType.UNKNOWN:
            # For unknown sessions, load minimal core set
            return ["BuiltIn", "Collections", "String"]

        profile = self.session_manager.get_session_profile(self.session_type)
        if not profile:
            return ["BuiltIn", "Collections", "String"]

        # Return core libraries for this session type
        return profile.core_libraries

    def get_optional_libraries(self) -> List[str]:
        """Get list of optional libraries for this session."""
        if self.session_type == SessionType.UNKNOWN:
            return []

        profile = self.session_manager.get_session_profile(self.session_type)
        if not profile:
            return []

        return profile.optional_libraries

    def should_load_library(self, library_name: str) -> bool:
        """Determine if a library should be loaded for this session."""
        if library_name in self.loaded_libraries:
            return False  # Already loaded

        required_libs = self.get_libraries_to_load()
        optional_libs = self.get_optional_libraries()

        return library_name in required_libs or library_name in optional_libs

    def mark_library_loaded(self, library_name: str) -> None:
        """Mark a library as loaded in this session."""
        self.loaded_libraries.add(library_name)

        # Update search order if needed
        if library_name not in self.search_order:
            # Add to search order based on session type priority
            profile = self.session_manager.get_session_profile(self.session_type)
            if profile and library_name in profile.search_order:
                # Insert in correct position based on profile
                insert_pos = len(self.search_order)
                for i, lib in enumerate(profile.search_order):
                    if lib == library_name:
                        insert_pos = min(i, len(self.search_order))
                        break
                self.search_order.insert(insert_pos, library_name)
            else:
                # Add to end
                self.search_order.append(library_name)

    def get_search_order(self) -> List[str]:
        """Get current library search order."""
        return self.search_order.copy()

    def get_session_info(self) -> Dict[str, Any]:
        """Get session information for debugging."""
        return {
            "session_id": self.session_id,
            "session_type": self.session_type.value,
            "loaded_libraries": list(self.loaded_libraries),
            "search_order": self.search_order,
            "keywords_used_count": len(self.keywords_used),
            "request_count": self.request_count,
            "recent_keywords": self.keywords_used[-5:] if self.keywords_used else [],
        }


# Global session manager instance
_session_manager = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager

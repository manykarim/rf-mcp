"""Session-related data models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import re
import logging

from .execution_models import ExecutionStep
from .browser_models import BrowserState

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


@dataclass
class ExecutionSession:
    """Manages execution state for a test session with intelligent library management."""
    session_id: str
    suite: Optional[Any] = None
    steps: List[ExecutionStep] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    imported_libraries: List[str] = field(default_factory=list)
    current_browser: Optional[str] = None
    browser_state: BrowserState = field(default_factory=BrowserState)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    # New fields from ActiveSession
    session_type: SessionType = SessionType.UNKNOWN
    loaded_libraries: Set[str] = field(default_factory=set)
    keywords_used: List[str] = field(default_factory=list)
    search_order: List[str] = field(default_factory=list)
    request_count: int = 0
    explicit_library_preference: Optional[str] = None  # For scenario-based explicit preferences
    scenario_text: Optional[str] = None  # Store original scenario for re-analysis
    auto_configured: bool = False  # Track if session was auto-configured
    
    def add_step(self, step: ExecutionStep) -> None:
        """Add a successful step to the session."""
        if step.is_successful:
            self.steps.append(step)
            self.last_activity = datetime.now()
    
    def update_activity(self) -> None:
        """Update the last activity timestamp."""
        self.last_activity = datetime.now()
    
    def is_browser_session(self) -> bool:
        """Check if this session has browser automation capabilities."""
        return (self.browser_state.has_browser_session() or 
                'Browser' in self.imported_libraries or 
                'SeleniumLibrary' in self.imported_libraries)
    
    def get_active_library(self) -> Optional[str]:
        """Get the currently active browser automation library."""
        return self.browser_state.active_library
    
    def set_variable(self, name: str, value: Any) -> None:
        """Set a session variable."""
        self.variables[name] = value
        self.update_activity()
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a session variable."""
        return self.variables.get(name, default)
    
    def import_library(self, library_name: str, force: bool = False) -> None:
        """
        Mark a library as imported in this session.
        
        Enforces exclusion rules - Browser Library and SeleniumLibrary cannot
        coexist in the same session, unless force=True is used to switch libraries.
        
        Args:
            library_name: Name of the library to import
            force: If True, allows switching between mutually exclusive libraries
            
        Raises:
            ValueError: If trying to import a conflicting library without force=True
        """
        if library_name not in self.imported_libraries:
            # Enforce web automation library exclusion
            web_automation_libs = ['Browser', 'SeleniumLibrary']
            
            if library_name in web_automation_libs:
                # Check if another web automation library is already imported
                existing_web_libs = [lib for lib in self.imported_libraries if lib in web_automation_libs]
                
                if existing_web_libs and library_name not in existing_web_libs:
                    if not force:
                        existing_lib = existing_web_libs[0]
                        raise ValueError(
                            f"Cannot import '{library_name}' - session already has '{existing_lib}'. "
                            f"Browser Library and SeleniumLibrary are mutually exclusive per session."
                        )
                    else:
                        # Force switch: remove existing web automation libraries
                        for existing_lib in existing_web_libs:
                            if existing_lib in self.imported_libraries:
                                self.imported_libraries.remove(existing_lib)
            
            self.imported_libraries.append(library_name)
            self.update_activity()
    
    def get_web_automation_library(self) -> Optional[str]:
        """Get the web automation library imported in this session."""
        web_automation_libs = ['Browser', 'SeleniumLibrary']
        for lib in self.imported_libraries:
            if lib in web_automation_libs:
                return lib
        return None
    
    def get_successful_steps(self) -> List[ExecutionStep]:
        """Get all successfully executed steps."""
        return [step for step in self.steps if step.is_successful]
    
    def get_failed_steps(self) -> List[ExecutionStep]:
        """Get all failed steps (Note: failed steps are not added to self.steps)."""
        # This would need to be tracked separately if needed
        return []
    
    @property
    def step_count(self) -> int:
        """Get the total number of successful steps."""
        return len(self.steps)
    
    @property
    def duration(self) -> float:
        """Calculate session duration in seconds."""
        return (self.last_activity - self.created_at).total_seconds()
    
    def cleanup(self) -> None:
        """Clean up session resources."""
        self.browser_state.reset()
        self.steps.clear()
        # Keep variables and imported_libraries for potential reuse
    
    # ===============================
    # Session Profiles (from SessionManager)
    # ===============================
    
    @classmethod
    def _get_session_profiles(cls) -> Dict[SessionType, SessionProfile]:
        """Get predefined session profiles."""
        return {
            SessionType.XML_PROCESSING: SessionProfile(
                session_type=SessionType.XML_PROCESSING,
                core_libraries=["BuiltIn", "XML", "Collections", "String", "OperatingSystem"],
                optional_libraries=["DateTime", "Process"],
                search_order=["XML", "BuiltIn", "Collections", "String", "OperatingSystem"],
                keywords_patterns=[
                    r'\b(parse|xml|xpath|element|attribute)\b',
                    r'\b(get element|set element|xml)\b'
                ],
                description="XML file processing and manipulation"
            ),
            
            SessionType.WEB_AUTOMATION: SessionProfile(
                session_type=SessionType.WEB_AUTOMATION,
                core_libraries=["BuiltIn", "Browser", "Collections", "String"],
                optional_libraries=["XML", "DateTime", "Screenshot"],
                search_order=["Browser", "BuiltIn", "Collections", "String", "XML"],
                keywords_patterns=[
                    r'\b(click|fill|navigate|browser|page|element|locator)\b',
                    r'\b(new page|go to|wait for|screenshot)\b',
                    r'\b(get text|get attribute|should contain)\b'
                ],
                description="Web browser automation testing"
            ),
            
            SessionType.API_TESTING: SessionProfile(
                session_type=SessionType.API_TESTING,
                core_libraries=["BuiltIn", "RequestsLibrary", "Collections", "String"],
                optional_libraries=["XML", "DateTime"],
                search_order=["RequestsLibrary", "BuiltIn", "Collections", "String", "XML"],
                keywords_patterns=[
                    r'\b(get request|post|put|delete|api|http)\b',
                    r'\b(create session|request|response|status)\b',
                    r'\b(json|rest|endpoint)\b'
                ],
                description="API and HTTP testing"
            ),
            
            SessionType.DATA_PROCESSING: SessionProfile(
                session_type=SessionType.DATA_PROCESSING,
                core_libraries=["BuiltIn", "Collections", "String", "DateTime", "XML"],
                optional_libraries=["OperatingSystem", "Process"],
                search_order=["Collections", "String", "DateTime", "XML", "BuiltIn"],
                keywords_patterns=[
                    r'\b(create list|append|remove|sort|filter)\b',
                    r'\b(convert to|get from|set to)\b',
                    r'\b(data|process|transform)\b'
                ],
                description="Data processing and manipulation"
            ),
            
            SessionType.SYSTEM_TESTING: SessionProfile(
                session_type=SessionType.SYSTEM_TESTING,
                core_libraries=["BuiltIn", "OperatingSystem", "Process", "Collections"],
                optional_libraries=["String", "DateTime", "SSHLibrary"],
                search_order=["OperatingSystem", "Process", "BuiltIn", "Collections"],
                keywords_patterns=[
                    r'\b(run process|start process|terminate|kill)\b',
                    r'\b(create file|remove file|directory|path)\b',
                    r'\b(environment|variable|system)\b'
                ],
                description="System and process testing"
            )
        }
    
    # ===============================
    # Scenario-Based Library Detection
    # ===============================
    
    def detect_explicit_library_preference(self, scenario_text: str) -> Optional[str]:
        """Detect explicit library preference from scenario text."""
        if not scenario_text:
            return None
        
        text_lower = scenario_text.lower()
        
        # Selenium patterns (highest priority for explicit mentions)
        selenium_patterns = [
            r'\b(use|using|with)\s+(selenium|seleniumlibrary|selenium\s*library)\b',
            r'\bselenium\b(?!.*browser)',  # Selenium mentioned but not "selenium browser"
            r'\bseleniumlibrary\b',
        ]
        
        # Browser Library patterns
        browser_patterns = [
            r'\b(use|using|with)\s+(browser|browserlibrary|browser\s*library|playwright)\b',
            r'\bbrowser\s*library\b',
            r'\bplaywright\b',
        ]
        
        # Check for explicit Selenium preference first
        for pattern in selenium_patterns:
            if re.search(pattern, text_lower):
                logger.info(f"Detected explicit SeleniumLibrary preference in scenario: {pattern}")
                return "SeleniumLibrary"
        
        # Check for explicit Browser Library preference
        for pattern in browser_patterns:
            if re.search(pattern, text_lower):
                logger.info(f"Detected explicit Browser Library preference in scenario: {pattern}")
                return "Browser"
        
        # Check for other library preferences
        if re.search(r'\b(xml|xpath)\b', text_lower):
            return "XML"
        if re.search(r'\b(api|http|rest|request)\b', text_lower):
            return "RequestsLibrary"
        
        return None
    
    def detect_session_type_from_scenario(self, scenario_text: str) -> SessionType:
        """Detect session type from scenario text."""
        if not scenario_text:
            return SessionType.UNKNOWN
        
        text_lower = scenario_text.lower()
        profiles = self._get_session_profiles()
        scores = {session_type: 0 for session_type in profiles.keys()}
        
        # Score each session type based on keyword patterns
        for session_type, profile in profiles.items():
            for pattern in profile.keywords_patterns:
                matches = len(re.findall(pattern, text_lower))
                scores[session_type] += matches
        
        # Find the session type with highest score
        if not scores or max(scores.values()) == 0:
            return SessionType.UNKNOWN
        
        best_type = max(scores, key=scores.get)
        
        # If multiple session types have similar high scores, it's mixed
        max_score = max(scores.values())
        high_scores = [k for k, v in scores.items() if v >= max_score * 0.6 and v > 0]
        if len(high_scores) > 1 and max_score > 1:
            return SessionType.MIXED
        
        return best_type
    
    def configure_from_scenario(self, scenario_text: str) -> None:
        """Configure session based on scenario analysis."""
        if self.auto_configured:
            logger.debug(f"Session {self.session_id} already auto-configured, skipping scenario analysis")
            return
        
        self.scenario_text = scenario_text
        
        # Detect explicit library preference (highest priority)
        self.explicit_library_preference = self.detect_explicit_library_preference(scenario_text)
        
        # Detect session type
        self.session_type = self.detect_session_type_from_scenario(scenario_text)
        
        # Configure session based on detected type and preferences
        self._apply_session_configuration()
        
        self.auto_configured = True
        self.update_activity()
        
        logger.info(f"Session {self.session_id} auto-configured: type={self.session_type.value}, explicit_lib={self.explicit_library_preference}")
    
    def _apply_session_configuration(self) -> None:
        """Apply configuration based on detected session type and preferences."""
        profiles = self._get_session_profiles()
        
        # If we have an explicit web automation library preference, set session type to web automation
        if self.explicit_library_preference in ["SeleniumLibrary", "Browser"]:
            self.session_type = SessionType.WEB_AUTOMATION
            logger.info(f"Session type set to WEB_AUTOMATION based on explicit library preference: {self.explicit_library_preference}")
        
        # Handle explicit library preferences for web automation
        if (self.explicit_library_preference in ["SeleniumLibrary", "Browser"] and 
            self.session_type == SessionType.WEB_AUTOMATION):
            
            # Create a custom profile for explicit library preference
            if self.explicit_library_preference == "SeleniumLibrary":
                # Use SeleniumLibrary instead of Browser
                profile = SessionProfile(
                    session_type=SessionType.WEB_AUTOMATION,
                    core_libraries=["BuiltIn", "SeleniumLibrary", "Collections", "String"],
                    optional_libraries=["XML", "DateTime", "Screenshot"],
                    search_order=["SeleniumLibrary", "BuiltIn", "Collections", "String", "XML"],
                    keywords_patterns=profiles[SessionType.WEB_AUTOMATION].keywords_patterns,
                    description="Web automation testing with SeleniumLibrary"
                )
            else:  # Browser Library
                profile = profiles[SessionType.WEB_AUTOMATION]
        else:
            # Use standard profile for detected session type
            profile = profiles.get(self.session_type)
        
        if not profile:
            # Fallback to minimal configuration
            self.search_order = ["BuiltIn", "Collections", "String"]
            return
        
        # Set search order from profile
        self.search_order = profile.search_order.copy()
        
        # Import the preferred library if explicit preference exists
        if self.explicit_library_preference:
            try:
                self.import_library(self.explicit_library_preference, force=True)
                logger.info(f"Auto-imported preferred library: {self.explicit_library_preference}")
            except ValueError as e:
                logger.warning(f"Could not import preferred library {self.explicit_library_preference}: {e}")
    
    # ===============================
    # ActiveSession Methods (merged functionality)
    # ===============================
    
    def record_keyword_usage(self, keyword_name: str) -> None:
        """Record that a keyword was used."""
        self.keywords_used.append(keyword_name.lower())
        self.request_count += 1
        
        # Re-evaluate session type every few requests if not explicitly configured
        if not self.auto_configured and (self.request_count <= 5 or self.request_count % 10 == 0):
            self._update_session_type_from_keywords()
    
    def _update_session_type_from_keywords(self) -> None:
        """Update session type based on keyword usage patterns."""
        if not self.keywords_used:
            return
        
        # Create a pseudo-scenario from recent keywords
        recent_keywords = ' '.join(self.keywords_used[-10:])  # Last 10 keywords
        old_type = self.session_type
        
        # Detect session type from keyword patterns
        new_type = self.detect_session_type_from_scenario(recent_keywords)
        
        if new_type != old_type and new_type != SessionType.UNKNOWN:
            logger.info(f"Session {self.session_id} type updated from {old_type.value} to {new_type.value} based on keyword usage")
            self.session_type = new_type
            self._update_search_order_from_type()
    
    def _update_search_order_from_type(self) -> None:
        """Update search order based on session type."""
        profiles = self._get_session_profiles()
        profile = profiles.get(self.session_type)
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
            logger.info(f"Session {self.session_id} search order updated: {self.search_order}")
    
    def get_libraries_to_load(self) -> List[str]:
        """Get list of libraries that should be loaded for this session."""
        if self.session_type == SessionType.UNKNOWN:
            # For unknown sessions, load minimal core set
            return ["BuiltIn", "Collections", "String"]
        
        profiles = self._get_session_profiles()
        profile = profiles.get(self.session_type)
        if not profile:
            return ["BuiltIn", "Collections", "String"]
        
        # Handle explicit library preferences
        if (self.explicit_library_preference == "SeleniumLibrary" and 
            self.session_type == SessionType.WEB_AUTOMATION):
            # Replace Browser with SeleniumLibrary in core libraries
            core_libs = profile.core_libraries.copy()
            if "Browser" in core_libs:
                core_libs[core_libs.index("Browser")] = "SeleniumLibrary"
            return core_libs
        
        # Return core libraries for this session type
        return profile.core_libraries
    
    def get_optional_libraries(self) -> List[str]:
        """Get list of optional libraries for this session."""
        if self.session_type == SessionType.UNKNOWN:
            return []
        
        profiles = self._get_session_profiles()
        profile = profiles.get(self.session_type)
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
            profiles = self._get_session_profiles()
            profile = profiles.get(self.session_type)
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
        """Get comprehensive session information."""
        return {
            "session_id": self.session_id,
            "session_type": self.session_type.value,
            "explicit_library_preference": self.explicit_library_preference,
            "auto_configured": self.auto_configured,
            "loaded_libraries": list(self.loaded_libraries),
            "imported_libraries": self.imported_libraries,
            "search_order": self.search_order,
            "keywords_used_count": len(self.keywords_used),
            "request_count": self.request_count,
            "recent_keywords": self.keywords_used[-5:] if self.keywords_used else [],
            "step_count": len(self.steps),
            "web_automation_library": self.get_web_automation_library(),
            "active_library": self.get_active_library(),
            "is_browser_session": self.is_browser_session(),
            "scenario_text": self.scenario_text[:100] + "..." if self.scenario_text and len(self.scenario_text) > 100 else self.scenario_text,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "duration": self.duration
        }
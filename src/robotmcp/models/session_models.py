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
    MOBILE_TESTING = "mobile_testing"
    DATABASE_TESTING = "database_testing"
    VISUAL_TESTING = "visual_testing"
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
            ),
            
            SessionType.MOBILE_TESTING: SessionProfile(
                session_type=SessionType.MOBILE_TESTING,
                core_libraries=["BuiltIn", "AppiumLibrary", "Collections", "String"],
                optional_libraries=["XML", "DateTime", "Screenshot", "ImageLibrary"],
                search_order=["AppiumLibrary", "BuiltIn", "Collections", "String", "XML"],
                keywords_patterns=[
                    r'\b(mobile|android|ios|device|app)\b',
                    r'\b(tap|swipe|scroll|pinch|zoom)\b',
                    r'\b(install|launch|close)\s+(app|application)\b',
                    r'\b(appium|mobile\s*automation)\b'
                ],
                description="Mobile application testing with Appium"
            ),
            
            SessionType.DATABASE_TESTING: SessionProfile(
                session_type=SessionType.DATABASE_TESTING,
                core_libraries=["BuiltIn", "DatabaseLibrary", "Collections", "String"],
                optional_libraries=["DateTime", "XML", "OperatingSystem"],
                search_order=["DatabaseLibrary", "BuiltIn", "Collections", "String"],
                keywords_patterns=[
                    r'\b(database|sql|query|table|record)\b',
                    r'\b(connect|disconnect|execute|select|insert|update|delete)\b',
                    r'\b(mysql|postgresql|sqlite|oracle|mongodb)\b',
                    r'\b(transaction|commit|rollback)\b'
                ],
                description="Database testing and validation"
            ),
            
            SessionType.VISUAL_TESTING: SessionProfile(
                session_type=SessionType.VISUAL_TESTING,
                core_libraries=["BuiltIn", "ImageLibrary", "Collections", "String"],
                optional_libraries=["Browser", "SeleniumLibrary", "Screenshot", "DateTime"],
                search_order=["ImageLibrary", "Browser", "BuiltIn", "Collections", "String"],
                keywords_patterns=[
                    r'\b(image|screenshot|visual|compare|pixel)\b',
                    r'\b(capture|template|match|similarity)\b',
                    r'\b(visual\s*testing|image\s*comparison)\b'
                ],
                description="Visual testing and image comparison"
            )
        }
    
    # ===============================
    # Scenario-Based Library Detection
    # ===============================
    
    def detect_explicit_library_preference(self, scenario_text: str) -> Optional[str]:
        """Detect explicit library preference from scenario text with enhanced patterns and confidence scoring."""
        if not scenario_text:
            return None
        
        text_lower = scenario_text.lower()
        library_scores = {}
        
        # Enhanced Selenium patterns (highest priority for explicit mentions)
        selenium_patterns = [
            (r'\b(use|using|with|via|through)\s+(selenium|seleniumlibrary|selenium\s*library)\b', 10),
            (r'\btest\s+automation\s+with\s+selenium\b', 8),
            (r'\bseleniumlibrary\b', 9),
            (r'\bwebdriver\b', 6),  # WebDriver often implies Selenium
            (r'\bselenium\b(?!.*browser)(?!.*grid)', 7),  # Selenium mentioned but not "selenium browser" or "selenium grid"
            (r'\b(selenium|webdriver)\s+(automation|testing|framework)\b', 8),
            (r'\bchrome\s*driver|firefox\s*driver|edge\s*driver\b', 5),
        ]
        
        # Enhanced Browser Library patterns
        browser_patterns = [
            (r'\b(use|using|with|via|through)\s+(browser|browserlibrary|browser\s*library|playwright)\b', 10),
            (r'\btest\s+automation\s+with\s+(browser\s*library|playwright)\b', 8),
            (r'\bbrowser\s*library\b', 9),
            (r'\bplaywright\b', 8),
            (r'\b(modern|new)\s+(browser|web)\s+(automation|testing)\b', 6),
            (r'\b(chromium|firefox|webkit)\s+(browser|automation)\b', 7),
            (r'\b(headless|headed)\s+(browser|testing)\b', 5),
            (r'\basync\s+(browser|web)\s+automation\b', 6),
        ]
        
        # Enhanced API testing patterns
        api_patterns = [
            (r'\b(use|using|with)\s+(requests|requestslibrary|requests\s*library)\b', 10),
            (r'\btest\s+automation\s+with\s+requests\b', 8),
            (r'\brequestslibrary\b', 9),
            (r'\b(rest|http)\s+(api|testing|automation)\b', 7),
            (r'\b(post|get|put|delete)\s+(request|endpoint)\b', 6),
            (r'\bjson\s+(api|response|request)\b', 5),
            (r'\bapi\s+(automation|testing|validation)\b', 6),
        ]
        
        # Enhanced XML processing patterns
        xml_patterns = [
            (r'\b(use|using|with)\s+(xml|xmllibrary|xml\s*library)\b', 10),
            (r'\btest\s+automation\s+with\s+xml\b', 8),
            (r'\bxmllibrary\b', 9),
            (r'\b(xml|xpath)\s+(processing|parsing|manipulation)\b', 7),
            (r'\b(parse|process|manipulate)\s+xml\b', 6),
            (r'\bxml\s+(validation|testing|automation)\b', 6),
        ]
        
        # Enhanced mobile testing patterns
        mobile_patterns = [
            (r'\b(use|using|with)\s+(appium|appiumlibrary|appium\s*library)\b', 10),
            (r'\btest\s+automation\s+with\s+appium\b', 8),
            (r'\bappiumlibrary\b', 9),
            (r'\b(mobile|android|ios)\s+(app|testing|automation)\b', 7),
            (r'\b(device|mobile)\s+(automation|testing)\b', 6),
            (r'\b(native|hybrid)\s+(app|mobile)\s+(testing|automation)\b', 6),
        ]
        
        # Enhanced database testing patterns
        database_patterns = [
            (r'\b(use|using|with)\s+(database|databaselibrary|database\s*library)\b', 10),
            (r'\btest\s+automation\s+with\s+database\b', 8),
            (r'\bdatabaselibrary\b', 9),
            (r'\b(sql|database)\s+(testing|automation|queries)\b', 7),
            (r'\b(mysql|postgresql|sqlite|oracle)\s+(database|testing)\b', 6),
            (r'\bdatabase\s+(validation|testing|automation)\b', 6),
        ]
        
        # Score all patterns
        all_patterns = [
            ("SeleniumLibrary", selenium_patterns),
            ("Browser", browser_patterns),
            ("RequestsLibrary", api_patterns),
            ("XML", xml_patterns),
            ("AppiumLibrary", mobile_patterns),
            ("DatabaseLibrary", database_patterns),
        ]
        
        for library_name, patterns in all_patterns:
            library_scores[library_name] = 0
            for pattern, weight in patterns:
                matches = len(re.findall(pattern, text_lower))
                library_scores[library_name] += matches * weight
        
        # Find highest scoring library
        if library_scores:
            best_library = max(library_scores, key=library_scores.get)
            max_score = library_scores[best_library]
            
            # Only return if confidence is high enough
            if max_score >= 5:  # Minimum confidence threshold
                logger.info(f"Detected explicit {best_library} preference (score: {max_score}) in scenario")
                return best_library
        
        # Fallback: Generic patterns for common libraries (lower confidence)
        fallback_patterns = [
            (r'\b(xml|xpath)\b', "XML"),
            (r'\b(api|http|rest|request)\b', "RequestsLibrary"),
            (r'\b(mobile|android|ios|device)\b', "AppiumLibrary"),
            (r'\b(database|sql|mysql|postgresql)\b', "DatabaseLibrary"),
        ]
        
        for pattern, library in fallback_patterns:
            if re.search(pattern, text_lower):
                logger.info(f"Fallback detection: {library} preference based on generic pattern")
                return library
        
        return None
    
    def detect_session_type_from_scenario(self, scenario_text: str) -> SessionType:
        """Detect session type from scenario text with enhanced multi-pass pattern matching."""
        if not scenario_text:
            return SessionType.UNKNOWN
        
        text_lower = scenario_text.lower()
        profiles = self._get_session_profiles()
        scores = {session_type: 0 for session_type in profiles.keys()}
        
        # Enhanced scoring with weighted patterns and context awareness
        pattern_weights = {
            # High confidence patterns get higher weight
            SessionType.WEB_AUTOMATION: [
                (r'\b(web|browser|page|form|button|click|fill)\b', 2),
                (r'\b(selenium|browser\s*library|playwright)\b', 4),  # Explicit library mentions
                (r'\b(login|registration|navigation|ui|website)\b', 2),
                (r'\b(url|link|element|locator|css|xpath)\b', 2),
                (r'\b(headless|headed|screenshot|automation)\b', 1),
                (r'\bhttps?://\S+', 3),  # URL detection
            ],
            SessionType.API_TESTING: [
                (r'\b(api|rest|http|endpoint|request)\b', 3),
                (r'\b(get|post|put|delete|patch)\s+(request|method|call)\b', 4),
                (r'\b(json|xml|response|status|header)\b', 2),
                (r'\b(oauth|token|authentication|authorization)\b', 2),
                (r'\bmicroservice|webhook|graphql\b', 2),
                (r'\bapi\s+(testing|automation|validation)\b', 3),
            ],
            SessionType.XML_PROCESSING: [
                (r'\b(xml|xpath|parse|element|attribute)\b', 3),
                (r'\b(document|node|tag|namespace)\b', 2),
                (r'\b(xslt|xsd|dtd|schema)\b', 3),
                (r'\bxml\s+(processing|parsing|validation)\b', 4),
            ],
            SessionType.DATA_PROCESSING: [
                (r'\b(data|process|transform|manipulate|convert)\b', 2),
                (r'\b(list|dictionary|collection|string|csv|excel)\b', 2),
                (r'\b(filter|sort|aggregate|merge|split)\b', 2),
                (r'\bdata\s+(processing|transformation|manipulation)\b', 3),
            ],
            SessionType.SYSTEM_TESTING: [
                (r'\b(system|process|file|directory|filesystem)\b', 2),
                (r'\b(command|shell|environment|terminal)\b', 2),
                (r'\b(ssh|ftp|sftp|scp)\b', 2),
                (r'\bsystem\s+(testing|automation|administration)\b', 3),
            ],
            SessionType.MOBILE_TESTING: [
                (r'\b(mobile|android|ios|device|app)\b', 3),
                (r'\b(appium|espresso|xcuitest)\b', 4),
                (r'\b(tap|swipe|scroll|pinch|gesture)\b', 3),
                (r'\bmobile\s+(testing|automation|app)\b', 4),
            ],
            SessionType.DATABASE_TESTING: [
                (r'\b(database|sql|query|table|record)\b', 3),
                (r'\b(mysql|postgresql|sqlite|oracle|mongodb)\b', 3),
                (r'\b(select|insert|update|delete|create)\s+(table|from|into)\b', 3),
                (r'\bdatabase\s+(testing|automation|validation)\b', 4),
            ]
        }
        
        # Score using weighted patterns
        for session_type, weighted_patterns in pattern_weights.items():
            for pattern, weight in weighted_patterns:
                matches = len(re.findall(pattern, text_lower))
                scores[session_type] += matches * weight
        
        # Also use original profile patterns as fallback
        for session_type, profile in profiles.items():
            for pattern in profile.keywords_patterns:
                matches = len(re.findall(pattern, text_lower))
                scores[session_type] += matches  # Default weight of 1
        
        # Find the session type with highest score
        if not scores or max(scores.values()) == 0:
            return SessionType.UNKNOWN
        
        best_type = max(scores, key=scores.get)
        max_score = max(scores.values())
        
        # Enhanced mixed-type detection with smarter thresholds
        high_scores = [k for k, v in scores.items() if v >= max_score * 0.7 and v > 1]
        if len(high_scores) > 1 and max_score > 2:
            logger.info(f"Multiple high-scoring session types detected: {[k.value for k in high_scores]}")
            return SessionType.MIXED
        
        logger.info(f"Detected session type: {best_type.value} (score: {max_score})")
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
        """Apply enhanced configuration with improved conflict resolution based on detected session type and preferences."""
        profiles = self._get_session_profiles()
        
        # Enhanced explicit library preference handling with conflict resolution
        profile = self._get_profile_for_preferences(profiles)
        
        if not profile:
            # Fallback to minimal configuration
            self.search_order = ["BuiltIn", "Collections", "String"]
            return
        
        # Apply conflict resolution before setting search order
        self._resolve_library_conflicts()
        
        # Set intelligent search order using RF Set Library Search Order concept
        self.search_order = self._build_intelligent_search_order(profile)
        
        # Import the preferred library if explicit preference exists
        if self.explicit_library_preference:
            try:
                self.import_library(self.explicit_library_preference, force=True)
                logger.info(f"Auto-imported preferred library: {self.explicit_library_preference}")
            except ValueError as e:
                logger.warning(f"Could not import preferred library {self.explicit_library_preference}: {e}")
    
    def _resolve_library_conflicts(self) -> None:
        """Resolve library conflicts based on explicit preferences and session type."""
        if not self.explicit_library_preference:
            return
        
        # Define exclusion groups with priority handling
        exclusion_groups = [
            {
                "libraries": {"Browser", "SeleniumLibrary"},
                "description": "Web automation libraries",
                "default_priority": "Browser"  # Modern default
            },
            # Future: Add more exclusion groups as needed
            # {
            #     "libraries": {"RequestsLibrary", "RESTLibrary"},
            #     "description": "HTTP/REST testing libraries",
            #     "default_priority": "RequestsLibrary"
            # }
        ]
        
        for group in exclusion_groups:
            if self.explicit_library_preference in group["libraries"]:
                # Remove conflicting libraries from imported libraries
                for conflicting_lib in group["libraries"]:
                    if (conflicting_lib != self.explicit_library_preference and 
                        conflicting_lib in self.imported_libraries):
                        self.imported_libraries.remove(conflicting_lib)
                        logger.info(f"Removed conflicting library {conflicting_lib} in favor of {self.explicit_library_preference}")
                
                # Update loaded libraries set to reflect preference
                for conflicting_lib in group["libraries"]:
                    if conflicting_lib != self.explicit_library_preference:
                        self.loaded_libraries.discard(conflicting_lib)
                
                break
    
    def _get_profile_for_preferences(self, profiles: Dict[SessionType, SessionProfile]) -> Optional[SessionProfile]:
        """Get the appropriate profile based on explicit preferences and session type."""
        # Handle explicit library preferences with session type override
        if self.explicit_library_preference:
            # Web automation libraries
            if self.explicit_library_preference in ["SeleniumLibrary", "Browser"]:
                self.session_type = SessionType.WEB_AUTOMATION
                logger.info(f"Session type set to WEB_AUTOMATION based on explicit library preference: {self.explicit_library_preference}")
                
                if self.explicit_library_preference == "SeleniumLibrary":
                    # Custom profile for SeleniumLibrary
                    return SessionProfile(
                        session_type=SessionType.WEB_AUTOMATION,
                        core_libraries=["BuiltIn", "SeleniumLibrary", "Collections", "String"],
                        optional_libraries=["XML", "DateTime", "Screenshot"],
                        search_order=["SeleniumLibrary", "BuiltIn", "Collections", "String", "XML"],
                        keywords_patterns=profiles[SessionType.WEB_AUTOMATION].keywords_patterns,
                        description="Web automation testing with SeleniumLibrary"
                    )
                else:  # Browser Library
                    return profiles[SessionType.WEB_AUTOMATION]
            
            # Mobile testing libraries
            elif self.explicit_library_preference == "AppiumLibrary":
                self.session_type = SessionType.MOBILE_TESTING
                return profiles[SessionType.MOBILE_TESTING]
            
            # API testing libraries
            elif self.explicit_library_preference == "RequestsLibrary":
                self.session_type = SessionType.API_TESTING
                return profiles[SessionType.API_TESTING]
            
            # Database testing libraries
            elif self.explicit_library_preference == "DatabaseLibrary":
                self.session_type = SessionType.DATABASE_TESTING
                return profiles[SessionType.DATABASE_TESTING]
            
            # XML processing libraries
            elif self.explicit_library_preference == "XML":
                self.session_type = SessionType.XML_PROCESSING
                return profiles[SessionType.XML_PROCESSING]
        
        # Use standard profile for detected session type
        return profiles.get(self.session_type)
    
    def _build_intelligent_search_order(self, profile: SessionProfile) -> List[str]:
        """Build intelligent library search order using RF Set Library Search Order concept."""
        search_order = []
        
        # 1. Explicit preference gets highest priority (like RF Set Library Search Order)
        if self.explicit_library_preference and self.explicit_library_preference not in search_order:
            search_order.append(self.explicit_library_preference)
        
        # 2. Core libraries from profile (in priority order)
        for lib in profile.core_libraries:
            if lib not in search_order:
                search_order.append(lib)
        
        # 3. Already loaded libraries (maintain existing order)
        for lib in self.loaded_libraries:
            if lib not in search_order:
                search_order.append(lib)
        
        # 4. Optional libraries from profile
        for lib in profile.optional_libraries:
            if lib not in search_order:
                search_order.append(lib)
        
        logger.debug(f"Built intelligent search order for {self.session_id}: {search_order}")
        return search_order
    
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
        """Get list of libraries that should be loaded for this session with intelligent prioritization."""
        if self.session_type == SessionType.UNKNOWN:
            # For unknown sessions, load minimal core set
            return ["BuiltIn", "Collections", "String"]
        
        profiles = self._get_session_profiles()
        profile = profiles.get(self.session_type)
        if not profile:
            return ["BuiltIn", "Collections", "String"]
        
        # Build library list using intelligent search order approach
        libraries_to_load = []
        
        # 1. Explicit preference gets highest priority
        if self.explicit_library_preference:
            libraries_to_load.append(self.explicit_library_preference)
        
        # 2. Core libraries from profile (excluding conflicts)
        for lib in profile.core_libraries:
            if lib not in libraries_to_load and not self._is_conflicting_library(lib):
                libraries_to_load.append(lib)
        
        # Ensure BuiltIn is always included (but not duplicate)
        if "BuiltIn" not in libraries_to_load:
            libraries_to_load.insert(-1 if libraries_to_load else 0, "BuiltIn")
        
        return libraries_to_load
    
    def _is_conflicting_library(self, library_name: str) -> bool:
        """Check if a library conflicts with already selected libraries."""
        if not self.explicit_library_preference:
            return False
        
        # Define exclusion groups (libraries that cannot coexist)
        exclusion_groups = [
            {"Browser", "SeleniumLibrary"},  # Web automation exclusion
            # Future: Add more exclusion groups as needed
        ]
        
        for group in exclusion_groups:
            if (library_name in group and 
                self.explicit_library_preference in group and 
                library_name != self.explicit_library_preference):
                logger.debug(f"Library {library_name} conflicts with explicit preference {self.explicit_library_preference}")
                return True
        
        return False
    
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
    
    def set_library_search_order(self, libraries: List[str]) -> None:
        """Set explicit library search order (similar to RF Set Library Search Order)."""
        # Validate that all libraries are loaded or loadable
        valid_libraries = []
        for lib in libraries:
            if lib in self.loaded_libraries or self.should_load_library(lib):
                valid_libraries.append(lib)
            else:
                logger.warning(f"Library {lib} not loaded and not in session profile, skipping")
        
        # Always ensure BuiltIn is in the search order
        if "BuiltIn" not in valid_libraries:
            valid_libraries.append("BuiltIn")
        
        old_order = self.search_order.copy()
        self.search_order = valid_libraries
        
        logger.info(f"Library search order updated from {old_order} to {self.search_order}")
    
    def resolve_keyword_library(self, keyword_name: str) -> Optional[str]:
        """Resolve which library should handle a keyword based on search order."""
        # Strip any existing library prefix
        clean_keyword = keyword_name.split('.', 1)[-1] if '.' in keyword_name else keyword_name
        
        # Check libraries in search order
        for library in self.search_order:
            if library in self.loaded_libraries:
                # TODO: In a real implementation, this would check if the library has the keyword
                # For now, we return the first loaded library in search order
                logger.debug(f"Resolved keyword '{clean_keyword}' to library '{library}' via search order")
                return library
        
        # Fallback to BuiltIn if no other library found
        return "BuiltIn"
    
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
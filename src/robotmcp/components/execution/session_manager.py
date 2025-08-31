"""Session management service with mobile platform detection."""

import logging
import uuid
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List

from robotmcp.models.config_models import ExecutionConfig
from robotmcp.models.session_models import ExecutionSession, PlatformType, MobileConfig, SessionType

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages execution sessions and their lifecycle."""

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.sessions: Dict[str, ExecutionSession] = {}

    def create_session_id(self) -> str:
        """Create a unique session ID."""
        return str(uuid.uuid4())

    def create_session(self, session_id: str) -> ExecutionSession:
        """Create a new execution session."""
        if session_id in self.sessions:
            logger.debug(
                f"Session '{session_id}' already exists, returning existing session"
            )
            return self.sessions[session_id]

        session = ExecutionSession(session_id=session_id)
        self.sessions[session_id] = session
        
        # Add reference to session manager for Phase 2 synchronization
        session._session_manager = self

        logger.info(f"Created new session: {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[ExecutionSession]:
        """Get an existing session by ID."""
        return self.sessions.get(session_id)

    def get_or_create_session(self, session_id: str) -> ExecutionSession:
        """Get existing session or create new one if it doesn't exist."""
        session = self.get_session(session_id)
        if session is None:
            session = self.create_session(session_id)
        else:
            # Update activity timestamp
            session.update_activity()
            # Ensure session manager reference exists for Phase 2
            if not hasattr(session, '_session_manager') or session._session_manager is None:
                session._session_manager = self
        return session

    def synchronize_requests_library_state(self, session: ExecutionSession) -> bool:
        """
        Synchronize RequestsLibrary session state between MCP and RF contexts.
        
        This is Phase 2 of the RequestsLibrary fix: Session State Synchronization.
        The issue is that RequestsLibrary session state is not properly initialized
        in the MCP context, leading to 500 errors even when library registration works.
        
        Args:
            session: ExecutionSession to synchronize
            
        Returns:
            True if synchronization was successful, False otherwise
        """
        try:
            from robot.running.context import EXECUTION_CONTEXTS
            
            # Check if we have an active RF context
            if not EXECUTION_CONTEXTS.current:
                logger.debug("No active RF execution context for RequestsLibrary sync")
                return False
            
            rf_context = EXECUTION_CONTEXTS.current
            
            # Try to get the RequestsLibrary instance from RF context
            try:
                requests_lib = rf_context.namespace.get_library_instance('RequestsLibrary')
                
                if not requests_lib:
                    logger.debug("RequestsLibrary not found in RF context during sync")
                    return False
                
                # Ensure proper session state initialization for RequestsLibrary
                if not hasattr(requests_lib, '_session_store'):
                    logger.debug("Initializing RequestsLibrary session store")
                    requests_lib._session_store = {}
                
                # Check if RequestsLibrary has a session attribute for default session
                if not hasattr(requests_lib, 'session'):
                    logger.debug("Initializing RequestsLibrary default session attribute")
                    # RequestsLibrary creates sessions on demand, but we ensure the structure exists
                    requests_lib.session = None  # This will be populated by RequestsLibrary when needed
                
                # Verify RequestsLibrary is in a working state
                if hasattr(requests_lib, 'builtin'):
                    logger.debug("RequestsLibrary has proper BuiltIn integration")
                else:
                    logger.debug("Setting up RequestsLibrary BuiltIn integration")
                    from robot.libraries.BuiltIn import BuiltIn
                    requests_lib.builtin = BuiltIn()
                
                logger.debug("RequestsLibrary session state synchronized successfully")
                return True
                
            except Exception as e:
                logger.warning(f"Failed to get RequestsLibrary instance during sync: {e}")
                return False
            
        except Exception as e:
            logger.error(f"RequestsLibrary session synchronization failed: {e}")
            import traceback
            logger.debug(f"RequestsLibrary sync traceback: {traceback.format_exc()}")
            return False

    def remove_session(self, session_id: str) -> bool:
        """Remove a session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.cleanup()
            del self.sessions[session_id]
            logger.info(f"Removed session: {session_id}")
            return True
        return False

    def cleanup_expired_sessions(self) -> int:
        """Clean up sessions that have been inactive for too long."""
        cutoff_time = datetime.now() - timedelta(
            seconds=self.config.SESSION_CLEANUP_TIMEOUT
        )
        expired_sessions = []

        for session_id, session in self.sessions.items():
            if session.last_activity < cutoff_time:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            self.remove_session(session_id)

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

        return len(expired_sessions)

    def apply_state_updates(
        self, session: ExecutionSession, state_updates: Dict[str, Any]
    ) -> None:
        """Apply state updates to a session."""
        if not state_updates:
            return

        # Update browser state
        browser_state = session.browser_state

        for key, value in state_updates.items():
            if key == "current_browser":
                if isinstance(value, dict):
                    browser_state.browser_type = value.get("type")
                elif value is None:
                    browser_state.browser_type = None

            elif key == "current_context":
                if isinstance(value, dict):
                    browser_state.context_id = value.get("id")
                elif value is None:
                    browser_state.context_id = None

            elif key == "current_page":
                if isinstance(value, dict):
                    browser_state.current_url = value.get("url")
                    browser_state.page_id = value.get("id")
                elif value is None:
                    browser_state.current_url = None
                    browser_state.page_id = None

            elif hasattr(browser_state, key):
                setattr(browser_state, key, value)

            elif key in ["variables", "session_variables"]:
                if isinstance(value, dict):
                    session.variables.update(value)

        session.update_activity()
        logger.debug(
            f"Applied state updates to session {session.session_id}: {list(state_updates.keys())}"
        )

    def get_session_count(self) -> int:
        """Get the total number of active sessions."""
        return len(self.sessions)

    def get_all_session_ids(self) -> list[str]:
        """Get list of all active session IDs."""
        return list(self.sessions.keys())

    def detect_platform_from_scenario(self, scenario: str) -> PlatformType:
        """
        Detect platform type from scenario description.
        
        Args:
            scenario: Natural language scenario description
            
        Returns:
            Detected platform type
        """
        scenario_lower = scenario.lower()
        
        # Mobile indicators
        mobile_keywords = ['app', 'mobile', 'android', 'ios', 'iphone', 'ipad', 
                          'device', 'appium', 'emulator', 'simulator', 'apk',
                          'bundle', 'tap', 'swipe', 'gesture']
        
        # Web indicators  
        web_keywords = ['browser', 'web', 'website', 'url', 'page', 'chrome',
                       'firefox', 'safari', 'edge', 'selenium', 'click link']
        
        # API indicators
        api_keywords = ['api', 'rest', 'soap', 'endpoint', 'request', 'response',
                       'json', 'xml', 'http', 'graphql']
        
        # Count keyword matches
        mobile_score = sum(1 for keyword in mobile_keywords if keyword in scenario_lower)
        web_score = sum(1 for keyword in web_keywords if keyword in scenario_lower)
        api_score = sum(1 for keyword in api_keywords if keyword in scenario_lower)
        
        # Determine platform based on scores
        if mobile_score > web_score and mobile_score > api_score:
            return PlatformType.MOBILE
        elif api_score > web_score:
            return PlatformType.API
        else:
            return PlatformType.WEB  # Default to web
    
    def initialize_mobile_session(self, session: ExecutionSession, scenario: str = None) -> None:
        """
        Initialize session with mobile configuration.
        
        Args:
            session: Session to initialize
            scenario: Optional scenario text for parsing requirements
        """
        session.platform_type = PlatformType.MOBILE
        session.session_type = SessionType.MOBILE_TESTING
        
        # Parse mobile requirements from scenario if provided
        if scenario:
            config = self.parse_mobile_requirements(scenario)
            session.mobile_config = config
            
        # Set mobile-specific exclusions and ensure AppiumLibrary is properly tracked
        session.loaded_libraries.add('AppiumLibrary')
        if 'AppiumLibrary' not in session.imported_libraries:
            session.imported_libraries.append('AppiumLibrary')  # Track both loaded and imported
        
        # Set mobile session context
        session.current_context = "NATIVE_APP"  # Default mobile context
        
        logger.info(f"Initialized mobile session: {session.session_id} with AppiumLibrary loaded")
    
    def parse_mobile_requirements(self, scenario: str) -> MobileConfig:
        """
        Parse mobile configuration from scenario text.
        
        Args:
            scenario: Natural language scenario
            
        Returns:
            MobileConfig with parsed requirements
        """
        config = MobileConfig()
        scenario_lower = scenario.lower()
        
        # Detect platform
        if 'android' in scenario_lower:
            config.platform_name = 'Android'
            config.automation_name = 'UiAutomator2'
            
            # Look for package/activity mentions
            package_match = re.search(r'com\.\w+(?:\.\w+)*', scenario)
            if package_match:
                config.app_package = package_match.group(0)
                
        elif 'ios' in scenario_lower or 'iphone' in scenario_lower or 'ipad' in scenario_lower:
            config.platform_name = 'iOS'
            config.automation_name = 'XCUITest'
            
            # Look for bundle ID mentions
            bundle_match = re.search(r'com\.\w+(?:\.\w+)*', scenario)
            if bundle_match:
                config.bundle_id = bundle_match.group(0)
        
        # Detect device name
        if 'emulator' in scenario_lower:
            config.device_name = 'emulator-5554'
        elif 'simulator' in scenario_lower:
            config.device_name = 'iPhone Simulator'
        elif 'real device' in scenario_lower or 'physical device' in scenario_lower:
            # Would need actual device UDID
            pass
            
        # Detect app path
        app_match = re.search(r'[\/\\][\w\/\\]+\.(apk|app|ipa)', scenario)
        if app_match:
            config.app_path = app_match.group(0)
            
        return config
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary information about a session."""
        session = self.get_session(session_id)
        if not session:
            return None

        info = {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "duration": session.duration,
            "step_count": session.step_count,
            "imported_libraries": session.imported_libraries,
            "active_library": session.get_active_library(),
            "has_browser_session": session.is_browser_session(),
            "variables_count": len(session.variables),
            "platform_type": session.platform_type.value,
            "is_mobile": session.is_mobile_session(),
            "current_url": session.browser_state.current_url,
            "browser_type": session.browser_state.browser_type,
            "libraries_loaded": getattr(session, 'libraries_loaded', False),
            "search_order": getattr(session, 'search_order', []),
            "loaded_libraries": list(getattr(session, 'loaded_libraries', set())),
        }
        
        # Add mobile-specific info if applicable
        if session.is_mobile_session() and session.mobile_config:
            info["mobile_config"] = {
                "platform_name": session.mobile_config.platform_name,
                "device_name": session.mobile_config.device_name,
                "app_package": session.mobile_config.app_package,
                "app_activity": session.mobile_config.app_activity,
                "automation_name": session.mobile_config.automation_name,
                "appium_server_url": session.mobile_config.appium_server_url
            }
            info["appium_session_id"] = session.appium_session_id
            info["current_context"] = session.current_context
            
        return info

    def get_all_sessions_info(self) -> Dict[str, Dict[str, Any]]:
        """Get summary information about all sessions."""
        return {
            session_id: self.get_session_info(session_id)
            for session_id in self.sessions.keys()
        }

    def cleanup_all_sessions(self) -> int:
        """Clean up all sessions (typically called on shutdown)."""
        count = len(self.sessions)
        session_ids = list(self.sessions.keys())

        for session_id in session_ids:
            self.remove_session(session_id)

        logger.info(f"Cleaned up all {count} sessions")
        return count
    
    def get_most_recent_session(self) -> Optional[ExecutionSession]:
        """Get the most recently active session."""
        if not self.sessions:
            return None
        
        most_recent = max(self.sessions.values(), key=lambda s: s.last_activity)
        return most_recent
    
    def get_sessions_with_steps(self) -> List[ExecutionSession]:
        """Get all sessions that have executed steps."""
        sessions_with_steps = [s for s in self.sessions.values() if s.step_count > 0]
        # Sort by last activity (most recent first)
        sessions_with_steps.sort(key=lambda s: s.last_activity, reverse=True)
        return sessions_with_steps
    
    def suggest_session_for_suite_build(self) -> Optional[str]:
        """Suggest best session ID for test suite building."""
        sessions_with_steps = self.get_sessions_with_steps()
        
        if not sessions_with_steps:
            return None
        
        # Return the most recently active session with steps
        return sessions_with_steps[0].session_id

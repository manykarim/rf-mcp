"""Session management service."""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List

from robotmcp.models.config_models import ExecutionConfig
from robotmcp.models.session_models import ExecutionSession

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
        return session

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

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary information about a session."""
        session = self.get_session(session_id)
        if not session:
            return None

        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "duration": session.duration,
            "step_count": session.step_count,
            "imported_libraries": session.imported_libraries,
            "active_library": session.get_active_library(),
            "has_browser_session": session.is_browser_session(),
            "variables_count": len(session.variables),
            "current_url": session.browser_state.current_url,
            "browser_type": session.browser_state.browser_type,
        }

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

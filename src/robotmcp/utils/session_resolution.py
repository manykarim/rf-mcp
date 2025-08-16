"""Session resolution utilities for intelligent session ID fallback logic."""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SessionResolver:
    """Provides intelligent session resolution with fallback logic."""
    
    def __init__(self, session_manager):
        """Initialize with session manager reference."""
        self.session_manager = session_manager
    
    def resolve_session_id(self, provided_session_id: str) -> Optional[str]:
        """
        Resolve session ID with intelligent fallback logic.
        
        Args:
            provided_session_id: Session ID provided by user (may be empty/invalid)
            
        Returns:
            Valid session ID or None if no suitable session found
        """
        # Step 1: Use provided session if it exists and has steps
        if provided_session_id and provided_session_id.strip():
            session = self.session_manager.get_session(provided_session_id.strip())
            if session and session.step_count > 0:
                logger.info(f"Using provided session: {provided_session_id}")
                return provided_session_id.strip()
            elif session and session.step_count == 0:
                logger.info(f"Provided session {provided_session_id} exists but has no steps")
            else:
                logger.info(f"Provided session {provided_session_id} not found")
        
        # Step 2: Use most recently active session with steps
        recent_session = self.get_most_recent_session_with_steps()
        if recent_session:
            logger.info(f"Using most recent active session: {recent_session.session_id}")
            return recent_session.session_id
        
        # Step 3: Use "default" session if it has steps
        default_session = self.session_manager.get_session("default")
        if default_session and default_session.step_count > 0:
            logger.info("Using default session")
            return "default"
        
        # Step 4: No suitable session found
        logger.warning("No session found with executed steps")
        return None
    
    def get_most_recent_session_with_steps(self) -> Optional[Any]:
        """Get the most recently active session that has executed steps."""
        sessions_with_steps = []
        
        for session_id, session in self.session_manager.sessions.items():
            if session.step_count > 0:
                sessions_with_steps.append(session)
        
        if not sessions_with_steps:
            return None
        
        # Sort by last activity (most recent first)
        sessions_with_steps.sort(key=lambda s: s.last_activity, reverse=True)
        return sessions_with_steps[0]
    
    def get_sessions_with_steps(self) -> List[Any]:
        """Get all sessions that have executed steps."""
        sessions_with_steps = []
        
        for session_id, session in self.session_manager.sessions.items():
            if session.step_count > 0:
                sessions_with_steps.append(session)
        
        # Sort by last activity (most recent first)
        sessions_with_steps.sort(key=lambda s: s.last_activity, reverse=True)
        return sessions_with_steps
    
    def suggest_session_for_suite_build(self) -> Optional[str]:
        """Suggest best session ID for test suite building."""
        sessions_with_steps = self.get_sessions_with_steps()
        
        if not sessions_with_steps:
            return None
        
        # Prefer sessions with more recent activity and more steps
        best_session = None
        best_score = 0
        
        now = datetime.now()
        
        for session in sessions_with_steps:
            # Calculate recency score (higher for more recent activity)
            time_diff = (now - session.last_activity).total_seconds()
            recency_score = max(0, 100 - (time_diff / 60))  # Decay over minutes
            
            # Calculate completeness score (more steps = higher score)
            completeness_score = min(session.step_count * 10, 100)
            
            # Calculate configuration score (auto-configured sessions preferred)
            config_score = 20 if getattr(session, 'auto_configured', False) else 0
            
            total_score = recency_score + completeness_score + config_score
            
            if total_score > best_score:
                best_score = total_score
                best_session = session
        
        return best_session.session_id if best_session else None
    
    def build_session_error_guidance(self, 
                                   session_id: str, 
                                   available_sessions: List[str]) -> Dict[str, Any]:
        """Build helpful error message for session issues."""
        sessions_with_steps = [s.session_id for s in self.get_sessions_with_steps()]
        suggested_session = self.suggest_session_for_suite_build()
        
        guidance = {
            "error": f"Session '{session_id}' not found or has no executed steps",
            "available_sessions": available_sessions,
            "sessions_with_steps": sessions_with_steps,
            "suggested_session": suggested_session,
            "suggestions": [
                "Use the session_id returned by analyze_scenario",
                "Check if you executed steps in the same session",
                "Use get_session_validation_status to check session state"
            ],
            "recommended_action": "Call analyze_scenario first to create a properly configured session"
        }
        
        if suggested_session:
            guidance["auto_recovery"] = {
                "available": True,
                "suggested_session_id": suggested_session,
                "message": f"Consider using session '{suggested_session}' which has executed steps"
            }
        else:
            guidance["auto_recovery"] = {
                "available": False,
                "message": "No sessions with executed steps found. Start with analyze_scenario."
            }
        
        return guidance
    
    def resolve_session_with_fallback(self, provided_session_id: str) -> Dict[str, Any]:
        """
        Resolve session with comprehensive fallback and guidance.
        
        Returns:
            Dictionary with resolution result and guidance
        """
        resolved_session_id = self.resolve_session_id(provided_session_id)
        
        if resolved_session_id:
            session = self.session_manager.get_session(resolved_session_id)
            return {
                "success": True,
                "session_id": resolved_session_id,
                "fallback_used": resolved_session_id != provided_session_id.strip(),
                "session_info": {
                    "step_count": session.step_count,
                    "last_activity": session.last_activity.isoformat(),
                    "auto_configured": getattr(session, 'auto_configured', False),
                    "session_type": getattr(session, 'session_type', {}).value if hasattr(getattr(session, 'session_type', None), 'value') else 'unknown'
                }
            }
        else:
            available_sessions = list(self.session_manager.sessions.keys())
            error_guidance = self.build_session_error_guidance(provided_session_id, available_sessions)
            
            return {
                "success": False,
                "session_id": None,
                "error_guidance": error_guidance
            }
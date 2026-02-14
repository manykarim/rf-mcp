"""Session Lookup Adapter.

Wraps the SessionManager for the Intent domain.
"""
from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class SessionLookupAdapter:
    """Adapts SessionManager for the Intent domain.

    Implements the SessionLookup protocol from services.py.
    """

    def __init__(self, session_manager) -> None:
        self._session_manager = session_manager

    def get_active_web_library(self, session_id: str) -> Optional[str]:
        session = self._session_manager.get_session(session_id)
        if session is None:
            return None
        return session.get_web_automation_library()

    def get_imported_libraries(self, session_id: str) -> List[str]:
        session = self._session_manager.get_session(session_id)
        if session is None:
            return []
        return list(session.imported_libraries)

    def get_platform_type(self, session_id: str) -> str:
        session = self._session_manager.get_session(session_id)
        if session is None:
            return "web"
        libs = session.imported_libraries
        if any("Appium" in lib for lib in libs):
            return "mobile"
        return "web"

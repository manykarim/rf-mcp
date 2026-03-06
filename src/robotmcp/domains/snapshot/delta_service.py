"""Delta State Retrieval Service (ADR-018).

Application-level service that manages per-session VersionedStateCache
instances with session-level LRU eviction.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .delta_aggregates import VersionedStateCache
from .delta_value_objects import StateDelta, StateVersion

logger = logging.getLogger(__name__)


@dataclass
class DeltaStateService:
    """Manages versioned state caches across multiple sessions.

    Each session gets its own VersionedStateCache. The service
    enforces a max number of active sessions with LRU eviction
    of the oldest session when capacity is reached.
    """
    max_versions_per_session: int = 5
    ttl_seconds: float = 300.0
    max_sessions: int = 50
    _caches: OrderedDict = field(default_factory=OrderedDict)

    def get_or_create_cache(self, session_id: str) -> VersionedStateCache:
        """Get an existing cache or create a new one for the session."""
        if session_id in self._caches:
            self._caches.move_to_end(session_id)
            return self._caches[session_id]

        # Evict oldest session if at capacity
        while len(self._caches) >= self.max_sessions:
            oldest = next(iter(self._caches))
            self._caches.pop(oldest)
            logger.debug("Evicted state cache for session %s", oldest)

        cache = VersionedStateCache(
            session_id=session_id,
            max_versions=self.max_versions_per_session,
            ttl_seconds=self.ttl_seconds,
        )
        self._caches[session_id] = cache
        return cache

    def record_full_state(
        self,
        session_id: str,
        sections: Dict[str, Any],
        section_names: List[str],
    ) -> StateVersion:
        """Record a full state snapshot, filtered to section_names."""
        cache = self.get_or_create_cache(session_id)
        filtered = {k: v for k, v in sections.items() if k in section_names}
        return cache.store_version(filtered)

    def compute_delta(
        self,
        session_id: str,
        since_version: int,
        sections: Dict[str, Any],
        section_names: List[str],
    ) -> Tuple[StateDelta, StateVersion]:
        """Compute a delta from since_version, filtered to section_names."""
        cache = self.get_or_create_cache(session_id)
        filtered = {k: v for k, v in sections.items() if k in section_names}
        return cache.compute_delta(since_version, filtered)

    def get_current_version(self, session_id: str) -> int:
        """Get the current version number for a session (0 if unknown)."""
        if session_id not in self._caches:
            return 0
        return self._caches[session_id].get_current_version_number()

    def clear_session(self, session_id: str) -> None:
        """Remove and clear the cache for a session."""
        cache = self._caches.pop(session_id, None)
        if cache:
            cache.clear()

    def clear_all(self) -> None:
        """Remove all session caches."""
        self._caches.clear()

    @property
    def active_sessions(self) -> int:
        """Number of sessions with active caches."""
        return len(self._caches)

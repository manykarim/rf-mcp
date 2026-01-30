"""Repository Protocol for the Element Registry Context.

The repository provides persistence for ElementRegistry aggregates,
abstracting the storage mechanism (in-memory, database, etc.).

This follows the Repository pattern from Domain-Driven Design,
providing a collection-like interface for aggregates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Protocol, runtime_checkable

from robotmcp.domains.element_registry.aggregates import ElementRegistry
from robotmcp.domains.element_registry.value_objects import RegistryId


@runtime_checkable
class ElementRegistryRepository(Protocol):
    """Repository protocol for ElementRegistry aggregates.

    This protocol defines the interface for persisting and retrieving
    ElementRegistry instances. Implementations can use various storage
    backends (memory, Redis, database, etc.).

    Thread Safety: Implementations should be thread-safe for concurrent access.
    """

    def save(self, registry: ElementRegistry) -> None:
        """Persist a registry to storage.

        If a registry with the same ID exists, it will be replaced.

        Args:
            registry: The ElementRegistry to persist
        """
        ...

    def get_for_session(self, session_id: str) -> Optional[ElementRegistry]:
        """Get the current registry for a session.

        Returns the most recent (highest version) non-invalidated registry
        for the given session.

        Args:
            session_id: The session identifier

        Returns:
            The current ElementRegistry, or None if not found
        """
        ...

    def get_by_id(self, registry_id: RegistryId) -> Optional[ElementRegistry]:
        """Get a specific registry by its ID.

        Args:
            registry_id: The unique registry identifier

        Returns:
            The ElementRegistry, or None if not found
        """
        ...

    def delete_for_session(self, session_id: str) -> None:
        """Remove all registries for a session.

        Called when a session ends to clean up resources.

        Args:
            session_id: The session identifier
        """
        ...


class InMemoryElementRegistryRepository(ABC):
    """In-memory implementation of ElementRegistryRepository.

    This implementation stores registries in memory, suitable for
    single-process deployments or testing.

    Thread Safety: Uses a simple dict, not thread-safe.
    For production use with concurrent access, use a thread-safe
    implementation or external storage.
    """

    def __init__(self) -> None:
        """Initialize the in-memory repository."""
        self._registries: Dict[str, ElementRegistry] = {}
        self._session_index: Dict[str, List[str]] = {}

    def save(self, registry: ElementRegistry) -> None:
        """Persist a registry to memory.

        Args:
            registry: The ElementRegistry to persist
        """
        registry_key = registry.registry_id.value
        session_id = registry.session_id

        # Store the registry
        self._registries[registry_key] = registry

        # Update session index
        if session_id not in self._session_index:
            self._session_index[session_id] = []
        if registry_key not in self._session_index[session_id]:
            self._session_index[session_id].append(registry_key)

    def get_for_session(self, session_id: str) -> Optional[ElementRegistry]:
        """Get the current registry for a session.

        Returns the most recent non-invalidated registry.

        Args:
            session_id: The session identifier

        Returns:
            The current ElementRegistry, or None if not found
        """
        if session_id not in self._session_index:
            return None

        registry_keys = self._session_index[session_id]
        if not registry_keys:
            return None

        # Find the most recent non-stale registry
        candidates = []
        for key in registry_keys:
            registry = self._registries.get(key)
            if registry and not registry.is_stale():
                candidates.append(registry)

        if not candidates:
            # If all are stale, return the most recent one anyway
            for key in reversed(registry_keys):
                registry = self._registries.get(key)
                if registry:
                    return registry
            return None

        # Return the one with highest version
        return max(candidates, key=lambda r: r.snapshot_version)

    def get_by_id(self, registry_id: RegistryId) -> Optional[ElementRegistry]:
        """Get a specific registry by its ID.

        Args:
            registry_id: The unique registry identifier

        Returns:
            The ElementRegistry, or None if not found
        """
        return self._registries.get(registry_id.value)

    def delete_for_session(self, session_id: str) -> None:
        """Remove all registries for a session.

        Args:
            session_id: The session identifier
        """
        if session_id not in self._session_index:
            return

        # Remove all registries for the session
        for registry_key in self._session_index[session_id]:
            self._registries.pop(registry_key, None)

        # Remove session from index
        del self._session_index[session_id]

    def get_all_for_session(self, session_id: str) -> List[ElementRegistry]:
        """Get all registries for a session (including stale ones).

        Useful for debugging and history tracking.

        Args:
            session_id: The session identifier

        Returns:
            List of all registries for the session
        """
        if session_id not in self._session_index:
            return []

        registries = []
        for key in self._session_index[session_id]:
            registry = self._registries.get(key)
            if registry:
                registries.append(registry)

        return sorted(registries, key=lambda r: r.snapshot_version)

    def cleanup_expired(self, max_age_seconds: int = 3600) -> int:
        """Remove registries older than the specified age.

        Args:
            max_age_seconds: Maximum age in seconds (default 1 hour)

        Returns:
            Number of registries removed
        """
        now = datetime.now()
        removed = 0

        expired_keys = []
        for key, registry in self._registries.items():
            age = (now - registry.created_at).total_seconds()
            if age > max_age_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            registry = self._registries.pop(key, None)
            if registry:
                removed += 1
                # Remove from session index
                session_id = registry.session_id
                if session_id in self._session_index:
                    if key in self._session_index[session_id]:
                        self._session_index[session_id].remove(key)
                    # Clean up empty session entries
                    if not self._session_index[session_id]:
                        del self._session_index[session_id]

        return removed

    def stats(self) -> Dict[str, object]:
        """Get repository statistics.

        Returns:
            Dictionary with repository stats
        """
        total_registries = len(self._registries)
        total_sessions = len(self._session_index)
        stale_count = sum(
            1 for r in self._registries.values() if r.is_stale()
        )
        total_refs = sum(
            len(r.refs) for r in self._registries.values()
        )

        return {
            "total_registries": total_registries,
            "total_sessions": total_sessions,
            "stale_registries": stale_count,
            "active_registries": total_registries - stale_count,
            "total_refs": total_refs,
        }

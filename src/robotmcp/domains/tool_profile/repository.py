"""Profile Repository Protocol.

The repository provides persistence for ToolProfile aggregates,
abstracting the storage mechanism (in-memory, database, etc.).

This follows the Repository pattern from Domain-Driven Design,
providing a collection-like interface for aggregates.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol

from .aggregates import ToolProfile


class ProfileRepository(Protocol):
    """Protocol for profile persistence (in-memory for v1).

    This protocol defines the interface for persisting and retrieving
    ToolProfile instances. Implementations can use various storage
    backends (memory, Redis, database, etc.).
    """

    def get(self, name: str) -> Optional[ToolProfile]:
        """Retrieve a profile by name.

        Args:
            name: The profile name.

        Returns:
            The ToolProfile if found, None otherwise.
        """
        ...

    def save(self, profile: ToolProfile) -> None:
        """Persist a profile.

        If a profile with the same name exists, it will be replaced.

        Args:
            profile: The ToolProfile to persist.
        """
        ...

    def list_all(self) -> List[ToolProfile]:
        """List all stored profiles.

        Returns:
            List of all ToolProfile instances.
        """
        ...


class InMemoryProfileRepository:
    """In-memory implementation of ProfileRepository.

    Useful for testing and for short-lived sessions where
    persistence is not required.

    Thread Safety: This implementation is NOT thread-safe.
    Use appropriate synchronization for concurrent access.
    """

    def __init__(self) -> None:
        """Initialize the in-memory repository."""
        self._profiles: Dict[str, ToolProfile] = {}

    def get(self, name: str) -> Optional[ToolProfile]:
        """Retrieve a profile by name.

        Args:
            name: The profile name.

        Returns:
            The ToolProfile if found, None otherwise.
        """
        return self._profiles.get(name)

    def save(self, profile: ToolProfile) -> None:
        """Persist a profile in memory.

        Args:
            profile: The ToolProfile to persist.
        """
        self._profiles[profile.name] = profile

    def list_all(self) -> List[ToolProfile]:
        """List all stored profiles.

        Returns:
            List of all ToolProfile instances.
        """
        return list(self._profiles.values())

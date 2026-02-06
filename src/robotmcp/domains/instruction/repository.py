"""Repository Protocol for the Instruction bounded context.

The repository provides persistence for InstructionConfig aggregates,
abstracting the storage mechanism (in-memory, database, etc.).

This follows the Repository pattern from Domain-Driven Design,
providing a collection-like interface for aggregates.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Protocol, runtime_checkable

from .aggregates import InstructionConfig


@runtime_checkable
class InstructionRepository(Protocol):
    """Repository protocol for InstructionConfig aggregates.

    This protocol defines the interface for persisting and retrieving
    InstructionConfig instances. Implementations can use various storage
    backends (memory, Redis, database, etc.).

    Thread Safety: Implementations should be thread-safe for concurrent access.
    """

    def save(self, config: InstructionConfig) -> None:
        """Persist an instruction configuration.

        If a configuration with the same ID exists, it will be replaced.

        Args:
            config: The InstructionConfig to persist.
        """
        ...

    def get_by_id(self, config_id: str) -> Optional[InstructionConfig]:
        """Retrieve configuration by its ID.

        Args:
            config_id: The unique configuration identifier.

        Returns:
            The InstructionConfig if found, None otherwise.
        """
        ...

    def get_for_session(self, session_id: str) -> Optional[InstructionConfig]:
        """Get configuration associated with a session.

        Args:
            session_id: The session identifier.

        Returns:
            The InstructionConfig for the session, None if not found.
        """
        ...

    def get_default(self) -> Optional[InstructionConfig]:
        """Get the global default configuration.

        Returns:
            The default InstructionConfig, None if not set.
        """
        ...

    def delete(self, config_id: str) -> bool:
        """Delete a configuration.

        Args:
            config_id: The configuration ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        ...


class InMemoryInstructionRepository:
    """In-memory implementation of InstructionRepository.

    Useful for testing and for short-lived sessions where
    persistence is not required.

    Thread Safety: This implementation is NOT thread-safe.
    Use appropriate synchronization for concurrent access.
    """

    def __init__(self) -> None:
        """Initialize the in-memory repository."""
        self._configs: Dict[str, InstructionConfig] = {}
        self._session_index: Dict[str, str] = {}  # session_id -> config_id
        self._default_config_id: Optional[str] = None

    def save(self, config: InstructionConfig) -> None:
        """Persist a configuration in memory.

        Args:
            config: The InstructionConfig to persist.
        """
        self._configs[config.config_id] = config

    def get_by_id(self, config_id: str) -> Optional[InstructionConfig]:
        """Retrieve configuration by ID.

        Args:
            config_id: The configuration identifier.

        Returns:
            The InstructionConfig if found, None otherwise.
        """
        return self._configs.get(config_id)

    def get_for_session(self, session_id: str) -> Optional[InstructionConfig]:
        """Get configuration for a session.

        Args:
            session_id: The session identifier.

        Returns:
            The InstructionConfig for the session, None if not found.
        """
        config_id = self._session_index.get(session_id)
        if config_id:
            return self._configs.get(config_id)
        return None

    def set_for_session(self, session_id: str, config: InstructionConfig) -> None:
        """Associate a configuration with a session.

        Args:
            session_id: The session identifier.
            config: The configuration to associate.
        """
        self._configs[config.config_id] = config
        self._session_index[session_id] = config.config_id

    def get_default(self) -> Optional[InstructionConfig]:
        """Get the global default configuration.

        Returns:
            The default InstructionConfig, None if not set.
        """
        if self._default_config_id:
            return self._configs.get(self._default_config_id)
        return None

    def set_default(self, config: InstructionConfig) -> None:
        """Set the global default configuration.

        Args:
            config: The configuration to set as default.
        """
        self._configs[config.config_id] = config
        self._default_config_id = config.config_id

    def delete(self, config_id: str) -> bool:
        """Delete a configuration.

        Also cleans up any session associations and default reference.

        Args:
            config_id: The configuration ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        if config_id in self._configs:
            del self._configs[config_id]

            # Clean up session index
            self._session_index = {
                sid: cid
                for sid, cid in self._session_index.items()
                if cid != config_id
            }

            # Clean up default reference
            if self._default_config_id == config_id:
                self._default_config_id = None

            return True
        return False

    def delete_for_session(self, session_id: str) -> bool:
        """Delete the configuration for a session.

        Args:
            session_id: The session identifier.

        Returns:
            True if deleted, False if not found.
        """
        config_id = self._session_index.get(session_id)
        if config_id:
            del self._session_index[session_id]
            # Note: We don't delete the config itself as it might be shared
            return True
        return False

    def get_all(self) -> List[InstructionConfig]:
        """Get all configurations.

        Returns:
            List of all stored InstructionConfig instances.
        """
        return list(self._configs.values())

    def get_all_session_ids(self) -> List[str]:
        """Get all session IDs with associated configurations.

        Returns:
            List of session IDs.
        """
        return list(self._session_index.keys())

    def clear(self) -> None:
        """Clear all configurations and associations."""
        self._configs.clear()
        self._session_index.clear()
        self._default_config_id = None

    def stats(self) -> Dict[str, object]:
        """Get repository statistics.

        Returns:
            Dictionary with repository stats.
        """
        return {
            "total_configs": len(self._configs),
            "session_associations": len(self._session_index),
            "has_default": self._default_config_id is not None,
        }

    def __len__(self) -> int:
        """Return total number of configurations."""
        return len(self._configs)

    def __contains__(self, config_id: str) -> bool:
        """Check if a configuration exists."""
        return config_id in self._configs

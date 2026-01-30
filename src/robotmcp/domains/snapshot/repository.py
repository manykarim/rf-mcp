"""
Repository Interface for the Snapshot bounded context.

Repositories provide a collection-like interface for accessing domain objects.
The interface is defined as a Protocol to allow for different implementations
(in-memory, database, etc.) without coupling the domain to infrastructure.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Protocol, runtime_checkable

from .aggregates import PageSnapshot
from .value_objects import SnapshotId


@runtime_checkable
class SnapshotRepository(Protocol):
    """
    Repository interface for PageSnapshot aggregate.

    Defines the contract for storing and retrieving snapshots.
    Implementations may use in-memory storage, databases, or other
    persistence mechanisms.
    """

    def save(self, snapshot: PageSnapshot) -> None:
        """
        Persist a snapshot.

        If a snapshot with the same ID already exists, it will be updated.

        Args:
            snapshot: The PageSnapshot to persist
        """
        ...

    def get_by_id(self, snapshot_id: SnapshotId) -> Optional[PageSnapshot]:
        """
        Retrieve a snapshot by its ID.

        Args:
            snapshot_id: The unique identifier of the snapshot

        Returns:
            The PageSnapshot if found, None otherwise
        """
        ...

    def get_latest_for_session(self, session_id: str) -> Optional[PageSnapshot]:
        """
        Get the most recent snapshot for a session.

        Args:
            session_id: The session identifier

        Returns:
            The most recent PageSnapshot for the session, None if no snapshots exist
        """
        ...

    def get_previous(self, snapshot: PageSnapshot) -> Optional[PageSnapshot]:
        """
        Get the snapshot before the given one for the same session.

        Args:
            snapshot: The reference snapshot

        Returns:
            The previous PageSnapshot, None if this is the first snapshot
        """
        ...

    def delete_older_than(self, session_id: str, keep_count: int = 5) -> int:
        """
        Clean up old snapshots, keeping the N most recent.

        Args:
            session_id: The session identifier
            keep_count: Number of recent snapshots to keep

        Returns:
            Number of snapshots deleted
        """
        ...


class InMemorySnapshotRepository:
    """
    In-memory implementation of SnapshotRepository.

    Useful for testing and for short-lived sessions where
    persistence is not required.
    """

    def __init__(self) -> None:
        self._snapshots: Dict[str, PageSnapshot] = {}
        self._session_snapshots: Dict[str, List[str]] = {}  # session_id -> list of snapshot_ids

    def save(self, snapshot: PageSnapshot) -> None:
        """Persist a snapshot in memory."""
        snapshot_id_str = str(snapshot.snapshot_id)
        self._snapshots[snapshot_id_str] = snapshot

        # Track snapshots by session
        if snapshot.session_id not in self._session_snapshots:
            self._session_snapshots[snapshot.session_id] = []

        if snapshot_id_str not in self._session_snapshots[snapshot.session_id]:
            self._session_snapshots[snapshot.session_id].append(snapshot_id_str)

    def get_by_id(self, snapshot_id: SnapshotId) -> Optional[PageSnapshot]:
        """Retrieve a snapshot by ID."""
        return self._snapshots.get(str(snapshot_id))

    def get_latest_for_session(self, session_id: str) -> Optional[PageSnapshot]:
        """Get the most recent snapshot for a session."""
        snapshot_ids = self._session_snapshots.get(session_id, [])
        if not snapshot_ids:
            return None

        # Get the most recent by created_at
        snapshots = [
            self._snapshots[sid] for sid in snapshot_ids
            if sid in self._snapshots
        ]
        if not snapshots:
            return None

        return max(snapshots, key=lambda s: s.created_at)

    def get_previous(self, snapshot: PageSnapshot) -> Optional[PageSnapshot]:
        """Get the snapshot before the given one."""
        session_id = snapshot.session_id
        snapshot_ids = self._session_snapshots.get(session_id, [])

        # Get all snapshots for this session, sorted by created_at
        snapshots = [
            self._snapshots[sid] for sid in snapshot_ids
            if sid in self._snapshots
        ]
        snapshots.sort(key=lambda s: s.created_at)

        # Find the position of the given snapshot
        current_idx = None
        for i, s in enumerate(snapshots):
            if s.snapshot_id == snapshot.snapshot_id:
                current_idx = i
                break

        if current_idx is None or current_idx == 0:
            return None

        return snapshots[current_idx - 1]

    def delete_older_than(self, session_id: str, keep_count: int = 5) -> int:
        """Clean up old snapshots, keeping the N most recent."""
        snapshot_ids = self._session_snapshots.get(session_id, [])
        if len(snapshot_ids) <= keep_count:
            return 0

        # Get all snapshots for this session, sorted by created_at
        snapshots = [
            (sid, self._snapshots[sid]) for sid in snapshot_ids
            if sid in self._snapshots
        ]
        snapshots.sort(key=lambda x: x[1].created_at, reverse=True)

        # Keep only the most recent
        to_keep = set(sid for sid, _ in snapshots[:keep_count])
        to_delete = [sid for sid, _ in snapshots[keep_count:]]

        deleted_count = 0
        for sid in to_delete:
            if sid in self._snapshots:
                del self._snapshots[sid]
                deleted_count += 1

        self._session_snapshots[session_id] = [
            sid for sid in snapshot_ids if sid in to_keep
        ]

        return deleted_count

    def get_all_for_session(self, session_id: str) -> List[PageSnapshot]:
        """Get all snapshots for a session, ordered by creation time."""
        snapshot_ids = self._session_snapshots.get(session_id, [])
        snapshots = [
            self._snapshots[sid] for sid in snapshot_ids
            if sid in self._snapshots
        ]
        snapshots.sort(key=lambda s: s.created_at)
        return snapshots

    def delete_for_session(self, session_id: str) -> int:
        """Delete all snapshots for a session."""
        snapshot_ids = self._session_snapshots.get(session_id, [])
        deleted_count = 0

        for sid in snapshot_ids:
            if sid in self._snapshots:
                del self._snapshots[sid]
                deleted_count += 1

        if session_id in self._session_snapshots:
            del self._session_snapshots[session_id]

        return deleted_count

    def count_for_session(self, session_id: str) -> int:
        """Count snapshots for a session."""
        return len(self._session_snapshots.get(session_id, []))

    def clear(self) -> None:
        """Clear all snapshots (for testing)."""
        self._snapshots.clear()
        self._session_snapshots.clear()

    def __len__(self) -> int:
        """Return total number of snapshots."""
        return len(self._snapshots)

    def __contains__(self, snapshot_id: SnapshotId) -> bool:
        """Check if a snapshot exists."""
        return str(snapshot_id) in self._snapshots

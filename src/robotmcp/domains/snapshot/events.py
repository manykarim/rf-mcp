"""
Domain Events for the Snapshot bounded context.

Domain events represent something that happened in the domain that domain experts
care about. They are used for communication between bounded contexts and for
event sourcing/auditing purposes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from .value_objects import ElementRef, SnapshotId


@dataclass
class SnapshotCaptured:
    """
    Emitted when a new snapshot is captured.

    This event signals that a page's accessibility tree has been
    captured and is available for use by other contexts.
    """
    snapshot_id: SnapshotId
    session_id: str
    node_count: int
    token_estimate: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Optional metadata
    url: Optional[str] = None
    title: Optional[str] = None
    selector: Optional[str] = None  # If snapshot was scoped to a selector

    @property
    def event_type(self) -> str:
        return "snapshot.captured"

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "snapshot_id": str(self.snapshot_id),
            "session_id": self.session_id,
            "node_count": self.node_count,
            "token_estimate": self.token_estimate,
            "timestamp": self.timestamp.isoformat(),
            "url": self.url,
            "title": self.title,
            "selector": self.selector
        }

    def __repr__(self) -> str:
        return (
            f"SnapshotCaptured(id={self.snapshot_id}, session={self.session_id}, "
            f"nodes={self.node_count}, tokens={self.token_estimate})"
        )


@dataclass
class SnapshotDiffComputed:
    """
    Emitted when an incremental diff is computed between two snapshots.

    This event contains information about what changed between
    the previous and current snapshots.
    """
    current_snapshot_id: SnapshotId
    previous_snapshot_id: SnapshotId
    added_refs: List[ElementRef]
    removed_refs: List[ElementRef]
    modified_refs: List[ElementRef] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def event_type(self) -> str:
        return "snapshot.diff_computed"

    @property
    def has_changes(self) -> bool:
        """Check if there were any changes."""
        return bool(self.added_refs or self.removed_refs or self.modified_refs)

    @property
    def change_count(self) -> int:
        """Total number of changes."""
        return len(self.added_refs) + len(self.removed_refs) + len(self.modified_refs)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "current_snapshot_id": str(self.current_snapshot_id),
            "previous_snapshot_id": str(self.previous_snapshot_id),
            "added_refs": [str(ref) for ref in self.added_refs],
            "removed_refs": [str(ref) for ref in self.removed_refs],
            "modified_refs": [str(ref) for ref in self.modified_refs],
            "timestamp": self.timestamp.isoformat()
        }

    def __repr__(self) -> str:
        return (
            f"SnapshotDiffComputed(current={self.current_snapshot_id}, "
            f"previous={self.previous_snapshot_id}, "
            f"added={len(self.added_refs)}, removed={len(self.removed_refs)}, "
            f"modified={len(self.modified_refs)})"
        )


@dataclass
class ListsFolded:
    """
    Emitted when list folding is applied to a snapshot.

    This event contains statistics about the compression achieved
    through list folding.
    """
    snapshot_id: SnapshotId
    lists_folded: int
    items_compressed: int
    compression_ratio: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Detailed statistics
    tokens_before: Optional[int] = None
    tokens_after: Optional[int] = None

    @property
    def event_type(self) -> str:
        return "snapshot.lists_folded"

    @property
    def tokens_saved(self) -> Optional[int]:
        """Calculate tokens saved by folding."""
        if self.tokens_before is not None and self.tokens_after is not None:
            return self.tokens_before - self.tokens_after
        return None

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "snapshot_id": str(self.snapshot_id),
            "lists_folded": self.lists_folded,
            "items_compressed": self.items_compressed,
            "compression_ratio": self.compression_ratio,
            "timestamp": self.timestamp.isoformat(),
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved
        }

    def __repr__(self) -> str:
        return (
            f"ListsFolded(id={self.snapshot_id}, lists={self.lists_folded}, "
            f"items={self.items_compressed}, ratio={self.compression_ratio:.2%})"
        )


@dataclass
class SnapshotExpired:
    """
    Emitted when a snapshot is expired/deleted from the repository.

    This event is useful for cleanup tracking and debugging.
    """
    snapshot_id: SnapshotId
    session_id: str
    reason: str  # "age", "limit", "manual", "session_ended"
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def event_type(self) -> str:
        return "snapshot.expired"

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "snapshot_id": str(self.snapshot_id),
            "session_id": self.session_id,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat()
        }

    def __repr__(self) -> str:
        return (
            f"SnapshotExpired(id={self.snapshot_id}, session={self.session_id}, "
            f"reason={self.reason})"
        )


@dataclass
class SnapshotFormatChanged:
    """
    Emitted when the snapshot format configuration is changed.

    Useful for auditing configuration changes.
    """
    session_id: str
    old_mode: str
    new_mode: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def event_type(self) -> str:
        return "snapshot.format_changed"

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "session_id": self.session_id,
            "old_mode": self.old_mode,
            "new_mode": self.new_mode,
            "timestamp": self.timestamp.isoformat()
        }

    def __repr__(self) -> str:
        return (
            f"SnapshotFormatChanged(session={self.session_id}, "
            f"{self.old_mode} -> {self.new_mode})"
        )

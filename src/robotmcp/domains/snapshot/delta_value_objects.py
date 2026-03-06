"""Delta State Retrieval Value Objects (ADR-018).

Immutable value objects for delta-based session state retrieval.
These VOs represent the core concepts of versioned state tracking
and section-level change detection for token-efficient state updates.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Tuple


class StateRetrievalMode(Enum):
    """Mode for retrieving session state."""
    FULL = "full"
    DELTA = "delta"
    NONE = "none"


class SectionChangeType(Enum):
    """Type of change detected in a state section."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class StateVersion:
    """Immutable version identifier for a state snapshot.

    Tracks version number, timestamp, session, and content hash
    to enable deterministic delta computation.
    """
    version_number: int
    timestamp: datetime
    session_id: str
    content_hash: str

    def __post_init__(self) -> None:
        if self.version_number < 0:
            raise ValueError(
                f"version_number cannot be negative: {self.version_number}"
            )
        if not self.session_id:
            raise ValueError("session_id cannot be empty")

    @classmethod
    def initial(cls, session_id: str, content_hash: str) -> "StateVersion":
        """Create the initial version (version 0) for a session."""
        return cls(
            version_number=0,
            timestamp=datetime.now(),
            session_id=session_id,
            content_hash=content_hash,
        )

    def next(self, content_hash: str) -> "StateVersion":
        """Create the next version with an incremented version number."""
        return StateVersion(
            version_number=self.version_number + 1,
            timestamp=datetime.now(),
            session_id=self.session_id,
            content_hash=content_hash,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version_number,
            "ts": self.timestamp.isoformat(),
            "session": self.session_id,
            "hash": self.content_hash[:8],
        }


@dataclass(frozen=True)
class DeltaSection:
    """A single section within a state delta.

    Represents whether a named section was added, removed,
    modified, or unchanged between two state versions.
    """
    section_name: str
    change_type: SectionChangeType
    value: Optional[Any] = None  # Only set for ADDED/MODIFIED

    def __post_init__(self) -> None:
        if self.change_type in (
            SectionChangeType.ADDED,
            SectionChangeType.MODIFIED,
        ) and self.value is None:
            raise ValueError(
                f"value required for {self.change_type.value} sections"
            )

    @property
    def is_changed(self) -> bool:
        """Return True if this section has any change."""
        return self.change_type != SectionChangeType.UNCHANGED

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "section": self.section_name,
            "change": self.change_type.value,
        }
        if self.value is not None:
            d["value"] = self.value
        return d


@dataclass(frozen=True)
class StateDelta:
    """The complete delta between two state versions.

    Contains all changed and unchanged sections along with
    token estimates for measuring savings.
    """
    from_version: int
    to_version: int
    changed_sections: Tuple[DeltaSection, ...]
    unchanged_sections: Tuple[str, ...]
    delta_token_estimate: int
    full_token_estimate: int

    def __post_init__(self) -> None:
        if self.to_version < self.from_version:
            raise ValueError(
                f"to_version ({self.to_version}) < from_version ({self.from_version})"
            )
        if self.delta_token_estimate < 0:
            raise ValueError("delta_token_estimate cannot be negative")
        if self.full_token_estimate < 0:
            raise ValueError("full_token_estimate cannot be negative")

    @property
    def has_changes(self) -> bool:
        """Return True if any sections changed."""
        return len(self.changed_sections) > 0

    @property
    def change_count(self) -> int:
        """Number of changed sections."""
        return len(self.changed_sections)

    @property
    def saved_tokens(self) -> int:
        """Tokens saved by using delta instead of full state."""
        return max(0, self.full_token_estimate - self.delta_token_estimate)

    @property
    def savings_ratio(self) -> float:
        """Fraction of tokens saved (0.0 to 1.0)."""
        if self.full_token_estimate == 0:
            return 0.0
        return self.saved_tokens / self.full_token_estimate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "changed_sections": [s.to_dict() for s in self.changed_sections],
            "unchanged_sections": list(self.unchanged_sections),
            "token_savings": {
                "delta_tokens": self.delta_token_estimate,
                "full_tokens": self.full_token_estimate,
                "saved_tokens": self.saved_tokens,
                "savings_ratio": round(self.savings_ratio, 3),
            },
        }

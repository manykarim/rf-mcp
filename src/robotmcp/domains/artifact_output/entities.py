"""Artifact Output Domain Entities (ADR-015).

Mutable domain objects with identity, following ADR-001 conventions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from .value_objects import ArtifactId, ArtifactReference


@dataclass
class Artifact:
    """A stored artifact representing externalized tool output."""

    id: ArtifactId
    reference: ArtifactReference
    tool_name: str
    field_name: str
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: Optional[datetime] = None

    __test__ = False  # Suppress pytest collection

    def record_access(self) -> None:
        """Update last-accessed timestamp."""
        self.accessed_at = datetime.now()

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check whether this artifact has exceeded its TTL."""
        age = (datetime.now() - self.created_at).total_seconds()
        return age > ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses."""
        return {
            "id": str(self.id),
            "tool": self.tool_name,
            "field": self.field_name,
            "session": self.session_id,
            "ref": self.reference.to_inline_dict(),
            "created": self.created_at.isoformat(),
            "accessed": (
                self.accessed_at.isoformat() if self.accessed_at else None
            ),
        }


@dataclass(frozen=True)
class ArtifactSlice:
    """A paginated slice of artifact content."""

    artifact_id: str
    offset: int
    limit: int
    content: str
    total_size: int

    __test__ = False  # Suppress pytest collection

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("offset cannot be negative")
        if self.limit < 0:
            raise ValueError("limit cannot be negative")

    @property
    def end_offset(self) -> int:
        """Byte offset after this slice."""
        return self.offset + len(self.content)

    @property
    def has_more(self) -> bool:
        """Whether there is more content beyond this slice."""
        return self.end_offset < self.total_size

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses."""
        return {
            "artifact_id": self.artifact_id,
            "offset": self.offset,
            "length": len(self.content),
            "total_size": self.total_size,
            "has_more": self.has_more,
            "content": self.content,
        }

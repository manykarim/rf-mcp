"""Instruction Domain Entities.

This module contains entities for the Instruction bounded context.
Entities have identity and lifecycle, unlike value objects.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .value_objects import InstructionContent


@dataclass
class InstructionVersion:
    """Tracks instruction versions for learning and analytics.

    Each time instructions are modified or loaded from a different
    source, a new version is created. This enables:
    - Tracking which instructions were in effect during errors
    - A/B testing of different instruction strategies
    - Rollback to previous known-good instructions

    Attributes:
        version_id: Unique identifier for this version.
        content_hash: SHA256 hash of content (first 16 chars).
        source: Where the content came from (default, custom:path, template:id).
        created_at: When this version was created.
        session_id: Optional session this version is associated with.
        metadata: Additional tracking information.
        application_count: Number of times this version was applied.
        success_rate: Running average of success rate.

    Examples:
        >>> from .value_objects import InstructionContent
        >>> content = InstructionContent("Use discovery tools first.", "default")
        >>> version = InstructionVersion.create(content)
        >>> version.record_application(success=True)
        >>> version.success_rate
        1.0
    """

    version_id: str
    content_hash: str
    source: str
    created_at: datetime
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Tracking metrics
    application_count: int = 0
    success_rate: Optional[float] = None

    @classmethod
    def create(
        cls,
        content: "InstructionContent",
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "InstructionVersion":
        """Create a new version from instruction content.

        Args:
            content: The instruction content to version.
            session_id: Optional session identifier.
            metadata: Optional additional metadata.

        Returns:
            New InstructionVersion instance.
        """
        content_hash = hashlib.sha256(content.value.encode()).hexdigest()[:16]

        return cls(
            version_id=str(uuid.uuid4()),
            content_hash=content_hash,
            source=content.source,
            created_at=datetime.now(),
            session_id=session_id,
            metadata=metadata or {},
        )

    def record_application(self, success: bool) -> None:
        """Record an instruction application outcome.

        Updates the application count and running success rate.

        Args:
            success: Whether the application was successful.
        """
        self.application_count += 1
        if self.success_rate is None:
            self.success_rate = 1.0 if success else 0.0
        else:
            # Running average
            self.success_rate = (
                self.success_rate * (self.application_count - 1)
                + (1.0 if success else 0.0)
            ) / self.application_count

    @property
    def is_effective(self) -> bool:
        """Check if this version has been applied at least once."""
        return self.application_count > 0

    @property
    def has_good_success_rate(self) -> bool:
        """Check if success rate is above threshold (80%)."""
        if self.success_rate is None:
            return True  # No data, assume good
        return self.success_rate >= 0.8

    @property
    def age_seconds(self) -> float:
        """Get age of this version in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    def with_metadata(self, key: str, value: Any) -> "InstructionVersion":
        """Create a new version with additional metadata.

        Args:
            key: Metadata key.
            value: Metadata value.

        Returns:
            New InstructionVersion with updated metadata.
        """
        new_metadata = dict(self.metadata)
        new_metadata[key] = value

        return InstructionVersion(
            version_id=self.version_id,
            content_hash=self.content_hash,
            source=self.source,
            created_at=self.created_at,
            session_id=self.session_id,
            metadata=new_metadata,
            application_count=self.application_count,
            success_rate=self.success_rate,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all version data.
        """
        return {
            "version_id": self.version_id,
            "content_hash": self.content_hash,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "session_id": self.session_id,
            "metadata": self.metadata,
            "application_count": self.application_count,
            "success_rate": self.success_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstructionVersion":
        """Create from dictionary representation.

        Args:
            data: Dictionary with version data.

        Returns:
            InstructionVersion instance.
        """
        return cls(
            version_id=data["version_id"],
            content_hash=data["content_hash"],
            source=data["source"],
            created_at=datetime.fromisoformat(data["created_at"]),
            session_id=data.get("session_id"),
            metadata=data.get("metadata", {}),
            application_count=data.get("application_count", 0),
            success_rate=data.get("success_rate"),
        )

    def __str__(self) -> str:
        return f"Version({self.version_id[:8]})"

    def __repr__(self) -> str:
        return (
            f"InstructionVersion("
            f"id={self.version_id[:8]!r}, "
            f"hash={self.content_hash!r}, "
            f"source={self.source!r}, "
            f"applications={self.application_count})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InstructionVersion):
            return NotImplemented
        return self.version_id == other.version_id

    def __hash__(self) -> int:
        return hash(self.version_id)

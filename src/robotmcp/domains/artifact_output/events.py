"""Artifact Output Domain Events (ADR-015).

Immutable event records following ADR-001 conventions.
All events expose to_dict() for serialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass(frozen=True)
class ArtifactCreated:
    """Raised when a new artifact is stored."""

    artifact_id: str
    tool_name: str
    field_name: str
    session_id: str
    byte_size: int
    token_estimate: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "ArtifactCreated",
            "artifact_id": self.artifact_id,
            "tool": self.tool_name,
            "field": self.field_name,
            "session": self.session_id,
            "size": self.byte_size,
            "tokens": self.token_estimate,
            "ts": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class ArtifactAccessed:
    """Raised when an artifact is read."""

    artifact_id: str
    tool_name: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "ArtifactAccessed",
            "artifact_id": self.artifact_id,
            "tool": self.tool_name,
            "ts": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class ArtifactExpired:
    """Raised when an artifact is evicted due to TTL."""

    artifact_id: str
    tool_name: str
    session_id: str
    age_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "ArtifactExpired",
            "artifact_id": self.artifact_id,
            "tool": self.tool_name,
            "session": self.session_id,
            "age_s": round(self.age_seconds, 1),
            "ts": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class LargeFieldExternalized:
    """Raised when a tool response field is replaced with an artifact ref."""

    tool_name: str
    field_path: str
    artifact_id: str
    original_tokens: int
    saved_tokens: int
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "LargeFieldExternalized",
            "tool": self.tool_name,
            "field": self.field_path,
            "artifact_id": self.artifact_id,
            "original_tokens": self.original_tokens,
            "saved_tokens": self.saved_tokens,
            "ts": self.timestamp.isoformat(),
        }

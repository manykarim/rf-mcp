"""Artifact Output Domain Value Objects (ADR-015).

Immutable types identified by their attributes, following ADR-001 conventions.
All use @dataclass(frozen=True) with __post_init__ validation.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Dict, Optional, Tuple


class OutputMode(Enum):
    """Controls how large tool outputs are handled."""

    INLINE = "inline"
    FILE = "file"
    AUTO = "auto"

    @classmethod
    def from_env(cls) -> "OutputMode":
        """Read output mode from ROBOTMCP_OUTPUT_MODE env var."""
        val = os.getenv("ROBOTMCP_OUTPUT_MODE", "auto").lower().strip()
        try:
            return cls(val)
        except ValueError:
            return cls.AUTO


@dataclass(frozen=True)
class ArtifactId:
    """Unique identifier for an artifact, format: art_{12 hex chars}."""

    value: str
    PATTERN: ClassVar[str] = r"^art_[a-f0-9]{12}$"

    def __post_init__(self) -> None:
        if not re.match(self.PATTERN, self.value):
            raise ValueError(
                f"Invalid ArtifactId format: '{self.value}'. "
                "Must match pattern 'art_{12 hex chars}'"
            )

    @classmethod
    def generate(cls) -> "ArtifactId":
        """Generate a new random ArtifactId."""
        return cls(value=f"art_{uuid.uuid4().hex[:12]}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ArtifactReference:
    """Lightweight reference to an externalized artifact."""

    artifact_id: str
    file_path: str
    content_hash: str
    byte_size: int
    token_estimate: int
    mime_type: str
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if self.byte_size < 0:
            raise ValueError("byte_size cannot be negative")
        if self.token_estimate < 0:
            raise ValueError("token_estimate cannot be negative")

    def to_inline_dict(self) -> Dict[str, Any]:
        """Compact dict for embedding in tool responses."""
        return {
            "artifact_id": self.artifact_id,
            "file": self.file_path,
            "size": self.byte_size,
            "tokens": self.token_estimate,
            "hash": self.content_hash[:8],
        }


@dataclass(frozen=True)
class ArtifactPolicy:
    """Configuration for artifact externalization behavior."""

    max_inline_tokens: int = 500
    artifact_dir: str = ".robotmcp_artifacts"
    retention_ttl_seconds: int = 3600
    max_artifacts: int = 100

    CHARS_PER_TOKEN: ClassVar[float] = 4.0

    def __post_init__(self) -> None:
        if self.max_inline_tokens < 0:
            raise ValueError("max_inline_tokens cannot be negative")
        if self.retention_ttl_seconds < 0:
            raise ValueError("retention_ttl_seconds cannot be negative")
        if self.max_artifacts < 1:
            raise ValueError("max_artifacts must be >= 1")

    @classmethod
    def from_env(cls) -> "ArtifactPolicy":
        """Build policy from environment variables."""
        return cls(
            max_inline_tokens=int(
                os.getenv("ROBOTMCP_MAX_INLINE_TOKENS", "500")
            ),
            artifact_dir=os.getenv(
                "ROBOTMCP_ARTIFACT_DIR", ".robotmcp_artifacts"
            ),
            retention_ttl_seconds=int(
                os.getenv("ROBOTMCP_ARTIFACT_TTL", "3600")
            ),
        )

    def should_externalize(self, content: str) -> bool:
        """Return True if content exceeds inline token threshold."""
        return len(content) / self.CHARS_PER_TOKEN > self.max_inline_tokens


@dataclass(frozen=True)
class ExternalizationRule:
    """Maps a tool+field to externalization behavior."""

    tool_name: str
    field_path: str
    summary_template: str = (
        "Content externalized to artifact {artifact_id} "
        "({byte_size} bytes, ~{token_estimate} tokens). "
        "Use fetch_artifact to retrieve."
    )

    def __post_init__(self) -> None:
        if not self.tool_name:
            raise ValueError("tool_name cannot be empty")
        if not self.field_path:
            raise ValueError("field_path cannot be empty")


@dataclass(frozen=True)
class ExternalizationResult:
    """Outcome of an externalization attempt."""

    summary: str
    artifact_ref: Optional[ArtifactReference]
    original_token_estimate: int
    saved_tokens: int

    def __post_init__(self) -> None:
        if self.saved_tokens < 0:
            raise ValueError("saved_tokens cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging / diagnostics."""
        d: Dict[str, Any] = {
            "summary": self.summary,
            "original_tokens": self.original_token_estimate,
            "saved_tokens": self.saved_tokens,
        }
        if self.artifact_ref:
            d["artifact"] = self.artifact_ref.to_inline_dict()
        return d

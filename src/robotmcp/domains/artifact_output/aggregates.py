"""Artifact Output Domain Aggregates (ADR-015).

Root aggregate for artifact lifecycle management.
"""

from __future__ import annotations

import hashlib
import logging
import pathlib
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .entities import Artifact, ArtifactSlice
from .events import ArtifactCreated, ArtifactExpired
from .value_objects import ArtifactId, ArtifactPolicy, ArtifactReference

logger = logging.getLogger(__name__)

_MIME_TO_EXT: Dict[str, str] = {
    "text/html": ".html",
    "application/json": ".json",
    "text/x-robot": ".robot",
    "text/plain": ".txt",
}


@dataclass
class ArtifactStore:
    """In-memory store for externalized artifacts with LRU eviction."""

    policy: ArtifactPolicy
    _artifacts: OrderedDict = field(default_factory=OrderedDict)
    _session_index: Dict[str, List[str]] = field(default_factory=dict)
    _pending_events: List = field(default_factory=list)

    __test__ = False  # Suppress pytest collection

    @classmethod
    def create(cls, policy: Optional[ArtifactPolicy] = None) -> "ArtifactStore":
        """Factory with optional custom policy."""
        policy = policy or ArtifactPolicy()
        # Resolve relative artifact_dir to absolute path
        art_dir = pathlib.Path(policy.artifact_dir)
        if not art_dir.is_absolute():
            art_dir = pathlib.Path.cwd() / art_dir
        policy = ArtifactPolicy(
            max_inline_tokens=policy.max_inline_tokens,
            artifact_dir=str(art_dir),
            retention_ttl_seconds=policy.retention_ttl_seconds,
            max_artifacts=policy.max_artifacts,
        )
        return cls(policy=policy)

    def create_artifact(
        self,
        content: str,
        tool_name: str,
        field_name: str,
        session_id: str,
        mime_type: str = "text/plain",
    ) -> Artifact:
        """Store content and return the artifact handle."""
        self._evict_if_full()
        art_id = ArtifactId.generate()
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        ext = _MIME_TO_EXT.get(mime_type, ".txt")
        file_path_str = f"{self.policy.artifact_dir}/{art_id.value}{ext}"
        ref = ArtifactReference(
            artifact_id=str(art_id),
            file_path=file_path_str,
            content_hash=content_hash,
            byte_size=len(content.encode()),
            token_estimate=len(content) // 4,
            mime_type=mime_type,
        )
        artifact = Artifact(
            id=art_id,
            reference=ref,
            tool_name=tool_name,
            field_name=field_name,
            session_id=session_id,
        )
        self._artifacts[str(art_id)] = (artifact, content)
        self._session_index.setdefault(session_id, []).append(str(art_id))
        # Write content to disk so agents can read via file path
        self._write_to_disk(file_path_str, content)
        self._pending_events.append(
            ArtifactCreated(
                artifact_id=str(art_id),
                tool_name=tool_name,
                field_name=field_name,
                session_id=session_id,
                byte_size=ref.byte_size,
                token_estimate=ref.token_estimate,
            )
        )
        return artifact

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Retrieve artifact by id, returning None if missing or expired."""
        entry = self._artifacts.get(artifact_id)
        if entry is None:
            return None
        artifact, _ = entry
        if artifact.is_expired(self.policy.retention_ttl_seconds):
            self._remove_artifact(artifact_id)
            return None
        artifact.record_access()
        return artifact

    def read_content(self, artifact_id: str) -> Optional[str]:
        """Return raw content for an artifact."""
        entry = self._artifacts.get(artifact_id)
        if entry is None:
            return None
        _, content = entry
        return content

    def read_slice(
        self, artifact_id: str, offset: int = 0, limit: int = 4000
    ) -> Optional[ArtifactSlice]:
        """Return a paginated slice of artifact content."""
        content = self.read_content(artifact_id)
        if content is None:
            return None
        sliced = content[offset : offset + limit]
        return ArtifactSlice(
            artifact_id=artifact_id,
            offset=offset,
            limit=limit,
            content=sliced,
            total_size=len(content),
        )

    def cleanup_expired(self) -> int:
        """Remove all expired artifacts, return count removed."""
        expired = [
            aid
            for aid, (art, _) in self._artifacts.items()
            if art.is_expired(self.policy.retention_ttl_seconds)
        ]
        for aid in expired:
            self._remove_artifact(aid)
        return len(expired)

    def cleanup_session(self, session_id: str) -> int:
        """Remove all artifacts for a session, return count removed."""
        aids = self._session_index.pop(session_id, [])
        for aid in aids:
            entry = self._artifacts.pop(aid, None)
            if entry:
                artifact, _ = entry
                self._delete_from_disk(artifact.reference.file_path)
        return len(aids)

    def list_artifacts(
        self, session_id: Optional[str] = None
    ) -> List[Artifact]:
        """List artifacts, optionally filtered by session."""
        if session_id:
            aids = self._session_index.get(session_id, [])
            return [
                self._artifacts[a][0]
                for a in aids
                if a in self._artifacts
            ]
        return [art for art, _ in self._artifacts.values()]

    def _evict_if_full(self) -> None:
        """Evict oldest artifacts when at capacity."""
        while len(self._artifacts) >= self.policy.max_artifacts:
            oldest_id = next(iter(self._artifacts))
            self._remove_artifact(oldest_id)

    def _remove_artifact(self, artifact_id: str) -> None:
        """Remove an artifact and emit expiry event."""
        entry = self._artifacts.pop(artifact_id, None)
        if entry:
            artifact, _ = entry
            self._delete_from_disk(artifact.reference.file_path)
            for aids in self._session_index.values():
                if artifact_id in aids:
                    aids.remove(artifact_id)
            self._pending_events.append(
                ArtifactExpired(
                    artifact_id=artifact_id,
                    tool_name=artifact.tool_name,
                    session_id=artifact.session_id,
                    age_seconds=(
                        datetime.now() - artifact.created_at
                    ).total_seconds(),
                )
            )

    @staticmethod
    def _write_to_disk(file_path: str, content: str) -> None:
        """Write artifact content to disk. Never raises."""
        try:
            p = pathlib.Path(file_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except Exception as exc:
            logger.debug("Failed to write artifact to %s: %s", file_path, exc)

    @staticmethod
    def _delete_from_disk(file_path: str) -> None:
        """Delete artifact file from disk. Never raises."""
        try:
            p = pathlib.Path(file_path)
            if p.exists():
                p.unlink(missing_ok=True)
        except Exception as exc:
            logger.debug("Failed to delete artifact %s: %s", file_path, exc)

    def drain_events(self) -> List:
        """Return and clear all pending domain events."""
        events = self._pending_events.copy()
        self._pending_events.clear()
        return events

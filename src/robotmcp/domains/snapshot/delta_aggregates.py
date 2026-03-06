"""Delta State Retrieval Aggregates (ADR-018).

VersionedStateCache is the aggregate root for per-session
versioned state management with LRU eviction and TTL expiry.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .delta_value_objects import (
    DeltaSection,
    SectionChangeType,
    StateDelta,
    StateVersion,
)
from .delta_entities import VersionedSnapshot


@dataclass
class VersionedStateCache:
    """Per-session cache of versioned state snapshots.

    Maintains an ordered history of state versions with LRU eviction
    and TTL-based expiry. Computes section-level deltas by comparing
    per-section content hashes between versions.
    """
    session_id: str
    max_versions: int = 5
    ttl_seconds: float = 300.0
    _versions: OrderedDict = field(default_factory=OrderedDict)
    _version_counter: int = field(default=0)
    _pending_events: List = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_versions < 1:
            raise ValueError("max_versions must be >= 1")
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")

    def store_version(self, sections: Dict[str, Any]) -> StateVersion:
        """Store a new state version and return its StateVersion.

        Computes content hash, creates a VersionedSnapshot, and
        applies LRU eviction if the cache exceeds max_versions.
        """
        content_hash = hashlib.md5(
            json.dumps(sections, default=str, sort_keys=True).encode()
        ).hexdigest()

        if self._version_counter == 0:
            version = StateVersion.initial(self.session_id, content_hash)
        else:
            latest = self.get_latest()
            if latest is not None:
                version = latest.version.next(content_hash)
            else:
                version = StateVersion.initial(self.session_id, content_hash)

        snapshot = VersionedSnapshot.create(version, sections)
        self._versions[version.version_number] = snapshot
        self._version_counter = version.version_number + 1

        # LRU eviction
        while len(self._versions) > self.max_versions:
            oldest_key = next(iter(self._versions))
            self._versions.pop(oldest_key)

        return version

    def get_version(self, version_number: int) -> Optional[VersionedSnapshot]:
        """Get a specific version, returning None if expired or absent."""
        snap = self._versions.get(version_number)
        if snap is None:
            return None
        # TTL check
        age = (datetime.now() - snap.created_at).total_seconds()
        if age > self.ttl_seconds:
            self._versions.pop(version_number, None)
            return None
        return snap

    def get_latest(self) -> Optional[VersionedSnapshot]:
        """Get the most recent version, or None if cache is empty."""
        if not self._versions:
            return None
        latest_key = max(self._versions.keys())
        return self._versions[latest_key]

    def get_current_version_number(self) -> int:
        """Get the highest stored version number, or 0 if empty."""
        if not self._versions:
            return 0
        return max(self._versions.keys())

    def compute_delta(
        self, since_version: int, current_sections: Dict[str, Any]
    ) -> Tuple[StateDelta, StateVersion]:
        """Compute a delta from since_version to the current sections.

        Always stores the current sections as a new version. If
        since_version is not available (expired or evicted), all
        sections are reported as ADDED (fallback to full).

        Returns:
            Tuple of (StateDelta, new StateVersion).
        """
        # Always store current state
        new_version = self.store_version(current_sections)
        new_snap = self._versions[new_version.version_number]
        old_snap = self.get_version(since_version)

        if old_snap is None:
            # Fallback: all sections are ADDED.
            # Clamp since_version so that from_version <= to_version
            # (the requested version was unavailable, so we report
            # from the new version to itself).
            effective_from = min(since_version, new_version.version_number)
            changed = tuple(
                DeltaSection(
                    section_name=name,
                    change_type=SectionChangeType.ADDED,
                    value=value,
                )
                for name, value in current_sections.items()
            )
            full_tokens = new_snap.estimate_tokens()
            return (
                StateDelta(
                    from_version=effective_from,
                    to_version=new_version.version_number,
                    changed_sections=changed,
                    unchanged_sections=(),
                    delta_token_estimate=full_tokens,
                    full_token_estimate=full_tokens,
                ),
                new_version,
            )

        # Compare section hashes
        changed: List[DeltaSection] = []
        unchanged: List[str] = []
        all_sections = set(
            list(new_snap.section_hashes.keys())
            + list(old_snap.section_hashes.keys())
        )

        for section_name in sorted(all_sections):
            old_hash = old_snap.get_section_hash(section_name)
            new_hash = new_snap.get_section_hash(section_name)

            if old_hash is None and new_hash is not None:
                changed.append(
                    DeltaSection(
                        section_name=section_name,
                        change_type=SectionChangeType.ADDED,
                        value=new_snap.get_section(section_name),
                    )
                )
            elif old_hash is not None and new_hash is None:
                changed.append(
                    DeltaSection(
                        section_name=section_name,
                        change_type=SectionChangeType.REMOVED,
                    )
                )
            elif old_hash != new_hash:
                changed.append(
                    DeltaSection(
                        section_name=section_name,
                        change_type=SectionChangeType.MODIFIED,
                        value=new_snap.get_section(section_name),
                    )
                )
            else:
                unchanged.append(section_name)

        full_tokens = new_snap.estimate_tokens()
        delta_tokens = sum(
            len(json.dumps(s.value, default=str)) // 4
            for s in changed
            if s.value is not None
        )
        delta_tokens += 45  # overhead for delta structure

        return (
            StateDelta(
                from_version=since_version,
                to_version=new_version.version_number,
                changed_sections=tuple(changed),
                unchanged_sections=tuple(unchanged),
                delta_token_estimate=delta_tokens,
                full_token_estimate=full_tokens,
            ),
            new_version,
        )

    def clear(self) -> None:
        """Clear all stored versions."""
        self._versions.clear()
        self._version_counter = 0

    def drain_events(self) -> List:
        """Drain and return all pending domain events."""
        events = self._pending_events.copy()
        self._pending_events.clear()
        return events

"""Delta State Retrieval Entities (ADR-018).

Entities with identity for versioned state snapshots.
VersionedSnapshot tracks a specific version of session state
with per-section content hashing for efficient delta detection.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from .delta_value_objects import StateVersion


@dataclass
class VersionedSnapshot:
    """A versioned snapshot of session state sections.

    Stores section data alongside per-section content hashes
    to enable O(n) delta computation via hash comparison.
    """
    version: StateVersion
    section_data: Dict[str, Any]
    section_hashes: Dict[str, str]
    created_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def create(
        cls, version: StateVersion, sections: Dict[str, Any]
    ) -> "VersionedSnapshot":
        """Create a new snapshot, computing hashes for each section."""
        hashes: Dict[str, str] = {}
        for name, value in sections.items():
            if isinstance(value, str):
                content = value
            else:
                content = json.dumps(value, default=str, sort_keys=True)
            hashes[name] = hashlib.md5(content.encode()).hexdigest()
        return cls(
            version=version,
            section_data=sections,
            section_hashes=hashes,
        )

    def get_section_hash(self, section_name: str) -> Optional[str]:
        """Get the content hash for a section, or None if absent."""
        return self.section_hashes.get(section_name)

    def has_section(self, section_name: str) -> bool:
        """Check whether a section exists in this snapshot."""
        return section_name in self.section_data

    def get_section(self, section_name: str) -> Optional[Any]:
        """Get the value of a section, or None if absent."""
        return self.section_data.get(section_name)

    def estimate_tokens(self) -> int:
        """Estimate total tokens across all sections (~4 chars/token)."""
        total_chars = 0
        for v in self.section_data.values():
            if isinstance(v, str):
                total_chars += len(v)
            else:
                total_chars += len(json.dumps(v, default=str))
        return total_chars // 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version.to_dict(),
            "sections": list(self.section_data.keys()),
            "tokens": self.estimate_tokens(),
        }

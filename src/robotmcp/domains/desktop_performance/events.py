"""Domain events for the desktop_performance bounded context.

All events are frozen dataclasses with ``to_dict()`` following the
project convention from ADR-001/011.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ElementCached:
    """Emitted when a desktop element is cached after first resolution."""

    session_id: str
    xpath: str
    application_name: Optional[str]
    resolution_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "element_cached",
            "session_id": self.session_id,
            "xpath": self.xpath,
            "application_name": self.application_name,
            "resolution_time_ms": round(self.resolution_time_ms, 2),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class CacheHit:
    """Emitted when a cached element is reused."""

    session_id: str
    xpath: str
    hit_count: int
    saved_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "cache_hit",
            "session_id": self.session_id,
            "xpath": self.xpath,
            "hit_count": self.hit_count,
            "saved_time_ms": round(self.saved_time_ms, 2),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class ElementInvalidated:
    """Emitted when a cached element is invalidated."""

    session_id: str
    xpath: str
    reason: str  # "stale_node" | "ttl_expired" | "explicit_clear" | "capacity_eviction"
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "element_invalidated",
            "session_id": self.session_id,
            "xpath": self.xpath,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class ApplicationScopeSet:
    """Emitted when an application root is discovered and set."""

    session_id: str
    application_name: str
    runtime_id: str
    descendant_count: Optional[int]
    discovery_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "application_scope_set",
            "session_id": self.session_id,
            "application_name": self.application_name,
            "runtime_id": self.runtime_id,
            "descendant_count": self.descendant_count,
            "discovery_time_ms": round(self.discovery_time_ms, 2),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class XPathTransformed:
    """Emitted when an absolute XPath is rewritten to relative."""

    session_id: str
    original_xpath: str
    relative_xpath: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "xpath_transformed",
            "session_id": self.session_id,
            "original_xpath": self.original_xpath,
            "relative_xpath": self.relative_xpath,
            "timestamp": self.timestamp.isoformat(),
        }

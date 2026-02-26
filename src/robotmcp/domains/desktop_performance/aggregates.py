"""Aggregates for the desktop_performance bounded context.

Aggregates enforce invariants and encapsulate domain logic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .entities import ApplicationRoot, CachedElement
from .value_objects import (
    CacheCapacity,
    CacheKey,
    CacheTTL,
    InteractionSpeed,
    PointerSpeedProfile,
    SPEED_PROFILES,
    XPathAxis,
    XPathTransform,
)

_monotonic = time.monotonic


# ---------------------------------------------------------------------------
# ElementCache (Aggregate Root)
# ---------------------------------------------------------------------------


@dataclass
class ElementCache:
    """Session-scoped cache of resolved desktop UI elements.

    Invariants:
    - Maximum capacity entries (LRU eviction on overflow)
    - All entries validated via is_valid() before return
    - TTL-based expiration as safety net
    - Fully isolated per session_id
    """

    __test__ = False

    session_id: str
    capacity: CacheCapacity = field(default_factory=CacheCapacity.default)
    ttl: CacheTTL = field(default_factory=CacheTTL.default)
    _entries: Dict[str, CachedElement] = field(default_factory=dict)
    _access_order: List[str] = field(default_factory=list)  # LRU tracking
    _stats: Dict[str, int] = field(
        default_factory=lambda: {
            "hits": 0,
            "misses": 0,
            "invalidations": 0,
            "evictions": 0,
        }
    )

    def lookup(self, cache_key: CacheKey) -> Optional[CachedElement]:
        """Look up a cached element, returning None on miss or invalid."""
        key_str = cache_key.key_str
        entry = self._entries.get(key_str)
        if entry is None:
            self._stats["misses"] += 1
            return None
        # TTL check
        if self.ttl.is_expired(entry.created_at, _monotonic()):
            self._remove(key_str)
            self._stats["invalidations"] += 1
            return None
        # Validity check (AT-SPI node still exists)
        if not entry.is_valid():
            self._remove(key_str)
            self._stats["invalidations"] += 1
            return None
        # Cache hit
        entry.record_hit()
        self._promote(key_str)
        self._stats["hits"] += 1
        return entry

    def store(self, element: CachedElement) -> None:
        """Store a resolved element, evicting LRU if at capacity."""
        key_str = element.cache_key.key_str
        if key_str in self._entries:
            self._entries[key_str] = element
            self._promote(key_str)
            return
        # Evict LRU if at capacity
        while len(self._entries) >= self.capacity.max_entries:
            self._evict_lru()
        self._entries[key_str] = element
        self._access_order.append(key_str)

    def invalidate_all(self) -> int:
        """Clear all cached entries. Returns count of evicted entries."""
        count = len(self._entries)
        self._entries.clear()
        self._access_order.clear()
        self._stats["invalidations"] += count
        return count

    def invalidate_by_prefix(self, xpath_prefix: str) -> int:
        """Invalidate entries matching an XPath prefix."""
        to_remove = [
            k
            for k, v in self._entries.items()
            if v.xpath_original.startswith(xpath_prefix)
        ]
        for k in to_remove:
            self._remove(k)
        self._stats["invalidations"] += len(to_remove)
        return len(to_remove)

    @property
    def size(self) -> int:
        return len(self._entries)

    def get_stats(self) -> Dict[str, Any]:
        total = self._stats["hits"] + self._stats["misses"]
        return {
            **self._stats,
            "size": self.size,
            "hit_rate": round(self._stats["hits"] / total, 4) if total > 0 else 0.0,
        }

    def to_response_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "size": self.size,
            "capacity": self.capacity.max_entries,
            "ttl_seconds": self.ttl.value_seconds,
            "stats": self.get_stats(),
            "entries": [e.to_dict() for e in self._entries.values()],
        }

    # --- Private helpers ---
    def _remove(self, key_str: str) -> None:
        self._entries.pop(key_str, None)
        try:
            self._access_order.remove(key_str)
        except ValueError:
            pass

    def _promote(self, key_str: str) -> None:
        try:
            self._access_order.remove(key_str)
        except ValueError:
            pass
        self._access_order.append(key_str)

    def _evict_lru(self) -> None:
        if self._access_order:
            oldest = self._access_order.pop(0)
            self._entries.pop(oldest, None)
            self._stats["evictions"] += 1


# ---------------------------------------------------------------------------
# ApplicationScope (Aggregate Root)
# ---------------------------------------------------------------------------


@dataclass
class ApplicationScope:
    """Manages the application-scoped root node for a desktop session.

    Lifecycle:
    1. Unscoped (no root) - queries traverse full desktop tree
    2. Discovering - finding application frame via absolute XPath
    3. Scoped - all queries use relative XPath with app root
    4. Invalidated - root node became invalid, needs re-discovery
    """

    __test__ = False

    session_id: str
    root: Optional[ApplicationRoot] = None
    _discovery_xpath: Optional[str] = None
    _state: str = "unscoped"  # unscoped | discovering | scoped | invalidated

    def __post_init__(self) -> None:
        if self.root is not None:
            self._state = "scoped"

    @classmethod
    def create(cls, session_id: str) -> "ApplicationScope":
        return cls(session_id=session_id)

    def set_discovery_xpath(self, xpath: str) -> None:
        """Set the XPath expression used to discover the application root."""
        self._discovery_xpath = xpath
        self._state = "discovering"

    def set_root(self, root: ApplicationRoot) -> None:
        """Set the discovered application root."""
        self.root = root
        self._state = "scoped"

    def invalidate(self) -> None:
        """Mark the scope as needing re-discovery."""
        self.root = None
        self._state = "invalidated"

    @property
    def is_scoped(self) -> bool:
        if self._state != "scoped" or self.root is None:
            return False
        return self.root.is_valid()

    @property
    def state(self) -> str:
        return self._state

    @property
    def discovery_xpath(self) -> Optional[str]:
        return self._discovery_xpath

    def transform_xpath(self, xpath: str) -> XPathTransform:
        """Transform an absolute XPath to relative if scope is active.

        When the XPath contains the scoped Frame/Window prefix, strips it
        to avoid redundant traversal (e.g., searching for Frame inside Frame).
        """
        if not self.is_scoped:
            return XPathTransform.identity(xpath)

        # Strip the scope's discovery prefix if it appears in the xpath
        # e.g. //control:Frame[@Name="Calculator"]//control:Button[@Name="2"]
        # becomes .//control:Button[@Name="2"] when scoped to Calculator
        if self._discovery_xpath and xpath.startswith(self._discovery_xpath):
            remainder = xpath[len(self._discovery_xpath):]
            if remainder.startswith("//"):
                return XPathTransform.to_relative(remainder)
            elif remainder.startswith("/"):
                # Single-slash descendant path
                return XPathTransform.to_relative("/" + remainder)

        return XPathTransform.to_relative(xpath)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "state": self._state,
            "discovery_xpath": self._discovery_xpath,
            "root": self.root.to_dict() if self.root else None,
        }


# ---------------------------------------------------------------------------
# InteractionProfile (Aggregate)
# ---------------------------------------------------------------------------


@dataclass
class InteractionProfile:
    """Session-level interaction speed configuration."""

    __test__ = False

    session_id: str
    pointer_profile: PointerSpeedProfile = field(
        default_factory=lambda: SPEED_PROFILES[InteractionSpeed.FAST]
    )

    @classmethod
    def create(
        cls,
        session_id: str,
        speed: InteractionSpeed = InteractionSpeed.FAST,
    ) -> "InteractionProfile":
        return cls(
            session_id=session_id,
            pointer_profile=SPEED_PROFILES.get(
                speed, SPEED_PROFILES[InteractionSpeed.FAST]
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "speed": self.pointer_profile.speed.value,
            "pointer_overrides": self.pointer_profile.to_overrides_dict(),
        }

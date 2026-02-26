"""Entities for the desktop_performance bounded context.

Entities are mutable objects with identity and lifecycle. They use
``__test__ = False`` to suppress pytest collection warnings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .value_objects import CacheKey

_monotonic = time.monotonic


@dataclass
class CachedElement:
    """A cached desktop UI element with its resolved node reference.

    Lifecycle:
    1. Created after first successful XPath resolution
    2. Reused on cache hits (verified via is_valid())
    3. Invalidated when is_valid() returns False or TTL expires
    4. Evicted on explicit clear or capacity overflow (LRU)
    """

    __test__ = False

    cache_key: CacheKey
    xpath_original: str
    xpath_relative: str  # The .// scoped form used for resolution
    # Note: UiNodeDescriptor and UiNode are PlatynUI types, stored as Any
    # to avoid hard dependency on platynui_native at import time
    descriptor: Any  # UiNodeDescriptor with cached .node
    resolved_node: Any  # UiNode from platynui_native
    application_root_id: Optional[str]  # RuntimeId of the app root used
    created_at: float  # time.monotonic()
    last_accessed: float  # time.monotonic(), updated on each hit
    hit_count: int = 0

    def is_valid(self) -> bool:
        """Check if the cached node is still valid in the AT-SPI tree."""
        if self.resolved_node is None:
            return False
        try:
            return self.resolved_node.is_valid()
        except Exception:
            return False

    def record_hit(self) -> None:
        """Record a cache hit, updating access time and count."""
        self.last_accessed = _monotonic()
        self.hit_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "xpath": self.xpath_original,
            "xpath_relative": self.xpath_relative,
            "application_root_id": self.application_root_id,
            "is_valid": self.is_valid(),
            "hit_count": self.hit_count,
            "age_seconds": round(_monotonic() - self.created_at, 2),
        }


@dataclass
class ApplicationRoot:
    """A cached reference to an application's top-level AT-SPI node.

    Used as the root parameter for relative XPath evaluation,
    scoping queries from 11,452 desktop nodes to ~50-200 app nodes.
    """

    __test__ = False

    session_id: str
    application_name: str  # e.g., "Calculator", "Firefox"
    root_node: Any  # UiNode from platynui_native
    root_runtime_id: str  # RuntimeId for cache key matching
    descendant_count: Optional[int] = None  # Populated on first tree walk
    discovered_at: float = field(default_factory=_monotonic)

    def is_valid(self) -> bool:
        if self.root_node is None:
            return False
        try:
            return self.root_node.is_valid()
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "application_name": self.application_name,
            "runtime_id": self.root_runtime_id,
            "descendant_count": self.descendant_count,
            "is_valid": self.is_valid(),
            "age_seconds": round(_monotonic() - self.discovered_at, 2),
        }

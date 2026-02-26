"""Desktop Performance bounded context — ADR-013.

Optimizes PlatynUI desktop keyword execution through:
1. Application scoping (290x faster via relative XPath)
2. Element caching (375x+ faster via pre-resolved descriptors)
3. Interaction speed profiles (22x faster pointer clicks)

Public API re-exports for container wiring and adapter use.
"""

from .value_objects import (
    CacheCapacity,
    CacheKey,
    CacheTTL,
    InteractionSpeed,
    PointerSpeedProfile,
    POINTER_SPEED_FAST,
    POINTER_SPEED_INSTANT,
    POINTER_SPEED_REALISTIC,
    SPEED_PROFILES,
    XPathAxis,
    XPathTransform,
)
from .entities import ApplicationRoot, CachedElement
from .aggregates import ApplicationScope, ElementCache, InteractionProfile
from .events import (
    ApplicationScopeSet,
    CacheHit,
    ElementCached,
    ElementInvalidated,
    XPathTransformed,
)
from .services import (
    ApplicationScopeManager,
    DesktopKeywordOptimizer,
    EventCollector,
    EventPublisherProtocol,
    RuntimeQueryProtocol,
)

__all__ = [
    # Value Objects
    "CacheCapacity",
    "CacheKey",
    "CacheTTL",
    "InteractionSpeed",
    "PointerSpeedProfile",
    "POINTER_SPEED_FAST",
    "POINTER_SPEED_INSTANT",
    "POINTER_SPEED_REALISTIC",
    "SPEED_PROFILES",
    "XPathAxis",
    "XPathTransform",
    # Entities
    "ApplicationRoot",
    "CachedElement",
    # Aggregates
    "ApplicationScope",
    "ElementCache",
    "InteractionProfile",
    # Events
    "ApplicationScopeSet",
    "CacheHit",
    "ElementCached",
    "ElementInvalidated",
    "XPathTransformed",
    # Services
    "ApplicationScopeManager",
    "DesktopKeywordOptimizer",
    "EventCollector",
    "EventPublisherProtocol",
    "RuntimeQueryProtocol",
]

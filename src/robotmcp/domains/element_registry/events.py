"""Domain Events for the Element Registry Context.

Domain events capture significant occurrences within the bounded context.
They are used for:
- Audit logging
- Cross-context communication
- Event-driven workflows
- Monitoring and debugging
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from robotmcp.domains.element_registry.value_objects import Locator, RegistryId
from robotmcp.domains.shared import AriaRole, ElementRef

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class ElementRegistered:
    """Emitted when a new element is registered in the registry.

    This event captures the creation of a mapping between an element
    reference and its locator, including the ARIA role for context.

    Use cases:
    - Audit trail of element registrations
    - Debugging element lookup issues
    - Metrics on registry usage
    """
    registry_id: RegistryId
    ref: ElementRef
    locator: Locator
    aria_role: AriaRole
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return (
            f"ElementRegistered(ref={self.ref.value}, "
            f"role={self.aria_role.value}, "
            f"locator={self.locator})"
        )


@dataclass(frozen=True)
class RegistryInvalidated:
    """Emitted when the registry is invalidated.

    This event is triggered when the registry becomes stale, typically
    due to navigation or significant DOM changes. All refs in the
    registry are considered invalid after this event.

    Reasons:
    - "navigation": Page navigation occurred
    - "dom_change": Significant DOM mutation detected
    - "manual": Explicitly invalidated by user/system
    - "timeout": Registry expired due to inactivity

    Use cases:
    - Triggering snapshot refresh
    - Notifying consumers of stale refs
    - Metrics on invalidation frequency
    """
    registry_id: RegistryId
    session_id: str
    reason: str
    refs_invalidated: int
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return (
            f"RegistryInvalidated(registry={self.registry_id.value}, "
            f"reason={self.reason}, "
            f"refs_invalidated={self.refs_invalidated})"
        )


@dataclass(frozen=True)
class StaleRefAccessed:
    """Emitted when a stale ref is accessed.

    This event is triggered when code attempts to use an element
    reference that belongs to an older snapshot version. It indicates
    a potential issue with ref lifecycle management.

    Use cases:
    - Debugging stale ref errors
    - Metrics on stale ref frequency
    - Identifying patterns that cause stale refs
    - Proactive snapshot refresh recommendations
    """
    registry_id: RegistryId
    ref: ElementRef
    snapshot_version_expected: int
    snapshot_version_actual: int
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return (
            f"StaleRefAccessed(ref={self.ref.value}, "
            f"expected_version={self.snapshot_version_expected}, "
            f"actual_version={self.snapshot_version_actual})"
        )


@dataclass(frozen=True)
class RefExpired:
    """Emitted when a ref expires due to age.

    This event is triggered when a ref is accessed or cleaned up
    after exceeding the expiration time (5 minutes per ADR-001).

    Use cases:
    - Tracking ref lifecycle
    - Identifying long-running sessions needing refresh
    - Metrics on ref usage patterns
    """
    registry_id: RegistryId
    ref: ElementRef
    age_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return (
            f"RefExpired(ref={self.ref.value}, "
            f"age={self.age_seconds:.1f}s)"
        )


@dataclass(frozen=True)
class RegistryCreated:
    """Emitted when a new registry is created.

    This event is triggered when a new ElementRegistry is instantiated,
    typically after a snapshot is captured.

    Use cases:
    - Session tracking
    - Registry lifecycle monitoring
    - Audit trail
    """
    registry_id: RegistryId
    session_id: str
    snapshot_version: int
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return (
            f"RegistryCreated(registry={self.registry_id.value}, "
            f"session={self.session_id}, "
            f"version={self.snapshot_version})"
        )


@dataclass(frozen=True)
class BulkRegistrationCompleted:
    """Emitted when bulk registration of elements completes.

    This event is triggered after processing an entire aria tree
    and registering all interactive elements.

    Use cases:
    - Performance monitoring
    - Capacity planning
    - Metrics on page complexity
    """
    registry_id: RegistryId
    elements_registered: int
    interactive_elements: int
    total_nodes_processed: int
    duration_ms: float
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return (
            f"BulkRegistrationCompleted(elements={self.elements_registered}, "
            f"interactive={self.interactive_elements}, "
            f"duration={self.duration_ms:.1f}ms)"
        )


@dataclass(frozen=True)
class MaxRefsExceeded:
    """Emitted when registration fails due to max refs limit.

    This event is triggered when an attempt to register an element
    fails because the registry has reached its maximum capacity.

    Use cases:
    - Alerting on capacity issues
    - Identifying pages with too many elements
    - Tuning max refs configuration
    """
    registry_id: RegistryId
    session_id: str
    current_refs: int
    max_refs: int
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return (
            f"MaxRefsExceeded(session={self.session_id}, "
            f"current={self.current_refs}, "
            f"max={self.max_refs})"
        )

"""Aggregates for the Element Registry Context.

The ElementRegistry is the aggregate root for this context. It maintains
the consistency boundary around element mappings and enforces all
invariants (max refs, expiration, stale detection).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

from robotmcp.domains.element_registry.entities import ElementMapping
from robotmcp.domains.element_registry.events import (
    ElementRegistered,
    RegistryInvalidated,
    StaleRefAccessed,
)
from robotmcp.domains.element_registry.value_objects import (
    Locator,
    LocatorStrategy,
    RegistryId,
    StaleRefError,
)
from robotmcp.domains.shared import AriaNode, AriaRole, ElementRef

if TYPE_CHECKING:
    pass


# Type alias for PageSnapshot - will be properly typed when snapshot context is implemented
class PageSnapshot:
    """Placeholder for PageSnapshot from Snapshot Context.

    This will be replaced with proper import when the Snapshot Context
    is implemented.
    """
    snapshot_id: str
    session_id: str
    version: int = 0

    def get_aria_tree(self) -> Optional[AriaNode]:
        """Get the aria tree root node."""
        return None


@dataclass
class ElementRegistry:
    """Aggregate root for element reference management.

    The ElementRegistry maintains mappings between short element references
    (e.g., e1, e42) and their locators. It enforces:

    - Ref ID format validation: ^e\\d{1,10}$
    - Max 10,000 refs per session
    - 5-minute ref expiration
    - Stale reference detection across snapshot versions

    Invariants:
    - All refs must match the format pattern
    - Number of refs cannot exceed MAX_REFS
    - Refs older than REF_EXPIRATION_SECONDS are invalid
    - Registry is invalidated on navigation or significant DOM changes
    """
    registry_id: RegistryId
    session_id: str
    snapshot_version: int
    refs: Dict[ElementRef, ElementMapping] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    invalidated_at: Optional[datetime] = None

    # Configuration constants per ADR-001
    MAX_REFS: int = 10000
    REF_EXPIRATION_SECONDS: int = 300  # 5 minutes

    # Ref format validation pattern
    REF_PATTERN: str = r"^e\d{1,10}$"

    # Internal counter for generating new refs
    _next_ref_index: int = field(default=1, init=False, repr=False)

    # Domain events collected during operations
    _events: List[object] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize internal state after dataclass init."""
        # Find the highest existing ref index to continue from
        if self.refs:
            max_index = max(ref.to_index() for ref in self.refs.keys())
            self._next_ref_index = max_index + 1

    def register_element(
        self,
        aria_node: AriaNode,
        locator: Locator,
    ) -> ElementRef:
        """Register an element from the aria tree.

        Creates a mapping between a new element reference and the
        provided locator based on the aria node's properties.

        Args:
            aria_node: The AriaNode representing the element
            locator: The Locator to use for finding the element

        Returns:
            The assigned ElementRef

        Raises:
            ValueError: If max refs exceeded or registry is invalidated
        """
        # Check if registry is invalidated
        if self.is_stale():
            raise ValueError(
                "Cannot register elements in an invalidated registry. "
                "Call refresh_from_snapshot first."
            )

        # Clean up expired refs first to make room
        self._cleanup_expired_refs()

        # Check max refs limit
        if len(self.refs) >= self.MAX_REFS:
            raise ValueError(
                f"Maximum element refs ({self.MAX_REFS}) exceeded for session. "
                "Call get_page_snapshot to get a fresh registry."
            )

        # Use the ref from the aria node if it has one and it's valid
        if aria_node.ref and self._validate_ref_format(aria_node.ref.value):
            ref = aria_node.ref
        else:
            # Generate a new ref
            ref = ElementRef.from_index(self._next_ref_index)
            self._next_ref_index += 1

        # Create the mapping
        mapping = ElementMapping.from_aria_node(aria_node, locator)
        # Override the ref if we generated a new one
        if mapping.ref != ref:
            mapping = ElementMapping(
                ref=ref,
                locator=locator,
                aria_role=aria_node.role,
                aria_name=aria_node.name,
                registered_at=datetime.now(),
                last_validated_at=datetime.now(),
            )

        self.refs[ref] = mapping

        # Record domain event
        self._events.append(
            ElementRegistered(
                registry_id=self.registry_id,
                ref=ref,
                locator=locator,
                aria_role=aria_node.role,
            )
        )

        return ref

    def get_locator(self, ref: ElementRef) -> Locator:
        """Get the locator for an element reference.

        Args:
            ref: The ElementRef to look up

        Returns:
            The Locator for the element

        Raises:
            StaleRefError: If the ref is from a different snapshot version
            KeyError: If the ref is not found in the registry
            ValueError: If the ref format is invalid
        """
        # Validate ref format
        if not self._validate_ref_format(ref.value):
            raise ValueError(f"Invalid ref format: {ref.value}")

        # Check for stale ref
        if self.is_stale():
            self._events.append(
                StaleRefAccessed(
                    registry_id=self.registry_id,
                    ref=ref,
                    snapshot_version_expected=self.snapshot_version,
                    snapshot_version_actual=-1,  # Unknown current version
                )
            )
            raise StaleRefError(
                ref=ref,
                snapshot_version_used=self.snapshot_version,
                current_snapshot_version=-1,
            )

        # Look up the mapping
        if ref not in self.refs:
            raise KeyError(f"Element ref not found: {ref.value}")

        mapping = self.refs[ref]

        # Check if mapping has expired
        if mapping.is_expired():
            # Remove expired mapping
            del self.refs[ref]
            raise KeyError(
                f"Element ref has expired: {ref.value}. "
                "Call get_page_snapshot for fresh refs."
            )

        return mapping.locator

    def get_mapping(self, ref: ElementRef) -> ElementMapping:
        """Get the full mapping for an element reference.

        Args:
            ref: The ElementRef to look up

        Returns:
            The full ElementMapping

        Raises:
            StaleRefError: If the ref is from a different snapshot version
            KeyError: If the ref is not found in the registry
        """
        # This also validates the ref
        _ = self.get_locator(ref)
        return self.refs[ref]

    def invalidate(self, reason: str = "manual") -> None:
        """Mark the registry as invalidated.

        This is called when navigation occurs or significant DOM changes
        are detected. All refs become stale after invalidation.

        Args:
            reason: The reason for invalidation ("navigation", "dom_change", "manual")
        """
        if self.invalidated_at is not None:
            return  # Already invalidated

        self.invalidated_at = datetime.now()
        refs_count = len(self.refs)

        # Record domain event
        self._events.append(
            RegistryInvalidated(
                registry_id=self.registry_id,
                session_id=self.session_id,
                reason=reason,
                refs_invalidated=refs_count,
            )
        )

    def is_stale(self) -> bool:
        """Check if the registry has been invalidated.

        Returns:
            True if the registry is invalidated and refs are stale
        """
        return self.invalidated_at is not None

    def refresh_from_snapshot(self, snapshot: PageSnapshot) -> "ElementRegistry":
        """Create a new registry from a fresh snapshot.

        This creates a new ElementRegistry based on the new snapshot,
        attempting to preserve mappings for elements that still exist.

        Args:
            snapshot: The new PageSnapshot to base the registry on

        Returns:
            A new ElementRegistry for the snapshot
        """
        # Create new registry
        new_registry = ElementRegistry(
            registry_id=RegistryId.for_session(
                self.session_id,
                snapshot.version if hasattr(snapshot, 'version') else self.snapshot_version + 1
            ),
            session_id=self.session_id,
            snapshot_version=snapshot.version if hasattr(snapshot, 'version') else self.snapshot_version + 1,
            created_at=datetime.now(),
        )

        # Try to migrate valid mappings if we have an aria tree
        aria_tree = snapshot.get_aria_tree() if hasattr(snapshot, 'get_aria_tree') else None
        if aria_tree:
            for ref, mapping in self.refs.items():
                if not mapping.is_expired():
                    # Try to find the element in the new tree
                    node = aria_tree.find_by_ref(ref)
                    if node and mapping.validate_against_node(node):
                        # Element still exists and matches - migrate mapping
                        mapping.refresh()
                        new_registry.refs[ref] = mapping

        return new_registry

    def bulk_register(
        self,
        aria_tree: AriaNode,
        locator_generator: "LocatorGenerator",
    ) -> Dict[ElementRef, Locator]:
        """Register all elements from an aria tree.

        Traverses the aria tree and registers all interactive elements,
        generating locators using the provided generator.

        Args:
            aria_tree: The root AriaNode to traverse
            locator_generator: A callable that generates Locators for nodes

        Returns:
            Dictionary mapping refs to their locators
        """
        registered: Dict[ElementRef, Locator] = {}

        for node in aria_tree.traverse():
            if node.is_interactive:
                locator = locator_generator.generate_locator(node)
                ref = self.register_element(node, locator)
                registered[ref] = locator

        return registered

    def _validate_ref_format(self, ref: str) -> bool:
        """Validate that a ref string matches the expected format.

        Args:
            ref: The ref string to validate

        Returns:
            True if the format is valid
        """
        return bool(re.match(self.REF_PATTERN, ref))

    def _cleanup_expired_refs(self) -> int:
        """Remove expired refs from the registry.

        Returns:
            Number of refs removed
        """
        now = datetime.now()
        expired_refs = [
            ref for ref, mapping in self.refs.items()
            if mapping.is_expired(now)
        ]

        for ref in expired_refs:
            del self.refs[ref]

        return len(expired_refs)

    def get_events(self) -> List[object]:
        """Get and clear collected domain events.

        Returns:
            List of domain events that occurred during operations
        """
        events = self._events.copy()
        self._events.clear()
        return events

    def stats(self) -> Dict[str, object]:
        """Get statistics about the registry.

        Returns:
            Dictionary with registry statistics
        """
        now = datetime.now()
        expired_count = sum(
            1 for mapping in self.refs.values()
            if mapping.is_expired(now)
        )

        return {
            "registry_id": self.registry_id.value,
            "session_id": self.session_id,
            "snapshot_version": self.snapshot_version,
            "total_refs": len(self.refs),
            "expired_refs": expired_count,
            "active_refs": len(self.refs) - expired_count,
            "is_stale": self.is_stale(),
            "created_at": self.created_at.isoformat(),
            "invalidated_at": (
                self.invalidated_at.isoformat()
                if self.invalidated_at
                else None
            ),
        }

    @classmethod
    def create_for_session(
        cls,
        session_id: str,
        snapshot_version: int = 1,
    ) -> "ElementRegistry":
        """Factory method to create a new registry for a session.

        Args:
            session_id: The session identifier
            snapshot_version: Initial snapshot version (default 1)

        Returns:
            A new ElementRegistry
        """
        return cls(
            registry_id=RegistryId.for_session(session_id, snapshot_version),
            session_id=session_id,
            snapshot_version=snapshot_version,
            created_at=datetime.now(),
        )


class LocatorGenerator:
    """Generates locators for AriaNodes.

    This is a strategy class that can be customized to generate
    different types of locators based on the node properties.
    """

    def __init__(
        self,
        prefer_testid: bool = True,
        prefer_aria_label: bool = True,
    ) -> None:
        """Initialize the generator with preferences.

        Args:
            prefer_testid: Prefer data-testid if available
            prefer_aria_label: Prefer aria-label if available
        """
        self.prefer_testid = prefer_testid
        self.prefer_aria_label = prefer_aria_label

    def generate_locator(self, node: AriaNode) -> Locator:
        """Generate a locator for an AriaNode.

        The strategy prioritizes stable locators in this order:
        1. data-testid (if prefer_testid and available)
        2. aria-label (if prefer_aria_label and name available)
        3. role + name combination
        4. CSS based on role

        Args:
            node: The AriaNode to generate a locator for

        Returns:
            A Locator for finding the element
        """
        props = node.properties

        # Check for data-testid
        if self.prefer_testid and "data-testid" in props:
            return Locator.create_safe(
                LocatorStrategy.DATA_TESTID,
                str(props["data-testid"]),
            )

        # Check for aria-label via node name
        if self.prefer_aria_label and node.name:
            return Locator.create_safe(
                LocatorStrategy.ARIA_LABEL,
                node.name,
            )

        # Use role-based locator
        if node.name:
            # Role with name
            return Locator.create_safe(
                LocatorStrategy.ROLE,
                f'{node.role.value}[name="{node.name}"]',
            )

        # Fallback to role-only
        return Locator.create_safe(
            LocatorStrategy.ROLE,
            node.role.value,
        )

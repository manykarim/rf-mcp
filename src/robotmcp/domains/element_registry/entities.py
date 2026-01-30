"""Entities for the Element Registry Context.

Entities have identity and lifecycle. In the Element Registry context,
ElementMapping is an entity that tracks the mapping between an element
reference and its locator, along with metadata for validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from robotmcp.domains.element_registry.value_objects import Locator
from robotmcp.domains.shared import AriaNode, AriaRole, ElementRef

if TYPE_CHECKING:
    pass


@dataclass
class ElementMapping:
    """Mapping between an element reference and its locator.

    An ElementMapping tracks:
    - The short element reference (e.g., e42)
    - The locator strategy and value for finding the element
    - ARIA metadata for validation
    - Timestamps for expiration tracking

    Entity Identity: The ElementRef serves as the identity.
    """
    ref: ElementRef
    locator: Locator
    aria_role: AriaRole
    aria_name: Optional[str] = None
    registered_at: datetime = field(default_factory=datetime.now)
    last_validated_at: Optional[datetime] = None

    # Expiration time in seconds (5 minutes per ADR-001)
    EXPIRATION_SECONDS: int = 300

    def validate_against_node(self, node: AriaNode) -> bool:
        """Validate this mapping against an AriaNode.

        Checks if the stored ARIA role and name still match the
        current state of the element in the accessibility tree.

        Args:
            node: The AriaNode to validate against

        Returns:
            True if the mapping is still valid (role and name match)
        """
        # Check role match
        if self.aria_role.value != node.role.value:
            return False

        # Check name match (both None or both equal)
        if self.aria_name != node.name:
            return False

        # Update validation timestamp
        self.last_validated_at = datetime.now()
        return True

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """Check if this mapping has expired.

        Mappings expire after EXPIRATION_SECONDS (5 minutes) to prevent
        stale references from accumulating.

        Args:
            now: Current time (defaults to datetime.now())

        Returns:
            True if the mapping has expired
        """
        if now is None:
            now = datetime.now()

        age_seconds = (now - self.registered_at).total_seconds()
        return age_seconds > self.EXPIRATION_SECONDS

    def time_until_expiration(self, now: Optional[datetime] = None) -> float:
        """Calculate seconds until this mapping expires.

        Args:
            now: Current time (defaults to datetime.now())

        Returns:
            Seconds until expiration (negative if already expired)
        """
        if now is None:
            now = datetime.now()

        age_seconds = (now - self.registered_at).total_seconds()
        return self.EXPIRATION_SECONDS - age_seconds

    def refresh(self) -> None:
        """Refresh the mapping's registration time.

        Call this when the mapping has been re-validated and should
        have its expiration timer reset.
        """
        self.registered_at = datetime.now()
        self.last_validated_at = datetime.now()

    @classmethod
    def from_aria_node(
        cls,
        node: AriaNode,
        locator: Locator,
    ) -> "ElementMapping":
        """Create an ElementMapping from an AriaNode.

        Factory method that extracts the necessary information from
        an AriaNode to create a mapping.

        Args:
            node: The AriaNode to create a mapping for
            locator: The Locator to use for finding the element

        Returns:
            A new ElementMapping
        """
        return cls(
            ref=node.ref,
            locator=locator,
            aria_role=node.role,
            aria_name=node.name,
            registered_at=datetime.now(),
            last_validated_at=datetime.now(),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the mapping
        """
        return {
            "ref": self.ref.value,
            "locator": {
                "strategy": self.locator.strategy.value,
                "value": self.locator.value,
            },
            "aria_role": self.aria_role.value,
            "aria_name": self.aria_name,
            "registered_at": self.registered_at.isoformat(),
            "last_validated_at": (
                self.last_validated_at.isoformat()
                if self.last_validated_at
                else None
            ),
        }

    def __hash__(self) -> int:
        """Hash based on ref identity."""
        return hash(self.ref)

    def __eq__(self, other: object) -> bool:
        """Equality based on ref identity."""
        if not isinstance(other, ElementMapping):
            return NotImplemented
        return self.ref == other.ref

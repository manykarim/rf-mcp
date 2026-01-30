"""Element Registry Context - Element reference mapping and stale detection.

This bounded context manages:
- Element reference registration and lookup
- Locator strategy mapping
- Stale reference detection
- Registry invalidation on navigation/DOM changes

Security Controls (per ADR-001):
- Ref ID format validation: ^e\\d{1,10}$
- Max 10,000 refs per session
- 5-minute ref expiration
- Input sanitization for locator values
"""

# Value Objects
from robotmcp.domains.element_registry.value_objects import (
    Locator,
    LocatorStrategy,
    RegistryId,
    StaleRefError,
)

# Entities
from robotmcp.domains.element_registry.entities import (
    ElementMapping,
)

# Aggregates
from robotmcp.domains.element_registry.aggregates import (
    ElementRegistry,
)

# Domain Events
from robotmcp.domains.element_registry.events import (
    ElementRegistered,
    RegistryInvalidated,
    StaleRefAccessed,
)

# Repository Protocol
from robotmcp.domains.element_registry.repository import (
    ElementRegistryRepository,
)

__all__ = [
    # Value Objects
    "Locator",
    "LocatorStrategy",
    "RegistryId",
    "StaleRefError",
    # Entities
    "ElementMapping",
    # Aggregates
    "ElementRegistry",
    # Events
    "ElementRegistered",
    "RegistryInvalidated",
    "StaleRefAccessed",
    # Repository
    "ElementRegistryRepository",
]

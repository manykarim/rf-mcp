"""Tool Profile Bounded Context (ADR-006).

Manages dynamic MCP tool gating for small-context LLM optimization.
Controls which tools are visible, how descriptions are formatted,
and how profiles transition through workflow phases.

The domain follows DDD patterns with:
- Value Objects: Immutable objects identified by their values
- Entities: Objects with identity and lifecycle
- Aggregates: Clusters of objects treated as a unit
- Domain Events: Record of something that happened in the domain
- Services: Business logic that doesn't fit in entities

Example usage:
    from robotmcp.domains.tool_profile import (
        ToolProfile,
        ProfilePresets,
        ToolProfileManager,
        ModelTier,
        ToolDescriptionMode,
    )

    # Create a profile for small-context LLMs
    profile = ProfilePresets.browser_exec()

    # Suggest a profile based on scenario
    manager = ToolProfileManager(tool_manager, descriptors)
    suggested = manager.suggest_profile("Browser login test", ModelTier.SMALL_CONTEXT)

For more details, see:
- docs/adr/ADR-006-tool-profile-bounded-context.md
"""

# Value Objects
from .value_objects import (
    ModelTier,
    ToolDescriptionMode,
    ToolTag,
    TokenBudget,
    ProfileTransition,
)

# Entities
from .entities import (
    ToolDescriptor,
)

# Aggregates
from .aggregates import (
    ToolProfile,
    ProfilePresets,
)

# Domain Events
from .events import (
    ProfileActivated,
    ProfileTransitioned,
    ToolsHidden,
    ToolsRevealed,
    TokenBudgetExceeded,
)

# Services
from .services import (
    ToolProfileManager,
    ToolManagerPort,
)

# Repository
from .repository import (
    ProfileRepository,
    InMemoryProfileRepository,
)

# Adapters
from .adapters import (
    ToolManagerAdapter,
)

__all__ = [
    # Value Objects
    "ModelTier",
    "ToolDescriptionMode",
    "ToolTag",
    "TokenBudget",
    "ProfileTransition",
    # Entities
    "ToolDescriptor",
    # Aggregates
    "ToolProfile",
    "ProfilePresets",
    # Domain Events
    "ProfileActivated",
    "ProfileTransitioned",
    "ToolsHidden",
    "ToolsRevealed",
    "TokenBudgetExceeded",
    # Services
    "ToolProfileManager",
    "ToolManagerPort",
    # Repository
    "ProfileRepository",
    "InMemoryProfileRepository",
    # Adapters
    "ToolManagerAdapter",
]

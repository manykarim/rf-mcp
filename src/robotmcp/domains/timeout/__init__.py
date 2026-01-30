"""Timeout Domain - Bounded Context for Timeout Management.

This module implements the Timeout bounded context as defined in
ADR-001. It provides domain-driven design components for managing
timeout configuration with the dual timeout strategy:

- 5 seconds for element actions (clicks, typing, etc.)
- 60 seconds for navigation operations (page loads, reloads)

The domain follows DDD patterns with:
- Value Objects: Immutable objects identified by their values
- Entities: Objects with identity and lifecycle
- Aggregates: Clusters of objects treated as a unit
- Domain Events: Record of something that happened in the domain
- Services: Business logic that doesn't fit in entities

Example usage:
    from robotmcp.domains.timeout import (
        TimeoutService,
        TimeoutPolicy,
        ActionType,
        Milliseconds,
    )

    # Create a service and policy
    service = TimeoutService()
    policy = service.create_default_policy("session_123")

    # Get timeout for an action
    timeout = service.get_timeout_for_action(policy, ActionType.CLICK)
    print(f"Click timeout: {timeout}")  # 5000ms

    # Get timeout for navigation
    timeout = service.get_timeout_for_action(policy, ActionType.NAVIGATE)
    print(f"Navigate timeout: {timeout}")  # 60000ms

    # Create custom override
    custom_policy = policy.with_override(ActionType.CLICK, Milliseconds(10000))

For more details, see:
- ADR-001: Domain-Driven Design Architecture for Token Optimization
"""

# Value Objects
from .value_objects import (
    PolicyId,
    Milliseconds,
    DefaultTimeouts,
)

# Entities
from .entities import (
    ActionType,
    TimeoutCategory,
)

# Aggregates
from .aggregates import (
    TimeoutPolicy,
)

# Domain Events
from .events import (
    TimeoutPolicyCreated,
    TimeoutExceeded,
    TimeoutOverrideApplied,
    TimeoutPolicyUpdated,
    TimeoutWarning,
)

# Services
from .services import (
    TimeoutService,
    TimeoutContextManager,
)

__all__ = [
    # Value Objects
    "PolicyId",
    "Milliseconds",
    "DefaultTimeouts",
    # Entities
    "ActionType",
    "TimeoutCategory",
    # Aggregates
    "TimeoutPolicy",
    # Domain Events
    "TimeoutPolicyCreated",
    "TimeoutExceeded",
    "TimeoutOverrideApplied",
    "TimeoutPolicyUpdated",
    "TimeoutWarning",
    # Services
    "TimeoutService",
    "TimeoutContextManager",
]

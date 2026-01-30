"""Domain-Driven Design bounded contexts for rf-mcp token optimization.

This package contains the DDD implementation for:
- Snapshot Context: Accessibility tree generation and compression
- Element Registry Context: Element reference mapping and stale detection
- Action Context: Tool execution and response filtering
- Timeout Context: Timeout configuration and enforcement

For architecture details, see:
/docs/adr/ADR-001-ddd-architecture-token-optimization.md
"""

from robotmcp.domains.shared import (
    AriaNode,
    AriaRole,
    ElementRef,
)

from robotmcp.domains.timeout import (
    # Value Objects
    PolicyId,
    Milliseconds,
    DefaultTimeouts,
    # Entities
    ActionType,
    TimeoutCategory,
    # Aggregates
    TimeoutPolicy,
    # Domain Events
    TimeoutPolicyCreated,
    TimeoutExceeded as TimeoutTimeoutExceeded,
    TimeoutOverrideApplied,
    TimeoutPolicyUpdated,
    TimeoutWarning,
    # Services
    TimeoutService,
    TimeoutContextManager,
)

from robotmcp.domains.action import (
    # Value Objects
    ActionParameters,
    ExecutionId,
    FilteredResponse,
    PreValidationResult,
    ResponseConfig,
    # Protocols
    BrowserAdapter,
    ElementRegistry,
    Locator,
    # Services
    PreValidator,
    ResponseBuilder,
    IncrementalResponseBuilder,
    # Aggregates
    ActionExecution,
    RetryableActionExecution,
    # Events
    ActionCompleted,
    ActionFailed,
    ActionRetrying,
    ActionStarted,
    PreValidationCompleted,
    TimeoutExceeded as ActionTimeoutExceeded,
)

__all__ = [
    # Shared Kernel
    "AriaNode",
    "AriaRole",
    "ElementRef",
    # Timeout Domain - Value Objects
    "PolicyId",
    "Milliseconds",
    "DefaultTimeouts",
    # Timeout Domain - Entities
    "ActionType",
    "TimeoutCategory",
    # Timeout Domain - Aggregates
    "TimeoutPolicy",
    # Timeout Domain - Events
    "TimeoutPolicyCreated",
    "TimeoutTimeoutExceeded",
    "TimeoutOverrideApplied",
    "TimeoutPolicyUpdated",
    "TimeoutWarning",
    # Timeout Domain - Services
    "TimeoutService",
    "TimeoutContextManager",
    # Action Domain - Value Objects
    "ActionParameters",
    "ExecutionId",
    "FilteredResponse",
    "PreValidationResult",
    "ResponseConfig",
    # Action Domain - Protocols
    "BrowserAdapter",
    "ElementRegistry",
    "Locator",
    # Action Domain - Services
    "PreValidator",
    "ResponseBuilder",
    "IncrementalResponseBuilder",
    # Action Domain - Aggregates
    "ActionExecution",
    "RetryableActionExecution",
    # Action Domain - Events
    "ActionCompleted",
    "ActionFailed",
    "ActionRetrying",
    "ActionStarted",
    "PreValidationCompleted",
    "ActionTimeoutExceeded",
]

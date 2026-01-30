"""Action Context - Manages tool execution, pre-validation, and response filtering.

This bounded context handles the complete action execution pipeline including:
- Pre-validation of element actionability
- Action execution with proper timeout management
- Token-optimized response building

The Action Context integrates with:
- Element Registry Context: For ref -> locator resolution
- Snapshot Context: For capturing page state after actions
- Timeout Context: For action/navigation timeout configuration
"""

from robotmcp.domains.action.value_objects import (
    ActionParameters,
    ExecutionId,
    FilteredResponse,
    PreValidationResult,
    ResponseConfig,
)
from robotmcp.domains.action.services import (
    BrowserAdapter,
    ElementRegistry,
    Locator,
    PreValidator,
)
from robotmcp.domains.action.aggregates import (
    ActionExecution,
    RetryableActionExecution,
)
# Re-export Milliseconds from timeout domain for convenience
from robotmcp.domains.timeout.value_objects import Milliseconds
from robotmcp.domains.action.events import (
    ActionCompleted,
    ActionFailed,
    ActionRetrying,
    ActionStarted,
    PreValidationCompleted,
    TimeoutExceeded,
)
from robotmcp.domains.action.response_builder import (
    IncrementalResponseBuilder,
    ResponseBuilder,
)

__all__ = [
    # Value Objects
    "ActionParameters",
    "ExecutionId",
    "FilteredResponse",
    "Milliseconds",
    "PreValidationResult",
    "ResponseConfig",
    # Protocols (for type hints and adapters)
    "BrowserAdapter",
    "ElementRegistry",
    "Locator",
    # Services
    "PreValidator",
    "ResponseBuilder",
    "IncrementalResponseBuilder",
    # Aggregates
    "ActionExecution",
    "RetryableActionExecution",
    # Events
    "ActionCompleted",
    "ActionFailed",
    "ActionRetrying",
    "ActionStarted",
    "PreValidationCompleted",
    "TimeoutExceeded",
]

"""Instruction Domain - Bounded Context for MCP Instructions Management.

This module implements the Instruction bounded context for managing
system-level instructions that guide LLMs to use discovery tools
appropriately and prevent guessing of keywords/locators.

The domain supports three instruction modes:
- OFF: No instructions are applied
- DEFAULT: Built-in discovery-first instructions are used
- CUSTOM: User-provided custom instructions from file

The domain follows DDD patterns with:
- Value Objects: Immutable objects identified by their values
- Entities: Objects with identity and lifecycle
- Aggregates: Clusters of objects treated as a unit
- Domain Events: Record of something that happened in the domain
- Services: Business logic that doesn't fit in entities

Example usage:
    from robotmcp.domains.instruction import (
        InstructionConfig,
        InstructionMode,
        InstructionResolver,
        InstructionValidator,
        FastMCPInstructionAdapter,
    )

    # Create configuration
    config = InstructionConfig.create_default()

    # Or from environment
    adapter = FastMCPInstructionAdapter()
    config = adapter.create_config_from_env()

    # Get instructions for FastMCP server
    instructions = adapter.get_server_instructions(config)

For more details, see:
- docs/architecture/instruction_domain_design.md
- docs/architecture/instruction_domain_diagrams.md
"""

# Value Objects
from .value_objects import (
    InstructionMode,
    InstructionContent,
    InstructionPath,
    InstructionTemplate,
)

# Entities
from .entities import (
    InstructionVersion,
)

# Aggregates
from .aggregates import (
    InstructionConfig,
)

# Domain Events
from .events import (
    InstructionApplied,
    InstructionOverridden,
    InstructionValidationFailed,
    InstructionContentLoaded,
)

# Services
from .services import (
    InstructionResolver,
    InstructionValidator,
    InstructionRenderer,
    ValidationResult,
)

# Repository
from .repository import (
    InstructionRepository,
    InMemoryInstructionRepository,
)

# Adapters
from .adapters import (
    FastMCPInstructionAdapter,
    InstructionTemplateType,
)

# Security
from .security import (
    SecurityError,
    PathTraversalError,
    PromptInjectionError,
    InvalidInstructionModeError,
    ContentValidationError,
    ValidInstructionMode,
    SecurityValidationResult,
    InstructionContentValidator,
    SecurePathValidator,
    SecureEnvironmentValidator,
    SecureLogger,
    InstructionSecurityService,
)

__all__ = [
    # Value Objects
    "InstructionMode",
    "InstructionContent",
    "InstructionPath",
    "InstructionTemplate",
    # Entities
    "InstructionVersion",
    # Aggregates
    "InstructionConfig",
    # Domain Events
    "InstructionApplied",
    "InstructionOverridden",
    "InstructionValidationFailed",
    "InstructionContentLoaded",
    # Services
    "InstructionResolver",
    "InstructionValidator",
    "InstructionRenderer",
    "ValidationResult",
    # Repository
    "InstructionRepository",
    "InMemoryInstructionRepository",
    # Adapters
    "FastMCPInstructionAdapter",
    "InstructionTemplateType",
    # Security
    "SecurityError",
    "PathTraversalError",
    "PromptInjectionError",
    "InvalidInstructionModeError",
    "ContentValidationError",
    "ValidInstructionMode",
    "SecurityValidationResult",
    "InstructionContentValidator",
    "SecurePathValidator",
    "SecureEnvironmentValidator",
    "SecureLogger",
    "InstructionSecurityService",
]

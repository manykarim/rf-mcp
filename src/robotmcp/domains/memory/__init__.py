"""Memory Domain — Bounded Context for Persistent Semantic Memory.

This module implements the Memory bounded context for storing and
recalling test automation knowledge across MCP sessions using
vector embeddings and semantic search.

The domain supports six memory types:
- WORKING_STEPS: Previously successful step sequences
- KEYWORDS: Keyword usage patterns and argument examples
- DOCUMENTATION: Library and framework documentation
- COMMON_ERRORS: Error messages paired with known fixes
- DOMAIN_KNOWLEDGE: Domain-specific knowledge and patterns
- LOCATORS: Structured locator-outcome mappings for web elements

Architecture: Python rf-mcp -> sqlite-vec (in-process) -> ~/.rf-mcp/memory.db
Embeddings: model2vec (256-dim) / fastembed (384-dim) / sentence-transformers (384-dim)
"""

# Value Objects
from .value_objects import (
    ConfidenceScore,
    EmbeddingVector,
    LocatorDescription,
    LocatorOutcome,
    LocatorRecallResult,
    LocatorStrategy,
    MemoryEntry,
    MemoryQuery,
    MemoryType,
    MemoryTypeEnum,
    RecallResult,
    SimilarityScore,
    StorageConfig,
    TimeDecayFactor,
)

# Entities
from .entities import (
    MemoryCollection,
    MemoryRecord,
)

# Aggregates
from .aggregates import (
    EmbeddingBackend,
    MemoryStore,
)

# Domain Events
from .events import (
    CollectionCreated,
    LocatorStored,
    MemoryDecayed,
    MemoryPruned,
    MemoryRecalled,
    MemoryStored,
)

# Services
from .services import (
    EmbeddingService,
    MemoryHookService,
    MemoryQueryService,
    create_memory_services,
)

# Repository
from .repository import (
    InMemoryMemoryRepository,
    MemoryRepository,
)

__all__ = [
    # Value Objects
    "MemoryType",
    "MemoryTypeEnum",
    "EmbeddingVector",
    "SimilarityScore",
    "ConfidenceScore",
    "TimeDecayFactor",
    "MemoryQuery",
    "MemoryEntry",
    "RecallResult",
    "StorageConfig",
    "LocatorStrategy",
    "LocatorOutcome",
    "LocatorDescription",
    "LocatorRecallResult",
    # Entities
    "MemoryRecord",
    "MemoryCollection",
    # Aggregates
    "MemoryStore",
    "EmbeddingBackend",
    # Domain Events
    "MemoryStored",
    "MemoryRecalled",
    "MemoryDecayed",
    "CollectionCreated",
    "MemoryPruned",
    "LocatorStored",
    # Services
    "EmbeddingService",
    "MemoryQueryService",
    "MemoryHookService",
    "create_memory_services",
    # Repository
    "MemoryRepository",
    "InMemoryMemoryRepository",
]

"""Snapshot Bounded Context for rf-mcp Token Optimization.

This module provides the domain model for managing accessibility tree snapshots,
incremental diffs, and content compression. It is part of the Domain-Driven Design
architecture for achieving 70-90% token reduction in rf-mcp.

Key Components:
- PageSnapshot: Aggregate root managing all snapshot-related state
- AriaTree/AriaNode: Entities representing the accessibility tree structure
- SnapshotDiffService: Domain service for computing incremental diffs
- ListFoldingService: Domain service for SimHash-based list compression

Example Usage:
    from robotmcp.domains.snapshot import (
        PageSnapshot,
        SnapshotId,
        SnapshotFormat,
        SnapshotDiffService,
        ListFoldingService,
        InMemorySnapshotRepository,
    )

    # Create a snapshot
    snapshot = PageSnapshot.create(session_id="sess_123", aria_tree=tree)

    # Configure format
    format = SnapshotFormat.incremental()

    # Apply list folding
    folding_service = ListFoldingService()
    folded_snapshot, event = folding_service.fold_snapshot(snapshot)

    # Compute diff with previous snapshot
    diff_service = SnapshotDiffService()
    diff = diff_service.compute_diff(current_snapshot, previous_snapshot)

    # Store in repository
    repo = InMemorySnapshotRepository()
    repo.save(snapshot)
"""

# Value Objects from models module (primary source for backwards compatibility)
from robotmcp.domains.snapshot.models import (
    AriaNode,
    AriaRole,
    AriaTree,
    CompressionStats,
    ElementRef,
    PageSnapshot,
    SnapshotFormat,
    SnapshotId,
)

# Diff tracking service
from robotmcp.domains.snapshot.diff_service import (
    ChangeType,
    NodeChange,
    SnapshotDiff,
    SnapshotDiffService,
    SnapshotManager,
)

# Domain Events
from robotmcp.domains.snapshot.events import (
    ListsFolded,
    SnapshotCaptured,
    SnapshotDiffComputed,
    SnapshotExpired,
    SnapshotFormatChanged,
)

# Repository
from robotmcp.domains.snapshot.repository import (
    InMemorySnapshotRepository,
    SnapshotRepository,
)

# Domain Services
from robotmcp.domains.snapshot.services import (
    ListFoldingService,
    SnapshotCaptureService,
)

__all__ = [
    # Value Objects
    "AriaRole",
    "CompressionStats",
    "ElementRef",
    "SnapshotFormat",
    "SnapshotId",
    # Entities
    "AriaNode",
    "AriaTree",
    # Aggregates
    "PageSnapshot",
    # Domain Events
    "ListsFolded",
    "SnapshotCaptured",
    "SnapshotDiffComputed",
    "SnapshotExpired",
    "SnapshotFormatChanged",
    # Repository
    "InMemorySnapshotRepository",
    "SnapshotRepository",
    # Domain Services
    "ListFoldingService",
    "SnapshotCaptureService",
    "SnapshotDiffService",
    "SnapshotManager",
    # Diff tracking (for backwards compatibility)
    "ChangeType",
    "NodeChange",
    "SnapshotDiff",
]

# Optional: Extended list folding (import only if available)
try:
    from robotmcp.domains.snapshot.list_folding import (
        FoldedListItem,
        SimHash,
        SimHashConfig,
    )
    __all__.extend([
        "FoldedListItem",
        "SimHash",
        "SimHashConfig",
    ])
except ImportError:
    # Extended list folding module not available
    pass

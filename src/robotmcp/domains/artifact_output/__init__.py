"""Artifact Output Bounded Context (ADR-015).

Externalizes large tool outputs to artifact files, returning
summaries + references to reduce token consumption.
"""

from .aggregates import ArtifactStore
from .entities import Artifact, ArtifactSlice
from .events import (
    ArtifactAccessed,
    ArtifactCreated,
    ArtifactExpired,
    LargeFieldExternalized,
)
from .services import (
    DEFAULT_RULES,
    ArtifactExternalizationService,
    ArtifactRetrievalService,
)
from .value_objects import (
    ArtifactId,
    ArtifactPolicy,
    ArtifactReference,
    ExternalizationResult,
    ExternalizationRule,
    OutputMode,
)

__all__ = [
    "OutputMode",
    "ArtifactId",
    "ArtifactReference",
    "ArtifactPolicy",
    "ExternalizationRule",
    "ExternalizationResult",
    "Artifact",
    "ArtifactSlice",
    "ArtifactStore",
    "ArtifactCreated",
    "ArtifactAccessed",
    "ArtifactExpired",
    "LargeFieldExternalized",
    "ArtifactExternalizationService",
    "ArtifactRetrievalService",
    "DEFAULT_RULES",
]

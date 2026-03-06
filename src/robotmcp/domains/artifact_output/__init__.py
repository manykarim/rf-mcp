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
    FETCH_ARTIFACT_SUMMARY_TEMPLATE,
    FILE_PATH_SUMMARY_TEMPLATE,
    ArtifactId,
    ArtifactPolicy,
    ArtifactReference,
    ExternalizationResult,
    ExternalizationRule,
    OutputMode,
    is_fetch_artifact_enabled,
)

__all__ = [
    "OutputMode",
    "ArtifactId",
    "ArtifactReference",
    "ArtifactPolicy",
    "ExternalizationRule",
    "ExternalizationResult",
    "is_fetch_artifact_enabled",
    "FETCH_ARTIFACT_SUMMARY_TEMPLATE",
    "FILE_PATH_SUMMARY_TEMPLATE",
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

"""Response Optimization Aggregate Root."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from .value_objects import (
    FieldAbbreviationMap,
    SnapshotCompressionMode,
    TruncationPolicy,
    Verbosity,
)


@dataclass
class ResponseOptimizationConfig:
    """Aggregate root for per-session response optimization settings.

    Invariants:
    - COMPACT verbosity always enables field abbreviation
    - Token budget must be positive
    - Session-scoped (one config per session)
    """
    config_id: str
    session_id: str
    verbosity: Verbosity
    field_abbreviation: FieldAbbreviationMap
    truncation: TruncationPolicy
    snapshot_compression: SnapshotCompressionMode
    token_budget: int  # Max tokens for a single tool response

    __test__ = False  # Suppress pytest collection

    def __post_init__(self) -> None:
        if self.token_budget <= 0:
            raise ValueError("Token budget must be positive")
        if self.verbosity == Verbosity.COMPACT and not self.field_abbreviation.enabled:
            raise ValueError("COMPACT verbosity requires field abbreviation to be enabled")

    @classmethod
    def create_default(cls, session_id: str) -> ResponseOptimizationConfig:
        """Standard config: minimal abbreviation, no snapshot compression."""
        return cls(
            config_id=str(uuid.uuid4()),
            session_id=session_id,
            verbosity=Verbosity.STANDARD,
            field_abbreviation=FieldAbbreviationMap.disabled(),
            truncation=TruncationPolicy.default(),
            snapshot_compression=SnapshotCompressionMode.NONE,
            token_budget=4000,
        )

    @classmethod
    def create_compact(cls, session_id: str) -> ResponseOptimizationConfig:
        """Compact config: full abbreviation + folded snapshots."""
        return cls(
            config_id=str(uuid.uuid4()),
            session_id=session_id,
            verbosity=Verbosity.COMPACT,
            field_abbreviation=FieldAbbreviationMap.standard(),
            truncation=TruncationPolicy.aggressive(),
            snapshot_compression=SnapshotCompressionMode.FOLDED_DIFF,
            token_budget=1500,
        )

    @classmethod
    def create_verbose(cls, session_id: str) -> ResponseOptimizationConfig:
        """Verbose config: no compression at all."""
        return cls(
            config_id=str(uuid.uuid4()),
            session_id=session_id,
            verbosity=Verbosity.VERBOSE,
            field_abbreviation=FieldAbbreviationMap.disabled(),
            truncation=TruncationPolicy.default(),
            snapshot_compression=SnapshotCompressionMode.NONE,
            token_budget=10000,
        )

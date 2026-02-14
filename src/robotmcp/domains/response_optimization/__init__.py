"""Response Optimization Bounded Context (ADR-008).

Orchestrates token-efficient response compression for MCP tool outputs.
Wraps existing compression subsystems through Protocol-based ports
with zero changes to the underlying implementations.
"""
from .aggregates import ResponseOptimizationConfig
from .value_objects import (
    Verbosity, FieldAbbreviationMap, TruncationPolicy,
    SnapshotCompressionMode, TokenEstimate,
)
from .services import ResponseCompressor, CompressionMetrics
from .events import (
    ResponseCompressed, SnapshotFolded,
    IncrementalDiffComputed, CompressionRatioLearned,
)

__all__ = [
    "ResponseOptimizationConfig",
    "Verbosity", "FieldAbbreviationMap", "TruncationPolicy",
    "SnapshotCompressionMode", "TokenEstimate",
    "ResponseCompressor", "CompressionMetrics",
    "ResponseCompressed", "SnapshotFolded",
    "IncrementalDiffComputed", "CompressionRatioLearned",
]

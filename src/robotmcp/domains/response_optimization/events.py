"""Response Optimization Domain Events."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass(frozen=True)
class ResponseCompressed:
    """Emitted after every tool response is compressed."""
    session_id: str
    tool_name: str
    raw_tokens: int
    compressed_tokens: int
    compression_ratio: float
    verbosity: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "ResponseCompressed",
            "session_id": self.session_id,
            "tool_name": self.tool_name,
            "raw_tokens": self.raw_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": round(self.compression_ratio, 4),
            "verbosity": self.verbosity,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class SnapshotFolded:
    """Emitted when SimHash list folding is applied."""
    session_id: str
    items_original: int
    items_after_folding: int
    fold_threshold: float
    tokens_before: int
    tokens_after: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "SnapshotFolded",
            "session_id": self.session_id,
            "items_original": self.items_original,
            "items_after_folding": self.items_after_folding,
            "fold_threshold": self.fold_threshold,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class IncrementalDiffComputed:
    """Emitted when incremental diff replaces full snapshot."""
    session_id: str
    full_snapshot_tokens: int
    diff_tokens: int
    change_count: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "IncrementalDiffComputed",
            "session_id": self.session_id,
            "full_snapshot_tokens": self.full_snapshot_tokens,
            "diff_tokens": self.diff_tokens,
            "change_count": self.change_count,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class CompressionRatioLearned:
    """Emitted when CompressionLearner updates a pattern."""
    page_type: str
    sample_count: int
    avg_compression_ratio: float
    optimal_fold_threshold: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": "CompressionRatioLearned",
            "page_type": self.page_type,
            "sample_count": self.sample_count,
            "avg_compression_ratio": round(self.avg_compression_ratio, 4),
            "optimal_fold_threshold": self.optimal_fold_threshold,
            "timestamp": self.timestamp.isoformat(),
        }

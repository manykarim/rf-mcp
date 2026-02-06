"""Compression pattern learning for ARIA snapshot optimization.

This module learns optimal compression strategies per page type
based on observed compression ratios and success rates.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import statistics

from .pattern_store import PatternStore


@dataclass
class CompressionPattern:
    """
    Learned compression pattern for a page type.

    Attributes:
        page_type: The classified page type (e.g., "search_results")
        url_pattern: Regex pattern matching URLs of this type
        avg_compression_ratio: Average observed compression ratio
        optimal_fold_threshold: Learned optimal SimHash threshold
        use_aggressive_filtering: Whether to use aggressive element filtering
        sample_count: Number of samples used to learn this pattern
        confidence: Confidence score (0.0 to 1.0) based on sample count
    """
    page_type: str
    url_pattern: str
    avg_compression_ratio: float
    optimal_fold_threshold: float
    use_aggressive_filtering: bool
    sample_count: int
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "page_type": self.page_type,
            "url_pattern": self.url_pattern,
            "avg_compression_ratio": self.avg_compression_ratio,
            "optimal_fold_threshold": self.optimal_fold_threshold,
            "use_aggressive_filtering": self.use_aggressive_filtering,
            "sample_count": self.sample_count,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompressionPattern":
        """Create from dictionary."""
        return cls(
            page_type=data.get("page_type", "unknown"),
            url_pattern=data.get("url_pattern", ""),
            avg_compression_ratio=data.get("avg_compression_ratio", 1.0),
            optimal_fold_threshold=data.get("optimal_fold_threshold", 0.85),
            use_aggressive_filtering=data.get("use_aggressive_filtering", False),
            sample_count=data.get("sample_count", 0),
            confidence=data.get("confidence", 0.0),
        )


@dataclass
class CompressionResult:
    """
    A single compression result for learning.

    Attributes:
        compression_ratio: Ratio of raw size to compressed size
        fold_threshold: SimHash threshold used
        success: Whether the compression was successful
        raw_size: Original size in characters
        compressed_size: Final size in characters
    """
    compression_ratio: float
    fold_threshold: float
    success: bool
    raw_size: int = 0
    compressed_size: int = 0


class CompressionPatternLearner:
    """
    Learn optimal compression strategies per page type.

    Tracks compression results and learns patterns that can be
    applied to future compressions of similar page types.

    Example:
        learner = CompressionPatternLearner()
        learner.record_compression_result(
            page_type="search_results",
            url="https://example.com/search",
            compression_ratio=5.2,
            fold_threshold=0.85,
            success=True
        )
        settings = learner.get_optimal_settings("search_results")
    """

    # Minimum samples before pattern is considered reliable
    MIN_SAMPLES_FOR_PATTERN = 10

    # Maximum results to keep in memory per page type
    MAX_RESULTS_PER_TYPE = 500

    def __init__(self, pattern_store: Optional[PatternStore] = None):
        """
        Initialize the compression pattern learner.

        Args:
            pattern_store: Pattern store for persistence. Creates default if None.
        """
        self.pattern_store = pattern_store or PatternStore()

        # In-memory tracking of compression results
        self.compression_ratios: Dict[str, List[float]] = defaultdict(list)
        self.fold_thresholds: Dict[str, List[float]] = defaultdict(list)
        self.success_counts: Dict[str, int] = defaultdict(int)
        self.failure_counts: Dict[str, int] = defaultdict(int)

        # Learned patterns cache
        self.learned_patterns: Dict[str, CompressionPattern] = {}

        # Load persisted patterns
        self._load_persisted_patterns()

    def _load_persisted_patterns(self) -> None:
        """Load previously learned patterns from storage."""
        for key in self.pattern_store.list_keys("compression"):
            data = self.pattern_store.retrieve("compression", key)
            if data:
                # Remove metadata keys before creating pattern
                pattern_data = {
                    k: v for k, v in data.items()
                    if not k.startswith("_")
                }
                self.learned_patterns[key] = CompressionPattern.from_dict(pattern_data)

    def record_compression_result(
        self,
        page_type: str,
        url: str,
        compression_ratio: float,
        fold_threshold: float,
        success: bool,
        raw_size: int = 0,
        compressed_size: int = 0,
    ) -> None:
        """
        Record a compression result for learning.

        Args:
            page_type: The classified page type
            url: The page URL (used for pattern generation)
            compression_ratio: Ratio of raw size to compressed size
            fold_threshold: SimHash threshold that was used
            success: Whether the compression was successful/usable
            raw_size: Original size in characters
            compressed_size: Final size in characters
        """
        if success and compression_ratio > 1.0:
            # Track successful compression
            ratios = self.compression_ratios[page_type]
            thresholds = self.fold_thresholds[page_type]

            ratios.append(compression_ratio)
            thresholds.append(fold_threshold)

            # Bound memory usage
            if len(ratios) > self.MAX_RESULTS_PER_TYPE:
                self.compression_ratios[page_type] = ratios[-self.MAX_RESULTS_PER_TYPE:]
            if len(thresholds) > self.MAX_RESULTS_PER_TYPE:
                self.fold_thresholds[page_type] = thresholds[-self.MAX_RESULTS_PER_TYPE:]

            self.success_counts[page_type] += 1
        else:
            self.failure_counts[page_type] += 1

        # Update learned pattern if enough samples
        total_samples = self.success_counts[page_type] + self.failure_counts[page_type]
        if total_samples >= self.MIN_SAMPLES_FOR_PATTERN:
            # Update every 10 samples or on first threshold crossing
            if (total_samples == self.MIN_SAMPLES_FOR_PATTERN or
                total_samples % 10 == 0):
                self._update_learned_pattern(page_type)

    def _update_learned_pattern(self, page_type: str) -> None:
        """
        Update and persist learned pattern for a page type.

        Args:
            page_type: The page type to update pattern for
        """
        ratios = self.compression_ratios.get(page_type, [])
        thresholds = self.fold_thresholds.get(page_type, [])

        if not ratios:
            return

        success_count = self.success_counts[page_type]
        failure_count = self.failure_counts[page_type]
        total_count = success_count + failure_count

        # Calculate statistics
        avg_ratio = statistics.mean(ratios)
        median_threshold = statistics.median(thresholds) if thresholds else 0.85

        # Determine if aggressive filtering should be used
        # Use aggressive filtering if average compression ratio is high
        use_aggressive = avg_ratio > 5.0

        # Calculate confidence based on sample count and success rate
        success_rate = success_count / max(total_count, 1)
        sample_confidence = min(len(ratios) / 100.0, 1.0)
        confidence = success_rate * sample_confidence

        pattern = CompressionPattern(
            page_type=page_type,
            url_pattern=f".*{page_type}.*",  # Simple pattern
            avg_compression_ratio=avg_ratio,
            optimal_fold_threshold=median_threshold,
            use_aggressive_filtering=use_aggressive,
            sample_count=len(ratios),
            confidence=confidence,
        )

        self.learned_patterns[page_type] = pattern

        # Persist to storage
        self.pattern_store.store("compression", page_type, pattern.to_dict())

    def get_optimal_settings(self, page_type: str) -> Optional[CompressionPattern]:
        """
        Get learned optimal settings for a page type.

        Args:
            page_type: The page type to get settings for

        Returns:
            CompressionPattern if learned, None otherwise
        """
        # Check memory cache first
        if page_type in self.learned_patterns:
            return self.learned_patterns[page_type]

        # Try to load from storage
        data = self.pattern_store.retrieve("compression", page_type)
        if data:
            pattern_data = {k: v for k, v in data.items() if not k.startswith("_")}
            pattern = CompressionPattern.from_dict(pattern_data)
            self.learned_patterns[page_type] = pattern
            return pattern

        return None

    def get_all_patterns(self) -> Dict[str, CompressionPattern]:
        """
        Get all learned compression patterns.

        Returns:
            Dictionary mapping page type to pattern
        """
        # Ensure all persisted patterns are loaded
        for key in self.pattern_store.list_keys("compression"):
            if key not in self.learned_patterns:
                data = self.pattern_store.retrieve("compression", key)
                if data:
                    pattern_data = {k: v for k, v in data.items() if not k.startswith("_")}
                    self.learned_patterns[key] = CompressionPattern.from_dict(pattern_data)

        return dict(self.learned_patterns)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about compression learning.

        Returns:
            Dictionary with learning statistics
        """
        stats = {
            "learned_patterns": len(self.learned_patterns),
            "page_types": {},
        }

        for page_type in set(list(self.compression_ratios.keys()) +
                           list(self.learned_patterns.keys())):
            ratios = self.compression_ratios.get(page_type, [])
            pattern = self.learned_patterns.get(page_type)

            stats["page_types"][page_type] = {
                "samples_in_memory": len(ratios),
                "success_count": self.success_counts.get(page_type, 0),
                "failure_count": self.failure_counts.get(page_type, 0),
                "has_learned_pattern": pattern is not None,
                "avg_compression_ratio": statistics.mean(ratios) if ratios else None,
                "pattern_confidence": pattern.confidence if pattern else None,
            }

        return stats

    def reset_learning(self, page_type: Optional[str] = None) -> None:
        """
        Reset learned patterns.

        Args:
            page_type: Specific page type to reset, or None for all
        """
        if page_type:
            self.compression_ratios.pop(page_type, None)
            self.fold_thresholds.pop(page_type, None)
            self.success_counts.pop(page_type, None)
            self.failure_counts.pop(page_type, None)
            self.learned_patterns.pop(page_type, None)
            self.pattern_store.delete("compression", page_type)
        else:
            self.compression_ratios.clear()
            self.fold_thresholds.clear()
            self.success_counts.clear()
            self.failure_counts.clear()
            self.learned_patterns.clear()
            for key in self.pattern_store.list_keys("compression"):
                self.pattern_store.delete("compression", key)

"""List folding pattern learning for token optimization.

This module learns optimal SimHash thresholds for list folding
based on observed token savings and list characteristics.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import statistics

from .pattern_store import PatternStore


@dataclass
class FoldingPattern:
    """
    Learned folding pattern for list structures.

    Attributes:
        list_type: The type of list (e.g., "product_grid", "search_results")
        optimal_threshold: Learned optimal SimHash threshold
        min_items_to_fold: Minimum items before folding is beneficial
        avg_items_per_page: Average list items observed
        fold_effectiveness: Average tokens saved per original item
        sample_count: Number of samples used
    """
    list_type: str
    optimal_threshold: float
    min_items_to_fold: int
    avg_items_per_page: int
    fold_effectiveness: float
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "list_type": self.list_type,
            "optimal_threshold": self.optimal_threshold,
            "min_items_to_fold": self.min_items_to_fold,
            "avg_items_per_page": self.avg_items_per_page,
            "fold_effectiveness": self.fold_effectiveness,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FoldingPattern":
        """Create from dictionary."""
        return cls(
            list_type=data.get("list_type", "unknown"),
            optimal_threshold=data.get("optimal_threshold", 0.85),
            min_items_to_fold=data.get("min_items_to_fold", 5),
            avg_items_per_page=data.get("avg_items_per_page", 0),
            fold_effectiveness=data.get("fold_effectiveness", 0.0),
            sample_count=data.get("sample_count", 0),
        )


@dataclass
class FoldResult:
    """
    A single fold operation result for learning.

    Attributes:
        threshold: SimHash threshold used
        items_original: Number of items before folding
        items_folded: Number of items after folding
        tokens_saved: Estimated tokens saved
        effectiveness: Tokens saved per original item
    """
    threshold: float
    items_original: int
    items_folded: int
    tokens_saved: int
    effectiveness: float


class FoldingPatternLearner:
    """
    Learn optimal list folding thresholds per list type.

    Tracks fold operation results and learns patterns that optimize
    token savings while preserving necessary information.

    Example:
        learner = FoldingPatternLearner()
        learner.record_fold_result(
            list_type="product_grid",
            threshold=0.85,
            items_original=50,
            items_folded=5,
            tokens_saved=450
        )
        optimal = learner.get_optimal_threshold("product_grid")
    """

    # Default threshold when no pattern is learned
    DEFAULT_THRESHOLD = 0.85

    # Minimum samples before pattern is reliable
    MIN_SAMPLES_FOR_PATTERN = 5

    # Maximum results to keep per list type
    MAX_RESULTS_PER_TYPE = 200

    # Minimum items before folding is considered
    MIN_ITEMS_TO_FOLD = 3

    def __init__(self, pattern_store: Optional[PatternStore] = None):
        """
        Initialize the folding pattern learner.

        Args:
            pattern_store: Pattern store for persistence. Creates default if None.
        """
        self.pattern_store = pattern_store or PatternStore()

        # In-memory tracking of fold results
        self.fold_results: Dict[str, List[FoldResult]] = defaultdict(list)

        # Learned patterns cache
        self.learned_patterns: Dict[str, FoldingPattern] = {}

        # Load persisted patterns
        self._load_persisted_patterns()

    def _load_persisted_patterns(self) -> None:
        """Load previously learned folding patterns."""
        for key in self.pattern_store.list_keys("folding"):
            data = self.pattern_store.retrieve("folding", key)
            if data:
                if "results" in data:
                    # Load results for continued learning
                    for result_data in data["results"]:
                        self.fold_results[key].append(FoldResult(
                            threshold=result_data.get("threshold", 0.85),
                            items_original=result_data.get("items", 0),
                            items_folded=result_data.get("items_folded", 0),
                            tokens_saved=result_data.get("tokens_saved", 0),
                            effectiveness=result_data.get("effectiveness", 0.0),
                        ))
                    # Rebuild pattern
                    self._update_learned_pattern(key)
                else:
                    # Old format - just pattern data
                    pattern_data = {k: v for k, v in data.items() if not k.startswith("_")}
                    self.learned_patterns[key] = FoldingPattern.from_dict(pattern_data)

    def record_fold_result(
        self,
        list_type: str,
        threshold: float,
        items_original: int,
        items_folded: int,
        tokens_saved: int,
    ) -> None:
        """
        Record a fold operation result for learning.

        Args:
            list_type: The type of list (derived from page type or element class)
            threshold: SimHash threshold used for folding
            items_original: Number of items before folding
            items_folded: Number of items after folding (includes summary item)
            tokens_saved: Estimated number of tokens saved
        """
        # Calculate effectiveness
        effectiveness = tokens_saved / max(items_original, 1)

        result = FoldResult(
            threshold=threshold,
            items_original=items_original,
            items_folded=items_folded,
            tokens_saved=tokens_saved,
            effectiveness=effectiveness,
        )

        results = self.fold_results[list_type]
        results.append(result)

        # Bound memory usage
        if len(results) > self.MAX_RESULTS_PER_TYPE:
            self.fold_results[list_type] = results[-self.MAX_RESULTS_PER_TYPE:]

        # Persist periodically
        if len(self.fold_results[list_type]) % 20 == 0:
            self._persist_results(list_type)

        # Update pattern if enough samples
        if len(self.fold_results[list_type]) >= self.MIN_SAMPLES_FOR_PATTERN:
            if (len(self.fold_results[list_type]) == self.MIN_SAMPLES_FOR_PATTERN or
                len(self.fold_results[list_type]) % 10 == 0):
                self._update_learned_pattern(list_type)

    def _persist_results(self, list_type: str) -> None:
        """
        Persist fold results to storage.

        Args:
            list_type: The list type key
        """
        results = self.fold_results.get(list_type, [])
        results_to_store = results[-self.MAX_RESULTS_PER_TYPE:]

        self.pattern_store.store("folding", list_type, {
            "results": [
                {
                    "threshold": r.threshold,
                    "items": r.items_original,
                    "items_folded": r.items_folded,
                    "tokens_saved": r.tokens_saved,
                    "effectiveness": r.effectiveness,
                }
                for r in results_to_store
            ]
        })

    def _update_learned_pattern(self, list_type: str) -> None:
        """
        Update learned pattern from results.

        Args:
            list_type: The list type key
        """
        results = self.fold_results.get(list_type, [])
        if not results:
            return

        # Group results by threshold to find best one
        threshold_effectiveness: Dict[float, List[float]] = defaultdict(list)
        for r in results:
            # Round threshold to nearest 0.05 for grouping
            rounded_threshold = round(r.threshold * 20) / 20
            threshold_effectiveness[rounded_threshold].append(r.effectiveness)

        # Find threshold with best average effectiveness
        best_threshold = self.DEFAULT_THRESHOLD
        best_avg_effectiveness = 0.0

        for threshold, effectivenesses in threshold_effectiveness.items():
            avg_eff = statistics.mean(effectivenesses)
            if avg_eff > best_avg_effectiveness:
                best_avg_effectiveness = avg_eff
                best_threshold = threshold

        # Calculate other pattern values
        all_items = [r.items_original for r in results]
        all_effectiveness = [r.effectiveness for r in results]

        avg_items = int(statistics.mean(all_items))
        avg_effectiveness = statistics.mean(all_effectiveness)

        # Determine minimum items to fold (items where folding was effective)
        effective_results = [r for r in results if r.effectiveness > 5.0]  # > 5 tokens/item
        if effective_results:
            min_items = min(r.items_original for r in effective_results)
        else:
            min_items = self.MIN_ITEMS_TO_FOLD

        pattern = FoldingPattern(
            list_type=list_type,
            optimal_threshold=best_threshold,
            min_items_to_fold=max(min_items, self.MIN_ITEMS_TO_FOLD),
            avg_items_per_page=avg_items,
            fold_effectiveness=avg_effectiveness,
            sample_count=len(results),
        )

        self.learned_patterns[list_type] = pattern

    def get_optimal_threshold(self, list_type: str, default: float = 0.85) -> float:
        """
        Get learned optimal threshold for a list type.

        Args:
            list_type: The list type to get threshold for
            default: Default threshold if no pattern is learned

        Returns:
            Optimal SimHash threshold
        """
        results = self.fold_results.get(list_type, [])

        if len(results) < self.MIN_SAMPLES_FOR_PATTERN:
            return default

        pattern = self.learned_patterns.get(list_type)
        if pattern:
            return pattern.optimal_threshold

        # Calculate on the fly if not cached
        threshold_effectiveness: Dict[float, List[float]] = defaultdict(list)
        for r in results:
            rounded_threshold = round(r.threshold * 20) / 20
            threshold_effectiveness[rounded_threshold].append(r.effectiveness)

        best_threshold = default
        best_avg = 0.0

        for threshold, effectivenesses in threshold_effectiveness.items():
            avg = statistics.mean(effectivenesses)
            if avg > best_avg:
                best_avg = avg
                best_threshold = threshold

        return best_threshold

    def get_pattern(self, list_type: str) -> Optional[FoldingPattern]:
        """
        Get the full learned pattern for a list type.

        Args:
            list_type: The list type

        Returns:
            FoldingPattern if available, None otherwise
        """
        return self.learned_patterns.get(list_type)

    def should_fold(self, list_type: str, item_count: int) -> bool:
        """
        Determine if a list should be folded based on learned patterns.

        Args:
            list_type: The list type
            item_count: Number of items in the list

        Returns:
            True if folding is recommended
        """
        pattern = self.learned_patterns.get(list_type)

        if pattern:
            return item_count >= pattern.min_items_to_fold

        return item_count >= self.MIN_ITEMS_TO_FOLD

    def get_all_patterns(self) -> Dict[str, FoldingPattern]:
        """
        Get all learned folding patterns.

        Returns:
            Dictionary mapping list type to pattern
        """
        return dict(self.learned_patterns)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about folding learning.

        Returns:
            Dictionary with learning statistics
        """
        stats = {
            "learned_patterns": len(self.learned_patterns),
            "total_results": sum(len(r) for r in self.fold_results.values()),
            "list_types": {},
        }

        for list_type in set(list(self.fold_results.keys()) +
                           list(self.learned_patterns.keys())):
            results = self.fold_results.get(list_type, [])
            pattern = self.learned_patterns.get(list_type)

            if results:
                avg_items = statistics.mean(r.items_original for r in results)
                avg_saved = statistics.mean(r.tokens_saved for r in results)
                avg_eff = statistics.mean(r.effectiveness for r in results)
            else:
                avg_items = avg_saved = avg_eff = None

            stats["list_types"][list_type] = {
                "samples": len(results),
                "avg_items_original": avg_items,
                "avg_tokens_saved": avg_saved,
                "avg_effectiveness": avg_eff,
                "has_learned_pattern": pattern is not None,
                "optimal_threshold": pattern.optimal_threshold if pattern else None,
            }

        return stats

    def reset_learning(self, list_type: Optional[str] = None) -> None:
        """
        Reset learned patterns.

        Args:
            list_type: Specific list type to reset, or None for all
        """
        if list_type:
            self.fold_results.pop(list_type, None)
            self.learned_patterns.pop(list_type, None)
            self.pattern_store.delete("folding", list_type)
        else:
            self.fold_results.clear()
            self.learned_patterns.clear()
            for key in self.pattern_store.list_keys("folding"):
                self.pattern_store.delete("folding", key)

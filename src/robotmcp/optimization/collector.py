"""Centralized performance metrics collection for rf-mcp.

This module provides a unified interface for collecting and analyzing
performance metrics across all optimization components.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
import statistics
import time

from .pattern_store import PatternStore
from .compression_learner import CompressionPatternLearner
from .folding_learner import FoldingPatternLearner
from .timeout_learner import TimeoutPatternLearner
from .ref_learner import RefUsageLearner


# Token metric targets
TOKEN_TARGETS = {
    "min_reduction_percent": 70.0,    # Minimum acceptable reduction
    "target_reduction_percent": 85.0, # Target reduction
    "max_reduction_percent": 95.0,    # Aggressive optimization ceiling
    "incremental_diff_target": 50.0,  # Additional reduction via diffs
    "list_fold_threshold": 0.85,      # SimHash similarity threshold
}

# Latency targets (in milliseconds)
LATENCY_TARGETS = {
    "aria_snapshot_max_ms": 200,       # Max acceptable snapshot time
    "ref_lookup_max_ms": 1,            # Ref lookup must be sub-millisecond
    "diff_compute_max_ms": 50,         # Diff should be fast
    "prevalidation_max_ms": 100,       # Quick actionability check
    "total_overhead_max_ms": 500,      # Total optimization overhead budget
}

# Effectiveness targets
EFFECTIVENESS_TARGETS = {
    "min_ref_hit_rate": 0.95,          # 95% ref lookup success
    "max_stale_ref_rate": 0.05,        # Max 5% stale refs
    "min_prevalidation_accuracy": 0.90,# 90% prediction accuracy
    "min_diff_accuracy": 0.99,         # 99% correct diffs
}

# Memory targets
MEMORY_TARGETS = {
    "max_refs_in_memory": 10000,       # Max refs before cleanup
    "max_snapshots_cached": 50,        # Max snapshots in cache
    "max_total_memory_mb": 50,         # Total memory budget
    "ref_memory_budget_bytes": 200,    # Budget per ref entry
}


@dataclass
class TokenMetrics:
    """
    Track token consumption per tool response.

    Attributes:
        raw_html_tokens: Token count of raw HTML DOM
        raw_html_chars: Character count of raw HTML
        aria_snapshot_tokens: Token count of ARIA snapshot
        aria_snapshot_chars: Character count of ARIA snapshot
        token_reduction_percent: Percentage reduction from raw to ARIA
        compression_ratio: Ratio of raw to ARIA size
        full_snapshot_tokens: Full snapshot token count
        diff_tokens: Incremental diff token count
        diff_reduction_percent: Additional savings from diff
        list_items_original: Original list item count
        list_items_folded: List items after folding
        fold_compression_ratio: Original / folded ratio
    """
    raw_html_tokens: int
    raw_html_chars: int
    aria_snapshot_tokens: int
    aria_snapshot_chars: int
    token_reduction_percent: float
    compression_ratio: float
    full_snapshot_tokens: int = 0
    diff_tokens: int = 0
    diff_reduction_percent: float = 0.0
    list_items_original: int = 0
    list_items_folded: int = 0
    fold_compression_ratio: float = 1.0


@dataclass
class LatencyMetrics:
    """
    Track latency at each optimization stage.

    Attributes:
        aria_snapshot_latency_ms: Time to capture ARIA snapshot
        ref_lookup_latency_ms: Time to resolve ref to locator
        diff_compute_latency_ms: Time to compute incremental diff
        prevalidation_latency_ms: Time for actionability checks
        total_optimization_overhead_ms: Total time for all optimizations
    """
    aria_snapshot_latency_ms: float = 0.0
    ref_lookup_latency_ms: float = 0.0
    diff_compute_latency_ms: float = 0.0
    prevalidation_latency_ms: float = 0.0
    total_optimization_overhead_ms: float = 0.0


@dataclass
class PerformanceMetricsCollector:
    """
    Centralized performance metrics collection.

    Coordinates all learning components and provides a unified
    interface for recording and analyzing optimization metrics.

    Example:
        collector = PerformanceMetricsCollector()
        collector.record_snapshot_generation(
            raw_chars=10000,
            aria_chars=1500,
            latency_ms=150,
            page_type="search_results",
            fold_threshold=0.85
        )
        summary = collector.get_summary()
    """

    # Pattern store for persistence
    pattern_store: PatternStore = field(default_factory=PatternStore)

    # Pattern learners
    compression_learner: Optional[CompressionPatternLearner] = None
    folding_learner: Optional[FoldingPatternLearner] = None
    timeout_learner: Optional[TimeoutPatternLearner] = None
    ref_learner: Optional[RefUsageLearner] = None

    # Token metrics history
    token_metrics: List[TokenMetrics] = field(default_factory=list)

    # Latency metrics history
    latency_metrics: List[LatencyMetrics] = field(default_factory=list)

    # Ref lookup tracking
    ref_lookups_total: int = 0
    ref_lookups_success: int = 0
    ref_lookups_stale: int = 0

    # Prevalidation tracking
    prevalidation_results: List[Dict[str, Any]] = field(default_factory=list)

    # Timeout tracking
    timeout_results: List[Dict[str, Any]] = field(default_factory=list)

    # Maximum history to keep in memory
    _max_metrics_history: int = 1000

    def __post_init__(self):
        """Initialize learners with shared pattern store."""
        if self.compression_learner is None:
            self.compression_learner = CompressionPatternLearner(self.pattern_store)
        if self.folding_learner is None:
            self.folding_learner = FoldingPatternLearner(self.pattern_store)
        if self.timeout_learner is None:
            self.timeout_learner = TimeoutPatternLearner(self.pattern_store)
        if self.ref_learner is None:
            self.ref_learner = RefUsageLearner(self.pattern_store)

    def record_snapshot_generation(
        self,
        raw_chars: int,
        aria_chars: int,
        latency_ms: float,
        page_type: str,
        fold_threshold: float,
        list_items_original: int = 0,
        list_items_folded: int = 0,
    ) -> None:
        """
        Record a snapshot generation event.

        Args:
            raw_chars: Character count of raw HTML
            aria_chars: Character count of ARIA snapshot
            latency_ms: Time taken in milliseconds
            page_type: Classified page type
            fold_threshold: SimHash threshold used
            list_items_original: Original list item count
            list_items_folded: List items after folding
        """
        # Estimate tokens (rough: 4 chars per token)
        raw_tokens = raw_chars // 4
        aria_tokens = aria_chars // 4

        # Calculate metrics
        reduction = ((raw_tokens - aria_tokens) / max(raw_tokens, 1)) * 100
        compression = raw_tokens / max(aria_tokens, 1)
        fold_ratio = list_items_original / max(list_items_folded, 1)

        token_metric = TokenMetrics(
            raw_html_tokens=raw_tokens,
            raw_html_chars=raw_chars,
            aria_snapshot_tokens=aria_tokens,
            aria_snapshot_chars=aria_chars,
            token_reduction_percent=reduction,
            compression_ratio=compression,
            full_snapshot_tokens=aria_tokens,
            diff_tokens=0,
            diff_reduction_percent=0,
            list_items_original=list_items_original,
            list_items_folded=list_items_folded,
            fold_compression_ratio=fold_ratio,
        )

        self.token_metrics.append(token_metric)

        # Bound history
        if len(self.token_metrics) > self._max_metrics_history:
            self.token_metrics = self.token_metrics[-self._max_metrics_history:]

        # Record latency
        latency_metric = LatencyMetrics(
            aria_snapshot_latency_ms=latency_ms,
            total_optimization_overhead_ms=latency_ms,
        )
        self.latency_metrics.append(latency_metric)

        if len(self.latency_metrics) > self._max_metrics_history:
            self.latency_metrics = self.latency_metrics[-self._max_metrics_history:]

        # Forward to compression learner
        self.compression_learner.record_compression_result(
            page_type=page_type,
            url="",
            compression_ratio=compression,
            fold_threshold=fold_threshold,
            success=True,
            raw_size=raw_chars,
            compressed_size=aria_chars,
        )

        # Forward to folding learner if applicable
        if list_items_original > 0:
            tokens_saved = (list_items_original - list_items_folded) * 10  # Estimate
            self.folding_learner.record_fold_result(
                list_type=page_type,
                threshold=fold_threshold,
                items_original=list_items_original,
                items_folded=list_items_folded,
                tokens_saved=tokens_saved,
            )

    def record_ref_lookup(
        self,
        ref: str,
        success: bool,
        stale: bool,
        latency_ms: float,
        page_type: str,
        session_id: str,
    ) -> None:
        """
        Record a ref lookup event.

        Args:
            ref: The ref that was looked up
            success: Whether the lookup succeeded
            stale: Whether the ref was stale
            latency_ms: Lookup time in milliseconds
            page_type: Classified page type
            session_id: Session identifier
        """
        self.ref_lookups_total += 1

        if success:
            self.ref_lookups_success += 1
        if stale:
            self.ref_lookups_stale += 1

        # Record latency
        if self.latency_metrics:
            self.latency_metrics[-1].ref_lookup_latency_ms = latency_ms
        else:
            self.latency_metrics.append(LatencyMetrics(ref_lookup_latency_ms=latency_ms))

        # Forward to ref learner
        self.ref_learner.record_ref_access(page_type, ref, session_id)

    def record_prevalidation(
        self,
        predicted: bool,
        actual: bool,
        latency_ms: float,
    ) -> None:
        """
        Record a pre-validation result.

        Args:
            predicted: Whether the element was predicted to be actionable
            actual: Whether the element was actually actionable
            latency_ms: Prevalidation time in milliseconds
        """
        self.prevalidation_results.append({
            "predicted": predicted,
            "actual": actual,
            "latency_ms": latency_ms,
            "correct": predicted == actual,
            "timestamp": time.time(),
        })

        # Bound history
        if len(self.prevalidation_results) > self._max_metrics_history:
            self.prevalidation_results = self.prevalidation_results[-self._max_metrics_history:]

        # Record latency
        if self.latency_metrics:
            self.latency_metrics[-1].prevalidation_latency_ms = latency_ms

    def record_timeout_result(
        self,
        page_type: str,
        action: str,
        timeout_ms: int,
        actual_ms: int,
        success: bool,
        triggered: bool,
    ) -> None:
        """
        Record a timeout result.

        Args:
            page_type: Classified page type
            action: The action performed
            timeout_ms: Configured timeout
            actual_ms: Actual duration
            success: Whether the operation succeeded
            triggered: Whether the timeout was triggered
        """
        self.timeout_results.append({
            "page_type": page_type,
            "action": action,
            "timeout_ms": timeout_ms,
            "actual_ms": actual_ms,
            "success": success,
            "triggered": triggered,
            "timestamp": time.time(),
        })

        # Bound history
        if len(self.timeout_results) > self._max_metrics_history:
            self.timeout_results = self.timeout_results[-self._max_metrics_history:]

        # Forward to timeout learner
        self.timeout_learner.record_timeout_result(
            page_type=page_type,
            action=action,
            timeout_ms=timeout_ms,
            duration_ms=actual_ms,
            success=success,
            triggered=triggered,
        )

    def record_diff_computation(
        self,
        full_tokens: int,
        diff_tokens: int,
        latency_ms: float,
    ) -> None:
        """
        Record an incremental diff computation.

        Args:
            full_tokens: Token count of full snapshot
            diff_tokens: Token count of diff
            latency_ms: Diff computation time
        """
        if self.token_metrics:
            metric = self.token_metrics[-1]
            metric.full_snapshot_tokens = full_tokens
            metric.diff_tokens = diff_tokens
            metric.diff_reduction_percent = (
                (full_tokens - diff_tokens) / max(full_tokens, 1) * 100
            )

        if self.latency_metrics:
            self.latency_metrics[-1].diff_compute_latency_ms = latency_ms

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all collected metrics.

        Returns:
            Dictionary with comprehensive metrics summary
        """
        # Token metrics summary
        if self.token_metrics:
            avg_reduction = statistics.mean(
                m.token_reduction_percent for m in self.token_metrics
            )
            avg_compression = statistics.mean(
                m.compression_ratio for m in self.token_metrics
            )
        else:
            avg_reduction = 0.0
            avg_compression = 1.0

        # Ref metrics
        ref_hit_rate = self.ref_lookups_success / max(self.ref_lookups_total, 1)
        stale_rate = self.ref_lookups_stale / max(self.ref_lookups_total, 1)

        # Prevalidation accuracy
        prevalidation_accuracy = 0.0
        if self.prevalidation_results:
            correct = sum(1 for r in self.prevalidation_results if r["correct"])
            prevalidation_accuracy = correct / len(self.prevalidation_results)

        # Latency summary
        avg_snapshot_latency = 0.0
        avg_ref_latency = 0.0
        if self.latency_metrics:
            snapshot_latencies = [
                m.aria_snapshot_latency_ms for m in self.latency_metrics
                if m.aria_snapshot_latency_ms > 0
            ]
            if snapshot_latencies:
                avg_snapshot_latency = statistics.mean(snapshot_latencies)

            ref_latencies = [
                m.ref_lookup_latency_ms for m in self.latency_metrics
                if m.ref_lookup_latency_ms > 0
            ]
            if ref_latencies:
                avg_ref_latency = statistics.mean(ref_latencies)

        # Timeout summary
        timeout_triggered_count = sum(
            1 for r in self.timeout_results if r["triggered"]
        )
        timeout_success_rate = sum(
            1 for r in self.timeout_results if r["success"]
        ) / max(len(self.timeout_results), 1)

        return {
            "token_metrics": {
                "avg_token_reduction_percent": round(avg_reduction, 2),
                "avg_compression_ratio": round(avg_compression, 2),
                "samples": len(self.token_metrics),
                "target_met": avg_reduction >= TOKEN_TARGETS["min_reduction_percent"],
                "target_percent": TOKEN_TARGETS["min_reduction_percent"],
            },
            "ref_metrics": {
                "total_lookups": self.ref_lookups_total,
                "hit_rate": round(ref_hit_rate, 4),
                "stale_rate": round(stale_rate, 4),
                "target_met": ref_hit_rate >= EFFECTIVENESS_TARGETS["min_ref_hit_rate"],
                "target_hit_rate": EFFECTIVENESS_TARGETS["min_ref_hit_rate"],
            },
            "prevalidation_metrics": {
                "accuracy": round(prevalidation_accuracy, 4),
                "samples": len(self.prevalidation_results),
                "target_met": prevalidation_accuracy >= EFFECTIVENESS_TARGETS["min_prevalidation_accuracy"],
                "target_accuracy": EFFECTIVENESS_TARGETS["min_prevalidation_accuracy"],
            },
            "latency_metrics": {
                "avg_snapshot_latency_ms": round(avg_snapshot_latency, 2),
                "avg_ref_lookup_latency_ms": round(avg_ref_latency, 4),
                "samples": len(self.latency_metrics),
                "snapshot_target_met": avg_snapshot_latency <= LATENCY_TARGETS["aria_snapshot_max_ms"],
                "ref_target_met": avg_ref_latency <= LATENCY_TARGETS["ref_lookup_max_ms"],
            },
            "timeout_metrics": {
                "total_operations": len(self.timeout_results),
                "timeout_triggered_count": timeout_triggered_count,
                "success_rate": round(timeout_success_rate, 4),
            },
            "learning_status": {
                "compression_patterns": len(self.compression_learner.learned_patterns),
                "folding_patterns": len(self.folding_learner.learned_patterns),
                "timeout_patterns": len(self.timeout_learner.learned_patterns),
                "ref_page_types": len(self.ref_learner.ref_frequencies),
            },
        }

    def get_detailed_report(self) -> Dict[str, Any]:
        """
        Get a detailed performance report.

        Returns:
            Comprehensive report with all metrics and learner stats
        """
        summary = self.get_summary()

        return {
            **summary,
            "compression_learner_stats": self.compression_learner.get_statistics(),
            "folding_learner_stats": self.folding_learner.get_statistics(),
            "timeout_learner_stats": self.timeout_learner.get_statistics(),
            "ref_learner_stats": self.ref_learner.get_statistics(),
            "pattern_store_stats": self.pattern_store.get_storage_stats(),
        }

    def persist_all_patterns(self) -> None:
        """Persist all learned patterns and metrics to storage."""
        # Persist ref learner data
        self.ref_learner.persist_all()

        # Save metrics summary
        summary = self.get_summary()
        timestamp = datetime.now().strftime("%Y%m%d-%H")
        self.pattern_store.store("metrics", f"summary-{timestamp}", summary)

    def get_optimization_recommendations(self, page_type: str) -> Dict[str, Any]:
        """
        Get optimization recommendations for a page type.

        Combines learned patterns from all learners to provide
        comprehensive optimization settings.

        Args:
            page_type: The page type to get recommendations for

        Returns:
            Dictionary with all optimization recommendations
        """
        recommendations = {
            "page_type": page_type,
            "compression": {},
            "folding": {},
            "timeouts": {},
            "ref_preload": [],
        }

        # Compression recommendations
        compression_pattern = self.compression_learner.get_optimal_settings(page_type)
        if compression_pattern:
            recommendations["compression"] = {
                "fold_threshold": compression_pattern.optimal_fold_threshold,
                "use_aggressive_filtering": compression_pattern.use_aggressive_filtering,
                "expected_compression_ratio": compression_pattern.avg_compression_ratio,
                "confidence": compression_pattern.confidence,
            }
        else:
            recommendations["compression"] = {
                "fold_threshold": TOKEN_TARGETS["list_fold_threshold"],
                "use_aggressive_filtering": False,
                "expected_compression_ratio": None,
                "confidence": 0.0,
            }

        # Folding recommendations
        folding_pattern = self.folding_learner.get_pattern(page_type)
        if folding_pattern:
            recommendations["folding"] = {
                "optimal_threshold": folding_pattern.optimal_threshold,
                "min_items_to_fold": folding_pattern.min_items_to_fold,
                "expected_effectiveness": folding_pattern.fold_effectiveness,
            }
        else:
            recommendations["folding"] = {
                "optimal_threshold": self.folding_learner.get_optimal_threshold(page_type),
                "min_items_to_fold": 5,
                "expected_effectiveness": None,
            }

        # Timeout recommendations for common actions
        for action in ["click", "fill", "navigate"]:
            timeouts = self.timeout_learner.get_optimal_timeouts(page_type, action)
            recommendations["timeouts"][action] = timeouts

        # Ref preload candidates
        recommendations["ref_preload"] = self.ref_learner.get_preload_candidates(
            page_type, top_n=10
        )

        return recommendations

    def reset_all_learning(self) -> None:
        """Reset all learned patterns and metrics."""
        self.compression_learner.reset_learning()
        self.folding_learner.reset_learning()
        self.timeout_learner.reset_learning()
        self.ref_learner.reset_learning()

        self.token_metrics.clear()
        self.latency_metrics.clear()
        self.prevalidation_results.clear()
        self.timeout_results.clear()
        self.ref_lookups_total = 0
        self.ref_lookups_success = 0
        self.ref_lookups_stale = 0

"""Tests for optimization and self-learning system.

This module tests the self-learning optimization functionality:
- PatternStore (persistent pattern storage)
- CompressionPatternLearner (learns optimal compression settings)
- TimeoutPatternLearner (learns optimal timeout values)
- PerformanceMetricsCollector (tracks performance metrics)
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


# =============================================================================
# Domain Models (to be moved to production code)
# =============================================================================


class PatternStore:
    """Persistent storage for learned patterns.

    Stores pattern data as JSON files with in-memory caching.
    """

    def __init__(self, storage_dir: Path):
        self._storage_dir = storage_dir
        self._cache: Dict[str, Any] = {}
        self._ensure_storage_dir()

    def _ensure_storage_dir(self) -> None:
        """Create storage directory if it doesn't exist."""
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, key: str) -> Path:
        """Get file path for a key."""
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self._storage_dir / f"{safe_key}.json"

    def store(self, key: str, value: Any) -> None:
        """Store a value with a key.

        Args:
            key: The key to store under
            value: The value to store (must be JSON-serializable)
        """
        file_path = self._get_file_path(key)
        with open(file_path, "w") as f:
            json.dump(value, f, indent=2)
        self._cache[key] = value

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value by key.

        Args:
            key: The key to retrieve

        Returns:
            The stored value, or None if not found
        """
        # Check cache first
        if key in self._cache:
            return self._cache[key]

        # Try to load from disk
        file_path = self._get_file_path(key)
        if file_path.exists():
            with open(file_path, "r") as f:
                value = json.load(f)
            self._cache[key] = value
            return value

        return None

    def list_keys(self) -> List[str]:
        """List all stored keys.

        Returns:
            List of all keys
        """
        keys = []
        for file_path in self._storage_dir.glob("*.json"):
            key = file_path.stem
            keys.append(key)
        return keys

    def delete(self, key: str) -> bool:
        """Delete a stored value.

        Args:
            key: The key to delete

        Returns:
            True if deleted, False if key not found
        """
        file_path = self._get_file_path(key)
        if file_path.exists():
            file_path.unlink()
            self._cache.pop(key, None)
            return True
        return False

    def cleanup(self, max_age_seconds: float = 86400 * 7) -> int:
        """Remove old entries.

        Args:
            max_age_seconds: Maximum age of entries to keep (default: 7 days)

        Returns:
            Number of entries removed
        """
        current_time = time.time()
        removed = 0

        for file_path in self._storage_dir.glob("*.json"):
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                key = file_path.stem
                file_path.unlink()
                self._cache.pop(key, None)
                removed += 1

        return removed

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()


# Mock version for conftest.py import
MockPatternStore = PatternStore


@dataclass
class CompressionResult:
    """Result of a compression operation."""

    page_type: str
    original_tokens: int
    compressed_tokens: int
    settings: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio (0-1, higher is better)."""
        if self.original_tokens == 0:
            return 0.0
        return 1 - (self.compressed_tokens / self.original_tokens)


class CompressionPatternLearner:
    """Learns optimal compression settings from results.

    Tracks compression results and determines optimal settings
    based on historical data.
    """

    MIN_SAMPLES_FOR_RECOMMENDATION = 5

    def __init__(self, pattern_store: PatternStore):
        self._store = pattern_store
        self._results: Dict[str, List[CompressionResult]] = {}
        self._load_from_store()

    def _load_from_store(self) -> None:
        """Load existing results from pattern store."""
        data = self._store.retrieve("compression_patterns")
        if data:
            for page_type, results in data.items():
                self._results[page_type] = [
                    CompressionResult(**r) for r in results
                ]

    def _save_to_store(self) -> None:
        """Save results to pattern store."""
        data = {
            page_type: [
                {
                    "page_type": r.page_type,
                    "original_tokens": r.original_tokens,
                    "compressed_tokens": r.compressed_tokens,
                    "settings": r.settings,
                    "timestamp": r.timestamp,
                }
                for r in results
            ]
            for page_type, results in self._results.items()
        }
        self._store.store("compression_patterns", data)

    def record_result(self, result: CompressionResult) -> None:
        """Record a compression result for learning.

        Args:
            result: The compression result to record
        """
        if result.page_type not in self._results:
            self._results[result.page_type] = []
        self._results[result.page_type].append(result)
        self._save_to_store()

    def get_optimal_settings(self, page_type: str) -> Optional[Dict[str, Any]]:
        """Get optimal compression settings for a page type.

        Args:
            page_type: The type of page

        Returns:
            Optimal settings dict, or None if not enough data
        """
        results = self._results.get(page_type, [])

        if len(results) < self.MIN_SAMPLES_FOR_RECOMMENDATION:
            return None

        # Find settings with best compression ratio
        best_result = max(results, key=lambda r: r.compression_ratio)
        return best_result.settings

    def get_average_compression_ratio(self, page_type: str) -> Optional[float]:
        """Get average compression ratio for a page type.

        Args:
            page_type: The type of page

        Returns:
            Average compression ratio, or None if no data
        """
        results = self._results.get(page_type, [])
        if not results:
            return None
        return statistics.mean(r.compression_ratio for r in results)

    def get_all_page_types(self) -> List[str]:
        """Get all known page types."""
        return list(self._results.keys())


@dataclass
class TimeoutResult:
    """Result of a timeout-related operation."""

    action: str
    actual_duration_ms: int
    success: bool
    configured_timeout_ms: Optional[int] = None
    timestamp: float = field(default_factory=time.time)


class TimeoutPatternLearner:
    """Learns optimal timeout values from execution results.

    Uses percentile-based analysis to recommend timeouts that
    will succeed for most operations while not being excessively long.
    """

    MIN_SAMPLES_FOR_RECOMMENDATION = 10
    DEFAULT_PERCENTILE = 95  # P95 for timeout recommendations

    def __init__(self, pattern_store: PatternStore):
        self._store = pattern_store
        self._results: Dict[str, List[TimeoutResult]] = {}
        self._load_from_store()

    def _load_from_store(self) -> None:
        """Load existing results from pattern store."""
        data = self._store.retrieve("timeout_patterns")
        if data:
            for action, results in data.items():
                self._results[action] = [TimeoutResult(**r) for r in results]

    def _save_to_store(self) -> None:
        """Save results to pattern store."""
        data = {
            action: [
                {
                    "action": r.action,
                    "actual_duration_ms": r.actual_duration_ms,
                    "success": r.success,
                    "configured_timeout_ms": r.configured_timeout_ms,
                    "timestamp": r.timestamp,
                }
                for r in results
            ]
            for action, results in self._results.items()
        }
        self._store.store("timeout_patterns", data)

    def record_result(self, result: TimeoutResult) -> None:
        """Record a timeout result for learning.

        Args:
            result: The timeout result to record
        """
        if result.action not in self._results:
            self._results[result.action] = []
        self._results[result.action].append(result)
        self._save_to_store()

    def get_optimal_timeout(
        self, action: str, percentile: int = None
    ) -> Optional[int]:
        """Get optimal timeout for an action based on P95.

        Args:
            action: The action type
            percentile: Percentile to use (default: 95)

        Returns:
            Recommended timeout in ms, or None if not enough data
        """
        percentile = percentile or self.DEFAULT_PERCENTILE
        results = self._results.get(action, [])

        # Only consider successful results for timeout calculation
        successful = [r for r in results if r.success]

        if len(successful) < self.MIN_SAMPLES_FOR_RECOMMENDATION:
            return None

        # Calculate percentile of actual durations
        durations = sorted(r.actual_duration_ms for r in successful)
        index = int(len(durations) * percentile / 100)
        p_value = durations[min(index, len(durations) - 1)]

        # Add 20% buffer for safety
        return int(p_value * 1.2)

    def get_optimal_timeouts(self) -> Dict[str, int]:
        """Get optimal timeouts for all known actions.

        Returns:
            Dictionary of action -> recommended timeout
        """
        result = {}
        for action in self._results.keys():
            timeout = self.get_optimal_timeout(action)
            if timeout is not None:
                result[action] = timeout
        return result

    def get_success_rate(self, action: str) -> Optional[float]:
        """Get success rate for an action.

        Args:
            action: The action type

        Returns:
            Success rate (0-1), or None if no data
        """
        results = self._results.get(action, [])
        if not results:
            return None
        successful = sum(1 for r in results if r.success)
        return successful / len(results)


@dataclass
class PerformanceMetric:
    """A single performance metric record."""

    operation: str
    duration_ms: int
    tokens_before: Optional[int] = None
    tokens_after: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class PerformanceMetricsCollector:
    """Collects and analyzes performance metrics.

    Tracks operation performance for optimization analysis.
    """

    def __init__(self, pattern_store: Optional[PatternStore] = None):
        self._store = pattern_store
        self._metrics: List[PerformanceMetric] = []
        if pattern_store:
            self._load_from_store()

    def _load_from_store(self) -> None:
        """Load existing metrics from pattern store."""
        if not self._store:
            return
        data = self._store.retrieve("performance_metrics")
        if data:
            self._metrics = [PerformanceMetric(**m) for m in data]

    def _save_to_store(self) -> None:
        """Save metrics to pattern store."""
        if not self._store:
            return
        data = [
            {
                "operation": m.operation,
                "duration_ms": m.duration_ms,
                "tokens_before": m.tokens_before,
                "tokens_after": m.tokens_after,
                "metadata": m.metadata,
                "timestamp": m.timestamp,
            }
            for m in self._metrics
        ]
        self._store.store("performance_metrics", data)

    def record(self, metric: PerformanceMetric) -> None:
        """Record a performance metric.

        Args:
            metric: The metric to record
        """
        self._metrics.append(metric)
        self._save_to_store()

    def record_snapshot_generation(
        self,
        duration_ms: int,
        tokens_before: int,
        tokens_after: int,
        **metadata,
    ) -> None:
        """Convenience method to record snapshot generation metrics.

        Args:
            duration_ms: Time taken in milliseconds
            tokens_before: Tokens before compression
            tokens_after: Tokens after compression
            **metadata: Additional metadata
        """
        self.record(
            PerformanceMetric(
                operation="snapshot_generation",
                duration_ms=duration_ms,
                tokens_before=tokens_before,
                tokens_after=tokens_after,
                metadata=metadata,
            )
        )

    def get_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics for metrics.

        Args:
            operation: Optional operation filter

        Returns:
            Summary statistics dictionary
        """
        metrics = self._metrics
        if operation:
            metrics = [m for m in metrics if m.operation == operation]

        if not metrics:
            return {"count": 0}

        durations = [m.duration_ms for m in metrics]
        summary = {
            "count": len(metrics),
            "avg_duration_ms": statistics.mean(durations),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
        }

        if len(durations) > 1:
            summary["std_duration_ms"] = statistics.stdev(durations)

        # Token metrics if available
        token_metrics = [
            m for m in metrics if m.tokens_before is not None
        ]
        if token_metrics:
            compressions = [
                1 - (m.tokens_after / m.tokens_before)
                for m in token_metrics
                if m.tokens_before and m.tokens_after
            ]
            if compressions:
                summary["avg_compression_ratio"] = statistics.mean(compressions)

        return summary

    def get_operations(self) -> List[str]:
        """Get list of all recorded operations."""
        return list(set(m.operation for m in self._metrics))


# =============================================================================
# Tests
# =============================================================================


class TestPatternStore:
    """Tests for PatternStore."""

    @pytest.fixture
    def store(self, tmp_path: Path) -> PatternStore:
        """Create a PatternStore with temporary storage."""
        return PatternStore(storage_dir=tmp_path / "patterns")

    def test_store_creates_json_file(self, store: PatternStore, tmp_path: Path):
        """Test that store creates a JSON file."""
        store.store("test_key", {"value": 42})

        file_path = tmp_path / "patterns" / "test_key.json"
        assert file_path.exists()

        with open(file_path) as f:
            data = json.load(f)
        assert data == {"value": 42}

    def test_retrieve_returns_stored_value(self, store: PatternStore):
        """Test retrieving a stored value."""
        store.store("test_key", {"value": 42})
        result = store.retrieve("test_key")

        assert result == {"value": 42}

    def test_retrieve_returns_none_for_missing(self, store: PatternStore):
        """Test retrieving a missing key returns None."""
        result = store.retrieve("nonexistent")
        assert result is None

    def test_cache_avoids_disk_read(self, store: PatternStore, tmp_path: Path):
        """Test that cache avoids disk reads."""
        store.store("test_key", {"value": 42})

        # Delete the file
        file_path = tmp_path / "patterns" / "test_key.json"
        file_path.unlink()

        # Should still be in cache
        result = store.retrieve("test_key")
        assert result == {"value": 42}

    def test_clear_cache_forces_disk_read(self, store: PatternStore, tmp_path: Path):
        """Test that clear_cache forces disk read."""
        store.store("test_key", {"value": 42})
        store.clear_cache()

        # With cache cleared and file existing, should read from disk
        result = store.retrieve("test_key")
        assert result == {"value": 42}

    def test_list_keys_returns_all_keys(self, store: PatternStore):
        """Test listing all keys."""
        store.store("key1", {"a": 1})
        store.store("key2", {"b": 2})
        store.store("key3", {"c": 3})

        keys = store.list_keys()
        assert set(keys) == {"key1", "key2", "key3"}

    def test_delete_removes_key(self, store: PatternStore):
        """Test deleting a key."""
        store.store("test_key", {"value": 42})
        result = store.delete("test_key")

        assert result is True
        assert store.retrieve("test_key") is None

    def test_delete_returns_false_for_missing(self, store: PatternStore):
        """Test deleting a missing key."""
        result = store.delete("nonexistent")
        assert result is False

    def test_cleanup_removes_old_entries(self, store: PatternStore, tmp_path: Path):
        """Test cleanup removes old entries."""
        store.store("old_key", {"value": "old"})

        # Make the file appear old
        file_path = tmp_path / "patterns" / "old_key.json"
        old_time = time.time() - 100  # 100 seconds ago
        os.utime(file_path, (old_time, old_time))

        # Cleanup with 50 second max age
        removed = store.cleanup(max_age_seconds=50)

        assert removed == 1
        assert store.retrieve("old_key") is None

    def test_store_handles_special_characters_in_key(self, store: PatternStore):
        """Test storing with special characters in key."""
        store.store("path/with/slashes", {"value": 42})
        result = store.retrieve("path/with/slashes")
        assert result == {"value": 42}


class TestCompressionPatternLearner:
    """Tests for CompressionPatternLearner."""

    @pytest.fixture
    def learner(self, tmp_path: Path) -> CompressionPatternLearner:
        """Create a learner with temporary storage."""
        store = PatternStore(storage_dir=tmp_path / "patterns")
        return CompressionPatternLearner(store)

    def test_record_compression_result(self, learner: CompressionPatternLearner):
        """Test recording a compression result."""
        result = CompressionResult(
            page_type="product_list",
            original_tokens=5000,
            compressed_tokens=500,
            settings={"fold_lists": True},
        )
        learner.record_result(result)

        assert "product_list" in learner.get_all_page_types()

    def test_get_optimal_settings_after_enough_samples(
        self, learner: CompressionPatternLearner
    ):
        """Test getting optimal settings after recording enough samples."""
        # Record enough samples
        for i in range(10):
            result = CompressionResult(
                page_type="product_list",
                original_tokens=5000,
                compressed_tokens=500 + i * 10,  # Vary slightly
                settings={"fold_lists": True, "threshold": 0.85},
            )
            learner.record_result(result)

        settings = learner.get_optimal_settings("product_list")
        assert settings is not None
        assert "fold_lists" in settings

    def test_get_optimal_settings_returns_none_without_enough_samples(
        self, learner: CompressionPatternLearner
    ):
        """Test that optimal settings returns None without enough samples."""
        result = CompressionResult(
            page_type="rare_page",
            original_tokens=1000,
            compressed_tokens=100,
            settings={"fold_lists": False},
        )
        learner.record_result(result)

        settings = learner.get_optimal_settings("rare_page")
        assert settings is None

    def test_patterns_persisted_to_storage(self, tmp_path: Path):
        """Test that patterns are persisted to storage."""
        store = PatternStore(storage_dir=tmp_path / "patterns")
        learner1 = CompressionPatternLearner(store)

        # Record some data
        for i in range(5):
            result = CompressionResult(
                page_type="test_page",
                original_tokens=1000,
                compressed_tokens=100,
                settings={"test": True},
            )
            learner1.record_result(result)

        # Create new learner with same store
        learner2 = CompressionPatternLearner(store)

        # Should have loaded the data
        assert "test_page" in learner2.get_all_page_types()

    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        result = CompressionResult(
            page_type="test",
            original_tokens=1000,
            compressed_tokens=100,
            settings={},
        )
        assert result.compression_ratio == 0.9  # 90% compression

    def test_get_average_compression_ratio(
        self, learner: CompressionPatternLearner
    ):
        """Test getting average compression ratio."""
        for tokens in [100, 200, 300]:  # Different compression levels
            result = CompressionResult(
                page_type="test",
                original_tokens=1000,
                compressed_tokens=tokens,
                settings={},
            )
            learner.record_result(result)

        avg = learner.get_average_compression_ratio("test")
        assert avg is not None
        assert 0 <= avg <= 1


class TestTimeoutPatternLearner:
    """Tests for TimeoutPatternLearner."""

    @pytest.fixture
    def learner(self, tmp_path: Path) -> TimeoutPatternLearner:
        """Create a learner with temporary storage."""
        store = PatternStore(storage_dir=tmp_path / "patterns")
        return TimeoutPatternLearner(store)

    def test_record_timeout_result(self, learner: TimeoutPatternLearner):
        """Test recording a timeout result."""
        result = TimeoutResult(
            action="click",
            actual_duration_ms=150,
            success=True,
        )
        learner.record_result(result)

        assert learner.get_success_rate("click") == 1.0

    def test_get_optimal_timeout_uses_p95(
        self, learner: TimeoutPatternLearner
    ):
        """Test that optimal timeout uses P95 of durations."""
        # Record many results with varying durations
        durations = [100, 120, 130, 140, 150, 160, 170, 180, 190, 500]
        for duration in durations:
            result = TimeoutResult(
                action="click",
                actual_duration_ms=duration,
                success=True,
            )
            learner.record_result(result)

        timeout = learner.get_optimal_timeout("click")
        assert timeout is not None
        # P95 of these values is around 190-500, with 20% buffer
        assert timeout >= 190 * 1.2

    def test_get_optimal_timeout_ignores_failures(
        self, learner: TimeoutPatternLearner
    ):
        """Test that failed operations are ignored for timeout calculation."""
        # Record successful operations with reasonable durations
        for _ in range(10):
            learner.record_result(
                TimeoutResult(action="click", actual_duration_ms=100, success=True)
            )

        # Record a failed operation with very long duration
        learner.record_result(
            TimeoutResult(action="click", actual_duration_ms=60000, success=False)
        )

        timeout = learner.get_optimal_timeout("click")
        # Should not be influenced by the 60s failure
        assert timeout is not None
        assert timeout < 60000

    def test_get_optimal_timeouts_returns_all_actions(
        self, learner: TimeoutPatternLearner
    ):
        """Test getting optimal timeouts for all actions."""
        # Record data for multiple actions
        for action in ["click", "fill", "navigate"]:
            for _ in range(15):
                learner.record_result(
                    TimeoutResult(
                        action=action,
                        actual_duration_ms=100 + hash(action) % 100,
                        success=True,
                    )
                )

        timeouts = learner.get_optimal_timeouts()
        assert "click" in timeouts
        assert "fill" in timeouts
        assert "navigate" in timeouts

    def test_get_success_rate(self, learner: TimeoutPatternLearner):
        """Test success rate calculation."""
        # 7 successes, 3 failures = 70% success rate
        for i in range(10):
            learner.record_result(
                TimeoutResult(
                    action="click",
                    actual_duration_ms=100,
                    success=i < 7,
                )
            )

        rate = learner.get_success_rate("click")
        assert rate == 0.7


class TestPerformanceMetricsCollector:
    """Tests for PerformanceMetricsCollector."""

    @pytest.fixture
    def collector(self, tmp_path: Path) -> PerformanceMetricsCollector:
        """Create a collector with temporary storage."""
        store = PatternStore(storage_dir=tmp_path / "patterns")
        return PerformanceMetricsCollector(pattern_store=store)

    def test_record_snapshot_generation(
        self, collector: PerformanceMetricsCollector
    ):
        """Test recording snapshot generation metrics."""
        collector.record_snapshot_generation(
            duration_ms=150,
            tokens_before=5000,
            tokens_after=500,
        )

        summary = collector.get_summary("snapshot_generation")
        assert summary["count"] == 1
        assert summary["avg_duration_ms"] == 150

    def test_record_generic_metric(
        self, collector: PerformanceMetricsCollector
    ):
        """Test recording a generic metric."""
        metric = PerformanceMetric(
            operation="element_lookup",
            duration_ms=5,
        )
        collector.record(metric)

        summary = collector.get_summary("element_lookup")
        assert summary["count"] == 1

    def test_get_summary_calculates_averages(
        self, collector: PerformanceMetricsCollector
    ):
        """Test that summary calculates correct averages."""
        for duration in [100, 150, 200, 250, 300]:
            collector.record(
                PerformanceMetric(
                    operation="test_op",
                    duration_ms=duration,
                )
            )

        summary = collector.get_summary("test_op")
        assert summary["count"] == 5
        assert summary["avg_duration_ms"] == 200  # (100+150+200+250+300)/5
        assert summary["min_duration_ms"] == 100
        assert summary["max_duration_ms"] == 300

    def test_get_summary_with_compression_ratio(
        self, collector: PerformanceMetricsCollector
    ):
        """Test summary includes compression ratio for token metrics."""
        collector.record_snapshot_generation(
            duration_ms=100,
            tokens_before=1000,
            tokens_after=100,  # 90% compression
        )
        collector.record_snapshot_generation(
            duration_ms=100,
            tokens_before=1000,
            tokens_after=200,  # 80% compression
        )

        summary = collector.get_summary("snapshot_generation")
        assert "avg_compression_ratio" in summary
        assert summary["avg_compression_ratio"] == pytest.approx(0.85, rel=0.01)

    def test_get_operations_lists_all(
        self, collector: PerformanceMetricsCollector
    ):
        """Test getting list of all operations."""
        collector.record(PerformanceMetric(operation="op1", duration_ms=100))
        collector.record(PerformanceMetric(operation="op2", duration_ms=100))
        collector.record(PerformanceMetric(operation="op1", duration_ms=100))

        operations = collector.get_operations()
        assert set(operations) == {"op1", "op2"}

    def test_collector_without_store(self):
        """Test collector works without persistent storage."""
        collector = PerformanceMetricsCollector(pattern_store=None)

        collector.record(PerformanceMetric(operation="test", duration_ms=100))

        summary = collector.get_summary("test")
        assert summary["count"] == 1

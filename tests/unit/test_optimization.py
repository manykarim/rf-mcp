"""Unit tests for the native self-learning optimization system.

Tests all components of the optimization package using only Python
standard library features.
"""

import json
import tempfile
import shutil
from pathlib import Path
import pytest

from robotmcp.optimization import (
    PatternStore,
    ComplexityLevel,
    PageComplexityProfile,
    PageAnalyzer,
    PageTypeClassification,
    CompressionPattern,
    CompressionPatternLearner,
    TimeoutPattern,
    TimeoutPatternLearner,
    FoldingPattern,
    FoldingPatternLearner,
    RefUsagePattern,
    RefUsageLearner,
    PerformanceMetricsCollector,
    TokenMetrics,
    LatencyMetrics,
    TOKEN_TARGETS,
)


class TestPatternStore:
    """Tests for PatternStore class."""

    @pytest.fixture
    def temp_storage_dir(self):
        """Create a temporary storage directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def pattern_store(self, temp_storage_dir):
        """Create a PatternStore with temporary storage."""
        return PatternStore(storage_dir=temp_storage_dir)

    def test_store_and_retrieve(self, pattern_store):
        """Test basic store and retrieve operations."""
        test_data = {"value": 42, "name": "test"}

        # Store
        result = pattern_store.store("test_ns", "test_key", test_data)
        assert result is True

        # Retrieve
        retrieved = pattern_store.retrieve("test_ns", "test_key")
        assert retrieved is not None
        assert retrieved["value"] == 42
        assert retrieved["name"] == "test"
        assert "_stored_at" in retrieved  # Metadata added

    def test_list_keys(self, pattern_store):
        """Test listing keys in a namespace."""
        pattern_store.store("ns1", "key1", {"a": 1})
        pattern_store.store("ns1", "key2", {"b": 2})
        pattern_store.store("ns2", "key3", {"c": 3})

        keys = pattern_store.list_keys("ns1")
        assert len(keys) == 2
        assert "key1" in keys
        assert "key2" in keys

    def test_delete(self, pattern_store):
        """Test deleting a pattern."""
        pattern_store.store("ns", "key", {"data": "test"})
        assert pattern_store.retrieve("ns", "key") is not None

        result = pattern_store.delete("ns", "key")
        assert result is True
        assert pattern_store.retrieve("ns", "key") is None

    def test_cleanup_old_entries(self, pattern_store):
        """Test cleanup of old entries."""
        # Store an entry
        pattern_store.store("ns", "key", {"data": "test"})

        # Cleanup with 0 days should remove it
        removed = pattern_store.cleanup_old_entries("ns", max_age_days=0)
        assert removed == 1

    def test_sanitize_filename(self, pattern_store):
        """Test filename sanitization."""
        # Store with special characters
        result = pattern_store.store("ns/sub", "key:special", {"data": "test"})
        assert result is True

        # Should be retrievable
        retrieved = pattern_store.retrieve("ns/sub", "key:special")
        assert retrieved is not None

    def test_cache_behavior(self, pattern_store):
        """Test that cache is populated on retrieve."""
        pattern_store.store("ns", "key", {"data": "cached"})

        # Clear cache
        pattern_store.clear_cache()
        assert len(pattern_store._cache) == 0

        # Retrieve should populate cache
        pattern_store.retrieve("ns", "key")
        assert len(pattern_store._cache) == 1


class TestPageAnalyzer:
    """Tests for PageAnalyzer and PageComplexityProfile."""

    def test_complexity_level_low(self):
        """Test low complexity classification."""
        profile = PageComplexityProfile(
            element_count=100,
            form_count=1,
            list_item_count=10,
            depth=5,
            has_shadow_dom=False,
            has_iframes=False,
        )
        assert profile.complexity_level == ComplexityLevel.LOW

    def test_complexity_level_medium(self):
        """Test medium complexity classification."""
        profile = PageComplexityProfile(
            element_count=600,
            form_count=2,
            list_item_count=60,
            depth=8,
            has_shadow_dom=False,
            has_iframes=False,
        )
        assert profile.complexity_level == ComplexityLevel.MEDIUM

    def test_complexity_level_high(self):
        """Test high complexity classification."""
        profile = PageComplexityProfile(
            element_count=1500,
            form_count=5,
            list_item_count=150,
            depth=15,
            has_shadow_dom=True,
            has_iframes=True,
        )
        assert profile.complexity_level == ComplexityLevel.HIGH

    def test_optimization_recommendations(self):
        """Test that recommendations are generated correctly."""
        profile = PageComplexityProfile(
            element_count=800,
            form_count=2,
            list_item_count=75,
            depth=12,
            has_shadow_dom=False,
            has_iframes=False,
        )
        recommendations = profile.get_optimization_recommendations()

        assert recommendations["use_aria_snapshot"] is True
        assert recommendations["enable_list_folding"] is True
        assert "fold_threshold" in recommendations
        assert 0.0 < recommendations["fold_threshold"] < 1.0

    def test_fold_threshold_calculation(self):
        """Test fold threshold varies with list size."""
        small_list = PageComplexityProfile(
            element_count=100, form_count=0, list_item_count=5,
            depth=3, has_shadow_dom=False, has_iframes=False
        )
        large_list = PageComplexityProfile(
            element_count=500, form_count=0, list_item_count=150,
            depth=5, has_shadow_dom=False, has_iframes=False
        )

        # Larger lists should have lower threshold (more aggressive folding)
        assert small_list._calculate_optimal_fold_threshold() > large_list._calculate_optimal_fold_threshold()

    def test_page_type_classification(self):
        """Test page type classification."""
        analyzer = PageAnalyzer()

        # Search results page
        result = analyzer.classify_page_type(
            url="https://example.com/search?q=test",
            title="Search Results",
            text_content="showing 25 results for test"
        )
        assert result.page_type == "search_results"
        assert result.confidence > 0.0

    def test_page_type_generic_fallback(self):
        """Test fallback to generic type."""
        analyzer = PageAnalyzer()

        result = analyzer.classify_page_type(
            url="https://example.com/xyz",
            title="Welcome",
            text_content="nothing specific here"
        )
        assert result.page_type == "generic"


class TestCompressionPatternLearner:
    """Tests for CompressionPatternLearner."""

    @pytest.fixture
    def temp_storage_dir(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def learner(self, temp_storage_dir):
        store = PatternStore(storage_dir=temp_storage_dir)
        return CompressionPatternLearner(pattern_store=store)

    def test_record_compression_result(self, learner):
        """Test recording compression results."""
        learner.record_compression_result(
            page_type="search",
            url="https://example.com/search",
            compression_ratio=5.0,
            fold_threshold=0.85,
            success=True
        )

        assert "search" in learner.compression_ratios
        assert len(learner.compression_ratios["search"]) == 1

    def test_pattern_learning_threshold(self, learner):
        """Test that patterns are learned after enough samples."""
        # Record 10 results (minimum for pattern)
        for i in range(10):
            learner.record_compression_result(
                page_type="listing",
                url=f"https://example.com/page{i}",
                compression_ratio=4.5 + (i * 0.1),
                fold_threshold=0.85,
                success=True
            )

        pattern = learner.get_optimal_settings("listing")
        assert pattern is not None
        assert pattern.sample_count == 10

    def test_no_pattern_below_threshold(self, learner):
        """Test that no pattern is returned with few samples."""
        learner.record_compression_result(
            page_type="rare",
            url="https://example.com/rare",
            compression_ratio=3.0,
            fold_threshold=0.85,
            success=True
        )

        pattern = learner.get_optimal_settings("rare")
        assert pattern is None

    def test_failed_compression_not_counted(self, learner):
        """Test that failed compressions don't affect patterns."""
        learner.record_compression_result(
            page_type="test",
            url="https://example.com/test",
            compression_ratio=5.0,
            fold_threshold=0.85,
            success=False
        )

        assert len(learner.compression_ratios.get("test", [])) == 0


class TestTimeoutPatternLearner:
    """Tests for TimeoutPatternLearner."""

    @pytest.fixture
    def temp_storage_dir(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def learner(self, temp_storage_dir):
        store = PatternStore(storage_dir=temp_storage_dir)
        return TimeoutPatternLearner(pattern_store=store)

    def test_record_timeout_result(self, learner):
        """Test recording timeout results."""
        learner.record_timeout_result(
            page_type="form",
            action="fill",
            timeout_ms=5000,
            duration_ms=1200,
            success=True,
            triggered=False
        )

        key = "form:fill"
        assert key in learner.timeout_results
        assert len(learner.timeout_results[key]) == 1

    def test_default_timeouts_with_few_samples(self, learner):
        """Test default timeouts are returned with few samples."""
        timeouts = learner.get_optimal_timeouts("unknown", "click")
        assert "action" in timeouts
        assert timeouts["action"] == 5000  # Default

    def test_learned_timeouts(self, learner):
        """Test timeouts are learned from data."""
        # Record 15 results
        for i in range(15):
            learner.record_timeout_result(
                page_type="fast_page",
                action="click",
                timeout_ms=5000,
                duration_ms=500 + (i * 10),  # 500-640ms
                success=True,
                triggered=False
            )

        timeouts = learner.get_optimal_timeouts("fast_page", "click")
        # Should be around p95 (630ms) + 20% buffer
        assert timeouts["action"] < 5000  # Less than default
        assert timeouts["action"] >= 756  # At least p95 + buffer

    def test_navigation_timeout_category(self, learner):
        """Test navigation actions get navigation timeouts."""
        timeouts = learner.get_optimal_timeouts("page", "navigate")
        assert "navigation" in timeouts


class TestFoldingPatternLearner:
    """Tests for FoldingPatternLearner."""

    @pytest.fixture
    def temp_storage_dir(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def learner(self, temp_storage_dir):
        store = PatternStore(storage_dir=temp_storage_dir)
        return FoldingPatternLearner(pattern_store=store)

    def test_record_fold_result(self, learner):
        """Test recording fold results."""
        learner.record_fold_result(
            list_type="product_grid",
            threshold=0.85,
            items_original=50,
            items_folded=5,
            tokens_saved=450
        )

        assert "product_grid" in learner.fold_results
        assert len(learner.fold_results["product_grid"]) == 1

    def test_default_threshold(self, learner):
        """Test default threshold with no data."""
        threshold = learner.get_optimal_threshold("unknown")
        assert threshold == 0.85  # Default

    def test_should_fold_calculation(self, learner):
        """Test should_fold logic."""
        assert learner.should_fold("any", 10) is True
        assert learner.should_fold("any", 2) is False


class TestRefUsageLearner:
    """Tests for RefUsageLearner."""

    @pytest.fixture
    def temp_storage_dir(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def learner(self, temp_storage_dir):
        store = PatternStore(storage_dir=temp_storage_dir)
        return RefUsageLearner(pattern_store=store)

    def test_record_ref_access(self, learner):
        """Test recording ref access."""
        learner.record_ref_access("search", "e1", "session-1")
        learner.record_ref_access("search", "e1", "session-1")
        learner.record_ref_access("search", "e2", "session-1")

        assert learner.ref_frequencies["search"]["e1"] == 2
        assert learner.ref_frequencies["search"]["e2"] == 1

    def test_get_preload_candidates(self, learner):
        """Test preload candidates are sorted by frequency."""
        learner.record_ref_access("page", "e1", "s1")
        learner.record_ref_access("page", "e2", "s1")
        learner.record_ref_access("page", "e2", "s1")
        learner.record_ref_access("page", "e3", "s1")
        learner.record_ref_access("page", "e3", "s1")
        learner.record_ref_access("page", "e3", "s1")

        candidates = learner.get_preload_candidates("page", top_n=2)
        assert candidates[0] == "e3"  # Most frequent
        assert candidates[1] == "e2"  # Second most frequent

    def test_predict_next_refs(self, learner):
        """Test next ref prediction from sequences."""
        # Record sequence: e1 -> e2 -> e3
        learner.record_ref_access("page", "e1", "s1")
        learner.record_ref_access("page", "e2", "s1")
        learner.record_ref_access("page", "e3", "s1")

        # Record another sequence: e1 -> e2 -> e4
        learner.record_ref_access("page", "e1", "s2")
        learner.record_ref_access("page", "e2", "s2")
        learner.record_ref_access("page", "e4", "s2")

        # After e1, e2 should be predicted
        predictions = learner.predict_next_refs("page", "e1", top_n=1)
        assert "e2" in predictions


class TestPerformanceMetricsCollector:
    """Tests for PerformanceMetricsCollector."""

    @pytest.fixture
    def temp_storage_dir(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def collector(self, temp_storage_dir):
        store = PatternStore(storage_dir=temp_storage_dir)
        return PerformanceMetricsCollector(pattern_store=store)

    def test_record_snapshot_generation(self, collector):
        """Test recording snapshot generation."""
        collector.record_snapshot_generation(
            raw_chars=10000,
            aria_chars=1500,
            latency_ms=150,
            page_type="search",
            fold_threshold=0.85
        )

        assert len(collector.token_metrics) == 1
        assert collector.token_metrics[0].raw_html_chars == 10000
        assert collector.token_metrics[0].aria_snapshot_chars == 1500

    def test_record_ref_lookup(self, collector):
        """Test recording ref lookup."""
        collector.record_ref_lookup(
            ref="e1",
            success=True,
            stale=False,
            latency_ms=0.5,
            page_type="form",
            session_id="s1"
        )

        assert collector.ref_lookups_total == 1
        assert collector.ref_lookups_success == 1
        assert collector.ref_lookups_stale == 0

    def test_record_prevalidation(self, collector):
        """Test recording prevalidation results."""
        collector.record_prevalidation(
            predicted=True,
            actual=True,
            latency_ms=50
        )

        assert len(collector.prevalidation_results) == 1
        assert collector.prevalidation_results[0]["correct"] is True

    def test_get_summary(self, collector):
        """Test summary generation."""
        # Add some metrics
        collector.record_snapshot_generation(
            raw_chars=8000,
            aria_chars=1000,
            latency_ms=100,
            page_type="test",
            fold_threshold=0.85
        )
        collector.record_ref_lookup("e1", True, False, 0.5, "test", "s1")
        collector.record_prevalidation(True, True, 30)

        summary = collector.get_summary()

        assert "token_metrics" in summary
        assert "ref_metrics" in summary
        assert "prevalidation_metrics" in summary
        assert "latency_metrics" in summary

        # Check token reduction is calculated
        assert summary["token_metrics"]["avg_token_reduction_percent"] > 0

    def test_get_optimization_recommendations(self, collector):
        """Test optimization recommendations."""
        recommendations = collector.get_optimization_recommendations("search")

        assert "compression" in recommendations
        assert "folding" in recommendations
        assert "timeouts" in recommendations
        assert "ref_preload" in recommendations

    def test_detailed_report(self, collector):
        """Test detailed report includes all stats."""
        report = collector.get_detailed_report()

        assert "compression_learner_stats" in report
        assert "folding_learner_stats" in report
        assert "timeout_learner_stats" in report
        assert "ref_learner_stats" in report
        assert "pattern_store_stats" in report


class TestTokenMetrics:
    """Tests for TokenMetrics dataclass."""

    def test_token_metrics_creation(self):
        """Test creating TokenMetrics."""
        metrics = TokenMetrics(
            raw_html_tokens=2500,
            raw_html_chars=10000,
            aria_snapshot_tokens=375,
            aria_snapshot_chars=1500,
            token_reduction_percent=85.0,
            compression_ratio=6.67,
        )

        assert metrics.raw_html_tokens == 2500
        assert metrics.token_reduction_percent == 85.0


class TestLatencyMetrics:
    """Tests for LatencyMetrics dataclass."""

    def test_latency_metrics_defaults(self):
        """Test LatencyMetrics defaults."""
        metrics = LatencyMetrics()

        assert metrics.aria_snapshot_latency_ms == 0.0
        assert metrics.ref_lookup_latency_ms == 0.0


class TestIntegration:
    """Integration tests for the optimization system."""

    @pytest.fixture
    def temp_storage_dir(self):
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_full_workflow(self, temp_storage_dir):
        """Test a complete optimization workflow."""
        store = PatternStore(storage_dir=temp_storage_dir)
        collector = PerformanceMetricsCollector(pattern_store=store)

        # Simulate multiple page loads
        for i in range(15):
            collector.record_snapshot_generation(
                raw_chars=10000 + (i * 100),
                aria_chars=1500 + (i * 10),
                latency_ms=100 + (i * 5),
                page_type="product_listing",
                fold_threshold=0.85,
                list_items_original=50,
                list_items_folded=5
            )

            collector.record_ref_lookup(
                ref=f"e{i}",
                success=True,
                stale=False,
                latency_ms=0.3,
                page_type="product_listing",
                session_id="test-session"
            )

            collector.record_timeout_result(
                page_type="product_listing",
                action="click",
                timeout_ms=5000,
                actual_ms=800 + (i * 20),
                success=True,
                triggered=False
            )

        # Get recommendations
        recommendations = collector.get_optimization_recommendations("product_listing")

        # Compression should have learned
        assert recommendations["compression"]["confidence"] > 0

        # Summary should show good metrics
        summary = collector.get_summary()
        assert summary["token_metrics"]["avg_token_reduction_percent"] > 70
        assert summary["ref_metrics"]["hit_rate"] == 1.0

        # Persist and verify
        collector.persist_all_patterns()

        # Check storage
        keys = store.list_keys("compression")
        assert "product_listing" in keys

    def test_persistence_across_instances(self, temp_storage_dir):
        """Test that patterns persist across collector instances."""
        store = PatternStore(storage_dir=temp_storage_dir)

        # First instance: record data
        collector1 = PerformanceMetricsCollector(pattern_store=store)
        for i in range(15):
            collector1.record_snapshot_generation(
                raw_chars=10000,
                aria_chars=1200,
                latency_ms=100,
                page_type="persistent_type",
                fold_threshold=0.85
            )

        collector1.persist_all_patterns()

        # Second instance: should have learned patterns
        store2 = PatternStore(storage_dir=temp_storage_dir)
        collector2 = PerformanceMetricsCollector(pattern_store=store2)

        pattern = collector2.compression_learner.get_optimal_settings("persistent_type")
        assert pattern is not None
        assert pattern.sample_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

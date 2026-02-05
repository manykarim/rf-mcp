"""Performance benchmarks for instruction learning system.

This module benchmarks the self-learning capabilities:
- Tool call recording: <1ms per tool call
- Pattern storage/retrieval: <5ms
- Recommendation generation: <10ms

Run with:
    uv run pytest tests/benchmarks/test_learning_performance.py -v --benchmark-only
"""

import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from robotmcp.optimization.instruction_learner import (
    InstructionEffectivenessLearner,
    InstructionMode,
    SessionTracker,
    ToolCallEvent,
    InstructionEffectivenessRecord,
    LLMBehaviorPattern,
    SuccessfulSequence,
)
from robotmcp.optimization.pattern_store import PatternStore


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_storage():
    """Create a temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def pattern_store(temp_storage):
    """Create a pattern store with temporary storage."""
    return PatternStore(storage_dir=temp_storage)


@pytest.fixture
def learner(temp_storage):
    """Create a learner with temporary storage."""
    store = PatternStore(storage_dir=temp_storage)
    return InstructionEffectivenessLearner(pattern_store=store)


@pytest.fixture
def populated_learner(temp_storage):
    """Create a learner with pre-populated data."""
    store = PatternStore(storage_dir=temp_storage)
    learner = InstructionEffectivenessLearner(pattern_store=store)

    # Populate with sample sessions
    for i in range(50):
        tracker = learner.start_session(
            session_id=f"session-{i}",
            instruction_mode="default",
            llm_type="claude-sonnet",
            scenario_type="web_automation",
        )
        tracker.record_tool_call("find_keywords", {"query": "click"}, True)
        tracker.record_tool_call("execute_step", {"keyword": "Click"}, True)
        learner.end_session(tracker)

    # Force consolidation
    learner._consolidate_patterns()

    return learner


# ============================================================================
# Tool Call Recording Benchmarks
# ============================================================================


class TestToolCallRecordingLatency:
    """Benchmark tool call recording time."""

    @pytest.mark.benchmark
    def test_record_tool_call_latency(
        self,
        benchmark_reporter,
    ):
        """Target: <1ms per tool call recording.

        Tool call recording should be fast as it happens on every
        MCP tool invocation during a session.
        """
        tracker = SessionTracker(
            session_id="benchmark-session",
            instruction_mode="default",
            llm_type="claude-sonnet",
            scenario_type="web_automation",
        )

        iterations = 10000
        start = time.perf_counter()
        for i in range(iterations):
            tracker.record_tool_call(
                tool_name=f"tool_{i % 10}",
                arguments={"arg1": f"value_{i}", "arg2": i},
                success=True,
            )
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="tool_call_recording",
            duration_ms=total_ms,
            target_ms=1.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Tool call recording {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 1ms target"
        )

    @pytest.mark.benchmark
    def test_record_discovery_tool_call(
        self,
        benchmark_reporter,
    ):
        """Benchmark recording discovery tool calls (special tracking)."""
        tracker = SessionTracker(
            session_id="benchmark-session",
            instruction_mode="default",
            llm_type="test-model",
        )

        discovery_tools = [
            "find_keywords",
            "get_keyword_info",
            "recommend_libraries",
            "analyze_scenario",
            "check_library_availability",
        ]

        iterations = 5000
        start = time.perf_counter()
        for i in range(iterations):
            tool = discovery_tools[i % len(discovery_tools)]
            tracker.record_tool_call(
                tool_name=tool,
                arguments={"query": f"search_{i}"},
                success=True,
            )
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="discovery_tool_recording",
            duration_ms=total_ms,
            target_ms=1.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Discovery tool recording {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 1ms target"
        )

    @pytest.mark.benchmark
    def test_record_failed_tool_call(
        self,
        benchmark_reporter,
    ):
        """Benchmark recording failed tool calls with error classification."""
        tracker = SessionTracker(
            session_id="benchmark-session",
            instruction_mode="default",
            llm_type="test-model",
        )

        errors = [
            "keyword not found: InvalidKeyword",
            "stale reference: element no longer exists",
            "timeout waiting for element",
            "connection error",
            "element not found: locator failed",
        ]

        iterations = 5000
        start = time.perf_counter()
        for i in range(iterations):
            error = errors[i % len(errors)]
            tracker.record_tool_call(
                tool_name="execute_step",
                arguments={"keyword": f"Keyword_{i}"},
                success=False,
                error=error,
            )
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="failed_tool_recording",
            duration_ms=total_ms,
            target_ms=1.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Failed tool recording {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 1ms target"
        )

    @pytest.mark.benchmark
    def test_argument_hashing(
        self,
        benchmark_reporter,
    ):
        """Benchmark privacy-preserving argument hashing."""
        tracker = SessionTracker(
            session_id="benchmark-session",
            instruction_mode="default",
            llm_type="test-model",
        )

        # Various argument patterns
        argument_patterns = [
            {"keyword": "Click", "ref": "e1"},
            {"keyword": "Fill Text", "ref": "e2", "text": "Hello"},
            {"query": "search term", "library": "Browser"},
            {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
            {},  # Empty arguments
        ]

        iterations = 10000
        start = time.perf_counter()
        for i in range(iterations):
            args = argument_patterns[i % len(argument_patterns)]
            _hash = tracker._hash_arguments(args)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="argument_hashing",
            duration_ms=total_ms,
            target_ms=0.5,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Argument hashing {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 0.5ms target"
        )


# ============================================================================
# Pattern Storage/Retrieval Benchmarks
# ============================================================================


class TestPatternStorageLatency:
    """Benchmark pattern storage and retrieval."""

    @pytest.mark.benchmark
    def test_pattern_store_write(
        self,
        pattern_store: PatternStore,
        benchmark_reporter,
    ):
        """Target: <5ms for storing a pattern."""
        iterations = 500
        start = time.perf_counter()
        for i in range(iterations):
            pattern_store.store(
                namespace="test_patterns",
                key=f"pattern_{i}",
                value={
                    "llm_type": f"model_{i % 10}",
                    "mode": "default",
                    "compliance": 0.95,
                    "success_rate": 0.88,
                },
            )
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="pattern_store_write",
            duration_ms=total_ms,
            target_ms=5.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Pattern store write {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 5ms target"
        )

    @pytest.mark.benchmark
    def test_pattern_store_read_cached(
        self,
        pattern_store: PatternStore,
        benchmark_reporter,
    ):
        """Target: <5ms for retrieving a cached pattern."""
        # Pre-populate store
        for i in range(100):
            pattern_store.store(
                namespace="test_patterns",
                key=f"pattern_{i}",
                value={"data": f"value_{i}"},
            )

        # Benchmark cached reads
        iterations = 5000
        start = time.perf_counter()
        for i in range(iterations):
            _data = pattern_store.retrieve("test_patterns", f"pattern_{i % 100}")
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="pattern_store_read_cached",
            duration_ms=total_ms,
            target_ms=5.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Pattern store read (cached) {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 5ms target"
        )

    @pytest.mark.benchmark
    def test_pattern_store_list_keys(
        self,
        pattern_store: PatternStore,
        benchmark_reporter,
    ):
        """Benchmark listing keys in a namespace."""
        # Pre-populate store with many keys
        for i in range(500):
            pattern_store.store(
                namespace="list_test",
                key=f"key_{i}",
                value={"index": i},
            )

        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            _keys = pattern_store.list_keys("list_test")
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="pattern_store_list_keys",
            duration_ms=total_ms,
            target_ms=50.0,  # Allow more time for filesystem operations
            iterations=iterations,
        )

        assert result.target_met, (
            f"Pattern store list keys {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 50ms target"
        )

    @pytest.mark.benchmark
    def test_behavior_pattern_storage(
        self,
        pattern_store: PatternStore,
        benchmark_reporter,
    ):
        """Benchmark storing and retrieving LLMBehaviorPattern objects."""
        patterns = []
        for i in range(10):
            pattern = LLMBehaviorPattern(
                llm_type=f"model_{i}",
                sample_count=100,
                preferred_instruction_mode="default",
                discovery_compliance_by_mode={"default": 0.95, "minimal": 0.85},
                invalid_keyword_rate_by_mode={"default": 0.02, "minimal": 0.08},
                success_rate_by_mode={"default": 0.90, "minimal": 0.82},
                best_scenario_types=["web_automation", "api_testing"],
                problematic_scenario_types=["mobile_testing"],
                confidence=0.85,
            )
            patterns.append(pattern)

        # Benchmark storing
        iterations = 100
        start = time.perf_counter()
        for i in range(iterations):
            pattern = patterns[i % len(patterns)]
            pattern_store.store(
                namespace="instruction_patterns",
                key=pattern.llm_type,
                value=pattern.to_dict(),
            )
        total_ms = (time.perf_counter() - start) * 1000

        store_result = benchmark_reporter.record_latency(
            name="behavior_pattern_store",
            duration_ms=total_ms,
            target_ms=5.0,
            iterations=iterations,
        )

        # Benchmark retrieving
        start = time.perf_counter()
        for i in range(iterations):
            pattern = patterns[i % len(patterns)]
            data = pattern_store.retrieve("instruction_patterns", pattern.llm_type)
            _restored = LLMBehaviorPattern.from_dict(data)
        total_ms = (time.perf_counter() - start) * 1000

        retrieve_result = benchmark_reporter.record_latency(
            name="behavior_pattern_retrieve",
            duration_ms=total_ms,
            target_ms=5.0,
            iterations=iterations,
        )

        assert store_result.target_met, (
            f"Behavior pattern store {store_result.avg_per_operation_ms:.4f}ms "
            f"exceeds 5ms target"
        )
        assert retrieve_result.target_met, (
            f"Behavior pattern retrieve {retrieve_result.avg_per_operation_ms:.4f}ms "
            f"exceeds 5ms target"
        )


# ============================================================================
# Recommendation Generation Benchmarks
# ============================================================================


class TestRecommendationLatency:
    """Benchmark recommendation generation time."""

    @pytest.mark.benchmark
    def test_recommendation_with_learned_pattern(
        self,
        populated_learner: InstructionEffectivenessLearner,
        benchmark_reporter,
    ):
        """Target: <10ms for generating recommendation with learned data."""
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            _rec = populated_learner.get_recommendation(
                llm_type="claude-sonnet",
                scenario_type="web_automation",
            )
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="recommendation_with_pattern",
            duration_ms=total_ms,
            target_ms=10.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Recommendation generation {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 10ms target"
        )

    @pytest.mark.benchmark
    def test_recommendation_heuristic_fallback(
        self,
        learner: InstructionEffectivenessLearner,
        benchmark_reporter,
    ):
        """Target: <10ms for generating heuristic recommendation."""
        llm_types = [
            "claude-opus",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "llama-2-70b",
            "unknown-model",
        ]

        iterations = 2000
        start = time.perf_counter()
        for i in range(iterations):
            llm = llm_types[i % len(llm_types)]
            _rec = learner.get_recommendation(llm, "web_automation")
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="recommendation_heuristic",
            duration_ms=total_ms,
            target_ms=10.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Heuristic recommendation {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 10ms target"
        )

    @pytest.mark.benchmark
    def test_mode_comparison_generation(
        self,
        populated_learner: InstructionEffectivenessLearner,
        benchmark_reporter,
    ):
        """Benchmark generating mode comparison reports."""
        iterations = 500
        start = time.perf_counter()
        for _ in range(iterations):
            _comparison = populated_learner.get_mode_comparison("claude-sonnet")
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="mode_comparison",
            duration_ms=total_ms,
            target_ms=10.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Mode comparison generation {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 10ms target"
        )

    @pytest.mark.benchmark
    def test_statistics_generation(
        self,
        populated_learner: InstructionEffectivenessLearner,
        benchmark_reporter,
    ):
        """Benchmark generating learning statistics."""
        iterations = 500
        start = time.perf_counter()
        for _ in range(iterations):
            _stats = populated_learner.get_statistics()
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="statistics_generation",
            duration_ms=total_ms,
            target_ms=10.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Statistics generation {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 10ms target"
        )


# ============================================================================
# Session Lifecycle Benchmarks
# ============================================================================


class TestSessionLifecycleLatency:
    """Benchmark complete session lifecycle."""

    @pytest.mark.benchmark
    def test_session_start_latency(
        self,
        learner: InstructionEffectivenessLearner,
        benchmark_reporter,
    ):
        """Benchmark session initialization."""
        iterations = 5000
        start = time.perf_counter()
        for i in range(iterations):
            tracker = learner.start_session(
                session_id=f"session_{i}",
                instruction_mode="default",
                llm_type="claude-sonnet",
                scenario_type="web_automation",
            )
            # Immediately remove to avoid memory growth
            learner._active_sessions.pop(f"session_{i}", None)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="session_start",
            duration_ms=total_ms,
            target_ms=1.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Session start {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 1ms target"
        )

    @pytest.mark.benchmark
    def test_session_end_latency(
        self,
        learner: InstructionEffectivenessLearner,
        benchmark_reporter,
    ):
        """Benchmark session finalization and record storage."""
        # Create sessions first
        trackers = []
        for i in range(100):
            tracker = learner.start_session(
                session_id=f"session_{i}",
                instruction_mode="default",
                llm_type="test-model",
            )
            tracker.record_tool_call("find_keywords", {}, True)
            tracker.record_tool_call("execute_step", {"keyword": "Click"}, True)
            trackers.append(tracker)

        iterations = len(trackers)
        start = time.perf_counter()
        for tracker in trackers:
            learner.end_session(tracker)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="session_end",
            duration_ms=total_ms,
            target_ms=10.0,  # Allow more time for I/O
            iterations=iterations,
        )

        assert result.target_met, (
            f"Session end {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 10ms target"
        )

    @pytest.mark.benchmark
    def test_complete_session_lifecycle(
        self,
        learner: InstructionEffectivenessLearner,
        benchmark_reporter,
    ):
        """Benchmark complete session: start -> record calls -> end."""
        iterations = 100
        start = time.perf_counter()
        for i in range(iterations):
            # Start session
            tracker = learner.start_session(
                session_id=f"session_{i}",
                instruction_mode="default",
                llm_type="claude-sonnet",
            )

            # Record typical tool sequence
            tracker.record_tool_call("find_keywords", {"query": "click"}, True)
            tracker.record_tool_call("get_keyword_info", {"name": "Click"}, True)
            tracker.record_tool_call("execute_step", {"keyword": "Click"}, True)

            # End session
            learner.end_session(tracker)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="complete_session_lifecycle",
            duration_ms=total_ms,
            target_ms=20.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Complete session lifecycle {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 20ms target"
        )


# ============================================================================
# Pattern Consolidation Benchmarks
# ============================================================================


class TestPatternConsolidationLatency:
    """Benchmark pattern consolidation operations."""

    @pytest.mark.benchmark
    def test_consolidation_latency(
        self,
        learner: InstructionEffectivenessLearner,
        benchmark_reporter,
    ):
        """Benchmark pattern consolidation from records."""
        # Add enough records to trigger consolidation
        for i in range(100):
            tracker = learner.start_session(
                session_id=f"session_{i}",
                instruction_mode="default" if i % 2 == 0 else "minimal",
                llm_type="consolidation-test",
            )
            tracker.record_tool_call("find_keywords", {}, True)
            tracker.record_tool_call("execute_step", {}, i % 3 == 0)  # 33% fail
            learner.end_session(tracker)

        iterations = 10
        start = time.perf_counter()
        for _ in range(iterations):
            learner._consolidate_patterns()
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="pattern_consolidation",
            duration_ms=total_ms,
            target_ms=100.0,  # Consolidation can be slower
            iterations=iterations,
        )

        assert result.target_met, (
            f"Pattern consolidation {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 100ms target"
        )

    @pytest.mark.benchmark
    def test_persist_all_latency(
        self,
        learner: InstructionEffectivenessLearner,
        benchmark_reporter,
    ):
        """Benchmark persisting all learned data."""
        # Populate learner
        for i in range(50):
            tracker = learner.start_session(
                session_id=f"session_{i}",
                instruction_mode="default",
                llm_type=f"model_{i % 5}",
            )
            tracker.record_tool_call("find_keywords", {}, True)
            learner.end_session(tracker)

        iterations = 5
        start = time.perf_counter()
        for _ in range(iterations):
            learner.persist_all()
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="persist_all",
            duration_ms=total_ms,
            target_ms=500.0,  # Allow more time for full persistence
            iterations=iterations,
        )

        assert result.target_met, (
            f"Persist all {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 500ms target"
        )


# ============================================================================
# Successful Sequence Tracking Benchmarks
# ============================================================================


class TestSequenceTrackingLatency:
    """Benchmark successful sequence tracking."""

    @pytest.mark.benchmark
    def test_sequence_hash_generation(
        self,
        benchmark_reporter,
    ):
        """Benchmark generating sequence hashes."""
        tracker = SessionTracker(
            session_id="test-session",
            instruction_mode="default",
            llm_type="test-model",
        )

        # Record some tool calls to create sequences
        for i in range(10):
            tracker.record_tool_call("find_keywords", {}, True)
            tracker.record_tool_call("execute_step", {"keyword": "Click"}, True)

        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            _hashes = tracker.get_successful_sequence_hashes()
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="sequence_hash_generation",
            duration_ms=total_ms,
            target_ms=5.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Sequence hash generation {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 5ms target"
        )

    @pytest.mark.benchmark
    def test_effectiveness_record_generation(
        self,
        benchmark_reporter,
    ):
        """Benchmark generating effectiveness records from sessions."""
        trackers = []
        for i in range(100):
            tracker = SessionTracker(
                session_id=f"session_{i}",
                instruction_mode="default",
                llm_type="test-model",
            )
            # Add varied tool calls
            tracker.record_tool_call("find_keywords", {}, True)
            if i % 3 == 0:
                tracker.record_tool_call("execute_step", {}, False, "keyword not found")
            tracker.record_tool_call("execute_step", {"keyword": "Click"}, True)
            trackers.append(tracker)

        iterations = len(trackers)
        start = time.perf_counter()
        for tracker in trackers:
            _record = tracker.get_effectiveness_record()
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="effectiveness_record_generation",
            duration_ms=total_ms,
            target_ms=1.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Effectiveness record generation {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 1ms target"
        )


# ============================================================================
# Data Structure Serialization Benchmarks
# ============================================================================


class TestSerializationLatency:
    """Benchmark data structure serialization/deserialization."""

    @pytest.mark.benchmark
    def test_tool_call_event_serialization(
        self,
        benchmark_reporter,
    ):
        """Benchmark ToolCallEvent serialization."""
        event = ToolCallEvent(
            tool_name="execute_step",
            timestamp=1234567890.0,
            success=True,
            is_discovery=False,
            is_state_tool=False,
            error_type=None,
            arguments_hash="abc123",
        )

        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            data = event.to_dict()
            _restored = ToolCallEvent.from_dict(data)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="tool_call_event_serialization",
            duration_ms=total_ms,
            target_ms=0.5,
            iterations=iterations,
        )

        assert result.target_met, (
            f"ToolCallEvent serialization {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 0.5ms target"
        )

    @pytest.mark.benchmark
    def test_effectiveness_record_serialization(
        self,
        benchmark_reporter,
    ):
        """Benchmark InstructionEffectivenessRecord serialization."""
        record = InstructionEffectivenessRecord(
            instruction_mode="default",
            llm_type="claude-sonnet",
            session_id="test-session",
            timestamp=1234567890.0,
            discovery_first_compliance=True,
            invalid_keyword_count=2,
            total_tool_calls=15,
            successful_sequences=3,
            failed_sequences=1,
            error_recovery_success=0.75,
            scenario_type="web_automation",
        )

        iterations = 5000
        start = time.perf_counter()
        for _ in range(iterations):
            data = record.to_dict()
            _restored = InstructionEffectivenessRecord.from_dict(data)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="effectiveness_record_serialization",
            duration_ms=total_ms,
            target_ms=1.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"EffectivenessRecord serialization {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 1ms target"
        )

    @pytest.mark.benchmark
    def test_successful_sequence_serialization(
        self,
        benchmark_reporter,
    ):
        """Benchmark SuccessfulSequence serialization."""
        sequence = SuccessfulSequence(
            sequence_hash="abc123xyz",
            tool_sequence=[
                "find_keywords",
                "get_keyword_info",
                "execute_step",
                "verify_result",
            ],
            scenario_type="web_automation",
            success_count=15,
            llm_types=["claude-sonnet", "gpt-4", "claude-opus"],
            instruction_modes=["default", "minimal"],
            avg_duration_ms=150.5,
        )

        iterations = 5000
        start = time.perf_counter()
        for _ in range(iterations):
            data = sequence.to_dict()
            _restored = SuccessfulSequence.from_dict(data)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="successful_sequence_serialization",
            duration_ms=total_ms,
            target_ms=1.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"SuccessfulSequence serialization {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 1ms target"
        )

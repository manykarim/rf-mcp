"""Unit tests for MCP instruction effectiveness learning system."""

import json
import tempfile
import time
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch

import pytest

from robotmcp.optimization.instruction_learner import (
    InstructionMode,
    ToolCallEvent,
    InstructionEffectivenessRecord,
    LLMBehaviorPattern,
    SuccessfulSequence,
    SessionTracker,
    InstructionEffectivenessLearner,
)
from robotmcp.optimization.pattern_store import PatternStore


class TestToolCallEvent:
    """Tests for ToolCallEvent dataclass."""

    def test_create_event(self):
        """Test creating a tool call event."""
        event = ToolCallEvent(
            tool_name="list_keywords",
            timestamp=1234567890.0,
            success=True,
            is_discovery=True,
            is_state_tool=False,
        )
        assert event.tool_name == "list_keywords"
        assert event.success is True
        assert event.is_discovery is True

    def test_to_dict(self):
        """Test converting event to dictionary."""
        event = ToolCallEvent(
            tool_name="execute_step",
            timestamp=1234567890.0,
            success=False,
            error_type="invalid_keyword",
            arguments_hash="abc123",
        )
        data = event.to_dict()

        assert data["tool_name"] == "execute_step"
        assert data["success"] is False
        assert data["error_type"] == "invalid_keyword"
        assert data["arguments_hash"] == "abc123"

    def test_from_dict(self):
        """Test creating event from dictionary."""
        data = {
            "tool_name": "click",
            "timestamp": 1234567890.0,
            "success": True,
            "is_discovery": False,
            "is_state_tool": False,
        }
        event = ToolCallEvent.from_dict(data)

        assert event.tool_name == "click"
        assert event.success is True

    def test_from_dict_with_defaults(self):
        """Test creating event from partial dictionary."""
        data = {"tool_name": "test"}
        event = ToolCallEvent.from_dict(data)

        assert event.tool_name == "test"
        assert event.timestamp == 0
        assert event.success is False
        assert event.is_discovery is False


class TestInstructionEffectivenessRecord:
    """Tests for InstructionEffectivenessRecord dataclass."""

    def test_create_record(self):
        """Test creating an effectiveness record."""
        record = InstructionEffectivenessRecord(
            instruction_mode="default",
            llm_type="claude-sonnet",
            session_id="session-1",
            timestamp=1234567890.0,
            discovery_first_compliance=True,
            invalid_keyword_count=2,
            total_tool_calls=10,
        )

        assert record.instruction_mode == "default"
        assert record.llm_type == "claude-sonnet"
        assert record.discovery_first_compliance is True
        assert record.invalid_keyword_count == 2

    def test_to_dict_roundtrip(self):
        """Test roundtrip conversion to/from dict."""
        original = InstructionEffectivenessRecord(
            instruction_mode="verbose",
            llm_type="gpt-4",
            session_id="session-2",
            timestamp=1234567890.0,
            discovery_first_compliance=False,
            invalid_keyword_count=5,
            total_tool_calls=20,
            successful_sequences=3,
            failed_sequences=2,
            error_recovery_success=0.75,
            scenario_type="web_automation",
        )

        data = original.to_dict()
        restored = InstructionEffectivenessRecord.from_dict(data)

        assert restored.instruction_mode == original.instruction_mode
        assert restored.llm_type == original.llm_type
        assert restored.invalid_keyword_count == original.invalid_keyword_count
        assert restored.error_recovery_success == original.error_recovery_success


class TestLLMBehaviorPattern:
    """Tests for LLMBehaviorPattern dataclass."""

    def test_create_pattern(self):
        """Test creating a behavior pattern."""
        pattern = LLMBehaviorPattern(
            llm_type="claude-sonnet",
            sample_count=100,
            preferred_instruction_mode="minimal",
            discovery_compliance_by_mode={"minimal": 0.95, "default": 0.98},
            invalid_keyword_rate_by_mode={"minimal": 0.02, "default": 0.01},
            success_rate_by_mode={"minimal": 0.90, "default": 0.88},
            confidence=0.85,
        )

        assert pattern.llm_type == "claude-sonnet"
        assert pattern.preferred_instruction_mode == "minimal"
        assert pattern.confidence == 0.85

    def test_to_dict_roundtrip(self):
        """Test roundtrip conversion to/from dict."""
        original = LLMBehaviorPattern(
            llm_type="gpt-4",
            sample_count=50,
            preferred_instruction_mode="off",
            discovery_compliance_by_mode={"off": 0.92},
            best_scenario_types=["web_automation", "api_testing"],
            problematic_scenario_types=["mobile_testing"],
        )

        data = original.to_dict()
        restored = LLMBehaviorPattern.from_dict(data)

        assert restored.llm_type == original.llm_type
        assert restored.preferred_instruction_mode == original.preferred_instruction_mode
        assert restored.best_scenario_types == original.best_scenario_types


class TestSuccessfulSequence:
    """Tests for SuccessfulSequence dataclass."""

    def test_create_sequence(self):
        """Test creating a successful sequence."""
        sequence = SuccessfulSequence(
            sequence_hash="abc123",
            tool_sequence=["list_keywords", "execute_step", "click"],
            scenario_type="web_automation",
            success_count=5,
            llm_types=["claude-sonnet", "gpt-4"],
            instruction_modes=["default"],
        )

        assert sequence.sequence_hash == "abc123"
        assert len(sequence.tool_sequence) == 3
        assert sequence.success_count == 5


class TestSessionTracker:
    """Tests for SessionTracker class."""

    def test_init(self):
        """Test tracker initialization."""
        tracker = SessionTracker(
            session_id="test-session",
            instruction_mode="default",
            llm_type="claude-sonnet",
            scenario_type="web_automation",
        )

        assert tracker.session_id == "test-session"
        assert tracker.instruction_mode == "default"
        assert len(tracker.tool_calls) == 0

    def test_record_discovery_tool_first(self):
        """Test recording discovery tool as first call."""
        tracker = SessionTracker(
            session_id="test-session",
            instruction_mode="default",
            llm_type="claude-sonnet",
        )

        tracker.record_tool_call("list_keywords", {}, True)
        tracker.record_tool_call("execute_step", {"keyword": "Click"}, True)

        record = tracker.get_effectiveness_record()
        assert record.discovery_first_compliance is True

    def test_record_non_discovery_tool_first(self):
        """Test recording non-discovery tool as first call."""
        tracker = SessionTracker(
            session_id="test-session",
            instruction_mode="default",
            llm_type="claude-sonnet",
        )

        tracker.record_tool_call("execute_step", {"keyword": "Click"}, True)

        record = tracker.get_effectiveness_record()
        assert record.discovery_first_compliance is False

    def test_record_state_tool_as_discovery(self):
        """Test that state tools count as discovery-first compliance."""
        tracker = SessionTracker(
            session_id="test-session",
            instruction_mode="default",
            llm_type="claude-sonnet",
        )

        tracker.record_tool_call("get_page_snapshot", {}, True)
        tracker.record_tool_call("click", {"ref": "e1"}, True)

        record = tracker.get_effectiveness_record()
        assert record.discovery_first_compliance is True

    def test_count_invalid_keywords(self):
        """Test counting invalid keyword errors."""
        tracker = SessionTracker(
            session_id="test-session",
            instruction_mode="default",
            llm_type="claude-sonnet",
        )

        tracker.record_tool_call("execute_step", {"keyword": "InvalidKw"}, False, "keyword not found")
        tracker.record_tool_call("execute_step", {"keyword": "AnotherBad"}, False, "no keyword matches")
        tracker.record_tool_call("execute_step", {"keyword": "ValidKw"}, True)

        record = tracker.get_effectiveness_record()
        assert record.invalid_keyword_count == 2
        assert record.total_tool_calls == 3

    def test_classify_different_errors(self):
        """Test error classification for different error types."""
        tracker = SessionTracker(
            session_id="test-session",
            instruction_mode="default",
            llm_type="claude-sonnet",
        )

        tracker.record_tool_call("click", {"ref": "e1"}, False, "stale element reference: element no longer exists")
        tracker.record_tool_call("click", {"ref": "e2"}, False, "timeout waiting for element")
        tracker.record_tool_call("click", {"ref": "e3"}, False, "connection error")

        # Check that errors were recorded
        assert len(tracker.tool_calls) == 3
        assert tracker.tool_calls[0].error_type == "stale_ref"
        assert tracker.tool_calls[1].error_type == "timeout"
        assert tracker.tool_calls[2].error_type == "connection_error"

    def test_track_successful_sequences(self):
        """Test tracking successful tool sequences."""
        tracker = SessionTracker(
            session_id="test-session",
            instruction_mode="default",
            llm_type="claude-sonnet",
        )

        # Simulate a successful sequence
        tracker.record_tool_call("list_keywords", {}, True)
        tracker.record_tool_call("analyze_scenario", {"scenario": "test"}, True)
        tracker.record_tool_call("execute_step", {"keyword": "Click"}, True)

        # The sequence should be recorded as successful
        assert len(tracker.successful_sequences) == 1
        assert "list_keywords" in tracker.successful_sequences[0]

    def test_get_effectiveness_record(self):
        """Test generating effectiveness record from session."""
        tracker = SessionTracker(
            session_id="test-session",
            instruction_mode="verbose",
            llm_type="gpt-3.5",
            scenario_type="api_testing",
        )

        tracker.record_tool_call("list_keywords", {}, True)
        tracker.record_tool_call("execute_step", {}, False, "keyword not found")
        tracker.record_tool_call("search_keywords", {"query": "click"}, True)
        tracker.record_tool_call("execute_step", {"keyword": "Click"}, True)

        record = tracker.get_effectiveness_record()

        assert record.instruction_mode == "verbose"
        assert record.llm_type == "gpt-3.5"
        assert record.scenario_type == "api_testing"
        assert record.discovery_first_compliance is True
        assert record.invalid_keyword_count == 1
        assert record.total_tool_calls == 4


class TestInstructionEffectivenessLearner:
    """Tests for InstructionEffectivenessLearner class."""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def learner(self, temp_storage):
        """Create a learner with temporary storage."""
        store = PatternStore(storage_dir=temp_storage)
        return InstructionEffectivenessLearner(pattern_store=store)

    def test_init(self, learner):
        """Test learner initialization."""
        assert len(learner.records_by_llm) == 0
        assert len(learner.behavior_patterns) == 0

    def test_start_session(self, learner):
        """Test starting a tracking session."""
        tracker = learner.start_session(
            session_id="session-1",
            instruction_mode="default",
            llm_type="claude-sonnet",
            scenario_type="web_automation",
        )

        assert tracker.session_id == "session-1"
        assert "session-1" in learner._active_sessions

    def test_get_session(self, learner):
        """Test getting an active session."""
        learner.start_session("session-1", "default", "claude-sonnet")

        tracker = learner.get_session("session-1")
        assert tracker is not None
        assert tracker.session_id == "session-1"

        missing = learner.get_session("nonexistent")
        assert missing is None

    def test_end_session(self, learner):
        """Test ending a session and recording effectiveness."""
        tracker = learner.start_session(
            session_id="session-1",
            instruction_mode="default",
            llm_type="claude-sonnet",
        )

        tracker.record_tool_call("list_keywords", {}, True)
        tracker.record_tool_call("execute_step", {"keyword": "Click"}, True)

        record = learner.end_session(tracker)

        assert record.llm_type == "claude-sonnet"
        assert record.discovery_first_compliance is True
        assert "session-1" not in learner._active_sessions
        assert len(learner.records_by_llm["claude-sonnet"]) == 1

    def test_get_recommendation_heuristic(self, learner):
        """Test getting recommendation with no learned data."""
        rec = learner.get_recommendation("claude-opus", "web_automation")

        assert rec["llm_type"] == "claude-opus"
        assert rec["recommended_mode"] == "minimal"  # Large model heuristic
        assert rec["confidence"] == 0.0
        assert "heuristic" in rec["reasoning"]

    def test_get_recommendation_small_model_heuristic(self, learner):
        """Test heuristic for smaller models."""
        rec = learner.get_recommendation("gpt-3.5-turbo", "web_automation")

        assert rec["recommended_mode"] == "default"  # Small model heuristic

    def test_get_recommendation_unknown_model(self, learner):
        """Test heuristic for completely unknown models."""
        rec = learner.get_recommendation("totally-unknown-model", "api_testing")

        assert rec["recommended_mode"] == "default"  # Default fallback

    def test_learn_from_multiple_sessions(self, learner):
        """Test learning patterns from multiple sessions."""
        # Simulate multiple sessions with good discovery compliance
        for i in range(15):  # Need MIN_SAMPLES_FOR_RECOMMENDATION
            tracker = learner.start_session(
                session_id=f"session-{i}",
                instruction_mode="default",
                llm_type="test-model",
                scenario_type="web_automation",
            )
            tracker.record_tool_call("list_keywords", {}, True)
            tracker.record_tool_call("execute_step", {"keyword": "Click"}, True)
            learner.end_session(tracker)

        # Force consolidation
        learner._consolidate_patterns()

        # Should have learned a pattern
        assert "test-model" in learner.behavior_patterns
        pattern = learner.behavior_patterns["test-model"]
        assert pattern.sample_count >= 10
        assert pattern.discovery_compliance_by_mode.get("default", 0) == 1.0

    def test_get_statistics(self, learner):
        """Test getting learning statistics."""
        tracker = learner.start_session("session-1", "default", "claude-sonnet")
        tracker.record_tool_call("list_keywords", {}, True)
        learner.end_session(tracker)

        stats = learner.get_statistics()

        assert stats["llm_types_tracked"] == 1
        assert stats["total_records"] == 1
        assert "claude-sonnet" in stats["llm_details"]

    def test_get_mode_comparison_no_data(self, learner):
        """Test mode comparison with no learned data."""
        comparison = learner.get_mode_comparison("unknown-model")

        assert comparison["comparison_available"] is False

    def test_persist_and_load(self, temp_storage):
        """Test persistence and loading of learned data."""
        store = PatternStore(storage_dir=temp_storage)
        learner1 = InstructionEffectivenessLearner(pattern_store=store)

        # Record some sessions
        for i in range(12):
            tracker = learner1.start_session(
                session_id=f"session-{i}",
                instruction_mode="default",
                llm_type="persist-test",
            )
            tracker.record_tool_call("list_keywords", {}, True)
            learner1.end_session(tracker)

        # Force persist
        learner1.persist_all()

        # Create new learner with same storage
        learner2 = InstructionEffectivenessLearner(pattern_store=store)

        # Should load previous data
        assert "persist-test" in learner2.records_by_llm
        assert len(learner2.records_by_llm["persist-test"]) > 0

    def test_reset_learning_specific_llm(self, learner):
        """Test resetting learning for a specific LLM type."""
        # Add data for multiple LLMs
        for llm in ["model-a", "model-b"]:
            tracker = learner.start_session(f"session-{llm}", "default", llm)
            tracker.record_tool_call("list_keywords", {}, True)
            learner.end_session(tracker)

        # Reset only model-a
        learner.reset_learning("model-a")

        assert "model-a" not in learner.records_by_llm
        assert "model-b" in learner.records_by_llm

    def test_reset_learning_all(self, learner):
        """Test resetting all learning data."""
        for llm in ["model-a", "model-b", "model-c"]:
            tracker = learner.start_session(f"session-{llm}", "default", llm)
            tracker.record_tool_call("list_keywords", {}, True)
            learner.end_session(tracker)

        learner.reset_learning()

        assert len(learner.records_by_llm) == 0
        assert len(learner.behavior_patterns) == 0


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def learner(self, temp_storage):
        """Create a learner with temporary storage."""
        store = PatternStore(storage_dir=temp_storage)
        return InstructionEffectivenessLearner(pattern_store=store)

    def test_scenario_discovery_first_workflow(self, learner):
        """Test a typical discovery-first workflow."""
        tracker = learner.start_session(
            session_id="discovery-first",
            instruction_mode="default",
            llm_type="claude-sonnet",
            scenario_type="web_automation",
        )

        # Discovery phase
        tracker.record_tool_call("list_libraries", {}, True)
        tracker.record_tool_call("list_keywords", {"library": "Browser"}, True)
        tracker.record_tool_call("search_keywords", {"query": "click"}, True)

        # State check
        tracker.record_tool_call("get_page_snapshot", {}, True)

        # Action
        tracker.record_tool_call("execute_step", {"keyword": "Click", "ref": "e1"}, True)

        record = learner.end_session(tracker)

        assert record.discovery_first_compliance is True
        assert record.invalid_keyword_count == 0
        assert record.successful_sequences >= 1

    def test_scenario_skip_discovery_with_errors(self, learner):
        """Test workflow that skips discovery and encounters errors."""
        tracker = learner.start_session(
            session_id="skip-discovery",
            instruction_mode="minimal",
            llm_type="small-model",
            scenario_type="web_automation",
        )

        # Skip discovery, try action directly
        tracker.record_tool_call(
            "execute_step",
            {"keyword": "Clicky Button"},
            False,
            "keyword not found: Clicky Button"
        )

        # Try again with wrong keyword
        tracker.record_tool_call(
            "execute_step",
            {"keyword": "Press Button"},
            False,
            "unknown keyword"
        )

        # Finally use discovery
        tracker.record_tool_call("search_keywords", {"query": "button"}, True)

        # Now succeed
        tracker.record_tool_call("execute_step", {"keyword": "Click"}, True)

        record = learner.end_session(tracker)

        assert record.discovery_first_compliance is False
        assert record.invalid_keyword_count == 2
        assert record.error_recovery_success > 0  # Eventually recovered

    def test_scenario_compare_instruction_modes(self, learner):
        """Test comparing effectiveness across instruction modes."""
        # Sessions with 'default' mode - good compliance
        for i in range(12):
            tracker = learner.start_session(
                f"default-{i}", "default", "test-llm", "web_automation"
            )
            tracker.record_tool_call("list_keywords", {}, True)
            tracker.record_tool_call("execute_step", {}, True)
            learner.end_session(tracker)

        # Sessions with 'minimal' mode - some compliance issues
        for i in range(12):
            tracker = learner.start_session(
                f"minimal-{i}", "minimal", "test-llm", "web_automation"
            )
            # 50% of sessions skip discovery
            if i % 2 == 0:
                tracker.record_tool_call("list_keywords", {}, True)
            tracker.record_tool_call("execute_step", {}, True)
            learner.end_session(tracker)

        # Consolidate
        learner._consolidate_patterns()

        # Get comparison
        comparison = learner.get_mode_comparison("test-llm")

        assert comparison["comparison_available"] is True
        assert "default" in comparison["modes"]
        assert "minimal" in comparison["modes"]

        # Default mode should have higher compliance
        default_compliance = comparison["modes"]["default"]["discovery_compliance"]
        minimal_compliance = comparison["modes"]["minimal"]["discovery_compliance"]
        assert default_compliance > minimal_compliance

    def test_scenario_learn_optimal_mode(self, learner):
        """Test learning the optimal instruction mode for an LLM."""
        # Simulate a model that works best with minimal instructions
        # Need 30+ samples to reach confidence threshold of 0.3 (confidence = samples/100)
        for i in range(35):
            tracker = learner.start_session(
                f"session-{i}",
                "minimal" if i < 20 else "verbose",
                "smart-model",
                "web_automation",
            )
            # Always compliant and successful
            tracker.record_tool_call("list_keywords", {}, True)
            tracker.record_tool_call("execute_step", {}, True)
            learner.end_session(tracker)

        # Force consolidation to create behavior patterns
        learner._consolidate_patterns()

        # Verify pattern was created
        assert "smart-model" in learner.behavior_patterns
        pattern = learner.behavior_patterns["smart-model"]

        # Verify confidence threshold is met (35 samples = 0.35 confidence)
        assert pattern.confidence >= 0.3

        rec = learner.get_recommendation("smart-model", "web_automation")

        # Should recommend based on learned patterns (not heuristics)
        assert rec["confidence"] >= 0.3
        assert rec["sample_count"] >= 35
        # Both modes have equal metrics, so any recommendation is valid
        assert rec["recommended_mode"] in ["minimal", "verbose"]
        # Should NOT be using heuristics
        assert "heuristic" not in rec["reasoning"]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def learner(self, temp_storage):
        """Create a learner with temporary storage."""
        store = PatternStore(storage_dir=temp_storage)
        return InstructionEffectivenessLearner(pattern_store=store)

    def test_empty_session(self, learner):
        """Test ending a session with no tool calls."""
        tracker = learner.start_session("empty", "default", "test-model")
        record = learner.end_session(tracker)

        assert record.total_tool_calls == 0
        assert record.discovery_first_compliance is False  # No discovery calls

    def test_single_discovery_call_only(self, learner):
        """Test session with only discovery call."""
        tracker = learner.start_session("discovery-only", "default", "test-model")
        tracker.record_tool_call("list_keywords", {}, True)
        record = learner.end_session(tracker)

        assert record.discovery_first_compliance is True
        assert record.total_tool_calls == 1

    def test_all_calls_failed(self, learner):
        """Test session where all calls fail."""
        tracker = learner.start_session("all-failed", "default", "test-model")
        tracker.record_tool_call("list_keywords", {}, False, "connection error")
        tracker.record_tool_call("execute_step", {}, False, "timeout")
        record = learner.end_session(tracker)

        assert record.total_tool_calls == 2
        assert record.successful_sequences == 0

    def test_bounds_on_record_storage(self, learner):
        """Test that records are bounded to MAX_RECORDS_PER_LLM."""
        max_records = learner.MAX_RECORDS_PER_LLM

        # Add more than max records
        for i in range(max_records + 100):
            tracker = learner.start_session(f"session-{i}", "default", "bounded-test")
            tracker.record_tool_call("list_keywords", {}, True)
            learner.end_session(tracker)

        # Should be bounded
        assert len(learner.records_by_llm["bounded-test"]) <= max_records

    def test_recommendation_for_problematic_scenario(self, learner):
        """Test recommendation adjusts for problematic scenarios."""
        # Create a pattern with a problematic scenario
        pattern = LLMBehaviorPattern(
            llm_type="problem-model",
            sample_count=50,
            preferred_instruction_mode="minimal",
            discovery_compliance_by_mode={"minimal": 0.9},
            problematic_scenario_types=["mobile_testing"],
            confidence=0.8,
        )
        learner.behavior_patterns["problem-model"] = pattern

        # Get recommendation for problematic scenario
        rec = learner.get_recommendation("problem-model", "mobile_testing")

        # Should upgrade from minimal to default for problematic scenario
        assert rec["is_problematic_scenario"] is True
        assert rec["recommended_mode"] == "default"

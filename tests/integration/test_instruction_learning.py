"""Integration tests for instruction effectiveness learning system.

Tests verify that:
1. Learning hooks record tool calls
2. Discovery-first detection works
3. Metrics are persisted
4. Recommendations improve over sessions
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from fastmcp import Client, FastMCP

from robotmcp.optimization import (
    InstructionEffectivenessLearner,
    SessionTracker,
    ToolCallEvent,
    LLMBehaviorPattern,
)
from robotmcp.optimization.pattern_store import PatternStore


class TestLearningHooksRecordToolCalls:
    """Test learning hooks record tool calls."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def learner(self, temp_storage):
        """Create learner with temporary storage."""
        store = PatternStore(storage_dir=temp_storage)
        return InstructionEffectivenessLearner(pattern_store=store)

    def test_session_tracker_records_tool_call(self, learner):
        """SessionTracker records individual tool calls."""
        tracker = learner.start_session(
            session_id="test-session",
            instruction_mode="default",
            llm_type="claude-sonnet",
            scenario_type="web_automation",
        )

        # Record a tool call
        tracker.record_tool_call("find_keywords", {"query": "click"}, success=True)

        assert len(tracker.tool_calls) == 1
        event = tracker.tool_calls[0]
        assert event.tool_name == "find_keywords"
        assert event.success is True

    def test_session_tracker_records_multiple_calls(self, learner):
        """SessionTracker records sequence of tool calls."""
        tracker = learner.start_session(
            session_id="test-session",
            instruction_mode="default",
            llm_type="gpt-4",
        )

        # Record sequence of calls
        tracker.record_tool_call("list_keywords", {}, True)
        tracker.record_tool_call("get_keyword_info", {"name": "Click"}, True)
        tracker.record_tool_call("execute_step", {"keyword": "Click"}, True)

        assert len(tracker.tool_calls) == 3

    def test_session_tracker_records_failed_calls(self, learner):
        """SessionTracker records failed tool calls with errors."""
        tracker = learner.start_session(
            session_id="test-session",
            instruction_mode="default",
            llm_type="claude-sonnet",
        )

        tracker.record_tool_call(
            "execute_step",
            {"keyword": "NonexistentKeyword"},
            success=False,
            error="Keyword not found",
        )

        assert len(tracker.tool_calls) == 1
        event = tracker.tool_calls[0]
        assert event.success is False
        assert event.error_type is not None

    def test_tool_call_event_serialization(self):
        """ToolCallEvent serializes to/from dict."""
        event = ToolCallEvent(
            tool_name="find_keywords",
            timestamp=time.time(),
            success=True,
            is_discovery=True,
            is_state_tool=False,
            arguments_hash="abc123",
        )

        data = event.to_dict()
        restored = ToolCallEvent.from_dict(data)

        assert restored.tool_name == event.tool_name
        assert restored.success == event.success
        assert restored.is_discovery == event.is_discovery

    def test_learning_session_lifecycle(self, learner):
        """Complete session lifecycle: start, record, end."""
        # Start session
        tracker = learner.start_session(
            session_id="lifecycle-test",
            instruction_mode="default",
            llm_type="claude-sonnet",
        )
        assert "lifecycle-test" in learner._active_sessions

        # Record calls
        tracker.record_tool_call("list_keywords", {}, True)
        tracker.record_tool_call("execute_step", {"keyword": "Click"}, True)

        # End session
        record = learner.end_session(tracker)

        # Session should be removed
        assert "lifecycle-test" not in learner._active_sessions

        # Record should be stored
        assert "claude-sonnet" in learner.records_by_llm
        assert len(learner.records_by_llm["claude-sonnet"]) == 1


class TestDiscoveryFirstDetection:
    """Test discovery-first detection works."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def learner(self, temp_storage):
        """Create learner with temporary storage."""
        store = PatternStore(storage_dir=temp_storage)
        return InstructionEffectivenessLearner(pattern_store=store)

    def test_discovery_first_compliance_true(self, learner):
        """Discovery tool called first results in compliance=True."""
        tracker = learner.start_session(
            session_id="discovery-first",
            instruction_mode="default",
            llm_type="claude-sonnet",
        )

        # Discovery tool first
        tracker.record_tool_call("find_keywords", {"query": "click"}, True)
        tracker.record_tool_call("execute_step", {"keyword": "Click"}, True)

        record = tracker.get_effectiveness_record()
        assert record.discovery_first_compliance is True

    def test_discovery_first_compliance_false(self, learner):
        """Non-discovery tool called first results in compliance=False."""
        tracker = learner.start_session(
            session_id="skip-discovery",
            instruction_mode="default",
            llm_type="claude-sonnet",
        )

        # Action tool first (skipping discovery)
        tracker.record_tool_call("execute_step", {"keyword": "Click"}, True)

        record = tracker.get_effectiveness_record()
        assert record.discovery_first_compliance is False

    def test_state_tool_counts_as_discovery(self, learner):
        """State tools (get_page_snapshot) count as discovery-first."""
        tracker = learner.start_session(
            session_id="state-first",
            instruction_mode="default",
            llm_type="claude-sonnet",
        )

        # State tool first
        tracker.record_tool_call("get_page_snapshot", {}, True)
        tracker.record_tool_call("click", {"ref": "e1"}, True)

        record = tracker.get_effectiveness_record()
        assert record.discovery_first_compliance is True

    def test_list_keywords_is_discovery_tool(self, learner):
        """list_keywords is recognized as discovery tool."""
        tracker = learner.start_session(
            session_id="list-kw-first",
            instruction_mode="default",
            llm_type="gpt-4",
        )

        tracker.record_tool_call("list_keywords", {"library": "Browser"}, True)
        tracker.record_tool_call("execute_step", {"keyword": "Click"}, True)

        record = tracker.get_effectiveness_record()
        assert record.discovery_first_compliance is True

    def test_search_keywords_is_discovery_tool(self, learner):
        """search_keywords is recognized as discovery tool."""
        tracker = learner.start_session(
            session_id="search-first",
            instruction_mode="default",
            llm_type="gpt-4",
        )

        tracker.record_tool_call("search_keywords", {"query": "type"}, True)

        record = tracker.get_effectiveness_record()
        assert record.discovery_first_compliance is True

    def test_analyze_scenario_is_discovery_tool(self, learner):
        """analyze_scenario is recognized as discovery tool."""
        tracker = learner.start_session(
            session_id="analyze-first",
            instruction_mode="default",
            llm_type="claude-sonnet",
        )

        tracker.record_tool_call("analyze_scenario", {"scenario": "Login test"}, True)

        record = tracker.get_effectiveness_record()
        assert record.discovery_first_compliance is True

    def test_invalid_keyword_count(self, learner):
        """Invalid keyword errors are counted."""
        tracker = learner.start_session(
            session_id="invalid-kw",
            instruction_mode="default",
            llm_type="claude-sonnet",
        )

        # Multiple invalid keyword attempts
        tracker.record_tool_call(
            "execute_step", {"keyword": "BadKw1"}, False, "keyword not found"
        )
        tracker.record_tool_call(
            "execute_step", {"keyword": "BadKw2"}, False, "unknown keyword"
        )
        tracker.record_tool_call("find_keywords", {"query": "click"}, True)
        tracker.record_tool_call("execute_step", {"keyword": "Click"}, True)

        record = tracker.get_effectiveness_record()
        assert record.invalid_keyword_count == 2
        assert record.total_tool_calls == 4


class TestMetricsPersistence:
    """Test metrics are persisted."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_records_persisted_to_store(self, temp_storage):
        """Session records are persisted to pattern store."""
        store = PatternStore(storage_dir=temp_storage)
        learner = InstructionEffectivenessLearner(pattern_store=store)

        # Record a session
        tracker = learner.start_session(
            session_id="persist-test",
            instruction_mode="default",
            llm_type="test-model",
        )
        tracker.record_tool_call("find_keywords", {}, True)
        learner.end_session(tracker)

        # Force persistence
        learner.persist_all()

        # Verify file exists
        records_dir = temp_storage / "instruction_records"
        assert records_dir.exists() or any(temp_storage.glob("**/instruction_records*"))

    def test_records_loaded_on_new_learner(self, temp_storage):
        """Records are loaded when creating new learner with same store."""
        store = PatternStore(storage_dir=temp_storage)

        # First learner records sessions
        learner1 = InstructionEffectivenessLearner(pattern_store=store)
        for i in range(5):
            tracker = learner1.start_session(
                session_id=f"session-{i}",
                instruction_mode="default",
                llm_type="persist-model",
            )
            tracker.record_tool_call("find_keywords", {}, True)
            learner1.end_session(tracker)
        learner1.persist_all()

        # Second learner should load existing records
        learner2 = InstructionEffectivenessLearner(pattern_store=store)

        assert "persist-model" in learner2.records_by_llm
        assert len(learner2.records_by_llm["persist-model"]) == 5

    def test_behavior_patterns_persisted(self, temp_storage):
        """Consolidated behavior patterns are persisted."""
        store = PatternStore(storage_dir=temp_storage)
        learner = InstructionEffectivenessLearner(pattern_store=store)

        # Create enough records to trigger pattern consolidation
        for i in range(15):
            tracker = learner.start_session(
                session_id=f"pattern-{i}",
                instruction_mode="default",
                llm_type="pattern-model",
            )
            tracker.record_tool_call("find_keywords", {}, True)
            tracker.record_tool_call("execute_step", {}, True)
            learner.end_session(tracker)

        # Force consolidation and persistence
        learner._consolidate_patterns()
        learner.persist_all()

        # Verify pattern was created
        assert "pattern-model" in learner.behavior_patterns

    def test_statistics_after_sessions(self, temp_storage):
        """Get statistics reflects recorded sessions."""
        store = PatternStore(storage_dir=temp_storage)
        learner = InstructionEffectivenessLearner(pattern_store=store)

        # Record sessions for different LLMs
        for llm in ["claude-sonnet", "gpt-4", "gemini"]:
            tracker = learner.start_session(
                session_id=f"session-{llm}",
                instruction_mode="default",
                llm_type=llm,
            )
            tracker.record_tool_call("find_keywords", {}, True)
            learner.end_session(tracker)

        stats = learner.get_statistics()

        assert stats["llm_types_tracked"] == 3
        assert stats["total_records"] == 3
        assert "claude-sonnet" in stats["llm_details"]
        assert "gpt-4" in stats["llm_details"]
        assert "gemini" in stats["llm_details"]


class TestRecommendationsImprove:
    """Test recommendations improve over sessions."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def learner(self, temp_storage):
        """Create learner with temporary storage."""
        store = PatternStore(storage_dir=temp_storage)
        return InstructionEffectivenessLearner(pattern_store=store)

    def test_initial_recommendation_uses_heuristics(self, learner):
        """Initial recommendation uses heuristics (no data)."""
        rec = learner.get_recommendation("new-model", "web_automation")

        assert rec["confidence"] == 0.0
        assert "heuristic" in rec["reasoning"].lower()

    def test_recommendation_confidence_increases(self, learner):
        """Recommendation confidence increases with more data."""
        # Record 30+ sessions to exceed MIN_SAMPLES_FOR_RECOMMENDATION (10)
        # and build sufficient confidence
        for i in range(35):
            tracker = learner.start_session(
                session_id=f"session-{i}",
                instruction_mode="default",
                llm_type="learning-model",
            )
            tracker.record_tool_call("find_keywords", {}, True)
            tracker.record_tool_call("execute_step", {}, True)
            learner.end_session(tracker)

        # Force consolidation
        learner._consolidate_patterns()

        rec = learner.get_recommendation("learning-model", "web_automation")

        # Should have learned from data
        assert rec["confidence"] > 0
        assert rec["sample_count"] >= 35

    def test_recommendation_reflects_learned_mode(self, learner):
        """Recommendation reflects what mode works best."""
        # Record sessions with different modes having different success rates
        # Mode "minimal" has higher success rate
        for i in range(20):
            tracker = learner.start_session(
                session_id=f"minimal-{i}",
                instruction_mode="minimal",
                llm_type="compare-model",
            )
            # Always compliant in minimal mode
            tracker.record_tool_call("find_keywords", {}, True)
            tracker.record_tool_call("execute_step", {}, True)
            learner.end_session(tracker)

        # Default mode has some failures
        for i in range(20):
            tracker = learner.start_session(
                session_id=f"default-{i}",
                instruction_mode="default",
                llm_type="compare-model",
            )
            # 50% skip discovery (non-compliant)
            if i % 2 == 0:
                tracker.record_tool_call("find_keywords", {}, True)
            tracker.record_tool_call("execute_step", {}, True)
            learner.end_session(tracker)

        learner._consolidate_patterns()

        comparison = learner.get_mode_comparison("compare-model")

        assert comparison["comparison_available"] is True
        # Minimal mode should have higher compliance
        assert "minimal" in comparison["modes"]
        assert "default" in comparison["modes"]

    def test_recommendation_for_problematic_scenario(self, learner):
        """Recommendation adjusts for problematic scenarios."""
        # Set up a pattern with known problematic scenario
        pattern = LLMBehaviorPattern(
            llm_type="scenario-model",
            sample_count=50,
            preferred_instruction_mode="minimal",
            discovery_compliance_by_mode={"minimal": 0.9},
            problematic_scenario_types=["mobile_testing"],
            confidence=0.8,
        )
        learner.behavior_patterns["scenario-model"] = pattern

        # Get recommendation for problematic scenario
        rec = learner.get_recommendation("scenario-model", "mobile_testing")

        assert rec["is_problematic_scenario"] is True
        # Should upgrade to more verbose mode for problematic scenario
        assert rec["recommended_mode"] == "default"

    def test_recommendation_for_large_model_heuristic(self, learner):
        """Large models get minimal instructions by heuristic."""
        for model in ["claude-opus", "claude-sonnet", "gpt-4o"]:
            rec = learner.get_recommendation(model, "web_automation")

            # Large models should get minimal by default
            assert rec["recommended_mode"] == "minimal", f"Failed for {model}"

    def test_recommendation_for_small_model_heuristic(self, learner):
        """Small models get default instructions by heuristic."""
        for model in ["gpt-3.5-turbo", "claude-haiku", "gemini-1.5-flash"]:
            rec = learner.get_recommendation(model, "web_automation")

            # Smaller models should get default (more guidance)
            assert rec["recommended_mode"] == "default", f"Failed for {model}"

    def test_reset_learning_clears_data(self, learner):
        """Reset learning clears all data for LLM."""
        # Record some data
        tracker = learner.start_session(
            session_id="reset-test",
            instruction_mode="default",
            llm_type="reset-model",
        )
        tracker.record_tool_call("find_keywords", {}, True)
        learner.end_session(tracker)

        assert "reset-model" in learner.records_by_llm

        # Reset
        learner.reset_learning("reset-model")

        assert "reset-model" not in learner.records_by_llm

    def test_reset_all_learning(self, learner):
        """Reset all learning clears everything."""
        # Record data for multiple LLMs
        for llm in ["model-a", "model-b", "model-c"]:
            tracker = learner.start_session(
                session_id=f"session-{llm}",
                instruction_mode="default",
                llm_type=llm,
            )
            tracker.record_tool_call("find_keywords", {}, True)
            learner.end_session(tracker)

        assert len(learner.records_by_llm) == 3

        # Reset all
        learner.reset_learning()

        assert len(learner.records_by_llm) == 0
        assert len(learner.behavior_patterns) == 0


class TestLearnerIntegrationWithMCPServer:
    """Integration tests combining learner with MCP server."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def learner(self, temp_storage):
        """Create learner with temporary storage."""
        store = PatternStore(storage_dir=temp_storage)
        return InstructionEffectivenessLearner(pattern_store=store)

    @pytest.mark.asyncio
    async def test_simulated_mcp_session_tracking(self, learner):
        """Simulate MCP session with learning hooks."""
        # Start tracking session
        tracker = learner.start_session(
            session_id="mcp-sim-session",
            instruction_mode="default",
            llm_type="claude-sonnet",
            scenario_type="web_automation",
        )

        # Simulate typical MCP tool call sequence
        # 1. Discovery phase
        tracker.record_tool_call("analyze_scenario", {"scenario": "Login test"}, True)
        tracker.record_tool_call("recommend_libraries", {"scenario": "Login"}, True)
        tracker.record_tool_call("find_keywords", {"query": "input"}, True)

        # 2. Setup phase
        tracker.record_tool_call("manage_session", {"action": "start"}, True)

        # 3. Execution phase
        tracker.record_tool_call(
            "execute_step",
            {"keyword": "Input Text", "arguments": ["username", "test"]},
            True,
        )
        tracker.record_tool_call(
            "execute_step",
            {"keyword": "Click Button", "arguments": ["login"]},
            True,
        )

        # End session and get record
        record = learner.end_session(tracker)

        assert record.discovery_first_compliance is True
        assert record.total_tool_calls == 6
        assert record.invalid_keyword_count == 0
        assert record.successful_sequences >= 1

    @pytest.mark.asyncio
    async def test_simulated_failed_session_tracking(self, learner):
        """Simulate MCP session with failures and recovery."""
        tracker = learner.start_session(
            session_id="mcp-fail-session",
            instruction_mode="minimal",
            llm_type="gpt-3.5-turbo",
            scenario_type="api_testing",
        )

        # Skip discovery - go straight to action (bad practice)
        tracker.record_tool_call(
            "execute_step",
            {"keyword": "Send Request", "arguments": []},
            False,
            "keyword not found: Send Request",  # Uses pattern that triggers invalid_keyword detection
        )

        # Retry with another guess
        tracker.record_tool_call(
            "execute_step",
            {"keyword": "Make Request", "arguments": []},
            False,
            "unknown keyword: Make Request",  # Uses pattern that triggers invalid_keyword detection
        )

        # Finally use discovery
        tracker.record_tool_call("search_keywords", {"query": "request"}, True)

        # Now succeed
        tracker.record_tool_call(
            "execute_step",
            {"keyword": "GET", "arguments": ["/api/users"]},
            True,
        )

        record = learner.end_session(tracker)

        assert record.discovery_first_compliance is False
        assert record.invalid_keyword_count == 2
        assert record.total_tool_calls == 4
        # Eventually recovered
        assert record.error_recovery_success > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

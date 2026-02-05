"""Unit tests for MCP instruction learning hooks integration."""

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from robotmcp.optimization.instruction_hooks import (
    SessionMetadata,
    InstructionLearningHooks,
    get_hooks,
    track_tool_call,
)
from robotmcp.optimization.pattern_store import PatternStore


class TestSessionMetadata:
    """Tests for SessionMetadata dataclass."""

    def test_create_metadata(self):
        """Test creating session metadata."""
        metadata = SessionMetadata(
            session_id="test-session",
            instruction_mode="default",
            llm_type="claude-sonnet",
            scenario_type="web_automation",
            start_time=1234567890.0,
        )

        assert metadata.session_id == "test-session"
        assert metadata.instruction_mode == "default"
        assert metadata.llm_type == "claude-sonnet"
        assert metadata.tool_call_count == 0
        assert metadata.first_tool_was_discovery is False

    def test_metadata_defaults(self):
        """Test metadata has correct defaults."""
        metadata = SessionMetadata(
            session_id="s1",
            instruction_mode="minimal",
            llm_type="gpt-4",
            scenario_type="api_testing",
            start_time=time.time(),
        )

        assert metadata.tool_call_count == 0
        assert metadata.first_tool_was_discovery is False


class TestInstructionLearningHooks:
    """Tests for InstructionLearningHooks class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the singleton before each test."""
        InstructionLearningHooks.reset_instance()
        yield
        InstructionLearningHooks.reset_instance()

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def hooks(self, temp_storage):
        """Create hooks with temporary storage."""
        store = PatternStore(storage_dir=temp_storage)
        return InstructionLearningHooks(pattern_store=store)

    def test_init(self, hooks):
        """Test hooks initialization."""
        assert hooks.is_enabled() is True
        assert len(hooks._session_metadata) == 0

    def test_singleton_pattern(self, temp_storage):
        """Test singleton pattern works correctly."""
        store = PatternStore(storage_dir=temp_storage)
        hooks1 = InstructionLearningHooks.get_instance(pattern_store=store)
        hooks2 = InstructionLearningHooks.get_instance()

        assert hooks1 is hooks2

    def test_singleton_reset(self, temp_storage):
        """Test singleton reset works."""
        store = PatternStore(storage_dir=temp_storage)
        hooks1 = InstructionLearningHooks.get_instance(pattern_store=store)
        InstructionLearningHooks.reset_instance()
        hooks2 = InstructionLearningHooks.get_instance(pattern_store=store)

        assert hooks1 is not hooks2

    def test_on_session_start(self, hooks):
        """Test starting a session creates tracker."""
        tracker = hooks.on_session_start(
            session_id="test-session",
            instruction_mode="default",
            llm_type="claude-sonnet",
            scenario_type="web_automation",
        )

        assert tracker is not None
        assert tracker.session_id == "test-session"
        assert "test-session" in hooks._session_metadata
        assert hooks._session_metadata["test-session"].llm_type == "claude-sonnet"

    def test_on_session_start_without_llm_type(self, hooks):
        """Test session start without explicit LLM type."""
        tracker = hooks.on_session_start(
            session_id="test-session",
            instruction_mode="default",
        )

        assert tracker is not None
        # Should detect or default to "unknown"
        assert hooks._session_metadata["test-session"].llm_type in ("unknown", "")

    @patch.dict(os.environ, {"ROBOTMCP_LLM_TYPE": "custom-model"})
    def test_llm_type_from_env(self, temp_storage):
        """Test LLM type detection from environment."""
        store = PatternStore(storage_dir=temp_storage)
        hooks = InstructionLearningHooks(pattern_store=store)

        tracker = hooks.on_session_start(
            session_id="test-session",
            instruction_mode="default",
        )

        assert hooks._session_metadata["test-session"].llm_type == "custom-model"

    @patch.dict(os.environ, {"ANTHROPIC_MODEL": "claude-3-opus"})
    def test_llm_type_from_anthropic_env(self, temp_storage):
        """Test LLM type detection from Anthropic model env."""
        store = PatternStore(storage_dir=temp_storage)
        hooks = InstructionLearningHooks(pattern_store=store)

        tracker = hooks.on_session_start(
            session_id="test-session",
            instruction_mode="default",
        )

        assert hooks._session_metadata["test-session"].llm_type == "claude-3-opus"

    def test_on_tool_call_discovery_first(self, hooks):
        """Test tracking tool calls with discovery first."""
        hooks.on_session_start(
            session_id="test-session",
            instruction_mode="default",
            llm_type="claude-sonnet",
        )

        # First call is a discovery tool
        hooks.on_tool_call(
            session_id="test-session",
            tool_name="find_keywords",
            arguments={"pattern": "click"},
            success=True,
        )

        metadata = hooks._session_metadata["test-session"]
        assert metadata.tool_call_count == 1
        assert metadata.first_tool_was_discovery is True

    def test_on_tool_call_action_first(self, hooks):
        """Test tracking tool calls with action first."""
        hooks.on_session_start(
            session_id="test-session",
            instruction_mode="default",
            llm_type="claude-sonnet",
        )

        # First call is an action tool (not discovery)
        hooks.on_tool_call(
            session_id="test-session",
            tool_name="execute_step",
            arguments={"keyword": "Click"},
            success=True,
        )

        metadata = hooks._session_metadata["test-session"]
        assert metadata.tool_call_count == 1
        assert metadata.first_tool_was_discovery is False

    def test_on_tool_call_state_tool_counts_as_discovery(self, hooks):
        """Test that state tools count as discovery-first."""
        hooks.on_session_start(
            session_id="test-session",
            instruction_mode="default",
            llm_type="claude-sonnet",
        )

        # First call is a state tool
        hooks.on_tool_call(
            session_id="test-session",
            tool_name="get_session_state",
            arguments={},
            success=True,
        )

        metadata = hooks._session_metadata["test-session"]
        assert metadata.first_tool_was_discovery is True

    def test_on_tool_call_with_error(self, hooks):
        """Test tracking failed tool calls."""
        hooks.on_session_start(
            session_id="test-session",
            instruction_mode="default",
            llm_type="claude-sonnet",
        )

        hooks.on_tool_call(
            session_id="test-session",
            tool_name="execute_step",
            arguments={"keyword": "InvalidKeyword"},
            success=False,
            error="keyword not found",
        )

        # Should still track the call
        metadata = hooks._session_metadata["test-session"]
        assert metadata.tool_call_count == 1

    def test_on_tool_call_extracts_error_from_result(self, hooks):
        """Test error extraction from result dict."""
        hooks.on_session_start(
            session_id="test-session",
            instruction_mode="default",
            llm_type="claude-sonnet",
        )

        hooks.on_tool_call(
            session_id="test-session",
            tool_name="execute_step",
            arguments={"keyword": "InvalidKeyword"},
            success=False,
            result={"success": False, "error": "keyword not found"},
        )

        # Error should be recorded by the underlying learner
        tracker = hooks.learner.get_session("test-session")
        assert tracker is not None
        assert len(tracker.tool_calls) == 1

    def test_on_tool_call_untracked_session(self, hooks):
        """Test tool call for untracked session is ignored."""
        # Don't start a session, just call a tool
        hooks.on_tool_call(
            session_id="nonexistent-session",
            tool_name="execute_step",
            arguments={},
            success=True,
        )

        # Should not raise an error
        assert "nonexistent-session" not in hooks._session_metadata

    def test_on_session_end(self, hooks):
        """Test ending a session returns metrics."""
        hooks.on_session_start(
            session_id="test-session",
            instruction_mode="default",
            llm_type="claude-sonnet",
            scenario_type="web_automation",
        )

        hooks.on_tool_call("test-session", "find_keywords", {}, True)
        hooks.on_tool_call("test-session", "execute_step", {"keyword": "Click"}, True)

        result = hooks.on_session_end("test-session")

        assert result is not None
        assert result["session_id"] == "test-session"
        assert result["instruction_mode"] == "default"
        assert result["llm_type"] == "claude-sonnet"
        assert result["discovery_first_compliance"] is True
        assert result["total_tool_calls"] == 2
        assert result["duration_ms"] >= 0

    def test_on_session_end_untracked(self, hooks):
        """Test ending untracked session returns None."""
        result = hooks.on_session_end("nonexistent-session")
        assert result is None

    def test_on_session_end_removes_metadata(self, hooks):
        """Test ending session removes metadata."""
        hooks.on_session_start("test-session", "default", "test-model")
        assert "test-session" in hooks._session_metadata

        hooks.on_session_end("test-session")
        assert "test-session" not in hooks._session_metadata

    def test_get_recommendation(self, hooks):
        """Test getting recommendation."""
        rec = hooks.get_recommendation("claude-sonnet", "web_automation")

        assert rec["llm_type"] == "claude-sonnet"
        assert "recommended_mode" in rec
        assert "confidence" in rec
        assert "reasoning" in rec

    def test_get_statistics(self, hooks):
        """Test getting statistics."""
        hooks.on_session_start("session-1", "default", "claude-sonnet")
        hooks.on_tool_call("session-1", "find_keywords", {}, True)
        hooks.on_session_end("session-1")

        stats = hooks.get_statistics()

        assert stats["enabled"] is True
        assert stats["llm_types_tracked"] >= 0
        assert "total_records" in stats

    def test_get_mode_comparison(self, hooks):
        """Test getting mode comparison."""
        comparison = hooks.get_mode_comparison("unknown-model")

        assert comparison["llm_type"] == "unknown-model"
        assert "comparison_available" in comparison

    def test_persist(self, hooks):
        """Test persisting learned data."""
        hooks.on_session_start("session-1", "default", "test-model")
        hooks.on_tool_call("session-1", "find_keywords", {}, True)
        hooks.on_session_end("session-1")

        # Should not raise
        hooks.persist()


class TestDisabledLearning:
    """Tests for disabled learning scenarios."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the singleton before each test."""
        InstructionLearningHooks.reset_instance()
        yield
        InstructionLearningHooks.reset_instance()

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @patch.dict(os.environ, {"ROBOTMCP_DISABLE_LEARNING": "1"})
    def test_learning_disabled_via_env(self, temp_storage):
        """Test learning is disabled via environment variable."""
        store = PatternStore(storage_dir=temp_storage)
        hooks = InstructionLearningHooks(pattern_store=store)

        assert hooks.is_enabled() is False

    @patch.dict(os.environ, {"ROBOTMCP_DISABLE_LEARNING": "true"})
    def test_learning_disabled_true_string(self, temp_storage):
        """Test learning disabled with 'true' string."""
        store = PatternStore(storage_dir=temp_storage)
        hooks = InstructionLearningHooks(pattern_store=store)

        assert hooks.is_enabled() is False

    @patch.dict(os.environ, {"ROBOTMCP_DISABLE_LEARNING": "yes"})
    def test_learning_disabled_yes_string(self, temp_storage):
        """Test learning disabled with 'yes' string."""
        store = PatternStore(storage_dir=temp_storage)
        hooks = InstructionLearningHooks(pattern_store=store)

        assert hooks.is_enabled() is False

    @patch.dict(os.environ, {"ROBOTMCP_DISABLE_LEARNING": "1"})
    def test_session_start_returns_none_when_disabled(self, temp_storage):
        """Test session start returns None when disabled."""
        store = PatternStore(storage_dir=temp_storage)
        hooks = InstructionLearningHooks(pattern_store=store)

        tracker = hooks.on_session_start("test-session", "default", "test-model")
        assert tracker is None

    @patch.dict(os.environ, {"ROBOTMCP_DISABLE_LEARNING": "1"})
    def test_tool_call_noop_when_disabled(self, temp_storage):
        """Test tool call is a no-op when disabled."""
        store = PatternStore(storage_dir=temp_storage)
        hooks = InstructionLearningHooks(pattern_store=store)

        # Should not raise
        hooks.on_tool_call("test-session", "execute_step", {}, True)

    @patch.dict(os.environ, {"ROBOTMCP_DISABLE_LEARNING": "1"})
    def test_session_end_returns_none_when_disabled(self, temp_storage):
        """Test session end returns None when disabled."""
        store = PatternStore(storage_dir=temp_storage)
        hooks = InstructionLearningHooks(pattern_store=store)

        result = hooks.on_session_end("test-session")
        assert result is None

    @patch.dict(os.environ, {"ROBOTMCP_DISABLE_LEARNING": "1"})
    def test_recommendation_returns_default_when_disabled(self, temp_storage):
        """Test recommendation returns default when disabled."""
        store = PatternStore(storage_dir=temp_storage)
        hooks = InstructionLearningHooks(pattern_store=store)

        rec = hooks.get_recommendation("claude-sonnet", "web_automation")

        assert rec["recommended_mode"] == "default"
        assert rec["confidence"] == 0.0
        assert "disabled" in rec["reasoning"].lower()

    @patch.dict(os.environ, {"ROBOTMCP_DISABLE_LEARNING": "1"})
    def test_statistics_shows_disabled(self, temp_storage):
        """Test statistics shows learning is disabled."""
        store = PatternStore(storage_dir=temp_storage)
        hooks = InstructionLearningHooks(pattern_store=store)

        stats = hooks.get_statistics()

        assert stats["enabled"] is False
        assert "reason" in stats


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the singleton before each test."""
        InstructionLearningHooks.reset_instance()
        yield
        InstructionLearningHooks.reset_instance()

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_get_hooks_returns_singleton(self, temp_storage):
        """Test get_hooks returns the singleton."""
        # Initialize with a pattern store first
        store = PatternStore(storage_dir=temp_storage)
        InstructionLearningHooks.get_instance(pattern_store=store)

        hooks1 = get_hooks()
        hooks2 = get_hooks()

        assert hooks1 is hooks2
        assert isinstance(hooks1, InstructionLearningHooks)

    def test_track_tool_call_convenience(self, temp_storage):
        """Test track_tool_call convenience function."""
        store = PatternStore(storage_dir=temp_storage)
        hooks = InstructionLearningHooks.get_instance(pattern_store=store)

        # Start a session first
        hooks.on_session_start("test-session", "default", "test-model")

        # Use convenience function
        track_tool_call(
            session_id="test-session",
            tool_name="find_keywords",
            arguments={"pattern": "click"},
            success=True,
        )

        # Verify it was tracked
        tracker = hooks.learner.get_session("test-session")
        assert len(tracker.tool_calls) == 1


class TestIntegrationScenarios:
    """Integration tests for realistic usage scenarios."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the singleton before each test."""
        InstructionLearningHooks.reset_instance()
        yield
        InstructionLearningHooks.reset_instance()

    @pytest.fixture
    def temp_storage(self):
        """Create a temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def hooks(self, temp_storage):
        """Create hooks with temporary storage."""
        store = PatternStore(storage_dir=temp_storage)
        return InstructionLearningHooks(pattern_store=store)

    def test_full_session_workflow(self, hooks):
        """Test a complete session workflow."""
        # Start session
        tracker = hooks.on_session_start(
            session_id="workflow-session",
            instruction_mode="default",
            llm_type="claude-sonnet",
            scenario_type="web_automation",
        )

        assert tracker is not None

        # Simulate typical tool call sequence
        hooks.on_tool_call("workflow-session", "analyze_scenario", {"scenario": "test login"}, True)
        hooks.on_tool_call("workflow-session", "recommend_libraries", {"scenario": "test login"}, True)
        hooks.on_tool_call("workflow-session", "find_keywords", {"pattern": "open browser"}, True)
        hooks.on_tool_call("workflow-session", "execute_step", {"keyword": "New Browser"}, True)
        hooks.on_tool_call("workflow-session", "execute_step", {"keyword": "Go To", "args": ["https://example.com"]}, True)

        # End session
        result = hooks.on_session_end("workflow-session")

        assert result["discovery_first_compliance"] is True
        assert result["total_tool_calls"] == 5
        assert result["invalid_keyword_count"] == 0

    def test_session_with_errors(self, hooks):
        """Test session with tool call errors."""
        hooks.on_session_start(
            session_id="error-session",
            instruction_mode="default",
            llm_type="gpt-4",
        )

        # Start with action (not discovery-first)
        hooks.on_tool_call("error-session", "execute_step", {"keyword": "InvalidKeyword"}, False, "keyword not found")
        hooks.on_tool_call("error-session", "execute_step", {"keyword": "AnotherBadKeyword"}, False, "unknown keyword")

        # Then use discovery
        hooks.on_tool_call("error-session", "find_keywords", {"pattern": "click"}, True)
        hooks.on_tool_call("error-session", "execute_step", {"keyword": "Click"}, True)

        result = hooks.on_session_end("error-session")

        assert result["discovery_first_compliance"] is False
        assert result["invalid_keyword_count"] == 2
        assert result["total_tool_calls"] == 4

    def test_multiple_sessions_same_llm(self, hooks):
        """Test multiple sessions for the same LLM type."""
        for i in range(5):
            hooks.on_session_start(f"session-{i}", "default", "test-model", "web_automation")
            hooks.on_tool_call(f"session-{i}", "find_keywords", {}, True)
            hooks.on_tool_call(f"session-{i}", "execute_step", {}, True)
            hooks.on_session_end(f"session-{i}")

        stats = hooks.get_statistics()

        assert stats["total_records"] >= 5

    def test_concurrent_sessions(self, hooks):
        """Test handling multiple concurrent sessions."""
        # Start multiple sessions
        hooks.on_session_start("session-a", "default", "model-a")
        hooks.on_session_start("session-b", "minimal", "model-b")
        hooks.on_session_start("session-c", "verbose", "model-c")

        # Interleave tool calls
        hooks.on_tool_call("session-a", "find_keywords", {}, True)
        hooks.on_tool_call("session-b", "execute_step", {}, True)  # No discovery first
        hooks.on_tool_call("session-c", "analyze_scenario", {}, True)
        hooks.on_tool_call("session-a", "execute_step", {}, True)
        hooks.on_tool_call("session-c", "execute_step", {}, True)

        # End sessions
        result_a = hooks.on_session_end("session-a")
        result_b = hooks.on_session_end("session-b")
        result_c = hooks.on_session_end("session-c")

        assert result_a["discovery_first_compliance"] is True
        assert result_b["discovery_first_compliance"] is False
        assert result_c["discovery_first_compliance"] is True

    def test_persist_and_load(self, temp_storage):
        """Test persistence across hooks instances."""
        store = PatternStore(storage_dir=temp_storage)

        # First instance - record sessions
        hooks1 = InstructionLearningHooks(pattern_store=store)
        for i in range(12):
            hooks1.on_session_start(f"session-{i}", "default", "persist-test")
            hooks1.on_tool_call(f"session-{i}", "find_keywords", {}, True)
            hooks1.on_session_end(f"session-{i}")
        hooks1.persist()

        # Second instance - should load persisted data
        hooks2 = InstructionLearningHooks(pattern_store=store)

        stats = hooks2.get_statistics()
        assert stats["total_records"] >= 12

    def test_session_start_ends_existing_session(self, hooks):
        """Test that starting a new session with same ID ends the existing one."""
        # Start first session
        hooks.on_session_start(
            session_id="reused-session",
            instruction_mode="default",
            llm_type="model-v1",
            scenario_type="web_automation",
        )
        hooks.on_tool_call("reused-session", "find_keywords", {}, True)
        hooks.on_tool_call("reused-session", "execute_step", {"keyword": "Click"}, True)

        # Start new session with same ID - should end the old one
        hooks.on_session_start(
            session_id="reused-session",
            instruction_mode="minimal",
            llm_type="model-v2",
            scenario_type="api_testing",
        )

        # Verify new session has fresh state
        metadata = hooks._session_metadata["reused-session"]
        assert metadata.llm_type == "model-v2"
        assert metadata.instruction_mode == "minimal"
        assert metadata.scenario_type == "api_testing"
        assert metadata.tool_call_count == 0  # Fresh session

        # Verify old session data was persisted (check learner records)
        stats = hooks.get_statistics()
        assert stats["total_records"] >= 1  # At least the ended session

    def test_shutdown_ends_all_sessions(self, hooks):
        """Test shutdown ends all active sessions."""
        # Start multiple sessions
        hooks.on_session_start("session-1", "default", "test-model")
        hooks.on_tool_call("session-1", "find_keywords", {}, True)

        hooks.on_session_start("session-2", "minimal", "test-model")
        hooks.on_tool_call("session-2", "execute_step", {}, True)

        hooks.on_session_start("session-3", "verbose", "test-model")
        hooks.on_tool_call("session-3", "analyze_scenario", {}, True)

        # Verify sessions are active
        assert len(hooks._session_metadata) == 3

        # Shutdown
        result = hooks.shutdown()

        # Verify all sessions ended
        assert result["enabled"] is True
        assert result["sessions_ended"] == 3
        assert result["persisted"] is True
        assert len(hooks._session_metadata) == 0

    def test_shutdown_persists_patterns(self, temp_storage):
        """Test shutdown persists patterns to disk."""
        store = PatternStore(storage_dir=temp_storage)
        hooks = InstructionLearningHooks(pattern_store=store)

        # Create sessions with tool calls
        hooks.on_session_start("session-1", "default", "test-model")
        hooks.on_tool_call("session-1", "find_keywords", {}, True)
        hooks.on_tool_call("session-1", "execute_step", {}, True)

        # Shutdown (should persist)
        result = hooks.shutdown()

        assert result["persisted"] is True
        assert "storage_dir" in result

        # Verify data persisted to disk
        records = store.list_keys("instruction_records")
        assert len(records) >= 1

    @patch.dict(os.environ, {"ROBOTMCP_DISABLE_LEARNING": "1"})
    def test_shutdown_when_disabled(self, temp_storage):
        """Test shutdown when learning is disabled."""
        store = PatternStore(storage_dir=temp_storage)
        hooks = InstructionLearningHooks(pattern_store=store)

        result = hooks.shutdown()

        assert result["enabled"] is False
        assert result["sessions_ended"] == 0
        assert result["persisted"] is False

    def test_shutdown_handles_empty_sessions(self, hooks):
        """Test shutdown with no active sessions."""
        result = hooks.shutdown()

        assert result["enabled"] is True
        assert result["sessions_ended"] == 0
        assert result["persisted"] is True

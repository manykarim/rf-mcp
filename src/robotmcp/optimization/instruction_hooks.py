"""Integration hooks for MCP tool execution with instruction learning.

This module provides integration between the MCP server tool execution and the
instruction effectiveness learning system. It tracks tool calls, sessions, and
learns which instruction modes work best for different LLM types.

Usage:
    # In server.py or tool handlers
    from robotmcp.optimization.instruction_hooks import InstructionLearningHooks

    hooks = InstructionLearningHooks.get_instance()

    # When session starts
    hooks.on_session_start(session_id, instruction_mode="default", llm_type="claude-sonnet")

    # For each tool call
    hooks.on_tool_call(session_id, "execute_step", {"keyword": "Click"}, success=True)

    # When session ends
    hooks.on_session_end(session_id)

    # Get recommendations
    rec = hooks.get_recommendation("claude-sonnet", "web_automation")
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .instruction_learner import InstructionEffectivenessLearner, SessionTracker
from .pattern_store import PatternStore

logger = logging.getLogger(__name__)


@dataclass
class SessionMetadata:
    """Metadata about an active session for learning.

    Attributes:
        session_id: Unique identifier for the session
        instruction_mode: Active instruction mode (off/minimal/default/verbose)
        llm_type: Identifier for the LLM type
        scenario_type: Type of scenario being executed
        start_time: Unix timestamp when session started
        tool_call_count: Number of tool calls in session
        first_tool_was_discovery: Whether first tool was a discovery tool
    """
    session_id: str
    instruction_mode: str
    llm_type: str
    scenario_type: str
    start_time: float
    tool_call_count: int = 0
    first_tool_was_discovery: bool = False


class InstructionLearningHooks:
    """Hooks for integrating instruction learning with MCP tool execution.

    This class provides a singleton interface for tracking tool calls and
    sessions to learn optimal instruction modes for different LLM types.

    Thread Safety:
        The class uses a singleton pattern but is not thread-safe. In async
        contexts, all operations should be performed in the same event loop.

    Example:
        hooks = InstructionLearningHooks.get_instance()

        # Session lifecycle
        hooks.on_session_start("session-1", "default", "claude-sonnet", "web_automation")
        hooks.on_tool_call("session-1", "find_keywords", {"pattern": "click"}, True)
        hooks.on_tool_call("session-1", "execute_step", {"keyword": "Click"}, True)
        hooks.on_session_end("session-1")

        # Get learned recommendations
        rec = hooks.get_recommendation("claude-sonnet", "web_automation")
    """

    _instance: Optional["InstructionLearningHooks"] = None

    # Discovery tools that should be called first (discovery-first compliance)
    DISCOVERY_TOOLS = {
        "find_keywords",           # Search for keywords by name/pattern
        "get_keyword_info",        # Get detailed keyword documentation
        "get_locator_guidance",    # Get guidance for element locators
        "analyze_scenario",        # Analyze test scenarios
        "recommend_libraries",     # Get library recommendations
        "check_library_availability",  # Check if libraries are available
        # Legacy/alternative tool names
        "list_keywords",           # Legacy name for find_keywords
        "search_keywords",         # Legacy name for find_keywords
        "list_libraries",          # List available libraries
        "discover_keywords",       # Discover keywords by intent
    }

    # State checking tools (also count for discovery-first compliance)
    STATE_TOOLS = {
        "get_session_state",       # Get current session state
        "get_page_snapshot",       # Get page snapshot/DOM
        "get_application_state",   # Get application state
    }

    # Action tools (execution tools)
    ACTION_TOOLS = {
        "execute_step",            # Execute a single test step
        "execute_flow",            # Execute a flow of steps
        "manage_session",          # Manage test sessions
        "build_test_suite",        # Build a test suite
        "run_test_suite",          # Run a test suite
        "set_library_search_order",  # Set library search order
    }

    def __init__(self, pattern_store: Optional[PatternStore] = None):
        """Initialize the hooks integration.

        Args:
            pattern_store: Pattern store for persistence. Creates default if None.
        """
        self.pattern_store = pattern_store or PatternStore()
        self.learner = InstructionEffectivenessLearner(pattern_store=self.pattern_store)
        self._session_metadata: Dict[str, SessionMetadata] = {}
        self._enabled = self._check_enabled()

        if self._enabled:
            logger.info("Instruction learning hooks initialized")
        else:
            logger.debug("Instruction learning hooks disabled via ROBOTMCP_DISABLE_LEARNING")

    def _check_enabled(self) -> bool:
        """Check if learning is enabled via environment variable."""
        disable_var = os.environ.get("ROBOTMCP_DISABLE_LEARNING", "").lower()
        return disable_var not in ("1", "true", "yes")

    @classmethod
    def get_instance(cls, pattern_store: Optional[PatternStore] = None) -> "InstructionLearningHooks":
        """Get or create the singleton instance.

        Args:
            pattern_store: Optional pattern store (only used on first call)

        Returns:
            The singleton InstructionLearningHooks instance
        """
        if cls._instance is None:
            cls._instance = cls(pattern_store=pattern_store)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (primarily for testing)."""
        cls._instance = None

    def on_session_start(
        self,
        session_id: str,
        instruction_mode: str = "default",
        llm_type: Optional[str] = None,
        scenario_type: str = "unknown",
    ) -> Optional[SessionTracker]:
        """Called when a session starts (typically via manage_session action="start").

        This starts tracking tool calls for the session to measure instruction
        mode effectiveness. If an existing session with the same ID exists,
        it will be ended first and its metrics persisted.

        Args:
            session_id: Unique identifier for the session
            instruction_mode: Active instruction mode (off/minimal/default/verbose)
            llm_type: Identifier for the LLM type (e.g., "claude-sonnet", "gpt-4")
            scenario_type: Type of scenario being executed

        Returns:
            SessionTracker for the session, or None if learning is disabled
        """
        if not self._enabled:
            return None

        # End any existing session with the same ID to persist its metrics
        existing_tracker = self.learner.get_session(session_id)
        if existing_tracker is not None:
            logger.debug(f"Ending existing session {session_id} before starting new one")
            self.on_session_end(session_id)

        # Try to detect LLM type from environment if not provided
        if llm_type is None:
            llm_type = self._detect_llm_type()

        # Store metadata
        metadata = SessionMetadata(
            session_id=session_id,
            instruction_mode=instruction_mode,
            llm_type=llm_type,
            scenario_type=scenario_type,
            start_time=time.time(),
        )
        self._session_metadata[session_id] = metadata

        # Start tracking with the learner
        tracker = self.learner.start_session(
            session_id=session_id,
            instruction_mode=instruction_mode,
            llm_type=llm_type,
            scenario_type=scenario_type,
        )

        logger.debug(
            f"Started instruction tracking for session {session_id}, "
            f"mode={instruction_mode}, llm={llm_type}, scenario={scenario_type}"
        )

        return tracker

    def _detect_llm_type(self) -> str:
        """Detect LLM type from environment variables or defaults.

        Returns:
            Detected LLM type string or "unknown"
        """
        # Check common environment variables for LLM type
        llm_type = os.environ.get("ROBOTMCP_LLM_TYPE")
        if llm_type:
            return llm_type

        # Check for Anthropic model
        anthropic_model = os.environ.get("ANTHROPIC_MODEL")
        if anthropic_model:
            return anthropic_model

        # Check for OpenAI model
        openai_model = os.environ.get("OPENAI_MODEL")
        if openai_model:
            return openai_model

        return "unknown"

    def on_tool_call(
        self,
        session_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        success: bool,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called after each tool call to record the outcome.

        This tracks tool call sequences, discovery-first compliance, and error
        patterns to learn optimal instruction modes.

        Args:
            session_id: Session identifier (must match a started session)
            tool_name: Name of the tool that was called
            arguments: Arguments passed to the tool
            success: Whether the call succeeded
            error: Error message if the call failed
            result: Optional result dict from the tool call
        """
        if not self._enabled:
            return

        # Get the session tracker
        tracker = self.learner.get_session(session_id)
        if tracker is None:
            # Session not tracked - might be an untracked session or learning disabled
            logger.debug(f"No tracker for session {session_id}, skipping tool call record")
            return

        # Update metadata
        metadata = self._session_metadata.get(session_id)
        if metadata:
            metadata.tool_call_count += 1

            # Check if this is the first tool call and if it's a discovery tool
            if metadata.tool_call_count == 1:
                is_discovery = tool_name in self.DISCOVERY_TOOLS
                is_state = tool_name in self.STATE_TOOLS
                metadata.first_tool_was_discovery = is_discovery or is_state

        # Extract error from result if not provided
        if error is None and result:
            if not result.get("success", True):
                error = result.get("error") or result.get("message")

        # Record the tool call
        tracker.record_tool_call(
            tool_name=tool_name,
            arguments=arguments or {},
            success=success,
            error=error,
        )

        logger.debug(
            f"Recorded tool call: session={session_id}, tool={tool_name}, "
            f"success={success}, error={error}"
        )

    def on_session_end(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Called when a session ends to finalize learning.

        This computes effectiveness metrics for the session and updates the
        learned patterns for the LLM type.

        Args:
            session_id: Session identifier to end

        Returns:
            Dictionary with session metrics, or None if session not found
        """
        if not self._enabled:
            return None

        # Get the session tracker
        tracker = self.learner.get_session(session_id)
        if tracker is None:
            logger.debug(f"No tracker for session {session_id}, cannot end")
            return None

        # Get metadata
        metadata = self._session_metadata.pop(session_id, None)

        # End the session and get effectiveness record
        record = self.learner.end_session(tracker)

        # Compute session duration
        duration_ms = 0
        if metadata:
            duration_ms = (time.time() - metadata.start_time) * 1000

        result = {
            "session_id": session_id,
            "instruction_mode": record.instruction_mode,
            "llm_type": record.llm_type,
            "scenario_type": record.scenario_type,
            "discovery_first_compliance": record.discovery_first_compliance,
            "invalid_keyword_count": record.invalid_keyword_count,
            "total_tool_calls": record.total_tool_calls,
            "successful_sequences": record.successful_sequences,
            "failed_sequences": record.failed_sequences,
            "error_recovery_success": record.error_recovery_success,
            "duration_ms": duration_ms,
        }

        logger.debug(
            f"Ended instruction tracking for session {session_id}: "
            f"compliance={record.discovery_first_compliance}, "
            f"invalid_keywords={record.invalid_keyword_count}, "
            f"tool_calls={record.total_tool_calls}"
        )

        return result

    def get_recommendation(
        self,
        llm_type: str,
        scenario_type: str = "unknown",
    ) -> Dict[str, Any]:
        """Get instruction mode recommendation for an LLM type.

        This returns the learned optimal instruction mode based on historical
        session data, or falls back to heuristics for unknown LLM types.

        Args:
            llm_type: Type of LLM to get recommendation for
            scenario_type: Type of scenario (for context-aware recommendations)

        Returns:
            Dictionary with recommendation details including:
            - recommended_mode: The suggested instruction mode
            - confidence: Confidence in the recommendation (0-1)
            - reasoning: Human-readable reasoning for the recommendation
            - sample_count: Number of sessions used for learning
        """
        if not self._enabled:
            return {
                "llm_type": llm_type,
                "scenario_type": scenario_type,
                "recommended_mode": "default",
                "confidence": 0.0,
                "reasoning": "Learning disabled - using default mode",
                "sample_count": 0,
            }

        return self.learner.get_recommendation(llm_type, scenario_type)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about instruction learning.

        Returns:
            Dictionary with learning statistics including tracked LLM types,
            total records, learned patterns, and active sessions.
        """
        if not self._enabled:
            return {
                "enabled": False,
                "reason": "Learning disabled via ROBOTMCP_DISABLE_LEARNING",
            }

        stats = self.learner.get_statistics()
        stats["enabled"] = True
        stats["active_tracked_sessions"] = len(self._session_metadata)
        return stats

    def get_mode_comparison(self, llm_type: str) -> Dict[str, Any]:
        """Get a comparison of instruction modes for an LLM type.

        Args:
            llm_type: The LLM type to compare modes for

        Returns:
            Dictionary with mode comparison metrics or indication that
            comparison is not available.
        """
        if not self._enabled:
            return {
                "llm_type": llm_type,
                "comparison_available": False,
                "reason": "Learning disabled",
            }

        return self.learner.get_mode_comparison(llm_type)

    def is_enabled(self) -> bool:
        """Check if learning is currently enabled.

        Returns:
            True if learning is enabled, False otherwise
        """
        return self._enabled

    def persist(self) -> None:
        """Persist all learned patterns to storage."""
        if self._enabled:
            self.learner.persist_all()
            logger.debug("Persisted instruction learning data")

    def shutdown(self) -> Dict[str, Any]:
        """Shutdown the learning system, ending all active sessions and persisting data.

        This should be called when the MCP server is shutting down to ensure
        all collected metrics are saved to disk.

        Returns:
            Dictionary with shutdown statistics
        """
        if not self._enabled:
            return {"enabled": False, "sessions_ended": 0, "persisted": False}

        # End all active sessions to persist their metrics
        active_session_ids = list(self._session_metadata.keys())
        sessions_ended = 0

        for session_id in active_session_ids:
            try:
                result = self.on_session_end(session_id)
                if result is not None:
                    sessions_ended += 1
                    logger.debug(
                        f"Ended session {session_id} on shutdown: "
                        f"tool_calls={result.get('total_tool_calls', 0)}"
                    )
            except Exception as e:
                logger.debug(f"Failed to end session {session_id} on shutdown: {e}")

        # Persist all learned patterns
        try:
            self.learner.persist_all()
            persisted = True
            logger.info(
                f"Instruction learning shutdown: ended {sessions_ended} sessions, "
                f"persisted patterns to {self.pattern_store.storage_dir}"
            )
        except Exception as e:
            persisted = False
            logger.warning(f"Failed to persist instruction learning data on shutdown: {e}")

        return {
            "enabled": True,
            "sessions_ended": sessions_ended,
            "persisted": persisted,
            "storage_dir": str(self.pattern_store.storage_dir),
        }


# Convenience functions for direct usage without singleton
def get_hooks() -> InstructionLearningHooks:
    """Get the singleton hooks instance.

    Returns:
        The InstructionLearningHooks singleton instance
    """
    return InstructionLearningHooks.get_instance()


def track_tool_call(
    session_id: str,
    tool_name: str,
    arguments: Dict[str, Any],
    success: bool,
    error: Optional[str] = None,
) -> None:
    """Convenience function to track a tool call.

    Args:
        session_id: Session identifier
        tool_name: Name of the tool called
        arguments: Arguments passed to the tool
        success: Whether the call succeeded
        error: Error message if failed
    """
    hooks = get_hooks()
    hooks.on_tool_call(session_id, tool_name, arguments, success, error)

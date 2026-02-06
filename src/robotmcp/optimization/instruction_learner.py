"""Self-learning system for MCP instruction effectiveness tracking.

This module provides tracking and learning capabilities to understand which
instruction modes work best for different LLM types and scenarios. It tracks
tool call sequences, discovery compliance, and adapts recommendations over time.

Storage: ~/.rf-mcp/instruction_learning/
"""

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import statistics

from .pattern_store import PatternStore


class InstructionMode(Enum):
    """Available instruction modes for MCP server."""
    OFF = "off"           # No instructions (for advanced users/large models)
    MINIMAL = "minimal"   # Brief hints only
    DEFAULT = "default"   # Standard discovery-first instructions
    VERBOSE = "verbose"   # Detailed step-by-step guidance


@dataclass
class ToolCallEvent:
    """A single tool call event for sequence tracking.

    Attributes:
        tool_name: Name of the tool called
        timestamp: Unix timestamp of the call
        success: Whether the call succeeded
        is_discovery: Whether this is a discovery tool (find_keywords, etc.)
        is_state_tool: Whether this is a state checking tool
        error_type: Type of error if failed (e.g., "invalid_keyword", "stale_ref")
        arguments_hash: Hash of arguments for pattern matching (privacy-preserving)
    """
    tool_name: str
    timestamp: float
    success: bool
    is_discovery: bool = False
    is_state_tool: bool = False
    error_type: Optional[str] = None
    arguments_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "tool_name": self.tool_name,
            "timestamp": self.timestamp,
            "success": self.success,
            "is_discovery": self.is_discovery,
            "is_state_tool": self.is_state_tool,
            "error_type": self.error_type,
            "arguments_hash": self.arguments_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCallEvent":
        """Create from dictionary."""
        return cls(
            tool_name=data.get("tool_name", "unknown"),
            timestamp=data.get("timestamp", 0),
            success=data.get("success", False),
            is_discovery=data.get("is_discovery", False),
            is_state_tool=data.get("is_state_tool", False),
            error_type=data.get("error_type"),
            arguments_hash=data.get("arguments_hash"),
        )


@dataclass
class InstructionEffectivenessRecord:
    """Record of instruction mode effectiveness for a session.

    Attributes:
        instruction_mode: The instruction mode used
        llm_type: Identifier for the LLM type (e.g., "claude-sonnet", "gpt-4")
        session_id: Unique session identifier
        timestamp: Unix timestamp of session start
        discovery_first_compliance: Did LLM call discovery tools before action?
        invalid_keyword_count: Number of invalid keyword calls
        total_tool_calls: Total number of tool calls
        successful_sequences: Number of successful tool call sequences
        failed_sequences: Number of failed sequences
        error_recovery_success: Rate of successful error recovery
        scenario_type: Type of scenario (e.g., "web_automation", "api_testing")
    """
    instruction_mode: str
    llm_type: str
    session_id: str
    timestamp: float
    discovery_first_compliance: bool = False
    invalid_keyword_count: int = 0
    total_tool_calls: int = 0
    successful_sequences: int = 0
    failed_sequences: int = 0
    error_recovery_success: float = 0.0
    scenario_type: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "instruction_mode": self.instruction_mode,
            "llm_type": self.llm_type,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "discovery_first_compliance": self.discovery_first_compliance,
            "invalid_keyword_count": self.invalid_keyword_count,
            "total_tool_calls": self.total_tool_calls,
            "successful_sequences": self.successful_sequences,
            "failed_sequences": self.failed_sequences,
            "error_recovery_success": self.error_recovery_success,
            "scenario_type": self.scenario_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstructionEffectivenessRecord":
        """Create from dictionary."""
        return cls(
            instruction_mode=data.get("instruction_mode", "default"),
            llm_type=data.get("llm_type", "unknown"),
            session_id=data.get("session_id", ""),
            timestamp=data.get("timestamp", 0),
            discovery_first_compliance=data.get("discovery_first_compliance", False),
            invalid_keyword_count=data.get("invalid_keyword_count", 0),
            total_tool_calls=data.get("total_tool_calls", 0),
            successful_sequences=data.get("successful_sequences", 0),
            failed_sequences=data.get("failed_sequences", 0),
            error_recovery_success=data.get("error_recovery_success", 0.0),
            scenario_type=data.get("scenario_type", "unknown"),
        )


@dataclass
class LLMBehaviorPattern:
    """Aggregated behavior pattern for an LLM type.

    Attributes:
        llm_type: Identifier for the LLM type
        sample_count: Number of sessions analyzed
        preferred_instruction_mode: Most effective instruction mode
        discovery_compliance_by_mode: Compliance rate per instruction mode
        invalid_keyword_rate_by_mode: Invalid keyword rate per mode
        success_rate_by_mode: Overall success rate per mode
        best_scenario_types: Scenario types this LLM handles well
        problematic_scenario_types: Scenario types that cause issues
        avg_tool_calls_per_session: Average number of tool calls
        confidence: Confidence in the pattern (based on sample count)
    """
    llm_type: str
    sample_count: int = 0
    preferred_instruction_mode: str = "default"
    discovery_compliance_by_mode: Dict[str, float] = field(default_factory=dict)
    invalid_keyword_rate_by_mode: Dict[str, float] = field(default_factory=dict)
    success_rate_by_mode: Dict[str, float] = field(default_factory=dict)
    best_scenario_types: List[str] = field(default_factory=list)
    problematic_scenario_types: List[str] = field(default_factory=list)
    avg_tool_calls_per_session: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "llm_type": self.llm_type,
            "sample_count": self.sample_count,
            "preferred_instruction_mode": self.preferred_instruction_mode,
            "discovery_compliance_by_mode": self.discovery_compliance_by_mode,
            "invalid_keyword_rate_by_mode": self.invalid_keyword_rate_by_mode,
            "success_rate_by_mode": self.success_rate_by_mode,
            "best_scenario_types": self.best_scenario_types,
            "problematic_scenario_types": self.problematic_scenario_types,
            "avg_tool_calls_per_session": self.avg_tool_calls_per_session,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMBehaviorPattern":
        """Create from dictionary."""
        return cls(
            llm_type=data.get("llm_type", "unknown"),
            sample_count=data.get("sample_count", 0),
            preferred_instruction_mode=data.get("preferred_instruction_mode", "default"),
            discovery_compliance_by_mode=data.get("discovery_compliance_by_mode", {}),
            invalid_keyword_rate_by_mode=data.get("invalid_keyword_rate_by_mode", {}),
            success_rate_by_mode=data.get("success_rate_by_mode", {}),
            best_scenario_types=data.get("best_scenario_types", []),
            problematic_scenario_types=data.get("problematic_scenario_types", []),
            avg_tool_calls_per_session=data.get("avg_tool_calls_per_session", 0.0),
            confidence=data.get("confidence", 0.0),
        )


@dataclass
class SuccessfulSequence:
    """A successful tool call sequence pattern.

    Attributes:
        sequence_hash: Hash of the tool sequence for matching
        tool_sequence: Ordered list of tool names in the sequence
        scenario_type: Type of scenario this sequence handles
        success_count: Number of times this sequence succeeded
        llm_types: LLM types that successfully used this sequence
        instruction_modes: Instruction modes under which this worked
        avg_duration_ms: Average duration of the sequence
    """
    sequence_hash: str
    tool_sequence: List[str]
    scenario_type: str
    success_count: int = 0
    llm_types: List[str] = field(default_factory=list)
    instruction_modes: List[str] = field(default_factory=list)
    avg_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "sequence_hash": self.sequence_hash,
            "tool_sequence": self.tool_sequence,
            "scenario_type": self.scenario_type,
            "success_count": self.success_count,
            "llm_types": self.llm_types,
            "instruction_modes": self.instruction_modes,
            "avg_duration_ms": self.avg_duration_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuccessfulSequence":
        """Create from dictionary."""
        return cls(
            sequence_hash=data.get("sequence_hash", ""),
            tool_sequence=data.get("tool_sequence", []),
            scenario_type=data.get("scenario_type", "unknown"),
            success_count=data.get("success_count", 0),
            llm_types=data.get("llm_types", []),
            instruction_modes=data.get("instruction_modes", []),
            avg_duration_ms=data.get("avg_duration_ms", 0.0),
        )


class SessionTracker:
    """Tracks tool calls within a single session.

    This class maintains the state of a session for effectiveness tracking,
    recording tool call events and computing metrics at session end.
    """

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

    # State checking tools
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

    # Error types that indicate invalid keyword usage
    INVALID_KEYWORD_ERRORS = {
        "keyword_not_found", "invalid_keyword", "unknown_keyword",
        "no_such_keyword", "keyword_error",
    }

    def __init__(
        self,
        session_id: str,
        instruction_mode: str = "default",
        llm_type: str = "unknown",
        scenario_type: str = "unknown",
    ):
        """Initialize the session tracker.

        Args:
            session_id: Unique identifier for this session
            instruction_mode: Active instruction mode
            llm_type: Type of LLM being used
            scenario_type: Type of scenario being executed
        """
        self.session_id = session_id
        self.instruction_mode = instruction_mode
        self.llm_type = llm_type
        self.scenario_type = scenario_type
        self.start_time = time.time()

        self.tool_calls: List[ToolCallEvent] = []
        self.current_sequence: List[ToolCallEvent] = []
        self.successful_sequences: List[List[str]] = []
        self.failed_sequences: List[List[str]] = []

        self._first_non_discovery_seen = False
        self._discovery_called_first = False
        self._invalid_keyword_count = 0
        self._error_recovery_attempts = 0
        self._error_recovery_successes = 0

    def _hash_arguments(self, arguments: Dict[str, Any]) -> str:
        """Create a privacy-preserving hash of arguments."""
        # Only include argument keys, not values
        keys_str = ",".join(sorted(arguments.keys()))
        return hashlib.md5(keys_str.encode()).hexdigest()[:8]

    def _classify_error(self, error: Optional[str]) -> Optional[str]:
        """Classify an error into known categories for learning."""
        if not error:
            return None

        error_lower = error.lower()

        # Most specific patterns first
        if any(kw in error_lower for kw in ["keyword not found", "no keyword", "unknown keyword"]):
            return "invalid_keyword"
        if any(kw in error_lower for kw in [
            "intercepts pointer events", "element click intercepted",
            "is not clickable at point", "other element would receive the click",
        ]):
            return "element_intercept"
        if "strict mode violation" in error_lower:
            return "strict_mode"
        if any(kw in error_lower for kw in [
            "invalid selector", "not a valid xpath", "selector syntax error",
            "unexpected token",
        ]):
            return "invalid_selector"
        if "element is outside of the viewport" in error_lower or (
            "outside" in error_lower and "viewport" in error_lower
        ):
            return "element_outside_viewport"
        if any(kw in error_lower for kw in [
            "element not interactable", "elementnotinteractableexception",
            "not currently visible", "element is not visible",
            "element is not enabled",
        ]):
            return "element_not_interactable"
        if any(kw in error_lower for kw in [
            "stale element reference", "element is not attached",
            "not attached to the page document", "not present in the current view",
            "expired from the internal cache", "target closed",
        ]):
            return "stale_ref"
        if any(kw in error_lower for kw in [
            "no such frame", "nosuchframeexception",
            "frame was detached", "err_aborted",
        ]):
            return "frame_error"
        if any(kw in error_lower for kw in [
            "unexpected alert open", "unexpectedalertpresentexception",
        ]):
            return "alert_dialog"
        if any(kw in error_lower for kw in [
            "no such window", "nosuchwindowexception",
        ]):
            return "window_error"
        if any(kw in error_lower for kw in [
            "invalid element state", "invalidelementstateexception",
            "may not be manipulated", "element is not editable",
        ]):
            return "invalid_state"
        if any(kw in error_lower for kw in [
            "no such element", "unable to locate element",
            "could not be located", "did not match any elements",
            "page should have contained element", "element not found",
        ]):
            return "element_not_found"
        if "timeout" in error_lower or "timed out" in error_lower:
            return "timeout"
        if "connection" in error_lower or "network" in error_lower:
            return "connection_error"

        return "other"

    def record_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Record a tool call event.

        Args:
            tool_name: Name of the tool called
            arguments: Arguments passed to the tool
            success: Whether the call succeeded
            error: Error message if failed
        """
        is_discovery = tool_name in self.DISCOVERY_TOOLS
        is_state = tool_name in self.STATE_TOOLS
        error_type = self._classify_error(error)

        event = ToolCallEvent(
            tool_name=tool_name,
            timestamp=time.time(),
            success=success,
            is_discovery=is_discovery,
            is_state_tool=is_state,
            error_type=error_type,
            arguments_hash=self._hash_arguments(arguments),
        )

        self.tool_calls.append(event)
        self.current_sequence.append(event)

        # Track discovery-first compliance
        if not self._first_non_discovery_seen:
            if is_discovery or is_state:
                self._discovery_called_first = True
            else:
                self._first_non_discovery_seen = True

        # Track invalid keyword calls
        if error_type == "invalid_keyword":
            self._invalid_keyword_count += 1

        # Track error recovery (discovery/state call after an error)
        if not success and error_type:
            self._error_recovery_attempts += 1
        elif success and self._error_recovery_attempts > 0 and (is_discovery or is_state):
            self._error_recovery_successes += 1

        # Handle sequence completion on success after multiple calls
        if success and len(self.current_sequence) > 1:
            # Check if this completes a meaningful sequence
            if self._is_sequence_complete(event):
                sequence_tools = [e.tool_name for e in self.current_sequence]
                self.successful_sequences.append(sequence_tools)
                self.current_sequence = []

        # Handle sequence failure
        if not success and len(self.current_sequence) > 1:
            sequence_tools = [e.tool_name for e in self.current_sequence]
            self.failed_sequences.append(sequence_tools)
            # Keep sequence open for potential recovery

    def _is_sequence_complete(self, event: ToolCallEvent) -> bool:
        """Determine if a sequence is complete based on the event."""
        # A sequence is complete when we see certain terminal actions
        terminal_tools = {
            "execute_step", "run_keyword", "click", "fill", "type_text",
            "submit", "navigate", "assert", "verify",
        }
        return any(t in event.tool_name.lower() for t in terminal_tools)

    def get_effectiveness_record(self) -> InstructionEffectivenessRecord:
        """Generate an effectiveness record for this session.

        Returns:
            InstructionEffectivenessRecord with session metrics
        """
        total_sequences = len(self.successful_sequences) + len(self.failed_sequences)
        error_recovery_rate = (
            self._error_recovery_successes / max(self._error_recovery_attempts, 1)
        )

        return InstructionEffectivenessRecord(
            instruction_mode=self.instruction_mode,
            llm_type=self.llm_type,
            session_id=self.session_id,
            timestamp=self.start_time,
            discovery_first_compliance=self._discovery_called_first,
            invalid_keyword_count=self._invalid_keyword_count,
            total_tool_calls=len(self.tool_calls),
            successful_sequences=len(self.successful_sequences),
            failed_sequences=len(self.failed_sequences),
            error_recovery_success=error_recovery_rate,
            scenario_type=self.scenario_type,
        )

    def get_successful_sequence_hashes(self) -> List[Tuple[str, List[str]]]:
        """Get hashes and tool lists for successful sequences.

        Returns:
            List of (hash, tool_list) tuples
        """
        results = []
        for sequence in self.successful_sequences:
            seq_str = "->".join(sequence)
            seq_hash = hashlib.md5(seq_str.encode()).hexdigest()[:12]
            results.append((seq_hash, sequence))
        return results


class InstructionEffectivenessLearner:
    """Learn optimal instruction modes based on LLM behavior patterns.

    This class aggregates session data to learn which instruction modes
    work best for different LLM types and scenarios. It provides
    recommendations that improve over time.

    Example:
        learner = InstructionEffectivenessLearner()

        # During session
        tracker = learner.start_session("session-1", "default", "claude-sonnet")
        tracker.record_tool_call("find_keywords", {}, True)
        tracker.record_tool_call("execute_step", {"keyword": "Click"}, True)

        # End session
        learner.end_session(tracker)

        # Get recommendations
        rec = learner.get_recommendation("claude-sonnet", "web_automation")
    """

    # Minimum samples needed for confident recommendations
    MIN_SAMPLES_FOR_RECOMMENDATION = 10

    # Maximum records to keep per LLM type
    MAX_RECORDS_PER_LLM = 500

    # Consolidation threshold (consolidate after N new records)
    CONSOLIDATION_THRESHOLD = 50

    def __init__(self, pattern_store: Optional[PatternStore] = None):
        """Initialize the learner.

        Args:
            pattern_store: Pattern store for persistence. Creates default if None.
        """
        self.pattern_store = pattern_store or PatternStore()

        # In-memory tracking
        self.records_by_llm: Dict[str, List[InstructionEffectivenessRecord]] = defaultdict(list)
        self.behavior_patterns: Dict[str, LLMBehaviorPattern] = {}
        self.successful_sequences: Dict[str, SuccessfulSequence] = {}

        # Active sessions
        self._active_sessions: Dict[str, SessionTracker] = {}

        # Records since last consolidation
        self._records_since_consolidation = 0

        # Load persisted data
        self._load_persisted_data()

    def _load_persisted_data(self) -> None:
        """Load previously stored patterns and records."""
        # Load behavior patterns
        for key in self.pattern_store.list_keys("instruction_patterns"):
            data = self.pattern_store.retrieve("instruction_patterns", key)
            if data and "llm_type" in data:
                pattern = LLMBehaviorPattern.from_dict(data)
                self.behavior_patterns[pattern.llm_type] = pattern

        # Load successful sequences
        for key in self.pattern_store.list_keys("instruction_sequences"):
            data = self.pattern_store.retrieve("instruction_sequences", key)
            if data and "sequence_hash" in data:
                sequence = SuccessfulSequence.from_dict(data)
                self.successful_sequences[sequence.sequence_hash] = sequence

        # Load recent records (for continued learning)
        for key in self.pattern_store.list_keys("instruction_records"):
            data = self.pattern_store.retrieve("instruction_records", key)
            if data and "records" in data:
                llm_type = data.get("llm_type", "unknown")
                for record_data in data["records"][-self.MAX_RECORDS_PER_LLM:]:
                    record = InstructionEffectivenessRecord.from_dict(record_data)
                    self.records_by_llm[llm_type].append(record)

    def start_session(
        self,
        session_id: str,
        instruction_mode: str = "default",
        llm_type: str = "unknown",
        scenario_type: str = "unknown",
    ) -> SessionTracker:
        """Start tracking a new session.

        Args:
            session_id: Unique identifier for this session
            instruction_mode: Active instruction mode
            llm_type: Type of LLM being used
            scenario_type: Type of scenario being executed

        Returns:
            SessionTracker for recording tool calls
        """
        tracker = SessionTracker(
            session_id=session_id,
            instruction_mode=instruction_mode,
            llm_type=llm_type,
            scenario_type=scenario_type,
        )
        self._active_sessions[session_id] = tracker
        return tracker

    def get_session(self, session_id: str) -> Optional[SessionTracker]:
        """Get an active session tracker.

        Args:
            session_id: Session identifier

        Returns:
            SessionTracker if found, None otherwise
        """
        return self._active_sessions.get(session_id)

    def end_session(self, tracker: SessionTracker) -> InstructionEffectivenessRecord:
        """End a session and record its effectiveness.

        Args:
            tracker: The session tracker to finalize

        Returns:
            The effectiveness record for this session
        """
        # Generate effectiveness record
        record = tracker.get_effectiveness_record()

        # Store record
        self.records_by_llm[tracker.llm_type].append(record)

        # Bound record count
        if len(self.records_by_llm[tracker.llm_type]) > self.MAX_RECORDS_PER_LLM:
            self.records_by_llm[tracker.llm_type] = (
                self.records_by_llm[tracker.llm_type][-self.MAX_RECORDS_PER_LLM:]
            )

        # Store successful sequences
        for seq_hash, seq_tools in tracker.get_successful_sequence_hashes():
            if seq_hash in self.successful_sequences:
                seq = self.successful_sequences[seq_hash]
                seq.success_count += 1
                if tracker.llm_type not in seq.llm_types:
                    seq.llm_types.append(tracker.llm_type)
                if tracker.instruction_mode not in seq.instruction_modes:
                    seq.instruction_modes.append(tracker.instruction_mode)
            else:
                self.successful_sequences[seq_hash] = SuccessfulSequence(
                    sequence_hash=seq_hash,
                    tool_sequence=seq_tools,
                    scenario_type=tracker.scenario_type,
                    success_count=1,
                    llm_types=[tracker.llm_type],
                    instruction_modes=[tracker.instruction_mode],
                )

        # Remove from active sessions
        self._active_sessions.pop(tracker.session_id, None)

        # Increment consolidation counter
        self._records_since_consolidation += 1

        # Trigger consolidation if threshold reached
        if self._records_since_consolidation >= self.CONSOLIDATION_THRESHOLD:
            self._consolidate_patterns()
            self._records_since_consolidation = 0

        # Persist record
        self._persist_records(tracker.llm_type)

        return record

    def _persist_records(self, llm_type: str) -> None:
        """Persist records for an LLM type.

        Args:
            llm_type: The LLM type to persist
        """
        records = self.records_by_llm.get(llm_type, [])
        self.pattern_store.store("instruction_records", llm_type, {
            "llm_type": llm_type,
            "records": [r.to_dict() for r in records[-self.MAX_RECORDS_PER_LLM:]],
        })

    def _consolidate_patterns(self) -> None:
        """Consolidate records into behavior patterns."""
        for llm_type, records in self.records_by_llm.items():
            if len(records) < self.MIN_SAMPLES_FOR_RECOMMENDATION:
                continue

            pattern = self._compute_behavior_pattern(llm_type, records)
            self.behavior_patterns[llm_type] = pattern

            # Persist pattern
            self.pattern_store.store(
                "instruction_patterns",
                llm_type,
                pattern.to_dict()
            )

        # Persist successful sequences
        for seq_hash, sequence in self.successful_sequences.items():
            if sequence.success_count >= 3:  # Only persist sequences with 3+ successes
                self.pattern_store.store(
                    "instruction_sequences",
                    seq_hash,
                    sequence.to_dict()
                )

    def _compute_behavior_pattern(
        self,
        llm_type: str,
        records: List[InstructionEffectivenessRecord],
    ) -> LLMBehaviorPattern:
        """Compute behavior pattern from records.

        Args:
            llm_type: The LLM type
            records: List of effectiveness records

        Returns:
            Computed LLMBehaviorPattern
        """
        # Group records by instruction mode
        by_mode: Dict[str, List[InstructionEffectivenessRecord]] = defaultdict(list)
        for record in records:
            by_mode[record.instruction_mode].append(record)

        # Compute metrics per mode
        discovery_compliance_by_mode: Dict[str, float] = {}
        invalid_keyword_rate_by_mode: Dict[str, float] = {}
        success_rate_by_mode: Dict[str, float] = {}

        for mode, mode_records in by_mode.items():
            if not mode_records:
                continue

            # Discovery compliance rate
            compliant = sum(1 for r in mode_records if r.discovery_first_compliance)
            discovery_compliance_by_mode[mode] = compliant / len(mode_records)

            # Invalid keyword rate (per tool call)
            total_calls = sum(r.total_tool_calls for r in mode_records)
            total_invalid = sum(r.invalid_keyword_count for r in mode_records)
            invalid_keyword_rate_by_mode[mode] = (
                total_invalid / max(total_calls, 1)
            )

            # Success rate (based on sequences)
            total_success = sum(r.successful_sequences for r in mode_records)
            total_failed = sum(r.failed_sequences for r in mode_records)
            total_sequences = total_success + total_failed
            success_rate_by_mode[mode] = (
                total_success / max(total_sequences, 1)
            )

        # Determine preferred mode (lowest invalid rate + highest compliance)
        preferred_mode = "default"
        best_score = -1.0

        for mode in by_mode.keys():
            compliance = discovery_compliance_by_mode.get(mode, 0)
            invalid_rate = invalid_keyword_rate_by_mode.get(mode, 1)
            success = success_rate_by_mode.get(mode, 0)

            # Score: high compliance + low invalid rate + high success
            score = compliance * 0.3 + (1 - invalid_rate) * 0.4 + success * 0.3

            if score > best_score:
                best_score = score
                preferred_mode = mode

        # Analyze scenario performance
        scenario_success: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))
        for record in records:
            current = scenario_success[record.scenario_type]
            scenario_success[record.scenario_type] = (
                current[0] + record.successful_sequences,
                current[1] + record.failed_sequences,
            )

        best_scenarios = []
        problematic_scenarios = []
        for scenario, (success, failed) in scenario_success.items():
            total = success + failed
            if total < 3:
                continue
            rate = success / total
            if rate >= 0.8:
                best_scenarios.append(scenario)
            elif rate < 0.5:
                problematic_scenarios.append(scenario)

        # Calculate average tool calls
        avg_tool_calls = statistics.mean(r.total_tool_calls for r in records)

        # Calculate confidence based on sample count
        confidence = min(1.0, len(records) / 100)  # Full confidence at 100 samples

        return LLMBehaviorPattern(
            llm_type=llm_type,
            sample_count=len(records),
            preferred_instruction_mode=preferred_mode,
            discovery_compliance_by_mode=discovery_compliance_by_mode,
            invalid_keyword_rate_by_mode=invalid_keyword_rate_by_mode,
            success_rate_by_mode=success_rate_by_mode,
            best_scenario_types=best_scenarios,
            problematic_scenario_types=problematic_scenarios,
            avg_tool_calls_per_session=avg_tool_calls,
            confidence=confidence,
        )

    def get_recommendation(
        self,
        llm_type: str,
        scenario_type: str = "unknown",
    ) -> Dict[str, Any]:
        """Get instruction mode recommendation for an LLM type.

        Args:
            llm_type: Type of LLM to get recommendation for
            scenario_type: Type of scenario (for context-aware recommendations)

        Returns:
            Dictionary with recommendation details
        """
        # Check for existing pattern
        pattern = self.behavior_patterns.get(llm_type)

        # Check recent records if no pattern
        records = self.records_by_llm.get(llm_type, [])

        if pattern and pattern.confidence >= 0.3:
            # Use learned pattern
            recommended_mode = pattern.preferred_instruction_mode

            # Adjust for problematic scenarios
            if scenario_type in pattern.problematic_scenario_types:
                # Use more verbose instructions for problematic scenarios
                if recommended_mode in ("off", "minimal"):
                    recommended_mode = "default"
                elif recommended_mode == "default":
                    recommended_mode = "verbose"

            return {
                "llm_type": llm_type,
                "scenario_type": scenario_type,
                "recommended_mode": recommended_mode,
                "confidence": pattern.confidence,
                "sample_count": pattern.sample_count,
                "discovery_compliance": pattern.discovery_compliance_by_mode.get(
                    recommended_mode, 0
                ),
                "invalid_keyword_rate": pattern.invalid_keyword_rate_by_mode.get(
                    recommended_mode, 0
                ),
                "is_problematic_scenario": scenario_type in pattern.problematic_scenario_types,
                "is_best_scenario": scenario_type in pattern.best_scenario_types,
                "reasoning": self._generate_reasoning(pattern, scenario_type),
            }

        # Fall back to heuristics for unknown LLMs
        return self._get_heuristic_recommendation(llm_type, scenario_type, len(records))

    def _get_heuristic_recommendation(
        self,
        llm_type: str,
        scenario_type: str,
        sample_count: int,
    ) -> Dict[str, Any]:
        """Get heuristic recommendation for unknown LLM types.

        Args:
            llm_type: Type of LLM
            scenario_type: Type of scenario
            sample_count: Number of samples we have

        Returns:
            Dictionary with heuristic recommendation
        """
        llm_lower = llm_type.lower()

        # Heuristics based on known LLM characteristics
        if any(x in llm_lower for x in ["gpt-4", "claude-3", "opus", "sonnet"]):
            # Large models often work well with minimal instructions
            recommended = "minimal"
            reasoning = "Large models typically need less guidance"
        elif any(x in llm_lower for x in ["gpt-3.5", "claude-instant", "haiku"]):
            # Smaller models benefit from default instructions
            recommended = "default"
            reasoning = "Smaller models benefit from discovery-first guidance"
        elif any(x in llm_lower for x in ["llama", "mistral", "mixtral"]):
            # Open source models may need more guidance
            recommended = "default"
            reasoning = "Open source models may need explicit workflow guidance"
        else:
            # Unknown models get default instructions
            recommended = "default"
            reasoning = "Default instructions for unknown model type"

        return {
            "llm_type": llm_type,
            "scenario_type": scenario_type,
            "recommended_mode": recommended,
            "confidence": 0.0,  # No learned confidence
            "sample_count": sample_count,
            "discovery_compliance": None,
            "invalid_keyword_rate": None,
            "is_problematic_scenario": False,
            "is_best_scenario": False,
            "reasoning": reasoning + " (heuristic - no learned data)",
        }

    def _generate_reasoning(
        self,
        pattern: LLMBehaviorPattern,
        scenario_type: str,
    ) -> str:
        """Generate human-readable reasoning for a recommendation.

        Args:
            pattern: The behavior pattern
            scenario_type: Type of scenario

        Returns:
            Reasoning string
        """
        parts = []

        mode = pattern.preferred_instruction_mode
        compliance = pattern.discovery_compliance_by_mode.get(mode, 0)
        invalid_rate = pattern.invalid_keyword_rate_by_mode.get(mode, 0)

        # Compliance reasoning
        if compliance >= 0.9:
            parts.append(f"High discovery compliance ({compliance:.0%}) with '{mode}' mode")
        elif compliance >= 0.7:
            parts.append(f"Good discovery compliance ({compliance:.0%}) with '{mode}' mode")
        else:
            parts.append(f"Discovery compliance needs improvement ({compliance:.0%})")

        # Invalid keyword reasoning
        if invalid_rate <= 0.05:
            parts.append(f"Very low invalid keyword rate ({invalid_rate:.1%})")
        elif invalid_rate <= 0.15:
            parts.append(f"Acceptable invalid keyword rate ({invalid_rate:.1%})")
        else:
            parts.append(f"High invalid keyword rate ({invalid_rate:.1%}) suggests more guidance needed")

        # Scenario-specific reasoning
        if scenario_type in pattern.best_scenario_types:
            parts.append(f"'{scenario_type}' scenarios perform well with this model")
        elif scenario_type in pattern.problematic_scenario_types:
            parts.append(f"'{scenario_type}' scenarios may need extra attention")

        # Sample count
        parts.append(f"Based on {pattern.sample_count} sessions")

        return "; ".join(parts)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about instruction learning.

        Returns:
            Dictionary with learning statistics
        """
        stats = {
            "llm_types_tracked": len(self.records_by_llm),
            "total_records": sum(len(r) for r in self.records_by_llm.values()),
            "learned_patterns": len(self.behavior_patterns),
            "successful_sequences": len(self.successful_sequences),
            "active_sessions": len(self._active_sessions),
            "records_since_consolidation": self._records_since_consolidation,
            "llm_details": {},
        }

        for llm_type, records in self.records_by_llm.items():
            pattern = self.behavior_patterns.get(llm_type)

            stats["llm_details"][llm_type] = {
                "record_count": len(records),
                "has_learned_pattern": pattern is not None,
                "preferred_mode": pattern.preferred_instruction_mode if pattern else None,
                "confidence": pattern.confidence if pattern else 0.0,
            }

        return stats

    def get_mode_comparison(self, llm_type: str) -> Dict[str, Any]:
        """Get a comparison of instruction modes for an LLM type.

        Args:
            llm_type: The LLM type to compare modes for

        Returns:
            Dictionary with mode comparison metrics
        """
        pattern = self.behavior_patterns.get(llm_type)

        if not pattern:
            return {
                "llm_type": llm_type,
                "comparison_available": False,
                "reason": "Not enough data for comparison",
            }

        modes = list(pattern.discovery_compliance_by_mode.keys())

        comparison = {
            "llm_type": llm_type,
            "comparison_available": True,
            "modes": {},
        }

        for mode in modes:
            comparison["modes"][mode] = {
                "discovery_compliance": pattern.discovery_compliance_by_mode.get(mode, 0),
                "invalid_keyword_rate": pattern.invalid_keyword_rate_by_mode.get(mode, 0),
                "success_rate": pattern.success_rate_by_mode.get(mode, 0),
                "is_recommended": mode == pattern.preferred_instruction_mode,
            }

        return comparison

    def persist_all(self) -> None:
        """Persist all learned data to storage."""
        # Force consolidation
        self._consolidate_patterns()

        # Persist all records
        for llm_type in self.records_by_llm:
            self._persist_records(llm_type)

    def reset_learning(self, llm_type: Optional[str] = None) -> None:
        """Reset learned patterns.

        Args:
            llm_type: Specific LLM type to reset (None for all)
        """
        if llm_type:
            self.records_by_llm.pop(llm_type, None)
            self.behavior_patterns.pop(llm_type, None)
            self.pattern_store.delete("instruction_patterns", llm_type)
            self.pattern_store.delete("instruction_records", llm_type)
        else:
            self.records_by_llm.clear()
            self.behavior_patterns.clear()
            self.successful_sequences.clear()

            for key in self.pattern_store.list_keys("instruction_patterns"):
                self.pattern_store.delete("instruction_patterns", key)
            for key in self.pattern_store.list_keys("instruction_records"):
                self.pattern_store.delete("instruction_records", key)
            for key in self.pattern_store.list_keys("instruction_sequences"):
                self.pattern_store.delete("instruction_sequences", key)

"""Timeout pattern learning for optimal timeout configuration.

This module learns optimal timeout configurations per page type and action
based on observed durations and success rates.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import statistics

from .pattern_store import PatternStore


@dataclass
class TimeoutPattern:
    """
    Learned timeout pattern for page types.

    Attributes:
        page_type: The classified page type
        action: The action type (e.g., "click", "fill", "navigate")
        optimal_timeout_ms: Learned optimal timeout in milliseconds
        p95_duration_ms: 95th percentile of observed durations
        success_rate: Rate of successful operations
        sample_count: Number of samples used
        timeout_triggered_count: Number of times timeout was triggered
    """
    page_type: str
    action: str
    optimal_timeout_ms: int
    p95_duration_ms: int
    success_rate: float
    sample_count: int
    timeout_triggered_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "page_type": self.page_type,
            "action": self.action,
            "optimal_timeout_ms": self.optimal_timeout_ms,
            "p95_duration_ms": self.p95_duration_ms,
            "success_rate": self.success_rate,
            "sample_count": self.sample_count,
            "timeout_triggered_count": self.timeout_triggered_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeoutPattern":
        """Create from dictionary."""
        return cls(
            page_type=data.get("page_type", "unknown"),
            action=data.get("action", "unknown"),
            optimal_timeout_ms=data.get("optimal_timeout_ms", 5000),
            p95_duration_ms=data.get("p95_duration_ms", 0),
            success_rate=data.get("success_rate", 0.0),
            sample_count=data.get("sample_count", 0),
            timeout_triggered_count=data.get("timeout_triggered_count", 0),
        )


@dataclass
class TimeoutResult:
    """
    A single timeout result for learning.

    Attributes:
        timeout_ms: Timeout that was configured
        duration_ms: Actual duration of the operation
        success: Whether the operation succeeded
        triggered: Whether the timeout was triggered
    """
    timeout_ms: int
    duration_ms: int
    success: bool
    triggered: bool


class TimeoutPatternLearner:
    """
    Learn optimal timeout configurations per scenario.

    Tracks timeout results and learns patterns that can be
    applied to future operations of similar types.

    Example:
        learner = TimeoutPatternLearner()
        learner.record_timeout_result(
            page_type="form_page",
            action="fill",
            timeout_ms=5000,
            duration_ms=1200,
            success=True,
            triggered=False
        )
        timeouts = learner.get_optimal_timeouts("form_page", "fill")
    """

    # Default timeouts by action category
    DEFAULT_TIMEOUTS = {
        "action": 5000,        # 5 seconds for clicks, fills, etc.
        "navigation": 60000,   # 60 seconds for page loads
        "prevalidation": 2000, # 2 seconds for actionability checks
    }

    # Actions that use action timeout
    ACTION_TIMEOUT_ACTIONS = {
        "click", "fill", "hover", "select", "check", "uncheck",
        "type", "press", "focus", "blur", "scroll", "drag"
    }

    # Minimum samples before pattern is reliable
    MIN_SAMPLES_FOR_PATTERN = 10

    # Maximum results to keep per key
    MAX_RESULTS_PER_KEY = 500

    # Buffer percentage for optimal timeout calculation
    TIMEOUT_BUFFER_PERCENT = 0.20  # 20% buffer

    def __init__(self, pattern_store: Optional[PatternStore] = None):
        """
        Initialize the timeout pattern learner.

        Args:
            pattern_store: Pattern store for persistence. Creates default if None.
        """
        self.pattern_store = pattern_store or PatternStore()

        # In-memory tracking: key is "page_type:action"
        self.timeout_results: Dict[str, List[TimeoutResult]] = defaultdict(list)

        # Learned patterns cache
        self.learned_patterns: Dict[str, TimeoutPattern] = {}

        # Load persisted patterns
        self._load_persisted_patterns()

    def _get_key(self, page_type: str, action: str) -> str:
        """Generate storage key from page type and action."""
        return f"{page_type}:{action}"

    def _load_persisted_patterns(self) -> None:
        """Load previously learned timeout patterns."""
        for key in self.pattern_store.list_keys("timeouts"):
            data = self.pattern_store.retrieve("timeouts", key)
            if data and "results" in data:
                # Load results for continued learning
                for result_data in data["results"]:
                    self.timeout_results[key].append(TimeoutResult(
                        timeout_ms=result_data.get("timeout", 5000),
                        duration_ms=result_data.get("duration", 0),
                        success=result_data.get("success", False),
                        triggered=result_data.get("timeout_triggered", False),
                    ))

                # Rebuild pattern from results
                self._update_learned_pattern(key)

    def record_timeout_result(
        self,
        page_type: str,
        action: str,
        timeout_ms: int,
        duration_ms: int,
        success: bool,
        triggered: bool,
    ) -> None:
        """
        Record a timeout result for learning.

        Args:
            page_type: The classified page type
            action: The action performed (e.g., "click", "fill", "navigate")
            timeout_ms: The timeout that was configured
            duration_ms: Actual duration of the operation
            success: Whether the operation succeeded
            triggered: Whether the timeout was triggered (operation timed out)
        """
        key = self._get_key(page_type, action)

        result = TimeoutResult(
            timeout_ms=timeout_ms,
            duration_ms=duration_ms,
            success=success,
            triggered=triggered,
        )

        results = self.timeout_results[key]
        results.append(result)

        # Bound memory usage
        if len(results) > self.MAX_RESULTS_PER_KEY:
            self.timeout_results[key] = results[-self.MAX_RESULTS_PER_KEY:]

        # Persist periodically
        if len(self.timeout_results[key]) % 50 == 0:
            self._persist_results(key)

        # Update pattern if enough samples
        if len(self.timeout_results[key]) >= self.MIN_SAMPLES_FOR_PATTERN:
            if (len(self.timeout_results[key]) == self.MIN_SAMPLES_FOR_PATTERN or
                len(self.timeout_results[key]) % 20 == 0):
                self._update_learned_pattern(key)

    def _persist_results(self, key: str) -> None:
        """
        Persist timeout results to storage.

        Args:
            key: The page_type:action key
        """
        results = self.timeout_results.get(key, [])
        # Keep only last MAX_RESULTS_PER_KEY
        results_to_store = results[-self.MAX_RESULTS_PER_KEY:]

        self.pattern_store.store("timeouts", key, {
            "results": [
                {
                    "timeout": r.timeout_ms,
                    "duration": r.duration_ms,
                    "success": r.success,
                    "timeout_triggered": r.triggered,
                }
                for r in results_to_store
            ]
        })

    def _update_learned_pattern(self, key: str) -> None:
        """
        Update learned pattern from results.

        Args:
            key: The page_type:action key
        """
        results = self.timeout_results.get(key, [])
        if not results:
            return

        # Parse key
        parts = key.split(":", 1)
        page_type = parts[0]
        action = parts[1] if len(parts) > 1 else "unknown"

        # Calculate statistics
        successful_durations = [r.duration_ms for r in results if r.success]
        timeout_triggered_count = sum(1 for r in results if r.triggered)
        success_count = sum(1 for r in results if r.success)

        if not successful_durations:
            return

        # Calculate 95th percentile of successful durations
        sorted_durations = sorted(successful_durations)
        p95_index = int(len(sorted_durations) * 0.95)
        p95_duration = sorted_durations[min(p95_index, len(sorted_durations) - 1)]

        # Calculate optimal timeout: p95 + buffer
        optimal_timeout = int(p95_duration * (1 + self.TIMEOUT_BUFFER_PERCENT))

        # Apply action-specific caps
        if action in self.ACTION_TIMEOUT_ACTIONS:
            optimal_timeout = min(optimal_timeout, 15000)  # Max 15s for actions
        else:
            optimal_timeout = min(optimal_timeout, 120000)  # Max 2min for navigation

        # Ensure minimum timeout
        optimal_timeout = max(optimal_timeout, 1000)  # At least 1 second

        success_rate = success_count / max(len(results), 1)

        pattern = TimeoutPattern(
            page_type=page_type,
            action=action,
            optimal_timeout_ms=optimal_timeout,
            p95_duration_ms=p95_duration,
            success_rate=success_rate,
            sample_count=len(results),
            timeout_triggered_count=timeout_triggered_count,
        )

        self.learned_patterns[key] = pattern

    def get_optimal_timeouts(
        self,
        page_type: str,
        action: str,
    ) -> Dict[str, int]:
        """
        Get learned optimal timeouts for a page type and action.

        Args:
            page_type: The page type
            action: The action to perform

        Returns:
            Dictionary with timeout values (keys: "action" or "navigation")
        """
        key = self._get_key(page_type, action)

        # Default timeouts
        if action in self.ACTION_TIMEOUT_ACTIONS:
            defaults = {"action": self.DEFAULT_TIMEOUTS["action"]}
        else:
            defaults = {"navigation": self.DEFAULT_TIMEOUTS["navigation"]}

        # Check if we have enough data
        results = self.timeout_results.get(key, [])
        if len(results) < self.MIN_SAMPLES_FOR_PATTERN:
            return defaults

        # Get pattern
        pattern = self.learned_patterns.get(key)
        if not pattern:
            self._update_learned_pattern(key)
            pattern = self.learned_patterns.get(key)

        if not pattern:
            return defaults

        # Return appropriate timeout type
        if action in self.ACTION_TIMEOUT_ACTIONS:
            return {"action": pattern.optimal_timeout_ms}
        else:
            return {"navigation": pattern.optimal_timeout_ms}

    def get_pattern(self, page_type: str, action: str) -> Optional[TimeoutPattern]:
        """
        Get the full learned pattern for a page type and action.

        Args:
            page_type: The page type
            action: The action

        Returns:
            TimeoutPattern if available, None otherwise
        """
        key = self._get_key(page_type, action)
        return self.learned_patterns.get(key)

    def get_all_patterns(self) -> Dict[str, TimeoutPattern]:
        """
        Get all learned timeout patterns.

        Returns:
            Dictionary mapping key to pattern
        """
        return dict(self.learned_patterns)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about timeout learning.

        Returns:
            Dictionary with learning statistics
        """
        stats = {
            "learned_patterns": len(self.learned_patterns),
            "total_results": sum(len(r) for r in self.timeout_results.values()),
            "patterns": {},
        }

        for key in set(list(self.timeout_results.keys()) +
                      list(self.learned_patterns.keys())):
            results = self.timeout_results.get(key, [])
            pattern = self.learned_patterns.get(key)

            successful = [r.duration_ms for r in results if r.success]

            stats["patterns"][key] = {
                "samples": len(results),
                "success_count": sum(1 for r in results if r.success),
                "timeout_triggered_count": sum(1 for r in results if r.triggered),
                "avg_duration_ms": statistics.mean(successful) if successful else None,
                "has_learned_pattern": pattern is not None,
                "optimal_timeout_ms": pattern.optimal_timeout_ms if pattern else None,
            }

        return stats

    def reset_learning(self, page_type: Optional[str] = None, action: Optional[str] = None) -> None:
        """
        Reset learned patterns.

        Args:
            page_type: Specific page type to reset (None for all)
            action: Specific action to reset (None for all actions of page_type)
        """
        if page_type and action:
            key = self._get_key(page_type, action)
            self.timeout_results.pop(key, None)
            self.learned_patterns.pop(key, None)
            self.pattern_store.delete("timeouts", key)
        elif page_type:
            # Remove all actions for this page type
            keys_to_remove = [
                k for k in list(self.timeout_results.keys())
                if k.startswith(f"{page_type}:")
            ]
            for key in keys_to_remove:
                self.timeout_results.pop(key, None)
                self.learned_patterns.pop(key, None)
                self.pattern_store.delete("timeouts", key)
        else:
            # Reset all
            self.timeout_results.clear()
            self.learned_patterns.clear()
            for key in self.pattern_store.list_keys("timeouts"):
                self.pattern_store.delete("timeouts", key)

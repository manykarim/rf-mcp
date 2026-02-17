"""Batch Execution Domain Entities.

Entities have identity and mutable state. Each entity is identified
by its position within a batch execution (step index).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .value_objects import StepStatus, StepTimeout


@dataclass
class BatchStep:
    """A single step in a batch execution.

    Identity is the composite of (batch, index). Each step describes
    a keyword invocation with positional arguments, an optional
    human-readable label, and an optional per-step timeout.

    Attributes:
        index: Zero-based position in the batch
        keyword: RF keyword name to execute
        args: Positional arguments for the keyword
        label: Optional human-readable label for diagnostics
        timeout: Optional per-step timeout in RF duration format
    """
    __test__ = False  # Suppress pytest collection

    index: int
    keyword: str
    args: List[str]
    label: Optional[str] = None
    timeout: Optional[StepTimeout] = None
    assign_to: Optional[str] = None

    @property
    def display_name(self) -> str:
        """Human-readable step name for logs and summaries."""
        if self.label:
            return self.label
        preview = ", ".join(self.args[:2])
        if len(self.args) > 2:
            preview += "..."
        return f"{self.keyword}({preview})"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for MCP responses."""
        d: Dict[str, Any] = {
            "index": self.index,
            "keyword": self.keyword,
            "args": list(self.args),
        }
        if self.label:
            d["label"] = self.label
        if self.timeout:
            d["timeout"] = self.timeout.rf_format
        if self.assign_to:
            d["assign_to"] = self.assign_to
        return d


@dataclass
class StepResult:
    """Result of executing a single batch step.

    Captures the outcome (status), return value, timing, and any
    error message produced during execution.

    Attributes:
        index: Zero-based step position matching the BatchStep
        keyword: The keyword that was executed
        args_resolved: Arguments after ``${STEP_N}`` resolution
        status: Outcome status of the step
        return_value: Value returned by the keyword (if any)
        time_ms: Wall-clock execution time in milliseconds
        error: Error message if the step failed
        label: Optional label copied from the BatchStep
    """
    __test__ = False  # Suppress pytest collection

    index: int
    keyword: str
    args_resolved: List[str]
    status: StepStatus
    return_value: Any = None
    time_ms: int = 0
    error: Optional[str] = None
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for MCP responses."""
        d: Dict[str, Any] = {
            "index": self.index,
            "keyword": self.keyword,
            "args_resolved": list(self.args_resolved),
            "status": self.status.value,
            "time_ms": self.time_ms,
        }
        if self.label:
            d["label"] = self.label
        if self.return_value is not None:
            d["return_value"] = self.return_value
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class RecoveryAttempt:
    """Record of a single recovery attempt for a failed step.

    Captures the strategy used, which recovery tier it belongs to,
    a description of what was attempted, and the outcome.

    Attributes:
        attempt_number: 1-based attempt counter
        strategy: Name of the recovery strategy (e.g., "retry", "refresh_page")
        tier: Recovery tier (1=retry, 2=DOM-aware, 3=AI-guided)
        action_description: Human-readable description of the recovery action
        result: Outcome string ("PASS" or "FAIL")
        time_ms: Wall-clock time for this recovery attempt in milliseconds
    """
    __test__ = False  # Suppress pytest collection

    attempt_number: int
    strategy: str
    tier: int
    action_description: str
    result: str  # "PASS" or "FAIL"
    time_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for MCP responses."""
        return {
            "attempt": self.attempt_number,
            "strategy": self.strategy,
            "tier": self.tier,
            "action": self.action_description,
            "result": self.result,
            "time_ms": self.time_ms,
        }


@dataclass
class FailureDetail:
    """Diagnostic context captured when a batch step fails.

    Collects environment state (screenshot, page source, URL) at
    the moment of failure, plus the log of recovery attempts if any.

    Attributes:
        step_index: Index of the step that failed
        error: Error message from the failed keyword execution
        screenshot_base64: Base64-encoded screenshot at failure time
        page_source_snippet: Truncated HTML source around the failure
        current_url: Browser URL at failure time
        page_title: Page title at failure time
        recovery_log: Ordered list of recovery attempts made
    """
    __test__ = False  # Suppress pytest collection

    step_index: int
    error: str
    screenshot_base64: Optional[str] = None
    page_source_snippet: Optional[str] = None
    current_url: Optional[str] = None
    page_title: Optional[str] = None
    recovery_log: List[RecoveryAttempt] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict for MCP responses."""
        d: Dict[str, Any] = {
            "step_index": self.step_index,
            "error": self.error,
        }
        if self.screenshot_base64:
            d["screenshot_base64"] = self.screenshot_base64
        if self.page_source_snippet:
            d["page_source_snippet"] = self.page_source_snippet
        if self.current_url:
            d["current_url"] = self.current_url
        if self.page_title:
            d["page_title"] = self.page_title
        if self.recovery_log:
            d["recovery_log"] = [a.to_dict() for a in self.recovery_log]
        return d

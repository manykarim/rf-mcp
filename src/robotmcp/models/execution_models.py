"""Execution-related data models."""

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionStep:
    """Represents a single execution step."""
    step_id: str
    keyword: str
    arguments: List[str]
    status: str = "pending"  # pending, running, pass, fail
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Any] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    
    # Variable assignment tracking for test suite generation
    assigned_variables: List[str] = field(default_factory=list)  # Variables assigned from this step
    assignment_type: Optional[str] = None  # "single", "multiple", "none"
    
    def mark_running(self) -> None:
        """Mark the step as currently running."""
        self.status = "running"
        self.start_time = datetime.now()
    
    def mark_success(self, result: Any = None) -> None:
        """Mark the step as successfully completed."""
        self.status = "pass"
        self.end_time = datetime.now()
        self.result = result
    
    def mark_failure(self, error: str) -> None:
        """Mark the step as failed."""
        self.status = "fail"
        self.end_time = datetime.now()
        self.error = error
    
    @property
    def execution_time(self) -> float:
        """Calculate execution time in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def is_successful(self) -> bool:
        """Check if the step completed successfully."""
        return self.status == "pass"
    
    @property
    def is_failed(self) -> bool:
        """Check if the step failed."""
        return self.status == "fail"
    
    @property
    def is_completed(self) -> bool:
        """Check if the step is completed (either success or failure)."""
        return self.status in ["pass", "fail"]


@dataclass
class TestInfo:
    """Represents a single test within a multi-test session (ADR-005)."""

    __test__ = False  # Not a pytest test class

    name: str
    status: str = "not_run"  # not_run → running → pass/fail
    documentation: str = ""
    tags: List[str] = field(default_factory=list)
    setup: Optional[Dict[str, Any]] = None  # {"keyword": ..., "arguments": [...]}
    teardown: Optional[Dict[str, Any]] = None  # {"keyword": ..., "arguments": [...]}
    steps: List[ExecutionStep] = field(default_factory=list)
    flow_blocks: List[Dict[str, Any]] = field(default_factory=list)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class TestRegistry:
    """Manages multiple tests within a session (ADR-005).

    Provides an ordered registry of tests with lifecycle management.
    In legacy mode (no start_test called), is_multi_test_mode() returns False
    and the session operates exactly as before.
    """

    __test__ = False  # Not a pytest test class

    tests: OrderedDict = field(default_factory=OrderedDict)
    current_test_name: Optional[str] = None
    _multi_test_activated: bool = False

    def start_test(
        self,
        name: str,
        documentation: str = "",
        tags: Optional[List[str]] = None,
        setup: Optional[Dict[str, Any]] = None,
        teardown: Optional[Dict[str, Any]] = None,
    ) -> TestInfo:
        """Start a new test. Auto-ends current test if one is active."""
        if self.current_test_name and self.current_test_name in self.tests:
            current = self.tests[self.current_test_name]
            if current.status == "running":
                current.status = "pass"
                current.ended_at = datetime.now()

        test = TestInfo(
            name=name,
            status="running",
            documentation=documentation,
            tags=tags or [],
            setup=setup,
            teardown=teardown,
            started_at=datetime.now(),
        )
        self.tests[name] = test
        self.current_test_name = name
        self._multi_test_activated = True
        return test

    def end_test(self, status: str = "pass", message: str = "") -> Optional[TestInfo]:
        """End the current test with the given status."""
        if not self.current_test_name or self.current_test_name not in self.tests:
            return None
        test = self.tests[self.current_test_name]
        test.status = status
        test.error_message = message if message else None
        test.ended_at = datetime.now()
        self.current_test_name = None
        return test

    def get_current_test(self) -> Optional[TestInfo]:
        """Get the currently active test, or None."""
        if self.current_test_name and self.current_test_name in self.tests:
            return self.tests[self.current_test_name]
        return None

    def is_multi_test_mode(self) -> bool:
        """True once start_test() has been called at least once."""
        return self._multi_test_activated

    def all_steps_flat(self) -> List[ExecutionStep]:
        """Return all steps from all tests in insertion order."""
        result = []
        for test in self.tests.values():
            result.extend(test.steps)
        return result

    def all_flow_blocks_flat(self) -> List[Dict[str, Any]]:
        """Return all flow blocks from all tests in insertion order."""
        result = []
        for test in self.tests.values():
            result.extend(test.flow_blocks)
        return result
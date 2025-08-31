"""Execution-related data models."""

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
"""Recording storage and management.

Stores executed keywords for later export as test suites.
Handles both regular RF keywords and AI-translated keywords.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StepType(Enum):
    """Type of recorded step."""

    REGULAR = "regular"  # Regular RF keyword
    AI = "ai"  # AI keyword (Do, Check, Ask)
    SETUP = "setup"
    TEARDOWN = "teardown"


@dataclass
class RecordedStep:
    """A recorded keyword execution step."""

    keyword: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    library: Optional[str] = None
    step_type: StepType = StepType.REGULAR
    timestamp: datetime = field(default_factory=datetime.now)
    result: Any = None
    success: bool = True
    error: Optional[str] = None

    # AI-specific fields
    prompt: Optional[str] = None  # Original AI prompt
    ai_keyword: Optional[str] = None  # Do, Check, or Ask
    executed_keywords: List["RecordedStep"] = field(default_factory=list)
    attempts: int = 1
    assigned_variable: Optional[str] = None  # Variable to assign result to (for Ask)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = {
            "keyword": self.keyword,
            "args": self.args,
            "kwargs": self.kwargs,
            "library": self.library,
            "type": self.step_type.value,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
        }

        if self.step_type == StepType.AI:
            data["prompt"] = self.prompt
            data["ai_keyword"] = self.ai_keyword
            data["executed_keywords"] = [kw.to_dict() for kw in self.executed_keywords]
            data["attempts"] = self.attempts
            if self.assigned_variable:
                data["assigned_variable"] = self.assigned_variable

        if self.error:
            data["error"] = self.error

        return data


class ExecutionRecorder:
    """Records keyword executions for test suite generation."""

    def __init__(self):
        """Initialize the recorder."""
        self._steps: List[RecordedStep] = []
        self._is_recording: bool = False
        self._current_ai_step: Optional[RecordedStep] = None
        self._test_metadata: Dict[str, Any] = {}
        self._libraries_used: set = set()
        self._pending_ai_assignment: Optional[List[str]] = None

    def set_pending_assignment(self, variables: List[str]) -> None:
        """Set pending variable assignment for the next AI step.

        Called by the listener when it detects an Ask/Do/Check keyword
        with variable assignment (e.g., ${var}= Ask ...).

        Args:
            variables: List of variable names being assigned
        """
        self._pending_ai_assignment = variables

    def get_and_clear_pending_assignment(self) -> Optional[List[str]]:
        """Get and clear the pending variable assignment.

        Returns:
            List of variable names or None
        """
        assignment = self._pending_ai_assignment
        self._pending_ai_assignment = None
        return assignment

    @property
    def is_recording(self) -> bool:
        """Check if recording is active."""
        return self._is_recording

    def start_recording(self) -> None:
        """Start recording keyword executions."""
        self._is_recording = True
        self._steps = []
        self._libraries_used = set()
        self._test_metadata = {
            "started_at": datetime.now().isoformat(),
        }
        logger.info("Recording started")

    def stop_recording(self) -> None:
        """Stop recording keyword executions."""
        self._is_recording = False
        self._test_metadata["stopped_at"] = datetime.now().isoformat()
        logger.info(f"Recording stopped. {len(self._steps)} steps recorded.")

    def record_step(
        self,
        keyword: str,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None,
        library: str = None,
        step_type: StepType = StepType.REGULAR,
        result: Any = None,
        success: bool = True,
        error: str = None,
    ) -> RecordedStep:
        """Record a keyword execution step.

        Args:
            keyword: Keyword name
            args: Positional arguments
            kwargs: Named arguments
            library: Library name
            step_type: Type of step
            result: Execution result
            success: Whether execution succeeded
            error: Error message if failed

        Returns:
            The recorded step
        """
        if not self._is_recording:
            return None

        step = RecordedStep(
            keyword=keyword,
            args=args or [],
            kwargs=kwargs or {},
            library=library,
            step_type=step_type,
            result=result,
            success=success,
            error=error,
        )

        # Track library usage
        if library:
            self._libraries_used.add(library)

        # If we're inside an AI step, add as sub-step
        if self._current_ai_step is not None:
            self._current_ai_step.executed_keywords.append(step)
        else:
            self._steps.append(step)

        logger.debug(f"Recorded step: {keyword} from {library}")
        return step

    def start_ai_step(
        self,
        prompt: str,
        ai_keyword: str,  # Do, Check, Ask
        assigned_variable: str = None,
    ) -> RecordedStep:
        """Start recording an AI keyword execution.

        Args:
            prompt: The natural language prompt
            ai_keyword: The AI keyword type (Do, Check, Ask)
            assigned_variable: Variable name to assign result to (for Ask)

        Returns:
            The AI step being recorded
        """
        if not self._is_recording:
            return None

        self._current_ai_step = RecordedStep(
            keyword=ai_keyword,
            args=[prompt],
            step_type=StepType.AI,
            prompt=prompt,
            ai_keyword=ai_keyword,
            executed_keywords=[],
            assigned_variable=assigned_variable,
        )

        return self._current_ai_step

    def end_ai_step(
        self,
        success: bool = True,
        result: Any = None,
        error: str = None,
        attempts: int = 1,
    ) -> Optional[RecordedStep]:
        """End the current AI keyword recording.

        Args:
            success: Whether the AI step succeeded
            result: Result value (for Ask keyword)
            error: Error message if failed
            attempts: Number of retry attempts

        Returns:
            The completed AI step or None
        """
        if self._current_ai_step is None:
            return None

        step = self._current_ai_step
        step.success = success
        step.result = result
        step.error = error
        step.attempts = attempts

        # Only record successful AI steps
        if success:
            self._steps.append(step)

        self._current_ai_step = None
        return step

    def get_steps(self) -> List[RecordedStep]:
        """Get all recorded steps.

        Returns:
            List of recorded steps
        """
        return self._steps.copy()

    def get_libraries_used(self) -> List[str]:
        """Get list of libraries used in recording.

        Returns:
            List of library names
        """
        return list(self._libraries_used)

    def get_metadata(self) -> Dict[str, Any]:
        """Get recording metadata.

        Returns:
            Recording metadata dictionary
        """
        return {
            **self._test_metadata,
            "step_count": len(self._steps),
            "libraries": self.get_libraries_used(),
        }

    def clear(self) -> None:
        """Clear all recorded steps."""
        self._steps = []
        self._current_ai_step = None
        self._libraries_used = set()
        self._test_metadata = {}
        self._pending_ai_assignment = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert recording to dictionary.

        Returns:
            Dictionary representation of the recording
        """
        return {
            "metadata": self.get_metadata(),
            "steps": [step.to_dict() for step in self._steps],
        }


# Singleton instance for global access
_recorder_instance: Optional[ExecutionRecorder] = None


def get_recorder() -> ExecutionRecorder:
    """Get the global recorder instance.

    Returns:
        ExecutionRecorder instance
    """
    global _recorder_instance
    if _recorder_instance is None:
        _recorder_instance = ExecutionRecorder()
    return _recorder_instance

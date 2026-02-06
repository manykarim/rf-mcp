"""Robot Framework Listener for recording ALL keyword executions.

Implements ROBOT_LISTENER_API_VERSION = 3 to capture every keyword call
with full arguments during test execution.
"""

import logging
from typing import Any, Dict, Optional

from robotmcp.lib.recorder import ExecutionRecorder, RecordedStep, StepType, get_recorder

logger = logging.getLogger(__name__)


class RecordingListener:
    """Robot Framework listener that captures all keyword executions.

    This listener hooks into Robot Framework's execution lifecycle to
    record every keyword call, including those executed by AI keywords.

    Attributes:
        ROBOT_LISTENER_API_VERSION: Listener API version (3)
    """

    ROBOT_LISTENER_API_VERSION = 3

    def __init__(self, recorder: ExecutionRecorder = None):
        """Initialize the recording listener.

        Args:
            recorder: ExecutionRecorder instance to use. If None, uses global.
        """
        self.recorder = recorder or get_recorder()
        self._keyword_stack: list = []
        self._current_test: Optional[str] = None
        self._current_suite: Optional[str] = None

        # Keywords to exclude from recording
        self._excluded_keywords = {
            "start recording",
            "stop recording",
            "export test suite",
            "get recorded steps",
            # AI wrapper keywords - these are handled by start_ai_step/end_ai_step
            # The sub-keywords executed by the AI are recorded via record_step
            "do",
            "check",
            "ask",
            # Internal RF keywords
            "log",
            "comment",
            "no operation",
        }

        # Libraries to exclude (optional)
        self._excluded_libraries: set = set()

    def start_suite(self, data, result):
        """Called when a test suite starts.

        Args:
            data: Suite data object
            result: Suite result object
        """
        self._current_suite = data.name
        logger.debug(f"Suite started: {self._current_suite}")

    def end_suite(self, data, result):
        """Called when a test suite ends.

        Args:
            data: Suite data object
            result: Suite result object
        """
        logger.debug(f"Suite ended: {self._current_suite}")
        self._current_suite = None

    def start_test(self, data, result):
        """Called when a test case starts.

        Args:
            data: Test data object
            result: Test result object
        """
        self._current_test = data.name
        logger.debug(f"Test started: {self._current_test}")

    def end_test(self, data, result):
        """Called when a test case ends.

        Args:
            data: Test data object
            result: Test result object
        """
        logger.debug(f"Test ended: {self._current_test}")
        self._current_test = None

    def start_keyword(self, data, result):
        """Called when a keyword starts.

        Args:
            data: Keyword data object with name, args, etc.
            result: Keyword result object
        """
        if not self.recorder.is_recording:
            return

        keyword_name = data.name
        keyword_lower = keyword_name.lower()

        # For AI keywords (ask, do, check), capture variable assignment before skipping
        # This allows AILibrary to detect ${var}= Ask ... syntax
        if keyword_lower in ("ask", "do", "check"):
            if hasattr(data, "assign") and data.assign:
                # data.assign is a tuple of variable names like ('${product_name}',)
                self.recorder.set_pending_assignment(list(data.assign))
            else:
                # Clear any stale pending assignment
                self.recorder.set_pending_assignment(None)

        # Skip excluded keywords
        if keyword_lower in self._excluded_keywords:
            self._keyword_stack.append(None)  # Placeholder for stack balance
            return

        # Get library name if available
        library = getattr(data, "libname", None) or self._infer_library(keyword_name)

        # Skip excluded libraries
        if library and library.lower() in self._excluded_libraries:
            self._keyword_stack.append(None)
            return

        # Parse arguments
        args = list(data.args) if data.args else []
        kwargs = {}

        # Separate kwargs from args (format: name=value)
        positional_args = []
        for arg in args:
            if isinstance(arg, str) and "=" in arg and not arg.startswith("$"):
                key, value = arg.split("=", 1)
                kwargs[key] = value
            else:
                positional_args.append(arg)

        # Determine step type
        step_type = self._determine_step_type(data)

        # Create and push step info
        step_info = {
            "keyword": keyword_name,
            "args": positional_args,
            "kwargs": kwargs,
            "library": library,
            "step_type": step_type,
        }
        self._keyword_stack.append(step_info)

        logger.debug(f"Keyword started: {keyword_name} ({library})")

    def end_keyword(self, data, result):
        """Called when a keyword ends.

        Args:
            data: Keyword data object
            result: Keyword result object with status
        """
        if not self.recorder.is_recording:
            return

        # Pop the corresponding step info
        if not self._keyword_stack:
            return

        step_info = self._keyword_stack.pop()
        if step_info is None:  # Was an excluded keyword
            return

        # Record the step with result
        success = result.passed if hasattr(result, "passed") else True
        error = str(result.message) if hasattr(result, "message") and result.message else None

        self.recorder.record_step(
            keyword=step_info["keyword"],
            args=step_info["args"],
            kwargs=step_info["kwargs"],
            library=step_info["library"],
            step_type=step_info["step_type"],
            result=None,  # Could capture return value if needed
            success=success,
            error=error if not success else None,
        )

        logger.debug(f"Keyword ended: {step_info['keyword']} (success={success})")

    def log_message(self, message):
        """Called when a log message is generated.

        Can be used to capture return values logged by keywords.

        Args:
            message: Log message object
        """
        # Could capture return values here if needed
        pass

    def _infer_library(self, keyword_name: str) -> Optional[str]:
        """Infer library name from keyword name pattern.

        Args:
            keyword_name: Keyword name, possibly with library prefix

        Returns:
            Inferred library name or None
        """
        if "." in keyword_name:
            # Format: Library.Keyword
            return keyword_name.split(".")[0]

        # Try to detect common patterns
        keyword_lower = keyword_name.lower()

        # Browser Library patterns
        browser_keywords = {
            "new browser", "new context", "new page", "go to",
            "click", "fill text", "get text", "take screenshot",
            "wait for elements state",
        }
        if keyword_lower in browser_keywords:
            return "Browser"

        # SeleniumLibrary patterns
        selenium_keywords = {
            "open browser", "close browser", "go to", "click element",
            "input text", "get text", "wait until element is visible",
            "get location", "get title",
        }
        if keyword_lower in selenium_keywords:
            return "SeleniumLibrary"

        # Built-in patterns
        builtin_keywords = {
            "log", "should be equal", "should contain", "set variable",
            "run keyword if", "run keywords", "sleep",
        }
        if keyword_lower in builtin_keywords:
            return "BuiltIn"

        return None

    def _determine_step_type(self, data) -> StepType:
        """Determine the type of step from keyword data.

        Args:
            data: Keyword data object

        Returns:
            StepType enum value
        """
        # Check for setup/teardown
        if hasattr(data, "type"):
            kw_type = str(data.type).lower()
            if "setup" in kw_type:
                return StepType.SETUP
            elif "teardown" in kw_type:
                return StepType.TEARDOWN

        return StepType.REGULAR

    def exclude_library(self, library_name: str) -> None:
        """Add a library to the exclusion list.

        Args:
            library_name: Library name to exclude
        """
        self._excluded_libraries.add(library_name.lower())

    def exclude_keyword(self, keyword_name: str) -> None:
        """Add a keyword to the exclusion list.

        Args:
            keyword_name: Keyword name to exclude
        """
        self._excluded_keywords.add(keyword_name.lower())

    def include_library(self, library_name: str) -> None:
        """Remove a library from the exclusion list.

        Args:
            library_name: Library name to include
        """
        self._excluded_libraries.discard(library_name.lower())

    def include_keyword(self, keyword_name: str) -> None:
        """Remove a keyword from the exclusion list.

        Args:
            keyword_name: Keyword name to include
        """
        self._excluded_keywords.discard(keyword_name.lower())

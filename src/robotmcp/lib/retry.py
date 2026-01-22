"""Intelligent retry mechanism with error feedback.

Implements retry logic that feeds error context back to the AI
for intelligent correction and re-execution.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from robot.api import logger as rf_logger

logger = logging.getLogger(__name__)


@dataclass
class RetryContext:
    """Context information for retry attempts."""

    original_prompt: str
    attempted_keyword: str
    attempted_args: List[str]
    error_message: str
    attempt_number: int
    max_retries: int
    page_state: Dict[str, Any] = field(default_factory=dict)
    available_keywords: List[str] = field(default_factory=list)
    previous_attempts: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for AI prompt."""
        return {
            "original_prompt": self.original_prompt,
            "attempted_keyword": self.attempted_keyword,
            "attempted_args": self.attempted_args,
            "error_message": self.error_message,
            "attempt_number": self.attempt_number,
            "max_retries": self.max_retries,
            "page_state": self.page_state,
            "available_keywords": self.available_keywords[:50],  # Limit for token efficiency
            "previous_attempts": self.previous_attempts,
        }


@dataclass
class ExecutionResult:
    """Result of a keyword execution attempt."""

    success: bool
    result: Any = None
    error: Optional[str] = None
    keyword: str = ""
    args: List[str] = field(default_factory=list)
    attempt: int = 1


class RetryHandler:
    """Handles intelligent retry logic with AI feedback."""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        exponential_backoff: bool = False,
    ):
        """Initialize the retry handler.

        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
            exponential_backoff: Whether to use exponential backoff
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_backoff = exponential_backoff

    async def execute_with_retry(
        self,
        prompt: str,
        execute_fn: Callable[..., ExecutionResult],
        generate_correction_fn: Callable[[RetryContext], str],
        get_page_state_fn: Callable[[], Dict[str, Any]],
        get_keywords_fn: Callable[[], List[str]],
    ) -> ExecutionResult:
        """Execute with intelligent retry on failure.

        Args:
            prompt: Original natural language prompt
            execute_fn: Function to execute the generated keyword
            generate_correction_fn: Function to generate corrected keyword from AI
            get_page_state_fn: Function to get current page state
            get_keywords_fn: Function to get available keywords

        Returns:
            ExecutionResult with success status and result/error
        """
        previous_attempts: List[Dict[str, Any]] = []
        last_result: Optional[ExecutionResult] = None

        for attempt in range(1, self.max_retries + 1):
            # First attempt or retry with correction
            if attempt == 1:
                # Initial execution - get page state for context
                initial_page_state = await self._run_async(get_page_state_fn)
                initial_keywords = await self._run_async(get_keywords_fn)
                retry_ctx = RetryContext(
                    original_prompt=prompt,
                    attempted_keyword="",
                    attempted_args=[],
                    error_message="",
                    attempt_number=0,
                    max_retries=self.max_retries,
                    page_state=initial_page_state,
                    available_keywords=initial_keywords,
                    previous_attempts=[],
                )
                keyword_call = await self._run_async(generate_correction_fn, retry_ctx)
            else:
                # Retry with error context
                retry_ctx = RetryContext(
                    original_prompt=prompt,
                    attempted_keyword=last_result.keyword if last_result else "",
                    attempted_args=last_result.args if last_result else [],
                    error_message=last_result.error if last_result else "",
                    attempt_number=attempt,
                    max_retries=self.max_retries,
                    page_state=await self._run_async(get_page_state_fn),
                    available_keywords=await self._run_async(get_keywords_fn),
                    previous_attempts=previous_attempts,
                )
                keyword_call = await self._run_async(generate_correction_fn, retry_ctx)

            # Log the attempt
            rf_logger.info(f"AILibrary: Attempt {attempt}/{self.max_retries}: {keyword_call}")
            logger.info(f"Executing attempt {attempt}: {keyword_call}")

            # Execute the keyword
            result = await self._run_async(execute_fn, keyword_call)
            result.attempt = attempt
            last_result = result

            if result.success:
                rf_logger.info(f"AILibrary: Success on attempt {attempt}")
                return result

            # Log failure
            rf_logger.warn(f"AILibrary: Attempt {attempt} failed: {result.error}")
            logger.warning(f"Attempt {attempt} failed: {result.error}")

            # Record attempt for context
            previous_attempts.append({
                "attempt": attempt,
                "keyword": result.keyword,
                "args": result.args,
                "error": result.error,
            })

            # Wait before retry (unless last attempt)
            if attempt < self.max_retries:
                delay = self._get_delay(attempt)
                rf_logger.debug(f"AILibrary: Waiting {delay}s before retry")
                await asyncio.sleep(delay)

        # All retries exhausted
        rf_logger.error(f"AILibrary: All {self.max_retries} attempts failed")
        return ExecutionResult(
            success=False,
            error=f"All {self.max_retries} attempts failed. Last error: {last_result.error if last_result else 'Unknown'}",
            keyword=last_result.keyword if last_result else "",
            args=last_result.args if last_result else [],
            attempt=self.max_retries,
        )

    def _get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt.

        Args:
            attempt: Current attempt number (1-based)

        Returns:
            Delay in seconds
        """
        if self.exponential_backoff:
            return self.retry_delay * (2 ** (attempt - 1))
        return self.retry_delay

    async def _run_async(self, fn: Callable, *args, **kwargs) -> Any:
        """Run a function asynchronously.

        If the function is async, await it directly.
        If it's sync, run it in a thread executor.
        """
        if asyncio.iscoroutinefunction(fn):
            return await fn(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))


def build_retry_prompt(context: RetryContext) -> str:
    """Build a prompt for the AI to generate a corrected keyword call.

    Args:
        context: RetryContext with error information

    Returns:
        Formatted prompt string for the AI
    """
    prompt_parts = [
        f"The previous keyword call failed. Please generate a corrected approach.",
        f"\nOriginal request: {context.original_prompt}",
        f"\nFailed keyword: {context.attempted_keyword}",
        f"Arguments: {context.attempted_args}",
        f"Error: {context.error_message}",
        f"\nAttempt: {context.attempt_number}/{context.max_retries}",
    ]

    if context.page_state:
        prompt_parts.append(f"\nCurrent page state: {context.page_state}")

    if context.previous_attempts:
        prompt_parts.append("\nPrevious attempts that failed:")
        for prev in context.previous_attempts:
            prompt_parts.append(f"  - {prev['keyword']} {prev['args']}: {prev['error']}")

    if context.available_keywords:
        # Include relevant keywords (first 20)
        prompt_parts.append(f"\nSome available keywords: {', '.join(context.available_keywords[:20])}")

    prompt_parts.append(
        "\n\nBased on the error and context, generate a SINGLE corrected keyword call. "
        "Respond with ONLY the keyword and arguments in this format:\n"
        "KEYWORD    arg1    arg2    ..."
    )

    return "\n".join(prompt_parts)

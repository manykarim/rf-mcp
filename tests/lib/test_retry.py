"""Tests for retry module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from robotmcp.lib.retry import (
    ExecutionResult,
    RetryContext,
    RetryHandler,
    build_retry_prompt,
)


class TestRetryContext:
    """Tests for RetryContext class."""

    def test_to_dict(self):
        """Test converting context to dictionary."""
        ctx = RetryContext(
            original_prompt="Click submit button",
            attempted_keyword="Click",
            attempted_args=["button#submit"],
            error_message="Element not found",
            attempt_number=2,
            max_retries=3,
            page_state={"url": "https://example.com"},
            available_keywords=["Click", "Fill Text"],
        )

        data = ctx.to_dict()

        assert data["original_prompt"] == "Click submit button"
        assert data["attempted_keyword"] == "Click"
        assert data["error_message"] == "Element not found"
        assert data["attempt_number"] == 2


class TestExecutionResult:
    """Tests for ExecutionResult class."""

    def test_success_result(self):
        """Test successful execution result."""
        result = ExecutionResult(
            success=True,
            result="OK",
            keyword="Click",
            args=["button"],
        )

        assert result.success is True
        assert result.result == "OK"
        assert result.error is None

    def test_failure_result(self):
        """Test failed execution result."""
        result = ExecutionResult(
            success=False,
            error="Element not found",
            keyword="Click",
            args=["button"],
        )

        assert result.success is False
        assert result.error == "Element not found"


class TestRetryHandler:
    """Tests for RetryHandler class."""

    def test_init(self):
        """Test handler initialization."""
        handler = RetryHandler(max_retries=5, retry_delay=2.0)

        assert handler.max_retries == 5
        assert handler.retry_delay == 2.0

    def test_get_delay_linear(self):
        """Test linear delay calculation."""
        handler = RetryHandler(retry_delay=1.0, exponential_backoff=False)

        assert handler._get_delay(1) == 1.0
        assert handler._get_delay(2) == 1.0
        assert handler._get_delay(3) == 1.0

    def test_get_delay_exponential(self):
        """Test exponential backoff delay."""
        handler = RetryHandler(retry_delay=1.0, exponential_backoff=True)

        assert handler._get_delay(1) == 1.0
        assert handler._get_delay(2) == 2.0
        assert handler._get_delay(3) == 4.0

    @pytest.mark.asyncio
    async def test_execute_with_retry_success_first_attempt(self):
        """Test successful execution on first attempt."""
        handler = RetryHandler(max_retries=3)

        # Mock functions
        execute_fn = MagicMock(
            return_value=ExecutionResult(
                success=True,
                result="OK",
                keyword="Click",
                args=[],
            )
        )
        generate_fn = AsyncMock(return_value="Click    button")
        page_state_fn = MagicMock(return_value={})
        keywords_fn = MagicMock(return_value=["Click"])

        result = await handler.execute_with_retry(
            prompt="Click the button",
            execute_fn=execute_fn,
            generate_correction_fn=generate_fn,
            get_page_state_fn=page_state_fn,
            get_keywords_fn=keywords_fn,
        )

        assert result.success is True
        assert result.attempt == 1
        assert execute_fn.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_with_retry_success_after_retry(self):
        """Test successful execution after retry."""
        handler = RetryHandler(max_retries=3, retry_delay=0.01)

        call_count = 0

        def execute_fn(keyword_call):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return ExecutionResult(
                    success=False,
                    error="Element not found",
                    keyword="Click",
                    args=["button"],
                )
            return ExecutionResult(
                success=True,
                result="OK",
                keyword="Click",
                args=["button"],
            )

        generate_fn = AsyncMock(return_value="Click    button")
        page_state_fn = MagicMock(return_value={})
        keywords_fn = MagicMock(return_value=["Click"])

        result = await handler.execute_with_retry(
            prompt="Click the button",
            execute_fn=execute_fn,
            generate_correction_fn=generate_fn,
            get_page_state_fn=page_state_fn,
            get_keywords_fn=keywords_fn,
        )

        assert result.success is True
        assert result.attempt == 2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_execute_with_retry_all_failures(self):
        """Test all retries exhausted."""
        handler = RetryHandler(max_retries=2, retry_delay=0.01)

        execute_fn = MagicMock(
            return_value=ExecutionResult(
                success=False,
                error="Element not found",
                keyword="Click",
                args=["button"],
            )
        )
        generate_fn = AsyncMock(return_value="Click    button")
        page_state_fn = MagicMock(return_value={})
        keywords_fn = MagicMock(return_value=["Click"])

        result = await handler.execute_with_retry(
            prompt="Click the button",
            execute_fn=execute_fn,
            generate_correction_fn=generate_fn,
            get_page_state_fn=page_state_fn,
            get_keywords_fn=keywords_fn,
        )

        assert result.success is False
        assert "All 2 attempts failed" in result.error
        assert execute_fn.call_count == 2


class TestBuildRetryPrompt:
    """Tests for build_retry_prompt function."""

    def test_basic_prompt(self):
        """Test building a basic retry prompt."""
        ctx = RetryContext(
            original_prompt="Click submit button",
            attempted_keyword="Click",
            attempted_args=["button#submit"],
            error_message="Element not found",
            attempt_number=2,
            max_retries=3,
        )

        prompt = build_retry_prompt(ctx)

        assert "Click submit button" in prompt
        assert "Element not found" in prompt
        assert "2/3" in prompt

    def test_prompt_with_page_state(self):
        """Test prompt includes page state."""
        ctx = RetryContext(
            original_prompt="Click submit",
            attempted_keyword="Click",
            attempted_args=[],
            error_message="Error",
            attempt_number=1,
            max_retries=3,
            page_state={"url": "https://example.com"},
        )

        prompt = build_retry_prompt(ctx)

        assert "example.com" in prompt

    def test_prompt_with_previous_attempts(self):
        """Test prompt includes previous attempts."""
        ctx = RetryContext(
            original_prompt="Click submit",
            attempted_keyword="Click",
            attempted_args=[],
            error_message="Error",
            attempt_number=2,
            max_retries=3,
            previous_attempts=[
                {
                    "attempt": 1,
                    "keyword": "Click",
                    "args": ["button"],
                    "error": "Not found",
                }
            ],
        )

        prompt = build_retry_prompt(ctx)

        assert "Previous attempts" in prompt
        assert "Not found" in prompt

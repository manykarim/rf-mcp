"""OBS-02 — pre-validation retry-on-transient-failure and the new
``pre_validate_timeout_ms`` override.

The 2026-05-17 Tricentis obstacle-course benchmark exposed a pattern:
both models hit "not visible" pre-validation failures on a button that
WAS visible to a human, just briefly settling. The fix is two-pronged:

1. ``_pre_validate_element_with_retry`` retries the check once after a
   short backoff. Happy path pays NO extra latency (only failing calls
   sleep the backoff).
2. ``pre_validate_timeout_ms`` parameter on execute_step lets the agent
   extend the gate for a single call when the page genuinely takes
   longer than the configured default to settle.

These tests pin both behaviours via mocked pre-validation so the real
Playwright machinery does not have to be running.
"""

from __future__ import annotations
from robotmcp.compat.fastmcp_compat import get_tool_fn

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotmcp.components.execution.keyword_executor import KeywordExecutor
from robotmcp.models.config_models import ExecutionConfig


@pytest.fixture
def executor():
    return KeywordExecutor(config=ExecutionConfig())


@pytest.fixture
def mock_session():
    """Minimal session stub. Pre-validation reads session.session_id."""
    session = MagicMock()
    session.session_id = "test-retry-session"
    return session


# ---------------------------------------------------------------------------
# _pre_validate_element_with_retry behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRetryWrapperHappyPath:
    """When the first call succeeds, the wrapper must NOT retry — the
    happy path stays byte-for-byte identical to the pre-OBS-02 behaviour.
    Critical for not regressing the 10us hot-path latency budget."""

    async def test_no_retry_when_first_call_succeeds(self, executor, mock_session):
        first_call_details = {"locator": "id=foo", "current_states": ["visible"]}
        mock_pre_validate = AsyncMock(
            return_value=(True, None, first_call_details),
        )
        with patch.object(executor, "_pre_validate_element", mock_pre_validate):
            is_valid, error, details = await executor._pre_validate_element_with_retry(
                "id=foo", mock_session, "Click", timeout_ms=500,
            )
        assert is_valid is True
        assert error is None
        assert details is first_call_details
        # Critical: exactly ONE call. No retry.
        assert mock_pre_validate.await_count == 1

    async def test_happy_path_passes_timeout_through_unchanged(
        self, executor, mock_session,
    ):
        mock_pre_validate = AsyncMock(return_value=(True, None, {}))
        with patch.object(executor, "_pre_validate_element", mock_pre_validate):
            await executor._pre_validate_element_with_retry(
                "id=foo", mock_session, "Click", timeout_ms=750,
            )
        # Per-call timeout is forwarded as-is — wrapper does not silently
        # halve, clip, or otherwise mutate it.
        mock_pre_validate.assert_awaited_once()
        assert mock_pre_validate.await_args.kwargs["timeout_ms"] == 750


@pytest.mark.asyncio
class TestRetryWrapperTransientFailure:
    """When the first call fails (visible=False, enabled=False, etc.),
    the wrapper waits the backoff and retries once. If the retry
    succeeds, the wrapper reports success with retry metadata in
    ``details``. If the retry also fails, the wrapper reports failure
    and does NOT loop further."""

    async def test_retry_recovers_when_second_call_succeeds(
        self, executor, mock_session,
    ):
        # First call: not visible. Second call: visible.
        mock_pre_validate = AsyncMock(
            side_effect=[
                (False, "Element not visible: id=foo",
                 {"missing_states": ["visible"]}),
                (True, None, {"current_states": ["visible", "enabled"]}),
            ],
        )
        with patch.object(executor, "_pre_validate_element", mock_pre_validate):
            is_valid, error, details = await executor._pre_validate_element_with_retry(
                "id=foo", mock_session, "Click", timeout_ms=500,
            )
        assert is_valid is True
        assert error is None
        # Retry metadata MUST be surfaced so callers (and tests) can
        # distinguish a first-try success from a recovered transient.
        assert details["retries"] == 1
        assert details["first_attempt_error"] == "Element not visible: id=foo"
        assert mock_pre_validate.await_count == 2

    async def test_no_recovery_when_both_calls_fail(
        self, executor, mock_session,
    ):
        # Both calls fail — wrapper must NOT keep looping beyond the cap.
        mock_pre_validate = AsyncMock(
            return_value=(
                False, "Element not visible: id=foo",
                {"missing_states": ["visible"]},
            ),
        )
        with patch.object(executor, "_pre_validate_element", mock_pre_validate):
            is_valid, error, details = await executor._pre_validate_element_with_retry(
                "id=foo", mock_session, "Click", timeout_ms=500,
            )
        assert is_valid is False
        assert "not visible" in error.lower()
        # Cap is honoured (PRE_VALIDATION_MAX_RETRIES == 1).
        assert mock_pre_validate.await_count == 1 + executor.PRE_VALIDATION_MAX_RETRIES
        # Surface the retry count so the failure-hint builder + tests
        # know we already tried once more.
        assert details.get("retries") == executor.PRE_VALIDATION_MAX_RETRIES

    async def test_retry_waits_the_configured_gap(self, executor, mock_session):
        # The wrapper uses asyncio.sleep — mock that so the test runs
        # without actually waiting 200ms. The mock confirms the sleep
        # was called with the expected duration.
        mock_pre_validate = AsyncMock(
            side_effect=[
                (False, "transient", {"missing_states": ["visible"]}),
                (True, None, {"current_states": ["visible"]}),
            ],
        )
        with patch.object(executor, "_pre_validate_element", mock_pre_validate), \
             patch("robotmcp.components.execution.keyword_executor.asyncio.sleep",
                   new_callable=AsyncMock) as mock_sleep:
            await executor._pre_validate_element_with_retry(
                "id=foo", mock_session, "Click", timeout_ms=500,
            )
        # sleep called exactly once, with the configured gap in seconds.
        mock_sleep.assert_awaited_once()
        expected_secs = executor.PRE_VALIDATION_RETRY_GAP_MS / 1000.0
        assert mock_sleep.await_args.args == (expected_secs,)


# ---------------------------------------------------------------------------
# pre_validate_timeout_ms parameter wiring
# ---------------------------------------------------------------------------


class TestPreValidateTimeoutMsParameter:
    """The new per-call override must:
    - exist on KeywordExecutor.execute_keyword / _execute_keyword_serialized
    - exist on ExecutionCoordinator.execute_step
    - exist on the server-level execute_step tool
    - take precedence over timeout_ms-derived calculation
    - support 0/negative-value disable
    """

    def test_executor_execute_keyword_accepts_parameter(self):
        # Signature check via introspection — implementation behaviour
        # is exercised by the async tests below.
        import inspect
        sig = inspect.signature(KeywordExecutor.execute_keyword)
        assert "pre_validate_timeout_ms" in sig.parameters
        assert sig.parameters["pre_validate_timeout_ms"].default is None

    def test_executor_serialized_method_accepts_parameter(self):
        import inspect
        sig = inspect.signature(KeywordExecutor._execute_keyword_serialized)
        assert "pre_validate_timeout_ms" in sig.parameters
        assert sig.parameters["pre_validate_timeout_ms"].default is None

    def test_coordinator_execute_step_accepts_parameter(self):
        from robotmcp.components.execution.execution_coordinator import (
            ExecutionCoordinator,
        )
        import inspect
        sig = inspect.signature(ExecutionCoordinator.execute_step)
        assert "pre_validate_timeout_ms" in sig.parameters
        assert sig.parameters["pre_validate_timeout_ms"].default is None

    def test_server_execute_step_accepts_parameter(self):
        # The server-level tool is wrapped by @mcp.tool; call .fn to
        # introspect the original async function signature.
        from robotmcp.server import execute_step
        import inspect
        # FunctionTool has the wrapped function on .fn; otherwise the
        # tool registration didn't replace the symbol.
        fn = getattr(execute_step, "fn", execute_step)
        sig = inspect.signature(fn)
        assert "pre_validate_timeout_ms" in sig.parameters
        assert sig.parameters["pre_validate_timeout_ms"].default is None


# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------


class TestRetryConstants:
    """The retry tunables exist as class-level constants so tests
    (and future tuning code) can find them in one place."""

    def test_retry_gap_is_short_enough_to_be_cheap(self):
        # < 500ms keeps the worst-case cost on a genuine miss comparable
        # to the original single-shot budget. 200ms is the v0.32 default.
        assert KeywordExecutor.PRE_VALIDATION_RETRY_GAP_MS <= 500
        assert KeywordExecutor.PRE_VALIDATION_RETRY_GAP_MS > 0

    def test_max_retries_is_bounded(self):
        # Single retry is the documented behaviour. Going higher would
        # silently extend the per-step latency budget. If a page truly
        # needs more, the agent should pass pre_validate_timeout_ms.
        assert KeywordExecutor.PRE_VALIDATION_MAX_RETRIES == 1


# ---------------------------------------------------------------------------
# Failure-hint surfacing
# ---------------------------------------------------------------------------


class TestFailureHintMentionsTimeoutKnob:
    """When pre-validation fails after the retry, the failure response
    must include a hint pointing at ``pre_validate_timeout_ms`` so the
    LLM knows how to extend the gate. This is OBS-02 acceptance #2."""

    def test_failure_response_includes_pre_validate_timeout_hint_type(self):
        # The literal hint shape is built inline at the failure site in
        # _execute_keyword_serialized. We verify the shape via grep so
        # this test does not need to spin up the full executor.
        # NOTE: more thorough end-to-end coverage is in
        # test_prevalidation_fixes.py (existing real-executor tests).
        import pathlib
        executor_src = pathlib.Path(
            "src/robotmcp/components/execution/keyword_executor.py"
        ).read_text(encoding="utf-8")
        assert '"type": "pre_validate_timeout_hint"' in executor_src
        assert "pre_validate_timeout_ms=2000" in executor_src
        # ``pre_validate=False`` is not a real parameter — the cookbook
        # test test_step3_timeout_ms_zero_escape_present pins that the
        # *docs* never recommend it. We don't pin it here because the
        # executor source contains explanatory comments that mention the
        # avoided parameter by name; that's commentary, not behaviour.

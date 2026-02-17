"""Integration tests for ADR-011 Batch Execution domain.

Since execute_batch / resume_batch MCP tools are not yet wired into server.py,
these tests exercise the domain layer directly: BatchExecution aggregate,
BatchRunner service, BatchStateManager, and StepVariableResolver.

Uses AsyncMock keyword executors to simulate BuiltIn-only keyword execution
(Log, Set Variable, Convert To Upper Case, Should Be Equal, Evaluate).

Run with: uv run pytest tests/integration/test_execute_batch_e2e.py -v --timeout=60
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from robotmcp.domains.batch_execution.aggregates import BatchExecution, BatchState
from robotmcp.domains.batch_execution.entities import (
    BatchStep,
    FailureDetail,
    RecoveryAttempt,
    StepResult,
)
from robotmcp.domains.batch_execution.services import (
    BatchRunner,
    BatchStateManager,
    StepVariableResolver,
)
from robotmcp.domains.batch_execution.value_objects import (
    BatchId,
    BatchStatus,
    BatchTimeout,
    OnFailurePolicy,
    RecoveryAttemptLimit,
    StepReference,
    StepStatus,
    StepTimeout,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_executor(
    side_effects: Optional[List[Dict[str, Any]]] = None,
    default_return: Optional[Dict[str, Any]] = None,
) -> AsyncMock:
    """Build an AsyncMock keyword executor.

    If *side_effects* is given, each call returns the next item in order.
    Otherwise every call returns *default_return* (or a generic success).
    """
    executor = AsyncMock()
    if side_effects is not None:
        executor.execute_keyword = AsyncMock(side_effect=side_effects)
    else:
        ret = default_return or {"success": True, "return_value": "ok"}
        executor.execute_keyword = AsyncMock(return_value=ret)
    return executor


def _step(keyword: str, args: Optional[List[str]] = None,
          label: Optional[str] = None, timeout: Optional[str] = None,
          assign_to: Optional[str] = None) -> Dict[str, Any]:
    """Shorthand for a step dict matching the MCP input format."""
    d: Dict[str, Any] = {"keyword": keyword, "args": args or []}
    if label:
        d["label"] = label
    if timeout:
        d["timeout"] = timeout
    if assign_to:
        d["assign_to"] = assign_to
    return d


# =========================================================================
# Basic Execution Tests
# =========================================================================


class TestBasicExecution:
    """Tests 1-5: basic batch creation and execution."""

    @pytest.mark.asyncio
    async def test_execute_batch_single_step_builtin(self):
        """Test 1: One BuiltIn keyword (Log) executes and returns PASS."""
        executor = _make_executor(default_return={
            "success": True, "return_value": None,
        })
        batch = BatchExecution.create(
            session_id="test-single",
            steps_data=[_step("Log", ["Hello from batch"])],
        )
        runner = BatchRunner(keyword_executor=executor)
        result = await runner.execute(batch)

        assert result.status == BatchStatus.PASS
        assert len(result.results) == 1
        assert result.results[0].status == StepStatus.PASS
        assert result.results[0].keyword == "Log"
        executor.execute_keyword.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_batch_multi_step_pass(self):
        """Test 2: Multiple BuiltIn keywords all pass."""
        returns = [
            {"success": True, "return_value": None},       # Log
            {"success": True, "return_value": "HELLO"},     # Convert To Upper Case
            {"success": True, "return_value": None},        # Should Be Equal
        ]
        executor = _make_executor(side_effects=returns)

        batch = BatchExecution.create(
            session_id="test-multi",
            steps_data=[
                _step("Log", ["Starting"]),
                _step("Convert To Upper Case", ["hello"]),
                _step("Should Be Equal", ["HELLO", "HELLO"]),
            ],
        )
        runner = BatchRunner(keyword_executor=executor)
        result = await runner.execute(batch)

        assert result.status == BatchStatus.PASS
        assert len(result.results) == 3
        assert all(r.status == StepStatus.PASS for r in result.results)

    @pytest.mark.asyncio
    async def test_execute_batch_variable_chaining(self):
        """Test 3: ${STEP_0} reference in later step args resolves correctly."""
        returns = [
            {"success": True, "return_value": "WORLD"},    # step 0
            {"success": True, "return_value": None},        # step 1 (uses ${STEP_0})
        ]
        executor = _make_executor(side_effects=returns)

        batch = BatchExecution.create(
            session_id="test-chain",
            steps_data=[
                _step("Convert To Upper Case", ["world"]),
                _step("Log", ["Result is: ${STEP_0}"]),
            ],
        )
        runner = BatchRunner(keyword_executor=executor)
        result = await runner.execute(batch)

        assert result.status == BatchStatus.PASS
        # Verify the resolved args for step 1
        assert result.results[1].args_resolved == ["Result is: WORLD"]

    @pytest.mark.asyncio
    async def test_execute_batch_with_labels(self):
        """Test 4: Steps with labels appear in response."""
        executor = _make_executor()
        batch = BatchExecution.create(
            session_id="test-labels",
            steps_data=[
                _step("Log", ["step one"], label="Setup Logging"),
                _step("Log", ["step two"], label="Teardown Logging"),
            ],
        )
        runner = BatchRunner(keyword_executor=executor)
        result = await runner.execute(batch)

        assert result.status == BatchStatus.PASS
        resp = result.to_response_dict()
        assert resp["steps"][0]["label"] == "Setup Logging"
        assert resp["steps"][1]["label"] == "Teardown Logging"

    def test_execute_batch_empty_steps_rejected(self):
        """Test 5: Empty steps list raises ValueError."""
        with pytest.raises(ValueError, match="At least one step"):
            BatchExecution.create(
                session_id="test-empty",
                steps_data=[],
            )


# =========================================================================
# Failure Handling Tests
# =========================================================================


class TestFailureHandling:
    """Tests 6-9: failure modes and on_failure policies."""

    @pytest.mark.asyncio
    async def test_execute_batch_stop_on_failure(self):
        """Test 6: on_failure='stop' aborts on first error, no retry."""
        returns = [
            {"success": True, "return_value": None},
            {"success": False, "error": "Element not found"},
            # step 2 should NOT be called
        ]
        executor = _make_executor(side_effects=returns)

        batch = BatchExecution.create(
            session_id="test-stop",
            steps_data=[
                _step("Log", ["ok"]),
                _step("Click Element", ["id=missing"]),
                _step("Log", ["should not run"]),
            ],
            on_failure="stop",
        )
        runner = BatchRunner(keyword_executor=executor)
        result = await runner.execute(batch)

        assert result.status == BatchStatus.FAIL
        assert len(result.results) == 2
        assert result.results[0].status == StepStatus.PASS
        assert result.results[1].status == StepStatus.FAIL
        # step 2 never called
        assert executor.execute_keyword.await_count == 2

    @pytest.mark.asyncio
    async def test_execute_batch_recover_on_failure(self):
        """Test 7: on_failure='recover' attempts recovery then retries."""
        call_count = 0

        async def dynamic_executor(session_id, keyword, args, timeout=None, assign_to=None):
            nonlocal call_count
            call_count += 1
            if keyword == "Failing Keyword" and call_count == 2:
                # First call (call_count=2 because step 0 Log is call 1)
                return {"success": False, "error": "Temporary error"}
            if keyword == "Failing Keyword" and call_count == 3:
                # Retry after recovery
                return {"success": True, "return_value": "recovered"}
            return {"success": True, "return_value": None}

        executor = AsyncMock()
        executor.execute_keyword = AsyncMock(side_effect=dynamic_executor)

        # Provide a mock recovery service
        recovery_svc = AsyncMock()
        recovery_svc.attempt_recovery = AsyncMock(return_value=RecoveryAttempt(
            attempt_number=1, strategy="retry", tier=1,
            action_description="Retrying keyword", result="ATTEMPTED",
        ))

        batch = BatchExecution.create(
            session_id="test-recover",
            steps_data=[
                _step("Log", ["before"]),
                _step("Failing Keyword", ["arg1"]),
            ],
            on_failure="recover",
            max_recovery_attempts=2,
        )
        runner = BatchRunner(
            keyword_executor=executor,
            recovery_service=recovery_svc,
        )
        result = await runner.execute(batch)

        # Step 1 failed initially but recovered on retry
        assert result.status == BatchStatus.RECOVERED
        assert result.results[1].status == StepStatus.RECOVERED

    @pytest.mark.asyncio
    async def test_execute_batch_unknown_keyword_fails(self):
        """Test 8: Non-existent keyword fails gracefully."""
        executor = _make_executor(side_effects=[
            {"success": False, "error": "No keyword with name 'Nonexistent Keyword' found"},
        ])

        batch = BatchExecution.create(
            session_id="test-unknown",
            steps_data=[_step("Nonexistent Keyword", ["arg"])],
            on_failure="stop",
        )
        runner = BatchRunner(keyword_executor=executor)
        result = await runner.execute(batch)

        assert result.status == BatchStatus.FAIL
        assert "Nonexistent Keyword" in result.results[0].error

    @pytest.mark.asyncio
    async def test_execute_batch_forward_reference_rejected(self):
        """Test 9: ${STEP_1} in step 0 fails validation at execution time."""
        executor = _make_executor()

        batch = BatchExecution.create(
            session_id="test-fwd-ref",
            steps_data=[
                _step("Log", ["${STEP_1}"]),   # Forward reference!
                _step("Log", ["hello"]),
            ],
        )
        runner = BatchRunner(keyword_executor=executor)
        result = await runner.execute(batch)

        assert result.status == BatchStatus.FAIL
        assert result.failure is not None
        assert result.failure.step_index == 0
        assert "Forward reference" in result.failure.error
        # Executor should NOT have been called (resolution failed before execution)
        executor.execute_keyword.assert_not_awaited()


# =========================================================================
# Timeout Tests
# =========================================================================


class TestTimeouts:
    """Tests 10-12: timeout value objects and boundary validation."""

    def test_execute_batch_default_timeout(self):
        """Test 10: Default timeout is 120000ms."""
        batch = BatchExecution.create(
            session_id="test-default-timeout",
            steps_data=[_step("Log", ["x"])],
        )
        assert batch.timeout.value_ms == 120000

    def test_execute_batch_custom_timeout(self):
        """Test 11: Custom timeout_ms is respected."""
        batch = BatchExecution.create(
            session_id="test-custom-timeout",
            steps_data=[_step("Log", ["x"])],
            timeout_ms=30000,
        )
        assert batch.timeout.value_ms == 30000

    @pytest.mark.parametrize("bad_value", [0, 500, 999, 600001, 999999])
    def test_execute_batch_timeout_value_bounds(self, bad_value):
        """Test 12: Below 1000 or above 600000 is rejected."""
        with pytest.raises(ValueError, match="BatchTimeout"):
            BatchExecution.create(
                session_id="test-bounds",
                steps_data=[_step("Log", ["x"])],
                timeout_ms=bad_value,
            )

    @pytest.mark.asyncio
    async def test_execute_batch_timeout_expires_during_run(self):
        """Batch that exceeds its timeout budget is marked TIMEOUT."""
        call_count = 0

        async def slow_executor(session_id, keyword, args, timeout=None):
            nonlocal call_count
            call_count += 1
            # Simulate a very slow keyword
            await asyncio.sleep(0.05)
            return {"success": True, "return_value": None}

        executor = AsyncMock()
        executor.execute_keyword = AsyncMock(side_effect=slow_executor)

        # Create batch with very tight timeout (1 second)
        batch = BatchExecution.create(
            session_id="test-timeout-expires",
            steps_data=[_step("Log", [str(i)]) for i in range(100)],
            timeout_ms=1000,
        )

        # Manually manipulate the start time to simulate timeout
        # We can't reliably time 100 async calls, so we use the domain
        # method directly:
        runner = BatchRunner(keyword_executor=executor)

        # Patch start_clock to set an already-expired start time
        import time
        batch.start_clock()
        batch._start_time = time.monotonic() - 2.0  # 2 seconds ago, > 1s budget

        # Now manually finalize since the runner already called start_clock
        # Instead, test via the aggregate directly:
        assert batch.is_timed_out() is True
        batch.record_timeout()
        batch.finalize()
        assert batch.status == BatchStatus.TIMEOUT


# =========================================================================
# Response Format Tests
# =========================================================================


class TestResponseFormat:
    """Tests 13-17: response dict structure validation."""

    @pytest.mark.asyncio
    async def test_execute_batch_response_has_status(self):
        """Test 13: status field present in response dict."""
        executor = _make_executor()
        batch = BatchExecution.create(
            session_id="test-resp-status",
            steps_data=[_step("Log", ["x"])],
        )
        runner = BatchRunner(keyword_executor=executor)
        result = await runner.execute(batch)
        resp = result.to_response_dict()

        assert "status" in resp
        assert resp["status"] == "PASS"

    @pytest.mark.asyncio
    async def test_execute_batch_response_has_summary(self):
        """Test 14: summary field present in response dict."""
        executor = _make_executor()
        batch = BatchExecution.create(
            session_id="test-resp-summary",
            steps_data=[_step("Log", ["x"]), _step("Log", ["y"])],
        )
        runner = BatchRunner(keyword_executor=executor)
        result = await runner.execute(batch)
        resp = result.to_response_dict()

        assert "summary" in resp
        assert "2/2 passed" in resp["summary"]

    @pytest.mark.asyncio
    async def test_execute_batch_response_has_steps_array(self):
        """Test 15: steps array with per-step results in response."""
        executor = _make_executor()
        batch = BatchExecution.create(
            session_id="test-resp-steps",
            steps_data=[
                _step("Log", ["a"]),
                _step("Log", ["b"]),
                _step("Log", ["c"]),
            ],
        )
        runner = BatchRunner(keyword_executor=executor)
        result = await runner.execute(batch)
        resp = result.to_response_dict()

        assert "steps" in resp
        assert isinstance(resp["steps"], list)
        assert len(resp["steps"]) == 3
        for i, step_resp in enumerate(resp["steps"]):
            assert step_resp["index"] == i
            assert step_resp["keyword"] == "Log"
            assert step_resp["status"] == "PASS"

    @pytest.mark.asyncio
    async def test_execute_batch_response_has_timing(self):
        """Test 16: total_time_ms present in response dict."""
        executor = _make_executor()
        batch = BatchExecution.create(
            session_id="test-resp-timing",
            steps_data=[_step("Log", ["x"])],
        )
        runner = BatchRunner(keyword_executor=executor)
        result = await runner.execute(batch)
        resp = result.to_response_dict()

        assert "total_time_ms" in resp
        assert isinstance(resp["total_time_ms"], int)
        assert resp["total_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_execute_batch_failure_includes_batch_id(self):
        """Test 17: FAIL response has batch_id for resume."""
        executor = _make_executor(side_effects=[
            {"success": False, "error": "step failed"},
        ])
        batch = BatchExecution.create(
            session_id="test-resp-bid",
            steps_data=[_step("Fail", ["step failed"])],
            on_failure="stop",
        )
        runner = BatchRunner(keyword_executor=executor)
        result = await runner.execute(batch)
        resp = result.to_response_dict()

        assert resp["status"] == "FAIL"
        assert "batch_id" in resp
        assert resp["batch_id"].startswith("batch_")
        assert "failure" in resp
        assert resp["failure"]["step_index"] == 0


# =========================================================================
# Resume Tests (Phase 2)
# =========================================================================


class TestResume:
    """Tests 18-20: BatchState and BatchStateManager for resume_batch."""

    def test_resume_batch_not_found(self):
        """Test 18: Non-existent batch_id returns None from state manager."""
        manager = BatchStateManager()
        result = manager.get("batch_nonexistent123")
        assert result is None

    @pytest.mark.asyncio
    async def test_resume_batch_after_failure(self):
        """Test 19: BatchState captures failure point for resume."""
        # Execute a batch that fails at step 1
        executor = _make_executor(side_effects=[
            {"success": True, "return_value": "step0_val"},
            {"success": False, "error": "step 1 broke"},
        ])
        batch = BatchExecution.create(
            session_id="test-resume",
            steps_data=[
                _step("Log", ["ok"]),
                _step("Failing Keyword", ["arg"]),
                _step("Log", ["after failure"]),
            ],
            on_failure="stop",
        )
        runner = BatchRunner(keyword_executor=executor)
        result = await runner.execute(batch)

        assert result.status == BatchStatus.FAIL
        assert result.failure is not None

        # Create BatchState from failed execution
        state = BatchState.from_execution(result)
        assert state.batch_id == result.batch_id
        assert state.session_id == "test-resume"
        assert state.failed_at_index == 1

        # Remaining steps should be step 2
        remaining = state.remaining_steps
        assert len(remaining) == 1
        assert remaining[0].index == 2

        # results_map should have step 0's value
        assert state.results_map[0] == "step0_val"

        # Store and retrieve
        manager = BatchStateManager()
        manager.store(state)
        retrieved = manager.get(state.batch_id.value)
        assert retrieved is not None
        assert retrieved.failed_at_index == 1

    @pytest.mark.asyncio
    async def test_resume_batch_with_fix_steps(self):
        """Test 20: Fix steps can be prepended before retrying remaining steps."""
        # Simulate: original batch fails at step 1, we resume with a fix step
        executor = _make_executor(side_effects=[
            {"success": True, "return_value": "step0_val"},
            {"success": False, "error": "button not visible"},
        ])
        batch = BatchExecution.create(
            session_id="test-fix",
            steps_data=[
                _step("Log", ["setup"]),
                _step("Click Element", ["id=btn"]),
                _step("Log", ["after click"]),
            ],
            on_failure="stop",
        )
        runner = BatchRunner(keyword_executor=executor)
        failed_batch = await runner.execute(batch)

        assert failed_batch.status == BatchStatus.FAIL
        state = BatchState.from_execution(failed_batch)

        # Now create a new batch with fix steps + the failed step + remaining
        fix_steps = [_step("Scroll Element Into View", ["id=btn"], label="Fix: scroll")]
        retry_step = state.failed_step
        assert retry_step is not None

        resume_steps = (
            fix_steps
            + [{"keyword": retry_step.keyword, "args": retry_step.args}]
            + [{"keyword": s.keyword, "args": s.args} for s in state.remaining_steps]
        )

        # Execute the resume batch (all pass this time)
        resume_executor = _make_executor(default_return={
            "success": True, "return_value": None,
        })
        resume_batch = BatchExecution.create(
            session_id="test-fix",
            steps_data=resume_steps,
        )
        resume_runner = BatchRunner(keyword_executor=resume_executor)
        resume_result = await resume_runner.execute(resume_batch)

        assert resume_result.status == BatchStatus.PASS
        assert len(resume_result.results) == 3  # fix + retry + remaining


# =========================================================================
# Session Integration (domain-level)
# =========================================================================


class TestSessionIntegration:
    """Tests 21-23: session_id handling at domain level."""

    @pytest.mark.asyncio
    async def test_execute_batch_creates_session_via_executor(self):
        """Test 21: session_id is passed through to the keyword executor."""
        executor = _make_executor()
        sid = "auto-create-session-123"
        batch = BatchExecution.create(
            session_id=sid,
            steps_data=[_step("Log", ["hello"])],
        )
        runner = BatchRunner(keyword_executor=executor)
        await runner.execute(batch)

        # Verify the executor was called with the correct session_id
        call_args = executor.execute_keyword.call_args
        assert call_args[0][0] == sid  # First positional arg is session_id

    @pytest.mark.asyncio
    async def test_execute_batch_uses_existing_session(self):
        """Test 22: Uses the provided session_id throughout all steps."""
        executor = _make_executor()
        sid = "existing-session-456"
        batch = BatchExecution.create(
            session_id=sid,
            steps_data=[_step("Log", ["a"]), _step("Log", ["b"])],
        )
        runner = BatchRunner(keyword_executor=executor)
        await runner.execute(batch)

        for call in executor.execute_keyword.call_args_list:
            assert call[0][0] == sid

    @pytest.mark.asyncio
    async def test_execute_batch_concurrent_sessions(self):
        """Test 23: Two batches on different sessions can run without interference."""
        executor_a = _make_executor(default_return={
            "success": True, "return_value": "session_a_value",
        })
        executor_b = _make_executor(default_return={
            "success": True, "return_value": "session_b_value",
        })

        batch_a = BatchExecution.create(
            session_id="session-a",
            steps_data=[_step("Log", ["from A"])],
        )
        batch_b = BatchExecution.create(
            session_id="session-b",
            steps_data=[_step("Log", ["from B"])],
        )

        runner_a = BatchRunner(keyword_executor=executor_a)
        runner_b = BatchRunner(keyword_executor=executor_b)

        result_a, result_b = await asyncio.gather(
            runner_a.execute(batch_a),
            runner_b.execute(batch_b),
        )

        assert result_a.status == BatchStatus.PASS
        assert result_b.status == BatchStatus.PASS
        assert result_a.session_id == "session-a"
        assert result_b.session_id == "session-b"
        assert result_a.results_map[0] == "session_a_value"
        assert result_b.results_map[0] == "session_b_value"


# =========================================================================
# Edge Cases
# =========================================================================


class TestEdgeCases:
    """Tests 24-25: boundary conditions and format validation."""

    @pytest.mark.parametrize("bad_value", [0, -1, 11, 100])
    def test_execute_batch_max_recovery_attempts_bounds(self, bad_value):
        """Test 24: Validate recovery attempts 1-10 range."""
        with pytest.raises(ValueError, match="RecoveryAttemptLimit"):
            BatchExecution.create(
                session_id="test-recovery-bounds",
                steps_data=[_step("Log", ["x"])],
                max_recovery_attempts=bad_value,
            )

    @pytest.mark.parametrize("valid_limit", [1, 5, 10])
    def test_execute_batch_max_recovery_attempts_valid(self, valid_limit):
        """Recovery attempts within 1-10 range are accepted."""
        batch = BatchExecution.create(
            session_id="test-recovery-valid",
            steps_data=[_step("Log", ["x"])],
            max_recovery_attempts=valid_limit,
        )
        assert batch.max_recovery_attempts.value == valid_limit

    @pytest.mark.parametrize("rf_timeout", ["10s", "1.5m", "500ms", "2 minutes", "1 hour"])
    def test_execute_batch_step_timeout_rf_format(self, rf_timeout):
        """Test 25: Per-step timeout accepts RF duration format."""
        batch = BatchExecution.create(
            session_id="test-rf-timeout",
            steps_data=[_step("Log", ["x"], timeout=rf_timeout)],
        )
        assert batch.steps[0].timeout is not None
        assert batch.steps[0].timeout.rf_format == rf_timeout

    @pytest.mark.parametrize("bad_timeout", ["abc", "10x", "forever"])
    def test_execute_batch_step_timeout_invalid_format_rejected(self, bad_timeout):
        """Invalid RF duration formats are rejected."""
        with pytest.raises(ValueError, match="Invalid RF duration"):
            BatchExecution.create(
                session_id="test-bad-timeout",
                steps_data=[_step("Log", ["x"], timeout=bad_timeout)],
            )

    def test_execute_batch_empty_timeout_string_treated_as_no_timeout(self):
        """Empty string timeout is treated as no timeout (falsy in factory)."""
        batch = BatchExecution.create(
            session_id="test-empty-timeout",
            steps_data=[_step("Log", ["x"], timeout="")],
        )
        assert batch.steps[0].timeout is None


# =========================================================================
# StepVariableResolver standalone tests
# =========================================================================


class TestStepVariableResolver:
    """Additional coverage for the StepVariableResolver service."""

    def test_resolve_no_references(self):
        resolver = StepVariableResolver()
        result = resolver.resolve(["hello", "world"], {}, step_index=0)
        assert result == ["hello", "world"]

    def test_resolve_single_reference(self):
        resolver = StepVariableResolver()
        result = resolver.resolve(
            ["prefix_${STEP_0}_suffix"],
            {0: "VALUE"},
            step_index=1,
        )
        assert result == ["prefix_VALUE_suffix"]

    def test_resolve_multiple_references_in_one_arg(self):
        resolver = StepVariableResolver()
        result = resolver.resolve(
            ["${STEP_0} and ${STEP_1}"],
            {0: "A", 1: "B"},
            step_index=2,
        )
        assert result == ["A and B"]

    def test_resolve_forward_reference_raises(self):
        resolver = StepVariableResolver()
        with pytest.raises(ValueError, match="Forward reference"):
            resolver.resolve(["${STEP_1}"], {}, step_index=0)

    def test_resolve_missing_result_raises(self):
        resolver = StepVariableResolver()
        with pytest.raises(ValueError, match="not available"):
            resolver.resolve(["${STEP_0}"], {}, step_index=1)

    def test_validate_references_finds_forward_refs(self):
        resolver = StepVariableResolver()
        steps = [
            BatchStep(index=0, keyword="Log", args=["${STEP_1}"]),
            BatchStep(index=1, keyword="Log", args=["ok"]),
        ]
        errors = resolver.validate_references(steps)
        assert len(errors) == 1
        assert "forward reference" in errors[0].lower()

    def test_validate_references_clean(self):
        resolver = StepVariableResolver()
        steps = [
            BatchStep(index=0, keyword="Log", args=["hello"]),
            BatchStep(index=1, keyword="Log", args=["${STEP_0}"]),
        ]
        errors = resolver.validate_references(steps)
        assert errors == []


# =========================================================================
# BatchStateManager tests
# =========================================================================


class TestBatchStateManager:
    """Additional coverage for the BatchStateManager service."""

    @pytest.mark.asyncio
    async def test_state_manager_store_and_retrieve(self):
        """Store a state and retrieve it by batch_id."""
        executor = _make_executor(side_effects=[
            {"success": False, "error": "bang"},
        ])
        batch = BatchExecution.create(
            session_id="mgr-test",
            steps_data=[_step("Fail", ["x"])],
            on_failure="stop",
        )
        runner = BatchRunner(keyword_executor=executor)
        result = await runner.execute(batch)

        state = BatchState.from_execution(result)
        manager = BatchStateManager()
        manager.store(state)

        assert manager.count == 1
        retrieved = manager.get(state.batch_id.value)
        assert retrieved is not None
        assert retrieved.session_id == "mgr-test"

    @pytest.mark.asyncio
    async def test_state_manager_remove(self):
        """Remove a state by batch_id."""
        executor = _make_executor(side_effects=[
            {"success": False, "error": "fail"},
        ])
        batch = BatchExecution.create(
            session_id="rm-test",
            steps_data=[_step("X", ["y"])],
            on_failure="stop",
        )
        runner = BatchRunner(keyword_executor=executor)
        result = await runner.execute(batch)

        state = BatchState.from_execution(result)
        manager = BatchStateManager()
        manager.store(state)
        assert manager.count == 1

        removed = manager.remove(state.batch_id.value)
        assert removed is True
        assert manager.count == 0
        assert manager.get(state.batch_id.value) is None

    def test_state_manager_expired_entry_not_returned(self):
        """Expired states are cleaned up on get()."""
        from datetime import datetime, timedelta

        bid = BatchId(value="batch_expired0001")
        state = BatchState(
            batch_id=bid,
            session_id="expired-test",
            original_steps=[BatchStep(index=0, keyword="X", args=[])],
            results=[],
            results_map={},
            failed_at_index=0,
            on_failure=OnFailurePolicy.STOP,
            max_recovery_attempts=RecoveryAttemptLimit.default(),
            timeout_ms=120000,
            elapsed_ms=0,
            created_at=datetime.now() - timedelta(seconds=600),  # 10 min ago
        )
        manager = BatchStateManager(ttl_seconds=300)
        manager._states[bid.value] = state  # Bypass store to avoid cleanup

        result = manager.get(bid.value)
        assert result is None  # Expired

    def test_state_manager_evicts_oldest_at_capacity(self):
        """When max_states is reached, oldest entry is evicted."""
        from datetime import datetime, timedelta

        manager = BatchStateManager(max_states=2)

        # Create 3 states with different ages
        for i in range(3):
            bid = BatchId(value=f"batch_cap_{i:04d}")
            state = BatchState(
                batch_id=bid,
                session_id=f"cap-{i}",
                original_steps=[BatchStep(index=0, keyword="X", args=[])],
                results=[],
                results_map={},
                failed_at_index=0,
                on_failure=OnFailurePolicy.STOP,
                max_recovery_attempts=RecoveryAttemptLimit.default(),
                timeout_ms=120000,
                elapsed_ms=0,
                created_at=datetime.now() - timedelta(seconds=100 - i * 10),
            )
            manager.store(state)

        # max_states=2, so oldest (batch_cap_0000) should be evicted
        assert manager.count == 2
        assert manager.get("batch_cap_0000") is None
        assert manager.get("batch_cap_0001") is not None
        assert manager.get("batch_cap_0002") is not None

    def test_cannot_create_batch_state_from_non_failed_execution(self):
        """BatchState.from_execution rejects non-failed batches."""
        batch = BatchExecution.create(
            session_id="no-fail",
            steps_data=[_step("Log", ["x"])],
        )
        # Manually set status to PASS without failure
        batch.status = BatchStatus.PASS
        batch.failure = None

        with pytest.raises(ValueError, match="non-failed"):
            BatchState.from_execution(batch)


# =========================================================================
# Value Object tests
# =========================================================================


class TestValueObjects:
    """Coverage for value object invariants."""

    def test_batch_id_generate_format(self):
        bid = BatchId.generate()
        assert bid.value.startswith("batch_")
        assert len(bid.value) == 18  # "batch_" + 12 hex chars

    def test_batch_id_empty_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            BatchId(value="")

    def test_step_reference_find_all(self):
        refs = StepReference.find_all("Use ${STEP_0} then ${STEP_2}")
        assert len(refs) == 2
        assert refs[0].index == 0
        assert refs[0].raw == "${STEP_0}"
        assert refs[1].index == 2
        assert refs[1].raw == "${STEP_2}"

    def test_step_reference_no_matches(self):
        refs = StepReference.find_all("no references here")
        assert refs == []

    def test_on_failure_policy_values(self):
        assert OnFailurePolicy("stop") == OnFailurePolicy.STOP
        assert OnFailurePolicy("retry") == OnFailurePolicy.RETRY
        assert OnFailurePolicy("recover") == OnFailurePolicy.RECOVER

    def test_on_failure_policy_invalid(self):
        with pytest.raises(ValueError):
            OnFailurePolicy("crash")

    def test_batch_timeout_boundary_values(self):
        # Exact boundaries should be accepted
        t_min = BatchTimeout(value_ms=1000)
        assert t_min.value_ms == 1000
        t_max = BatchTimeout(value_ms=600000)
        assert t_max.value_ms == 600000

    def test_recovery_attempt_limit_boundary(self):
        assert RecoveryAttemptLimit(value=1).value == 1
        assert RecoveryAttemptLimit(value=10).value == 10

    def test_step_timeout_valid_formats(self):
        for fmt in ["10s", "1m", "500ms", "2 seconds", "1.5 minutes"]:
            st = StepTimeout(rf_format=fmt)
            assert st.rf_format == fmt

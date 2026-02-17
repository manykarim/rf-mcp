"""Tests for batch_execution services."""
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from robotmcp.domains.batch_execution.aggregates import (
    BatchExecution,
    BatchState,
)
from robotmcp.domains.batch_execution.entities import (
    BatchStep,
    FailureDetail,
    RecoveryAttempt,
)
from robotmcp.domains.batch_execution.services import (
    BatchRunner,
    BatchStateManager,
    StepVariableResolver,
)
from robotmcp.domains.batch_execution.value_objects import (
    BatchId,
    BatchStatus,
    OnFailurePolicy,
    RecoveryAttemptLimit,
    StepStatus,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_batch(steps_data=None, **kwargs):
    if steps_data is None:
        steps_data = [{"keyword": "Log", "args": ["hello"]}]
    defaults = {"session_id": "s1", "steps_data": steps_data}
    defaults.update(kwargs)
    return BatchExecution.create(**defaults)


def _success_result(return_value=None):
    return {"success": True, "return_value": return_value}


def _fail_result(error="Error"):
    return {"success": False, "error": error}


# ── StepVariableResolver ─────────────────────────────────────────────


class TestStepVariableResolver:
    def test_resolve_no_refs(self):
        resolver = StepVariableResolver()
        result = resolver.resolve(["plain text"], {}, 0)
        assert result == ["plain text"]

    def test_resolve_valid_back_ref(self):
        resolver = StepVariableResolver()
        result = resolver.resolve(
            ["Bearer ${STEP_0}"], {0: "token123"}, step_index=1,
        )
        assert result == ["Bearer token123"]

    def test_resolve_multiple_refs(self):
        resolver = StepVariableResolver()
        result = resolver.resolve(
            ["${STEP_0}:${STEP_1}"], {0: "user", 1: "pass"}, step_index=2,
        )
        assert result == ["user:pass"]

    def test_resolve_forward_ref_raises(self):
        resolver = StepVariableResolver()
        with pytest.raises(ValueError, match="Forward reference"):
            resolver.resolve(["${STEP_1}"], {}, step_index=0)

    def test_resolve_self_ref_raises(self):
        resolver = StepVariableResolver()
        with pytest.raises(ValueError, match="Forward reference"):
            resolver.resolve(["${STEP_0}"], {}, step_index=0)

    def test_resolve_missing_result_raises(self):
        resolver = StepVariableResolver()
        with pytest.raises(ValueError, match="not available"):
            resolver.resolve(["${STEP_0}"], {}, step_index=1)

    def test_resolve_empty_args(self):
        resolver = StepVariableResolver()
        assert resolver.resolve([], {}, 0) == []

    def test_resolve_no_match_unchanged(self):
        resolver = StepVariableResolver()
        result = resolver.resolve(["hello world"], {}, 0)
        assert result == ["hello world"]

    def test_validate_references_valid(self):
        resolver = StepVariableResolver()
        steps = [
            BatchStep(index=0, keyword="A", args=["plain"]),
            BatchStep(index=1, keyword="B", args=["${STEP_0}"]),
        ]
        errors = resolver.validate_references(steps)
        assert errors == []

    def test_validate_references_forward_ref(self):
        resolver = StepVariableResolver()
        steps = [
            BatchStep(index=0, keyword="A", args=["${STEP_1}"]),
            BatchStep(index=1, keyword="B", args=[]),
        ]
        errors = resolver.validate_references(steps)
        assert len(errors) == 1
        assert "forward reference" in errors[0]

    def test_validate_references_multiple_errors(self):
        resolver = StepVariableResolver()
        steps = [
            BatchStep(index=0, keyword="A", args=["${STEP_1}", "${STEP_2}"]),
            BatchStep(index=1, keyword="B", args=["${STEP_2}"]),
        ]
        errors = resolver.validate_references(steps)
        assert len(errors) == 3

    def test_validate_references_empty_steps(self):
        resolver = StepVariableResolver()
        assert resolver.validate_references([]) == []


# ── BatchRunner ──────────────────────────────────────────────────────


class TestBatchRunnerAllPass:
    __test__ = True

    @pytest.mark.asyncio
    async def test_all_steps_pass(self):
        executor = AsyncMock()
        executor.execute_keyword = AsyncMock(return_value=_success_result("ok"))

        runner = BatchRunner(keyword_executor=executor)
        batch = _make_batch(steps_data=[
            {"keyword": "Log", "args": ["a"]},
            {"keyword": "Log", "args": ["b"]},
        ])
        result = await runner.execute(batch)
        assert result.status == BatchStatus.PASS
        assert len(result.results) == 2
        assert all(r.status == StepStatus.PASS for r in result.results)

    @pytest.mark.asyncio
    async def test_return_values_in_results_map(self):
        executor = AsyncMock()
        executor.execute_keyword = AsyncMock(
            side_effect=[_success_result("v0"), _success_result("v1")]
        )
        runner = BatchRunner(keyword_executor=executor)
        batch = _make_batch(steps_data=[
            {"keyword": "A", "args": []},
            {"keyword": "B", "args": []},
        ])
        result = await runner.execute(batch)
        assert result.results_map[0] == "v0"
        assert result.results_map[1] == "v1"


class TestBatchRunnerFailStop:
    __test__ = True

    @pytest.mark.asyncio
    async def test_first_fail_stops(self):
        executor = AsyncMock()
        executor.execute_keyword = AsyncMock(return_value=_fail_result("boom"))

        runner = BatchRunner(keyword_executor=executor)
        batch = _make_batch(
            steps_data=[
                {"keyword": "Fail", "args": []},
                {"keyword": "Log", "args": ["never"]},
            ],
            on_failure="stop",
        )
        result = await runner.execute(batch)
        assert result.status == BatchStatus.FAIL
        assert len(result.results) == 1
        assert result.failure is not None
        assert result.failure.step_index == 0

    @pytest.mark.asyncio
    async def test_exception_stops(self):
        executor = AsyncMock()
        executor.execute_keyword = AsyncMock(
            side_effect=RuntimeError("connection lost")
        )

        runner = BatchRunner(keyword_executor=executor)
        batch = _make_batch(on_failure="stop")
        result = await runner.execute(batch)
        assert result.status == BatchStatus.FAIL

    @pytest.mark.asyncio
    async def test_evidence_collected_on_stop(self):
        executor = AsyncMock()
        executor.execute_keyword = AsyncMock(return_value=_fail_result("err"))

        evidence_collector = AsyncMock()
        evidence_collector.collect_evidence = AsyncMock(return_value={
            "screenshot_base64": "abc=",
            "current_url": "http://x",
        })

        runner = BatchRunner(
            keyword_executor=executor,
            evidence_collector=evidence_collector,
        )
        batch = _make_batch(on_failure="stop")
        result = await runner.execute(batch)
        assert result.failure.screenshot_base64 == "abc="
        assert result.failure.current_url == "http://x"


class TestBatchRunnerRecovery:
    __test__ = True

    @pytest.mark.asyncio
    async def test_recovery_success(self):
        executor = AsyncMock()
        executor.execute_keyword = AsyncMock(
            side_effect=[
                _fail_result("err"),
                _success_result("recovered_val"),
            ]
        )
        recovery = AsyncMock()
        recovery.attempt_recovery = AsyncMock(return_value=RecoveryAttempt(
            attempt_number=1, strategy="wait", tier=1,
            action_description="waited", result="PASS",
        ))

        runner = BatchRunner(
            keyword_executor=executor,
            recovery_service=recovery,
        )
        batch = _make_batch(on_failure="recover")
        result = await runner.execute(batch)
        assert result.status == BatchStatus.RECOVERED
        assert result.results[0].status == StepStatus.RECOVERED

    @pytest.mark.asyncio
    async def test_recovery_exhausted(self):
        executor = AsyncMock()
        executor.execute_keyword = AsyncMock(return_value=_fail_result("err"))
        recovery = AsyncMock()
        recovery.attempt_recovery = AsyncMock(return_value=None)

        runner = BatchRunner(
            keyword_executor=executor,
            recovery_service=recovery,
        )
        batch = _make_batch(on_failure="recover", max_recovery_attempts=2)
        result = await runner.execute(batch)
        assert result.status == BatchStatus.FAIL

    @pytest.mark.asyncio
    async def test_retry_without_recovery_service(self):
        call_count = 0

        async def _execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return _fail_result("err")
            return _success_result("ok")

        executor = AsyncMock()
        executor.execute_keyword = AsyncMock(side_effect=_execute)

        runner = BatchRunner(keyword_executor=executor)
        batch = _make_batch(on_failure="retry")
        result = await runner.execute(batch)
        assert result.status == BatchStatus.RECOVERED

    @pytest.mark.asyncio
    async def test_recovery_exception_continues(self):
        executor = AsyncMock()
        executor.execute_keyword = AsyncMock(
            side_effect=[
                _fail_result("err"),
                _success_result("ok"),
            ]
        )
        recovery = AsyncMock()
        recovery.attempt_recovery = AsyncMock(
            side_effect=RuntimeError("recovery broken")
        )

        runner = BatchRunner(
            keyword_executor=executor,
            recovery_service=recovery,
        )
        batch = _make_batch(on_failure="recover")
        result = await runner.execute(batch)
        assert result.status == BatchStatus.RECOVERED


class TestBatchRunnerTimeout:
    __test__ = True

    @pytest.mark.asyncio
    async def test_timeout_before_step(self):
        """Simulate timeout by having the first step succeed but take
        enough virtual time that the second step sees timeout."""
        executor = AsyncMock()
        call_count = 0

        async def _slow_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Simulate first step succeeding; manipulate clock
                # to make the batch appear timed out before step 2
                batch._start_time = time.monotonic() - 2.0
                return _success_result("v0")
            return _success_result("v1")

        executor.execute_keyword = AsyncMock(side_effect=_slow_execute)

        runner = BatchRunner(keyword_executor=executor)
        batch = _make_batch(
            steps_data=[
                {"keyword": "A", "args": []},
                {"keyword": "B", "args": []},
            ],
            timeout_ms=1000,
        )
        result = await runner.execute(batch)
        assert result.status == BatchStatus.TIMEOUT
        # Only the first step should have executed
        assert len(result.results) == 1

    @pytest.mark.asyncio
    async def test_variable_resolution_error(self):
        executor = AsyncMock()
        runner = BatchRunner(keyword_executor=executor)
        batch = _make_batch(steps_data=[
            {"keyword": "A", "args": ["${STEP_1}"]},
            {"keyword": "B", "args": []},
        ])
        result = await runner.execute(batch)
        assert result.status == BatchStatus.FAIL
        assert "resolution error" in result.failure.error.lower() or "Forward" in result.failure.error

    @pytest.mark.asyncio
    async def test_step_timeout_passed_to_executor(self):
        executor = AsyncMock()
        executor.execute_keyword = AsyncMock(return_value=_success_result())

        runner = BatchRunner(keyword_executor=executor)
        batch = _make_batch(steps_data=[
            {"keyword": "Wait", "args": [], "timeout": "15s"},
        ])
        await runner.execute(batch)
        call_args = executor.execute_keyword.call_args
        assert call_args.kwargs.get("timeout") == "15s" or call_args[1].get("timeout") == "15s"

    @pytest.mark.asyncio
    async def test_assign_to_passed_to_executor(self):
        executor = AsyncMock()
        executor.execute_keyword = AsyncMock(return_value=_success_result("captured"))

        runner = BatchRunner(keyword_executor=executor)
        batch = _make_batch(steps_data=[
            {"keyword": "Get Text", "args": ["id=x"], "assign_to": "my_var"},
        ])
        await runner.execute(batch)
        call_args = executor.execute_keyword.call_args
        assert call_args.kwargs.get("assign_to") == "my_var"

    @pytest.mark.asyncio
    async def test_assign_to_none_when_not_specified(self):
        executor = AsyncMock()
        executor.execute_keyword = AsyncMock(return_value=_success_result())

        runner = BatchRunner(keyword_executor=executor)
        batch = _make_batch(steps_data=[
            {"keyword": "Log", "args": ["hello"]},
        ])
        await runner.execute(batch)
        call_args = executor.execute_keyword.call_args
        assert call_args.kwargs.get("assign_to") is None

    @pytest.mark.asyncio
    async def test_assign_to_passed_on_retry(self):
        """assign_to must be passed on retry calls too."""
        call_count = 0

        async def _execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return _fail_result("err")
            return _success_result("ok")

        executor = AsyncMock()
        executor.execute_keyword = AsyncMock(side_effect=_execute)

        runner = BatchRunner(keyword_executor=executor)
        batch = _make_batch(
            steps_data=[{"keyword": "Get Text", "args": ["id=x"], "assign_to": "val"}],
            on_failure="retry",
        )
        await runner.execute(batch)
        # Both the initial call and retry should have assign_to
        for call in executor.execute_keyword.call_args_list:
            assert call.kwargs.get("assign_to") == "val"


# ── BatchStateManager ────────────────────────────────────────────────


class TestBatchStateManager:
    def _make_state(self, batch_id_value="batch_abc"):
        """Create a minimal BatchState for testing."""
        return BatchState(
            batch_id=BatchId(value=batch_id_value),
            session_id="s1",
            original_steps=[BatchStep(index=0, keyword="A", args=[])],
            results=[],
            results_map={},
            failed_at_index=0,
            on_failure=OnFailurePolicy.RECOVER,
            max_recovery_attempts=RecoveryAttemptLimit.default(),
            timeout_ms=120000,
            elapsed_ms=1000.0,
        )

    def test_store_and_get(self):
        mgr = BatchStateManager()
        state = self._make_state()
        mgr.store(state)
        retrieved = mgr.get("batch_abc")
        assert retrieved is state

    def test_get_missing(self):
        mgr = BatchStateManager()
        assert mgr.get("nonexistent") is None

    def test_remove_existing(self):
        mgr = BatchStateManager()
        state = self._make_state()
        mgr.store(state)
        assert mgr.remove("batch_abc") is True
        assert mgr.get("batch_abc") is None

    def test_remove_missing(self):
        mgr = BatchStateManager()
        assert mgr.remove("nonexistent") is False

    def test_count(self):
        mgr = BatchStateManager()
        assert mgr.count == 0
        mgr.store(self._make_state("batch_1"))
        assert mgr.count == 1
        mgr.store(self._make_state("batch_2"))
        assert mgr.count == 2

    def test_ttl_expiry(self):
        mgr = BatchStateManager(ttl_seconds=1.0)
        state = self._make_state()
        state.created_at = datetime.now() - timedelta(seconds=2)
        mgr._states[state.batch_id.value] = state
        assert mgr.get("batch_abc") is None

    def test_max_states_eviction(self):
        mgr = BatchStateManager(max_states=2)
        s1 = self._make_state("batch_1")
        s1.created_at = datetime.now() - timedelta(seconds=10)
        s2 = self._make_state("batch_2")
        s2.created_at = datetime.now() - timedelta(seconds=5)
        mgr.store(s1)
        mgr.store(s2)
        # Third store should evict oldest
        s3 = self._make_state("batch_3")
        mgr.store(s3)
        assert mgr.count == 2
        assert mgr.get("batch_1") is None  # Oldest evicted

    def test_cleanup_expired_on_store(self):
        mgr = BatchStateManager(ttl_seconds=1.0, max_states=10)
        expired = self._make_state("batch_old")
        expired.created_at = datetime.now() - timedelta(seconds=2)
        mgr._states[expired.batch_id.value] = expired
        assert mgr.count == 1
        # Storing new state triggers cleanup
        mgr.store(self._make_state("batch_new"))
        assert mgr.get("batch_old") is None

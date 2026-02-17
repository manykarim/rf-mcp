"""Tests for batch_execution aggregates."""
import time

import pytest

from robotmcp.domains.batch_execution.aggregates import (
    BatchExecution,
    BatchState,
)
from robotmcp.domains.batch_execution.entities import (
    BatchStep,
    FailureDetail,
    RecoveryAttempt,
    StepResult,
)
from robotmcp.domains.batch_execution.value_objects import (
    BatchId,
    BatchStatus,
    BatchTimeout,
    OnFailurePolicy,
    RecoveryAttemptLimit,
    StepStatus,
    StepTimeout,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_batch(steps_data=None, **kwargs):
    """Shorthand factory for tests."""
    if steps_data is None:
        steps_data = [{"keyword": "Log", "args": ["hello"]}]
    defaults = {
        "session_id": "test-session",
        "steps_data": steps_data,
    }
    defaults.update(kwargs)
    return BatchExecution.create(**defaults)


def _make_step(index=0, keyword="Log", args=None, label=None, timeout=None):
    return BatchStep(
        index=index, keyword=keyword,
        args=args or [], label=label, timeout=timeout,
    )


# ── BatchExecution.create() ──────────────────────────────────────────


class TestBatchExecutionCreate:
    __test__ = True

    def test_create_minimal(self):
        batch = _make_batch()
        assert batch.session_id == "test-session"
        assert len(batch.steps) == 1
        assert batch.steps[0].keyword == "Log"
        assert batch.status is None

    def test_create_generates_batch_id(self):
        batch = _make_batch()
        assert batch.batch_id.value.startswith("batch_")

    def test_create_default_policy(self):
        batch = _make_batch()
        assert batch.on_failure == OnFailurePolicy.RECOVER

    def test_create_stop_policy(self):
        batch = _make_batch(on_failure="stop")
        assert batch.on_failure == OnFailurePolicy.STOP

    def test_create_retry_policy(self):
        batch = _make_batch(on_failure="retry")
        assert batch.on_failure == OnFailurePolicy.RETRY

    def test_create_default_recovery_attempts(self):
        batch = _make_batch()
        assert batch.max_recovery_attempts.value == 2

    def test_create_custom_recovery_attempts(self):
        batch = _make_batch(max_recovery_attempts=5)
        assert batch.max_recovery_attempts.value == 5

    def test_create_default_timeout(self):
        batch = _make_batch()
        assert batch.timeout.value_ms == 120000

    def test_create_custom_timeout(self):
        batch = _make_batch(timeout_ms=30000)
        assert batch.timeout.value_ms == 30000

    def test_create_multiple_steps(self):
        batch = _make_batch(steps_data=[
            {"keyword": "Log", "args": ["a"]},
            {"keyword": "Click", "args": ["btn"]},
            {"keyword": "Get Text", "args": ["id=x"]},
        ])
        assert len(batch.steps) == 3
        assert batch.steps[0].index == 0
        assert batch.steps[1].index == 1
        assert batch.steps[2].index == 2

    def test_create_with_step_label(self):
        batch = _make_batch(steps_data=[
            {"keyword": "Log", "args": ["x"], "label": "log step"},
        ])
        assert batch.steps[0].label == "log step"

    def test_create_with_step_timeout(self):
        batch = _make_batch(steps_data=[
            {"keyword": "Wait", "args": [], "timeout": "10s"},
        ])
        assert batch.steps[0].timeout.rf_format == "10s"

    def test_create_empty_steps_rejected(self):
        with pytest.raises(ValueError, match="At least one step"):
            _make_batch(steps_data=[])

    def test_create_step_without_args(self):
        batch = _make_batch(steps_data=[{"keyword": "No Operation"}])
        assert batch.steps[0].args == []

    def test_create_with_assign_to(self):
        batch = _make_batch(steps_data=[
            {"keyword": "Get Text", "args": ["id=x"], "assign_to": "my_var"},
        ])
        assert batch.steps[0].assign_to == "my_var"

    def test_create_without_assign_to(self):
        batch = _make_batch(steps_data=[
            {"keyword": "Log", "args": ["hello"]},
        ])
        assert batch.steps[0].assign_to is None

    def test_create_mixed_assign_to(self):
        batch = _make_batch(steps_data=[
            {"keyword": "Get Text", "args": ["id=x"], "assign_to": "val1"},
            {"keyword": "Log", "args": ["${val1}"]},
            {"keyword": "Get Text", "args": ["id=y"], "assign_to": "val2"},
        ])
        assert batch.steps[0].assign_to == "val1"
        assert batch.steps[1].assign_to is None
        assert batch.steps[2].assign_to == "val2"


# ── BatchExecution post_init ─────────────────────────────────────────


class TestBatchExecutionPostInit:
    __test__ = True

    def test_no_steps_raises(self):
        with pytest.raises(ValueError, match="at least 1 step"):
            BatchExecution(
                batch_id=BatchId.generate(),
                session_id="s1",
                steps=[],
            )


# ── Clock management ─────────────────────────────────────────────────


class TestBatchExecutionClock:
    __test__ = True

    def test_elapsed_ms_before_start(self):
        batch = _make_batch()
        assert batch.elapsed_ms == 0.0

    def test_start_clock(self):
        batch = _make_batch()
        batch.start_clock()
        time.sleep(0.01)
        assert batch.elapsed_ms > 0

    def test_is_timed_out_false(self):
        batch = _make_batch(timeout_ms=600000)
        batch.start_clock()
        assert not batch.is_timed_out()

    def test_is_timed_out_true(self):
        batch = _make_batch(timeout_ms=1000)
        batch._start_time = time.monotonic() - 2.0
        assert batch.is_timed_out()


# ── Argument resolution ──────────────────────────────────────────────


class TestBatchExecutionResolveArgs:
    __test__ = True

    def test_no_refs(self):
        batch = _make_batch()
        step = _make_step(args=["plain text"])
        result = batch.resolve_args(step)
        assert result == ["plain text"]

    def test_valid_back_ref(self):
        batch = _make_batch()
        batch.results_map[0] = "token123"
        step = _make_step(index=1, args=["Bearer ${STEP_0}"])
        result = batch.resolve_args(step)
        assert result == ["Bearer token123"]

    def test_multiple_refs(self):
        batch = _make_batch()
        batch.results_map[0] = "user"
        batch.results_map[1] = "pass"
        step = _make_step(index=2, args=["${STEP_0}:${STEP_1}"])
        result = batch.resolve_args(step)
        assert result == ["user:pass"]

    def test_forward_ref_rejected(self):
        batch = _make_batch()
        step = _make_step(index=0, args=["${STEP_1}"])
        with pytest.raises(ValueError, match="Forward reference"):
            batch.resolve_args(step)

    def test_self_ref_rejected(self):
        batch = _make_batch()
        step = _make_step(index=0, args=["${STEP_0}"])
        with pytest.raises(ValueError, match="Forward reference"):
            batch.resolve_args(step)

    def test_missing_result_ref(self):
        batch = _make_batch()
        step = _make_step(index=1, args=["${STEP_0}"])
        with pytest.raises(ValueError, match="not available"):
            batch.resolve_args(step)

    def test_no_args(self):
        batch = _make_batch()
        step = _make_step(args=[])
        assert batch.resolve_args(step) == []


# ── Result recording ─────────────────────────────────────────────────


class TestBatchExecutionRecording:
    __test__ = True

    def test_record_success(self):
        batch = _make_batch()
        step = batch.steps[0]
        batch.record_success(step, ["hello"], "ok", 42)
        assert len(batch.results) == 1
        assert batch.results[0].status == StepStatus.PASS
        assert batch.results[0].return_value == "ok"
        assert batch.results_map[0] == "ok"

    def test_record_success_none_return(self):
        batch = _make_batch()
        step = batch.steps[0]
        batch.record_success(step, [], None, 10)
        assert batch.results_map[0] is None

    def test_record_recovery(self):
        batch = _make_batch()
        step = batch.steps[0]
        attempts = [RecoveryAttempt(
            attempt_number=1, strategy="wait", tier=1,
            action_description="waited", result="PASS", time_ms=100,
        )]
        batch.record_recovery(step, ["a"], "val", 200, "orig error", attempts)
        assert len(batch.results) == 1
        assert batch.results[0].status == StepStatus.RECOVERED
        assert batch.results[0].error == "orig error"
        assert batch.results_map[0] == "val"

    def test_record_failure(self):
        batch = _make_batch()
        step = batch.steps[0]
        detail = FailureDetail(step_index=0, error="not found")
        batch.record_failure(step, ["a"], "not found", 50, detail)
        assert batch.status == BatchStatus.FAIL
        assert batch.failure is detail
        assert len(batch.results) == 1
        assert batch.results[0].status == StepStatus.FAIL

    def test_record_timeout(self):
        batch = _make_batch()
        batch.record_timeout()
        assert batch.status == BatchStatus.TIMEOUT


# ── Finalize ─────────────────────────────────────────────────────────


class TestBatchExecutionFinalize:
    __test__ = True

    def test_finalize_all_pass(self):
        batch = _make_batch()
        step = batch.steps[0]
        batch.record_success(step, [], None, 10)
        batch.finalize()
        assert batch.status == BatchStatus.PASS

    def test_finalize_recovered(self):
        batch = _make_batch()
        step = batch.steps[0]
        batch.record_recovery(step, [], "v", 10, "err", [])
        batch.finalize()
        assert batch.status == BatchStatus.RECOVERED

    def test_finalize_fail_already_set(self):
        batch = _make_batch()
        step = batch.steps[0]
        detail = FailureDetail(step_index=0, error="err")
        batch.record_failure(step, [], "err", 0, detail)
        batch.finalize()
        assert batch.status == BatchStatus.FAIL

    def test_finalize_timeout_already_set(self):
        batch = _make_batch()
        batch.record_timeout()
        batch.finalize()
        assert batch.status == BatchStatus.TIMEOUT

    def test_finalize_idempotent(self):
        batch = _make_batch()
        step = batch.steps[0]
        batch.record_success(step, [], None, 10)
        batch.finalize()
        batch.finalize()
        assert batch.status == BatchStatus.PASS


# ── to_response_dict ─────────────────────────────────────────────────


class TestBatchExecutionResponseDict:
    __test__ = True

    def test_pass_response(self):
        batch = _make_batch()
        batch.start_clock()
        step = batch.steps[0]
        batch.record_success(step, ["hello"], "ok", 10)
        batch.finalize()
        d = batch.to_response_dict()
        assert d["status"] == "PASS"
        assert d["steps_executed"] == 1
        assert d["steps_total"] == 1
        assert "failure" not in d

    def test_fail_response_has_batch_id(self):
        batch = _make_batch()
        batch.start_clock()
        step = batch.steps[0]
        detail = FailureDetail(step_index=0, error="bad")
        batch.record_failure(step, [], "bad", 10, detail)
        batch.finalize()
        d = batch.to_response_dict()
        assert d["status"] == "FAIL"
        assert d["batch_id"] == batch.batch_id.value
        assert d["failure"]["error"] == "bad"

    def test_recovered_response(self):
        batch = _make_batch()
        batch.start_clock()
        step = batch.steps[0]
        batch.record_recovery(step, [], "v", 10, "err", [])
        batch.finalize()
        d = batch.to_response_dict()
        assert d["status"] == "RECOVERED"
        assert "1 recovered" in d["summary"]

    def test_timeout_response(self):
        batch = _make_batch()
        batch.start_clock()
        batch.record_timeout()
        batch.finalize()
        d = batch.to_response_dict()
        assert d["status"] == "TIMEOUT"
        assert "Timeout" in d["summary"]

    def test_unknown_status(self):
        batch = _make_batch()
        batch.start_clock()
        d = batch.to_response_dict()
        assert d["status"] == "UNKNOWN"


# ── _build_summary ───────────────────────────────────────────────────


class TestBatchExecutionBuildSummary:
    __test__ = True

    def test_pass_summary(self):
        batch = _make_batch()
        step = batch.steps[0]
        batch.record_success(step, [], None, 0)
        batch.finalize()
        s = batch._build_summary(1, 1)
        assert s == "1/1 passed"

    def test_recovered_summary(self):
        batch = _make_batch(steps_data=[
            {"keyword": "A", "args": []},
            {"keyword": "B", "args": []},
        ])
        batch.record_success(batch.steps[0], [], None, 0)
        batch.record_recovery(batch.steps[1], [], "v", 0, "e", [])
        batch.finalize()
        s = batch._build_summary(2, 2)
        assert "1 recovered" in s

    def test_fail_summary(self):
        batch = _make_batch()
        step = batch.steps[0]
        detail = FailureDetail(step_index=0, error="err")
        batch.record_failure(step, [], "err", 0, detail)
        batch.finalize()
        s = batch._build_summary(1, 1)
        assert "step 0 failed" in s

    def test_timeout_summary(self):
        batch = _make_batch()
        batch.start_clock()
        batch.record_timeout()
        batch.finalize()
        s = batch._build_summary(0, 1)
        assert "Timeout" in s


# ── BatchState ───────────────────────────────────────────────────────


class TestBatchState:
    __test__ = True

    def test_from_execution_with_failure(self):
        batch = _make_batch(steps_data=[
            {"keyword": "A", "args": []},
            {"keyword": "B", "args": []},
            {"keyword": "C", "args": []},
        ])
        batch.start_clock()
        batch.record_success(batch.steps[0], [], "val0", 10)
        detail = FailureDetail(step_index=1, error="err")
        batch.record_failure(batch.steps[1], [], "err", 20, detail)

        state = BatchState.from_execution(batch)
        assert state.batch_id == batch.batch_id
        assert state.session_id == batch.session_id
        assert state.failed_at_index == 1
        assert len(state.results) == 2
        assert state.results_map[0] == "val0"

    def test_from_execution_no_failure_raises(self):
        batch = _make_batch()
        with pytest.raises(ValueError, match="non-failed"):
            BatchState.from_execution(batch)

    def test_remaining_timeout_ms(self):
        batch = _make_batch(timeout_ms=10000)
        batch.start_clock()
        detail = FailureDetail(step_index=0, error="err")
        batch.record_failure(batch.steps[0], [], "err", 0, detail)
        state = BatchState.from_execution(batch)
        assert state.remaining_timeout_ms >= 0
        assert state.remaining_timeout_ms <= 10000

    def test_remaining_steps(self):
        batch = _make_batch(steps_data=[
            {"keyword": "A", "args": []},
            {"keyword": "B", "args": []},
            {"keyword": "C", "args": []},
        ])
        batch.start_clock()
        batch.record_success(batch.steps[0], [], None, 0)
        detail = FailureDetail(step_index=1, error="err")
        batch.record_failure(batch.steps[1], [], "err", 0, detail)
        state = BatchState.from_execution(batch)
        remaining = state.remaining_steps
        assert len(remaining) == 1
        assert remaining[0].index == 2
        assert remaining[0].keyword == "C"

    def test_failed_step(self):
        batch = _make_batch(steps_data=[
            {"keyword": "A", "args": []},
            {"keyword": "B", "args": []},
        ])
        batch.start_clock()
        detail = FailureDetail(step_index=0, error="err")
        batch.record_failure(batch.steps[0], [], "err", 0, detail)
        state = BatchState.from_execution(batch)
        assert state.failed_step.keyword == "A"

    def test_failed_step_not_found(self):
        batch = _make_batch()
        batch.start_clock()
        detail = FailureDetail(step_index=0, error="err")
        batch.record_failure(batch.steps[0], [], "err", 0, detail)
        state = BatchState.from_execution(batch)
        # Mutate the original_steps to force not-found
        state.original_steps = []
        assert state.failed_step is None

    def test_is_expired_false(self):
        batch = _make_batch()
        batch.start_clock()
        detail = FailureDetail(step_index=0, error="err")
        batch.record_failure(batch.steps[0], [], "err", 0, detail)
        state = BatchState.from_execution(batch)
        assert not state.is_expired(ttl_seconds=300)

    def test_is_expired_true(self):
        from datetime import datetime, timedelta

        batch = _make_batch()
        batch.start_clock()
        detail = FailureDetail(step_index=0, error="err")
        batch.record_failure(batch.steps[0], [], "err", 0, detail)
        state = BatchState.from_execution(batch)
        state.created_at = datetime.now() - timedelta(seconds=600)
        assert state.is_expired(ttl_seconds=300)

    def test_remaining_timeout_ms_already_elapsed(self):
        batch = _make_batch(timeout_ms=1000)
        batch._start_time = time.monotonic() - 2.0
        detail = FailureDetail(step_index=0, error="err")
        batch.record_failure(batch.steps[0], [], "err", 0, detail)
        state = BatchState.from_execution(batch)
        assert state.remaining_timeout_ms == 0

"""Tests for batch_execution events."""
from datetime import datetime

import pytest

from robotmcp.domains.batch_execution.events import (
    BatchCompleted,
    BatchFailed,
    BatchResumed,
    BatchStarted,
    BatchTimedOut,
    StepExecuted,
    StepFailed,
    StepRecovered,
)


class TestBatchStarted:
    def test_creation(self):
        e = BatchStarted(
            batch_id="b1", session_id="s1", step_count=3,
            on_failure="recover", timeout_ms=120000,
        )
        assert e.batch_id == "b1"
        assert e.step_count == 3

    def test_frozen(self):
        e = BatchStarted(
            batch_id="b1", session_id="s1", step_count=1,
            on_failure="stop", timeout_ms=5000,
        )
        with pytest.raises(AttributeError):
            e.batch_id = "other"

    def test_to_dict(self):
        e = BatchStarted(
            batch_id="b1", session_id="s1", step_count=2,
            on_failure="recover", timeout_ms=120000,
        )
        d = e.to_dict()
        assert d["event_type"] == "batch_started"
        assert d["batch_id"] == "b1"
        assert d["session_id"] == "s1"
        assert d["step_count"] == 2
        assert d["timeout_ms"] == 120000

    def test_timestamp_auto(self):
        e = BatchStarted(
            batch_id="b1", session_id="s1", step_count=1,
            on_failure="stop", timeout_ms=1000,
        )
        assert isinstance(e.timestamp, datetime)


class TestStepExecuted:
    def test_to_dict(self):
        e = StepExecuted(
            batch_id="b1", step_index=0, keyword="Log", time_ms=42,
        )
        d = e.to_dict()
        assert d["event_type"] == "step_executed"
        assert d["step_index"] == 0
        assert d["keyword"] == "Log"
        assert "return_value" not in d

    def test_to_dict_with_return_value(self):
        e = StepExecuted(
            batch_id="b1", step_index=0, keyword="Get Text",
            time_ms=10, return_value="hello",
        )
        d = e.to_dict()
        assert d["return_value"] == "hello"

    def test_frozen(self):
        e = StepExecuted(
            batch_id="b1", step_index=0, keyword="Log", time_ms=0,
        )
        with pytest.raises(AttributeError):
            e.keyword = "other"


class TestStepFailed:
    def test_to_dict(self):
        e = StepFailed(
            batch_id="b1", step_index=1, keyword="Click", error="not found",
        )
        d = e.to_dict()
        assert d["event_type"] == "step_failed"
        assert d["error"] == "not found"

    def test_frozen(self):
        e = StepFailed(
            batch_id="b1", step_index=0, keyword="K", error="e",
        )
        with pytest.raises(AttributeError):
            e.error = "other"


class TestStepRecovered:
    def test_to_dict(self):
        e = StepRecovered(
            batch_id="b1", step_index=0, keyword="Click",
            strategy="wait_and_retry", attempt_number=1, time_ms=2000,
        )
        d = e.to_dict()
        assert d["event_type"] == "step_recovered"
        assert d["strategy"] == "wait_and_retry"
        assert d["attempt_number"] == 1


class TestBatchCompleted:
    def test_to_dict(self):
        e = BatchCompleted(
            batch_id="b1", status="PASS",
            steps_executed=3, steps_total=3, total_time_ms=500,
        )
        d = e.to_dict()
        assert d["event_type"] == "batch_completed"
        assert d["status"] == "PASS"
        assert d["steps_executed"] == 3

    def test_frozen(self):
        e = BatchCompleted(
            batch_id="b1", status="PASS",
            steps_executed=1, steps_total=1, total_time_ms=10,
        )
        with pytest.raises(AttributeError):
            e.status = "FAIL"


class TestBatchFailed:
    def test_to_dict(self):
        e = BatchFailed(
            batch_id="b1", failed_step_index=2, error="err",
            steps_completed=2, steps_total=5,
        )
        d = e.to_dict()
        assert d["event_type"] == "batch_failed"
        assert d["failed_step_index"] == 2
        assert d["steps_completed"] == 2


class TestBatchTimedOut:
    def test_to_dict(self):
        e = BatchTimedOut(
            batch_id="b1", steps_completed=1, steps_total=3, elapsed_ms=120000,
        )
        d = e.to_dict()
        assert d["event_type"] == "batch_timed_out"
        assert d["elapsed_ms"] == 120000

    def test_frozen(self):
        e = BatchTimedOut(
            batch_id="b1", steps_completed=0, steps_total=1, elapsed_ms=1000,
        )
        with pytest.raises(AttributeError):
            e.elapsed_ms = 0


class TestBatchResumed:
    def test_to_dict(self):
        e = BatchResumed(
            batch_id="b1", session_id="s1",
            resumed_from_index=2, fix_steps_count=1,
        )
        d = e.to_dict()
        assert d["event_type"] == "batch_resumed"
        assert d["resumed_from_index"] == 2
        assert d["fix_steps_count"] == 1

    def test_frozen(self):
        e = BatchResumed(
            batch_id="b1", session_id="s1",
            resumed_from_index=0, fix_steps_count=0,
        )
        with pytest.raises(AttributeError):
            e.batch_id = "other"

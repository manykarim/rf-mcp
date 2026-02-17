"""Tests for batch_execution entities."""
import pytest

from robotmcp.domains.batch_execution.entities import (
    BatchStep,
    FailureDetail,
    RecoveryAttempt,
    StepResult,
)
from robotmcp.domains.batch_execution.value_objects import StepStatus, StepTimeout


# ── BatchStep ────────────────────────────────────────────────────────


class TestBatchStep:
    __test__ = True  # Override the entity's __test__ = False

    def test_creation_minimal(self):
        step = BatchStep(index=0, keyword="Log", args=["hello"])
        assert step.index == 0
        assert step.keyword == "Log"
        assert step.args == ["hello"]
        assert step.label is None
        assert step.timeout is None

    def test_creation_with_label(self):
        step = BatchStep(index=1, keyword="Click", args=["btn"], label="click login")
        assert step.label == "click login"

    def test_creation_with_timeout(self):
        timeout = StepTimeout(rf_format="10s")
        step = BatchStep(index=0, keyword="Wait", args=[], timeout=timeout)
        assert step.timeout is timeout

    def test_display_name_with_label(self):
        step = BatchStep(index=0, keyword="Log", args=["x"], label="my label")
        assert step.display_name == "my label"

    def test_display_name_without_label_no_args(self):
        step = BatchStep(index=0, keyword="Log", args=[])
        assert step.display_name == "Log()"

    def test_display_name_without_label_one_arg(self):
        step = BatchStep(index=0, keyword="Log", args=["hello"])
        assert step.display_name == "Log(hello)"

    def test_display_name_without_label_two_args(self):
        step = BatchStep(index=0, keyword="Click", args=["btn", "timeout=5s"])
        assert step.display_name == "Click(btn, timeout=5s)"

    def test_display_name_without_label_three_args_truncated(self):
        step = BatchStep(index=0, keyword="Kw", args=["a", "b", "c"])
        assert step.display_name == "Kw(a, b...)"

    def test_to_dict_minimal(self):
        step = BatchStep(index=0, keyword="Log", args=["hello"])
        d = step.to_dict()
        assert d == {"index": 0, "keyword": "Log", "args": ["hello"]}
        assert "label" not in d
        assert "timeout" not in d

    def test_to_dict_with_label(self):
        step = BatchStep(index=0, keyword="Log", args=[], label="my step")
        d = step.to_dict()
        assert d["label"] == "my step"

    def test_to_dict_with_timeout(self):
        step = BatchStep(
            index=0, keyword="Wait", args=[],
            timeout=StepTimeout(rf_format="5s"),
        )
        d = step.to_dict()
        assert d["timeout"] == "5s"

    def test_to_dict_args_is_copy(self):
        args = ["a", "b"]
        step = BatchStep(index=0, keyword="Kw", args=args)
        d = step.to_dict()
        d["args"].append("c")
        assert step.args == ["a", "b"]

    def test_creation_with_assign_to(self):
        step = BatchStep(index=0, keyword="Get Text", args=["id=x"], assign_to="my_var")
        assert step.assign_to == "my_var"

    def test_creation_without_assign_to(self):
        step = BatchStep(index=0, keyword="Log", args=["hello"])
        assert step.assign_to is None

    def test_to_dict_with_assign_to(self):
        step = BatchStep(index=0, keyword="Get Text", args=["id=x"], assign_to="result")
        d = step.to_dict()
        assert d["assign_to"] == "result"

    def test_to_dict_without_assign_to(self):
        step = BatchStep(index=0, keyword="Log", args=["x"])
        d = step.to_dict()
        assert "assign_to" not in d


# ── StepResult ───────────────────────────────────────────────────────


class TestStepResult:
    __test__ = True

    def test_creation_pass(self):
        r = StepResult(
            index=0, keyword="Log", args_resolved=["hi"],
            status=StepStatus.PASS, time_ms=42,
        )
        assert r.status == StepStatus.PASS
        assert r.return_value is None
        assert r.error is None

    def test_creation_fail_with_error(self):
        r = StepResult(
            index=1, keyword="Click", args_resolved=["btn"],
            status=StepStatus.FAIL, error="Not found", time_ms=100,
        )
        assert r.error == "Not found"

    def test_to_dict_pass(self):
        r = StepResult(
            index=0, keyword="Log", args_resolved=["hi"],
            status=StepStatus.PASS, time_ms=10,
        )
        d = r.to_dict()
        assert d["status"] == "PASS"
        assert d["time_ms"] == 10
        assert "return_value" not in d
        assert "error" not in d

    def test_to_dict_with_return_value(self):
        r = StepResult(
            index=0, keyword="Get Text", args_resolved=["id=foo"],
            status=StepStatus.PASS, return_value="bar", time_ms=5,
        )
        d = r.to_dict()
        assert d["return_value"] == "bar"

    def test_to_dict_with_error(self):
        r = StepResult(
            index=0, keyword="Click", args_resolved=["btn"],
            status=StepStatus.FAIL, error="not found", time_ms=50,
        )
        d = r.to_dict()
        assert d["error"] == "not found"

    def test_to_dict_with_label(self):
        r = StepResult(
            index=0, keyword="Log", args_resolved=["x"],
            status=StepStatus.PASS, label="step one", time_ms=1,
        )
        d = r.to_dict()
        assert d["label"] == "step one"

    def test_to_dict_fail_status(self):
        r = StepResult(
            index=0, keyword="Click", args_resolved=[],
            status=StepStatus.FAIL, time_ms=0,
        )
        assert r.to_dict()["status"] == "FAIL"

    def test_to_dict_recovered_status(self):
        r = StepResult(
            index=0, keyword="Click", args_resolved=[],
            status=StepStatus.RECOVERED, time_ms=0,
        )
        assert r.to_dict()["status"] == "RECOVERED"

    def test_to_dict_skipped_status(self):
        r = StepResult(
            index=0, keyword="Click", args_resolved=[],
            status=StepStatus.SKIPPED, time_ms=0,
        )
        assert r.to_dict()["status"] == "SKIPPED"


# ── RecoveryAttempt ──────────────────────────────────────────────────


class TestRecoveryAttempt:
    __test__ = True

    def test_creation(self):
        a = RecoveryAttempt(
            attempt_number=1, strategy="wait_and_retry", tier=1,
            action_description="Wait 2s", result="PASS", time_ms=2000,
        )
        assert a.attempt_number == 1
        assert a.strategy == "wait_and_retry"
        assert a.tier == 1
        assert a.result == "PASS"

    def test_to_dict(self):
        a = RecoveryAttempt(
            attempt_number=2, strategy="reload", tier=2,
            action_description="Reload page", result="FAIL", time_ms=3000,
        )
        d = a.to_dict()
        assert d == {
            "attempt": 2,
            "strategy": "reload",
            "tier": 2,
            "action": "Reload page",
            "result": "FAIL",
            "time_ms": 3000,
        }

    def test_default_time_ms(self):
        a = RecoveryAttempt(
            attempt_number=1, strategy="s", tier=1,
            action_description="d", result="PASS",
        )
        assert a.time_ms == 0


# ── FailureDetail ────────────────────────────────────────────────────


class TestFailureDetail:
    __test__ = True

    def test_creation_minimal(self):
        f = FailureDetail(step_index=2, error="Element not found")
        assert f.step_index == 2
        assert f.error == "Element not found"
        assert f.screenshot_base64 is None
        assert f.page_source_snippet is None
        assert f.current_url is None
        assert f.page_title is None
        assert f.recovery_log == []

    def test_creation_full(self):
        f = FailureDetail(
            step_index=0, error="err",
            screenshot_base64="abc=", page_source_snippet="<html>",
            current_url="http://x", page_title="Page",
        )
        assert f.screenshot_base64 == "abc="
        assert f.current_url == "http://x"

    def test_to_dict_minimal(self):
        f = FailureDetail(step_index=1, error="err")
        d = f.to_dict()
        assert d == {"step_index": 1, "error": "err"}
        assert "screenshot_base64" not in d
        assert "recovery_log" not in d

    def test_to_dict_with_screenshot(self):
        f = FailureDetail(step_index=0, error="e", screenshot_base64="x==")
        d = f.to_dict()
        assert d["screenshot_base64"] == "x=="

    def test_to_dict_with_page_source(self):
        f = FailureDetail(step_index=0, error="e", page_source_snippet="<p>")
        d = f.to_dict()
        assert d["page_source_snippet"] == "<p>"

    def test_to_dict_with_url(self):
        f = FailureDetail(step_index=0, error="e", current_url="http://x")
        assert f.to_dict()["current_url"] == "http://x"

    def test_to_dict_with_page_title(self):
        f = FailureDetail(step_index=0, error="e", page_title="Title")
        assert f.to_dict()["page_title"] == "Title"

    def test_to_dict_with_recovery_log(self):
        attempt = RecoveryAttempt(
            attempt_number=1, strategy="wait", tier=1,
            action_description="waited", result="FAIL", time_ms=100,
        )
        f = FailureDetail(step_index=0, error="e", recovery_log=[attempt])
        d = f.to_dict()
        assert len(d["recovery_log"]) == 1
        assert d["recovery_log"][0]["attempt"] == 1

    def test_to_dict_empty_recovery_log_omitted(self):
        f = FailureDetail(step_index=0, error="e", recovery_log=[])
        d = f.to_dict()
        assert "recovery_log" not in d

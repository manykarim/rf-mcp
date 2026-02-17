"""Tests for batch_execution value objects."""
import re

import pytest

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


# ── BatchStatus ──────────────────────────────────────────────────────


class TestBatchStatus:
    def test_pass_value(self):
        assert BatchStatus.PASS.value == "PASS"

    def test_fail_value(self):
        assert BatchStatus.FAIL.value == "FAIL"

    def test_recovered_value(self):
        assert BatchStatus.RECOVERED.value == "RECOVERED"

    def test_timeout_value(self):
        assert BatchStatus.TIMEOUT.value == "TIMEOUT"

    def test_is_str_enum(self):
        assert isinstance(BatchStatus.PASS, str)

    def test_string_comparison(self):
        assert BatchStatus.PASS == "PASS"
        assert BatchStatus.FAIL == "FAIL"

    def test_all_members(self):
        members = {m.value for m in BatchStatus}
        assert members == {"PASS", "FAIL", "RECOVERED", "TIMEOUT"}


# ── StepStatus ───────────────────────────────────────────────────────


class TestStepStatus:
    def test_pass_value(self):
        assert StepStatus.PASS.value == "PASS"

    def test_fail_value(self):
        assert StepStatus.FAIL.value == "FAIL"

    def test_recovered_value(self):
        assert StepStatus.RECOVERED.value == "RECOVERED"

    def test_skipped_value(self):
        assert StepStatus.SKIPPED.value == "SKIPPED"

    def test_is_str_enum(self):
        assert isinstance(StepStatus.SKIPPED, str)
        assert StepStatus.SKIPPED == "SKIPPED"

    def test_all_members(self):
        members = {m.value for m in StepStatus}
        assert members == {"PASS", "FAIL", "RECOVERED", "SKIPPED"}


# ── OnFailurePolicy ─────────────────────────────────────────────────


class TestOnFailurePolicy:
    def test_stop_value(self):
        assert OnFailurePolicy.STOP.value == "stop"

    def test_retry_value(self):
        assert OnFailurePolicy.RETRY.value == "retry"

    def test_recover_value(self):
        assert OnFailurePolicy.RECOVER.value == "recover"

    def test_is_str_enum(self):
        assert isinstance(OnFailurePolicy.STOP, str)
        assert OnFailurePolicy.STOP == "stop"

    def test_from_string(self):
        assert OnFailurePolicy("stop") == OnFailurePolicy.STOP
        assert OnFailurePolicy("retry") == OnFailurePolicy.RETRY
        assert OnFailurePolicy("recover") == OnFailurePolicy.RECOVER


# ── BatchId ──────────────────────────────────────────────────────────


class TestBatchId:
    def test_create_with_value(self):
        bid = BatchId(value="batch_abc123def456")
        assert bid.value == "batch_abc123def456"

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            BatchId(value="")

    def test_generate_produces_prefix(self):
        bid = BatchId.generate()
        assert bid.value.startswith("batch_")

    def test_generate_has_12_hex_chars(self):
        bid = BatchId.generate()
        suffix = bid.value[len("batch_"):]
        assert len(suffix) == 12
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_generate_unique(self):
        ids = {BatchId.generate().value for _ in range(50)}
        assert len(ids) == 50

    def test_frozen(self):
        bid = BatchId(value="batch_abc")
        with pytest.raises(AttributeError):
            bid.value = "other"

    def test_equality(self):
        a = BatchId(value="batch_abc")
        b = BatchId(value="batch_abc")
        assert a == b

    def test_inequality(self):
        a = BatchId(value="batch_abc")
        b = BatchId(value="batch_xyz")
        assert a != b


# ── StepReference ────────────────────────────────────────────────────


class TestStepReference:
    def test_create(self):
        ref = StepReference(index=0, raw="${STEP_0}")
        assert ref.index == 0
        assert ref.raw == "${STEP_0}"

    def test_negative_index_rejected(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            StepReference(index=-1, raw="${STEP_-1}")

    def test_frozen(self):
        ref = StepReference(index=0, raw="${STEP_0}")
        with pytest.raises(AttributeError):
            ref.index = 1

    def test_find_all_no_refs(self):
        refs = StepReference.find_all("hello world")
        assert refs == []

    def test_find_all_single_ref(self):
        refs = StepReference.find_all("Login as ${STEP_0}")
        assert len(refs) == 1
        assert refs[0].index == 0
        assert refs[0].raw == "${STEP_0}"

    def test_find_all_two_refs(self):
        refs = StepReference.find_all("Login ${STEP_0} with ${STEP_1}")
        assert len(refs) == 2
        assert refs[0].index == 0
        assert refs[1].index == 1

    def test_find_all_preserves_order(self):
        refs = StepReference.find_all("${STEP_3} then ${STEP_1}")
        assert refs[0].index == 3
        assert refs[1].index == 1

    def test_pattern_is_compiled_regex(self):
        assert isinstance(StepReference.PATTERN, re.Pattern)

    def test_pattern_matches_step_n(self):
        assert StepReference.PATTERN.search("${STEP_0}")
        assert StepReference.PATTERN.search("${STEP_99}")

    def test_pattern_no_match_wrong_format(self):
        assert StepReference.PATTERN.search("$STEP_0") is None
        assert StepReference.PATTERN.search("{STEP_0}") is None


# ── BatchTimeout ─────────────────────────────────────────────────────


class TestBatchTimeout:
    def test_valid_min(self):
        t = BatchTimeout(value_ms=1000)
        assert t.value_ms == 1000

    def test_valid_max(self):
        t = BatchTimeout(value_ms=600000)
        assert t.value_ms == 600000

    def test_valid_mid(self):
        t = BatchTimeout(value_ms=30000)
        assert t.value_ms == 30000

    def test_below_min(self):
        with pytest.raises(ValueError, match="1000-600000"):
            BatchTimeout(value_ms=999)

    def test_above_max(self):
        with pytest.raises(ValueError, match="1000-600000"):
            BatchTimeout(value_ms=600001)

    def test_zero_rejected(self):
        with pytest.raises(ValueError):
            BatchTimeout(value_ms=0)

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            BatchTimeout(value_ms=-1)

    def test_default_is_120000(self):
        t = BatchTimeout.default()
        assert t.value_ms == 120000

    def test_frozen(self):
        t = BatchTimeout(value_ms=5000)
        with pytest.raises(AttributeError):
            t.value_ms = 6000

    def test_class_constants(self):
        assert BatchTimeout.MIN_MS == 1000
        assert BatchTimeout.MAX_MS == 600000
        assert BatchTimeout.DEFAULT_MS == 120000


# ── StepTimeout ──────────────────────────────────────────────────────


class TestStepTimeout:
    def test_seconds_short(self):
        t = StepTimeout(rf_format="10s")
        assert t.rf_format == "10s"

    def test_seconds_long(self):
        t = StepTimeout(rf_format="10 seconds")
        assert t.rf_format == "10 seconds"

    def test_seconds_singular(self):
        t = StepTimeout(rf_format="1 second")
        assert t.rf_format == "1 second"

    def test_minutes_short(self):
        t = StepTimeout(rf_format="1.5m")
        assert t.rf_format == "1.5m"

    def test_minutes_long(self):
        t = StepTimeout(rf_format="2 minutes")
        assert t.rf_format == "2 minutes"

    def test_minutes_singular(self):
        t = StepTimeout(rf_format="1 minute")
        assert t.rf_format == "1 minute"

    def test_milliseconds_short(self):
        t = StepTimeout(rf_format="500ms")
        assert t.rf_format == "500ms"

    def test_milliseconds_long(self):
        t = StepTimeout(rf_format="500 milliseconds")
        assert t.rf_format == "500 milliseconds"

    def test_hours_short(self):
        t = StepTimeout(rf_format="1h")
        assert t.rf_format == "1h"

    def test_hours_long(self):
        t = StepTimeout(rf_format="2 hours")
        assert t.rf_format == "2 hours"

    def test_sec_abbreviation(self):
        t = StepTimeout(rf_format="5sec")
        assert t.rf_format == "5sec"

    def test_min_abbreviation(self):
        t = StepTimeout(rf_format="3min")
        assert t.rf_format == "3min"

    def test_decimal_seconds(self):
        t = StepTimeout(rf_format="1.5s")
        assert t.rf_format == "1.5s"

    def test_invalid_format_no_unit(self):
        with pytest.raises(ValueError, match="Invalid RF duration"):
            StepTimeout(rf_format="10")

    def test_invalid_format_garbage(self):
        with pytest.raises(ValueError, match="Invalid RF duration"):
            StepTimeout(rf_format="abc")

    def test_invalid_format_empty(self):
        with pytest.raises(ValueError, match="Invalid RF duration"):
            StepTimeout(rf_format="")

    def test_frozen(self):
        t = StepTimeout(rf_format="10s")
        with pytest.raises(AttributeError):
            t.rf_format = "20s"

    def test_whitespace_in_format(self):
        t = StepTimeout(rf_format="  10s  ")
        assert t.rf_format == "  10s  "


# ── RecoveryAttemptLimit ─────────────────────────────────────────────


class TestRecoveryAttemptLimit:
    def test_valid_min(self):
        r = RecoveryAttemptLimit(value=1)
        assert r.value == 1

    def test_valid_max(self):
        r = RecoveryAttemptLimit(value=10)
        assert r.value == 10

    def test_valid_mid(self):
        r = RecoveryAttemptLimit(value=5)
        assert r.value == 5

    def test_below_min(self):
        with pytest.raises(ValueError, match="1-10"):
            RecoveryAttemptLimit(value=0)

    def test_above_max(self):
        with pytest.raises(ValueError, match="1-10"):
            RecoveryAttemptLimit(value=11)

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            RecoveryAttemptLimit(value=-1)

    def test_default_is_2(self):
        r = RecoveryAttemptLimit.default()
        assert r.value == 2

    def test_frozen(self):
        r = RecoveryAttemptLimit(value=3)
        with pytest.raises(AttributeError):
            r.value = 4

    def test_class_constants(self):
        assert RecoveryAttemptLimit.MIN == 1
        assert RecoveryAttemptLimit.MAX == 10
        assert RecoveryAttemptLimit.DEFAULT == 2

"""Tests for recovery domain events."""
from datetime import datetime

import pytest

from robotmcp.domains.recovery.events import (
    ErrorClassified,
    EvidenceCollected,
    RecoveryAttempted,
    RecoveryFailed,
    RecoveryStrategySelected,
    RecoverySucceeded,
)


class TestErrorClassified:
    def test_creation(self):
        e = ErrorClassified(
            error_message="not found", classification="ElementNotFound",
        )
        assert e.error_message == "not found"
        assert e.classification == "ElementNotFound"

    def test_frozen(self):
        e = ErrorClassified(error_message="x", classification="Unknown")
        with pytest.raises(AttributeError):
            e.classification = "other"

    def test_to_dict(self):
        e = ErrorClassified(
            error_message="err", classification="TimeoutException",
        )
        d = e.to_dict()
        assert d["event_type"] == "error_classified"
        assert d["error_message"] == "err"
        assert d["classification"] == "TimeoutException"

    def test_timestamp_auto(self):
        e = ErrorClassified(error_message="x", classification="Unknown")
        assert isinstance(e.timestamp, datetime)


class TestRecoveryStrategySelected:
    def test_to_dict(self):
        e = RecoveryStrategySelected(
            plan_id="p1", classification="ElementNotFound",
            strategy="wait_and_retry", tier=1,
        )
        d = e.to_dict()
        assert d["event_type"] == "recovery_strategy_selected"
        assert d["plan_id"] == "p1"
        assert d["tier"] == 1

    def test_frozen(self):
        e = RecoveryStrategySelected(
            plan_id="p1", classification="X", strategy="s", tier=1,
        )
        with pytest.raises(AttributeError):
            e.tier = 2


class TestRecoveryAttempted:
    def test_to_dict(self):
        e = RecoveryAttempted(
            plan_id="p1", strategy="wait", tier=1, actions_count=2,
        )
        d = e.to_dict()
        assert d["event_type"] == "recovery_attempted"
        assert d["actions_count"] == 2


class TestRecoverySucceeded:
    def test_to_dict(self):
        e = RecoverySucceeded(
            plan_id="p1", strategy="wait", tier=1, total_time_ms=3000,
        )
        d = e.to_dict()
        assert d["event_type"] == "recovery_succeeded"
        assert d["total_time_ms"] == 3000


class TestRecoveryFailed:
    def test_to_dict(self):
        e = RecoveryFailed(
            plan_id="p1", classification="ElementNotFound",
            strategies_tried=["wait", "reload"],
        )
        d = e.to_dict()
        assert d["event_type"] == "recovery_failed"
        assert d["strategies_tried"] == ["wait", "reload"]

    def test_to_dict_empty_strategies(self):
        e = RecoveryFailed(
            plan_id="p1", classification="Unknown",
        )
        d = e.to_dict()
        assert d["strategies_tried"] == []

    def test_frozen(self):
        e = RecoveryFailed(plan_id="p1", classification="X")
        with pytest.raises(AttributeError):
            e.plan_id = "other"


class TestEvidenceCollected:
    def test_to_dict_with_url(self):
        e = EvidenceCollected(
            plan_id="p1", has_screenshot=True, has_page_source=True,
            current_url="http://x.com",
        )
        d = e.to_dict()
        assert d["event_type"] == "evidence_collected"
        assert d["has_screenshot"] is True
        assert d["has_page_source"] is True
        assert d["current_url"] == "http://x.com"

    def test_to_dict_without_url(self):
        e = EvidenceCollected(
            plan_id="p1", has_screenshot=False, has_page_source=False,
        )
        d = e.to_dict()
        assert "current_url" not in d

    def test_frozen(self):
        e = EvidenceCollected(
            plan_id="p1", has_screenshot=False, has_page_source=False,
        )
        with pytest.raises(AttributeError):
            e.has_screenshot = True

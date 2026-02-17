"""Tests for recovery domain value objects."""
import re

import pytest

from robotmcp.domains.recovery.value_objects import (
    ErrorClassification,
    ErrorPattern,
    RecoveryAction,
    RecoveryStrategy,
    RecoveryTier,
)


# ── RecoveryTier ─────────────────────────────────────────────────────


class TestRecoveryTier:
    def test_tier1_value(self):
        assert RecoveryTier.TIER_1.value == 1

    def test_tier2_value(self):
        assert RecoveryTier.TIER_2.value == 2

    def test_is_int_enum(self):
        assert isinstance(RecoveryTier.TIER_1, int)
        assert RecoveryTier.TIER_1 == 1

    def test_all_members(self):
        members = {m.value for m in RecoveryTier}
        assert members == {1, 2}


# ── ErrorClassification ─────────────────────────────────────────────


class TestErrorClassification:
    def test_element_not_found(self):
        assert ErrorClassification.ELEMENT_NOT_FOUND.value == "ElementNotFound"

    def test_element_not_interactable(self):
        assert ErrorClassification.ELEMENT_NOT_INTERACTABLE.value == "ElementNotInteractable"

    def test_click_intercepted(self):
        assert ErrorClassification.ELEMENT_CLICK_INTERCEPTED.value == "ElementClickIntercepted"

    def test_timeout(self):
        assert ErrorClassification.TIMEOUT_EXCEPTION.value == "TimeoutException"

    def test_unexpected_alert(self):
        assert ErrorClassification.UNEXPECTED_ALERT.value == "UnexpectedAlertPresent"

    def test_stale_element(self):
        assert ErrorClassification.STALE_ELEMENT.value == "StaleElementReference"

    def test_navigation_drift(self):
        assert ErrorClassification.NAVIGATION_DRIFT.value == "NavigationDrift"

    def test_error_page(self):
        assert ErrorClassification.ERROR_PAGE.value == "ErrorPage"

    def test_session_loss(self):
        assert ErrorClassification.SESSION_LOSS.value == "SessionLoss"

    def test_unknown(self):
        assert ErrorClassification.UNKNOWN.value == "Unknown"

    def test_all_10_members(self):
        assert len(ErrorClassification) == 10

    def test_is_str_enum(self):
        assert isinstance(ErrorClassification.UNKNOWN, str)


# ── ErrorPattern ─────────────────────────────────────────────────────


class TestErrorPattern:
    def test_creation(self):
        pat = re.compile(r"not found", re.IGNORECASE)
        ep = ErrorPattern(
            classification=ErrorClassification.ELEMENT_NOT_FOUND,
            pattern=pat, priority=10,
        )
        assert ep.classification == ErrorClassification.ELEMENT_NOT_FOUND
        assert ep.priority == 10

    def test_from_string(self):
        ep = ErrorPattern.from_string(
            ErrorClassification.TIMEOUT_EXCEPTION,
            r"timeout|timed out",
            priority=8,
        )
        assert ep.classification == ErrorClassification.TIMEOUT_EXCEPTION
        assert ep.priority == 8

    def test_from_string_case_insensitive(self):
        ep = ErrorPattern.from_string(
            ErrorClassification.ELEMENT_NOT_FOUND,
            r"not found",
        )
        assert ep.pattern.search("NOT FOUND") is not None

    def test_regex_matching(self):
        ep = ErrorPattern.from_string(
            ErrorClassification.ELEMENT_NOT_FOUND,
            r"not found|unable to locate",
        )
        assert ep.pattern.search("Element not found in DOM") is not None
        assert ep.pattern.search("unable to locate element") is not None
        assert ep.pattern.search("everything ok") is None

    def test_default_priority(self):
        ep = ErrorPattern.from_string(
            ErrorClassification.UNKNOWN, r"x",
        )
        assert ep.priority == 0

    def test_frozen(self):
        ep = ErrorPattern.from_string(
            ErrorClassification.UNKNOWN, r"x",
        )
        with pytest.raises(AttributeError):
            ep.priority = 99

    def test_priority_ordering(self):
        low = ErrorPattern.from_string(ErrorClassification.ERROR_PAGE, r"404", 5)
        high = ErrorPattern.from_string(
            ErrorClassification.ELEMENT_NOT_FOUND, r"not found", 10,
        )
        sorted_patterns = sorted([low, high], key=lambda p: -p.priority)
        assert sorted_patterns[0].classification == ErrorClassification.ELEMENT_NOT_FOUND


# ── RecoveryAction ───────────────────────────────────────────────────


class TestRecoveryAction:
    def test_creation(self):
        a = RecoveryAction(
            keyword="Sleep", args=("2s",), description="Wait",
        )
        assert a.keyword == "Sleep"
        assert a.args == ("2s",)
        assert a.description == "Wait"

    def test_defaults(self):
        a = RecoveryAction(keyword="No Operation")
        assert a.args == ()
        assert a.description == ""

    def test_to_dict(self):
        a = RecoveryAction(
            keyword="Sleep", args=("2s",), description="Wait 2s",
        )
        d = a.to_dict()
        assert d == {
            "keyword": "Sleep",
            "args": ["2s"],
            "description": "Wait 2s",
        }

    def test_frozen(self):
        a = RecoveryAction(keyword="Sleep")
        with pytest.raises(AttributeError):
            a.keyword = "Other"


# ── RecoveryStrategy ─────────────────────────────────────────────────


class TestRecoveryStrategy:
    def test_creation(self):
        s = RecoveryStrategy(
            name="wait_and_retry",
            tier=RecoveryTier.TIER_1,
            applicable_to=(ErrorClassification.ELEMENT_NOT_FOUND,),
            actions=(
                RecoveryAction(keyword="Sleep", args=("2s",)),
            ),
            description="Wait then retry",
        )
        assert s.name == "wait_and_retry"
        assert s.tier == RecoveryTier.TIER_1

    def test_applies_to_true(self):
        s = RecoveryStrategy(
            name="s", tier=RecoveryTier.TIER_1,
            applicable_to=(
                ErrorClassification.ELEMENT_NOT_FOUND,
                ErrorClassification.TIMEOUT_EXCEPTION,
            ),
        )
        assert s.applies_to(ErrorClassification.ELEMENT_NOT_FOUND) is True
        assert s.applies_to(ErrorClassification.TIMEOUT_EXCEPTION) is True

    def test_applies_to_false(self):
        s = RecoveryStrategy(
            name="s", tier=RecoveryTier.TIER_1,
            applicable_to=(ErrorClassification.ELEMENT_NOT_FOUND,),
        )
        assert s.applies_to(ErrorClassification.SESSION_LOSS) is False

    def test_applies_to_empty(self):
        s = RecoveryStrategy(name="s", tier=RecoveryTier.TIER_1)
        assert s.applies_to(ErrorClassification.UNKNOWN) is False

    def test_to_dict(self):
        action = RecoveryAction(keyword="Sleep", args=("1s",), description="wait")
        s = RecoveryStrategy(
            name="wait", tier=RecoveryTier.TIER_1,
            applicable_to=(ErrorClassification.ELEMENT_NOT_FOUND,),
            actions=(action,),
            description="Wait then retry",
        )
        d = s.to_dict()
        assert d["name"] == "wait"
        assert d["tier"] == 1
        assert d["applicable_to"] == ["ElementNotFound"]
        assert len(d["actions"]) == 1
        assert d["description"] == "Wait then retry"

    def test_frozen(self):
        s = RecoveryStrategy(name="s", tier=RecoveryTier.TIER_1)
        with pytest.raises(AttributeError):
            s.name = "other"

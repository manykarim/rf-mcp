"""Tests for recovery domain aggregates."""
import pytest

from robotmcp.domains.recovery.aggregates import RecoveryEngine
from robotmcp.domains.recovery.value_objects import (
    ErrorClassification,
    ErrorPattern,
    RecoveryAction,
    RecoveryStrategy,
    RecoveryTier,
)


# ── RecoveryEngine.with_defaults() ──────────────────────────────────


class TestRecoveryEngineDefaults:
    def test_with_defaults_pattern_count(self):
        engine = RecoveryEngine.with_defaults()
        assert engine.pattern_count == 10

    def test_with_defaults_strategy_count(self):
        engine = RecoveryEngine.with_defaults()
        assert engine.strategy_count == 9

    def test_empty_engine(self):
        engine = RecoveryEngine()
        assert engine.pattern_count == 0
        assert engine.strategy_count == 0


# ── classify ─────────────────────────────────────────────────────────


class TestRecoveryEngineClassify:
    def test_element_not_found(self):
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("Element not found in page")
        assert result == ErrorClassification.ELEMENT_NOT_FOUND

    def test_element_locator_not_found(self):
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("Locator 'id=submit' not found")
        assert result == ErrorClassification.ELEMENT_NOT_FOUND

    def test_element_selector_not_found(self):
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("Selector '#btn' not found on page")
        assert result == ErrorClassification.ELEMENT_NOT_FOUND

    def test_unable_to_locate(self):
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("unable to locate element #btn")
        assert result == ErrorClassification.ELEMENT_NOT_FOUND

    def test_click_intercepted(self):
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("click intercepted by overlay")
        assert result == ErrorClassification.ELEMENT_CLICK_INTERCEPTED

    def test_other_element_would_receive(self):
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("other element would receive the click")
        assert result == ErrorClassification.ELEMENT_CLICK_INTERCEPTED

    def test_timeout(self):
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("Timeout exceeded: waited 30s")
        assert result == ErrorClassification.TIMEOUT_EXCEPTION

    def test_timed_out(self):
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("Connection timed out")
        assert result == ErrorClassification.TIMEOUT_EXCEPTION

    def test_stale_element(self):
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("StaleElementReferenceException")
        assert result == ErrorClassification.STALE_ELEMENT

    def test_not_attached_to_page(self):
        engine = RecoveryEngine.with_defaults()
        # Use the exact RF error text that starts with the stale pattern
        result = engine.classify("stale element reference: not attached to page")
        assert result == ErrorClassification.STALE_ELEMENT

    def test_session_loss(self):
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("session expired, please re-authenticate")
        assert result == ErrorClassification.SESSION_LOSS

    def test_session_loss_401(self):
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("Got 401 unauthorized response")
        assert result == ErrorClassification.SESSION_LOSS

    def test_navigation_drift(self):
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("unexpected url, expected /dashboard")
        assert result == ErrorClassification.NAVIGATION_DRIFT

    def test_error_page_404(self):
        engine = RecoveryEngine.with_defaults()
        # Use a message that matches ERROR_PAGE's \b404\b but not ELEMENT_NOT_FOUND
        result = engine.classify("Server returned 404 response")
        assert result == ErrorClassification.ERROR_PAGE

    def test_error_page_500(self):
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("Server returned 500 response")
        assert result == ErrorClassification.ERROR_PAGE

    def test_element_not_interactable(self):
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("Element is not interactable at this point")
        assert result == ErrorClassification.ELEMENT_NOT_INTERACTABLE

    def test_unexpected_alert(self):
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("UnexpectedAlertPresentException: alert open")
        assert result == ErrorClassification.UNEXPECTED_ALERT

    def test_unknown(self):
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("Some completely unrecognized error message")
        assert result == ErrorClassification.UNKNOWN

    def test_empty_message(self):
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("")
        assert result == ErrorClassification.UNKNOWN

    # ── Variable error classification (Bug 2 fix) ───────────────────

    def test_variable_not_found_classified_as_unknown(self):
        """Variable errors must NOT be classified as ElementNotFound."""
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("Variable '${cart_count_1}' not found.")
        assert result == ErrorClassification.UNKNOWN

    def test_variable_not_found_real_error_message(self):
        """Real RF error message format from production failure."""
        engine = RecoveryEngine.with_defaults()
        result = engine.classify(
            "Keyword execution failed: Variable '${cart_count_1}' not found."
        )
        assert result == ErrorClassification.UNKNOWN

    def test_variable_not_found_different_var_name(self):
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("Variable '${my_result}' not found.")
        assert result == ErrorClassification.UNKNOWN

    def test_variable_not_found_no_recovery_strategy(self):
        """Variable errors (UNKNOWN) should yield no recovery strategy."""
        engine = RecoveryEngine.with_defaults()
        classification = engine.classify("Variable '${x}' not found.")
        strategy = engine.select_strategy(classification, attempt_number=1)
        assert strategy is None

    def test_element_not_found_still_works(self):
        """Ensure element-level 'not found' still correctly classified."""
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("Element with locator '#submit' not found")
        assert result == ErrorClassification.ELEMENT_NOT_FOUND

    def test_no_element_still_works(self):
        """Ensure 'no element' pattern still matches."""
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("no element matches the given selector")
        assert result == ErrorClassification.ELEMENT_NOT_FOUND

    def test_cannot_find_still_works(self):
        """Ensure 'cannot find' pattern still matches."""
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("cannot find element on page")
        assert result == ErrorClassification.ELEMENT_NOT_FOUND

    def test_did_not_match_still_works(self):
        """Ensure 'did not match' pattern still matches."""
        engine = RecoveryEngine.with_defaults()
        result = engine.classify("Locator did not match any elements")
        assert result == ErrorClassification.ELEMENT_NOT_FOUND


# ── select_strategy ──────────────────────────────────────────────────


class TestRecoveryEngineSelectStrategy:
    def test_tier1_first_attempt_element_not_found(self):
        engine = RecoveryEngine.with_defaults()
        strategy = engine.select_strategy(
            ErrorClassification.ELEMENT_NOT_FOUND, attempt_number=1,
        )
        assert strategy is not None
        assert strategy.tier == RecoveryTier.TIER_1
        assert strategy.name == "wait_and_retry"

    def test_tier2_second_attempt_element_not_found(self):
        engine = RecoveryEngine.with_defaults()
        strategy = engine.select_strategy(
            ErrorClassification.ELEMENT_NOT_FOUND, attempt_number=2,
        )
        assert strategy is not None
        assert strategy.tier == RecoveryTier.TIER_2
        assert strategy.name == "navigate_back"

    def test_tier1_click_intercepted(self):
        engine = RecoveryEngine.with_defaults()
        strategy = engine.select_strategy(
            ErrorClassification.ELEMENT_CLICK_INTERCEPTED, attempt_number=1,
        )
        assert strategy is not None
        assert strategy.name == "dismiss_overlay"

    def test_tier1_timeout(self):
        engine = RecoveryEngine.with_defaults()
        strategy = engine.select_strategy(
            ErrorClassification.TIMEOUT_EXCEPTION, attempt_number=1,
        )
        assert strategy is not None
        assert strategy.name == "extended_timeout"

    def test_tier1_stale_element(self):
        engine = RecoveryEngine.with_defaults()
        strategy = engine.select_strategy(
            ErrorClassification.STALE_ELEMENT, attempt_number=1,
        )
        assert strategy is not None
        assert strategy.name == "stale_retry"

    def test_session_loss_returns_none(self):
        engine = RecoveryEngine.with_defaults()
        strategy = engine.select_strategy(
            ErrorClassification.SESSION_LOSS, attempt_number=1,
        )
        assert strategy is None

    def test_unknown_returns_none(self):
        engine = RecoveryEngine.with_defaults()
        strategy = engine.select_strategy(
            ErrorClassification.UNKNOWN, attempt_number=1,
        )
        assert strategy is None

    def test_tier2_error_page(self):
        engine = RecoveryEngine.with_defaults()
        strategy = engine.select_strategy(
            ErrorClassification.ERROR_PAGE, attempt_number=2,
        )
        assert strategy is not None
        assert strategy.name == "reload_page"

    def test_fallback_to_any_tier(self):
        engine = RecoveryEngine.with_defaults()
        # Navigation drift only has Tier 2 strategy
        strategy = engine.select_strategy(
            ErrorClassification.NAVIGATION_DRIFT, attempt_number=1,
        )
        assert strategy is not None
        assert strategy.name == "navigate_back"

    def test_unexpected_alert(self):
        engine = RecoveryEngine.with_defaults()
        strategy = engine.select_strategy(
            ErrorClassification.UNEXPECTED_ALERT, attempt_number=1,
        )
        assert strategy is not None
        assert strategy.name == "handle_alert"


# ── register_pattern / register_strategy ─────────────────────────────


class TestRecoveryEngineRegistration:
    def test_register_pattern(self):
        engine = RecoveryEngine()
        pat = ErrorPattern.from_string(
            ErrorClassification.ELEMENT_NOT_FOUND, r"custom error", 20,
        )
        engine.register_pattern(pat)
        assert engine.pattern_count == 1
        assert engine.classify("custom error here") == ErrorClassification.ELEMENT_NOT_FOUND

    def test_register_strategy(self):
        engine = RecoveryEngine()
        strategy = RecoveryStrategy(
            name="custom_strategy",
            tier=RecoveryTier.TIER_1,
            applicable_to=(ErrorClassification.ELEMENT_NOT_FOUND,),
        )
        engine.register_strategy(strategy)
        assert engine.strategy_count == 1

    def test_register_strategy_overwrites_same_name(self):
        engine = RecoveryEngine()
        s1 = RecoveryStrategy(name="s", tier=RecoveryTier.TIER_1)
        s2 = RecoveryStrategy(
            name="s", tier=RecoveryTier.TIER_2,
            applicable_to=(ErrorClassification.ERROR_PAGE,),
        )
        engine.register_strategy(s1)
        engine.register_strategy(s2)
        assert engine.strategy_count == 1

    def test_pattern_count_property(self):
        engine = RecoveryEngine()
        assert engine.pattern_count == 0
        engine.register_pattern(
            ErrorPattern.from_string(ErrorClassification.UNKNOWN, r"x"),
        )
        assert engine.pattern_count == 1

    def test_strategy_count_property(self):
        engine = RecoveryEngine()
        assert engine.strategy_count == 0
        engine.register_strategy(
            RecoveryStrategy(name="s", tier=RecoveryTier.TIER_1),
        )
        assert engine.strategy_count == 1

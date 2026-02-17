"""Recovery Domain Aggregate Root.

The RecoveryEngine is the aggregate root for the Recovery bounded
context. It owns the error pattern registry and strategy catalog,
and enforces invariants across both collections.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .value_objects import (
    ErrorClassification, ErrorPattern, RecoveryAction,
    RecoveryStrategy, RecoveryTier,
)


@dataclass
class RecoveryEngine:
    """Aggregate root: owns error pattern registry and strategy catalog.

    Invariants:
        - Each strategy name is unique within the catalog.
        - Patterns are evaluated in priority order (highest first).
        - SESSION_LOSS and UNKNOWN classifications yield no strategy.

    Concurrency:
        Like IntentRegistry, the engine is read-heavy, write-rare.
        Writes occur only at initialization or when custom patterns
        are registered. No locking is needed.
    """
    _error_patterns: List[ErrorPattern] = field(default_factory=list)
    _strategies: Dict[str, RecoveryStrategy] = field(default_factory=dict)

    @classmethod
    def with_defaults(cls) -> RecoveryEngine:
        """Create engine pre-populated with spec-defined patterns and strategies."""
        engine = cls()
        engine._register_default_patterns()
        engine._register_default_strategies()
        return engine

    def classify(self, error_message: str) -> ErrorClassification:
        """Classify an error message using registered patterns.

        Patterns are evaluated in descending priority order. The first
        match wins. If no pattern matches, returns UNKNOWN.

        Args:
            error_message: The error text from a failed keyword execution.

        Returns:
            The matching ErrorClassification.
        """
        sorted_patterns = sorted(self._error_patterns, key=lambda p: -p.priority)
        for pattern in sorted_patterns:
            if pattern.pattern.search(error_message):
                return pattern.classification
        return ErrorClassification.UNKNOWN

    def select_strategy(
        self,
        classification: ErrorClassification,
        attempt_number: int = 1,
    ) -> Optional[RecoveryStrategy]:
        """Select best strategy for an error classification and attempt number.

        Strategy selection follows an escalation model:
        - First attempt: prefer Tier 1 (keyword-specific heuristics).
        - Subsequent attempts: prefer Tier 2 (context-aware strategies).
        - If the preferred tier has no applicable strategy, fall back to
          any tier.

        Args:
            classification: The classified error type.
            attempt_number: 1-based attempt counter (1 = first try).

        Returns:
            A RecoveryStrategy if one applies, None for unrecoverable errors.
        """
        if classification == ErrorClassification.SESSION_LOSS:
            return None
        if classification == ErrorClassification.UNKNOWN:
            return None

        preferred_tier = (
            RecoveryTier.TIER_1 if attempt_number <= 1 else RecoveryTier.TIER_2
        )

        # Try preferred tier first
        for strategy in self._strategies.values():
            if strategy.tier == preferred_tier and strategy.applies_to(classification):
                return strategy

        # Fallback to any tier
        for strategy in self._strategies.values():
            if strategy.applies_to(classification):
                return strategy

        return None

    def register_pattern(self, pattern: ErrorPattern) -> None:
        """Register a custom error pattern."""
        self._error_patterns.append(pattern)

    def register_strategy(self, strategy: RecoveryStrategy) -> None:
        """Register a custom recovery strategy."""
        self._strategies[strategy.name] = strategy

    @property
    def pattern_count(self) -> int:
        """Number of registered error patterns."""
        return len(self._error_patterns)

    @property
    def strategy_count(self) -> int:
        """Number of registered recovery strategies."""
        return len(self._strategies)

    # ============================================================
    # Default pattern and strategy registration
    # ============================================================

    def _register_default_patterns(self) -> None:
        """Register ADR-011 spec error patterns.

        Pattern priority determines evaluation order (higher = first).
        Tier 1 patterns (specific element errors) have higher priority
        than Tier 2 patterns (page-level errors).
        """
        patterns = [
            # Variable errors (priority 15 — higher than element patterns)
            # Must be checked BEFORE ELEMENT_NOT_FOUND to prevent
            # "Variable '${x}' not found" matching "not found"
            (
                ErrorClassification.UNKNOWN,
                r"Variable\s+'[^']+'\s+not found",
                15,
            ),
            # Tier 1: Element-level errors (priority 10)
            (
                ErrorClassification.ELEMENT_NOT_FOUND,
                r"(?:element|locator|selector).*not found|unable to locate|did not match|no element|cannot find",
                10,
            ),
            (
                ErrorClassification.ELEMENT_NOT_INTERACTABLE,
                r"not interactable|not visible|not enabled|element is not",
                10,
            ),
            (
                ErrorClassification.ELEMENT_CLICK_INTERCEPTED,
                r"click intercepted|intercepts pointer|other element would receive",
                10,
            ),
            (
                ErrorClassification.TIMEOUT_EXCEPTION,
                r"timeout|timed out|Timeout exceeded|TimeoutError",
                8,
            ),
            (
                ErrorClassification.UNEXPECTED_ALERT,
                r"unexpected alert|alert open|UnexpectedAlertPresentException",
                10,
            ),
            (
                ErrorClassification.STALE_ELEMENT,
                r"stale element|not attached to page|StaleElementReferenceException",
                10,
            ),
            # Tier 2: Page-level errors (priority 5-7)
            (
                ErrorClassification.NAVIGATION_DRIFT,
                r"unexpected url|unexpected page|unexpected redirect|wrong page",
                5,
            ),
            (
                ErrorClassification.ERROR_PAGE,
                r"\b404\b|\b500\b|internal server error|page not found|service unavailable",
                5,
            ),
            (
                ErrorClassification.SESSION_LOSS,
                r"session expired|login redirect|unauthorized|session not found|401",
                7,
            ),
        ]
        for classification, pattern_str, priority in patterns:
            self._error_patterns.append(
                ErrorPattern.from_string(classification, pattern_str, priority)
            )

    def _register_default_strategies(self) -> None:
        """Register ADR-011 spec recovery strategies.

        Tier 1 strategies are keyword-specific heuristics that can be
        applied without understanding the page context.

        Tier 2 strategies are context-aware and may require page state
        inspection or navigation.
        """
        # ---- Tier 1: Keyword-specific ----

        self._strategies["wait_and_retry"] = RecoveryStrategy(
            name="wait_and_retry",
            tier=RecoveryTier.TIER_1,
            applicable_to=(ErrorClassification.ELEMENT_NOT_FOUND,),
            actions=(
                RecoveryAction(
                    keyword="Sleep",
                    args=("2s",),
                    description="Wait for element to appear",
                ),
            ),
            description="Wait 2s then retry the failed keyword",
        )

        self._strategies["scroll_and_wait"] = RecoveryStrategy(
            name="scroll_and_wait",
            tier=RecoveryTier.TIER_1,
            applicable_to=(ErrorClassification.ELEMENT_NOT_INTERACTABLE,),
            actions=(
                RecoveryAction(
                    keyword="Execute Javascript",
                    args=("window.scrollBy(0, 300)",),
                    description="Scroll down",
                ),
                RecoveryAction(
                    keyword="Sleep",
                    args=("1s",),
                    description="Wait for visibility",
                ),
            ),
            description="Scroll element into view and wait",
        )

        self._strategies["dismiss_overlay"] = RecoveryStrategy(
            name="dismiss_overlay",
            tier=RecoveryTier.TIER_1,
            applicable_to=(ErrorClassification.ELEMENT_CLICK_INTERCEPTED,),
            actions=(
                RecoveryAction(
                    keyword="Execute Javascript",
                    args=(
                        "document.querySelectorAll("
                        "'[class*=overlay],[class*=modal],[class*=popup],"
                        "[class*=cookie],[class*=banner]'"
                        ").forEach(e => e.remove())",
                    ),
                    description="Remove overlay/modal/popup elements from DOM",
                ),
            ),
            description="Remove overlays blocking clicks then retry",
        )

        self._strategies["extended_timeout"] = RecoveryStrategy(
            name="extended_timeout",
            tier=RecoveryTier.TIER_1,
            applicable_to=(ErrorClassification.TIMEOUT_EXCEPTION,),
            actions=(),  # No action -- retry with extended timeout
            description="Retry with 2x timeout",
        )

        self._strategies["handle_alert"] = RecoveryStrategy(
            name="handle_alert",
            tier=RecoveryTier.TIER_1,
            applicable_to=(ErrorClassification.UNEXPECTED_ALERT,),
            actions=(
                RecoveryAction(
                    keyword="Handle Alert",
                    args=("DISMISS",),
                    description="Dismiss unexpected alert",
                ),
            ),
            description="Dismiss alert and retry",
        )

        self._strategies["stale_retry"] = RecoveryStrategy(
            name="stale_retry",
            tier=RecoveryTier.TIER_1,
            applicable_to=(ErrorClassification.STALE_ELEMENT,),
            actions=(
                RecoveryAction(
                    keyword="Sleep",
                    args=("1s",),
                    description="Wait for DOM to stabilize",
                ),
            ),
            description="Wait for stable DOM then retry",
        )

        # ---- Tier 2: Context-aware ----

        self._strategies["navigate_back"] = RecoveryStrategy(
            name="navigate_back",
            tier=RecoveryTier.TIER_2,
            applicable_to=(
                ErrorClassification.NAVIGATION_DRIFT,
                ErrorClassification.ELEMENT_NOT_FOUND,
            ),
            actions=(
                RecoveryAction(
                    keyword="Go Back",
                    args=(),
                    description="Navigate back",
                ),
                RecoveryAction(
                    keyword="Sleep",
                    args=("2s",),
                    description="Wait for page load",
                ),
            ),
            description="Navigate back and retry",
        )

        self._strategies["reload_page"] = RecoveryStrategy(
            name="reload_page",
            tier=RecoveryTier.TIER_2,
            applicable_to=(
                ErrorClassification.ERROR_PAGE,
                ErrorClassification.TIMEOUT_EXCEPTION,
            ),
            actions=(
                RecoveryAction(
                    keyword="Reload Page",
                    args=(),
                    description="Reload current page",
                ),
                RecoveryAction(
                    keyword="Sleep",
                    args=("3s",),
                    description="Wait for page load",
                ),
            ),
            description="Reload page and retry",
        )

        self._strategies["session_loss_detection"] = RecoveryStrategy(
            name="session_loss_detection",
            tier=RecoveryTier.TIER_2,
            applicable_to=(ErrorClassification.SESSION_LOSS,),
            actions=(),  # No recovery action -- unrecoverable
            description="Session lost -- unrecoverable, LLM must handle",
        )

"""Recovery Domain Value Objects.

Immutable types that carry no identity. Equality is structural.
"""
from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Tuple


class RecoveryTier(int, enum.Enum):
    """Recovery strategy tier classification.

    Tier 1: Keyword-specific heuristics (wait, scroll, dismiss overlay).
    Tier 2: Context-aware strategies (navigate back, reload page).
    """
    TIER_1 = 1
    TIER_2 = 2


class ErrorClassification(str, enum.Enum):
    """Error taxonomy for recovery strategy selection.

    Each classification maps to one or more recovery strategies.
    The classifier tests error messages against registered patterns
    and returns the highest-priority match.
    """
    ELEMENT_NOT_FOUND = "ElementNotFound"
    ELEMENT_NOT_INTERACTABLE = "ElementNotInteractable"
    ELEMENT_CLICK_INTERCEPTED = "ElementClickIntercepted"
    TIMEOUT_EXCEPTION = "TimeoutException"
    UNEXPECTED_ALERT = "UnexpectedAlertPresent"
    STALE_ELEMENT = "StaleElementReference"
    NAVIGATION_DRIFT = "NavigationDrift"
    ERROR_PAGE = "ErrorPage"
    SESSION_LOSS = "SessionLoss"
    UNKNOWN = "Unknown"


@dataclass(frozen=True)
class ErrorPattern:
    """Maps an error message regex pattern to an ErrorClassification.

    Attributes:
        classification: The error category this pattern detects.
        pattern: Compiled regex (case-insensitive matching).
        priority: Higher values are matched first; ties are arbitrary.
    """
    classification: ErrorClassification
    pattern: re.Pattern  # type: ignore[type-arg]
    priority: int = 0

    @classmethod
    def from_string(
        cls,
        classification: ErrorClassification,
        pattern_str: str,
        priority: int = 0,
    ) -> ErrorPattern:
        """Convenience factory from a raw regex string."""
        return cls(
            classification=classification,
            pattern=re.compile(pattern_str, re.IGNORECASE),
            priority=priority,
        )


@dataclass(frozen=True)
class RecoveryAction:
    """A single recovery keyword to execute.

    Attributes:
        keyword: RF keyword name (e.g. "Sleep", "Execute Javascript").
        args: Positional arguments for the keyword.
        description: Human-readable explanation for diagnostics.
    """
    keyword: str
    args: Tuple[str, ...] = ()
    description: str = ""

    def to_dict(self):
        return {
            "keyword": self.keyword,
            "args": list(self.args),
            "description": self.description,
        }


@dataclass(frozen=True)
class RecoveryStrategy:
    """A named recovery approach with tier and applicable error types.

    Attributes:
        name: Unique strategy identifier (e.g. "wait_and_retry").
        tier: Which tier this strategy belongs to.
        applicable_to: Error classifications this strategy handles.
        actions: Ordered recovery actions to execute before retry.
        description: Human-readable description for diagnostics.
    """
    name: str
    tier: RecoveryTier
    applicable_to: Tuple[ErrorClassification, ...] = ()
    actions: Tuple[RecoveryAction, ...] = ()
    description: str = ""

    def applies_to(self, classification: ErrorClassification) -> bool:
        """Check if this strategy handles the given error classification."""
        return classification in self.applicable_to

    def to_dict(self):
        return {
            "name": self.name,
            "tier": self.tier.value,
            "applicable_to": [c.value for c in self.applicable_to],
            "actions": [a.to_dict() for a in self.actions],
            "description": self.description,
        }

"""Intent Domain Value Objects.

Immutable types that carry no identity. Equality is structural.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Dict, List, Optional, Tuple


class IntentVerb(str, Enum):
    """Enumeration of supported intent verbs.

    Each verb represents a high-level user action that maps to one or
    more RF keywords depending on the active library. The set is
    deliberately small (8 verbs) to minimize decision entropy for
    small LLMs.
    """
    NAVIGATE = "navigate"
    CLICK = "click"
    FILL = "fill"
    HOVER = "hover"
    SELECT = "select"
    ASSERT_VISIBLE = "assert_visible"
    EXTRACT_TEXT = "extract_text"
    WAIT_FOR = "wait_for"

    # Desktop-specific intents (ADR-012: PlatynUI)
    ACTIVATE = "activate"
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    RESTORE = "restore"
    FOCUS = "focus"
    CLOSE_WINDOW = "close_window"
    INSPECT = "inspect"


class LocatorStrategy(Enum):
    """Locator strategy hints that accompany an IntentTarget.

    When the LLM provides a strategy hint, the resolver can skip
    heuristic detection and apply the correct normalization directly.
    """
    CSS = "css"
    XPATH = "xpath"
    TEXT = "text"
    ID = "id"
    NAME = "name"
    LINK = "link"
    PARTIAL_LINK = "partial_link"
    ACCESSIBILITY_ID = "accessibility_id"
    AUTO = "auto"   # Let resolver detect strategy from target string
    PLATYNUI_XPATH = "platynui_xpath"  # //control:*, //item:*, //app:*, //native:*


@dataclass(frozen=True)
class IntentTarget:
    """A locator string with optional strategy hint.

    This value object wraps the raw locator that the LLM provides
    and carries an optional strategy hint to assist normalization.

    Attributes:
        locator: Raw locator string (e.g., "text=Login", "#submit", "//button")
        strategy: Optional strategy hint; AUTO means resolver will detect
        original_locator: Preserved original before any normalization

    Invariants:
        - locator must not be empty for intents that require a target
        - locator length must not exceed MAX_LOCATOR_LENGTH

    Examples:
        >>> target = IntentTarget(locator="text=Login")
        >>> target = IntentTarget(locator="#submit", strategy=LocatorStrategy.CSS)
        >>> target = IntentTarget(locator="Submit")  # strategy=AUTO, resolver detects
    """
    locator: str
    strategy: LocatorStrategy = LocatorStrategy.AUTO
    original_locator: Optional[str] = None

    MAX_LOCATOR_LENGTH: ClassVar[int] = 10000

    def __post_init__(self) -> None:
        if len(self.locator) > self.MAX_LOCATOR_LENGTH:
            raise ValueError(
                f"Locator exceeds max length of {self.MAX_LOCATOR_LENGTH}"
            )
        if "\x00" in self.locator:
            raise ValueError("Locator must not contain null bytes")

    @property
    def has_explicit_strategy(self) -> bool:
        """True if strategy is not AUTO (LLM provided a hint)."""
        return self.strategy != LocatorStrategy.AUTO

    @property
    def has_prefix(self) -> bool:
        """True if the locator string itself contains a strategy prefix.

        Detects both Browser Library (css=, text=, id=, xpath=) and
        SeleniumLibrary (css:, id:, xpath:, link:) prefix conventions.
        """
        known_prefixes = (
            "css=", "xpath=", "text=", "id=",
            "css:", "xpath:", "id:", "name:", "link:",
            "partial link:", "class:", "tag:", "dom:",
            "accessibility_id=",
        )
        return any(self.locator.startswith(p) for p in known_prefixes)


@dataclass(frozen=True)
class NormalizedLocator:
    """Result of locator normalization for a specific target library.

    Attributes:
        value: The normalized locator string ready for the target library
        source_locator: The original IntentTarget locator
        target_library: The library this was normalized for
        strategy_applied: Which strategy was used for normalization
        was_transformed: Whether the locator was actually changed
    """
    value: str
    source_locator: str
    target_library: str
    strategy_applied: str
    was_transformed: bool


@dataclass(frozen=True)
class ResolvedIntent:
    """The output of intent resolution: a concrete keyword + arguments.

    This is what the IntentResolver produces and what the MCP tool
    adapter passes to execute_step.

    Attributes:
        keyword: The RF keyword name to execute (e.g., "Click Element")
        arguments: Positional arguments for the keyword
        library: The library that owns the keyword
        intent_verb: The original intent verb
        normalized_locator: The locator after normalization (if applicable)
        metadata: Additional resolution metadata for diagnostics
    """
    keyword: str
    arguments: List[str]
    library: str
    intent_verb: IntentVerb
    normalized_locator: Optional[NormalizedLocator] = None
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FallbackStep:
    """A single keyword execution in a fallback sequence."""
    keyword: str
    arguments: Tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class NavigateFallbackSequence:
    """Ordered steps to recover from a navigate failure.

    Attributes:
        library: The library this fallback applies to
        error_pattern: Regex pattern to match error messages
        steps: Ordered tuple of FallbackStep to execute
        description: Human-readable description for diagnostics
    """
    library: str
    error_pattern: str
    steps: Tuple[FallbackStep, ...]
    description: str

    def matches_error(self, error_message: str) -> bool:
        """Check if this sequence handles the given error."""
        if not error_message:
            return False
        return bool(re.search(self.error_pattern, error_message, re.IGNORECASE))

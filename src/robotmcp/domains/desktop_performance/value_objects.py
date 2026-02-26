"""Value objects for the desktop_performance bounded context.

All value objects are frozen dataclasses with validation in __post_init__,
ClassVar constants, and factory classmethods — following the established
conventions from ADR-001/006/007/008/011.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, Optional, Tuple


class XPathAxis(str, Enum):
    """XPath axis classification."""

    ABSOLUTE = "absolute"  # //control:Button[@Name='OK']
    RELATIVE = "relative"  # .//control:Button[@Name='OK']


class InteractionSpeed(str, Enum):
    """Pointer/keyboard interaction speed presets."""

    INSTANT = "instant"  # CI/headless: zero delays
    FAST = "fast"  # Default for automation: minimal delays
    REALISTIC = "realistic"  # Visual debugging: human-like delays


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLATYNUI_XPATH_RE = re.compile(
    r"^\.{0,2}//"
    r"(?:control|item|app|group|edit|text|menu|menuitem|combobox|treeitem|tab|tabitem|dialog|window)"
    r":",
    re.IGNORECASE,
)


def _is_platynui_xpath(locator: str) -> bool:
    """Return True if *locator* looks like a PlatynUI XPath expression."""
    if not locator:
        return False
    # Fast path for common prefixes
    if locator.startswith("//") or locator.startswith(".//"):
        return bool(_PLATYNUI_XPATH_RE.match(locator))
    # Also accept bare /Window or /control: forms
    if locator.startswith("/"):
        return True
    return False


# ---------------------------------------------------------------------------
# CacheKey
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheKey:
    """Immutable identifier for a cached desktop element.

    Combines XPath expression with session context to ensure
    cross-session isolation.
    """

    xpath: str
    session_id: str

    def __post_init__(self) -> None:
        if not self.xpath:
            raise ValueError("CacheKey.xpath must not be empty")
        if not self.session_id:
            raise ValueError("CacheKey.session_id must not be empty")

    @classmethod
    def from_keyword_args(
        cls,
        session_id: str,
        keyword: str,
        arguments: Tuple[str, ...],
    ) -> Optional["CacheKey"]:
        """Extract cache key from keyword arguments.

        Only PlatynUI keywords whose first argument is an XPath locator
        are cacheable.
        """
        if not arguments:
            return None
        locator = arguments[0]
        if not _is_platynui_xpath(locator):
            return None
        return cls(xpath=locator, session_id=session_id)

    @property
    def key_str(self) -> str:
        """String representation used as dict key."""
        return f"{self.session_id}:{self.xpath}"


# ---------------------------------------------------------------------------
# CacheTTL
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheTTL:
    """Time-to-live for cached desktop elements."""

    value_seconds: float

    MIN_SECONDS: ClassVar[float] = 1.0
    MAX_SECONDS: ClassVar[float] = 300.0
    DEFAULT_SECONDS: ClassVar[float] = 60.0

    def __post_init__(self) -> None:
        if not (self.MIN_SECONDS <= self.value_seconds <= self.MAX_SECONDS):
            raise ValueError(
                f"CacheTTL must be {self.MIN_SECONDS}-{self.MAX_SECONDS}s, "
                f"got {self.value_seconds}"
            )

    @classmethod
    def default(cls) -> "CacheTTL":
        return cls(value_seconds=cls.DEFAULT_SECONDS)

    @classmethod
    def short(cls) -> "CacheTTL":
        """Short TTL for dynamic UIs."""
        return cls(value_seconds=5.0)

    def is_expired(self, cached_at: float, now: float) -> bool:
        return (now - cached_at) > self.value_seconds


# ---------------------------------------------------------------------------
# XPathTransform
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XPathTransform:
    """Result of transforming an absolute XPath to a relative one.

    The relative form scopes AT-SPI traversal to the application subtree,
    reducing query time from ~7.5s to ~0.02-0.27s.
    """

    original: str
    transformed: str
    axis: XPathAxis
    scoping_applied: bool

    def __post_init__(self) -> None:
        if not self.original:
            raise ValueError("XPathTransform.original must not be empty")
        if not self.transformed:
            raise ValueError("XPathTransform.transformed must not be empty")

    @classmethod
    def to_relative(cls, xpath: str) -> "XPathTransform":
        """Convert absolute XPath to relative for scoped evaluation.

        //control:Button[@Name='OK'] -> .//control:Button[@Name='OK']
        """
        if xpath.startswith(".//") or xpath.startswith("./"):
            return cls(
                original=xpath,
                transformed=xpath,
                axis=XPathAxis.RELATIVE,
                scoping_applied=False,
            )
        if xpath.startswith("//"):
            relative = "." + xpath
            return cls(
                original=xpath,
                transformed=relative,
                axis=XPathAxis.ABSOLUTE,
                scoping_applied=True,
            )
        # Single-slash or other forms: leave unchanged
        return cls(
            original=xpath,
            transformed=xpath,
            axis=XPathAxis.ABSOLUTE,
            scoping_applied=False,
        )

    @classmethod
    def identity(cls, xpath: str) -> "XPathTransform":
        """No-op transform (used when scoping is not active)."""
        axis = XPathAxis.RELATIVE if xpath.startswith(".") else XPathAxis.ABSOLUTE
        return cls(
            original=xpath,
            transformed=xpath,
            axis=axis,
            scoping_applied=False,
        )


# ---------------------------------------------------------------------------
# CacheCapacity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheCapacity:
    """Maximum number of cached elements per session."""

    max_entries: int

    MIN_ENTRIES: ClassVar[int] = 10
    MAX_ENTRIES: ClassVar[int] = 1000
    DEFAULT_ENTRIES: ClassVar[int] = 200

    def __post_init__(self) -> None:
        if not (self.MIN_ENTRIES <= self.max_entries <= self.MAX_ENTRIES):
            raise ValueError(
                f"CacheCapacity must be {self.MIN_ENTRIES}-{self.MAX_ENTRIES}, "
                f"got {self.max_entries}"
            )

    @classmethod
    def default(cls) -> "CacheCapacity":
        return cls(max_entries=cls.DEFAULT_ENTRIES)


# ---------------------------------------------------------------------------
# PointerSpeedProfile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PointerSpeedProfile:
    """Frozen configuration for pointer movement timing.

    Maps to platynui_native.PointerOverrides constructor kwargs.
    """

    speed: InteractionSpeed
    after_move_delay_ms: float
    after_input_delay_ms: float
    press_release_delay_ms: float
    after_click_delay_ms: float
    motion_mode: int  # 0=DIRECT, 1=LINEAR, 2=BEZIER
    max_move_duration_ms: float
    speed_factor: float

    def to_overrides_dict(self) -> Dict[str, Any]:
        """Convert to platynui_native.PointerOverrides kwargs."""
        return {
            "after_move_delay_ms": self.after_move_delay_ms,
            "after_input_delay_ms": self.after_input_delay_ms,
            "press_release_delay_ms": self.press_release_delay_ms,
            "after_click_delay_ms": self.after_click_delay_ms,
            "motion": self.motion_mode,
            "max_move_duration_ms": self.max_move_duration_ms,
            "speed_factor": self.speed_factor,
        }


# ClassVar-style presets (set after class definition for frozen dataclass)
POINTER_SPEED_INSTANT = PointerSpeedProfile(
    speed=InteractionSpeed.INSTANT,
    after_move_delay_ms=0,
    after_input_delay_ms=0,
    press_release_delay_ms=0,
    after_click_delay_ms=0,
    motion_mode=0,
    max_move_duration_ms=0,
    speed_factor=100.0,
)

POINTER_SPEED_FAST = PointerSpeedProfile(
    speed=InteractionSpeed.FAST,
    after_move_delay_ms=0,
    after_input_delay_ms=0,
    press_release_delay_ms=5,
    after_click_delay_ms=10,
    motion_mode=0,
    max_move_duration_ms=50,
    speed_factor=10.0,
)

POINTER_SPEED_REALISTIC = PointerSpeedProfile(
    speed=InteractionSpeed.REALISTIC,
    after_move_delay_ms=50,
    after_input_delay_ms=20,
    press_release_delay_ms=20,
    after_click_delay_ms=50,
    motion_mode=2,
    max_move_duration_ms=500,
    speed_factor=1.0,
)

SPEED_PROFILES: Dict[InteractionSpeed, PointerSpeedProfile] = {
    InteractionSpeed.INSTANT: POINTER_SPEED_INSTANT,
    InteractionSpeed.FAST: POINTER_SPEED_FAST,
    InteractionSpeed.REALISTIC: POINTER_SPEED_REALISTIC,
}

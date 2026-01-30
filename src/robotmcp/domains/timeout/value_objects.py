"""Timeout Domain Value Objects.

This module contains immutable value objects for the Timeout bounded context.
Value objects are identified by their attributes rather than by identity.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class PolicyId:
    """Unique identifier for a timeout policy.

    Used to track and reference timeout policies across sessions.
    """
    value: str

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"PolicyId({self.value!r})"

    @classmethod
    def for_session(cls, session_id: str) -> "PolicyId":
        """Create a policy ID for a given session."""
        return cls(value=f"policy_{session_id}")


@dataclass(frozen=True)
class Milliseconds:
    """Type-safe milliseconds value object.

    Represents a duration in milliseconds with validation to ensure
    non-negative values. Provides conversion utilities for working
    with seconds.

    Examples:
        >>> ms = Milliseconds(5000)
        >>> ms.to_seconds()
        5.0
        >>> ms = Milliseconds.seconds(2.5)
        >>> ms.value
        2500
    """
    value: int

    def __post_init__(self) -> None:
        """Validate that the timeout value is non-negative."""
        if self.value < 0:
            raise ValueError(f"Timeout cannot be negative, got {self.value}")

    @classmethod
    def seconds(cls, seconds: float) -> "Milliseconds":
        """Create a Milliseconds value from seconds.

        Args:
            seconds: Duration in seconds (can be fractional).

        Returns:
            Milliseconds value object.

        Examples:
            >>> Milliseconds.seconds(5.0)
            Milliseconds(value=5000)
        """
        return cls(value=int(seconds * 1000))

    def to_seconds(self) -> float:
        """Convert milliseconds to seconds.

        Returns:
            Duration in seconds as a float.
        """
        return self.value / 1000.0

    def __str__(self) -> str:
        return f"{self.value}ms"

    def __repr__(self) -> str:
        return f"Milliseconds(value={self.value})"

    def __add__(self, other: "Milliseconds") -> "Milliseconds":
        """Add two Milliseconds values."""
        if not isinstance(other, Milliseconds):
            return NotImplemented
        return Milliseconds(self.value + other.value)

    def __sub__(self, other: "Milliseconds") -> "Milliseconds":
        """Subtract two Milliseconds values."""
        if not isinstance(other, Milliseconds):
            return NotImplemented
        return Milliseconds(self.value - other.value)

    def __lt__(self, other: "Milliseconds") -> bool:
        """Compare less than."""
        if not isinstance(other, Milliseconds):
            return NotImplemented
        return self.value < other.value

    def __le__(self, other: "Milliseconds") -> bool:
        """Compare less than or equal."""
        if not isinstance(other, Milliseconds):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other: "Milliseconds") -> bool:
        """Compare greater than."""
        if not isinstance(other, Milliseconds):
            return NotImplemented
        return self.value > other.value

    def __ge__(self, other: "Milliseconds") -> bool:
        """Compare greater than or equal."""
        if not isinstance(other, Milliseconds):
            return NotImplemented
        return self.value >= other.value


class DefaultTimeouts:
    """Default timeout configuration matching Playwright MCP patterns.

    This class defines the default timeout values used throughout the system.
    The dual timeout strategy uses:
    - Short timeouts (5s) for element actions like clicks and typing
    - Long timeouts (60s) for navigation operations like page loads

    These values are based on Playwright MCP's proven configuration that
    balances responsiveness with reliability.

    Attributes:
        ACTION: Default timeout for element actions (5 seconds).
        NAVIGATION: Default timeout for page navigation (60 seconds).
        ASSERTION: Default timeout for assertion retries (10 seconds).
        READ: Default timeout for read operations (2 seconds).
        MIN_TIMEOUT: Minimum allowed timeout value (100ms).
        MAX_ACTION_TIMEOUT: Maximum allowed action timeout (30 seconds).
        MAX_NAVIGATION_TIMEOUT: Maximum allowed navigation timeout (5 minutes).
    """

    # Core timeout defaults (matching Playwright MCP)
    ACTION = Milliseconds(5000)       # 5 seconds for clicks, typing, etc.
    NAVIGATION = Milliseconds(60000)  # 60 seconds for page loads
    ASSERTION = Milliseconds(10000)   # 10 seconds for assertion retries
    READ = Milliseconds(2000)         # 2 seconds for read operations

    # Bounds for validation
    MIN_TIMEOUT = Milliseconds(100)           # Minimum 100ms
    MAX_ACTION_TIMEOUT = Milliseconds(30000)  # Maximum 30 seconds for actions
    MAX_NAVIGATION_TIMEOUT = Milliseconds(300000)  # Maximum 5 minutes for navigation

    @classmethod
    def get_default_for_category(cls, category: str) -> Milliseconds:
        """Get the default timeout for a given category.

        Args:
            category: One of "navigation", "action", or "read".

        Returns:
            The default Milliseconds value for the category.

        Raises:
            ValueError: If the category is not recognized.
        """
        defaults = {
            "navigation": cls.NAVIGATION,
            "action": cls.ACTION,
            "read": cls.READ,
            "assertion": cls.ASSERTION,
        }
        if category not in defaults:
            raise ValueError(f"Unknown timeout category: {category}")
        return defaults[category]

    @classmethod
    def get_bounds_for_category(cls, category: str) -> tuple[Milliseconds, Milliseconds]:
        """Get the min/max bounds for a given category.

        Args:
            category: One of "navigation", "action", or "read".

        Returns:
            Tuple of (min_timeout, max_timeout) for the category.
        """
        if category == "navigation":
            return (cls.MIN_TIMEOUT, cls.MAX_NAVIGATION_TIMEOUT)
        else:
            return (cls.MIN_TIMEOUT, cls.MAX_ACTION_TIMEOUT)

"""Tests for Timeout Context bounded context.

This module tests the timeout configuration functionality:
- Milliseconds value object (time unit handling)
- TimeoutPolicy aggregate (dual timeout strategy)
- Timeout bounds and validation
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import pytest


# =============================================================================
# Domain Models (to be moved to production code)
# =============================================================================


class TimeoutError(Exception):
    """Base exception for timeout-related errors."""

    pass


class InvalidTimeoutError(TimeoutError):
    """Exception raised when a timeout value is invalid."""

    def __init__(self, value: int, reason: str):
        self.value = value
        self.reason = reason
        super().__init__(f"Invalid timeout {value}ms: {reason}")


@dataclass(frozen=True)
class Milliseconds:
    """Value object representing time in milliseconds.

    Provides type-safe time handling and unit conversions.
    Cannot be negative.
    """

    value: int

    def __post_init__(self):
        if self.value < 0:
            raise ValueError("Milliseconds cannot be negative")

    @classmethod
    def from_seconds(cls, seconds: float) -> "Milliseconds":
        """Create Milliseconds from seconds."""
        return cls(value=int(seconds * 1000))

    def to_seconds(self) -> float:
        """Convert to seconds."""
        return self.value / 1000.0

    def __add__(self, other: "Milliseconds") -> "Milliseconds":
        return Milliseconds(self.value + other.value)

    def __sub__(self, other: "Milliseconds") -> "Milliseconds":
        return Milliseconds(self.value - other.value)

    def __mul__(self, factor: float) -> "Milliseconds":
        return Milliseconds(int(self.value * factor))

    def __lt__(self, other: "Milliseconds") -> bool:
        return self.value < other.value

    def __le__(self, other: "Milliseconds") -> bool:
        return self.value <= other.value

    def __gt__(self, other: "Milliseconds") -> bool:
        return self.value > other.value

    def __ge__(self, other: "Milliseconds") -> bool:
        return self.value >= other.value

    def __str__(self) -> str:
        return f"{self.value}ms"


class ActionType(Enum):
    """Types of browser actions with different timeout requirements."""

    # Quick actions - use action timeout
    CLICK = "click"
    TYPE = "type"
    FILL = "fill"
    HOVER = "hover"
    SELECT = "select"
    CHECK = "check"
    UNCHECK = "uncheck"
    FOCUS = "focus"
    BLUR = "blur"
    PRESS = "press"
    SCROLL = "scroll"

    # Navigation actions - use navigation timeout
    NAVIGATE = "navigate"
    RELOAD = "reload"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"

    # Wait actions - use custom timeout
    WAIT_FOR_ELEMENT = "wait_for_element"
    WAIT_FOR_SELECTOR = "wait_for_selector"
    WAIT_FOR_TEXT = "wait_for_text"
    WAIT_FOR_NAVIGATION = "wait_for_navigation"

    # Read actions - use action timeout
    GET_TEXT = "get_text"
    GET_VALUE = "get_value"
    GET_ATTRIBUTE = "get_attribute"
    GET_SNAPSHOT = "get_snapshot"


class TimeoutPolicy:
    """Aggregate for managing timeout configuration.

    Implements the dual timeout strategy:
    - Action timeout: Fast timeout for element interactions (default 5s)
    - Navigation timeout: Longer timeout for page loads (default 60s)

    Also handles custom overrides and validation.
    """

    # Default timeouts in milliseconds
    DEFAULT_ACTION_TIMEOUT = 5000
    DEFAULT_NAVIGATION_TIMEOUT = 60000
    DEFAULT_ASSERTION_RETRY = 10000

    # Absolute bounds
    MIN_TIMEOUT = 100  # 100ms minimum
    MAX_TIMEOUT = 300000  # 5 minutes maximum

    # Action classification
    NAVIGATION_ACTIONS = {
        ActionType.NAVIGATE,
        ActionType.RELOAD,
        ActionType.GO_BACK,
        ActionType.GO_FORWARD,
        ActionType.WAIT_FOR_NAVIGATION,
    }

    def __init__(
        self,
        action_timeout: Optional[int] = None,
        navigation_timeout: Optional[int] = None,
        assertion_retry: Optional[int] = None,
        min_timeout: Optional[int] = None,
        max_timeout: Optional[int] = None,
    ):
        """Initialize timeout policy.

        Args:
            action_timeout: Timeout for element actions in ms
            navigation_timeout: Timeout for navigation in ms
            assertion_retry: Retry timeout for assertions in ms
            min_timeout: Minimum allowed timeout in ms
            max_timeout: Maximum allowed timeout in ms
        """
        self._min = min_timeout or self.MIN_TIMEOUT
        self._max = max_timeout or self.MAX_TIMEOUT

        # Validate and set timeouts
        self._action_timeout = self._validate_and_clamp(
            action_timeout or self.DEFAULT_ACTION_TIMEOUT
        )
        self._navigation_timeout = self._validate_and_clamp(
            navigation_timeout or self.DEFAULT_NAVIGATION_TIMEOUT
        )
        self._assertion_retry = self._validate_and_clamp(
            assertion_retry or self.DEFAULT_ASSERTION_RETRY
        )

        # Custom overrides per action type
        self._custom_overrides: Dict[ActionType, Milliseconds] = {}

    @property
    def action_timeout(self) -> Milliseconds:
        """Default timeout for element actions."""
        return Milliseconds(self._action_timeout)

    @property
    def navigation_timeout(self) -> Milliseconds:
        """Timeout for navigation operations."""
        return Milliseconds(self._navigation_timeout)

    @property
    def assertion_retry(self) -> Milliseconds:
        """Retry timeout for assertions."""
        return Milliseconds(self._assertion_retry)

    def get_timeout_for_action(
        self, action: ActionType, custom_override: Optional[int] = None
    ) -> Milliseconds:
        """Get the appropriate timeout for an action.

        Args:
            action: The type of action
            custom_override: Optional custom timeout to use

        Returns:
            The timeout in milliseconds
        """
        # Check for custom override
        if custom_override is not None:
            return Milliseconds(self._validate_and_clamp(custom_override))

        # Check for action-specific override
        if action in self._custom_overrides:
            return self._custom_overrides[action]

        # Use category-based timeout
        if action in self.NAVIGATION_ACTIONS:
            return self.navigation_timeout
        else:
            return self.action_timeout

    def set_custom_timeout(
        self, action: ActionType, timeout_ms: int
    ) -> None:
        """Set a custom timeout for a specific action type.

        Args:
            action: The action type to configure
            timeout_ms: The timeout in milliseconds
        """
        validated = self._validate_and_clamp(timeout_ms)
        self._custom_overrides[action] = Milliseconds(validated)

    def clear_custom_timeout(self, action: ActionType) -> None:
        """Clear custom timeout for an action type."""
        self._custom_overrides.pop(action, None)

    def _validate_and_clamp(self, timeout_ms: int) -> int:
        """Validate and clamp a timeout value to bounds.

        Args:
            timeout_ms: The timeout value in milliseconds

        Returns:
            The clamped timeout value
        """
        if timeout_ms < 0:
            raise InvalidTimeoutError(timeout_ms, "timeout cannot be negative")

        # Clamp to bounds
        if timeout_ms < self._min:
            return self._min
        if timeout_ms > self._max:
            return self._max
        return timeout_ms

    def validate(self, timeout_ms: int) -> Milliseconds:
        """Validate a timeout value and return as Milliseconds.

        Args:
            timeout_ms: The timeout value to validate

        Returns:
            The validated timeout as Milliseconds

        Raises:
            InvalidTimeoutError: If the timeout is invalid
        """
        validated = self._validate_and_clamp(timeout_ms)
        return Milliseconds(validated)

    def to_dict(self) -> Dict[str, int]:
        """Export configuration as dictionary."""
        return {
            "action_timeout": self._action_timeout,
            "navigation_timeout": self._navigation_timeout,
            "assertion_retry": self._assertion_retry,
            "min_timeout": self._min,
            "max_timeout": self._max,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, int]) -> "TimeoutPolicy":
        """Create policy from dictionary configuration."""
        return cls(
            action_timeout=config.get("action_timeout"),
            navigation_timeout=config.get("navigation_timeout"),
            assertion_retry=config.get("assertion_retry"),
            min_timeout=config.get("min_timeout"),
            max_timeout=config.get("max_timeout"),
        )


# =============================================================================
# Tests
# =============================================================================


class TestMilliseconds:
    """Tests for Milliseconds value object."""

    def test_cannot_be_negative(self):
        """Test that Milliseconds cannot be negative."""
        with pytest.raises(ValueError, match="cannot be negative"):
            Milliseconds(-1)

        with pytest.raises(ValueError, match="cannot be negative"):
            Milliseconds(-1000)

    def test_zero_is_allowed(self):
        """Test that zero milliseconds is allowed."""
        ms = Milliseconds(0)
        assert ms.value == 0

    def test_from_seconds_conversion(self):
        """Test conversion from seconds."""
        ms = Milliseconds.from_seconds(5.0)
        assert ms.value == 5000

        ms = Milliseconds.from_seconds(0.5)
        assert ms.value == 500

        ms = Milliseconds.from_seconds(1.234)
        assert ms.value == 1234

    def test_to_seconds_conversion(self):
        """Test conversion to seconds."""
        ms = Milliseconds(5000)
        assert ms.to_seconds() == 5.0

        ms = Milliseconds(500)
        assert ms.to_seconds() == 0.5

        ms = Milliseconds(1234)
        assert ms.to_seconds() == 1.234

    def test_addition(self):
        """Test adding two Milliseconds."""
        ms1 = Milliseconds(1000)
        ms2 = Milliseconds(500)
        result = ms1 + ms2
        assert result.value == 1500

    def test_subtraction(self):
        """Test subtracting Milliseconds."""
        ms1 = Milliseconds(1000)
        ms2 = Milliseconds(500)
        result = ms1 - ms2
        assert result.value == 500

    def test_subtraction_cannot_go_negative(self):
        """Test that subtraction resulting in negative raises error."""
        ms1 = Milliseconds(500)
        ms2 = Milliseconds(1000)
        with pytest.raises(ValueError, match="cannot be negative"):
            _ = ms1 - ms2

    def test_multiplication(self):
        """Test multiplying Milliseconds by factor."""
        ms = Milliseconds(1000)
        result = ms * 2.5
        assert result.value == 2500

    def test_comparison_operators(self):
        """Test comparison operators."""
        ms1 = Milliseconds(1000)
        ms2 = Milliseconds(2000)
        ms3 = Milliseconds(1000)

        assert ms1 < ms2
        assert ms1 <= ms2
        assert ms1 <= ms3
        assert ms2 > ms1
        assert ms2 >= ms1
        assert ms1 >= ms3

    def test_string_representation(self):
        """Test string conversion."""
        ms = Milliseconds(5000)
        assert str(ms) == "5000ms"

    def test_equality(self):
        """Test equality comparison."""
        ms1 = Milliseconds(1000)
        ms2 = Milliseconds(1000)
        ms3 = Milliseconds(2000)

        assert ms1 == ms2
        assert ms1 != ms3

    def test_immutability(self):
        """Test that Milliseconds is immutable."""
        ms = Milliseconds(1000)
        with pytest.raises((AttributeError, TypeError)):
            ms.value = 2000


class TestTimeoutPolicy:
    """Tests for TimeoutPolicy aggregate."""

    @pytest.fixture
    def policy(self) -> TimeoutPolicy:
        """Create a default policy for testing."""
        return TimeoutPolicy()

    def test_default_timeouts(self, policy):
        """Test default timeout values."""
        assert policy.action_timeout.value == 5000
        assert policy.navigation_timeout.value == 60000
        assert policy.assertion_retry.value == 10000

    def test_get_timeout_for_click_returns_action_timeout(self, policy):
        """Test that click uses action timeout."""
        timeout = policy.get_timeout_for_action(ActionType.CLICK)
        assert timeout == policy.action_timeout

    def test_get_timeout_for_type_returns_action_timeout(self, policy):
        """Test that type uses action timeout."""
        timeout = policy.get_timeout_for_action(ActionType.TYPE)
        assert timeout == policy.action_timeout

    def test_get_timeout_for_hover_returns_action_timeout(self, policy):
        """Test that hover uses action timeout."""
        timeout = policy.get_timeout_for_action(ActionType.HOVER)
        assert timeout == policy.action_timeout

    def test_get_timeout_for_navigate_returns_navigation_timeout(self, policy):
        """Test that navigate uses navigation timeout."""
        timeout = policy.get_timeout_for_action(ActionType.NAVIGATE)
        assert timeout == policy.navigation_timeout

    def test_get_timeout_for_reload_returns_navigation_timeout(self, policy):
        """Test that reload uses navigation timeout."""
        timeout = policy.get_timeout_for_action(ActionType.RELOAD)
        assert timeout == policy.navigation_timeout

    def test_get_timeout_for_go_back_returns_navigation_timeout(self, policy):
        """Test that go_back uses navigation timeout."""
        timeout = policy.get_timeout_for_action(ActionType.GO_BACK)
        assert timeout == policy.navigation_timeout

    def test_custom_override_applied(self, policy):
        """Test that custom override is applied."""
        timeout = policy.get_timeout_for_action(ActionType.CLICK, custom_override=3000)
        assert timeout.value == 3000

    def test_action_specific_override(self, policy):
        """Test setting action-specific override."""
        policy.set_custom_timeout(ActionType.CLICK, 2000)
        timeout = policy.get_timeout_for_action(ActionType.CLICK)
        assert timeout.value == 2000

        # Other actions should not be affected
        timeout2 = policy.get_timeout_for_action(ActionType.TYPE)
        assert timeout2.value == 5000

    def test_clear_custom_timeout(self, policy):
        """Test clearing custom timeout."""
        policy.set_custom_timeout(ActionType.CLICK, 2000)
        policy.clear_custom_timeout(ActionType.CLICK)
        timeout = policy.get_timeout_for_action(ActionType.CLICK)
        assert timeout == policy.action_timeout

    def test_validate_clamps_to_bounds(self, policy):
        """Test that validate clamps values to bounds."""
        # Below minimum - should clamp to min
        result = policy.validate(50)
        assert result.value == TimeoutPolicy.MIN_TIMEOUT

        # Above maximum - should clamp to max
        result = policy.validate(500000)
        assert result.value == TimeoutPolicy.MAX_TIMEOUT

    def test_min_timeout_enforced(self):
        """Test that minimum timeout is enforced."""
        policy = TimeoutPolicy(action_timeout=50)  # Below default min
        assert policy.action_timeout.value >= TimeoutPolicy.MIN_TIMEOUT

    def test_max_timeout_enforced(self):
        """Test that maximum timeout is enforced."""
        policy = TimeoutPolicy(action_timeout=500000)  # Above default max
        assert policy.action_timeout.value <= TimeoutPolicy.MAX_TIMEOUT

    def test_custom_bounds(self):
        """Test custom min/max bounds."""
        policy = TimeoutPolicy(
            min_timeout=500,
            max_timeout=10000,
            action_timeout=200,  # Below custom min
            navigation_timeout=50000,  # Above custom max
        )

        assert policy.action_timeout.value == 500  # Clamped to custom min
        assert policy.navigation_timeout.value == 10000  # Clamped to custom max

    def test_negative_timeout_raises_error(self):
        """Test that negative timeout raises error."""
        policy = TimeoutPolicy()
        with pytest.raises(InvalidTimeoutError, match="cannot be negative"):
            policy.validate(-100)

    def test_to_dict_export(self, policy):
        """Test exporting configuration as dictionary."""
        config = policy.to_dict()

        assert "action_timeout" in config
        assert "navigation_timeout" in config
        assert "assertion_retry" in config
        assert "min_timeout" in config
        assert "max_timeout" in config
        assert config["action_timeout"] == 5000

    def test_from_dict_import(self):
        """Test creating policy from dictionary."""
        config = {
            "action_timeout": 3000,
            "navigation_timeout": 30000,
            "assertion_retry": 5000,
        }
        policy = TimeoutPolicy.from_dict(config)

        assert policy.action_timeout.value == 3000
        assert policy.navigation_timeout.value == 30000
        assert policy.assertion_retry.value == 5000


class TestTimeoutPolicyEdgeCases:
    """Edge case tests for TimeoutPolicy."""

    def test_zero_timeout_clamped_to_min(self):
        """Test that zero timeout is clamped to minimum."""
        policy = TimeoutPolicy()
        result = policy.validate(0)
        assert result.value == TimeoutPolicy.MIN_TIMEOUT

    def test_all_action_types_have_timeout(self):
        """Test that all action types return a valid timeout."""
        policy = TimeoutPolicy()

        for action in ActionType:
            timeout = policy.get_timeout_for_action(action)
            assert timeout.value > 0

    def test_wait_actions_use_appropriate_timeout(self):
        """Test wait actions use appropriate timeouts."""
        policy = TimeoutPolicy()

        # WAIT_FOR_NAVIGATION should use navigation timeout
        timeout = policy.get_timeout_for_action(ActionType.WAIT_FOR_NAVIGATION)
        assert timeout == policy.navigation_timeout

        # Other wait actions should use action timeout
        timeout = policy.get_timeout_for_action(ActionType.WAIT_FOR_ELEMENT)
        assert timeout == policy.action_timeout

    def test_read_actions_use_action_timeout(self):
        """Test read actions use action timeout."""
        policy = TimeoutPolicy()

        read_actions = [
            ActionType.GET_TEXT,
            ActionType.GET_VALUE,
            ActionType.GET_ATTRIBUTE,
            ActionType.GET_SNAPSHOT,
        ]

        for action in read_actions:
            timeout = policy.get_timeout_for_action(action)
            assert timeout == policy.action_timeout

    def test_override_priority(self):
        """Test that custom override takes priority over action-specific."""
        policy = TimeoutPolicy()

        # Set action-specific override
        policy.set_custom_timeout(ActionType.CLICK, 2000)

        # Custom override should take priority
        timeout = policy.get_timeout_for_action(ActionType.CLICK, custom_override=1000)
        assert timeout.value == 1000


class TestInvalidTimeoutError:
    """Tests for InvalidTimeoutError exception."""

    def test_error_contains_value_and_reason(self):
        """Test that error contains the value and reason."""
        error = InvalidTimeoutError(-100, "timeout cannot be negative")
        message = str(error)

        assert "-100" in message
        assert "cannot be negative" in message

    def test_error_stores_attributes(self):
        """Test that error stores value and reason."""
        error = InvalidTimeoutError(-100, "test reason")
        assert error.value == -100
        assert error.reason == "test reason"

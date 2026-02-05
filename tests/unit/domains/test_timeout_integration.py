"""Tests for Timeout Integration - Keyword Classification and Policy Application.

This module tests the integration of the timeout system with keyword execution:
- Keyword classification to ActionType (click=5s, navigate=60s, read=2s)
- Timeout policy retrieval from the container
- Timeout configuration passed to keyword execution
"""

from __future__ import annotations

import pytest
from typing import Dict, Optional, Set
from unittest.mock import MagicMock, Mock, patch

# Import production code from timeout domain
from robotmcp.domains.timeout import (
    ActionType,
    TimeoutCategory,
    TimeoutPolicy,
    TimeoutService,
    Milliseconds,
    DefaultTimeouts,
)
from robotmcp.container import ServiceContainer, get_container, reset_container


# =============================================================================
# Keyword Classification Tests
# =============================================================================


class TestKeywordToActionTypeMapping:
    """Tests for mapping Robot Framework keywords to ActionType."""

    # Mapping of common keywords to their expected ActionType
    CLICK_KEYWORDS = [
        "Click",
        "Click Element",
        "Click Button",
        "Click Link",
        "Click Image",
        "Double Click",
        "Right Click",
    ]

    FILL_KEYWORDS = [
        "Fill",
        "Fill Text",
        "Input Text",
        "Type Text",
        "Clear Text",
    ]

    NAVIGATION_KEYWORDS = [
        "Go To",
        "Navigate To",
        "Open Browser",
        "New Page",
        "Reload",
        "Go Back",
        "Go Forward",
    ]

    READ_KEYWORDS = [
        "Get Text",
        "Get Value",
        "Get Attribute",
        "Get Property",
        "Get Element States",
        "Get Aria Snapshot",
        "Take Screenshot",
    ]

    def test_classify_click_keywords(self):
        """Test that click-related keywords are classified correctly."""
        for keyword in self.CLICK_KEYWORDS:
            action_type = self._classify_keyword(keyword)
            category = TimeoutCategory.categorize(action_type)
            assert category == "action", f"Keyword '{keyword}' should be 'action' category"

    def test_classify_fill_keywords(self):
        """Test that fill/type keywords are classified correctly."""
        for keyword in self.FILL_KEYWORDS:
            action_type = self._classify_keyword(keyword)
            category = TimeoutCategory.categorize(action_type)
            assert category == "action", f"Keyword '{keyword}' should be 'action' category"

    def test_classify_navigation_keywords(self):
        """Test that navigation keywords are classified correctly."""
        for keyword in self.NAVIGATION_KEYWORDS:
            action_type = self._classify_keyword(keyword)
            category = TimeoutCategory.categorize(action_type)
            assert category == "navigation", f"Keyword '{keyword}' should be 'navigation' category"

    def test_classify_read_keywords(self):
        """Test that read keywords are classified correctly."""
        for keyword in self.READ_KEYWORDS:
            action_type = self._classify_keyword(keyword)
            category = TimeoutCategory.categorize(action_type)
            assert category == "read", f"Keyword '{keyword}' should be 'read' category"

    def _classify_keyword(self, keyword: str) -> ActionType:
        """Classify a keyword string to an ActionType.

        This helper maps Robot Framework keyword names to ActionType enum values
        based on common patterns.
        """
        keyword_lower = keyword.lower().replace(" ", "_")

        # Navigation keywords
        if any(nav in keyword_lower for nav in ["go_to", "navigate", "new_page", "reload", "go_back", "go_forward", "open_browser"]):
            return ActionType.NAVIGATE

        # Click keywords
        if any(click in keyword_lower for click in ["click", "double_click", "right_click"]):
            return ActionType.CLICK

        # Fill/Type keywords
        if any(fill in keyword_lower for fill in ["fill", "input", "type", "clear"]):
            return ActionType.FILL

        # Read keywords
        if any(read in keyword_lower for read in ["get_text", "get_value", "get_attribute", "get_property", "get_element", "get_aria", "screenshot"]):
            return ActionType.GET_TEXT

        # Default to action
        return ActionType.CLICK


class TestActionTypeClassification:
    """Tests for ActionType categorization via TimeoutCategory."""

    def test_click_action_is_element_action(self):
        """Test that CLICK is classified as element action (5s timeout)."""
        assert ActionType.CLICK in TimeoutCategory.ELEMENT_ACTIONS
        assert TimeoutCategory.categorize(ActionType.CLICK) == "action"

    def test_navigate_action_is_navigation(self):
        """Test that NAVIGATE is classified as navigation (60s timeout)."""
        assert ActionType.NAVIGATE in TimeoutCategory.NAVIGATION_ACTIONS
        assert TimeoutCategory.categorize(ActionType.NAVIGATE) == "navigation"

    def test_get_text_is_read_action(self):
        """Test that GET_TEXT is classified as read action (2s timeout)."""
        assert ActionType.GET_TEXT in TimeoutCategory.READ_ACTIONS
        assert TimeoutCategory.categorize(ActionType.GET_TEXT) == "read"

    def test_fill_is_element_action(self):
        """Test that FILL is classified as element action."""
        assert ActionType.FILL in TimeoutCategory.ELEMENT_ACTIONS
        assert TimeoutCategory.categorize(ActionType.FILL) == "action"

    def test_reload_is_navigation(self):
        """Test that RELOAD is classified as navigation."""
        assert ActionType.RELOAD in TimeoutCategory.NAVIGATION_ACTIONS
        assert TimeoutCategory.categorize(ActionType.RELOAD) == "navigation"

    def test_screenshot_is_read_action(self):
        """Test that SCREENSHOT is classified as read action."""
        assert ActionType.SCREENSHOT in TimeoutCategory.READ_ACTIONS
        assert TimeoutCategory.categorize(ActionType.SCREENSHOT) == "read"

    def test_all_action_types_have_category(self):
        """Test that every ActionType can be categorized."""
        for action in ActionType:
            category = TimeoutCategory.categorize(action)
            assert category in ("navigation", "action", "read", "assertion"), \
                f"ActionType.{action.name} should have a valid category"


# =============================================================================
# Timeout Value Tests (Default Configuration)
# =============================================================================


class TestDefaultTimeoutValues:
    """Tests for default timeout values."""

    def test_action_timeout_is_5_seconds(self):
        """Test that default action timeout is 5 seconds (5000ms)."""
        assert DefaultTimeouts.ACTION.value == 5000
        assert DefaultTimeouts.ACTION.to_seconds() == 5.0

    def test_navigation_timeout_is_60_seconds(self):
        """Test that default navigation timeout is 60 seconds (60000ms)."""
        assert DefaultTimeouts.NAVIGATION.value == 60000
        assert DefaultTimeouts.NAVIGATION.to_seconds() == 60.0

    def test_read_timeout_is_2_seconds(self):
        """Test that default read timeout is 2 seconds (2000ms)."""
        assert DefaultTimeouts.READ.value == 2000
        assert DefaultTimeouts.READ.to_seconds() == 2.0

    def test_assertion_timeout_is_10_seconds(self):
        """Test that default assertion timeout is 10 seconds (10000ms)."""
        assert DefaultTimeouts.ASSERTION.value == 10000
        assert DefaultTimeouts.ASSERTION.to_seconds() == 10.0

    def test_min_timeout_is_100ms(self):
        """Test that minimum timeout is 100ms."""
        assert DefaultTimeouts.MIN_TIMEOUT.value == 100

    def test_max_action_timeout_is_30_seconds(self):
        """Test that maximum action timeout is 30 seconds."""
        assert DefaultTimeouts.MAX_ACTION_TIMEOUT.value == 30000

    def test_max_navigation_timeout_is_5_minutes(self):
        """Test that maximum navigation timeout is 5 minutes."""
        assert DefaultTimeouts.MAX_NAVIGATION_TIMEOUT.value == 300000


class TestTimeoutPolicyGetTimeoutFor:
    """Tests for TimeoutPolicy.get_timeout_for() method."""

    @pytest.fixture
    def policy(self) -> TimeoutPolicy:
        """Create a default TimeoutPolicy for testing."""
        return TimeoutPolicy.create_default("test_session")

    def test_click_timeout_is_5_seconds(self, policy):
        """Test that click action uses 5 second timeout."""
        timeout = policy.get_timeout_for(ActionType.CLICK)
        assert timeout.value == 5000

    def test_navigate_timeout_is_60_seconds(self, policy):
        """Test that navigate action uses 60 second timeout."""
        timeout = policy.get_timeout_for(ActionType.NAVIGATE)
        assert timeout.value == 60000

    def test_get_text_timeout_is_2_seconds(self, policy):
        """Test that get_text action uses 2 second timeout."""
        timeout = policy.get_timeout_for(ActionType.GET_TEXT)
        assert timeout.value == 2000

    def test_fill_timeout_is_5_seconds(self, policy):
        """Test that fill action uses 5 second timeout."""
        timeout = policy.get_timeout_for(ActionType.FILL)
        assert timeout.value == 5000

    def test_reload_timeout_is_60_seconds(self, policy):
        """Test that reload action uses 60 second timeout."""
        timeout = policy.get_timeout_for(ActionType.RELOAD)
        assert timeout.value == 60000

    def test_wait_for_element_timeout_is_10_seconds(self, policy):
        """Test that wait_for_element uses assertion timeout (10s)."""
        timeout = policy.get_timeout_for(ActionType.WAIT_FOR_ELEMENT)
        assert timeout.value == 10000

    def test_all_element_actions_use_action_timeout(self, policy):
        """Test that all element actions use the action timeout."""
        for action in TimeoutCategory.ELEMENT_ACTIONS:
            timeout = policy.get_timeout_for(action)
            assert timeout.value == 5000, f"{action.name} should use 5000ms timeout"

    def test_all_navigation_actions_use_navigation_timeout(self, policy):
        """Test that all navigation actions use the navigation timeout."""
        for action in TimeoutCategory.NAVIGATION_ACTIONS:
            timeout = policy.get_timeout_for(action)
            assert timeout.value == 60000, f"{action.name} should use 60000ms timeout"

    def test_all_read_actions_use_read_timeout(self, policy):
        """Test that all read actions use the read timeout."""
        for action in TimeoutCategory.READ_ACTIONS:
            timeout = policy.get_timeout_for(action)
            assert timeout.value == 2000, f"{action.name} should use 2000ms timeout"


# =============================================================================
# Container Integration Tests
# =============================================================================


class TestContainerTimeoutIntegration:
    """Tests for timeout policy retrieval from the container."""

    @pytest.fixture(autouse=True)
    def reset_container_state(self):
        """Reset the container before and after each test."""
        reset_container()
        yield
        reset_container()

    def test_container_provides_timeout_service(self):
        """Test that the container provides a TimeoutService."""
        container = get_container()
        service = container.timeout_service

        assert service is not None
        assert isinstance(service, TimeoutService)

    def test_container_creates_default_policy_for_session(self):
        """Test that the container creates a default policy for a session."""
        container = get_container()
        policy = container.get_timeout_policy("session_123")

        assert policy is not None
        assert isinstance(policy, TimeoutPolicy)
        assert policy.session_id == "session_123"

    def test_container_returns_same_policy_for_same_session(self):
        """Test that the container returns the same policy instance for a session."""
        container = get_container()
        policy1 = container.get_timeout_policy("session_123")
        policy2 = container.get_timeout_policy("session_123")

        # Should be the same object (identity)
        assert policy1 is policy2

    def test_container_returns_different_policies_for_different_sessions(self):
        """Test that different sessions get different policies."""
        container = get_container()
        policy1 = container.get_timeout_policy("session_1")
        policy2 = container.get_timeout_policy("session_2")

        # Should be different objects
        assert policy1 is not policy2
        assert policy1.session_id == "session_1"
        assert policy2.session_id == "session_2"

    def test_container_policy_has_correct_defaults(self):
        """Test that policies from the container have correct default values."""
        container = get_container()
        policy = container.get_timeout_policy("test_session")

        assert policy.action_timeout.value == 5000
        assert policy.navigation_timeout.value == 60000
        assert policy.read_timeout.value == 2000
        assert policy.assertion_retry.value == 10000

    def test_container_clear_session_removes_policy(self):
        """Test that clearing a session removes its timeout policy."""
        container = get_container()
        policy1 = container.get_timeout_policy("session_to_clear")

        container.clear_session("session_to_clear")

        # Getting policy again should create a new one
        policy2 = container.get_timeout_policy("session_to_clear")
        assert policy1 is not policy2


# =============================================================================
# Timeout Service Tests
# =============================================================================


class TestTimeoutService:
    """Tests for TimeoutService."""

    @pytest.fixture
    def service(self) -> TimeoutService:
        """Create a TimeoutService for testing."""
        return TimeoutService()

    def test_create_default_policy(self, service):
        """Test creating a default timeout policy."""
        policy = service.create_default_policy("session_1")

        assert policy is not None
        assert policy.session_id == "session_1"
        assert policy.action_timeout.value == 5000
        assert policy.navigation_timeout.value == 60000

    def test_get_policy_returns_created_policy(self, service):
        """Test that get_policy returns the created policy."""
        created = service.create_default_policy("session_1")
        retrieved = service.get_policy("session_1")

        assert retrieved is created

    def test_get_policy_returns_none_for_unknown_session(self, service):
        """Test that get_policy returns None for unknown sessions."""
        policy = service.get_policy("unknown_session")
        assert policy is None

    def test_get_or_create_policy_creates_if_missing(self, service):
        """Test that get_or_create_policy creates a policy if none exists."""
        policy = service.get_or_create_policy("new_session")

        assert policy is not None
        assert policy.session_id == "new_session"

    def test_get_or_create_policy_returns_existing(self, service):
        """Test that get_or_create_policy returns existing policy."""
        created = service.create_default_policy("session_1")
        retrieved = service.get_or_create_policy("session_1")

        assert retrieved is created

    def test_get_timeout_for_action_uses_policy(self, service):
        """Test getting timeout for an action using a policy."""
        policy = service.create_default_policy("session_1")

        timeout = service.get_timeout_for_action(policy, ActionType.CLICK)
        assert timeout.value == 5000

        timeout = service.get_timeout_for_action(policy, ActionType.NAVIGATE)
        assert timeout.value == 60000


# =============================================================================
# Custom Timeout Override Tests
# =============================================================================


class TestTimeoutOverrides:
    """Tests for custom timeout overrides."""

    @pytest.fixture
    def policy(self) -> TimeoutPolicy:
        """Create a default TimeoutPolicy for testing."""
        return TimeoutPolicy.create_default("test_session")

    def test_custom_override_for_click(self, policy):
        """Test setting a custom timeout override for click."""
        custom_policy = policy.with_override(ActionType.CLICK, Milliseconds(10000))

        timeout = custom_policy.get_timeout_for(ActionType.CLICK)
        assert timeout.value == 10000

    def test_override_does_not_affect_other_actions(self, policy):
        """Test that overriding one action doesn't affect others."""
        custom_policy = policy.with_override(ActionType.CLICK, Milliseconds(10000))

        # Click should be overridden
        assert custom_policy.get_timeout_for(ActionType.CLICK).value == 10000

        # Fill should still use default
        assert custom_policy.get_timeout_for(ActionType.FILL).value == 5000

        # Navigate should still use default
        assert custom_policy.get_timeout_for(ActionType.NAVIGATE).value == 60000

    def test_clear_override_restores_default(self, policy):
        """Test that clearing an override restores the default."""
        custom_policy = policy.with_override(ActionType.CLICK, Milliseconds(10000))
        restored_policy = custom_policy.clear_override(ActionType.CLICK)

        timeout = restored_policy.get_timeout_for(ActionType.CLICK)
        assert timeout.value == 5000

    def test_override_is_clamped_to_bounds(self, policy):
        """Test that overrides are clamped to valid bounds."""
        # Try to set very short timeout - should be clamped to minimum
        short_policy = policy.with_override(ActionType.CLICK, Milliseconds(50))
        assert short_policy.get_timeout_for(ActionType.CLICK).value >= 100

        # Try to set very long timeout - should be clamped to maximum
        long_policy = policy.with_override(ActionType.CLICK, Milliseconds(500000))
        assert long_policy.get_timeout_for(ActionType.CLICK).value <= 30000

    def test_has_override_returns_true_when_override_exists(self, policy):
        """Test has_override returns True when an override exists."""
        assert not policy.has_override(ActionType.CLICK)

        custom_policy = policy.with_override(ActionType.CLICK, Milliseconds(10000))
        assert custom_policy.has_override(ActionType.CLICK)

    def test_policy_immutability(self, policy):
        """Test that with_override returns a new policy, not mutating original."""
        original_timeout = policy.get_timeout_for(ActionType.CLICK).value

        custom_policy = policy.with_override(ActionType.CLICK, Milliseconds(10000))

        # Original should be unchanged
        assert policy.get_timeout_for(ActionType.CLICK).value == original_timeout
        assert not policy.has_override(ActionType.CLICK)

        # New policy should have the override
        assert custom_policy.get_timeout_for(ActionType.CLICK).value == 10000
        assert custom_policy.has_override(ActionType.CLICK)


# =============================================================================
# Milliseconds Value Object Tests
# =============================================================================


class TestMillisecondsValueObject:
    """Tests for Milliseconds value object."""

    def test_create_milliseconds(self):
        """Test creating Milliseconds from integer value."""
        ms = Milliseconds(5000)
        assert ms.value == 5000

    def test_create_from_seconds(self):
        """Test creating Milliseconds from seconds."""
        ms = Milliseconds.seconds(5.0)
        assert ms.value == 5000

        ms = Milliseconds.seconds(2.5)
        assert ms.value == 2500

    def test_to_seconds_conversion(self):
        """Test converting Milliseconds to seconds."""
        ms = Milliseconds(5000)
        assert ms.to_seconds() == 5.0

        ms = Milliseconds(2500)
        assert ms.to_seconds() == 2.5

    def test_cannot_be_negative(self):
        """Test that Milliseconds cannot be negative."""
        with pytest.raises(ValueError, match="negative"):
            Milliseconds(-1)

    def test_zero_is_allowed(self):
        """Test that zero milliseconds is allowed."""
        ms = Milliseconds(0)
        assert ms.value == 0

    def test_string_representation(self):
        """Test string representation of Milliseconds."""
        ms = Milliseconds(5000)
        assert str(ms) == "5000ms"

    def test_comparison_operators(self):
        """Test comparison operators."""
        ms1 = Milliseconds(1000)
        ms2 = Milliseconds(2000)
        ms3 = Milliseconds(1000)

        assert ms1 < ms2
        assert ms2 > ms1
        assert ms1 <= ms3
        assert ms1 >= ms3
        assert ms1 == ms3
        assert ms1 != ms2

    def test_addition(self):
        """Test adding Milliseconds."""
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

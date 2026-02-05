"""Integration Tests for Timeout Behavior.

This module tests the end-to-end timeout behavior in keyword execution:
- execute_step with default timeout
- execute_step with custom timeout_ms
- Click timeout is 5s by default
- Navigation timeout is 60s by default
- Pre-validation fails fast for invisible elements
"""

from __future__ import annotations

import asyncio
import time
import pytest
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

# Import production code
from robotmcp.domains.timeout import (
    ActionType,
    TimeoutCategory,
    TimeoutPolicy,
    TimeoutService,
    Milliseconds,
    DefaultTimeouts,
)
from robotmcp.domains.action import (
    PreValidator,
    PreValidationResult,
)
from robotmcp.container import ServiceContainer, get_container, reset_container


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_container_fixture():
    """Reset the container before and after each test."""
    reset_container()
    yield
    reset_container()


@pytest.fixture
def timeout_service() -> TimeoutService:
    """Create a fresh TimeoutService for testing."""
    return TimeoutService()


@pytest.fixture
def timeout_policy(timeout_service) -> TimeoutPolicy:
    """Create a default timeout policy."""
    return timeout_service.create_default_policy("test_session")


# =============================================================================
# Execute Step Default Timeout Tests
# =============================================================================


class TestExecuteStepDefaultTimeout:
    """Tests for execute_step with default timeout configuration."""

    def test_click_uses_default_5_second_timeout(self, timeout_policy):
        """Test that click action uses the default 5 second timeout."""
        timeout = timeout_policy.get_timeout_for(ActionType.CLICK)
        assert timeout.value == 5000, "Click should default to 5000ms timeout"

    def test_fill_uses_default_5_second_timeout(self, timeout_policy):
        """Test that fill action uses the default 5 second timeout."""
        timeout = timeout_policy.get_timeout_for(ActionType.FILL)
        assert timeout.value == 5000, "Fill should default to 5000ms timeout"

    def test_navigate_uses_default_60_second_timeout(self, timeout_policy):
        """Test that navigate action uses the default 60 second timeout."""
        timeout = timeout_policy.get_timeout_for(ActionType.NAVIGATE)
        assert timeout.value == 60000, "Navigate should default to 60000ms timeout"

    def test_read_uses_default_2_second_timeout(self, timeout_policy):
        """Test that read action uses the default 2 second timeout."""
        timeout = timeout_policy.get_timeout_for(ActionType.GET_TEXT)
        assert timeout.value == 2000, "Get Text should default to 2000ms timeout"

    def test_container_provides_policy_with_correct_defaults(self):
        """Test that container provides policy with correct default timeouts."""
        container = get_container()
        policy = container.get_timeout_policy("integration_test_session")

        assert policy.action_timeout.value == 5000
        assert policy.navigation_timeout.value == 60000
        assert policy.read_timeout.value == 2000
        assert policy.assertion_retry.value == 10000


class TestExecuteStepCustomTimeout:
    """Tests for execute_step with custom timeout_ms override."""

    def test_custom_timeout_overrides_default(self, timeout_policy):
        """Test that custom timeout override takes precedence over default."""
        # Create policy with custom override
        custom_policy = timeout_policy.with_override(ActionType.CLICK, Milliseconds(10000))

        timeout = custom_policy.get_timeout_for(ActionType.CLICK)
        assert timeout.value == 10000, "Custom override should be 10000ms"

    def test_custom_timeout_does_not_affect_other_actions(self, timeout_policy):
        """Test that custom timeout for one action doesn't affect others."""
        custom_policy = timeout_policy.with_override(ActionType.CLICK, Milliseconds(10000))

        # Click should be overridden
        click_timeout = custom_policy.get_timeout_for(ActionType.CLICK)
        assert click_timeout.value == 10000

        # Fill should still use default
        fill_timeout = custom_policy.get_timeout_for(ActionType.FILL)
        assert fill_timeout.value == 5000

    def test_timeout_service_applies_override(self, timeout_service):
        """Test that TimeoutService correctly applies overrides."""
        policy = timeout_service.create_default_policy("session_1")

        # Set custom override via service
        updated_policy = timeout_service.set_override(
            "session_1", ActionType.NAVIGATE, Milliseconds(30000)
        )

        timeout = updated_policy.get_timeout_for(ActionType.NAVIGATE)
        assert timeout.value == 30000

    def test_custom_timeout_is_clamped_to_bounds(self, timeout_policy):
        """Test that custom timeout values are clamped to valid bounds."""
        # Very short timeout should be clamped to minimum (100ms)
        short_policy = timeout_policy.with_override(ActionType.CLICK, Milliseconds(50))
        short_timeout = short_policy.get_timeout_for(ActionType.CLICK)
        assert short_timeout.value >= 100, "Timeout should be clamped to minimum 100ms"

        # Very long timeout should be clamped to maximum (30000ms for actions)
        long_policy = timeout_policy.with_override(ActionType.CLICK, Milliseconds(500000))
        long_timeout = long_policy.get_timeout_for(ActionType.CLICK)
        assert long_timeout.value <= 30000, "Timeout should be clamped to maximum 30000ms"


# =============================================================================
# Click Timeout Tests (5 seconds default)
# =============================================================================


class TestClickTimeoutBehavior:
    """Tests for click action timeout behavior."""

    def test_click_timeout_is_5_seconds_by_default(self, timeout_policy):
        """Test that click timeout is 5 seconds (5000ms) by default."""
        timeout = timeout_policy.get_timeout_for(ActionType.CLICK)
        assert timeout.value == 5000

    def test_click_timeout_can_be_overridden(self, timeout_policy):
        """Test that click timeout can be customized."""
        custom_policy = timeout_policy.with_override(ActionType.CLICK, Milliseconds(3000))
        timeout = custom_policy.get_timeout_for(ActionType.CLICK)
        assert timeout.value == 3000

    def test_all_click_variants_use_same_timeout(self, timeout_policy):
        """Test that click variants use the action timeout."""
        # These are all element actions that should use action_timeout
        click_actions = [
            ActionType.CLICK,
            ActionType.FILL,
            ActionType.TYPE,
            ActionType.HOVER,
            ActionType.FOCUS,
            ActionType.CHECK,
            ActionType.UNCHECK,
        ]

        for action in click_actions:
            timeout = timeout_policy.get_timeout_for(action)
            assert timeout.value == 5000, f"{action.name} should use 5000ms timeout"

    def test_click_category_is_action(self):
        """Test that click is categorized as 'action'."""
        category = TimeoutCategory.categorize(ActionType.CLICK)
        assert category == "action"

    def test_click_is_in_element_actions_set(self):
        """Test that click is in the ELEMENT_ACTIONS set."""
        assert ActionType.CLICK in TimeoutCategory.ELEMENT_ACTIONS


# =============================================================================
# Navigation Timeout Tests (60 seconds default)
# =============================================================================


class TestNavigationTimeoutBehavior:
    """Tests for navigation action timeout behavior."""

    def test_navigation_timeout_is_60_seconds_by_default(self, timeout_policy):
        """Test that navigation timeout is 60 seconds (60000ms) by default."""
        timeout = timeout_policy.get_timeout_for(ActionType.NAVIGATE)
        assert timeout.value == 60000

    def test_navigation_timeout_can_be_overridden(self, timeout_policy):
        """Test that navigation timeout can be customized."""
        custom_policy = timeout_policy.with_override(ActionType.NAVIGATE, Milliseconds(30000))
        timeout = custom_policy.get_timeout_for(ActionType.NAVIGATE)
        assert timeout.value == 30000

    def test_all_navigation_actions_use_navigation_timeout(self, timeout_policy):
        """Test that all navigation actions use the navigation timeout."""
        navigation_actions = [
            ActionType.NAVIGATE,
            ActionType.RELOAD,
            ActionType.GO_BACK,
            ActionType.GO_FORWARD,
            ActionType.WAIT_FOR_NAVIGATION,
        ]

        for action in navigation_actions:
            timeout = timeout_policy.get_timeout_for(action)
            assert timeout.value == 60000, f"{action.name} should use 60000ms timeout"

    def test_navigate_category_is_navigation(self):
        """Test that navigate is categorized as 'navigation'."""
        category = TimeoutCategory.categorize(ActionType.NAVIGATE)
        assert category == "navigation"

    def test_navigate_is_in_navigation_actions_set(self):
        """Test that navigate is in the NAVIGATION_ACTIONS set."""
        assert ActionType.NAVIGATE in TimeoutCategory.NAVIGATION_ACTIONS

    def test_navigation_timeout_has_higher_max_bound(self):
        """Test that navigation has a higher maximum timeout bound."""
        _, nav_max = DefaultTimeouts.get_bounds_for_category("navigation")
        _, action_max = DefaultTimeouts.get_bounds_for_category("action")

        assert nav_max > action_max
        assert nav_max.value == 300000  # 5 minutes
        assert action_max.value == 30000  # 30 seconds


# =============================================================================
# Pre-Validation Fast Failure Tests
# =============================================================================


class TestPreValidationFailsFast:
    """Tests that pre-validation fails fast for invisible elements."""

    @pytest.fixture
    def pre_validator(self) -> PreValidator:
        """Create a PreValidator for testing."""
        return PreValidator()

    def test_pre_validation_returns_quickly_for_invisible(self, pre_validator):
        """Test that pre-validation doesn't wait the full timeout for invisible elements."""
        # This test verifies the design intent: pre-validation should detect
        # problems immediately, not wait for the action timeout to expire.

        # Mock registry and adapter
        from tests.unit.test_pre_validation import MockElementRegistry, MockBrowserAdapter

        registry = MockElementRegistry()
        adapter = MockBrowserAdapter(
            visible=False,
            enabled=True,
            states={"enabled", "attached", "hidden"},
        )

        from robotmcp.domains.shared.kernel import ElementRef
        ref = ElementRef(value="e1")

        # Measure execution time
        start = time.perf_counter()
        result = pre_validator.run_all_checks("click", ref, registry, adapter)
        elapsed = time.perf_counter() - start

        # Should fail fast - not wait for action timeout (5s)
        assert elapsed < 0.5, f"Pre-validation took {elapsed*1000:.2f}ms, should be <500ms (not wait for 5s timeout)"
        assert result.passed is False
        assert "visible" in result.failed_checks

    def test_pre_validation_returns_quickly_for_disabled(self, pre_validator):
        """Test that pre-validation doesn't wait for disabled elements."""
        from tests.unit.test_pre_validation import MockElementRegistry, MockBrowserAdapter

        registry = MockElementRegistry()
        adapter = MockBrowserAdapter(
            visible=True,
            enabled=False,
            states={"visible", "attached", "disabled"},
        )

        from robotmcp.domains.shared.kernel import ElementRef
        ref = ElementRef(value="e1")

        start = time.perf_counter()
        result = pre_validator.run_all_checks("click", ref, registry, adapter)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, f"Pre-validation took {elapsed*1000:.2f}ms, should be <500ms"
        assert result.passed is False
        assert "enabled" in result.failed_checks

    def test_pre_validation_skips_for_navigation(self, pre_validator):
        """Test that pre-validation is skipped for navigation actions."""
        from tests.unit.test_pre_validation import MockElementRegistry, MockBrowserAdapter

        registry = MockElementRegistry()
        adapter = MockBrowserAdapter()

        from robotmcp.domains.shared.kernel import ElementRef
        ref = ElementRef(value="e1")

        start = time.perf_counter()
        result = pre_validator.run_all_checks("navigate", ref, registry, adapter)
        elapsed = time.perf_counter() - start

        # Should return immediately with skipped status
        assert elapsed < 0.1, f"Navigation validation took {elapsed*1000:.2f}ms, should be <100ms"
        assert result.passed is True
        assert "skipped" in result.checks_performed


# =============================================================================
# Container Timeout Settings Integration Tests
# =============================================================================


class TestContainerTimeoutSettings:
    """Tests for container-based timeout settings retrieval."""

    def test_get_timeout_settings_for_click(self):
        """Test getting timeout settings for click action type."""
        container = get_container()
        settings = container.get_timeout_settings("click")

        # Should return a dict with timeout_ms key
        assert "timeout_ms" in settings
        assert settings["timeout_ms"] == 5000

    def test_get_timeout_settings_for_navigate(self):
        """Test getting timeout settings for navigate action type."""
        container = get_container()
        settings = container.get_timeout_settings("navigate")

        assert "timeout_ms" in settings
        assert settings["timeout_ms"] == 60000

    def test_get_timeout_settings_for_go_to(self):
        """Test getting timeout settings for go_to action type."""
        container = get_container()
        settings = container.get_timeout_settings("go_to")

        # go_to is a navigation action
        assert "timeout_ms" in settings
        assert settings["timeout_ms"] == 60000

    def test_get_timeout_settings_for_reload(self):
        """Test getting timeout settings for reload action type."""
        container = get_container()
        settings = container.get_timeout_settings("reload")

        assert "timeout_ms" in settings
        assert settings["timeout_ms"] == 60000

    def test_get_timeout_settings_for_unknown_action(self):
        """Test getting timeout settings for unknown action defaults to action timeout."""
        container = get_container()
        settings = container.get_timeout_settings("unknown_action")

        # Unknown actions should default to action timeout
        assert "timeout_ms" in settings
        assert settings["timeout_ms"] == 5000


# =============================================================================
# Timeout Recording and Events Tests
# =============================================================================


class TestTimeoutEventsAndRecording:
    """Tests for timeout event recording and publishing."""

    def test_timeout_exceeded_event_is_recorded(self, timeout_service):
        """Test that timeout exceeded events are recorded."""
        event = timeout_service.record_timeout_exceeded(
            execution_id="exec_123",
            action=ActionType.CLICK,
            timeout_used=Milliseconds(5000),
            elapsed=Milliseconds(6000),
            error_message="Element not found within timeout",
        )

        assert event is not None
        assert event.execution_id == "exec_123"
        assert event.action_type == ActionType.CLICK
        assert event.timeout_used.value == 5000
        assert event.elapsed.value == 6000

    def test_timeout_warning_is_issued_when_threshold_exceeded(self, timeout_service):
        """Test that timeout warnings are issued when elapsed time exceeds threshold."""
        warning = timeout_service.check_timeout_warning(
            execution_id="exec_123",
            action=ActionType.CLICK,
            timeout_configured=Milliseconds(5000),
            elapsed=Milliseconds(4000),  # 80% of timeout
            warning_threshold=0.75,  # Warn at 75%
        )

        assert warning is not None
        assert warning.execution_id == "exec_123"
        assert warning.action_type == ActionType.CLICK

    def test_no_warning_when_below_threshold(self, timeout_service):
        """Test that no warning is issued when elapsed time is below threshold."""
        warning = timeout_service.check_timeout_warning(
            execution_id="exec_123",
            action=ActionType.CLICK,
            timeout_configured=Milliseconds(5000),
            elapsed=Milliseconds(1000),  # 20% of timeout
            warning_threshold=0.75,  # Warn at 75%
        )

        assert warning is None


# =============================================================================
# Policy Serialization Tests
# =============================================================================


class TestTimeoutPolicySerialization:
    """Tests for timeout policy serialization/deserialization."""

    def test_policy_to_dict(self, timeout_policy):
        """Test converting policy to dictionary."""
        data = timeout_policy.to_dict()

        assert "policy_id" in data
        assert "session_id" in data
        assert "action_timeout_ms" in data
        assert "navigation_timeout_ms" in data
        assert "read_timeout_ms" in data
        assert data["action_timeout_ms"] == 5000
        assert data["navigation_timeout_ms"] == 60000
        assert data["read_timeout_ms"] == 2000

    def test_policy_from_dict(self):
        """Test creating policy from dictionary."""
        data = {
            "policy_id": "policy_test",
            "session_id": "test_session",
            "action_timeout_ms": 3000,
            "navigation_timeout_ms": 45000,
            "read_timeout_ms": 1500,
            "assertion_retry_ms": 8000,
        }

        policy = TimeoutPolicy.from_dict(data)

        assert policy.action_timeout.value == 3000
        assert policy.navigation_timeout.value == 45000
        assert policy.read_timeout.value == 1500
        assert policy.assertion_retry.value == 8000

    def test_policy_round_trip(self, timeout_policy):
        """Test that policy survives round-trip serialization."""
        # Add a custom override
        custom_policy = timeout_policy.with_override(ActionType.CLICK, Milliseconds(7500))

        # Serialize and deserialize
        data = custom_policy.to_dict()
        restored = TimeoutPolicy.from_dict(data)

        # Check values are preserved
        assert restored.action_timeout.value == custom_policy.action_timeout.value
        assert restored.navigation_timeout.value == custom_policy.navigation_timeout.value
        assert restored.get_timeout_for(ActionType.CLICK).value == 7500


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestTimeoutConcurrentAccess:
    """Tests for concurrent access to timeout policies."""

    def test_multiple_sessions_are_isolated(self, timeout_service):
        """Test that multiple sessions have isolated policies."""
        policy1 = timeout_service.create_default_policy("session_1")
        policy2 = timeout_service.create_default_policy("session_2")

        # Modify session 1's policy
        timeout_service.set_override("session_1", ActionType.CLICK, Milliseconds(10000))

        # Session 2 should not be affected
        policy1_updated = timeout_service.get_policy("session_1")
        policy2_unchanged = timeout_service.get_policy("session_2")

        assert policy1_updated.get_timeout_for(ActionType.CLICK).value == 10000
        assert policy2_unchanged.get_timeout_for(ActionType.CLICK).value == 5000

    def test_container_sessions_are_isolated(self):
        """Test that container-managed sessions are isolated."""
        container = get_container()

        # Get policies for two different sessions
        policy1 = container.get_timeout_policy("session_a")
        policy2 = container.get_timeout_policy("session_b")

        # They should be different objects
        assert policy1 is not policy2
        assert policy1.session_id == "session_a"
        assert policy2.session_id == "session_b"

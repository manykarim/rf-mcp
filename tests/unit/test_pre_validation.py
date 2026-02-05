"""Tests for Pre-Validation Service.

This module tests the pre-validation functionality:
- Pre-validation detects invisible elements
- Pre-validation detects disabled elements
- Pre-validation passes for valid elements
- Pre-validation timeout is fast (<500ms)
"""

from __future__ import annotations

import time
import pytest
from typing import Set
from unittest.mock import MagicMock, Mock, patch
from dataclasses import dataclass

# Import production code
from robotmcp.domains.action import (
    PreValidator,
    PreValidationResult,
)
from robotmcp.domains.shared.kernel import ElementRef


# =============================================================================
# Helper Functions
# =============================================================================


def create_element_ref(ref_str: str) -> ElementRef:
    """Create an ElementRef from a string like 'e1'.

    Args:
        ref_str: String in format 'e{number}' (e.g., 'e1', 'e42')

    Returns:
        ElementRef instance
    """
    return ElementRef(value=ref_str)


# =============================================================================
# Mock Implementations for Testing
# =============================================================================


@dataclass
class MockLocator:
    """Mock locator for testing."""

    selector: str

    def to_browser_library(self) -> str:
        return self.selector

    def to_selenium_library(self) -> str:
        return self.selector


class MockElementRegistry:
    """Mock element registry for testing."""

    def __init__(
        self,
        refs: Set[str] = None,
        stale: bool = False,
        locator_map: dict = None,
    ):
        self._refs = refs or {"e1", "e2", "e3"}
        self._stale = stale
        self._locator_map = locator_map or {
            "e1": MockLocator("css=#element1"),
            "e2": MockLocator("css=#element2"),
            "e3": MockLocator("css=#element3"),
        }

    def has_ref(self, ref: ElementRef) -> bool:
        # ElementRef has .value attribute containing the string like "e1"
        return ref.value in self._refs

    def is_stale(self) -> bool:
        return self._stale

    def get_locator(self, ref: ElementRef) -> MockLocator:
        # ElementRef has .value attribute containing the string like "e1"
        if ref.value not in self._locator_map:
            raise KeyError(f"Unknown ref: {ref.value}")
        return self._locator_map[ref.value]


class MockBrowserAdapter:
    """Mock browser adapter for testing."""

    def __init__(
        self,
        visible: bool = True,
        enabled: bool = True,
        editable: bool = True,
        stable: bool = True,
        receives_pointer: bool = True,
        states: Set[str] = None,
    ):
        self._visible = visible
        self._enabled = enabled
        self._editable = editable
        self._stable = stable
        self._receives_pointer = receives_pointer
        self._states = states or {"visible", "enabled", "attached"}

    def is_element_visible(self, locator: str) -> bool:
        return self._visible

    def is_element_enabled(self, locator: str) -> bool:
        return self._enabled

    def is_element_editable(self, locator: str) -> bool:
        return self._editable

    def is_element_stable(self, locator: str) -> bool:
        return self._stable

    def receives_pointer_events(self, locator: str) -> bool:
        return self._receives_pointer

    def get_element_states(self, locator: str) -> Set[str]:
        return self._states


# =============================================================================
# Pre-Validator Tests
# =============================================================================


class TestPreValidatorBasics:
    """Basic tests for PreValidator service."""

    @pytest.fixture
    def validator(self) -> PreValidator:
        """Create a PreValidator for testing."""
        return PreValidator()

    def test_should_validate_click(self, validator):
        """Test that click action requires validation."""
        assert validator.should_validate("click") is True
        assert validator.should_validate("Click") is True
        assert validator.should_validate("Click Element") is True

    def test_should_validate_fill(self, validator):
        """Test that fill action requires validation."""
        assert validator.should_validate("fill") is True
        assert validator.should_validate("Fill Text") is True
        assert validator.should_validate("Input Text") is True

    def test_should_not_validate_navigation(self, validator):
        """Test that navigation actions don't require element validation."""
        assert validator.should_validate("navigate") is False
        assert validator.should_validate("go_to") is False
        assert validator.should_validate("Go To") is False
        assert validator.should_validate("reload") is False
        assert validator.should_validate("go_back") is False
        assert validator.should_validate("go_forward") is False

    def test_should_not_validate_page_level_actions(self, validator):
        """Test that page-level actions don't require element validation."""
        assert validator.should_validate("get_url") is False
        assert validator.should_validate("get_title") is False
        assert validator.should_validate("get_page_source") is False
        assert validator.should_validate("take_screenshot") is False

    def test_get_required_states_for_click(self, validator):
        """Test required states for click action."""
        states = validator.get_required_states("click")
        assert "visible" in states
        assert "enabled" in states
        assert "stable" in states

    def test_get_required_states_for_fill(self, validator):
        """Test required states for fill action."""
        states = validator.get_required_states("fill")
        assert "visible" in states
        assert "enabled" in states
        assert "editable" in states

    def test_get_required_states_for_hover(self, validator):
        """Test required states for hover action."""
        states = validator.get_required_states("hover")
        assert "visible" in states
        assert "stable" in states

    def test_get_required_states_for_focus(self, validator):
        """Test required states for focus action."""
        states = validator.get_required_states("focus")
        assert "visible" in states


# =============================================================================
# Pre-Validation Detection Tests
# =============================================================================


class TestPreValidationDetectsInvisible:
    """Tests that pre-validation detects invisible elements."""

    @pytest.fixture
    def validator(self) -> PreValidator:
        """Create a PreValidator for testing."""
        return PreValidator()

    @pytest.fixture
    def registry(self) -> MockElementRegistry:
        """Create a mock element registry."""
        return MockElementRegistry()

    def test_detects_invisible_element(self, validator, registry):
        """Test that pre-validation detects invisible elements."""
        # Create adapter with invisible element
        adapter = MockBrowserAdapter(
            visible=False,
            enabled=True,
            states={"enabled", "attached", "hidden"},
        )

        ref = create_element_ref("e1")
        result = validator.run_all_checks("click", ref, registry, adapter)

        assert result.passed is False
        assert "visible" in result.failed_checks
        assert "visible" in result.missing_states

    def test_passes_for_visible_element(self, validator, registry):
        """Test that pre-validation passes for visible elements."""
        adapter = MockBrowserAdapter(
            visible=True,
            enabled=True,
            stable=True,
            states={"visible", "enabled", "attached", "stable"},
        )

        ref = create_element_ref("e1")
        result = validator.run_all_checks("click", ref, registry, adapter)

        assert result.passed is True
        assert "visible" not in result.failed_checks

    def test_failure_reason_mentions_visibility(self, validator, registry):
        """Test that failure reason mentions visibility issue."""
        adapter = MockBrowserAdapter(
            visible=False,
            enabled=True,
            states={"enabled", "attached", "hidden"},
        )

        ref = create_element_ref("e1")
        result = validator.run_all_checks("click", ref, registry, adapter)

        failure_reason = result.failure_reason
        assert failure_reason is not None
        assert "visible" in failure_reason.lower()


class TestPreValidationDetectsDisabled:
    """Tests that pre-validation detects disabled elements."""

    @pytest.fixture
    def validator(self) -> PreValidator:
        """Create a PreValidator for testing."""
        return PreValidator()

    @pytest.fixture
    def registry(self) -> MockElementRegistry:
        """Create a mock element registry."""
        return MockElementRegistry()

    def test_detects_disabled_element(self, validator, registry):
        """Test that pre-validation detects disabled elements."""
        adapter = MockBrowserAdapter(
            visible=True,
            enabled=False,
            states={"visible", "attached", "disabled"},
        )

        ref = create_element_ref("e1")
        result = validator.run_all_checks("click", ref, registry, adapter)

        assert result.passed is False
        assert "enabled" in result.failed_checks
        assert "enabled" in result.missing_states

    def test_passes_for_enabled_element(self, validator, registry):
        """Test that pre-validation passes for enabled elements."""
        adapter = MockBrowserAdapter(
            visible=True,
            enabled=True,
            stable=True,
            states={"visible", "enabled", "attached", "stable"},
        )

        ref = create_element_ref("e1")
        result = validator.run_all_checks("click", ref, registry, adapter)

        assert result.passed is True
        assert "enabled" not in result.failed_checks

    def test_failure_reason_mentions_disabled(self, validator, registry):
        """Test that failure reason mentions disabled state."""
        adapter = MockBrowserAdapter(
            visible=True,
            enabled=False,
            states={"visible", "attached", "disabled"},
        )

        ref = create_element_ref("e1")
        result = validator.run_all_checks("click", ref, registry, adapter)

        failure_reason = result.failure_reason
        assert failure_reason is not None
        assert "enabled" in failure_reason.lower()

    def test_hover_does_not_require_enabled(self, validator, registry):
        """Test that hover action does not require enabled state."""
        adapter = MockBrowserAdapter(
            visible=True,
            enabled=False,
            stable=True,
            states={"visible", "attached", "disabled", "stable"},
        )

        ref = create_element_ref("e1")
        result = validator.run_all_checks("hover", ref, registry, adapter)

        # Hover does not require enabled, so should pass if visible and stable
        assert result.passed is True


class TestPreValidationDetectsEditable:
    """Tests that pre-validation detects non-editable elements for fill actions."""

    @pytest.fixture
    def validator(self) -> PreValidator:
        """Create a PreValidator for testing."""
        return PreValidator()

    @pytest.fixture
    def registry(self) -> MockElementRegistry:
        """Create a mock element registry."""
        return MockElementRegistry()

    def test_detects_non_editable_for_fill(self, validator, registry):
        """Test that fill action detects non-editable elements."""
        adapter = MockBrowserAdapter(
            visible=True,
            enabled=True,
            editable=False,
            states={"visible", "enabled", "attached", "readonly"},
        )

        ref = create_element_ref("e1")
        result = validator.run_all_checks("fill", ref, registry, adapter)

        assert result.passed is False
        assert "editable" in result.failed_checks

    def test_passes_for_editable_element(self, validator, registry):
        """Test that fill action passes for editable elements."""
        adapter = MockBrowserAdapter(
            visible=True,
            enabled=True,
            editable=True,
            states={"visible", "enabled", "attached", "editable"},
        )

        ref = create_element_ref("e1")
        result = validator.run_all_checks("fill", ref, registry, adapter)

        assert result.passed is True
        assert "editable" not in result.failed_checks


# =============================================================================
# Pre-Validation Passes for Valid Elements
# =============================================================================


class TestPreValidationPassesForValid:
    """Tests that pre-validation passes for valid elements."""

    @pytest.fixture
    def validator(self) -> PreValidator:
        """Create a PreValidator for testing."""
        return PreValidator()

    @pytest.fixture
    def registry(self) -> MockElementRegistry:
        """Create a mock element registry."""
        return MockElementRegistry()

    @pytest.fixture
    def valid_adapter(self) -> MockBrowserAdapter:
        """Create a mock adapter with all valid states."""
        return MockBrowserAdapter(
            visible=True,
            enabled=True,
            editable=True,
            stable=True,
            receives_pointer=True,
            states={"visible", "enabled", "attached", "editable", "stable"},
        )

    def test_click_passes_for_valid_element(self, validator, registry, valid_adapter):
        """Test that click validation passes for a valid element."""
        ref = create_element_ref("e1")
        result = validator.run_all_checks("click", ref, registry, valid_adapter)

        assert result.passed is True
        assert len(result.failed_checks) == 0
        assert len(result.missing_states) == 0

    def test_fill_passes_for_valid_element(self, validator, registry, valid_adapter):
        """Test that fill validation passes for a valid element."""
        ref = create_element_ref("e1")
        result = validator.run_all_checks("fill", ref, registry, valid_adapter)

        assert result.passed is True
        assert len(result.failed_checks) == 0

    def test_hover_passes_for_valid_element(self, validator, registry, valid_adapter):
        """Test that hover validation passes for a valid element."""
        ref = create_element_ref("e1")
        result = validator.run_all_checks("hover", ref, registry, valid_adapter)

        assert result.passed is True

    def test_select_passes_for_valid_element(self, validator, registry, valid_adapter):
        """Test that select validation passes for a valid element."""
        ref = create_element_ref("e1")
        result = validator.run_all_checks("select", ref, registry, valid_adapter)

        assert result.passed is True

    def test_check_passes_for_valid_element(self, validator, registry, valid_adapter):
        """Test that check validation passes for a valid element."""
        ref = create_element_ref("e1")
        result = validator.run_all_checks("check", ref, registry, valid_adapter)

        assert result.passed is True


# =============================================================================
# Pre-Validation Performance Tests
# =============================================================================


class TestPreValidationPerformance:
    """Tests for pre-validation performance - should be fast (<500ms)."""

    @pytest.fixture
    def validator(self) -> PreValidator:
        """Create a PreValidator for testing."""
        return PreValidator()

    @pytest.fixture
    def registry(self) -> MockElementRegistry:
        """Create a mock element registry."""
        return MockElementRegistry()

    def test_validation_is_fast_for_visible_element(self, validator, registry):
        """Test that validation completes quickly for a visible element."""
        adapter = MockBrowserAdapter(
            visible=True,
            enabled=True,
            stable=True,
            states={"visible", "enabled", "attached", "stable"},
        )

        ref = create_element_ref("e1")

        start = time.perf_counter()
        result = validator.run_all_checks("click", ref, registry, adapter)
        elapsed = time.perf_counter() - start

        # Should complete in less than 500ms (much less with mocks)
        assert elapsed < 0.5, f"Validation took {elapsed*1000:.2f}ms, should be <500ms"
        assert result.passed is True

    def test_validation_is_fast_for_invisible_element(self, validator, registry):
        """Test that validation fails fast for an invisible element (<500ms)."""
        adapter = MockBrowserAdapter(
            visible=False,
            enabled=True,
            states={"enabled", "attached", "hidden"},
        )

        ref = create_element_ref("e1")

        start = time.perf_counter()
        result = validator.run_all_checks("click", ref, registry, adapter)
        elapsed = time.perf_counter() - start

        # Should fail fast - not wait for any timeout
        assert elapsed < 0.5, f"Validation took {elapsed*1000:.2f}ms, should be <500ms"
        assert result.passed is False

    def test_validation_is_fast_for_disabled_element(self, validator, registry):
        """Test that validation fails fast for a disabled element (<500ms)."""
        adapter = MockBrowserAdapter(
            visible=True,
            enabled=False,
            states={"visible", "attached", "disabled"},
        )

        ref = create_element_ref("e1")

        start = time.perf_counter()
        result = validator.run_all_checks("click", ref, registry, adapter)
        elapsed = time.perf_counter() - start

        # Should fail fast - not wait for any timeout
        assert elapsed < 0.5, f"Validation took {elapsed*1000:.2f}ms, should be <500ms"
        assert result.passed is False

    def test_quick_validation_is_very_fast(self, validator, registry):
        """Test that quick validation (registry-only) is very fast (<100ms)."""
        ref = create_element_ref("e1")

        start = time.perf_counter()
        result = validator.validate_quick("click", ref, registry)
        elapsed = time.perf_counter() - start

        # Quick validation should be very fast (no browser calls)
        assert elapsed < 0.1, f"Quick validation took {elapsed*1000:.2f}ms, should be <100ms"
        assert result.passed is True

    def test_multiple_validations_are_fast(self, validator, registry):
        """Test that multiple validations complete quickly."""
        adapter = MockBrowserAdapter(
            visible=True,
            enabled=True,
            stable=True,
            states={"visible", "enabled", "attached", "stable"},
        )

        start = time.perf_counter()

        # Run 10 validations
        for i in range(10):
            ref = create_element_ref(f"e{(i % 3) + 1}")
            validator.run_all_checks("click", ref, registry, adapter)

        elapsed = time.perf_counter() - start

        # 10 validations should complete in less than 1 second
        assert elapsed < 1.0, f"10 validations took {elapsed*1000:.2f}ms, should be <1000ms"


# =============================================================================
# Pre-Validation Registry Checks
# =============================================================================


class TestPreValidationRegistryChecks:
    """Tests for pre-validation registry checks."""

    @pytest.fixture
    def validator(self) -> PreValidator:
        """Create a PreValidator for testing."""
        return PreValidator()

    def test_fails_for_stale_registry(self, validator):
        """Test that validation fails if the registry is stale."""
        stale_registry = MockElementRegistry(stale=True)
        adapter = MockBrowserAdapter()

        ref = create_element_ref("e1")
        result = validator.run_all_checks("click", ref, stale_registry, adapter)

        assert result.passed is False
        assert "registry_fresh" in result.failed_checks

    def test_fails_for_unknown_ref(self, validator):
        """Test that validation fails if the element ref is not in registry."""
        registry = MockElementRegistry(refs={"e1", "e2"})
        adapter = MockBrowserAdapter()

        # Try to validate a ref that doesn't exist
        ref = create_element_ref("e999")
        result = validator.run_all_checks("click", ref, registry, adapter)

        assert result.passed is False
        assert "element_exists" in result.failed_checks

    def test_passes_for_fresh_registry_with_valid_ref(self, validator):
        """Test that validation passes for fresh registry with valid ref."""
        registry = MockElementRegistry(stale=False, refs={"e1", "e2", "e3"})
        adapter = MockBrowserAdapter()

        ref = create_element_ref("e1")
        result = validator.run_all_checks("click", ref, registry, adapter)

        assert "registry_fresh" not in result.failed_checks
        assert "element_exists" not in result.failed_checks


# =============================================================================
# Pre-Validation Skipped for Navigation
# =============================================================================


class TestPreValidationSkippedForNavigation:
    """Tests that pre-validation is skipped for navigation actions."""

    @pytest.fixture
    def validator(self) -> PreValidator:
        """Create a PreValidator for testing."""
        return PreValidator()

    @pytest.fixture
    def registry(self) -> MockElementRegistry:
        """Create a mock element registry."""
        return MockElementRegistry()

    def test_skipped_for_navigate(self, validator, registry):
        """Test that validation is skipped for navigate action."""
        adapter = MockBrowserAdapter()
        ref = create_element_ref("e1")

        result = validator.run_all_checks("navigate", ref, registry, adapter)

        # Should return a skipped result
        assert result.passed is True
        assert "skipped" in result.checks_performed

    def test_skipped_for_go_to(self, validator, registry):
        """Test that validation is skipped for go_to action."""
        adapter = MockBrowserAdapter()
        ref = create_element_ref("e1")

        result = validator.run_all_checks("go_to", ref, registry, adapter)

        assert result.passed is True
        assert "skipped" in result.checks_performed

    def test_skipped_for_reload(self, validator, registry):
        """Test that validation is skipped for reload action."""
        adapter = MockBrowserAdapter()
        ref = create_element_ref("e1")

        result = validator.run_all_checks("reload", ref, registry, adapter)

        assert result.passed is True


# =============================================================================
# PreValidationResult Tests
# =============================================================================


class TestPreValidationResult:
    """Tests for PreValidationResult value object."""

    def test_success_result_creation(self):
        """Test creating a success result."""
        result = PreValidationResult.success(
            checks_performed=["visible", "enabled", "stable"],
            current_states={"visible", "enabled", "stable"},
        )

        assert result.passed is True
        assert len(result.failed_checks) == 0
        assert len(result.missing_states) == 0
        assert result.failure_reason is None

    def test_failure_result_creation(self):
        """Test creating a failure result."""
        result = PreValidationResult.failure(
            checks_performed=["visible", "enabled"],
            failed_checks=["visible"],
            current_states={"enabled", "hidden"},
            missing_states={"visible"},
        )

        assert result.passed is False
        assert "visible" in result.failed_checks
        assert "visible" in result.missing_states
        assert result.failure_reason is not None

    def test_skipped_result_creation(self):
        """Test creating a skipped result."""
        result = PreValidationResult.skipped()

        assert result.passed is True
        assert "skipped" in result.checks_performed

    def test_summary_for_success(self):
        """Test summary string for success result."""
        result = PreValidationResult.success(
            checks_performed=["visible", "enabled", "stable"],
            current_states={"visible", "enabled", "stable"},
        )

        summary = result.summary
        assert "PASS" in summary
        assert "3" in summary  # 3 checks

    def test_summary_for_failure(self):
        """Test summary string for failure result."""
        result = PreValidationResult.failure(
            checks_performed=["visible", "enabled"],
            failed_checks=["visible"],
            current_states={"enabled", "hidden"},
            missing_states={"visible"},
        )

        summary = result.summary
        assert "FAIL" in summary
        assert "1" in summary  # 1 failed
        assert "2" in summary  # 2 total checks

"""Tests for Action Context bounded context.

This module tests the action execution functionality:
- PreValidator service (actionability checks)
- ActionExecution aggregate (coordinated action execution)
- ResponseConfig (verbosity and response filtering)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, Mock

import pytest


# =============================================================================
# Domain Models (to be moved to production code)
# =============================================================================


class ElementState(Enum):
    """Possible states of an element."""

    ATTACHED = "attached"
    DETACHED = "detached"
    VISIBLE = "visible"
    HIDDEN = "hidden"
    ENABLED = "enabled"
    DISABLED = "disabled"
    EDITABLE = "editable"
    READONLY = "readonly"
    CHECKED = "checked"
    UNCHECKED = "unchecked"
    FOCUSED = "focused"
    STABLE = "stable"
    ANIMATING = "animating"


@dataclass
class ValidationResult:
    """Result of a pre-validation check."""

    passed: bool
    check_name: str
    message: str = ""
    element_states: List[ElementState] = field(default_factory=list)

    @classmethod
    def success(cls, check_name: str) -> "ValidationResult":
        """Create a successful validation result."""
        return cls(passed=True, check_name=check_name)

    @classmethod
    def failure(cls, check_name: str, message: str) -> "ValidationResult":
        """Create a failed validation result."""
        return cls(passed=False, check_name=check_name, message=message)


@dataclass
class PreValidationReport:
    """Complete report from all pre-validation checks."""

    results: List[ValidationResult]
    passed: bool = field(init=False)
    failure_reasons: List[str] = field(init=False)

    def __post_init__(self):
        self.passed = all(r.passed for r in self.results)
        self.failure_reasons = [
            f"{r.check_name}: {r.message}" for r in self.results if not r.passed
        ]


class PreValidator:
    """Service for pre-validating element actionability.

    Performs automatic checks before executing actions:
    1. Element attached to DOM
    2. Element visible (non-empty bounding box)
    3. Element stable (not animating)
    4. Element enabled (not disabled)
    5. Element receives pointer events (not obscured)
    """

    # Actions that require visibility check
    VISIBILITY_REQUIRED = {"click", "fill", "type", "hover", "select", "check"}

    # Actions that require enabled check
    ENABLED_REQUIRED = {"click", "fill", "type", "select", "check", "uncheck"}

    # Actions that require stability check
    STABILITY_REQUIRED = {"click", "fill", "type"}

    def __init__(self, state_provider: Optional[Any] = None):
        """Initialize with optional state provider.

        Args:
            state_provider: Object that can provide element states
                           (e.g., browser adapter)
        """
        self._state_provider = state_provider

    def run_all_checks(
        self,
        action: str,
        element_states: Set[ElementState],
    ) -> PreValidationReport:
        """Run all applicable checks for an action.

        Args:
            action: The action type (e.g., 'click', 'fill')
            element_states: Current states of the element

        Returns:
            PreValidationReport with all check results
        """
        results = []

        # Always check attached
        results.append(self.check_element_attached(element_states))

        # Conditional checks based on action type
        action_lower = action.lower()

        if action_lower in self.VISIBILITY_REQUIRED:
            results.append(self.check_element_visible(element_states))

        if action_lower in self.ENABLED_REQUIRED:
            results.append(self.check_element_enabled(element_states))

        if action_lower in self.STABILITY_REQUIRED:
            results.append(self.check_element_stable(element_states))

        return PreValidationReport(results=results)

    def check_element_attached(
        self, states: Set[ElementState]
    ) -> ValidationResult:
        """Check if element is attached to DOM."""
        if ElementState.DETACHED in states:
            return ValidationResult.failure(
                "attached", "Element is not attached to the DOM"
            )
        if ElementState.ATTACHED in states:
            return ValidationResult.success("attached")
        # If no explicit state, assume attached
        return ValidationResult.success("attached")

    def check_element_visible(
        self, states: Set[ElementState]
    ) -> ValidationResult:
        """Check if element is visible."""
        if ElementState.HIDDEN in states:
            return ValidationResult.failure(
                "visible", "Element is not visible (hidden)"
            )
        if ElementState.VISIBLE in states:
            return ValidationResult.success("visible")
        # If no explicit state, assume visible
        return ValidationResult.success("visible")

    def check_element_enabled(
        self, states: Set[ElementState]
    ) -> ValidationResult:
        """Check if element is enabled."""
        if ElementState.DISABLED in states:
            return ValidationResult.failure(
                "enabled", "Element is disabled and cannot be interacted with"
            )
        if ElementState.ENABLED in states:
            return ValidationResult.success("enabled")
        return ValidationResult.success("enabled")

    def check_element_stable(
        self, states: Set[ElementState]
    ) -> ValidationResult:
        """Check if element is stable (not animating)."""
        if ElementState.ANIMATING in states:
            return ValidationResult.failure(
                "stable", "Element is animating and not stable"
            )
        if ElementState.STABLE in states:
            return ValidationResult.success("stable")
        return ValidationResult.success("stable")


class SnapshotMode(Enum):
    """Modes for including snapshots in responses."""

    FULL = "full"  # Complete accessibility tree
    INCREMENTAL = "incremental"  # Only changes since last snapshot
    NONE = "none"  # No snapshot in response


@dataclass
class ResponseConfig:
    """Configuration for action responses.

    Controls what information is included in action responses
    to optimize token usage.
    """

    include_snapshot: bool = True
    snapshot_mode: SnapshotMode = SnapshotMode.INCREMENTAL
    include_console: bool = False
    include_network: bool = False
    include_timing: bool = True
    verbosity: str = "standard"  # compact, standard, verbose

    @classmethod
    def minimal(cls) -> "ResponseConfig":
        """Create minimal response configuration."""
        return cls(
            include_snapshot=False,
            include_console=False,
            include_network=False,
            include_timing=False,
            verbosity="compact",
        )

    @classmethod
    def full(cls) -> "ResponseConfig":
        """Create full response configuration."""
        return cls(
            include_snapshot=True,
            snapshot_mode=SnapshotMode.FULL,
            include_console=True,
            include_network=True,
            include_timing=True,
            verbosity="verbose",
        )


@dataclass
class ActionResponse:
    """Response from an action execution."""

    success: bool
    action: str
    element_ref: str
    element_description: str = ""
    error: Optional[str] = None
    snapshot: Optional[str] = None
    execution_time_ms: Optional[int] = None
    console_logs: Optional[List[str]] = None
    network_requests: Optional[List[Dict]] = None

    def to_dict(self, config: ResponseConfig) -> Dict[str, Any]:
        """Convert to dictionary respecting response config."""
        result: Dict[str, Any] = {
            "success": self.success,
            "action": self.action,
            "ref": self.element_ref,
        }

        if self.element_description:
            result["element"] = self.element_description

        if not self.success and self.error:
            result["error"] = self.error

        if config.include_snapshot and self.snapshot:
            result["snapshot"] = self.snapshot
            result["snapshot_mode"] = config.snapshot_mode.value

        if config.include_timing and self.execution_time_ms is not None:
            result["time_ms"] = self.execution_time_ms

        if config.include_console and self.console_logs:
            result["console"] = self.console_logs

        if config.include_network and self.network_requests:
            result["network"] = self.network_requests

        return result


class ActionExecution:
    """Aggregate for coordinated action execution.

    Combines pre-validation, action execution, and response building.
    """

    def __init__(
        self,
        pre_validator: PreValidator,
        response_config: Optional[ResponseConfig] = None,
    ):
        self._pre_validator = pre_validator
        self._response_config = response_config or ResponseConfig()

    @property
    def response_config(self) -> ResponseConfig:
        return self._response_config

    def execute_with_pre_validation(
        self,
        action: str,
        ref: str,
        element_states: Set[ElementState],
        execute_fn,
        element_description: str = "",
    ) -> ActionResponse:
        """Execute an action with pre-validation.

        Args:
            action: The action type
            ref: Element reference
            element_states: Current element states
            execute_fn: Function to execute the action
            element_description: Human-readable element description

        Returns:
            ActionResponse with result
        """
        import time

        start_time = time.time()

        # Run pre-validation
        validation = self._pre_validator.run_all_checks(action, element_states)

        if not validation.passed:
            return ActionResponse(
                success=False,
                action=action,
                element_ref=ref,
                element_description=element_description,
                error="; ".join(validation.failure_reasons),
            )

        # Execute the action
        try:
            result = execute_fn()
            execution_time = int((time.time() - start_time) * 1000)

            return ActionResponse(
                success=True,
                action=action,
                element_ref=ref,
                element_description=element_description,
                snapshot=result.get("snapshot") if isinstance(result, dict) else None,
                execution_time_ms=execution_time,
            )
        except Exception as e:
            return ActionResponse(
                success=False,
                action=action,
                element_ref=ref,
                element_description=element_description,
                error=str(e),
            )

    def build_response(
        self,
        action_response: ActionResponse,
        config: Optional[ResponseConfig] = None,
    ) -> Dict[str, Any]:
        """Build response dictionary respecting verbosity.

        Args:
            action_response: The action response to convert
            config: Optional override for response config

        Returns:
            Response dictionary
        """
        effective_config = config or self._response_config
        return action_response.to_dict(effective_config)


# =============================================================================
# Tests
# =============================================================================


class TestPreValidator:
    """Tests for PreValidator service."""

    @pytest.fixture
    def validator(self) -> PreValidator:
        """Create a PreValidator for testing."""
        return PreValidator()

    def test_run_all_checks_for_click(self, validator):
        """Test that click action runs appropriate checks."""
        states = {ElementState.ATTACHED, ElementState.VISIBLE, ElementState.ENABLED}
        report = validator.run_all_checks("click", states)

        # Should run attached, visible, enabled, and stable checks
        check_names = [r.check_name for r in report.results]
        assert "attached" in check_names
        assert "visible" in check_names
        assert "enabled" in check_names
        assert "stable" in check_names
        assert report.passed

    def test_run_all_checks_for_fill(self, validator):
        """Test that fill action runs appropriate checks."""
        states = {ElementState.ATTACHED, ElementState.VISIBLE, ElementState.ENABLED}
        report = validator.run_all_checks("fill", states)

        assert report.passed
        check_names = [r.check_name for r in report.results]
        assert "attached" in check_names
        assert "visible" in check_names
        assert "enabled" in check_names

    def test_run_all_checks_for_hover(self, validator):
        """Test that hover action runs appropriate checks."""
        states = {ElementState.ATTACHED, ElementState.VISIBLE}
        report = validator.run_all_checks("hover", states)

        assert report.passed
        check_names = [r.check_name for r in report.results]
        assert "attached" in check_names
        assert "visible" in check_names

    def test_check_element_visible_when_visible(self, validator):
        """Test visibility check when element is visible."""
        states = {ElementState.VISIBLE}
        result = validator.check_element_visible(states)

        assert result.passed
        assert result.check_name == "visible"

    def test_check_element_visible_when_hidden(self, validator):
        """Test visibility check when element is hidden."""
        states = {ElementState.HIDDEN}
        result = validator.check_element_visible(states)

        assert not result.passed
        assert "not visible" in result.message.lower()

    def test_check_element_enabled_when_enabled(self, validator):
        """Test enabled check when element is enabled."""
        states = {ElementState.ENABLED}
        result = validator.check_element_enabled(states)

        assert result.passed
        assert result.check_name == "enabled"

    def test_check_element_enabled_when_disabled(self, validator):
        """Test enabled check when element is disabled."""
        states = {ElementState.DISABLED}
        result = validator.check_element_enabled(states)

        assert not result.passed
        assert "disabled" in result.message.lower()

    def test_check_element_attached_when_attached(self, validator):
        """Test attached check when element is attached."""
        states = {ElementState.ATTACHED}
        result = validator.check_element_attached(states)

        assert result.passed

    def test_check_element_attached_when_detached(self, validator):
        """Test attached check when element is detached."""
        states = {ElementState.DETACHED}
        result = validator.check_element_attached(states)

        assert not result.passed
        assert "not attached" in result.message.lower()

    def test_check_element_stable_when_stable(self, validator):
        """Test stability check when element is stable."""
        states = {ElementState.STABLE}
        result = validator.check_element_stable(states)

        assert result.passed

    def test_check_element_stable_when_animating(self, validator):
        """Test stability check when element is animating."""
        states = {ElementState.ANIMATING}
        result = validator.check_element_stable(states)

        assert not result.passed
        assert "animating" in result.message.lower()

    def test_failed_check_returns_failure_reason(self, validator):
        """Test that failed checks include a failure reason."""
        states = {ElementState.DISABLED, ElementState.HIDDEN}
        report = validator.run_all_checks("click", states)

        assert not report.passed
        assert len(report.failure_reasons) >= 2
        assert any("disabled" in reason.lower() for reason in report.failure_reasons)
        assert any("visible" in reason.lower() for reason in report.failure_reasons)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_success_factory(self):
        """Test creating a success result."""
        result = ValidationResult.success("attached")
        assert result.passed
        assert result.check_name == "attached"
        assert result.message == ""

    def test_failure_factory(self):
        """Test creating a failure result."""
        result = ValidationResult.failure("visible", "Element is hidden")
        assert not result.passed
        assert result.check_name == "visible"
        assert result.message == "Element is hidden"


class TestPreValidationReport:
    """Tests for PreValidationReport dataclass."""

    def test_all_passed(self):
        """Test report when all checks pass."""
        results = [
            ValidationResult.success("attached"),
            ValidationResult.success("visible"),
            ValidationResult.success("enabled"),
        ]
        report = PreValidationReport(results=results)

        assert report.passed
        assert len(report.failure_reasons) == 0

    def test_some_failed(self):
        """Test report when some checks fail."""
        results = [
            ValidationResult.success("attached"),
            ValidationResult.failure("visible", "Element is hidden"),
            ValidationResult.success("enabled"),
        ]
        report = PreValidationReport(results=results)

        assert not report.passed
        assert len(report.failure_reasons) == 1
        assert "visible" in report.failure_reasons[0]


class TestActionExecution:
    """Tests for ActionExecution aggregate."""

    @pytest.fixture
    def executor(self) -> ActionExecution:
        """Create an ActionExecution for testing."""
        validator = PreValidator()
        return ActionExecution(validator)

    def test_execute_with_pre_validation_success(self, executor):
        """Test successful action execution with pre-validation."""
        states = {ElementState.ATTACHED, ElementState.VISIBLE, ElementState.ENABLED}
        execute_fn = Mock(return_value={"snapshot": "test snapshot"})

        response = executor.execute_with_pre_validation(
            action="click",
            ref="e42",
            element_states=states,
            execute_fn=execute_fn,
            element_description="Submit button",
        )

        assert response.success
        assert response.action == "click"
        assert response.element_ref == "e42"
        execute_fn.assert_called_once()

    def test_execute_with_pre_validation_failure(self, executor):
        """Test action execution when pre-validation fails."""
        states = {ElementState.DISABLED}  # Missing visible and attached
        execute_fn = Mock()

        response = executor.execute_with_pre_validation(
            action="click",
            ref="e42",
            element_states=states,
            execute_fn=execute_fn,
        )

        assert not response.success
        assert "disabled" in response.error.lower()
        execute_fn.assert_not_called()  # Should not execute if validation fails

    def test_execute_with_execution_error(self, executor):
        """Test action execution when the action itself fails."""
        states = {ElementState.ATTACHED, ElementState.VISIBLE, ElementState.ENABLED}
        execute_fn = Mock(side_effect=Exception("Click failed"))

        response = executor.execute_with_pre_validation(
            action="click",
            ref="e42",
            element_states=states,
            execute_fn=execute_fn,
        )

        assert not response.success
        assert "Click failed" in response.error

    def test_build_response_includes_snapshot(self, executor):
        """Test that build_response includes snapshot when configured."""
        action_response = ActionResponse(
            success=True,
            action="click",
            element_ref="e42",
            snapshot="- button [ref=e42]",
        )

        config = ResponseConfig(include_snapshot=True)
        result = executor.build_response(action_response, config)

        assert "snapshot" in result
        assert result["snapshot"] == "- button [ref=e42]"

    def test_build_response_excludes_snapshot(self, executor):
        """Test that build_response excludes snapshot when not configured."""
        action_response = ActionResponse(
            success=True,
            action="click",
            element_ref="e42",
            snapshot="- button [ref=e42]",
        )

        config = ResponseConfig(include_snapshot=False)
        result = executor.build_response(action_response, config)

        assert "snapshot" not in result

    def test_build_response_respects_verbosity_compact(self, executor):
        """Test response building with compact verbosity."""
        action_response = ActionResponse(
            success=True,
            action="click",
            element_ref="e42",
            element_description="Submit button",
            execution_time_ms=150,
        )

        config = ResponseConfig.minimal()
        result = executor.build_response(action_response, config)

        # Minimal config should exclude timing
        assert "time_ms" not in result

    def test_build_response_respects_verbosity_verbose(self, executor):
        """Test response building with verbose verbosity."""
        action_response = ActionResponse(
            success=True,
            action="click",
            element_ref="e42",
            element_description="Submit button",
            execution_time_ms=150,
            console_logs=["Log 1", "Log 2"],
        )

        config = ResponseConfig.full()
        result = executor.build_response(action_response, config)

        assert "time_ms" in result
        assert "console" in result


class TestResponseConfig:
    """Tests for ResponseConfig dataclass."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = ResponseConfig()

        assert config.include_snapshot is True
        assert config.snapshot_mode == SnapshotMode.INCREMENTAL
        assert config.include_console is False
        assert config.include_network is False
        assert config.include_timing is True
        assert config.verbosity == "standard"

    def test_minimal_factory(self):
        """Test minimal configuration factory."""
        config = ResponseConfig.minimal()

        assert config.include_snapshot is False
        assert config.include_console is False
        assert config.include_network is False
        assert config.include_timing is False
        assert config.verbosity == "compact"

    def test_full_factory(self):
        """Test full configuration factory."""
        config = ResponseConfig.full()

        assert config.include_snapshot is True
        assert config.snapshot_mode == SnapshotMode.FULL
        assert config.include_console is True
        assert config.include_network is True
        assert config.include_timing is True
        assert config.verbosity == "verbose"

    def test_incremental_snapshot_mode(self):
        """Test incremental snapshot mode."""
        config = ResponseConfig(snapshot_mode=SnapshotMode.INCREMENTAL)
        assert config.snapshot_mode == SnapshotMode.INCREMENTAL

    def test_full_snapshot_mode(self):
        """Test full snapshot mode."""
        config = ResponseConfig(snapshot_mode=SnapshotMode.FULL)
        assert config.snapshot_mode == SnapshotMode.FULL

    def test_no_snapshot_mode(self):
        """Test no snapshot mode."""
        config = ResponseConfig(snapshot_mode=SnapshotMode.NONE)
        assert config.snapshot_mode == SnapshotMode.NONE


class TestActionResponse:
    """Tests for ActionResponse dataclass."""

    def test_to_dict_success(self):
        """Test converting success response to dict."""
        response = ActionResponse(
            success=True,
            action="click",
            element_ref="e42",
            element_description="Submit button",
        )

        config = ResponseConfig()
        result = response.to_dict(config)

        assert result["success"] is True
        assert result["action"] == "click"
        assert result["ref"] == "e42"
        assert result["element"] == "Submit button"

    def test_to_dict_failure(self):
        """Test converting failure response to dict."""
        response = ActionResponse(
            success=False,
            action="click",
            element_ref="e42",
            error="Element not found",
        )

        config = ResponseConfig()
        result = response.to_dict(config)

        assert result["success"] is False
        assert result["error"] == "Element not found"

    def test_to_dict_with_timing(self):
        """Test response with timing information."""
        response = ActionResponse(
            success=True,
            action="click",
            element_ref="e42",
            execution_time_ms=150,
        )

        config = ResponseConfig(include_timing=True)
        result = response.to_dict(config)

        assert result["time_ms"] == 150

    def test_to_dict_without_timing(self):
        """Test response without timing information."""
        response = ActionResponse(
            success=True,
            action="click",
            element_ref="e42",
            execution_time_ms=150,
        )

        config = ResponseConfig(include_timing=False)
        result = response.to_dict(config)

        assert "time_ms" not in result

"""Action Context Value Objects.

These immutable value objects define the core data structures for the Action Context.
They are designed to be token-efficient and support the pre-validation and response
filtering pipeline as specified in ADR-001.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set


@dataclass(frozen=True)
class ExecutionId:
    """Unique identifier for an action execution.

    Provides traceability for action executions through the system,
    enabling correlation of events and debugging.

    Format: "exec_{8-char-hex}"
    """
    value: str

    def __post_init__(self) -> None:
        """Validate execution ID format."""
        if not self.value:
            raise ValueError("ExecutionId value cannot be empty")
        if not self.value.startswith("exec_"):
            raise ValueError(
                f"Invalid ExecutionId format: '{self.value}'. "
                "Must start with 'exec_'"
            )

    @classmethod
    def generate(cls) -> "ExecutionId":
        """Generate a new unique ExecutionId.

        Returns:
            A new ExecutionId with a unique hex suffix
        """
        return cls(value=f"exec_{uuid.uuid4().hex[:8]}")

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)


@dataclass(frozen=True)
class ActionParameters:
    """Parameters for an action.

    Encapsulates the input parameters for browser actions, including
    the primary value (e.g., text to type), additional options,
    and optional timeout override.

    Attributes:
        value: Primary action value (e.g., text for fill, URL for navigate)
        options: Additional action-specific options (e.g., force, noWaitAfter)
        timeout_override: Optional custom timeout in milliseconds
    """
    value: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)
    timeout_override: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.timeout_override is not None and self.timeout_override < 0:
            raise ValueError(
                f"timeout_override must be non-negative, got {self.timeout_override}"
            )

    def with_option(self, key: str, value: Any) -> "ActionParameters":
        """Create a new ActionParameters with an additional option.

        Args:
            key: Option name
            value: Option value

        Returns:
            New ActionParameters with the option added
        """
        new_options = {**self.options, key: value}
        return ActionParameters(
            value=self.value,
            options=new_options,
            timeout_override=self.timeout_override,
        )

    def with_timeout(self, timeout_ms: int) -> "ActionParameters":
        """Create a new ActionParameters with a timeout override.

        Args:
            timeout_ms: Timeout in milliseconds

        Returns:
            New ActionParameters with the timeout set
        """
        return ActionParameters(
            value=self.value,
            options=self.options,
            timeout_override=timeout_ms,
        )

    @property
    def has_timeout_override(self) -> bool:
        """Check if a custom timeout is specified."""
        return self.timeout_override is not None


@dataclass(frozen=True)
class PreValidationResult:
    """Result of pre-validation checks.

    Contains the outcome of all pre-validation checks performed before
    an action execution, enabling early failure detection and clear
    error reporting.

    Attributes:
        passed: Whether all required checks passed
        checks_performed: List of check names that were run
        failed_checks: List of check names that failed
        current_states: Set of states the element currently has
        missing_states: Set of required states the element lacks
    """
    passed: bool
    checks_performed: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)
    current_states: Set[str] = field(default_factory=set)
    missing_states: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Convert mutable defaults to frozen sets for immutability."""
        # Note: frozen=True on the class prevents attribute reassignment,
        # but we need to ensure the sets are properly frozen
        object.__setattr__(self, 'current_states', frozenset(self.current_states))
        object.__setattr__(self, 'missing_states', frozenset(self.missing_states))

    @property
    def failure_reason(self) -> Optional[str]:
        """Get a human-readable failure reason.

        Returns:
            A descriptive message if validation failed, None if passed
        """
        if self.passed:
            return None

        if self.failed_checks:
            checks_str = ", ".join(self.failed_checks)
            return f"Element failed checks: {checks_str}"

        if self.missing_states:
            states_str = ", ".join(sorted(self.missing_states))
            return f"Element missing required states: {states_str}"

        return "Pre-validation failed for unknown reason"

    @property
    def summary(self) -> str:
        """Get a concise summary of the validation result.

        Returns:
            A short summary string suitable for logging
        """
        if self.passed:
            return f"PASS ({len(self.checks_performed)} checks)"
        return f"FAIL ({len(self.failed_checks)}/{len(self.checks_performed)} checks failed)"

    @classmethod
    def success(
        cls,
        checks_performed: List[str],
        current_states: Set[str],
    ) -> "PreValidationResult":
        """Create a successful validation result.

        Args:
            checks_performed: List of checks that were run
            current_states: Set of states the element has

        Returns:
            A PreValidationResult indicating success
        """
        return cls(
            passed=True,
            checks_performed=checks_performed,
            failed_checks=[],
            current_states=current_states,
            missing_states=set(),
        )

    @classmethod
    def failure(
        cls,
        checks_performed: List[str],
        failed_checks: List[str],
        current_states: Set[str],
        missing_states: Set[str],
    ) -> "PreValidationResult":
        """Create a failed validation result.

        Args:
            checks_performed: List of checks that were run
            failed_checks: List of checks that failed
            current_states: Set of states the element has
            missing_states: Set of required states the element lacks

        Returns:
            A PreValidationResult indicating failure
        """
        return cls(
            passed=False,
            checks_performed=checks_performed,
            failed_checks=failed_checks,
            current_states=current_states,
            missing_states=missing_states,
        )

    @classmethod
    def skipped(cls) -> "PreValidationResult":
        """Create a result for skipped validation.

        Used when pre-validation is not applicable (e.g., navigation actions).

        Returns:
            A PreValidationResult indicating validation was skipped
        """
        return cls(
            passed=True,
            checks_performed=["skipped"],
            failed_checks=[],
            current_states=set(),
            missing_states=set(),
        )


@dataclass(frozen=True)
class ResponseConfig:
    """Configuration for response filtering.

    Controls what data is included in action responses to optimize
    token consumption. Allows callers to customize the verbosity
    and content of responses based on their needs.

    Attributes:
        include_snapshot: Whether to include page snapshot in response
        snapshot_mode: Type of snapshot (full, incremental, or none)
        include_screenshot: Whether to include a screenshot
        include_console: Whether to include console logs
        include_tabs: Whether to include open tabs information
        verbosity: Response detail level (compact, standard, verbose)
    """
    include_snapshot: bool = True
    snapshot_mode: Literal["full", "incremental", "none"] = "incremental"
    include_screenshot: bool = False
    include_console: bool = False
    include_tabs: bool = False
    verbosity: Literal["compact", "standard", "verbose"] = "standard"

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_snapshot_modes = {"full", "incremental", "none"}
        if self.snapshot_mode not in valid_snapshot_modes:
            raise ValueError(
                f"Invalid snapshot_mode: '{self.snapshot_mode}'. "
                f"Must be one of: {valid_snapshot_modes}"
            )

        valid_verbosity = {"compact", "standard", "verbose"}
        if self.verbosity not in valid_verbosity:
            raise ValueError(
                f"Invalid verbosity: '{self.verbosity}'. "
                f"Must be one of: {valid_verbosity}"
            )

    @classmethod
    def minimal(cls) -> "ResponseConfig":
        """Create a minimal response configuration for token efficiency.

        Returns:
            ResponseConfig with minimal data inclusion
        """
        return cls(
            include_snapshot=False,
            snapshot_mode="none",
            include_screenshot=False,
            include_console=False,
            include_tabs=False,
            verbosity="compact",
        )

    @classmethod
    def standard(cls) -> "ResponseConfig":
        """Create the default standard response configuration.

        Returns:
            ResponseConfig with standard data inclusion
        """
        return cls(
            include_snapshot=True,
            snapshot_mode="incremental",
            include_screenshot=False,
            include_console=False,
            include_tabs=False,
            verbosity="standard",
        )

    @classmethod
    def full(cls) -> "ResponseConfig":
        """Create a full response configuration with all data.

        Returns:
            ResponseConfig with maximum data inclusion
        """
        return cls(
            include_snapshot=True,
            snapshot_mode="full",
            include_screenshot=True,
            include_console=True,
            include_tabs=True,
            verbosity="verbose",
        )

    def with_snapshot(
        self,
        include: bool = True,
        mode: Literal["full", "incremental", "none"] = "incremental",
    ) -> "ResponseConfig":
        """Create a new config with modified snapshot settings.

        Args:
            include: Whether to include snapshot
            mode: Snapshot mode

        Returns:
            New ResponseConfig with modified snapshot settings
        """
        return ResponseConfig(
            include_snapshot=include,
            snapshot_mode=mode,
            include_screenshot=self.include_screenshot,
            include_console=self.include_console,
            include_tabs=self.include_tabs,
            verbosity=self.verbosity,
        )


@dataclass(frozen=True)
class FilteredResponse:
    """Token-optimized response from an action.

    Represents the final response returned to the MCP client, containing
    only the requested data to minimize token consumption.

    Attributes:
        success: Whether the action completed successfully
        action: The action type that was executed
        ref: Element reference if applicable
        result: The action result value
        error: Error message if action failed
        snapshot: YAML snapshot if included
        token_estimate: Estimated token count for this response
    """
    success: bool
    action: str
    ref: Optional[str]
    result: Any
    error: Optional[str]
    snapshot: Optional[str]
    token_estimate: int

    def __post_init__(self) -> None:
        """Validate response state."""
        if self.token_estimate < 0:
            raise ValueError(
                f"token_estimate must be non-negative, got {self.token_estimate}"
            )

    @property
    def is_error(self) -> bool:
        """Check if this response represents an error."""
        return not self.success or self.error is not None

    @property
    def has_snapshot(self) -> bool:
        """Check if this response includes a snapshot."""
        return self.snapshot is not None and len(self.snapshot) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for MCP response.

        Returns:
            Dictionary representation suitable for MCP protocol
        """
        response: Dict[str, Any] = {
            "success": self.success,
            "action": self.action,
        }

        if self.ref is not None:
            response["ref"] = self.ref

        if self.result is not None:
            response["result"] = self.result

        if self.error is not None:
            response["error"] = self.error

        if self.snapshot is not None:
            response["snapshot"] = self.snapshot

        response["token_estimate"] = self.token_estimate

        return response

    @classmethod
    def success_response(
        cls,
        action: str,
        result: Any = None,
        ref: Optional[str] = None,
        snapshot: Optional[str] = None,
        token_estimate: int = 0,
    ) -> "FilteredResponse":
        """Create a successful response.

        Args:
            action: The action type
            result: The action result
            ref: Element reference if applicable
            snapshot: YAML snapshot if included
            token_estimate: Estimated token count

        Returns:
            A FilteredResponse indicating success
        """
        return cls(
            success=True,
            action=action,
            ref=ref,
            result=result,
            error=None,
            snapshot=snapshot,
            token_estimate=token_estimate,
        )

    @classmethod
    def error_response(
        cls,
        action: str,
        error: str,
        ref: Optional[str] = None,
        token_estimate: int = 0,
    ) -> "FilteredResponse":
        """Create an error response.

        Args:
            action: The action type
            error: Error message
            ref: Element reference if applicable
            token_estimate: Estimated token count

        Returns:
            A FilteredResponse indicating failure
        """
        return cls(
            success=False,
            action=action,
            ref=ref,
            result=None,
            error=error,
            snapshot=None,
            token_estimate=token_estimate,
        )

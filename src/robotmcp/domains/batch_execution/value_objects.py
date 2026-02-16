"""Batch Execution Domain Value Objects.

Immutable types that carry no identity. Equality is structural.
All value objects use frozen dataclasses with __post_init__ validation.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, List


class BatchStatus(str, Enum):
    """Overall status of a batch execution.

    Values:
        PASS: All steps succeeded without recovery
        FAIL: At least one step failed and could not be recovered
        RECOVERED: All steps completed but at least one required recovery
        TIMEOUT: The batch exceeded its time budget
    """
    PASS = "PASS"
    FAIL = "FAIL"
    RECOVERED = "RECOVERED"
    TIMEOUT = "TIMEOUT"


class StepStatus(str, Enum):
    """Outcome status of a single batch step.

    Values:
        PASS: Step succeeded on first attempt
        FAIL: Step failed and recovery was not possible or not attempted
        RECOVERED: Step failed initially but succeeded after recovery
        SKIPPED: Step was not executed (e.g., due to earlier failure in stop mode)
    """
    PASS = "PASS"
    FAIL = "FAIL"
    RECOVERED = "RECOVERED"
    SKIPPED = "SKIPPED"


class OnFailurePolicy(str, Enum):
    """Policy that controls what happens when a step fails.

    Values:
        STOP: Abort the batch immediately on first failure
        RETRY: Retry the failed step (same keyword + args, no recovery logic)
        RECOVER: Attempt tiered recovery before giving up
    """
    STOP = "stop"
    RETRY = "retry"
    RECOVER = "recover"


@dataclass(frozen=True)
class BatchId:
    """Unique identifier for a batch execution.

    Format: ``batch_<12 hex chars>`` (e.g., ``batch_a1b2c3d4e5f6``).

    Invariants:
        - value must not be empty

    Examples:
        >>> bid = BatchId.generate()
        >>> bid = BatchId(value="batch_abc123def456")
    """
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("BatchId cannot be empty")

    @classmethod
    def generate(cls) -> BatchId:
        """Generate a new unique BatchId."""
        return cls(value=f"batch_{uuid.uuid4().hex[:12]}")


@dataclass(frozen=True)
class StepReference:
    """A reference to a previous step's return value within argument text.

    Parsed from ``${STEP_N}`` patterns in step arguments. Used by the
    aggregate root to resolve inter-step data dependencies at execution time.

    Attributes:
        index: Zero-based step index being referenced
        raw: The original pattern string (e.g., ``${STEP_0}``)

    Invariants:
        - index must be >= 0

    Examples:
        >>> refs = StepReference.find_all("Login as ${STEP_0} with ${STEP_1}")
        >>> len(refs)
        2
        >>> refs[0].index
        0
    """
    index: int
    raw: str

    PATTERN: ClassVar[re.Pattern] = re.compile(r'\$\{STEP_(\d+)\}')

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"Step index must be >= 0, got {self.index}")

    @classmethod
    def find_all(cls, text: str) -> List[StepReference]:
        """Find all ``${STEP_N}`` references in the given text.

        Args:
            text: The argument string to scan for step references

        Returns:
            List of StepReference instances, in order of appearance
        """
        return [
            cls(index=int(m.group(1)), raw=m.group(0))
            for m in cls.PATTERN.finditer(text)
        ]


@dataclass(frozen=True)
class BatchTimeout:
    """Timeout budget for the entire batch execution.

    Attributes:
        value_ms: Timeout in milliseconds

    Invariants:
        - value_ms must be between MIN_MS (1000) and MAX_MS (600000)

    Examples:
        >>> timeout = BatchTimeout.default()  # 120000ms
        >>> timeout = BatchTimeout(value_ms=30000)
    """
    value_ms: int

    MIN_MS: ClassVar[int] = 1000
    MAX_MS: ClassVar[int] = 600000
    DEFAULT_MS: ClassVar[int] = 120000

    def __post_init__(self) -> None:
        if not (self.MIN_MS <= self.value_ms <= self.MAX_MS):
            raise ValueError(
                f"BatchTimeout must be {self.MIN_MS}-{self.MAX_MS}ms, "
                f"got {self.value_ms}"
            )

    @classmethod
    def default(cls) -> BatchTimeout:
        """Create a BatchTimeout with the default value (120s)."""
        return cls(value_ms=cls.DEFAULT_MS)


@dataclass(frozen=True)
class StepTimeout:
    """Per-step timeout in Robot Framework duration format.

    Accepts standard RF duration strings such as ``10s``, ``1.5m``,
    ``500ms``, ``2 minutes``, etc.

    Attributes:
        rf_format: The duration string in RF format

    Invariants:
        - rf_format must match the RF duration pattern

    Examples:
        >>> timeout = StepTimeout(rf_format="10s")
        >>> timeout = StepTimeout(rf_format="1.5 minutes")
    """
    rf_format: str

    _RF_DURATION: ClassVar[re.Pattern] = re.compile(
        r'^\d+(\.\d+)?\s*'
        r'(s|sec|seconds?|m|min|minutes?|h|hours?|ms|milliseconds?)$',
        re.IGNORECASE,
    )

    def __post_init__(self) -> None:
        if not self._RF_DURATION.match(self.rf_format.strip()):
            raise ValueError(
                f"Invalid RF duration format: {self.rf_format!r}"
            )


@dataclass(frozen=True)
class RecoveryAttemptLimit:
    """Maximum number of recovery attempts allowed per failed step.

    Attributes:
        value: The maximum attempt count

    Invariants:
        - value must be between MIN (1) and MAX (10)

    Examples:
        >>> limit = RecoveryAttemptLimit.default()  # 2
        >>> limit = RecoveryAttemptLimit(value=5)
    """
    value: int

    MIN: ClassVar[int] = 1
    MAX: ClassVar[int] = 10
    DEFAULT: ClassVar[int] = 2

    def __post_init__(self) -> None:
        if not (self.MIN <= self.value <= self.MAX):
            raise ValueError(
                f"RecoveryAttemptLimit must be {self.MIN}-{self.MAX}, "
                f"got {self.value}"
            )

    @classmethod
    def default(cls) -> RecoveryAttemptLimit:
        """Create a RecoveryAttemptLimit with the default value (2)."""
        return cls(value=cls.DEFAULT)

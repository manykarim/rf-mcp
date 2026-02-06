"""Instruction Domain Events.

This module contains domain events for the Instruction bounded context.
Domain events represent something that happened in the domain that
domain experts care about.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class InstructionApplied:
    """Event emitted when instructions are successfully applied.

    Published when instructions are provided to FastMCP or an LLM.
    Used for analytics, learning, and debugging.

    Attributes:
        config_id: The configuration that was applied.
        version_id: The specific version of instructions.
        mode: The instruction mode (default/custom).
        content_source: Where the content came from.
        token_count: Estimated token count of instructions.
        session_id: Optional session identifier.
        timestamp: When the instructions were applied.
    """

    config_id: str
    version_id: str
    mode: str
    content_source: str
    token_count: int
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_type": "InstructionApplied",
            "config_id": self.config_id,
            "version_id": self.version_id,
            "mode": self.mode,
            "content_source": self.content_source,
            "token_count": self.token_count,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class InstructionOverridden:
    """Event emitted when instructions are overridden.

    Published when custom instructions replace default ones,
    or when mode changes from default to custom.

    Attributes:
        config_id: The configuration that was modified.
        previous_mode: The mode before the change.
        new_mode: The mode after the change.
        previous_source: Where previous content came from.
        new_source: Where new content comes from.
        reason: Why the override occurred.
        timestamp: When the override happened.
    """

    config_id: str
    previous_mode: str
    new_mode: str
    previous_source: Optional[str]
    new_source: str
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_type": "InstructionOverridden",
            "config_id": self.config_id,
            "previous_mode": self.previous_mode,
            "new_mode": self.new_mode,
            "previous_source": self.previous_source,
            "new_source": self.new_source,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class InstructionValidationFailed:
    """Event emitted when instruction validation fails.

    Published when instruction content fails validation checks,
    helping identify misconfiguration or malformed instructions.

    Attributes:
        config_id: The configuration that failed validation.
        validation_errors: List of specific validation errors.
        source: Where the invalid content came from.
        attempted_mode: The mode that was being configured.
        timestamp: When the validation failed.
    """

    config_id: str
    validation_errors: List[str]
    source: str
    attempted_mode: str
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def error_count(self) -> int:
        """Get the number of validation errors."""
        return len(self.validation_errors)

    @property
    def first_error(self) -> Optional[str]:
        """Get the first validation error."""
        return self.validation_errors[0] if self.validation_errors else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_type": "InstructionValidationFailed",
            "config_id": self.config_id,
            "validation_errors": self.validation_errors,
            "error_count": self.error_count,
            "source": self.source,
            "attempted_mode": self.attempted_mode,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class InstructionContentLoaded:
    """Event emitted when instruction content is loaded from source.

    Published when content is successfully read from file or template.
    Useful for tracking load times and content sizes.

    Attributes:
        config_id: The configuration being loaded.
        source: Where the content was loaded from.
        content_length: Length of content in characters.
        token_estimate: Estimated token count.
        load_time_ms: Time taken to load in milliseconds.
        timestamp: When the content was loaded.
    """

    config_id: str
    source: str
    content_length: int
    token_estimate: int
    load_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_large_content(self) -> bool:
        """Check if content is considered large (> 5000 chars)."""
        return self.content_length > 5000

    @property
    def is_slow_load(self) -> bool:
        """Check if load was slow (> 100ms)."""
        return self.load_time_ms > 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_type": "InstructionContentLoaded",
            "config_id": self.config_id,
            "source": self.source,
            "content_length": self.content_length,
            "token_estimate": self.token_estimate,
            "load_time_ms": self.load_time_ms,
            "is_large_content": self.is_large_content,
            "is_slow_load": self.is_slow_load,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class InstructionModeChanged:
    """Event emitted when instruction mode is changed.

    Published when the mode switches between off/default/custom.

    Attributes:
        config_id: The configuration that changed.
        previous_mode: The mode before the change.
        new_mode: The mode after the change.
        trigger: What triggered the change (env, api, etc.).
        timestamp: When the change occurred.
    """

    config_id: str
    previous_mode: str
    new_mode: str
    trigger: str = "api"
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_enabling(self) -> bool:
        """Check if this change is enabling instructions."""
        return self.previous_mode == "off" and self.new_mode != "off"

    @property
    def is_disabling(self) -> bool:
        """Check if this change is disabling instructions."""
        return self.previous_mode != "off" and self.new_mode == "off"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_type": "InstructionModeChanged",
            "config_id": self.config_id,
            "previous_mode": self.previous_mode,
            "new_mode": self.new_mode,
            "trigger": self.trigger,
            "is_enabling": self.is_enabling,
            "is_disabling": self.is_disabling,
            "timestamp": self.timestamp.isoformat(),
        }

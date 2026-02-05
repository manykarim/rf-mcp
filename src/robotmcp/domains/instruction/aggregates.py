"""Instruction Domain Aggregates.

This module contains the aggregate roots for the Instruction bounded context.
Aggregates are clusters of domain objects that are treated as a unit
for data changes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .value_objects import InstructionMode, InstructionContent, InstructionPath
from .entities import InstructionVersion

if TYPE_CHECKING:
    pass


@dataclass
class InstructionConfig:
    """Aggregate root for instruction configuration.

    InstructionConfig manages all settings related to LLM instructions,
    including mode selection, content resolution, and version tracking.

    This aggregate ensures consistency of instruction settings and
    provides the entry point for all instruction-related operations.

    Invariants:
    - If mode is CUSTOM, custom_path must be set and valid
    - If mode is OFF, no content is generated
    - Version history is maintained for all changes (max 10)
    - Content token estimate must not exceed max_token_budget

    Attributes:
        config_id: Unique identifier for this configuration.
        mode: Current instruction mode (off/default/custom).
        custom_path: Path to custom instructions file (for CUSTOM mode).
        current_content: Currently resolved instruction content.
        current_version: Version tracking for current content.
        version_history: History of previous versions (max 10).
        include_tool_list: Whether to include available tools in instructions.
        include_session_context: Whether to include session info.
        max_token_budget: Maximum tokens allowed for instructions.

    Examples:
        >>> config = InstructionConfig.create_default()
        >>> config.is_enabled
        True
        >>> config.mode.value
        'default'

        >>> config = InstructionConfig.create_off()
        >>> config.is_enabled
        False
    """

    config_id: str
    mode: InstructionMode
    custom_path: Optional[InstructionPath] = None
    current_content: Optional[InstructionContent] = None
    current_version: Optional[InstructionVersion] = None
    version_history: List[InstructionVersion] = field(default_factory=list)

    # Configuration options
    include_tool_list: bool = True
    include_session_context: bool = False
    max_token_budget: int = 1000

    # History limit
    MAX_VERSION_HISTORY: int = 10

    @classmethod
    def create_default(cls, config_id: Optional[str] = None) -> "InstructionConfig":
        """Create configuration with default mode.

        Args:
            config_id: Optional explicit config ID.

        Returns:
            InstructionConfig with DEFAULT mode.
        """
        return cls(
            config_id=config_id or str(uuid.uuid4()),
            mode=InstructionMode.default(),
        )

    @classmethod
    def create_off(cls, config_id: Optional[str] = None) -> "InstructionConfig":
        """Create configuration with instructions disabled.

        Args:
            config_id: Optional explicit config ID.

        Returns:
            InstructionConfig with OFF mode.
        """
        return cls(
            config_id=config_id or str(uuid.uuid4()),
            mode=InstructionMode.off(),
        )

    @classmethod
    def create_custom(
        cls, path: InstructionPath, config_id: Optional[str] = None
    ) -> "InstructionConfig":
        """Create configuration with custom instructions from file.

        Args:
            path: Path to the custom instructions file.
            config_id: Optional explicit config ID.

        Returns:
            InstructionConfig with CUSTOM mode.
        """
        return cls(
            config_id=config_id or str(uuid.uuid4()),
            mode=InstructionMode.custom(),
            custom_path=path,
        )

    def with_mode(self, mode: InstructionMode) -> "InstructionConfig":
        """Create new config with different mode.

        When switching modes, content is cleared and will need
        to be resolved again.

        Args:
            mode: New instruction mode.

        Returns:
            New InstructionConfig with updated mode.
        """
        return InstructionConfig(
            config_id=self.config_id,
            mode=mode,
            custom_path=self.custom_path if mode.uses_custom_file else None,
            current_content=None,  # Will be resolved fresh
            current_version=None,
            version_history=list(self.version_history),
            include_tool_list=self.include_tool_list,
            include_session_context=self.include_session_context,
            max_token_budget=self.max_token_budget,
        )

    def with_custom_path(self, path: InstructionPath) -> "InstructionConfig":
        """Create new config with custom instruction path.

        Automatically switches mode to CUSTOM.

        Args:
            path: Path to custom instructions file.

        Returns:
            New InstructionConfig with custom path and CUSTOM mode.
        """
        return InstructionConfig(
            config_id=self.config_id,
            mode=InstructionMode.custom(),
            custom_path=path,
            current_content=None,
            current_version=None,
            version_history=list(self.version_history),
            include_tool_list=self.include_tool_list,
            include_session_context=self.include_session_context,
            max_token_budget=self.max_token_budget,
        )

    def with_content(
        self,
        content: InstructionContent,
        session_id: Optional[str] = None,
    ) -> "InstructionConfig":
        """Create new config with resolved content.

        Creates a new version for the content and updates
        version history.

        Args:
            content: The resolved instruction content.
            session_id: Optional session for version tracking.

        Returns:
            New InstructionConfig with content and version.
        """
        version = InstructionVersion.create(content, session_id)

        # Update history
        new_history = list(self.version_history)
        if self.current_version:
            new_history.append(self.current_version)

        # Keep only last N versions
        if len(new_history) > self.MAX_VERSION_HISTORY:
            new_history = new_history[-self.MAX_VERSION_HISTORY :]

        return InstructionConfig(
            config_id=self.config_id,
            mode=self.mode,
            custom_path=self.custom_path,
            current_content=content,
            current_version=version,
            version_history=new_history,
            include_tool_list=self.include_tool_list,
            include_session_context=self.include_session_context,
            max_token_budget=self.max_token_budget,
        )

    def with_options(
        self,
        include_tool_list: Optional[bool] = None,
        include_session_context: Optional[bool] = None,
        max_token_budget: Optional[int] = None,
    ) -> "InstructionConfig":
        """Create new config with updated options.

        Args:
            include_tool_list: Whether to include tool list.
            include_session_context: Whether to include session context.
            max_token_budget: Maximum token budget.

        Returns:
            New InstructionConfig with updated options.
        """
        return InstructionConfig(
            config_id=self.config_id,
            mode=self.mode,
            custom_path=self.custom_path,
            current_content=self.current_content,
            current_version=self.current_version,
            version_history=list(self.version_history),
            include_tool_list=(
                include_tool_list
                if include_tool_list is not None
                else self.include_tool_list
            ),
            include_session_context=(
                include_session_context
                if include_session_context is not None
                else self.include_session_context
            ),
            max_token_budget=(
                max_token_budget
                if max_token_budget is not None
                else self.max_token_budget
            ),
        )

    def validate(self) -> List[str]:
        """Validate configuration consistency.

        Checks all invariants and returns any validation errors.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        # Invariant: Custom mode requires path
        if self.mode.uses_custom_file and not self.custom_path:
            errors.append("Custom mode requires custom_path to be set")

        # Invariant: Custom path must exist
        if self.custom_path and not self.custom_path.exists:
            errors.append(f"Custom instruction file not found: {self.custom_path.value}")

        # Invariant: Content must be within budget
        if self.current_content:
            if self.current_content.token_estimate > self.max_token_budget:
                errors.append(
                    f"Content exceeds token budget: "
                    f"{self.current_content.token_estimate} > {self.max_token_budget}"
                )

        # Invariant: History size
        if len(self.version_history) > self.MAX_VERSION_HISTORY:
            errors.append(
                f"Version history exceeds limit: "
                f"{len(self.version_history)} > {self.MAX_VERSION_HISTORY}"
            )

        return errors

    @property
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0

    @property
    def is_enabled(self) -> bool:
        """Check if instructions are enabled."""
        return self.mode.is_enabled

    @property
    def has_content(self) -> bool:
        """Check if content has been resolved."""
        return self.current_content is not None

    @property
    def version_count(self) -> int:
        """Get total number of versions including current."""
        return len(self.version_history) + (1 if self.current_version else 0)

    def get_previous_version(self) -> Optional[InstructionVersion]:
        """Get the most recent previous version.

        Returns:
            Previous version, or None if no history.
        """
        if self.version_history:
            return self.version_history[-1]
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all configuration data.
        """
        return {
            "config_id": self.config_id,
            "mode": self.mode.value,
            "custom_path": self.custom_path.value if self.custom_path else None,
            "has_content": self.has_content,
            "content_source": (
                self.current_content.source if self.current_content else None
            ),
            "content_tokens": (
                self.current_content.token_estimate if self.current_content else None
            ),
            "current_version": (
                self.current_version.to_dict() if self.current_version else None
            ),
            "version_count": self.version_count,
            "include_tool_list": self.include_tool_list,
            "include_session_context": self.include_session_context,
            "max_token_budget": self.max_token_budget,
        }

    def __str__(self) -> str:
        return f"InstructionConfig(mode={self.mode.value}, enabled={self.is_enabled})"

    def __repr__(self) -> str:
        return (
            f"InstructionConfig("
            f"id={self.config_id[:8]!r}, "
            f"mode={self.mode.value!r}, "
            f"has_content={self.has_content})"
        )

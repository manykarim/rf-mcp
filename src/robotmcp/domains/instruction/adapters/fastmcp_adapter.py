"""FastMCP Instruction Adapter - Anti-Corruption Layer.

This adapter translates domain instruction concepts to the FastMCP
server's expected format, ensuring the domain model is not polluted
by infrastructure concerns.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Callable, Dict, Optional, TYPE_CHECKING

from ..value_objects import InstructionPath, InstructionTemplate
from ..aggregates import InstructionConfig
from ..services import InstructionResolver, InstructionValidator, InstructionRenderer
from ..events import InstructionApplied, InstructionOverridden

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class InstructionTemplateType(Enum):
    """Built-in instruction template types.

    As defined in ADR-002, these templates provide different levels of
    guidance for LLMs of varying capabilities.
    """
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    BROWSER_FOCUSED = "browser-focused"
    API_FOCUSED = "api-focused"

    @classmethod
    def from_string(cls, value: str) -> "InstructionTemplateType":
        """Create from string value (case-insensitive).

        Args:
            value: Template name string.

        Returns:
            InstructionTemplateType enum value.

        Raises:
            ValueError: If value is not a valid template type.
        """
        normalized = value.lower().strip()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(
            f"Invalid template type: '{value}'. "
            f"Valid values: {', '.join(m.value for m in cls)}"
        )


class FastMCPInstructionAdapter:
    """Anti-Corruption Layer adapting instructions to FastMCP format.

    This adapter translates domain instruction concepts to the FastMCP
    server's expected format, ensuring the domain model is not polluted
    by infrastructure concerns.

    Responsibilities:
    - Convert InstructionContent to FastMCP-compatible strings
    - Handle FastMCP-specific formatting requirements
    - Manage instruction injection into server startup
    - Provide fallback behavior when instructions are disabled
    - Load configuration from environment variables

    Attributes:
        _resolver: Service for resolving instruction content.
        _validator: Service for validating instructions.
        _renderer: Service for rendering instructions.
        _event_publisher: Optional callback for domain events.

    Examples:
        >>> adapter = FastMCPInstructionAdapter()
        >>> config = adapter.create_config_from_env()
        >>> instructions = adapter.get_server_instructions(config)
    """

    # Environment variable names (as per ADR-002)
    ENV_INSTRUCTIONS = "ROBOTMCP_INSTRUCTIONS"
    ENV_INSTRUCTIONS_TEMPLATE = "ROBOTMCP_INSTRUCTIONS_TEMPLATE"
    ENV_INSTRUCTIONS_FILE = "ROBOTMCP_INSTRUCTIONS_FILE"

    # Default template type
    DEFAULT_TEMPLATE_TYPE = InstructionTemplateType.STANDARD

    def __init__(
        self,
        resolver: Optional[InstructionResolver] = None,
        validator: Optional[InstructionValidator] = None,
        renderer: Optional[InstructionRenderer] = None,
        event_publisher: Optional[Callable[[object], None]] = None,
        template_name: Optional[str] = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            resolver: Optional custom resolver (creates default if None).
            validator: Optional custom validator (creates default if None).
            renderer: Optional custom renderer (creates default if None).
            event_publisher: Optional callback for domain events.
            template_name: Optional template name for resolver initialization.
        """
        # Get template name from environment if not provided
        if template_name is None:
            template_name = os.environ.get(
                self.ENV_INSTRUCTIONS_TEMPLATE, "standard"
            ).lower().strip()

        self._template_name = template_name
        self._resolver = resolver or InstructionResolver(
            template_name=template_name,
            event_publisher=event_publisher,
        )
        self._validator = validator or InstructionValidator(event_publisher=event_publisher)
        self._renderer = renderer or InstructionRenderer()
        self._event_publisher = event_publisher
        self._logger = logging.getLogger(__name__)

    def get_server_instructions(
        self,
        config: InstructionConfig,
        context: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """Get instructions formatted for FastMCP server.

        This is the primary method used when starting the MCP server.
        It resolves, validates, and renders instructions based on
        the configuration.

        Args:
            config: Instruction configuration.
            context: Optional context for template rendering.
            session_id: Optional session for tracking.

        Returns:
            Formatted instruction string, or None if disabled.

        Note:
            If custom instructions fail validation, falls back to default.
        """
        if not config.is_enabled:
            self._logger.debug("Instructions disabled, returning None")
            return None

        # Resolve content
        try:
            content = self._resolver.resolve(config, context)
        except FileNotFoundError as e:
            self._logger.error(f"Custom instruction file not found: {e}")
            # Fall back to default
            if config.mode.uses_custom_file:
                self._logger.warning("Falling back to default instructions")
                default_config = InstructionConfig.create_default()
                content = self._resolver.resolve(default_config, context)
            else:
                return None
        except Exception as e:
            self._logger.error(f"Failed to resolve instructions: {e}")
            return None

        if not content:
            return None

        # Validate
        validation = self._validator.validate(content, config)
        if not validation.is_valid:
            self._logger.error(f"Instruction validation failed: {validation.errors}")
            # Fall back to default if custom failed
            if config.mode.uses_custom_file:
                self._logger.warning("Falling back to default instructions due to validation failure")
                self._publish_event(
                    InstructionOverridden(
                        config_id=config.config_id,
                        previous_mode="custom",
                        new_mode="default",
                        previous_source=content.source,
                        new_source="template:discovery_first",
                        reason=f"Validation failed: {validation.errors[0]}",
                    )
                )
                default_config = InstructionConfig.create_default()
                content = self._resolver.resolve(default_config, context)
                if not content:
                    return None

        # Log warnings
        for warning in validation.warnings:
            self._logger.warning(f"Instruction warning: {warning}")

        # Render for FastMCP (generic format, no special wrapper needed)
        rendered = self._renderer.render(
            content,
            target=InstructionRenderer.TargetFormat.GENERIC,
            include_wrapper=False,
        )

        # Update config with content for version tracking
        config = config.with_content(content, session_id)

        # Publish applied event
        if config.current_version:
            self._publish_event(
                InstructionApplied(
                    config_id=config.config_id,
                    version_id=config.current_version.version_id,
                    mode=config.mode.value,
                    content_source=content.source,
                    token_count=content.token_estimate,
                    session_id=session_id,
                )
            )

        self._logger.info(
            f"Instructions resolved: mode={config.mode.value}, "
            f"source={content.source}, tokens={content.token_estimate}"
        )

        return rendered

    def create_config_from_env(self) -> InstructionConfig:
        """Create instruction config from environment variables.

        Environment variables (as per ADR-002):
        - ROBOTMCP_INSTRUCTIONS: Mode control ("off", "default", "custom")
        - ROBOTMCP_INSTRUCTIONS_TEMPLATE: Template selection
          ("minimal", "standard", "detailed", "browser-focused", "api-focused")
        - ROBOTMCP_INSTRUCTIONS_FILE: Path to custom instructions file

        Returns:
            InstructionConfig based on environment.

        Examples:
            ROBOTMCP_INSTRUCTIONS=off -> Instructions disabled
            ROBOTMCP_INSTRUCTIONS=default -> Default instructions with standard template
            ROBOTMCP_INSTRUCTIONS_TEMPLATE=detailed -> Use detailed template
            ROBOTMCP_INSTRUCTIONS=custom + ROBOTMCP_INSTRUCTIONS_FILE=./my.txt -> Custom file
        """
        # Read environment variables
        mode_str = os.environ.get(self.ENV_INSTRUCTIONS, "default").lower().strip()
        template_str = os.environ.get(self.ENV_INSTRUCTIONS_TEMPLATE, "standard").lower().strip()
        custom_file = os.environ.get(self.ENV_INSTRUCTIONS_FILE, "").strip()

        self._logger.debug(
            f"Loading config from env: mode={mode_str!r}, "
            f"template={template_str!r}, file={custom_file!r}"
        )

        # Determine template type
        try:
            template_type = InstructionTemplateType.from_string(template_str)
        except ValueError as e:
            self._logger.warning(f"Invalid template type: {e}. Using standard.")
            template_type = self.DEFAULT_TEMPLATE_TYPE

        # Store template type for later use by resolver
        self._current_template_type = template_type

        # Determine mode
        if mode_str == "off":
            self._logger.info("Instructions disabled via environment")
            return InstructionConfig.create_off()

        if mode_str == "custom":
            # Custom mode requires a file path
            if custom_file:
                try:
                    path = InstructionPath(custom_file)
                    self._logger.info(f"Using custom instructions from: {custom_file}")
                    return InstructionConfig.create_custom(path)
                except ValueError as e:
                    self._logger.error(f"Invalid instruction path '{custom_file}': {e}")
                    self._logger.warning("Falling back to default instructions")
            else:
                self._logger.warning(
                    "ROBOTMCP_INSTRUCTIONS=custom but ROBOTMCP_INSTRUCTIONS_FILE not set. "
                    "Falling back to default."
                )

        # Default mode - use the selected template
        self._logger.info(
            f"Using default instructions with '{template_type.value}' template"
        )
        return InstructionConfig.create_default()

    def get_template_type(self) -> InstructionTemplateType:
        """Get the currently configured template type.

        Returns:
            The template type from environment or default.
        """
        if hasattr(self, "_current_template_type"):
            return self._current_template_type

        template_str = os.environ.get(self.ENV_INSTRUCTIONS_TEMPLATE, "standard").lower().strip()
        try:
            return InstructionTemplateType.from_string(template_str)
        except ValueError:
            return self.DEFAULT_TEMPLATE_TYPE

    def get_default_tools_context(self) -> Dict[str, str]:
        """Get default context with available tools.

        Returns:
            Context dictionary with available_tools key.
        """
        return {
            "available_tools": (
                "find_keywords, get_keyword_info, get_session_state, "
                "get_locator_guidance, analyze_scenario, recommend_libraries, "
                "check_library_availability"
            )
        }

    def validate_file_path(self, path: str) -> bool:
        """Validate a potential instruction file path.

        Args:
            path: The path to validate.

        Returns:
            True if path is valid, False otherwise.
        """
        try:
            instruction_path = InstructionPath(path)
            return instruction_path.exists
        except ValueError:
            return False

    def _publish_event(self, event: object) -> None:
        """Publish a domain event.

        Args:
            event: The event to publish.
        """
        if self._event_publisher:
            try:
                self._event_publisher(event)
            except Exception as e:
                self._logger.error(f"Failed to publish event: {e}")

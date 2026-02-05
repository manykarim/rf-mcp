"""Instruction Domain Services.

This module contains domain services for the Instruction bounded context.
Domain services contain business logic that doesn't naturally fit
within an entity or value object.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, ClassVar, Dict, List, Optional, Set, TYPE_CHECKING

from .value_objects import InstructionContent, InstructionPath, InstructionTemplate
from .aggregates import InstructionConfig
from .events import InstructionContentLoaded, InstructionValidationFailed

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class InstructionResolver:
    """Service for resolving which instructions to use.

    Resolves instruction content based on configuration:
    - OFF mode: Returns None
    - DEFAULT mode: Returns built-in default instructions
    - CUSTOM mode: Loads and returns custom instructions from file

    The resolver handles template rendering and context injection.

    Attributes:
        _default_template: Template used for DEFAULT mode.
        _event_publisher: Optional callback for publishing events.
        _cache: Cache for resolved custom file content.

    Examples:
        >>> resolver = InstructionResolver()
        >>> config = InstructionConfig.create_default()
        >>> content = resolver.resolve(config, {"available_tools": "get_snapshot"})
        >>> "get_snapshot" in content.value
        True
    """

    def __init__(
        self,
        default_template: Optional[InstructionTemplate] = None,
        template_name: Optional[str] = None,
        event_publisher: Optional[Callable[[object], None]] = None,
    ) -> None:
        """Initialize the resolver.

        Args:
            default_template: Explicit template for default mode.
            template_name: Template name to look up (used if default_template is None).
            event_publisher: Optional callback for events.
        """
        if default_template is not None:
            self._default_template = default_template
        elif template_name is not None:
            self._default_template = InstructionTemplate.get_by_name(template_name)
        else:
            self._default_template = InstructionTemplate.standard()
        self._event_publisher = event_publisher
        self._cache: Dict[str, InstructionContent] = {}
        self._logger = logging.getLogger(__name__)

    def resolve(
        self,
        config: InstructionConfig,
        context: Optional[Dict[str, str]] = None,
    ) -> Optional[InstructionContent]:
        """Resolve instruction content based on configuration.

        Args:
            config: The instruction configuration.
            context: Optional context for template rendering.

        Returns:
            Resolved InstructionContent, or None if mode is OFF.

        Raises:
            FileNotFoundError: If custom file doesn't exist.
            ValueError: If content is invalid.
        """
        if not config.is_enabled:
            self._logger.debug("Instructions disabled (mode=off)")
            return None

        start_time = datetime.now()
        content: Optional[InstructionContent] = None

        try:
            if config.mode.uses_default_template:
                content = self._resolve_default(context or {})
            elif config.mode.uses_custom_file:
                content = self._resolve_custom(config.custom_path, context or {})

            if content:
                load_time = (datetime.now() - start_time).total_seconds() * 1000
                self._publish_event(
                    InstructionContentLoaded(
                        config_id=config.config_id,
                        source=content.source,
                        content_length=len(content),
                        token_estimate=content.token_estimate,
                        load_time_ms=load_time,
                    )
                )

            return content

        except Exception as e:
            self._logger.error(f"Failed to resolve instructions: {e}")
            raise

    def _resolve_default(self, context: Dict[str, str]) -> InstructionContent:
        """Resolve default template with context.

        Args:
            context: Context values for template placeholders.

        Returns:
            Rendered instruction content.
        """
        # Provide defaults for required placeholders
        if "available_tools" not in context:
            context["available_tools"] = (
                "find_keywords, get_keyword_info, get_session_state, "
                "get_locator_guidance, analyze_scenario, recommend_libraries, "
                "check_library_availability"
            )

        self._logger.debug(
            f"Resolving default template with context keys: {list(context.keys())}"
        )
        return self._default_template.render(context)

    def _resolve_custom(
        self,
        path: Optional[InstructionPath],
        context: Dict[str, str],
    ) -> InstructionContent:
        """Load and resolve custom instructions from file.

        Args:
            path: Path to custom instructions file.
            context: Context for placeholder substitution.

        Returns:
            Instruction content from file.

        Raises:
            ValueError: If path is None.
            FileNotFoundError: If file doesn't exist.
        """
        if not path:
            raise ValueError("Custom mode requires instruction path")

        # Check cache
        cache_key = str(path.value)
        if cache_key in self._cache:
            self._logger.debug(f"Using cached content for {path.value}")
            return self._cache[cache_key]

        # Load from file
        resolved_path = path.resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Instruction file not found: {resolved_path}")

        self._logger.info(f"Loading custom instructions from {resolved_path}")
        content_text = resolved_path.read_text(encoding="utf-8")

        # Apply any context substitutions
        for key, value in context.items():
            content_text = content_text.replace(f"{{{key}}}", value)

        content = InstructionContent(value=content_text, source=f"custom:{path.value}")

        # Cache the result
        self._cache[cache_key] = content

        return content

    def clear_cache(self) -> None:
        """Clear the content cache."""
        self._cache.clear()
        self._logger.debug("Instruction cache cleared")

    def invalidate_cache(self, path: str) -> bool:
        """Invalidate a specific cache entry.

        Args:
            path: The path to invalidate.

        Returns:
            True if entry was removed, False if not found.
        """
        if path in self._cache:
            del self._cache[path]
            return True
        return False

    def _publish_event(self, event: object) -> None:
        """Publish a domain event."""
        if self._event_publisher:
            try:
                self._event_publisher(event)
            except Exception as e:
                self._logger.error(f"Failed to publish event: {e}")


@dataclass
class ValidationResult:
    """Result of instruction validation.

    Attributes:
        is_valid: Whether validation passed.
        errors: List of error messages.
        warnings: List of warning messages.
    """

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Allow boolean evaluation."""
        return self.is_valid

    @property
    def has_warnings(self) -> bool:
        """Check if there are warnings."""
        return len(self.warnings) > 0

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class InstructionValidator:
    """Service for validating instruction content.

    Validates:
    - Content length and format
    - No dangerous patterns (script injection, etc.)
    - Required elements are present
    - Token budget compliance

    Attributes:
        DANGEROUS_PATTERNS: Patterns that are rejected.
        RECOMMENDED_KEYWORDS: Keywords that should appear.

    Examples:
        >>> validator = InstructionValidator()
        >>> content = InstructionContent("Use discovery tools.", "default")
        >>> result = validator.validate(content)
        >>> result.is_valid
        True
    """

    # Dangerous patterns to reject
    DANGEROUS_PATTERNS: ClassVar[List[re.Pattern]] = [
        re.compile(r"<script\b", re.IGNORECASE),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"\beval\s*\(", re.IGNORECASE),
        re.compile(r"\bexec\s*\(", re.IGNORECASE),
        re.compile(r"__import__\s*\(", re.IGNORECASE),
    ]

    # Recommended keywords that should appear in good instructions
    RECOMMENDED_KEYWORDS: ClassVar[Set[str]] = {
        "discovery",
        "snapshot",
        "element",
        "locator",
    }

    def __init__(
        self,
        max_token_budget: int = 1000,
        event_publisher: Optional[Callable[[object], None]] = None,
    ) -> None:
        """Initialize the validator.

        Args:
            max_token_budget: Default maximum token budget.
            event_publisher: Optional callback for events.
        """
        self._max_token_budget = max_token_budget
        self._event_publisher = event_publisher
        self._logger = logging.getLogger(__name__)

    def validate(
        self,
        content: InstructionContent,
        config: Optional[InstructionConfig] = None,
    ) -> ValidationResult:
        """Validate instruction content.

        Args:
            content: The content to validate.
            config: Optional config for additional constraints.

        Returns:
            ValidationResult with errors and warnings.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.search(content.value):
                errors.append(f"Dangerous pattern detected: {pattern.pattern}")

        # Check token budget
        budget = config.max_token_budget if config else self._max_token_budget
        if content.token_estimate > budget:
            errors.append(
                f"Content exceeds token budget: {content.token_estimate} > {budget}"
            )

        # Check for recommended keywords (warnings only)
        content_lower = content.value.lower()
        missing_keywords = [
            kw for kw in self.RECOMMENDED_KEYWORDS if kw not in content_lower
        ]
        if missing_keywords:
            warnings.append(
                f"Missing recommended keywords: {', '.join(missing_keywords)}"
            )

        # Check for very short content
        if content.token_estimate < 20:
            warnings.append("Instructions are very short, may not be effective")

        # Publish event if validation failed
        if errors:
            self._publish_event(
                InstructionValidationFailed(
                    config_id=config.config_id if config else "unknown",
                    validation_errors=errors,
                    source=content.source,
                    attempted_mode=config.mode.value if config else "unknown",
                )
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _publish_event(self, event: object) -> None:
        """Publish a domain event."""
        if self._event_publisher:
            try:
                self._event_publisher(event)
            except Exception as e:
                self._logger.error(f"Failed to publish event: {e}")


class InstructionRenderer:
    """Service for rendering instructions for different LLM types.

    Adapts instruction content for specific LLM requirements:
    - Claude: Optimized for Anthropic's Claude models (XML tags)
    - OpenAI: Optimized for GPT models (markdown)
    - Generic: Neutral format for other models

    Handles format variations like markdown, plain text, XML tags.

    Examples:
        >>> renderer = InstructionRenderer()
        >>> content = InstructionContent("Use discovery.", "default")
        >>> rendered = renderer.render(content, TargetFormat.CLAUDE)
        >>> "<instructions>" in rendered
        True
    """

    class TargetFormat(Enum):
        """Target LLM format for rendering."""

        CLAUDE = "claude"
        OPENAI = "openai"
        GENERIC = "generic"

    def __init__(self) -> None:
        """Initialize the renderer."""
        self._logger = logging.getLogger(__name__)

    def render(
        self,
        content: InstructionContent,
        target: "InstructionRenderer.TargetFormat" = None,
        include_wrapper: bool = True,
    ) -> str:
        """Render instruction content for target LLM.

        Args:
            content: The instruction content.
            target: Target LLM format (defaults to GENERIC).
            include_wrapper: Whether to include format-specific wrapper.

        Returns:
            Rendered instruction string.
        """
        if target is None:
            target = self.TargetFormat.GENERIC

        if target == self.TargetFormat.CLAUDE:
            return self._render_for_claude(content, include_wrapper)
        elif target == self.TargetFormat.OPENAI:
            return self._render_for_openai(content, include_wrapper)
        else:
            return self._render_generic(content, include_wrapper)

    def _render_for_claude(
        self, content: InstructionContent, include_wrapper: bool
    ) -> str:
        """Render for Claude with XML-style tags."""
        if include_wrapper:
            return f"""<instructions>
{content.value}
</instructions>"""
        return content.value

    def _render_for_openai(
        self, content: InstructionContent, include_wrapper: bool
    ) -> str:
        """Render for OpenAI with markdown style."""
        if include_wrapper:
            return f"""# System Instructions

{content.value}

---"""
        return content.value

    def _render_generic(
        self, content: InstructionContent, include_wrapper: bool
    ) -> str:
        """Render with minimal formatting."""
        if include_wrapper:
            return f"""[INSTRUCTIONS]
{content.value}
[/INSTRUCTIONS]"""
        return content.value

    def render_with_context(
        self,
        content: InstructionContent,
        context: Dict[str, str],
        target: "InstructionRenderer.TargetFormat" = None,
    ) -> str:
        """Render with additional context appended.

        Args:
            content: The instruction content.
            context: Additional context to include.
            target: Target LLM format.

        Returns:
            Rendered instruction with context.
        """
        rendered = self.render(content, target, include_wrapper=False)

        # Append context as key-value pairs
        if context:
            context_lines = [f"- {k}: {v}" for k, v in context.items()]
            context_section = "\n\nContext:\n" + "\n".join(context_lines)
            rendered += context_section

        if target == self.TargetFormat.CLAUDE:
            return f"<instructions>\n{rendered}\n</instructions>"
        elif target == self.TargetFormat.OPENAI:
            return f"# System Instructions\n\n{rendered}\n\n---"
        else:
            return f"[INSTRUCTIONS]\n{rendered}\n[/INSTRUCTIONS]"

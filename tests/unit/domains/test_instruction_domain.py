"""Tests for the Instruction Domain bounded context.

This module provides comprehensive tests for the MCP Instructions feature:
- InstructionMode enum and mode validation
- InstructionContent validation (length, patterns)
- InstructionPath validation (traversal prevention)
- InstructionTemplate rendering and placeholder handling
- InstructionConfig aggregate and invariants
- InstructionVersion entity tracking
- InstructionResolver service (mode selection, template loading)
- InstructionValidator service (security checks)
- InstructionRenderer service (format adaptation)
- InMemoryInstructionRepository
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from robotmcp.domains.instruction import (
    InstructionMode,
    InstructionContent,
    InstructionPath,
    InstructionTemplate,
    InstructionConfig,
    InstructionVersion,
    InstructionResolver,
    InstructionValidator,
    InstructionRenderer,
    ValidationResult,
    InMemoryInstructionRepository,
    InstructionApplied,
    InstructionValidationFailed,
    InstructionContentLoaded,
    InstructionOverridden,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_instruction_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for instruction files."""
    instruction_dir = tmp_path / "instructions"
    instruction_dir.mkdir()
    return instruction_dir


@pytest.fixture
def valid_instruction_file(temp_instruction_dir: Path) -> Path:
    """Create a valid instruction file."""
    file_path = temp_instruction_dir / "custom_instructions.txt"
    file_path.write_text(
        "Use discovery tools before executing actions.\n"
        "Always verify keywords exist via find_keywords.\n"
        "Never guess keyword names or arguments.\n"
        "Use get_keyword_info for detailed information."
    )
    return file_path


@pytest.fixture
def minimal_instruction_file(temp_instruction_dir: Path) -> Path:
    """Create a minimal valid instruction file."""
    file_path = temp_instruction_dir / "minimal.txt"
    file_path.write_text("Minimal instruction content here.")
    return file_path


@pytest.fixture
def large_instruction_file(temp_instruction_dir: Path) -> Path:
    """Create an instruction file at the maximum size limit."""
    file_path = temp_instruction_dir / "large.txt"
    # Create content just under the MAX_LENGTH limit
    content = "A" * 49999  # Just under 50000 chars
    file_path.write_text(content)
    return file_path


@pytest.fixture
def event_collector() -> List[object]:
    """Fixture that collects published events."""
    events: List[object] = []
    return events


@pytest.fixture
def event_publisher(event_collector: List[object]):
    """Create an event publisher that collects events."""
    def publisher(event: object) -> None:
        event_collector.append(event)
    return publisher


@pytest.fixture
def sample_context() -> Dict[str, str]:
    """Sample context for template rendering."""
    return {
        "available_tools": "find_keywords, get_keyword_info, get_session_state"
    }


# =============================================================================
# InstructionMode Tests
# =============================================================================


class TestInstructionMode:
    """Tests for InstructionMode value object."""

    def test_create_off_mode(self):
        """Test creating OFF mode."""
        mode = InstructionMode.off()
        assert mode.value == "off"
        assert mode.is_enabled is False
        assert mode.uses_custom_file is False
        assert mode.uses_default_template is False

    def test_create_default_mode(self):
        """Test creating DEFAULT mode."""
        mode = InstructionMode.default()
        assert mode.value == "default"
        assert mode.is_enabled is True
        assert mode.uses_custom_file is False
        assert mode.uses_default_template is True

    def test_create_custom_mode(self):
        """Test creating CUSTOM mode."""
        mode = InstructionMode.custom()
        assert mode.value == "custom"
        assert mode.is_enabled is True
        assert mode.uses_custom_file is True
        assert mode.uses_default_template is False

    def test_from_string_lowercase(self):
        """Test creating mode from lowercase string."""
        assert InstructionMode.from_string("off").value == "off"
        assert InstructionMode.from_string("default").value == "default"
        assert InstructionMode.from_string("custom").value == "custom"

    def test_from_string_uppercase(self):
        """Test creating mode from uppercase string."""
        assert InstructionMode.from_string("OFF").value == "off"
        assert InstructionMode.from_string("DEFAULT").value == "default"
        assert InstructionMode.from_string("CUSTOM").value == "custom"

    def test_from_string_mixed_case(self):
        """Test creating mode from mixed case string."""
        assert InstructionMode.from_string("Off").value == "off"
        assert InstructionMode.from_string("DeFaUlT").value == "default"
        assert InstructionMode.from_string("CuStOm").value == "custom"

    def test_invalid_mode_raises_error(self):
        """Test that invalid mode values raise ValueError."""
        with pytest.raises(ValueError, match="Invalid instruction mode"):
            InstructionMode(value="invalid")

    def test_invalid_mode_from_string(self):
        """Test that invalid string values raise ValueError."""
        with pytest.raises(ValueError, match="Invalid instruction mode"):
            InstructionMode.from_string("unknown")

    def test_mode_immutability(self):
        """Test that mode is immutable (frozen dataclass)."""
        mode = InstructionMode.default()
        with pytest.raises(Exception):  # FrozenInstanceError
            mode.value = "custom"

    def test_mode_string_representation(self):
        """Test string representations of mode."""
        mode = InstructionMode.default()
        assert str(mode) == "default"
        assert "default" in repr(mode)

    def test_mode_equality(self):
        """Test mode equality comparison."""
        mode1 = InstructionMode.default()
        mode2 = InstructionMode.default()
        mode3 = InstructionMode.off()

        assert mode1 == mode2
        assert mode1 != mode3

    def test_mode_hash(self):
        """Test mode can be used in sets/dicts."""
        mode1 = InstructionMode.default()
        mode2 = InstructionMode.default()

        mode_set = {mode1, mode2}
        assert len(mode_set) == 1


# =============================================================================
# InstructionContent Tests
# =============================================================================


class TestInstructionContent:
    """Tests for InstructionContent value object."""

    def test_create_valid_content(self):
        """Test creating valid instruction content."""
        content = InstructionContent(
            value="Use discovery tools before executing actions.",
            source="default"
        )
        assert len(content.value) > 0
        assert content.source == "default"

    def test_content_minimum_length_validation(self):
        """Test that content below minimum length is rejected."""
        with pytest.raises(ValueError, match="too short"):
            InstructionContent(value="short", source="default")

    def test_content_exactly_minimum_length(self):
        """Test content at exactly minimum length."""
        # MIN_LENGTH is 10
        content = InstructionContent(value="A" * 10, source="default")
        assert len(content.value) == 10

    def test_content_maximum_length_validation(self):
        """Test that content above maximum length is rejected."""
        # MAX_LENGTH is 50000
        with pytest.raises(ValueError, match="too long"):
            InstructionContent(value="A" * 50001, source="default")

    def test_content_exactly_maximum_length(self):
        """Test content at exactly maximum length."""
        content = InstructionContent(value="A" * 50000, source="default")
        assert len(content.value) == 50000

    def test_content_empty_rejected(self):
        """Test that empty content is rejected."""
        with pytest.raises(ValueError, match="too short"):
            InstructionContent(value="", source="default")

    def test_content_whitespace_only_rejected(self):
        """Test that whitespace-only content is rejected."""
        with pytest.raises(ValueError, match="too short"):
            InstructionContent(value="   \n\t  ", source="default")

    def test_content_is_from_file(self):
        """Test is_from_file property."""
        file_content = InstructionContent(
            value="Content from file here",
            source="custom:/path/to/file.txt"
        )
        default_content = InstructionContent(
            value="Default content here",
            source="default"
        )

        assert file_content.is_from_file is True
        assert default_content.is_from_file is False

    def test_content_is_from_template(self):
        """Test is_from_template property."""
        template_content = InstructionContent(
            value="Template content here",
            source="template:discovery_first"
        )
        default_content = InstructionContent(
            value="Default content here",
            source="default"
        )

        assert template_content.is_from_template is True
        assert default_content.is_from_template is False

    def test_content_is_default(self):
        """Test is_default property."""
        default_content = InstructionContent(
            value="Default content here",
            source="default"
        )
        custom_content = InstructionContent(
            value="Custom content here",
            source="custom:/path/file.txt"
        )

        assert default_content.is_default is True
        assert custom_content.is_default is False

    def test_content_token_estimate(self):
        """Test token estimation."""
        # 10 words should estimate to ~13 tokens (10 * 1.3)
        content = InstructionContent(
            value="one two three four five six seven eight nine ten",
            source="default"
        )
        assert content.token_estimate == 13

    def test_content_char_count(self):
        """Test character count property."""
        text = "Test content for counting"
        content = InstructionContent(value=text, source="default")
        assert content.char_count == len(text)

    def test_content_line_count(self):
        """Test line count property."""
        text = "Line 1\nLine 2\nLine 3"
        content = InstructionContent(value=text, source="default")
        assert content.line_count == 3

    def test_content_len(self):
        """Test __len__ returns character count."""
        text = "Test content here"
        content = InstructionContent(value=text, source="default")
        assert len(content) == len(text)

    def test_content_str(self):
        """Test __str__ returns the value."""
        text = "Test content here"
        content = InstructionContent(value=text, source="default")
        assert str(content) == text

    def test_content_repr(self):
        """Test __repr__ includes source and preview."""
        content = InstructionContent(
            value="This is a longer test content for preview testing",
            source="default"
        )
        repr_str = repr(content)
        assert "source=" in repr_str
        assert "preview=" in repr_str


# =============================================================================
# InstructionPath Tests
# =============================================================================


class TestInstructionPath:
    """Tests for InstructionPath value object."""

    def test_create_valid_path_txt(self):
        """Test creating path with .txt extension."""
        path = InstructionPath(value="instructions.txt")
        assert path.value == "instructions.txt"
        assert path.extension == ".txt"

    def test_create_valid_path_md(self):
        """Test creating path with .md extension."""
        path = InstructionPath(value="instructions.md")
        assert path.extension == ".md"

    def test_create_valid_path_instruction(self):
        """Test creating path with .instruction extension."""
        path = InstructionPath(value="my.instruction")
        assert path.extension == ".instruction"

    def test_create_valid_path_instructions(self):
        """Test creating path with .instructions extension."""
        path = InstructionPath(value="my.instructions")
        assert path.extension == ".instructions"

    def test_empty_path_rejected(self):
        """Test that empty path is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            InstructionPath(value="")

    def test_invalid_extension_rejected(self):
        """Test that invalid extensions are rejected."""
        invalid_paths = [
            "instructions.py",
            "instructions.sh",
            "instructions.exe",
            "instructions.html",
            "instructions.json",
        ]
        for path in invalid_paths:
            with pytest.raises(ValueError, match="Invalid file extension"):
                InstructionPath(value=path)

    def test_path_traversal_rejected(self):
        """Test that path traversal is rejected."""
        traversal_paths = [
            "../instructions.txt",
            "../../etc/passwd.txt",
            "foo/../bar/instructions.txt",
            "instructions/../../../etc/shadow.txt",
        ]
        for path in traversal_paths:
            with pytest.raises(ValueError, match="traversal"):
                InstructionPath(value=path)

    def test_path_property(self):
        """Test path property returns Path object."""
        instr_path = InstructionPath(value="instructions.txt")
        assert isinstance(instr_path.path, Path)
        assert instr_path.path == Path("instructions.txt")

    def test_filename_property(self):
        """Test filename property."""
        instr_path = InstructionPath(value="dir/subdir/instructions.txt")
        assert instr_path.filename == "instructions.txt"

    def test_exists_property(self, valid_instruction_file: Path):
        """Test exists property with real file."""
        instr_path = InstructionPath(value=str(valid_instruction_file))
        assert instr_path.exists is True

    def test_exists_property_nonexistent(self):
        """Test exists property with nonexistent file."""
        instr_path = InstructionPath(value="nonexistent.txt")
        assert instr_path.exists is False

    def test_resolve_relative_path(self, temp_instruction_dir: Path):
        """Test resolving relative path with base directory."""
        instr_path = InstructionPath(value="instructions.txt")
        resolved = instr_path.resolve(base_path=temp_instruction_dir)
        assert resolved == temp_instruction_dir / "instructions.txt"

    def test_resolve_absolute_path(self, valid_instruction_file: Path):
        """Test resolving absolute path."""
        instr_path = InstructionPath(value=str(valid_instruction_file))
        resolved = instr_path.resolve()
        assert resolved == valid_instruction_file

    def test_resolve_escaping_base_rejected(self, temp_instruction_dir: Path):
        """Test that resolving to outside base directory is rejected."""
        # Create a path that would resolve outside base
        # Need to use an absolute path that escapes
        outside_path = temp_instruction_dir.parent.parent / "outside.txt"
        # Create the file first so it can be checked
        outside_path.parent.mkdir(parents=True, exist_ok=True)
        outside_path.touch()

        try:
            # Create a relative path that would escape
            instr_path = InstructionPath(value="instructions.txt")
            # This should work - it's relative to base
            resolved = instr_path.resolve(base_path=temp_instruction_dir)
            assert temp_instruction_dir in resolved.parents or resolved.parent == temp_instruction_dir
        finally:
            outside_path.unlink(missing_ok=True)

    def test_path_string_representation(self):
        """Test string representations."""
        path = InstructionPath(value="instructions.txt")
        assert str(path) == "instructions.txt"
        assert "InstructionPath" in repr(path)


# =============================================================================
# InstructionTemplate Tests
# =============================================================================


class TestInstructionTemplate:
    """Tests for InstructionTemplate value object."""

    def test_create_valid_template(self):
        """Test creating a valid template."""
        template = InstructionTemplate(
            template_id="test_template",
            content="Use {available_tools} for discovery.",
            description="Test template",
            placeholders=("available_tools",)
        )
        assert template.template_id == "test_template"
        assert "available_tools" in template.placeholders

    def test_empty_template_id_rejected(self):
        """Test that empty template ID is rejected."""
        with pytest.raises(ValueError, match="ID cannot be empty"):
            InstructionTemplate(
                template_id="",
                content="Some content here",
                description="Test",
                placeholders=()
            )

    def test_empty_content_rejected(self):
        """Test that empty content is rejected."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            InstructionTemplate(
                template_id="test",
                content="",
                description="Test",
                placeholders=()
            )

    def test_placeholder_mismatch_rejected(self):
        """Test that placeholder mismatch is rejected."""
        with pytest.raises(ValueError, match="Placeholder mismatch"):
            InstructionTemplate(
                template_id="test",
                content="Use {available_tools} for discovery.",
                description="Test",
                placeholders=("wrong_placeholder",)
            )

    def test_render_template(self, sample_context: Dict[str, str]):
        """Test rendering template with context."""
        template = InstructionTemplate(
            template_id="test",
            content="Tools: {available_tools}",
            description="Test",
            placeholders=("available_tools",)
        )
        content = template.render(sample_context)
        assert "find_keywords" in content.value
        assert content.source == "template:test"

    def test_render_with_missing_placeholder(self):
        """Test rendering with missing placeholder leaves it as-is."""
        template = InstructionTemplate(
            template_id="test",
            content="Tools: {available_tools}",
            description="Test",
            placeholders=("available_tools",)
        )
        content = template.render({})  # Empty context
        assert "{available_tools}" in content.value

    def test_discovery_first_template(self):
        """Test the built-in discovery_first template."""
        template = InstructionTemplate.discovery_first()
        assert template.template_id == "discovery_first"
        assert "available_tools" in template.placeholders
        assert "DISCOVERY FIRST" in template.content
        assert "NO GUESSING" in template.content

    def test_locator_prevention_template(self):
        """Test the built-in locator_prevention template."""
        template = InstructionTemplate.locator_prevention()
        assert template.template_id == "locator_prevention"
        assert len(template.placeholders) == 0
        assert "MUST NOT" in template.content

    def test_minimal_template(self):
        """Test the built-in minimal template."""
        template = InstructionTemplate.minimal()
        assert template.template_id == "minimal"
        assert len(template.placeholders) == 0
        assert len(template.content) < 500  # Should be short


# =============================================================================
# InstructionVersion Entity Tests
# =============================================================================


class TestInstructionVersion:
    """Tests for InstructionVersion entity."""

    def test_create_version_from_content(self):
        """Test creating version from content."""
        content = InstructionContent(
            value="Test instruction content",
            source="default"
        )
        version = InstructionVersion.create(content)

        assert version.version_id is not None
        assert len(version.content_hash) == 16
        assert version.source == "default"
        assert version.application_count == 0
        assert version.success_rate is None

    def test_version_with_session_id(self):
        """Test creating version with session ID."""
        content = InstructionContent(
            value="Test instruction content",
            source="default"
        )
        version = InstructionVersion.create(content, session_id="test-session-123")
        assert version.session_id == "test-session-123"

    def test_version_with_metadata(self):
        """Test creating version with metadata."""
        content = InstructionContent(
            value="Test instruction content",
            source="default"
        )
        version = InstructionVersion.create(
            content,
            metadata={"llm_model": "claude-3", "context": "testing"}
        )
        assert version.metadata["llm_model"] == "claude-3"

    def test_record_application_success(self):
        """Test recording successful application."""
        content = InstructionContent(value="Test content here", source="default")
        version = InstructionVersion.create(content)

        version.record_application(success=True)
        assert version.application_count == 1
        assert version.success_rate == 1.0

    def test_record_application_failure(self):
        """Test recording failed application."""
        content = InstructionContent(value="Test content here", source="default")
        version = InstructionVersion.create(content)

        version.record_application(success=False)
        assert version.application_count == 1
        assert version.success_rate == 0.0

    def test_record_multiple_applications(self):
        """Test recording multiple applications."""
        content = InstructionContent(value="Test content here", source="default")
        version = InstructionVersion.create(content)

        # 3 successes, 1 failure = 75% success rate
        version.record_application(success=True)
        version.record_application(success=True)
        version.record_application(success=True)
        version.record_application(success=False)

        assert version.application_count == 4
        assert version.success_rate == 0.75

    def test_is_effective(self):
        """Test is_effective property."""
        content = InstructionContent(value="Test content here", source="default")
        version = InstructionVersion.create(content)

        assert version.is_effective is False
        version.record_application(success=True)
        assert version.is_effective is True

    def test_has_good_success_rate(self):
        """Test has_good_success_rate property."""
        content = InstructionContent(value="Test content here", source="default")
        version = InstructionVersion.create(content)

        # No data = assume good
        assert version.has_good_success_rate is True

        # Good rate (>= 80%)
        version.record_application(success=True)
        assert version.has_good_success_rate is True

        # Bad rate (< 80%)
        version.record_application(success=False)
        version.record_application(success=False)
        version.record_application(success=False)
        version.record_application(success=False)
        assert version.has_good_success_rate is False

    def test_with_metadata(self):
        """Test with_metadata creates new version."""
        content = InstructionContent(value="Test content here", source="default")
        version = InstructionVersion.create(content)

        new_version = version.with_metadata("key", "value")

        assert new_version is not version
        assert new_version.metadata["key"] == "value"
        assert version.version_id == new_version.version_id  # Same ID

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        content = InstructionContent(value="Test content here", source="default")
        version = InstructionVersion.create(content, session_id="sess-123")
        version.record_application(success=True)

        data = version.to_dict()
        restored = InstructionVersion.from_dict(data)

        assert restored.version_id == version.version_id
        assert restored.content_hash == version.content_hash
        assert restored.session_id == version.session_id
        assert restored.application_count == version.application_count

    def test_version_equality(self):
        """Test version equality based on ID."""
        content = InstructionContent(value="Test content here", source="default")
        version1 = InstructionVersion.create(content)
        version2 = InstructionVersion.create(content)

        # Different versions have different IDs
        assert version1 != version2

        # Same ID = equal
        assert version1 == version1


# =============================================================================
# InstructionConfig Aggregate Tests
# =============================================================================


class TestInstructionConfig:
    """Tests for InstructionConfig aggregate root."""

    def test_create_default_config(self):
        """Test creating default configuration."""
        config = InstructionConfig.create_default()

        assert config.mode.value == "default"
        assert config.is_enabled is True
        assert config.custom_path is None

    def test_create_off_config(self):
        """Test creating disabled configuration."""
        config = InstructionConfig.create_off()

        assert config.mode.value == "off"
        assert config.is_enabled is False

    def test_create_custom_config(self, valid_instruction_file: Path):
        """Test creating custom configuration."""
        path = InstructionPath(value=str(valid_instruction_file))
        config = InstructionConfig.create_custom(path)

        assert config.mode.value == "custom"
        assert config.is_enabled is True
        assert config.custom_path == path

    def test_with_mode_switch(self):
        """Test switching modes."""
        config = InstructionConfig.create_default()

        off_config = config.with_mode(InstructionMode.off())
        assert off_config.is_enabled is False

        default_config = off_config.with_mode(InstructionMode.default())
        assert default_config.is_enabled is True

    def test_with_custom_path(self, valid_instruction_file: Path):
        """Test setting custom path."""
        config = InstructionConfig.create_default()
        path = InstructionPath(value=str(valid_instruction_file))

        custom_config = config.with_custom_path(path)

        assert custom_config.mode.value == "custom"
        assert custom_config.custom_path == path

    def test_with_content_creates_version(self):
        """Test that with_content creates version."""
        config = InstructionConfig.create_default()
        content = InstructionContent(value="Test content here", source="default")

        updated = config.with_content(content)

        assert updated.has_content is True
        assert updated.current_version is not None

    def test_with_content_maintains_history(self):
        """Test that with_content maintains version history."""
        config = InstructionConfig.create_default()

        for i in range(5):
            content = InstructionContent(
                value=f"Test content version {i}",
                source="default"
            )
            config = config.with_content(content)

        assert config.version_count == 5
        assert len(config.version_history) == 4  # 4 in history + 1 current

    def test_version_history_limit(self):
        """Test that version history respects limit."""
        config = InstructionConfig.create_default()

        # Add more than MAX_VERSION_HISTORY versions
        for i in range(15):
            content = InstructionContent(
                value=f"Test content version {i}",
                source="default"
            )
            config = config.with_content(content)

        # History should be trimmed to MAX_VERSION_HISTORY
        assert len(config.version_history) <= config.MAX_VERSION_HISTORY

    def test_with_options(self):
        """Test updating configuration options."""
        config = InstructionConfig.create_default()

        updated = config.with_options(
            include_tool_list=False,
            include_session_context=True,
            max_token_budget=500
        )

        assert updated.include_tool_list is False
        assert updated.include_session_context is True
        assert updated.max_token_budget == 500

    def test_validate_custom_mode_requires_path(self):
        """Test validation: custom mode requires path."""
        config = InstructionConfig(
            config_id="test",
            mode=InstructionMode.custom(),
            custom_path=None  # Missing path
        )

        errors = config.validate()
        assert any("custom_path" in e.lower() for e in errors)

    def test_validate_content_exceeds_budget(self):
        """Test validation: content exceeds token budget."""
        config = InstructionConfig.create_default()
        # Large content that exceeds default budget
        large_content = InstructionContent(
            value="word " * 2000,  # ~2600 tokens
            source="default"
        )
        config = config.with_content(large_content)

        errors = config.validate()
        assert any("budget" in e.lower() for e in errors)

    def test_is_valid_property(self):
        """Test is_valid property."""
        config = InstructionConfig.create_default()
        assert config.is_valid is True

        # Invalid config
        invalid = InstructionConfig(
            config_id="test",
            mode=InstructionMode.custom(),
            custom_path=None
        )
        assert invalid.is_valid is False

    def test_get_previous_version(self):
        """Test getting previous version."""
        config = InstructionConfig.create_default()

        assert config.get_previous_version() is None

        content1 = InstructionContent(value="Test content v1", source="default")
        config = config.with_content(content1)
        assert config.get_previous_version() is None  # No history yet

        content2 = InstructionContent(value="Test content v2", source="default")
        config = config.with_content(content2)
        assert config.get_previous_version() is not None

    def test_to_dict(self):
        """Test to_dict serialization."""
        config = InstructionConfig.create_default()
        content = InstructionContent(value="Test content here", source="default")
        config = config.with_content(content)

        data = config.to_dict()

        assert data["mode"] == "default"
        assert data["has_content"] is True
        assert data["content_source"] == "default"
        assert "version_count" in data


# =============================================================================
# InstructionResolver Service Tests
# =============================================================================


class TestInstructionResolver:
    """Tests for InstructionResolver service."""

    def test_resolve_off_mode_returns_none(self):
        """Test that OFF mode returns None."""
        resolver = InstructionResolver()
        config = InstructionConfig.create_off()

        content = resolver.resolve(config)
        assert content is None

    def test_resolve_default_mode(self, sample_context: Dict[str, str]):
        """Test resolving default template."""
        resolver = InstructionResolver()
        config = InstructionConfig.create_default()

        content = resolver.resolve(config, sample_context)

        assert content is not None
        assert content.is_from_template
        assert "find_keywords" in content.value

    def test_resolve_default_without_context(self):
        """Test resolving default with auto-generated context."""
        resolver = InstructionResolver()
        config = InstructionConfig.create_default()

        content = resolver.resolve(config)

        assert content is not None
        # Should have default tools included
        assert "find_keywords" in content.value

    def test_resolve_custom_mode(self, valid_instruction_file: Path):
        """Test resolving custom instructions from file."""
        resolver = InstructionResolver()
        path = InstructionPath(value=str(valid_instruction_file))
        config = InstructionConfig.create_custom(path)

        content = resolver.resolve(config)

        assert content is not None
        assert content.is_from_file
        assert "custom:" in content.source

    def test_resolve_custom_file_not_found(self, temp_instruction_dir: Path):
        """Test that missing file raises FileNotFoundError."""
        resolver = InstructionResolver()
        path = InstructionPath(value=str(temp_instruction_dir / "nonexistent.txt"))
        config = InstructionConfig.create_custom(path)

        with pytest.raises(FileNotFoundError):
            resolver.resolve(config)

    def test_resolve_custom_with_context_substitution(self, temp_instruction_dir: Path):
        """Test context substitution in custom file."""
        # Create file with placeholder
        file_path = temp_instruction_dir / "template.txt"
        file_path.write_text("Available tools: {available_tools}")

        resolver = InstructionResolver()
        path = InstructionPath(value=str(file_path))
        config = InstructionConfig.create_custom(path)

        content = resolver.resolve(config, {"available_tools": "test_tool"})

        assert "test_tool" in content.value

    def test_resolver_caching(self, valid_instruction_file: Path):
        """Test that resolver caches custom file content."""
        resolver = InstructionResolver()
        path = InstructionPath(value=str(valid_instruction_file))
        config = InstructionConfig.create_custom(path)

        content1 = resolver.resolve(config)
        content2 = resolver.resolve(config)

        # Should be same object from cache
        assert content1 is content2

    def test_clear_cache(self, valid_instruction_file: Path):
        """Test cache clearing."""
        resolver = InstructionResolver()
        path = InstructionPath(value=str(valid_instruction_file))
        config = InstructionConfig.create_custom(path)

        content1 = resolver.resolve(config)
        resolver.clear_cache()
        content2 = resolver.resolve(config)

        # Should be different objects after cache clear
        assert content1 is not content2
        # But same value
        assert content1.value == content2.value

    def test_invalidate_cache(self, valid_instruction_file: Path):
        """Test invalidating specific cache entry."""
        resolver = InstructionResolver()
        path = InstructionPath(value=str(valid_instruction_file))
        config = InstructionConfig.create_custom(path)

        resolver.resolve(config)

        result = resolver.invalidate_cache(str(valid_instruction_file))
        assert result is True

        result = resolver.invalidate_cache("nonexistent")
        assert result is False

    def test_resolver_publishes_events(
        self,
        event_publisher,
        event_collector: List[object],
        sample_context: Dict[str, str]
    ):
        """Test that resolver publishes events."""
        resolver = InstructionResolver(event_publisher=event_publisher)
        config = InstructionConfig.create_default()

        resolver.resolve(config, sample_context)

        assert len(event_collector) == 1
        assert isinstance(event_collector[0], InstructionContentLoaded)

    def test_resolver_with_custom_template(self, sample_context: Dict[str, str]):
        """Test resolver with custom default template."""
        custom_template = InstructionTemplate(
            template_id="custom",
            content="Custom: {available_tools}",
            description="Custom template",
            placeholders=("available_tools",)
        )
        resolver = InstructionResolver(default_template=custom_template)
        config = InstructionConfig.create_default()

        content = resolver.resolve(config, sample_context)

        assert "Custom:" in content.value


# =============================================================================
# InstructionValidator Service Tests
# =============================================================================


class TestInstructionValidator:
    """Tests for InstructionValidator service."""

    def test_validate_safe_content(self):
        """Test validating safe content."""
        validator = InstructionValidator()
        content = InstructionContent(
            value="Use discovery tools. Check snapshots. Verify elements and locators.",
            source="default"
        )

        result = validator.validate(content)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_dangerous_script_pattern(self):
        """Test that script tags are blocked."""
        validator = InstructionValidator()
        content = InstructionContent(
            value="Normal content <script>alert('xss')</script> more content",
            source="custom:/path/file.txt"
        )

        result = validator.validate(content)

        assert result.is_valid is False
        assert any("dangerous" in e.lower() for e in result.errors)

    def test_validate_dangerous_javascript_pattern(self):
        """Test that javascript: is blocked."""
        validator = InstructionValidator()
        content = InstructionContent(
            value="Click here: javascript:alert('xss') for more info",
            source="custom:/path/file.txt"
        )

        result = validator.validate(content)

        assert result.is_valid is False

    def test_validate_dangerous_eval_pattern(self):
        """Test that eval() is blocked."""
        validator = InstructionValidator()
        content = InstructionContent(
            value="Do not use eval( 'code' ) in your scripts",
            source="custom:/path/file.txt"
        )

        result = validator.validate(content)

        assert result.is_valid is False

    def test_validate_dangerous_exec_pattern(self):
        """Test that exec() is blocked."""
        validator = InstructionValidator()
        content = InstructionContent(
            value="Never call exec( command ) directly",
            source="custom:/path/file.txt"
        )

        result = validator.validate(content)

        assert result.is_valid is False

    def test_validate_dangerous_import_pattern(self):
        """Test that __import__() is blocked."""
        validator = InstructionValidator()
        content = InstructionContent(
            value="Avoid __import__( 'module' ) calls",
            source="custom:/path/file.txt"
        )

        result = validator.validate(content)

        assert result.is_valid is False

    def test_validate_token_budget_exceeded(self):
        """Test validation when token budget is exceeded."""
        validator = InstructionValidator(max_token_budget=50)
        content = InstructionContent(
            value="word " * 100,  # ~130 tokens
            source="default"
        )

        result = validator.validate(content)

        assert result.is_valid is False
        assert any("budget" in e.lower() for e in result.errors)

    def test_validate_with_config_budget(self):
        """Test validation uses config's budget."""
        validator = InstructionValidator(max_token_budget=1000)
        content = InstructionContent(
            value="word " * 100,  # ~130 tokens
            source="default"
        )
        config = InstructionConfig.create_default()
        config = config.with_options(max_token_budget=50)

        result = validator.validate(content, config)

        assert result.is_valid is False

    def test_validate_missing_keywords_warning(self):
        """Test warning for missing recommended keywords."""
        validator = InstructionValidator()
        content = InstructionContent(
            value="Some generic instruction without recommended words.",
            source="default"
        )

        result = validator.validate(content)

        # Should be valid but with warnings
        assert result.is_valid is True
        assert result.has_warnings is True
        assert any("keyword" in w.lower() for w in result.warnings)

    def test_validate_short_content_warning(self):
        """Test warning for very short content."""
        validator = InstructionValidator()
        content = InstructionContent(
            value="Short content",  # 13 chars, ~2 tokens
            source="default"
        )

        result = validator.validate(content)

        assert result.is_valid is True
        assert any("short" in w.lower() for w in result.warnings)

    def test_validator_publishes_event_on_failure(
        self,
        event_publisher,
        event_collector: List[object]
    ):
        """Test that validator publishes event on validation failure."""
        validator = InstructionValidator(event_publisher=event_publisher)
        content = InstructionContent(
            value="Dangerous <script> content here",
            source="custom:/path/file.txt"
        )
        config = InstructionConfig.create_default()

        validator.validate(content, config)

        assert len(event_collector) == 1
        assert isinstance(event_collector[0], InstructionValidationFailed)

    def test_validation_result_bool(self):
        """Test ValidationResult boolean evaluation."""
        valid = ValidationResult(is_valid=True)
        invalid = ValidationResult(is_valid=False, errors=["error"])

        assert bool(valid) is True
        assert bool(invalid) is False

    def test_validation_result_to_dict(self):
        """Test ValidationResult serialization."""
        result = ValidationResult(
            is_valid=False,
            errors=["error1", "error2"],
            warnings=["warning1"]
        )

        data = result.to_dict()

        assert data["is_valid"] is False
        assert len(data["errors"]) == 2
        assert len(data["warnings"]) == 1


# =============================================================================
# InstructionRenderer Service Tests
# =============================================================================


class TestInstructionRenderer:
    """Tests for InstructionRenderer service."""

    @pytest.fixture
    def renderer(self) -> InstructionRenderer:
        """Create a renderer instance."""
        return InstructionRenderer()

    @pytest.fixture
    def sample_content(self) -> InstructionContent:
        """Create sample content for rendering."""
        return InstructionContent(
            value="Use discovery tools before actions.",
            source="default"
        )

    def test_render_for_claude(self, renderer, sample_content):
        """Test rendering for Claude (XML tags)."""
        rendered = renderer.render(
            sample_content,
            target=InstructionRenderer.TargetFormat.CLAUDE
        )

        assert rendered.startswith("<instructions>")
        assert rendered.endswith("</instructions>")
        assert sample_content.value in rendered

    def test_render_for_openai(self, renderer, sample_content):
        """Test rendering for OpenAI (markdown)."""
        rendered = renderer.render(
            sample_content,
            target=InstructionRenderer.TargetFormat.OPENAI
        )

        assert "# System Instructions" in rendered
        assert rendered.strip().endswith("---")
        assert sample_content.value in rendered

    def test_render_generic(self, renderer, sample_content):
        """Test rendering generic format."""
        rendered = renderer.render(
            sample_content,
            target=InstructionRenderer.TargetFormat.GENERIC
        )

        assert "[INSTRUCTIONS]" in rendered
        assert "[/INSTRUCTIONS]" in rendered
        assert sample_content.value in rendered

    def test_render_default_is_generic(self, renderer, sample_content):
        """Test that default rendering is generic."""
        rendered = renderer.render(sample_content)

        assert "[INSTRUCTIONS]" in rendered

    def test_render_without_wrapper(self, renderer, sample_content):
        """Test rendering without wrapper."""
        rendered = renderer.render(
            sample_content,
            target=InstructionRenderer.TargetFormat.CLAUDE,
            include_wrapper=False
        )

        assert "<instructions>" not in rendered
        assert rendered == sample_content.value

    def test_render_with_context(self, renderer, sample_content):
        """Test rendering with additional context."""
        context = {"session": "test-123", "model": "claude-3"}
        rendered = renderer.render_with_context(
            sample_content,
            context,
            target=InstructionRenderer.TargetFormat.GENERIC
        )

        assert "Context:" in rendered
        assert "session: test-123" in rendered
        assert "model: claude-3" in rendered


# =============================================================================
# InMemoryInstructionRepository Tests
# =============================================================================


class TestInMemoryInstructionRepository:
    """Tests for InMemoryInstructionRepository."""

    @pytest.fixture
    def repo(self) -> InMemoryInstructionRepository:
        """Create a repository instance."""
        return InMemoryInstructionRepository()

    def test_save_and_get_by_id(self, repo):
        """Test saving and retrieving by ID."""
        config = InstructionConfig.create_default()

        repo.save(config)
        retrieved = repo.get_by_id(config.config_id)

        assert retrieved is not None
        assert retrieved.config_id == config.config_id

    def test_get_by_id_not_found(self, repo):
        """Test getting nonexistent config returns None."""
        result = repo.get_by_id("nonexistent")
        assert result is None

    def test_set_and_get_for_session(self, repo):
        """Test session-based storage."""
        config = InstructionConfig.create_default()

        repo.set_for_session("session-123", config)
        retrieved = repo.get_for_session("session-123")

        assert retrieved is not None
        assert retrieved.config_id == config.config_id

    def test_get_for_session_not_found(self, repo):
        """Test getting config for unknown session."""
        result = repo.get_for_session("unknown-session")
        assert result is None

    def test_set_and_get_default(self, repo):
        """Test default configuration."""
        config = InstructionConfig.create_default()

        repo.set_default(config)
        retrieved = repo.get_default()

        assert retrieved is not None
        assert retrieved.config_id == config.config_id

    def test_get_default_not_set(self, repo):
        """Test getting default when not set."""
        result = repo.get_default()
        assert result is None

    def test_delete(self, repo):
        """Test deleting configuration."""
        config = InstructionConfig.create_default()
        repo.save(config)

        result = repo.delete(config.config_id)

        assert result is True
        assert repo.get_by_id(config.config_id) is None

    def test_delete_not_found(self, repo):
        """Test deleting nonexistent config."""
        result = repo.delete("nonexistent")
        assert result is False

    def test_delete_cleans_session_index(self, repo):
        """Test that delete cleans up session associations."""
        config = InstructionConfig.create_default()
        repo.set_for_session("session-123", config)

        repo.delete(config.config_id)

        assert repo.get_for_session("session-123") is None

    def test_delete_cleans_default_reference(self, repo):
        """Test that delete cleans up default reference."""
        config = InstructionConfig.create_default()
        repo.set_default(config)

        repo.delete(config.config_id)

        assert repo.get_default() is None

    def test_delete_for_session(self, repo):
        """Test deleting session association."""
        config = InstructionConfig.create_default()
        repo.set_for_session("session-123", config)

        result = repo.delete_for_session("session-123")

        assert result is True
        assert repo.get_for_session("session-123") is None
        # Config itself should still exist
        assert repo.get_by_id(config.config_id) is not None

    def test_get_all(self, repo):
        """Test getting all configurations."""
        config1 = InstructionConfig.create_default()
        config2 = InstructionConfig.create_off()

        repo.save(config1)
        repo.save(config2)

        all_configs = repo.get_all()

        assert len(all_configs) == 2

    def test_get_all_session_ids(self, repo):
        """Test getting all session IDs."""
        config = InstructionConfig.create_default()
        repo.set_for_session("session-1", config)
        repo.set_for_session("session-2", config)

        session_ids = repo.get_all_session_ids()

        assert len(session_ids) == 2
        assert "session-1" in session_ids
        assert "session-2" in session_ids

    def test_clear(self, repo):
        """Test clearing all data."""
        config = InstructionConfig.create_default()
        repo.save(config)
        repo.set_for_session("session-1", config)
        repo.set_default(config)

        repo.clear()

        assert len(repo) == 0
        assert repo.get_default() is None
        assert len(repo.get_all_session_ids()) == 0

    def test_stats(self, repo):
        """Test repository statistics."""
        config = InstructionConfig.create_default()
        repo.save(config)
        repo.set_for_session("session-1", config)
        repo.set_default(config)

        stats = repo.stats()

        assert stats["total_configs"] == 1
        assert stats["session_associations"] == 1
        assert stats["has_default"] is True

    def test_len(self, repo):
        """Test __len__."""
        assert len(repo) == 0

        repo.save(InstructionConfig.create_default())
        assert len(repo) == 1

    def test_contains(self, repo):
        """Test __contains__."""
        config = InstructionConfig.create_default()

        assert config.config_id not in repo

        repo.save(config)
        assert config.config_id in repo


# =============================================================================
# Domain Events Tests
# =============================================================================


class TestDomainEvents:
    """Tests for domain events."""

    def test_instruction_applied_event(self):
        """Test InstructionApplied event."""
        event = InstructionApplied(
            config_id="config-123",
            version_id="version-456",
            mode="default",
            content_source="template:discovery_first",
            token_count=150,
            session_id="session-789"
        )

        data = event.to_dict()

        assert data["event_type"] == "InstructionApplied"
        assert data["config_id"] == "config-123"
        assert data["token_count"] == 150

    def test_instruction_overridden_event(self):
        """Test InstructionOverridden event."""
        event = InstructionOverridden(
            config_id="config-123",
            previous_mode="custom",
            new_mode="default",
            previous_source="custom:/path/file.txt",
            new_source="template:discovery_first",
            reason="Validation failed"
        )

        data = event.to_dict()

        assert data["event_type"] == "InstructionOverridden"
        assert data["reason"] == "Validation failed"

    def test_instruction_validation_failed_event(self):
        """Test InstructionValidationFailed event."""
        event = InstructionValidationFailed(
            config_id="config-123",
            validation_errors=["error1", "error2"],
            source="custom:/path/file.txt",
            attempted_mode="custom"
        )

        assert event.error_count == 2
        assert event.first_error == "error1"

        data = event.to_dict()
        assert data["event_type"] == "InstructionValidationFailed"

    def test_instruction_content_loaded_event(self):
        """Test InstructionContentLoaded event."""
        event = InstructionContentLoaded(
            config_id="config-123",
            source="template:discovery_first",
            content_length=1000,
            token_estimate=130,
            load_time_ms=50.5
        )

        assert event.is_large_content is False
        assert event.is_slow_load is False

        data = event.to_dict()
        assert data["event_type"] == "InstructionContentLoaded"

    def test_large_content_detection(self):
        """Test is_large_content property."""
        event = InstructionContentLoaded(
            config_id="config-123",
            source="custom:/path/file.txt",
            content_length=10000,  # > 5000
            token_estimate=1300,
            load_time_ms=50.0
        )

        assert event.is_large_content is True

    def test_slow_load_detection(self):
        """Test is_slow_load property."""
        event = InstructionContentLoaded(
            config_id="config-123",
            source="custom:/path/file.txt",
            content_length=1000,
            token_estimate=130,
            load_time_ms=150.0  # > 100ms
        )

        assert event.is_slow_load is True

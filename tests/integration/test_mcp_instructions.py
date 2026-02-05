"""Integration tests for MCP Instructions with FastMCP server.

Tests verify that:
1. Server initializes with default instructions
2. Server initializes with instructions=off
3. Server initializes with custom template
4. Server initializes with custom file
5. Instruction content appears in MCP initialize response
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
from fastmcp import Client, FastMCP

from robotmcp.domains.instruction import (
    InstructionConfig,
    InstructionMode,
    InstructionTemplate,
    InstructionPath,
    FastMCPInstructionAdapter,
)


class TestServerDefaultInstructions:
    """Test server initializes with default instructions."""

    def test_adapter_creates_default_config(self):
        """FastMCPInstructionAdapter creates default config when no env vars set."""
        adapter = FastMCPInstructionAdapter()
        # Clear all instruction-related env vars
        env_overrides = {
            "ROBOTMCP_INSTRUCTIONS": "",
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "",
            "ROBOTMCP_INSTRUCTIONS_FILE": "",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            config = adapter.create_config_from_env()

            assert config.is_enabled is True
            assert config.mode.value == InstructionMode.DEFAULT

    def test_default_instructions_contain_discovery_content(self):
        """Default instructions include discovery-first guidance."""
        adapter = FastMCPInstructionAdapter()
        config = InstructionConfig.create_default()

        instructions = adapter.get_server_instructions(config)

        assert instructions is not None
        assert len(instructions) > 0
        # Check for key discovery-first phrases
        # Template uses "DISCOVER before EXECUTE" pattern
        assert "DISCOVER" in instructions.upper()
        assert "find_keywords" in instructions or "KEYWORD" in instructions.upper()

    @pytest.mark.asyncio
    async def test_fastmcp_server_with_default_instructions(self):
        """FastMCP server receives instructions in initialize response."""
        adapter = FastMCPInstructionAdapter()
        config = InstructionConfig.create_default()
        instructions = adapter.get_server_instructions(config)

        mcp = FastMCP("Test Server", instructions=instructions)

        async with Client(mcp) as client:
            init_result = client.initialize_result

            assert init_result is not None
            assert init_result.instructions is not None
            assert len(init_result.instructions) > 0
            # Verify content matches what adapter generated
            assert init_result.instructions == instructions


class TestServerInstructionsOff:
    """Test server initializes with instructions=off."""

    def test_adapter_creates_off_config_from_env(self):
        """FastMCPInstructionAdapter creates off config when ROBOTMCP_INSTRUCTIONS=off."""
        adapter = FastMCPInstructionAdapter()

        # Set off mode via ROBOTMCP_INSTRUCTIONS env var
        env_overrides = {
            "ROBOTMCP_INSTRUCTIONS": "off",
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "",
            "ROBOTMCP_INSTRUCTIONS_FILE": "",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            config = adapter.create_config_from_env()

            assert config.is_enabled is False
            assert config.mode.value == InstructionMode.OFF

    def test_adapter_creates_off_config_case_insensitive(self):
        """ROBOTMCP_INSTRUCTIONS=OFF is case insensitive."""
        adapter = FastMCPInstructionAdapter()

        # Test case insensitivity
        for off_value in ["off", "OFF", "Off"]:
            env_overrides = {
                "ROBOTMCP_INSTRUCTIONS": off_value,
                "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "",
                "ROBOTMCP_INSTRUCTIONS_FILE": "",
            }
            with patch.dict(os.environ, env_overrides, clear=False):
                config = adapter.create_config_from_env()
                assert config.is_enabled is False, f"Failed for value: {off_value}"

    def test_off_mode_returns_none_instructions(self):
        """OFF mode returns None for instructions."""
        adapter = FastMCPInstructionAdapter()
        config = InstructionConfig.create_off()

        instructions = adapter.get_server_instructions(config)

        assert instructions is None

    @pytest.mark.asyncio
    async def test_fastmcp_server_with_no_instructions(self):
        """FastMCP server has no instructions when mode is off."""
        # Create server with None instructions (simulating off mode)
        mcp = FastMCP("Test Server", instructions=None)

        async with Client(mcp) as client:
            init_result = client.initialize_result

            assert init_result is not None
            # When None, MCP should return None for instructions
            assert init_result.instructions is None


class TestServerCustomTemplate:
    """Test server initializes with custom template."""

    def test_discovery_first_template_renders(self):
        """Discovery first template renders with context."""
        template = InstructionTemplate.discovery_first()

        content = template.render({"available_tools": "find_keywords, get_keyword_info"})

        assert content is not None
        assert "find_keywords" in content.value
        assert "get_keyword_info" in content.value

    def test_locator_prevention_template_renders(self):
        """Locator prevention template renders."""
        template = InstructionTemplate.locator_prevention()

        content = template.render({})

        assert content is not None
        assert "MUST NOT" in content.value
        assert "MUST" in content.value

    def test_minimal_template_renders(self):
        """Minimal template renders."""
        template = InstructionTemplate.minimal()

        content = template.render({})

        assert content is not None
        assert len(content.value) > 0
        # Minimal should be shorter than default
        assert content.token_estimate < 100

    @pytest.mark.asyncio
    async def test_fastmcp_server_with_minimal_template(self):
        """FastMCP server with minimal template instructions."""
        template = InstructionTemplate.minimal()
        content = template.render({})

        mcp = FastMCP("Test Server", instructions=content.value)

        async with Client(mcp) as client:
            init_result = client.initialize_result

            assert init_result is not None
            assert init_result.instructions is not None
            assert init_result.instructions == content.value


class TestServerCustomFile:
    """Test server initializes with custom file."""

    @pytest.fixture
    def custom_instructions_file(self):
        """Create a temporary custom instructions file."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
        ) as f:
            f.write(
                """Custom Instructions for Robot Framework MCP

1. Always use discovery tools first
2. Never guess keyword names
3. Check library availability before use

Available discovery tools: {available_tools}
"""
            )
            f.flush()
            yield Path(f.name)

        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    def test_instruction_path_validation(self, custom_instructions_file):
        """InstructionPath validates file path."""
        path = InstructionPath(str(custom_instructions_file))

        assert path.exists is True
        assert path.extension == ".txt"

    def test_adapter_loads_custom_file(self, custom_instructions_file):
        """FastMCPInstructionAdapter loads custom file."""
        adapter = FastMCPInstructionAdapter()
        path = InstructionPath(str(custom_instructions_file))
        config = InstructionConfig.create_custom(path)

        instructions = adapter.get_server_instructions(
            config, context={"available_tools": "find_keywords, execute_step"}
        )

        assert instructions is not None
        assert "Custom Instructions" in instructions
        assert "find_keywords" in instructions

    def test_adapter_creates_custom_config_from_env(self, custom_instructions_file):
        """FastMCPInstructionAdapter creates custom config from env."""
        adapter = FastMCPInstructionAdapter()

        # Set custom mode with file path using ROBOTMCP_* env vars
        env_overrides = {
            "ROBOTMCP_INSTRUCTIONS": "custom",
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "",
            "ROBOTMCP_INSTRUCTIONS_FILE": str(custom_instructions_file),
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            config = adapter.create_config_from_env()

            assert config.is_enabled is True
            assert config.mode.value == InstructionMode.CUSTOM
            assert config.custom_path is not None

    @pytest.mark.asyncio
    async def test_fastmcp_server_with_custom_file(self, custom_instructions_file):
        """FastMCP server receives custom file instructions."""
        adapter = FastMCPInstructionAdapter()
        path = InstructionPath(str(custom_instructions_file))
        config = InstructionConfig.create_custom(path)
        instructions = adapter.get_server_instructions(
            config, context={"available_tools": "test_tool"}
        )

        mcp = FastMCP("Test Server", instructions=instructions)

        async with Client(mcp) as client:
            init_result = client.initialize_result

            assert init_result is not None
            assert init_result.instructions is not None
            assert "Custom Instructions" in init_result.instructions


class TestInvalidFileFallback:
    """Test invalid file path falls back to default."""

    def test_invalid_path_extension_raises(self):
        """Invalid file extension raises ValueError."""
        with pytest.raises(ValueError, match="Invalid file extension"):
            InstructionPath("/tmp/instructions.json")

    def test_nonexistent_file_fallback_to_default(self):
        """Non-existent custom file falls back to default instructions."""
        adapter = FastMCPInstructionAdapter()

        # Create config with non-existent file
        path = InstructionPath("/tmp/nonexistent_instructions.txt")
        config = InstructionConfig.create_custom(path)

        # Should fall back to default and not raise
        instructions = adapter.get_server_instructions(config)

        assert instructions is not None
        # Should contain default content (uses "DISCOVER before EXECUTE" pattern)
        assert "DISCOVER" in instructions.upper()

    def test_adapter_validates_file_path(self):
        """FastMCPInstructionAdapter validates file paths."""
        adapter = FastMCPInstructionAdapter()

        assert adapter.validate_file_path("/tmp/test.txt") is False  # Doesn't exist
        assert adapter.validate_file_path("/tmp/test.json") is False  # Wrong ext


class TestInstructionContentInResponse:
    """Test instruction content appears correctly in MCP initialize response."""

    @pytest.mark.asyncio
    async def test_instructions_preserved_in_response(self):
        """Instructions are preserved exactly in initialize response."""
        test_instructions = "Test instruction content with special chars: @#$%"

        mcp = FastMCP("Test Server", instructions=test_instructions)

        async with Client(mcp) as client:
            init_result = client.initialize_result

            assert init_result.instructions == test_instructions

    @pytest.mark.asyncio
    async def test_multiline_instructions_preserved(self):
        """Multiline instructions are preserved."""
        test_instructions = """Line 1
Line 2
Line 3 with indentation
    - Bullet point 1
    - Bullet point 2
"""
        mcp = FastMCP("Test Server", instructions=test_instructions)

        async with Client(mcp) as client:
            init_result = client.initialize_result

            assert init_result.instructions == test_instructions
            assert "\n" in init_result.instructions

    @pytest.mark.asyncio
    async def test_unicode_instructions_preserved(self):
        """Unicode characters in instructions are preserved."""
        test_instructions = "Instructions with unicode: cafe, Zurich, Tokyo (東京)"

        mcp = FastMCP("Test Server", instructions=test_instructions)

        async with Client(mcp) as client:
            init_result = client.initialize_result

            assert init_result.instructions == test_instructions
            assert "東京" in init_result.instructions

    @pytest.mark.asyncio
    async def test_server_info_alongside_instructions(self):
        """Server info is present alongside instructions."""
        mcp = FastMCP("Test Server Name", instructions="Test instructions")

        async with Client(mcp) as client:
            init_result = client.initialize_result

            assert init_result.serverInfo is not None
            assert init_result.serverInfo.name == "Test Server Name"
            assert init_result.instructions == "Test instructions"


class TestInstructionModeValue:
    """Test InstructionMode value object."""

    def test_off_mode(self):
        """OFF mode properties."""
        mode = InstructionMode.off()

        assert mode.value == "off"
        assert mode.is_enabled is False
        assert mode.uses_custom_file is False
        assert mode.uses_default_template is False

    def test_default_mode(self):
        """DEFAULT mode properties."""
        mode = InstructionMode.default()

        assert mode.value == "default"
        assert mode.is_enabled is True
        assert mode.uses_custom_file is False
        assert mode.uses_default_template is True

    def test_custom_mode(self):
        """CUSTOM mode properties."""
        mode = InstructionMode.custom()

        assert mode.value == "custom"
        assert mode.is_enabled is True
        assert mode.uses_custom_file is True
        assert mode.uses_default_template is False

    def test_from_string(self):
        """Create mode from string."""
        mode = InstructionMode.from_string("OFF")

        assert mode.value == "off"

    def test_invalid_mode_raises(self):
        """Invalid mode value raises ValueError."""
        with pytest.raises(ValueError, match="Invalid instruction mode"):
            InstructionMode("invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

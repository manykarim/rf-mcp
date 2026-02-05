"""Integration tests for MCP instruction modes via environment variables.

Tests verify that:
1. ROBOTMCP_INSTRUCTIONS=off disables instructions
2. ROBOTMCP_INSTRUCTIONS=default uses standard template
3. ROBOTMCP_INSTRUCTIONS_TEMPLATE selects correct template
4. ROBOTMCP_INSTRUCTIONS_FILE loads custom file
5. Invalid file path falls back to default
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
    InstructionPath,
    InstructionTemplate,
    FastMCPInstructionAdapter,
)


class TestEnvironmentModeOff:
    """Test ROBOTMCP_INSTRUCTIONS=off disables instructions."""

    def test_mode_off_via_env(self):
        """ROBOTMCP_INSTRUCTIONS=off creates disabled config."""
        adapter = FastMCPInstructionAdapter()

        env_overrides = {
            "ROBOTMCP_INSTRUCTIONS": "off",
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "",
            "ROBOTMCP_INSTRUCTIONS_FILE": "",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            config = adapter.create_config_from_env()

            assert config.is_enabled is False
            assert config.mode.value == "off"

    def test_mode_off_case_insensitive(self):
        """OFF mode is case insensitive."""
        adapter = FastMCPInstructionAdapter()

        for value in ["off", "OFF", "Off", "oFf"]:
            env_overrides = {
                "ROBOTMCP_INSTRUCTIONS": value,
                "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "",
                "ROBOTMCP_INSTRUCTIONS_FILE": "",
            }
            with patch.dict(os.environ, env_overrides, clear=False):
                config = adapter.create_config_from_env()
                assert config.is_enabled is False, f"Failed for value: {value}"

    def test_off_mode_returns_none(self):
        """OFF mode returns None for server instructions."""
        adapter = FastMCPInstructionAdapter()
        config = InstructionConfig.create_off()

        instructions = adapter.get_server_instructions(config)

        assert instructions is None

    @pytest.mark.asyncio
    async def test_server_with_off_mode(self):
        """Server with OFF mode has None instructions."""
        adapter = FastMCPInstructionAdapter()

        env_overrides = {
            "ROBOTMCP_INSTRUCTIONS": "off",
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "",
            "ROBOTMCP_INSTRUCTIONS_FILE": "",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            config = adapter.create_config_from_env()
            instructions = adapter.get_server_instructions(config)

        mcp = FastMCP("Test", instructions=instructions)

        async with Client(mcp) as client:
            assert client.initialize_result.instructions is None


class TestEnvironmentModeDefault:
    """Test ROBOTMCP_INSTRUCTIONS=default uses standard template."""

    def test_mode_default_via_env(self):
        """ROBOTMCP_INSTRUCTIONS=default creates default config."""
        adapter = FastMCPInstructionAdapter()

        env_overrides = {
            "ROBOTMCP_INSTRUCTIONS": "default",
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "",
            "ROBOTMCP_INSTRUCTIONS_FILE": "",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            config = adapter.create_config_from_env()

            assert config.is_enabled is True
            assert config.mode.value == "default"
            assert config.mode.uses_default_template is True

    def test_default_mode_uses_discovery_first_template(self):
        """Default mode uses discovery-first template."""
        adapter = FastMCPInstructionAdapter()
        config = InstructionConfig.create_default()

        instructions = adapter.get_server_instructions(config)

        assert instructions is not None
        # Check for discovery-first content (uses "DISCOVER before EXECUTE" pattern)
        assert "DISCOVER" in instructions.upper()

    def test_no_env_vars_uses_default(self):
        """No environment variables (or empty) uses default mode."""
        adapter = FastMCPInstructionAdapter()

        env_overrides = {
            "ROBOTMCP_INSTRUCTIONS": "",
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "",
            "ROBOTMCP_INSTRUCTIONS_FILE": "",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            config = adapter.create_config_from_env()

            assert config.is_enabled is True
            assert config.mode.uses_default_template is True

    @pytest.mark.asyncio
    async def test_server_with_default_mode(self):
        """Server with default mode has discovery instructions."""
        adapter = FastMCPInstructionAdapter()

        env_overrides = {
            "ROBOTMCP_INSTRUCTIONS": "default",
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "",
            "ROBOTMCP_INSTRUCTIONS_FILE": "",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            config = adapter.create_config_from_env()
            instructions = adapter.get_server_instructions(config)

        mcp = FastMCP("Test", instructions=instructions)

        async with Client(mcp) as client:
            init = client.initialize_result
            assert init.instructions is not None
            assert len(init.instructions) > 100  # Substantial instructions


class TestTemplateSelection:
    """Test different instruction templates."""

    def test_discovery_first_template_content(self):
        """Discovery first template contains expected sections."""
        template = InstructionTemplate.discovery_first()
        content = template.render({"available_tools": "find_keywords, get_keyword_info"})

        value = content.value
        # Check for key sections
        assert "DISCOVERY FIRST" in value.upper() or "discovery" in value.lower()
        assert "NO GUESSING" in value.upper() or "never guess" in value.lower()
        assert "find_keywords" in value

    def test_locator_prevention_template_content(self):
        """Locator prevention template focuses on keyword validation."""
        template = InstructionTemplate.locator_prevention()
        content = template.render({})

        value = content.value
        assert "MUST NOT" in value
        assert "MUST" in value
        assert "keyword" in value.lower()

    def test_minimal_template_content(self):
        """Minimal template is concise."""
        template = InstructionTemplate.minimal()
        content = template.render({})

        value = content.value
        # Should be shorter
        assert len(value) < 500
        assert content.token_estimate < 100

    def test_template_placeholder_substitution(self):
        """Templates substitute placeholders correctly."""
        template = InstructionTemplate.discovery_first()

        content = template.render({
            "available_tools": "custom_tool_1, custom_tool_2, custom_tool_3"
        })

        assert "custom_tool_1" in content.value
        assert "custom_tool_2" in content.value
        assert "custom_tool_3" in content.value


class TestEnvironmentModeCustomFile:
    """Test ROBOTMCP_INSTRUCTIONS_FILE loads custom file."""

    @pytest.fixture
    def custom_file(self):
        """Create temporary custom instructions file."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
        ) as f:
            f.write("CUSTOM FILE INSTRUCTIONS\n\n")
            f.write("1. Follow these custom rules\n")
            f.write("2. Use {available_tools}\n")
            f.flush()
            yield Path(f.name)

        Path(f.name).unlink(missing_ok=True)

    def test_custom_mode_via_env(self, custom_file):
        """ROBOTMCP_INSTRUCTIONS=custom + ROBOTMCP_INSTRUCTIONS_FILE creates custom config."""
        adapter = FastMCPInstructionAdapter()

        env_overrides = {
            "ROBOTMCP_INSTRUCTIONS": "custom",
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "",
            "ROBOTMCP_INSTRUCTIONS_FILE": str(custom_file),
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            config = adapter.create_config_from_env()

            assert config.is_enabled is True
            assert config.mode.uses_custom_file is True
            assert config.custom_path is not None

    def test_custom_file_content_loaded(self, custom_file):
        """Custom file content is loaded into instructions."""
        adapter = FastMCPInstructionAdapter()

        env_overrides = {
            "ROBOTMCP_INSTRUCTIONS": "custom",
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "",
            "ROBOTMCP_INSTRUCTIONS_FILE": str(custom_file),
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            config = adapter.create_config_from_env()
            instructions = adapter.get_server_instructions(
                config, context={"available_tools": "my_tool"}
            )

            assert instructions is not None
            assert "CUSTOM FILE INSTRUCTIONS" in instructions
            assert "my_tool" in instructions

    @pytest.mark.asyncio
    async def test_server_with_custom_file(self, custom_file):
        """Server with custom file has file content in instructions."""
        adapter = FastMCPInstructionAdapter()

        env_overrides = {
            "ROBOTMCP_INSTRUCTIONS": "custom",
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "",
            "ROBOTMCP_INSTRUCTIONS_FILE": str(custom_file),
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            config = adapter.create_config_from_env()
            instructions = adapter.get_server_instructions(config)

        mcp = FastMCP("Test", instructions=instructions)

        async with Client(mcp) as client:
            init = client.initialize_result
            assert init.instructions is not None
            assert "CUSTOM FILE INSTRUCTIONS" in init.instructions


class TestInvalidFileFallback:
    """Test invalid file path falls back to default."""

    def test_nonexistent_file_fallback(self):
        """Non-existent file falls back to default mode."""
        adapter = FastMCPInstructionAdapter()

        # Create config with non-existent path
        path = InstructionPath("/tmp/definitely_not_a_real_file_12345.txt")
        config = InstructionConfig.create_custom(path)

        # Should not raise, falls back to default
        instructions = adapter.get_server_instructions(config)

        assert instructions is not None
        # Should be default content (uses "DISCOVER before EXECUTE" pattern)
        assert "DISCOVER" in instructions.upper()

    def test_invalid_extension_raises(self):
        """Invalid file extension raises during path creation."""
        with pytest.raises(ValueError, match="Invalid file extension"):
            InstructionPath("/tmp/instructions.exe")

    def test_invalid_extension_list(self):
        """Check which extensions are invalid."""
        invalid_extensions = [".py", ".json", ".yaml", ".sh", ".exe", ".bat"]

        for ext in invalid_extensions:
            with pytest.raises(ValueError, match="Invalid file extension"):
                InstructionPath(f"/tmp/instructions{ext}")

    def test_valid_extension_list(self):
        """Check which extensions are valid."""
        valid_extensions = [".txt", ".md", ".instruction", ".instructions"]

        for ext in valid_extensions:
            # Should not raise
            path = InstructionPath(f"/tmp/instructions{ext}")
            assert path.extension == ext

    def test_path_traversal_blocked(self):
        """Path traversal is blocked."""
        with pytest.raises(ValueError, match="Path traversal"):
            InstructionPath("../../../etc/passwd")

    def test_env_var_with_invalid_path_uses_default(self):
        """Invalid path in env var falls back to default."""
        adapter = FastMCPInstructionAdapter()

        env_overrides = {
            "ROBOTMCP_INSTRUCTIONS": "custom",
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "",
            "ROBOTMCP_INSTRUCTIONS_FILE": "/nonexistent/path.txt",
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            config = adapter.create_config_from_env()
            instructions = adapter.get_server_instructions(config)

            # Should fall back to default, not fail
            assert instructions is not None

    @pytest.mark.asyncio
    async def test_server_fallback_after_invalid_file(self):
        """Server starts with fallback instructions after invalid file."""
        adapter = FastMCPInstructionAdapter()

        path = InstructionPath("/tmp/nonexistent_file.txt")
        config = InstructionConfig.create_custom(path)
        instructions = adapter.get_server_instructions(config)

        mcp = FastMCP("Test", instructions=instructions)

        async with Client(mcp) as client:
            init = client.initialize_result
            assert init.instructions is not None
            # Verify fallback content (uses "DISCOVER before EXECUTE" pattern)
            assert "DISCOVER" in init.instructions.upper()


class TestEnvironmentVariablePrecedence:
    """Test precedence of environment variables."""

    @pytest.fixture
    def custom_file(self):
        """Create temporary custom instructions file."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
        ) as f:
            f.write("CUSTOM FROM FILE\n")
            f.flush()
            yield Path(f.name)

        Path(f.name).unlink(missing_ok=True)

    def test_off_mode_overrides_file(self, custom_file):
        """ROBOTMCP_INSTRUCTIONS=off disables even if file is specified."""
        adapter = FastMCPInstructionAdapter()

        env_overrides = {
            "ROBOTMCP_INSTRUCTIONS": "off",
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "",
            "ROBOTMCP_INSTRUCTIONS_FILE": str(custom_file),
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            config = adapter.create_config_from_env()

            # Mode=off should win
            assert config.is_enabled is False

    def test_file_env_used_with_custom_mode(self, custom_file):
        """ROBOTMCP_INSTRUCTIONS_FILE is used when mode=custom."""
        adapter = FastMCPInstructionAdapter()

        env_overrides = {
            "ROBOTMCP_INSTRUCTIONS": "custom",
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "",
            "ROBOTMCP_INSTRUCTIONS_FILE": str(custom_file),
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            config = adapter.create_config_from_env()
            instructions = adapter.get_server_instructions(config)

            assert "CUSTOM FROM FILE" in instructions

    def test_explicit_mode_default_ignores_file(self, custom_file):
        """ROBOTMCP_INSTRUCTIONS=default ignores file env."""
        adapter = FastMCPInstructionAdapter()

        env_overrides = {
            "ROBOTMCP_INSTRUCTIONS": "default",
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "",
            "ROBOTMCP_INSTRUCTIONS_FILE": str(custom_file),
        }
        with patch.dict(os.environ, env_overrides, clear=False):
            config = adapter.create_config_from_env()

            # Mode=default should use default template
            assert config.mode.uses_default_template is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

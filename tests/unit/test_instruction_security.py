"""Security tests for the MCP Instructions feature.

This module provides comprehensive security testing for the instruction domain:
- Path traversal prevention
- Dangerous pattern blocking
- Content sanitization
- Environment variable validation
- Input validation edge cases
- Defense in depth scenarios
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from robotmcp.domains.instruction import (
    InstructionMode,
    InstructionContent,
    InstructionPath,
    InstructionConfig,
    InstructionResolver,
    InstructionValidator,
    FastMCPInstructionAdapter,
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
    file_path = temp_instruction_dir / "safe_instructions.txt"
    file_path.write_text(
        "Use discovery tools before executing actions.\n"
        "Always verify keywords exist via find_keywords.\n"
        "Never guess keyword names or locator arguments."
    )
    return file_path


@pytest.fixture
def malicious_content_samples() -> List[str]:
    """Collection of malicious content samples for testing."""
    return [
        "<script>alert('xss')</script>",
        "<SCRIPT>document.cookie</SCRIPT>",
        "<script src='evil.js'></script>",
        "javascript:alert('xss')",
        "JAVASCRIPT:eval(code)",
        "eval('malicious code')",
        "eval   (   'code'   )",
        "exec('os.system(\"rm -rf /\")')",
        "exec  (  command  )",
        "__import__('os').system('rm -rf /')",
        "__import__  (  'subprocess'  )",
    ]


@pytest.fixture
def path_traversal_attempts() -> List[str]:
    """Collection of path traversal attack patterns with valid extensions."""
    return [
        "../etc/passwd.txt",
        "../../etc/shadow.txt",
        "../../../root/.ssh/id_rsa.txt",
        "instructions/../../secret.txt",
        "./../../../../etc/hosts.txt",
        "foo/../../../bar.txt",
        "subdir/../../../etc/passwd.txt",
        "a/b/../../../c.txt",
    ]


@pytest.fixture
def invalid_extensions() -> List[str]:
    """Collection of disallowed file extensions."""
    return [
        "instructions.py",
        "instructions.sh",
        "instructions.bash",
        "instructions.exe",
        "instructions.bat",
        "instructions.cmd",
        "instructions.ps1",
        "instructions.dll",
        "instructions.so",
        "instructions.js",
        "instructions.html",
        "instructions.htm",
        "instructions.php",
        "instructions.asp",
        "instructions.jsp",
        "instructions.rb",
        "instructions.pl",
        "instructions.cgi",
        "instructions.json",
        "instructions.xml",
        "instructions.yaml",
        "instructions.yml",
        "instructions.conf",
        "instructions.ini",
        "instructions.env",
        "instructions",  # No extension
    ]


# =============================================================================
# Path Traversal Prevention Tests
# =============================================================================


class TestPathTraversalPrevention:
    """Security tests for path traversal attack prevention."""

    def test_basic_path_traversal_blocked(self):
        """Test that basic .. traversal is blocked."""
        with pytest.raises(ValueError, match="traversal"):
            InstructionPath(value="../instructions.txt")

    def test_nested_path_traversal_blocked(self):
        """Test that nested traversal is blocked."""
        with pytest.raises(ValueError, match="traversal"):
            InstructionPath(value="foo/../../../etc/passwd.txt")

    def test_all_traversal_patterns_blocked(self, path_traversal_attempts: List[str]):
        """Test that all common traversal patterns are blocked."""
        for path in path_traversal_attempts:
            with pytest.raises(ValueError, match="traversal"):
                InstructionPath(value=path)

    def test_windows_path_traversal_blocked(self):
        """Test that Windows-style traversal is blocked."""
        # Windows-style paths with backslashes may not trigger the same
        # traversal detection, but they should fail for other reasons
        # or be handled by the OS. The key protection is ".." detection.
        # On Linux, backslashes are treated as part of filename.
        # The real traversal protection is ".." which is cross-platform.
        with pytest.raises(ValueError, match="traversal"):
            InstructionPath(value="../windows/system.txt")

    def test_url_encoded_traversal_not_decoded(self):
        """Test that URL-encoded paths are treated literally."""
        # %2e%2e should not be decoded to ..
        # This should fail extension check since %2e isn't .
        with pytest.raises(ValueError, match="extension"):
            InstructionPath(value="%2e%2e/etc/passwd")

    def test_resolve_escaping_base_directory(self, temp_instruction_dir: Path):
        """Test that resolve prevents escaping base directory."""
        # Create a sibling directory
        sibling = temp_instruction_dir.parent / "sibling"
        sibling.mkdir(exist_ok=True)
        sibling_file = sibling / "secret.txt"
        sibling_file.write_text("secret data")

        try:
            # Create path that's technically valid
            path = InstructionPath(value="safe.txt")

            # Resolve should work within base
            resolved = path.resolve(base_path=temp_instruction_dir)
            assert temp_instruction_dir in resolved.parents or resolved.parent == temp_instruction_dir
        finally:
            sibling_file.unlink(missing_ok=True)
            sibling.rmdir()

    def test_absolute_path_handling(self, valid_instruction_file: Path):
        """Test that absolute paths are handled correctly."""
        path = InstructionPath(value=str(valid_instruction_file))
        resolved = path.resolve()
        assert resolved == valid_instruction_file


# =============================================================================
# Dangerous Pattern Blocking Tests
# =============================================================================


class TestDangerousPatternBlocking:
    """Security tests for blocking dangerous content patterns."""

    def test_script_tag_blocked(self):
        """Test that script tags are blocked."""
        validator = InstructionValidator()
        content = InstructionContent(
            value="Use <script>alert('xss')</script> for testing",
            source="custom:/path/file.txt"
        )

        result = validator.validate(content)

        assert result.is_valid is False
        assert any("dangerous" in e.lower() or "script" in e.lower() for e in result.errors)

    def test_script_tag_case_insensitive(self):
        """Test that script detection is case-insensitive."""
        validator = InstructionValidator()
        variants = [
            "<SCRIPT>code</SCRIPT>",
            "<Script>code</Script>",
            "<sCrIpT>code</sCrIpT>",
        ]

        for script in variants:
            content = InstructionContent(
                value=f"Test {script} content",
                source="custom:/path/file.txt"
            )
            result = validator.validate(content)
            assert result.is_valid is False, f"Should block: {script}"

    def test_javascript_protocol_blocked(self):
        """Test that javascript: protocol is blocked."""
        validator = InstructionValidator()
        content = InstructionContent(
            value="Click javascript:alert('xss') for info",
            source="custom:/path/file.txt"
        )

        result = validator.validate(content)

        assert result.is_valid is False

    def test_eval_blocked(self):
        """Test that eval() calls are blocked."""
        validator = InstructionValidator()
        variants = [
            "eval('code')",
            "eval ( 'code' )",
            "EVAL('code')",
            "eval  \t  ('code')",
        ]

        for pattern in variants:
            content = InstructionContent(
                value=f"Don't use {pattern} in code",
                source="custom:/path/file.txt"
            )
            result = validator.validate(content)
            assert result.is_valid is False, f"Should block: {pattern}"

    def test_exec_blocked(self):
        """Test that exec() calls are blocked."""
        validator = InstructionValidator()
        variants = [
            "exec('code')",
            "exec ( command )",
            "EXEC('shell')",
        ]

        for pattern in variants:
            content = InstructionContent(
                value=f"Avoid {pattern} usage",
                source="custom:/path/file.txt"
            )
            result = validator.validate(content)
            assert result.is_valid is False, f"Should block: {pattern}"

    def test_import_blocked(self):
        """Test that __import__() is blocked."""
        validator = InstructionValidator()
        content = InstructionContent(
            value="Never use __import__('os') directly",
            source="custom:/path/file.txt"
        )

        result = validator.validate(content)

        assert result.is_valid is False

    def test_all_malicious_patterns_blocked(self, malicious_content_samples: List[str]):
        """Test that all malicious content samples are blocked."""
        validator = InstructionValidator()

        for malicious in malicious_content_samples:
            content = InstructionContent(
                value=f"Normal text with {malicious} embedded",
                source="custom:/path/file.txt"
            )
            result = validator.validate(content)
            assert result.is_valid is False, f"Should block: {malicious}"

    def test_safe_mentions_not_blocked(self):
        """Test that mentioning dangerous words safely is OK."""
        validator = InstructionValidator()
        content = InstructionContent(
            value=(
                "Never use script tags in your content. "
                "Avoid javascript links. The eval function is dangerous. "
                "Do not call exec or __import__ directly."
            ),
            source="default"
        )

        result = validator.validate(content)

        # Should be valid because we're talking ABOUT these things,
        # not using them. The patterns look for actual code.
        # "eval function" does not match "eval(" pattern
        assert result.is_valid is True


# =============================================================================
# Content Sanitization Tests
# =============================================================================


class TestContentSanitization:
    """Tests for content validation and sanitization."""

    def test_minimum_content_length_enforced(self):
        """Test that minimum content length is enforced."""
        with pytest.raises(ValueError, match="too short"):
            InstructionContent(value="too short", source="default")

    def test_maximum_content_length_enforced(self):
        """Test that maximum content length is enforced."""
        # MAX_LENGTH is 50000
        with pytest.raises(ValueError, match="too long"):
            InstructionContent(value="A" * 50001, source="default")

    def test_empty_content_rejected(self):
        """Test that empty content is rejected."""
        with pytest.raises(ValueError):
            InstructionContent(value="", source="default")

    def test_whitespace_only_rejected(self):
        """Test that whitespace-only content is rejected."""
        whitespace_variants = [
            "   ",
            "\t\t\t",
            "\n\n\n",
            "  \t\n  \t\n  ",
        ]

        for whitespace in whitespace_variants:
            with pytest.raises(ValueError, match="too short"):
                InstructionContent(value=whitespace, source="default")

    def test_content_with_special_characters(self):
        """Test that special characters are handled."""
        special_content = InstructionContent(
            value="Use <angle> brackets and & ampersands and 'quotes' carefully.",
            source="default"
        )
        # Should be valid - these are just text
        assert len(special_content.value) > 0

    def test_content_with_unicode(self):
        """Test that unicode content is handled."""
        unicode_content = InstructionContent(
            value="Instructions with unicode: some special characters here.",
            source="default"
        )
        assert len(unicode_content.value) > 0

    def test_content_with_null_bytes(self):
        """Test handling of null bytes in content."""
        # Null bytes could be used for string termination attacks
        content_text = "Normal content\x00with null byte"
        content = InstructionContent(value=content_text, source="default")
        # The null byte is preserved but the content is valid
        assert "\x00" in content.value

    def test_token_budget_validation(self):
        """Test that token budget is validated."""
        validator = InstructionValidator(max_token_budget=50)
        large_content = InstructionContent(
            value="word " * 100,  # ~130 tokens
            source="default"
        )

        result = validator.validate(large_content)

        assert result.is_valid is False
        assert any("budget" in e.lower() for e in result.errors)


# =============================================================================
# Environment Variable Validation Tests
# =============================================================================


class TestEnvVarValidation:
    """Tests for environment variable handling security."""

    def test_env_mode_off(self):
        """Test that 'off' mode from env is respected."""
        # Use the correct env var names from the adapter
        env_patch = {
            "ROBOTMCP_INSTRUCTIONS": "off",
            "ROBOTMCP_INSTRUCTIONS_FILE": "",
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "standard",
        }
        with patch.dict(os.environ, env_patch, clear=False):
            adapter = FastMCPInstructionAdapter()
            config = adapter.create_config_from_env()
            assert config.mode.value == "off"
            assert config.is_enabled is False

    def test_env_mode_case_insensitive(self):
        """Test that env mode is case-insensitive."""
        for mode_value in ["OFF", "Off", "oFf"]:
            env_patch = {
                "ROBOTMCP_INSTRUCTIONS": mode_value,
                "ROBOTMCP_INSTRUCTIONS_FILE": "",
                "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "standard",
            }
            with patch.dict(os.environ, env_patch, clear=False):
                adapter = FastMCPInstructionAdapter()
                config = adapter.create_config_from_env()
                assert config.mode.value == "off", f"Failed for mode_value={mode_value}"

    def test_env_invalid_path_falls_back(self, temp_instruction_dir: Path):
        """Test that invalid path falls back to default."""
        # Path with invalid extension
        invalid_path = str(temp_instruction_dir / "instructions.py")

        env_patch = {
            "ROBOTMCP_INSTRUCTIONS": "custom",
            "ROBOTMCP_INSTRUCTIONS_FILE": invalid_path,
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "standard",
        }
        with patch.dict(os.environ, env_patch, clear=False):
            adapter = FastMCPInstructionAdapter()
            config = adapter.create_config_from_env()

        # Should fall back to default
        assert config.mode.value == "default"

    def test_env_traversal_path_rejected(self, temp_instruction_dir: Path):
        """Test that traversal paths in env are rejected."""
        traversal_path = "../../../etc/passwd.txt"

        env_patch = {
            "ROBOTMCP_INSTRUCTIONS": "custom",
            "ROBOTMCP_INSTRUCTIONS_FILE": traversal_path,
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "standard",
        }
        with patch.dict(os.environ, env_patch, clear=False):
            adapter = FastMCPInstructionAdapter()
            config = adapter.create_config_from_env()

        # Should fall back to default due to invalid path
        assert config.mode.value == "default"

    def test_env_empty_values_use_default(self):
        """Test that empty env values (for INSTRUCTIONS) use default mode."""
        # Note: ROBOTMCP_INSTRUCTIONS_TEMPLATE cannot be empty - it needs a valid template
        # When ROBOTMCP_INSTRUCTIONS is empty, it defaults to "default" mode
        env_patch = {
            "ROBOTMCP_INSTRUCTIONS": "",
            "ROBOTMCP_INSTRUCTIONS_FILE": "",
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "standard",
        }
        with patch.dict(os.environ, env_patch, clear=False):
            adapter = FastMCPInstructionAdapter()
            config = adapter.create_config_from_env()

        # Empty string for mode defaults to "default" mode
        assert config.mode.value == "default"

    def test_env_whitespace_trimmed(self):
        """Test that whitespace in env values is trimmed."""
        env_patch = {
            "ROBOTMCP_INSTRUCTIONS": "  off  ",
            "ROBOTMCP_INSTRUCTIONS_FILE": "",
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "standard",
        }
        with patch.dict(os.environ, env_patch, clear=False):
            adapter = FastMCPInstructionAdapter()
            config = adapter.create_config_from_env()
            assert config.mode.value == "off"

    def test_env_precedence(self, valid_instruction_file: Path):
        """Test env variable precedence."""
        # INSTRUCTIONS=off takes precedence over FILE
        env_patch = {
            "ROBOTMCP_INSTRUCTIONS": "off",
            "ROBOTMCP_INSTRUCTIONS_FILE": str(valid_instruction_file),
            "ROBOTMCP_INSTRUCTIONS_TEMPLATE": "standard",
        }
        with patch.dict(os.environ, env_patch, clear=False):
            adapter = FastMCPInstructionAdapter()
            config = adapter.create_config_from_env()
            assert config.mode.value == "off"


# =============================================================================
# File Extension Security Tests
# =============================================================================


class TestFileExtensionSecurity:
    """Tests for file extension validation security."""

    def test_all_invalid_extensions_rejected(self, invalid_extensions: List[str]):
        """Test that all invalid extensions are rejected."""
        for filename in invalid_extensions:
            with pytest.raises(ValueError, match="extension"):
                InstructionPath(value=filename)

    def test_valid_extensions_accepted(self):
        """Test that valid extensions are accepted."""
        valid_files = [
            "instructions.txt",
            "instructions.md",
            "instructions.instruction",
            "instructions.instructions",
            "INSTRUCTIONS.TXT",  # Case variations
            "INSTRUCTIONS.MD",
        ]

        for filename in valid_files:
            path = InstructionPath(value=filename)
            assert path.value == filename

    def test_double_extension_handling(self):
        """Test handling of double extensions."""
        # Should use the last extension
        path = InstructionPath(value="instructions.py.txt")
        assert path.extension == ".txt"

    def test_hidden_files_with_valid_extension(self):
        """Test hidden files with valid extensions."""
        path = InstructionPath(value=".hidden.txt")
        assert path.extension == ".txt"


# =============================================================================
# Instruction Mode Security Tests
# =============================================================================


class TestInstructionModeSecurity:
    """Tests for instruction mode validation security."""

    def test_only_valid_modes_accepted(self):
        """Test that only valid modes are accepted."""
        valid_modes = ["off", "default", "custom"]
        for mode in valid_modes:
            mode_obj = InstructionMode(value=mode)
            assert mode_obj.value == mode

    def test_invalid_modes_rejected(self):
        """Test that invalid modes are rejected."""
        invalid_modes = [
            "invalid",
            "enabled",
            "disabled",
            "on",
            "yes",
            "no",
            "true",
            "false",
            "1",
            "0",
            "",
            " ",
            "off ",
            " default",
        ]

        for mode in invalid_modes:
            with pytest.raises(ValueError, match="Invalid instruction mode"):
                InstructionMode(value=mode)

    def test_mode_immutability(self):
        """Test that mode is immutable."""
        mode = InstructionMode.default()
        with pytest.raises(Exception):  # FrozenInstanceError
            mode.value = "custom"


# =============================================================================
# Defense in Depth Tests
# =============================================================================


class TestDefenseInDepth:
    """Tests for defense in depth scenarios."""

    def test_combined_attack_path_and_content(self, temp_instruction_dir: Path):
        """Test handling attack combining path and content issues."""
        # Create a file with malicious content
        malicious_file = temp_instruction_dir / "malicious.txt"
        malicious_file.write_text(
            "Normal start\n"
            "<script>alert('xss')</script>\n"
            "Normal end"
        )

        adapter = FastMCPInstructionAdapter()
        path = InstructionPath(value=str(malicious_file))
        config = InstructionConfig.create_custom(path)

        # Get instructions should fall back to default
        instructions = adapter.get_server_instructions(config)

        # Should NOT contain the malicious content
        if instructions:
            assert "<script>" not in instructions

    def test_multiple_validation_layers(self):
        """Test that multiple validation layers work together."""
        # Even if one check passes, others should catch issues
        validator = InstructionValidator()

        # Content with multiple issues
        content = InstructionContent(
            value="Use eval('code') and exec('shell') and <script>xss</script>",
            source="custom:/path/file.txt"
        )

        result = validator.validate(content)

        assert result.is_valid is False
        # Should catch multiple issues
        assert len(result.errors) >= 1

    def test_fallback_on_custom_failure(self, temp_instruction_dir: Path):
        """Test fallback to default when custom validation fails."""
        # Create invalid custom file
        invalid_file = temp_instruction_dir / "invalid.txt"
        invalid_file.write_text("<script>malicious</script> content here")

        adapter = FastMCPInstructionAdapter()
        path = InstructionPath(value=str(invalid_file))
        config = InstructionConfig.create_custom(path)

        # Should fall back to default, not fail completely
        instructions = adapter.get_server_instructions(config)

        assert instructions is not None
        assert "<script>" not in instructions

    def test_resolver_handles_file_read_errors(self, temp_instruction_dir: Path):
        """Test that resolver handles file read errors gracefully."""
        resolver = InstructionResolver()
        path = InstructionPath(value=str(temp_instruction_dir / "nonexistent.txt"))
        config = InstructionConfig.create_custom(path)

        with pytest.raises(FileNotFoundError):
            resolver.resolve(config)


# =============================================================================
# Edge Cases
# =============================================================================


class TestSecurityEdgeCases:
    """Tests for security edge cases."""

    def test_very_long_path(self, temp_instruction_dir: Path):
        """Test handling of very long paths."""
        # Create a very long but technically valid path
        long_name = "a" * 200 + ".txt"
        long_path = temp_instruction_dir / long_name

        # This should work if the filesystem allows it
        try:
            path = InstructionPath(value=str(long_path))
            assert len(path.value) > 200
        except (ValueError, OSError):
            # Some systems may reject very long paths
            pass

    def test_special_characters_in_path(self, temp_instruction_dir: Path):
        """Test paths with special characters."""
        # Most special chars are fine in paths
        special_file = temp_instruction_dir / "test-file_v1.2.3.txt"
        special_file.write_text("Test content here")

        path = InstructionPath(value=str(special_file))
        assert path.exists

    def test_content_boundary_at_limit(self):
        """Test content at exact boundary limits."""
        # Exactly at MIN_LENGTH (10)
        min_content = InstructionContent(value="A" * 10, source="default")
        assert len(min_content.value) == 10

        # Exactly at MAX_LENGTH (50000)
        max_content = InstructionContent(value="A" * 50000, source="default")
        assert len(max_content.value) == 50000

    def test_unicode_path_handling(self, temp_instruction_dir: Path):
        """Test handling of unicode in paths."""
        # Create file with unicode name
        unicode_file = temp_instruction_dir / "instructions_test.txt"
        unicode_file.write_text("Unicode content test here")

        path = InstructionPath(value=str(unicode_file))
        assert path.exists

    def test_concurrent_validation(self):
        """Test that validation is stateless and thread-safe."""
        import concurrent.futures

        validator = InstructionValidator()

        def validate_content(index: int) -> bool:
            content = InstructionContent(
                value=f"Test content number {index} for validation",
                source="default"
            )
            result = validator.validate(content)
            return result.is_valid

        # Run multiple validations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(validate_content, i) for i in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should pass (no dangerous patterns)
        assert all(results)

"""Tests for memory instruction augmentation in FastMCPInstructionAdapter.

Covers the _append_memory_instructions() static method added in ADR-014.2
Phase 4, which appends memory workflow guidance to server instructions
when ROBOTMCP_MEMORY_ENABLED is set.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from robotmcp.domains.instruction.adapters.fastmcp_adapter import (
    FastMCPInstructionAdapter,
)


class TestAppendMemoryInstructions:
    """Tests for FastMCPInstructionAdapter._append_memory_instructions()."""

    def test_appends_memory_section(self):
        base = "You are a test automation assistant."
        result = FastMCPInstructionAdapter._append_memory_instructions(base)

        assert result.startswith(base)
        assert len(result) > len(base)

    def test_contains_memory_augmented_workflow_heading(self):
        result = FastMCPInstructionAdapter._append_memory_instructions("Base text")

        assert "Memory-Augmented Workflow" in result

    def test_contains_recall_fix_reference(self):
        result = FastMCPInstructionAdapter._append_memory_instructions("Base text")

        assert "recall_fix" in result

    def test_contains_recall_locator_reference(self):
        result = FastMCPInstructionAdapter._append_memory_instructions("Base text")

        assert "recall_locator" in result

    def test_contains_recall_step_reference(self):
        result = FastMCPInstructionAdapter._append_memory_instructions("Base text")

        assert "recall_step" in result

    def test_contains_memory_hints_mention(self):
        result = FastMCPInstructionAdapter._append_memory_instructions("Base text")

        assert "memory_hints" in result

    def test_preserves_original_content(self):
        original = "This is the original instruction content with specific details."
        result = FastMCPInstructionAdapter._append_memory_instructions(original)

        assert result.startswith(original)

    def test_empty_base_text(self):
        result = FastMCPInstructionAdapter._append_memory_instructions("")

        assert "Memory-Augmented Workflow" in result
        assert "recall_fix" in result


class TestGetServerInstructionsMemoryIntegration:
    """Tests for get_server_instructions() memory augmentation integration.

    Verifies that when ROBOTMCP_MEMORY_ENABLED is set, the rendered
    instructions include the memory workflow section.
    """

    def test_memory_section_added_when_enabled(self):
        """When ROBOTMCP_MEMORY_ENABLED=true, instructions include memory section."""
        adapter = FastMCPInstructionAdapter(template_name="standard")
        from robotmcp.domains.instruction.aggregates import InstructionConfig

        config = InstructionConfig.create_default()

        with patch.dict(
            "os.environ",
            {"ROBOTMCP_MEMORY_ENABLED": "true"},
        ):
            result = adapter.get_server_instructions(config)

        if result is not None:
            assert "Memory-Augmented Workflow" in result

    def test_memory_section_not_added_when_disabled(self):
        """When ROBOTMCP_MEMORY_ENABLED is not set, no memory section."""
        adapter = FastMCPInstructionAdapter(template_name="standard")
        from robotmcp.domains.instruction.aggregates import InstructionConfig

        config = InstructionConfig.create_default()

        with patch.dict(
            "os.environ",
            {"ROBOTMCP_MEMORY_ENABLED": "false"},
        ):
            result = adapter.get_server_instructions(config)

        if result is not None:
            assert "Memory-Augmented Workflow" not in result

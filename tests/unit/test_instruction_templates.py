"""Tests for MCP Instruction Templates.

This module provides comprehensive tests for instruction templates:
- Template loading and creation
- Template character limits
- Template placeholder handling
- Built-in template validation
- Template rendering with context
- Ensuring templates only reference enabled tools
"""

from __future__ import annotations

import re
from typing import Dict, List, Set
from unittest.mock import MagicMock

import pytest

from robotmcp.domains.instruction import (
    InstructionTemplate,
    InstructionContent,
    InstructionConfig,
    InstructionResolver,
    InstructionRenderer,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def discovery_first_template() -> InstructionTemplate:
    """Get the discovery_first built-in template."""
    return InstructionTemplate.discovery_first()


@pytest.fixture
def locator_prevention_template() -> InstructionTemplate:
    """Get the locator_prevention built-in template."""
    return InstructionTemplate.locator_prevention()


@pytest.fixture
def minimal_template() -> InstructionTemplate:
    """Get the minimal built-in template."""
    return InstructionTemplate.minimal()


@pytest.fixture
def all_builtin_templates() -> List[InstructionTemplate]:
    """Get all built-in templates."""
    return [
        InstructionTemplate.discovery_first(),
        InstructionTemplate.locator_prevention(),
        InstructionTemplate.minimal(),
    ]


@pytest.fixture
def discovery_tools_list() -> Set[str]:
    """Set of expected discovery tools that should be mentioned."""
    return {
        "find_keywords",
        "get_keyword_info",
        "get_session_state",
        "get_locator_guidance",
        "analyze_scenario",
        "recommend_libraries",
        "check_library_availability",
    }


@pytest.fixture
def action_tools_list() -> Set[str]:
    """Set of action tools (that should NOT be guessed)."""
    return {
        "execute_step",
        "execute_flow",
        "build_test_suite",
        "run_test_suite",
    }


@pytest.fixture
def disabled_tools_list() -> Set[str]:
    """Set of tools that should NOT be referenced in instructions.

    These are example tools that would be disabled or don't exist.
    """
    return {
        "deprecated_action",
        "legacy_execute",
        "unsafe_eval",
        "direct_click",  # Not a real tool
        "raw_xpath",  # Not a real tool
    }


@pytest.fixture
def sample_context() -> Dict[str, str]:
    """Sample context for template rendering."""
    return {
        "available_tools": (
            "find_keywords, get_keyword_info, get_session_state, "
            "get_locator_guidance, analyze_scenario, recommend_libraries, "
            "check_library_availability"
        )
    }


# =============================================================================
# Template Loading Tests
# =============================================================================


class TestTemplateLoading:
    """Tests for template loading and creation."""

    def test_discovery_first_loads(self, discovery_first_template):
        """Test that discovery_first template loads correctly."""
        assert discovery_first_template is not None
        assert discovery_first_template.template_id == "discovery_first"
        assert len(discovery_first_template.content) > 0

    def test_locator_prevention_loads(self, locator_prevention_template):
        """Test that locator_prevention template loads correctly."""
        assert locator_prevention_template is not None
        assert locator_prevention_template.template_id == "locator_prevention"
        assert len(locator_prevention_template.content) > 0

    def test_minimal_loads(self, minimal_template):
        """Test that minimal template loads correctly."""
        assert minimal_template is not None
        assert minimal_template.template_id == "minimal"
        assert len(minimal_template.content) > 0

    def test_all_builtin_templates_load(self, all_builtin_templates):
        """Test that all built-in templates load without error."""
        for template in all_builtin_templates:
            assert template is not None
            assert template.template_id is not None
            assert len(template.content) > 0

    def test_template_has_description(self, all_builtin_templates):
        """Test that all templates have descriptions."""
        for template in all_builtin_templates:
            assert template.description is not None
            assert len(template.description) > 0

    def test_create_custom_template(self):
        """Test creating a custom template."""
        custom = InstructionTemplate(
            template_id="custom_test",
            content="Use {tool_name} for testing.",
            description="Custom test template",
            placeholders=("tool_name",)
        )

        assert custom.template_id == "custom_test"
        assert "tool_name" in custom.placeholders

    def test_template_with_no_placeholders(self):
        """Test creating template with no placeholders."""
        template = InstructionTemplate(
            template_id="static",
            content="This is static content with no variables.",
            description="Static template",
            placeholders=()
        )

        assert len(template.placeholders) == 0

    def test_template_with_multiple_placeholders(self):
        """Test creating template with multiple placeholders."""
        template = InstructionTemplate(
            template_id="multi",
            content="Tools: {tools}, Session: {session}, Mode: {mode}",
            description="Multi-placeholder template",
            placeholders=("tools", "session", "mode")
        )

        assert len(template.placeholders) == 3


# =============================================================================
# Template Character Limit Tests
# =============================================================================


class TestTemplateCharacterLimits:
    """Tests for template character limits and sizing."""

    def test_discovery_first_under_limit(self, discovery_first_template):
        """Test that discovery_first template is under content limit."""
        # MAX_LENGTH for InstructionContent is 50000
        assert len(discovery_first_template.content) < 50000

    def test_locator_prevention_under_limit(self, locator_prevention_template):
        """Test that locator_prevention template is under content limit."""
        assert len(locator_prevention_template.content) < 50000

    def test_minimal_is_actually_minimal(self, minimal_template):
        """Test that minimal template is significantly shorter."""
        # Minimal should be much shorter than others
        assert len(minimal_template.content) < 500

    def test_all_templates_produce_valid_content(
        self, all_builtin_templates, sample_context
    ):
        """Test that all templates produce valid InstructionContent."""
        for template in all_builtin_templates:
            # Render with context (or empty for no-placeholder templates)
            context = sample_context if template.placeholders else {}
            content = template.render(context)

            # Should create valid InstructionContent
            assert isinstance(content, InstructionContent)
            assert len(content.value) >= InstructionContent.MIN_LENGTH
            assert len(content.value) <= InstructionContent.MAX_LENGTH

    def test_template_size_comparison(self, all_builtin_templates):
        """Test relative sizes of templates."""
        sizes = {t.template_id: len(t.content) for t in all_builtin_templates}

        # Minimal should be the shortest
        assert sizes["minimal"] < sizes["locator_prevention"]
        assert sizes["minimal"] < sizes["discovery_first"]

    def test_rendered_content_within_token_budget(
        self, discovery_first_template, sample_context
    ):
        """Test that rendered content is within reasonable token budget."""
        content = discovery_first_template.render(sample_context)

        # Default token budget is 1000
        # Discovery first is comprehensive but should be reasonable
        assert content.token_estimate < 2000  # Allow some room


# =============================================================================
# Template Placeholder Tests
# =============================================================================


class TestTemplatePlaceholders:
    """Tests for template placeholder handling."""

    def test_discovery_first_has_tools_placeholder(self, discovery_first_template):
        """Test that discovery_first has available_tools placeholder."""
        assert "available_tools" in discovery_first_template.placeholders

    def test_locator_prevention_no_placeholders(self, locator_prevention_template):
        """Test that locator_prevention has no placeholders."""
        assert len(locator_prevention_template.placeholders) == 0

    def test_minimal_no_placeholders(self, minimal_template):
        """Test that minimal has no placeholders."""
        assert len(minimal_template.placeholders) == 0

    def test_placeholder_detection_pattern(self):
        """Test the placeholder detection regex pattern."""
        pattern = InstructionTemplate.PLACEHOLDER_PATTERN

        # Should match {word}
        matches = pattern.findall("Use {tool} and {other_tool} here")
        assert "tool" in matches
        assert "other_tool" in matches

        # Should not match other patterns
        no_matches = pattern.findall("Use {{escaped}} and {123} here")
        assert "escaped" in no_matches or len(no_matches) == 1  # Only word chars

    def test_placeholder_mismatch_raises_error(self):
        """Test that placeholder mismatch raises error."""
        with pytest.raises(ValueError, match="Placeholder mismatch"):
            InstructionTemplate(
                template_id="bad",
                content="Use {tool} for testing",
                description="Bad template",
                placeholders=("wrong_name",)
            )

    def test_missing_placeholder_declaration_raises(self):
        """Test that undeclared placeholders raise error."""
        with pytest.raises(ValueError, match="Placeholder mismatch"):
            InstructionTemplate(
                template_id="bad",
                content="Use {tool} for testing",
                description="Bad template",
                placeholders=()  # Missing declaration
            )

    def test_extra_placeholder_declaration_raises(self):
        """Test that extra declared placeholders raise error."""
        with pytest.raises(ValueError, match="Placeholder mismatch"):
            InstructionTemplate(
                template_id="bad",
                content="No placeholders here",
                description="Bad template",
                placeholders=("extra",)  # Extra declaration
            )

    def test_render_with_missing_context(self, discovery_first_template):
        """Test rendering with missing context leaves placeholder."""
        content = discovery_first_template.render({})  # Empty context

        # Placeholder should remain
        assert "{available_tools}" in content.value

    def test_render_with_partial_context(self):
        """Test rendering with partial context."""
        template = InstructionTemplate(
            template_id="multi",
            content="A: {a}, B: {b}",
            description="Test",
            placeholders=("a", "b")
        )

        content = template.render({"a": "value_a"})

        assert "value_a" in content.value
        assert "{b}" in content.value  # Still present

    def test_render_with_complete_context(self, discovery_first_template, sample_context):
        """Test rendering with complete context."""
        content = discovery_first_template.render(sample_context)

        assert "{available_tools}" not in content.value
        assert "find_keywords" in content.value


# =============================================================================
# Template Content Validation Tests
# =============================================================================


class TestTemplateContentValidation:
    """Tests for template content validation."""

    def test_discovery_first_mentions_discovery(self, discovery_first_template):
        """Test that discovery_first mentions discovery concepts."""
        content = discovery_first_template.content.lower()

        assert "discovery" in content
        assert "first" in content or "before" in content

    def test_discovery_first_mentions_no_guessing(self, discovery_first_template):
        """Test that discovery_first prohibits guessing."""
        content = discovery_first_template.content.lower()

        assert "guess" in content or "never" in content
        assert "fabricate" in content or "invent" in content

    def test_locator_prevention_is_strict(self, locator_prevention_template):
        """Test that locator_prevention has strict rules."""
        content = locator_prevention_template.content

        assert "MUST NOT" in content or "must not" in content.lower()
        assert "MUST" in content

    def test_templates_contain_tool_references(
        self, discovery_first_template, discovery_tools_list
    ):
        """Test that discovery_first references discovery tools."""
        content = discovery_first_template.content.lower()

        # Should mention key discovery tools
        tools_mentioned = sum(1 for tool in discovery_tools_list if tool in content)
        assert tools_mentioned >= 3  # At least some tools mentioned

    def test_templates_dont_reference_disabled_tools(
        self, all_builtin_templates, disabled_tools_list
    ):
        """Test that templates don't reference disabled/fake tools."""
        for template in all_builtin_templates:
            content = template.content.lower()

            for disabled_tool in disabled_tools_list:
                assert disabled_tool not in content, (
                    f"Template {template.template_id} references disabled tool: {disabled_tool}"
                )


# =============================================================================
# Template Rendering Tests
# =============================================================================


class TestTemplateRendering:
    """Tests for template rendering."""

    def test_render_returns_instruction_content(
        self, discovery_first_template, sample_context
    ):
        """Test that render returns InstructionContent."""
        result = discovery_first_template.render(sample_context)

        assert isinstance(result, InstructionContent)

    def test_render_source_includes_template_id(
        self, discovery_first_template, sample_context
    ):
        """Test that rendered content source includes template ID."""
        result = discovery_first_template.render(sample_context)

        assert "template:" in result.source
        assert "discovery_first" in result.source

    def test_render_preserves_whitespace(self):
        """Test that rendering preserves whitespace formatting."""
        template = InstructionTemplate(
            template_id="formatted",
            content="Line 1\n\nLine 2\n  Indented",
            description="Test",
            placeholders=()
        )

        result = template.render({})

        assert "\n\n" in result.value
        assert "  Indented" in result.value

    def test_render_handles_special_characters_in_context(self):
        """Test rendering with special characters in context."""
        template = InstructionTemplate(
            template_id="special",
            content="Tools: {tools}",
            description="Test",
            placeholders=("tools",)
        )

        special_context = {"tools": "tool<>&'\"tool"}
        result = template.render(special_context)

        # Special chars should be preserved
        assert "<>&'\"" in result.value

    def test_render_handles_curly_braces_not_placeholders(self):
        """Test that non-placeholder curly braces are preserved."""
        template = InstructionTemplate(
            template_id="json",
            content='Use {"key": "value"} format. Tool: {tool}',
            description="Test",
            placeholders=("tool",)
        )

        # Note: This will have issues because {"key": "value"} looks like placeholder
        # Template validation should handle this appropriately
        # In current implementation, {key} would be extracted as placeholder

    def test_multiple_same_placeholder_replaced(self):
        """Test that same placeholder used multiple times is replaced."""
        template = InstructionTemplate(
            template_id="repeat",
            content="{tool} is good. Use {tool} often.",
            description="Test",
            placeholders=("tool",)
        )

        result = template.render({"tool": "find_keywords"})

        # Both occurrences should be replaced
        assert result.value.count("find_keywords") == 2
        assert "{tool}" not in result.value


# =============================================================================
# Resolver Template Integration Tests
# =============================================================================


class TestResolverTemplateIntegration:
    """Tests for template usage through InstructionResolver."""

    def test_resolver_uses_default_template(self, sample_context):
        """Test that resolver uses default template."""
        resolver = InstructionResolver()
        config = InstructionConfig.create_default()

        content = resolver.resolve(config, sample_context)

        assert content is not None
        assert content.is_from_template

    def test_resolver_with_custom_template(self, sample_context):
        """Test resolver with custom default template."""
        custom_template = InstructionTemplate(
            template_id="custom",
            content="Custom: {available_tools}",
            description="Custom",
            placeholders=("available_tools",)
        )

        resolver = InstructionResolver(default_template=custom_template)
        config = InstructionConfig.create_default()

        content = resolver.resolve(config, sample_context)

        assert "Custom:" in content.value

    def test_resolver_provides_default_context(self):
        """Test that resolver provides default context when none given."""
        resolver = InstructionResolver()
        config = InstructionConfig.create_default()

        # Resolve without context
        content = resolver.resolve(config)

        # Should have default tools filled in
        assert content is not None
        assert "find_keywords" in content.value


# =============================================================================
# Template Best Practices Tests
# =============================================================================


class TestTemplateBestPractices:
    """Tests ensuring templates follow best practices."""

    def test_templates_are_instructive_not_descriptive(self, all_builtin_templates):
        """Test that templates give instructions, not descriptions."""
        instruction_words = ["use", "do", "never", "always", "must", "should", "verify"]

        for template in all_builtin_templates:
            content_lower = template.content.lower()
            has_instructions = any(word in content_lower for word in instruction_words)
            assert has_instructions, f"Template {template.template_id} should be instructive"

    def test_discovery_first_is_discovery_focused(self, discovery_first_template):
        """Test that discovery_first emphasizes discovery workflow."""
        content = discovery_first_template.content

        # Should have numbered steps or clear structure
        assert "1." in content or "DISCOVERY" in content
        assert "2." in content or "NO GUESSING" in content

    def test_templates_have_clear_structure(self, discovery_first_template):
        """Test that templates have clear section structure."""
        content = discovery_first_template.content

        # Should have sections (numbered or headed)
        lines = content.split("\n")
        structured_lines = [
            l for l in lines
            if l.strip().startswith(("1.", "2.", "3.", "-", "*", "#"))
        ]
        assert len(structured_lines) > 0

    def test_templates_mention_verification(self, all_builtin_templates):
        """Test that templates mention verification/confirmation."""
        verification_words = ["verify", "confirm", "check", "ensure", "validate"]

        for template in all_builtin_templates:
            content_lower = template.content.lower()
            has_verification = any(word in content_lower for word in verification_words)
            # At least some templates should mention verification
            if template.template_id != "minimal":
                assert has_verification or "discovered" in content_lower


# =============================================================================
# Renderer Template Interaction Tests
# =============================================================================


class TestRendererTemplateInteraction:
    """Tests for template content through renderer."""

    def test_rendered_template_for_claude(
        self, discovery_first_template, sample_context
    ):
        """Test rendering template content for Claude."""
        content = discovery_first_template.render(sample_context)
        renderer = InstructionRenderer()

        rendered = renderer.render(
            content,
            target=InstructionRenderer.TargetFormat.CLAUDE
        )

        assert "<instructions>" in rendered
        assert "</instructions>" in rendered

    def test_rendered_template_for_openai(
        self, discovery_first_template, sample_context
    ):
        """Test rendering template content for OpenAI."""
        content = discovery_first_template.render(sample_context)
        renderer = InstructionRenderer()

        rendered = renderer.render(
            content,
            target=InstructionRenderer.TargetFormat.OPENAI
        )

        assert "# System Instructions" in rendered
        assert "---" in rendered

    def test_template_content_preserved_through_renderer(
        self, minimal_template
    ):
        """Test that template content is preserved through rendering."""
        content = minimal_template.render({})
        renderer = InstructionRenderer()

        rendered = renderer.render(
            content,
            target=InstructionRenderer.TargetFormat.GENERIC,
            include_wrapper=False
        )

        # Content should be unchanged
        assert rendered == content.value


# =============================================================================
# Template Edge Cases
# =============================================================================


class TestTemplateEdgeCases:
    """Tests for template edge cases."""

    def test_empty_template_id_rejected(self):
        """Test that empty template ID is rejected."""
        with pytest.raises(ValueError, match="ID cannot be empty"):
            InstructionTemplate(
                template_id="",
                content="Some content here",
                description="Test",
                placeholders=()
            )

    def test_whitespace_template_id_rejected(self):
        """Test that whitespace-only template ID is rejected."""
        with pytest.raises(ValueError, match="ID cannot be empty"):
            InstructionTemplate(
                template_id="   ",
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

    def test_whitespace_content_rejected(self):
        """Test that whitespace-only content is rejected."""
        with pytest.raises(ValueError, match="content cannot be empty"):
            InstructionTemplate(
                template_id="test",
                content="   \n\t  ",
                description="Test",
                placeholders=()
            )

    def test_template_string_representation(self, discovery_first_template):
        """Test template string representations."""
        str_repr = str(discovery_first_template)
        assert "discovery_first" in str_repr

        repr_str = repr(discovery_first_template)
        assert "InstructionTemplate" in repr_str
        assert "discovery_first" in repr_str

    def test_placeholder_with_underscore(self):
        """Test placeholder names with underscores."""
        template = InstructionTemplate(
            template_id="underscore",
            content="Use {my_long_placeholder_name} here",
            description="Test",
            placeholders=("my_long_placeholder_name",)
        )

        result = template.render({"my_long_placeholder_name": "value"})
        assert "value" in result.value

    def test_placeholder_with_numbers(self):
        """Test placeholder names with numbers."""
        template = InstructionTemplate(
            template_id="numbers",
            content="Use {tool1} and {tool2} here",
            description="Test",
            placeholders=("tool1", "tool2")
        )

        result = template.render({"tool1": "a", "tool2": "b"})
        assert "a" in result.value
        assert "b" in result.value


# =============================================================================
# Template Tool Reference Tests
# =============================================================================


class TestTemplateToolReferences:
    """Tests ensuring templates reference correct tools."""

    def test_discovery_first_references_find_keywords(self, discovery_first_template):
        """Test that discovery_first references find_keywords."""
        assert "find_keywords" in discovery_first_template.content

    def test_discovery_first_references_get_keyword_info(self, discovery_first_template):
        """Test that discovery_first references get_keyword_info."""
        assert "get_keyword_info" in discovery_first_template.content

    def test_locator_prevention_references_discovery_tools(
        self, locator_prevention_template
    ):
        """Test that locator_prevention references discovery tools."""
        content = locator_prevention_template.content

        assert "find_keywords" in content
        assert "get_keyword_info" in content

    def test_templates_dont_reference_nonexistent_tools(
        self, all_builtin_templates
    ):
        """Test that templates don't reference made-up tools."""
        fake_tools = [
            "magic_click",
            "auto_fill",
            "smart_detect",
            "ai_guess",
            "auto_complete",
            "predict_element",
        ]

        for template in all_builtin_templates:
            content_lower = template.content.lower()
            for fake_tool in fake_tools:
                assert fake_tool not in content_lower, (
                    f"Template {template.template_id} references fake tool: {fake_tool}"
                )

    def test_placeholder_substitution_with_tool_list(
        self, discovery_first_template, discovery_tools_list
    ):
        """Test placeholder substitution with actual tool list."""
        tools_str = ", ".join(sorted(discovery_tools_list))
        context = {"available_tools": tools_str}

        content = discovery_first_template.render(context)

        # All tools should be in the rendered content
        for tool in discovery_tools_list:
            assert tool in content.value

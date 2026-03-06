"""Unit tests for ADR-016 Ultra-Slim Tool Profile: SchemaMode, SlimToolSchema,
slim_exec preset, ToolDescriptor slim fields, SchemaSimplified event.

Run with: uv run pytest tests/unit/test_tool_profile_slim.py -v
"""

__test__ = True

import json

import pytest

from robotmcp.domains.tool_profile.aggregates import ProfilePresets, ToolProfile
from robotmcp.domains.tool_profile.entities import ToolDescriptor
from robotmcp.domains.tool_profile.events import SchemaSimplified
from robotmcp.domains.tool_profile.value_objects import (
    AutoProfileSelection,
    ModelTier,
    SchemaMode,
    SlimToolSchema,
    TokenBudget,
    ToolDescriptionMode,
    ToolTag,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def full_schema():
    """A representative full JSON schema for testing simplification."""
    return {
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "The Robot Framework keyword to execute.",
            },
            "args": {
                "type": "array",
                "description": "Arguments for the keyword.",
                "items": {"type": "string"},
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds.",
            },
            "on_error": {
                "description": "What to do on error.",
                "anyOf": [
                    {"type": "string"},
                    {"type": "object", "properties": {"retry": {"type": "integer"}}},
                ],
            },
        },
        "required": ["keyword"],
    }


@pytest.fixture
def schema_with_oneof():
    """Schema with oneOf union."""
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "The action.",
            },
            "target": {
                "description": "The target element.",
                "oneOf": [
                    {"type": "string"},
                    {"type": "integer"},
                ],
            },
        },
        "required": ["action", "target"],
    }


@pytest.fixture
def minimal_schema():
    """Schema with only required properties."""
    return {
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "The keyword to run.",
            },
        },
        "required": ["keyword"],
    }


@pytest.fixture
def sample_descriptor():
    """A ToolDescriptor with all fields including slim variants."""
    return ToolDescriptor(
        tool_name="execute_step",
        tags=frozenset({ToolTag.CORE, ToolTag.EXECUTION}),
        description_full="Execute a single Robot Framework keyword with arguments.",
        description_compact="Run keyword",
        description_minimal="Run",
        schema_full={
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "Keyword name"},
                "args": {"type": "array", "description": "Arguments"},
            },
            "required": ["keyword"],
        },
        schema_compact={
            "type": "object",
            "properties": {
                "keyword": {"type": "string"},
                "args": {"type": "array"},
            },
            "required": ["keyword"],
        },
        token_estimate_full=600,
        token_estimate_compact=200,
        token_estimate_minimal=100,
        description_slim="Run KW",
        schema_slim={
            "type": "object",
            "properties": {"keyword": {"type": "string"}},
            "required": ["keyword"],
        },
        token_estimate_slim=40,
    )


# =============================================================================
# SchemaMode Enum
# =============================================================================


class TestSchemaModeEnum:
    """Test SchemaMode enum values and properties."""

    def test_full_value(self):
        assert SchemaMode.FULL.value == "full"

    def test_standard_value(self):
        assert SchemaMode.STANDARD.value == "standard"

    def test_minimal_value(self):
        assert SchemaMode.MINIMAL.value == "minimal"

    def test_has_exactly_3_members(self):
        assert len(SchemaMode) == 3

    def test_all_values_unique(self):
        values = [m.value for m in SchemaMode]
        assert len(values) == len(set(values))


class TestSchemaModeStripOptionalFields:
    """Test strips_optional_fields property."""

    def test_full_does_not_strip(self):
        assert SchemaMode.FULL.strips_optional_fields is False

    def test_standard_does_not_strip(self):
        assert SchemaMode.STANDARD.strips_optional_fields is False

    def test_minimal_strips(self):
        assert SchemaMode.MINIMAL.strips_optional_fields is True


class TestSchemaModeStripDescriptions:
    """Test strips_property_descriptions property."""

    def test_full_does_not_strip(self):
        assert SchemaMode.FULL.strips_property_descriptions is False

    def test_standard_strips(self):
        assert SchemaMode.STANDARD.strips_property_descriptions is True

    def test_minimal_strips(self):
        assert SchemaMode.MINIMAL.strips_property_descriptions is True


class TestSchemaModeFlattenUnions:
    """Test flattens_unions property."""

    def test_full_does_not_flatten(self):
        assert SchemaMode.FULL.flattens_unions is False

    def test_standard_does_not_flatten(self):
        assert SchemaMode.STANDARD.flattens_unions is False

    def test_minimal_flattens(self):
        assert SchemaMode.MINIMAL.flattens_unions is True


# =============================================================================
# SlimToolSchema - from_full_schema MINIMAL mode
# =============================================================================


class TestSlimToolSchemaMinimal:
    """Test SlimToolSchema.from_full_schema with MINIMAL mode."""

    def test_keeps_only_required_properties(self, full_schema):
        slim = SlimToolSchema.from_full_schema("test", full_schema, SchemaMode.MINIMAL)
        assert "keyword" in slim.properties
        assert "args" not in slim.properties
        assert "timeout" not in slim.properties
        assert "on_error" not in slim.properties

    def test_strips_property_descriptions(self, full_schema):
        slim = SlimToolSchema.from_full_schema("test", full_schema, SchemaMode.MINIMAL)
        for prop_def in slim.properties.values():
            if isinstance(prop_def, dict):
                assert "description" not in prop_def

    def test_flattens_anyof_union(self):
        schema = {
            "type": "object",
            "properties": {
                "target": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "integer"},
                    ],
                },
            },
            "required": ["target"],
        }
        slim = SlimToolSchema.from_full_schema("test", schema, SchemaMode.MINIMAL)
        assert slim.properties["target"] == {"type": "string"}

    def test_flattens_oneof_union(self, schema_with_oneof):
        slim = SlimToolSchema.from_full_schema("test", schema_with_oneof, SchemaMode.MINIMAL)
        # target had oneOf, should be flattened to first variant
        assert slim.properties["target"] == {"type": "string"}

    def test_flattens_empty_anyof_to_string(self):
        schema = {
            "type": "object",
            "properties": {
                "x": {"anyOf": []},
            },
            "required": ["x"],
        }
        slim = SlimToolSchema.from_full_schema("test", schema, SchemaMode.MINIMAL)
        assert slim.properties["x"] == {"type": "string"}

    def test_flattens_empty_oneof_to_string(self):
        schema = {
            "type": "object",
            "properties": {
                "x": {"oneOf": []},
            },
            "required": ["x"],
        }
        slim = SlimToolSchema.from_full_schema("test", schema, SchemaMode.MINIMAL)
        assert slim.properties["x"] == {"type": "string"}

    def test_required_tuple_matches(self, full_schema):
        slim = SlimToolSchema.from_full_schema("test", full_schema, SchemaMode.MINIMAL)
        assert slim.required == ("keyword",)

    def test_tool_name_preserved(self, full_schema):
        slim = SlimToolSchema.from_full_schema("my_tool", full_schema, SchemaMode.MINIMAL)
        assert slim.tool_name == "my_tool"

    def test_schema_mode_preserved(self, full_schema):
        slim = SlimToolSchema.from_full_schema("test", full_schema, SchemaMode.MINIMAL)
        assert slim.schema_mode == SchemaMode.MINIMAL

    def test_token_estimate_is_positive(self, full_schema):
        slim = SlimToolSchema.from_full_schema("test", full_schema, SchemaMode.MINIMAL)
        assert slim.token_estimate > 0

    def test_token_estimate_reasonable(self, full_schema):
        slim = SlimToolSchema.from_full_schema("test", full_schema, SchemaMode.MINIMAL)
        # For a minimal schema with one property, should be small
        schema_json = json.dumps(slim.to_schema())
        expected = len(schema_json) // 4
        assert slim.token_estimate == expected


# =============================================================================
# SlimToolSchema - from_full_schema STANDARD mode
# =============================================================================


class TestSlimToolSchemaStandard:
    """Test SlimToolSchema.from_full_schema with STANDARD mode."""

    def test_keeps_all_properties(self, full_schema):
        slim = SlimToolSchema.from_full_schema("test", full_schema, SchemaMode.STANDARD)
        assert "keyword" in slim.properties
        assert "args" in slim.properties
        assert "timeout" in slim.properties
        assert "on_error" in slim.properties

    def test_strips_descriptions(self, full_schema):
        slim = SlimToolSchema.from_full_schema("test", full_schema, SchemaMode.STANDARD)
        for prop_def in slim.properties.values():
            if isinstance(prop_def, dict):
                assert "description" not in prop_def

    def test_does_not_flatten_unions(self, full_schema):
        slim = SlimToolSchema.from_full_schema("test", full_schema, SchemaMode.STANDARD)
        # on_error had anyOf, should still have it (no flattening in STANDARD)
        assert "anyOf" in slim.properties["on_error"]

    def test_required_includes_all_original(self, full_schema):
        slim = SlimToolSchema.from_full_schema("test", full_schema, SchemaMode.STANDARD)
        assert "keyword" in slim.required

    def test_token_estimate_larger_than_minimal(self, full_schema):
        minimal = SlimToolSchema.from_full_schema("test", full_schema, SchemaMode.MINIMAL)
        standard = SlimToolSchema.from_full_schema("test", full_schema, SchemaMode.STANDARD)
        assert standard.token_estimate > minimal.token_estimate


# =============================================================================
# SlimToolSchema - from_full_schema FULL mode
# =============================================================================


class TestSlimToolSchemaFull:
    """Test SlimToolSchema.from_full_schema with FULL mode."""

    def test_keeps_all_properties(self, full_schema):
        slim = SlimToolSchema.from_full_schema("test", full_schema, SchemaMode.FULL)
        assert len(slim.properties) == 4

    def test_keeps_descriptions(self, full_schema):
        slim = SlimToolSchema.from_full_schema("test", full_schema, SchemaMode.FULL)
        assert "description" in slim.properties["keyword"]

    def test_keeps_unions(self, full_schema):
        slim = SlimToolSchema.from_full_schema("test", full_schema, SchemaMode.FULL)
        assert "anyOf" in slim.properties["on_error"]

    def test_is_equivalent_to_original(self, full_schema):
        slim = SlimToolSchema.from_full_schema("test", full_schema, SchemaMode.FULL)
        output = slim.to_schema()
        assert output["properties"] == full_schema["properties"]
        assert output["required"] == full_schema["required"]


# =============================================================================
# SlimToolSchema - edge cases
# =============================================================================


class TestSlimToolSchemaEdgeCases:
    """Test edge cases for SlimToolSchema."""

    def test_empty_properties_raises(self):
        with pytest.raises(ValueError, match="at least one property"):
            SlimToolSchema(
                tool_name="test",
                properties={},
                required=(),
                schema_mode=SchemaMode.MINIMAL,
                token_estimate=0,
            )

    def test_no_required_fields_minimal_preserves_first_property(self):
        """If MINIMAL mode would remove all props (none required), keep the first."""
        schema = {
            "type": "object",
            "properties": {
                "alpha": {"type": "string", "description": "First prop"},
                "beta": {"type": "integer", "description": "Second prop"},
            },
            "required": [],
        }
        slim = SlimToolSchema.from_full_schema("test", schema, SchemaMode.MINIMAL)
        assert len(slim.properties) == 1
        assert "alpha" in slim.properties

    def test_schema_with_no_properties_key(self):
        """Schema missing 'properties' key entirely."""
        schema = {"type": "object"}
        # This should fallback: no properties -> empty -> error path
        # Since there are no properties at all, it will raise
        with pytest.raises(ValueError, match="at least one property"):
            SlimToolSchema.from_full_schema("test", schema, SchemaMode.MINIMAL)

    def test_to_schema_round_trip(self, full_schema):
        slim = SlimToolSchema.from_full_schema("test", full_schema, SchemaMode.MINIMAL)
        output = slim.to_schema()
        assert output["type"] == "object"
        assert "properties" in output
        assert "required" in output
        assert isinstance(output["required"], list)

    def test_frozen_dataclass(self, full_schema):
        slim = SlimToolSchema.from_full_schema("test", full_schema, SchemaMode.MINIMAL)
        with pytest.raises(AttributeError):
            slim.tool_name = "changed"

    def test_all_required_properties_in_minimal(self, schema_with_oneof):
        """When all properties are required, MINIMAL keeps all of them."""
        slim = SlimToolSchema.from_full_schema("test", schema_with_oneof, SchemaMode.MINIMAL)
        assert "action" in slim.properties
        assert "target" in slim.properties

    def test_non_dict_property_value_handled(self):
        """Property values that aren't dicts (unusual but valid JSON Schema) are preserved."""
        schema = {
            "type": "object",
            "properties": {
                "x": True,  # boolean schema
            },
            "required": ["x"],
        }
        slim = SlimToolSchema.from_full_schema("test", schema, SchemaMode.MINIMAL)
        assert slim.properties["x"] is True

    def test_multiple_required_fields(self):
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "integer"},
                "c": {"type": "boolean"},
            },
            "required": ["a", "b"],
        }
        slim = SlimToolSchema.from_full_schema("test", schema, SchemaMode.MINIMAL)
        assert set(slim.required) == {"a", "b"}
        assert "c" not in slim.properties


# =============================================================================
# ToolDescriptor - slim fields and schema_mode methods
# =============================================================================


class TestToolDescriptorSlimFields:
    """Test ToolDescriptor's ADR-016 slim fields."""

    def test_slim_fields_default_to_none(self):
        desc = ToolDescriptor(
            tool_name="test",
            tags=frozenset({ToolTag.CORE}),
            description_full="Full",
            description_compact="Compact",
            description_minimal="Min",
            schema_full={"type": "object"},
        )
        assert desc.description_slim is None
        assert desc.schema_slim is None
        assert desc.token_estimate_slim is None

    def test_slim_fields_can_be_set(self, sample_descriptor):
        assert sample_descriptor.description_slim == "Run KW"
        assert sample_descriptor.schema_slim is not None
        assert sample_descriptor.token_estimate_slim == 40


class TestToolDescriptorSchemaModeMethods:
    """Test description_for_schema_mode and schema_for_schema_mode."""

    def test_description_for_schema_mode_full(self, sample_descriptor):
        result = sample_descriptor.description_for_schema_mode(SchemaMode.FULL)
        assert result == sample_descriptor.description_full

    def test_description_for_schema_mode_standard(self, sample_descriptor):
        result = sample_descriptor.description_for_schema_mode(SchemaMode.STANDARD)
        assert result == sample_descriptor.description_compact

    def test_description_for_schema_mode_minimal_with_slim(self, sample_descriptor):
        result = sample_descriptor.description_for_schema_mode(SchemaMode.MINIMAL)
        assert result == "Run KW"

    def test_description_for_schema_mode_minimal_fallback(self):
        """When description_slim is None, falls back to description_minimal."""
        desc = ToolDescriptor(
            tool_name="test",
            tags=frozenset({ToolTag.CORE}),
            description_full="Full",
            description_compact="Compact",
            description_minimal="Min",
            schema_full={"type": "object"},
        )
        result = desc.description_for_schema_mode(SchemaMode.MINIMAL)
        assert result == "Min"

    def test_schema_for_schema_mode_full(self, sample_descriptor):
        result = sample_descriptor.schema_for_schema_mode(SchemaMode.FULL)
        assert result == sample_descriptor.schema_full

    def test_schema_for_schema_mode_standard(self, sample_descriptor):
        result = sample_descriptor.schema_for_schema_mode(SchemaMode.STANDARD)
        assert result == sample_descriptor.schema_compact

    def test_schema_for_schema_mode_standard_fallback(self):
        """When schema_compact is None, falls back to schema_full."""
        desc = ToolDescriptor(
            tool_name="test",
            tags=frozenset({ToolTag.CORE}),
            description_full="Full",
            description_compact="Compact",
            description_minimal="Min",
            schema_full={"type": "object", "properties": {"a": {"type": "string"}}},
        )
        result = desc.schema_for_schema_mode(SchemaMode.STANDARD)
        assert result == desc.schema_full

    def test_schema_for_schema_mode_minimal_with_slim(self, sample_descriptor):
        result = sample_descriptor.schema_for_schema_mode(SchemaMode.MINIMAL)
        assert result == sample_descriptor.schema_slim

    def test_schema_for_schema_mode_minimal_fallback_to_compact(self):
        """When schema_slim is None, falls back to schema_compact."""
        desc = ToolDescriptor(
            tool_name="test",
            tags=frozenset({ToolTag.CORE}),
            description_full="Full",
            description_compact="Compact",
            description_minimal="Min",
            schema_full={"type": "object"},
            schema_compact={"type": "object", "properties": {"a": {"type": "string"}}},
        )
        result = desc.schema_for_schema_mode(SchemaMode.MINIMAL)
        assert result == desc.schema_compact

    def test_schema_for_schema_mode_minimal_fallback_to_full(self):
        """When both schema_slim and schema_compact are None, falls back to schema_full."""
        desc = ToolDescriptor(
            tool_name="test",
            tags=frozenset({ToolTag.CORE}),
            description_full="Full",
            description_compact="Compact",
            description_minimal="Min",
            schema_full={"type": "object"},
        )
        result = desc.schema_for_schema_mode(SchemaMode.MINIMAL)
        assert result == desc.schema_full


# =============================================================================
# slim_exec Profile Preset
# =============================================================================


class TestSlimExecPreset:
    """Test the slim_exec profile preset (ADR-016)."""

    def test_has_4_tools(self):
        p = ProfilePresets.slim_exec()
        assert p.tool_count == 4

    def test_tool_names(self):
        p = ProfilePresets.slim_exec()
        expected = frozenset({
            "manage_session", "intent_action",
            "execute_batch", "get_session_state",
        })
        assert p.tool_names == expected

    def test_name(self):
        p = ProfilePresets.slim_exec()
        assert p.name == "slim_exec"

    def test_description_mode_minimal(self):
        p = ProfilePresets.slim_exec()
        assert p.description_mode == ToolDescriptionMode.MINIMAL

    def test_schema_mode_minimal(self):
        p = ProfilePresets.slim_exec()
        assert p.schema_mode == SchemaMode.MINIMAL

    def test_model_tier_small_7b(self):
        p = ProfilePresets.slim_exec()
        assert p.model_tier == ModelTier.SMALL_7B

    def test_has_token_budget(self):
        p = ProfilePresets.slim_exec()
        assert p.token_budget is not None
        assert p.token_budget.context_window == 8192

    def test_has_core_and_execution_tags(self):
        p = ProfilePresets.slim_exec()
        assert ToolTag.CORE in p.tags
        assert ToolTag.EXECUTION in p.tags

    def test_is_frozen(self):
        p = ProfilePresets.slim_exec()
        with pytest.raises(AttributeError):
            p.name = "changed"


# =============================================================================
# desktop_exec Profile Preset
# =============================================================================


class TestDesktopExecPreset:
    """Test the desktop_exec profile preset."""

    def test_has_5_tools(self):
        p = ProfilePresets.desktop_exec()
        assert p.tool_count == 5

    def test_tool_names(self):
        p = ProfilePresets.desktop_exec()
        expected = frozenset({
            "manage_session", "execute_step",
            "get_session_state", "find_keywords",
            "intent_action",
        })
        assert p.tool_names == expected

    def test_name(self):
        p = ProfilePresets.desktop_exec()
        assert p.name == "desktop_exec"

    def test_description_mode(self):
        p = ProfilePresets.desktop_exec()
        assert p.description_mode == ToolDescriptionMode.COMPACT

    def test_model_tier(self):
        p = ProfilePresets.desktop_exec()
        assert p.model_tier == ModelTier.SMALL_CONTEXT

    def test_schema_mode_defaults_to_full(self):
        p = ProfilePresets.desktop_exec()
        assert p.schema_mode == SchemaMode.FULL


# =============================================================================
# ToolProfile schema_mode field
# =============================================================================


class TestToolProfileSchemaMode:
    """Test the schema_mode field on ToolProfile aggregate."""

    def test_default_schema_mode_is_full(self):
        p = ToolProfile(
            name="test",
            tool_names=frozenset({"manage_session"}),
            description_mode=ToolDescriptionMode.FULL,
            model_tier=ModelTier.LARGE_CONTEXT,
        )
        assert p.schema_mode == SchemaMode.FULL

    def test_explicit_schema_mode(self):
        p = ToolProfile(
            name="test",
            tool_names=frozenset({"manage_session"}),
            description_mode=ToolDescriptionMode.MINIMAL,
            schema_mode=SchemaMode.MINIMAL,
            model_tier=ModelTier.SMALL_7B,
            token_budget=TokenBudget.for_context_window(8192),
        )
        assert p.schema_mode == SchemaMode.MINIMAL

    def test_with_additional_tool_preserves_schema_mode(self):
        p = ToolProfile(
            name="test",
            tool_names=frozenset({"manage_session"}),
            description_mode=ToolDescriptionMode.MINIMAL,
            schema_mode=SchemaMode.MINIMAL,
            model_tier=ModelTier.SMALL_7B,
            token_budget=TokenBudget.for_context_window(8192),
        )
        p2 = p.with_additional_tool("execute_step")
        assert p2.schema_mode == SchemaMode.MINIMAL

    def test_without_tool_preserves_schema_mode(self):
        p = ToolProfile(
            name="test",
            tool_names=frozenset({"manage_session", "execute_step"}),
            description_mode=ToolDescriptionMode.MINIMAL,
            schema_mode=SchemaMode.STANDARD,
            model_tier=ModelTier.SMALL_7B,
            token_budget=TokenBudget.for_context_window(8192),
        )
        p2 = p.without_tool("execute_step")
        assert p2.schema_mode == SchemaMode.STANDARD

    def test_small_7b_without_budget_raises(self):
        with pytest.raises(ValueError, match="SMALL_7B"):
            ToolProfile(
                name="no_budget",
                tool_names=frozenset({"manage_session"}),
                description_mode=ToolDescriptionMode.MINIMAL,
                model_tier=ModelTier.SMALL_7B,
                token_budget=None,
            )


# =============================================================================
# Backward Compatibility of Existing Profiles
# =============================================================================


class TestBackwardCompatibility:
    """Ensure existing profiles still work correctly with new schema_mode field."""

    def test_browser_exec_schema_mode_defaults_to_full(self):
        p = ProfilePresets.browser_exec()
        assert p.schema_mode == SchemaMode.FULL

    def test_api_exec_schema_mode_defaults_to_full(self):
        p = ProfilePresets.api_exec()
        assert p.schema_mode == SchemaMode.FULL

    def test_discovery_schema_mode_defaults_to_full(self):
        p = ProfilePresets.discovery()
        assert p.schema_mode == SchemaMode.FULL

    def test_minimal_exec_schema_mode_defaults_to_full(self):
        p = ProfilePresets.minimal_exec()
        assert p.schema_mode == SchemaMode.FULL

    def test_full_schema_mode_defaults_to_full(self):
        p = ProfilePresets.full()
        assert p.schema_mode == SchemaMode.FULL

    def test_existing_profiles_unchanged_tool_count(self):
        assert ProfilePresets.browser_exec().tool_count == 6
        assert ProfilePresets.api_exec().tool_count == 5
        assert ProfilePresets.discovery().tool_count == 6
        assert ProfilePresets.minimal_exec().tool_count == 3

    def test_existing_profiles_unchanged_description_mode(self):
        assert ProfilePresets.browser_exec().description_mode == ToolDescriptionMode.COMPACT
        assert ProfilePresets.api_exec().description_mode == ToolDescriptionMode.COMPACT
        assert ProfilePresets.minimal_exec().description_mode == ToolDescriptionMode.MINIMAL
        assert ProfilePresets.full().description_mode == ToolDescriptionMode.FULL


# =============================================================================
# SchemaSimplified Event
# =============================================================================


class TestSchemaSimplifiedEvent:
    """Test SchemaSimplified domain event."""

    def test_construction(self):
        event = SchemaSimplified(
            tool_name="execute_step",
            profile_name="slim_exec",
            schema_mode="minimal",
            properties_removed=frozenset({"args", "timeout"}),
            token_reduction=150,
        )
        assert event.tool_name == "execute_step"
        assert event.profile_name == "slim_exec"

    def test_to_dict_has_event_type(self):
        event = SchemaSimplified(
            tool_name="execute_step",
            profile_name="slim_exec",
            schema_mode="minimal",
            properties_removed=frozenset({"args"}),
            token_reduction=100,
        )
        d = event.to_dict()
        assert d["event_type"] == "SchemaSimplified"
        assert d["tool"] == "execute_step"
        assert d["profile"] == "slim_exec"
        assert d["schema_mode"] == "minimal"
        assert d["removed"] == ["args"]
        assert d["saved_tokens"] == 100
        assert "ts" in d

    def test_to_dict_removed_sorted(self):
        event = SchemaSimplified(
            tool_name="test",
            profile_name="slim_exec",
            schema_mode="minimal",
            properties_removed=frozenset({"zebra", "alpha", "middle"}),
            token_reduction=50,
        )
        d = event.to_dict()
        assert d["removed"] == ["alpha", "middle", "zebra"]

    def test_empty_properties_removed(self):
        event = SchemaSimplified(
            tool_name="test",
            profile_name="slim_exec",
            schema_mode="standard",
            properties_removed=frozenset(),
            token_reduction=0,
        )
        d = event.to_dict()
        assert d["removed"] == []
        assert d["saved_tokens"] == 0

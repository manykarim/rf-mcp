"""Tests for ADR-016 slim profile extensions: SchemaMode, SlimToolSchema, AutoProfileSelection."""

__test__ = True

import json

import pytest

from robotmcp.domains.shared.kernel import ModelTier
from robotmcp.domains.tool_profile.value_objects import (
    AutoProfileSelection,
    SchemaMode,
    SlimToolSchema,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

SAMPLE_SCHEMA = {
    "properties": {
        "session_id": {"type": "string", "description": "The session ID"},
        "keyword": {"type": "string", "description": "RF keyword to execute"},
        "args": {
            "anyOf": [{"type": "array"}, {"type": "null"}],
            "description": "Arguments",
        },
    },
    "required": ["session_id", "keyword"],
}


@pytest.fixture
def sample_schema():
    """Return a deep-copy of SAMPLE_SCHEMA so tests can mutate freely."""
    return json.loads(json.dumps(SAMPLE_SCHEMA))


# ===================================================================
# SchemaMode enum
# ===================================================================


class TestSchemaMode:
    """Tests for the SchemaMode enum."""

    def test_full_value(self):
        assert SchemaMode.FULL.value == "full"

    def test_standard_value(self):
        assert SchemaMode.STANDARD.value == "standard"

    def test_minimal_value(self):
        assert SchemaMode.MINIMAL.value == "minimal"

    def test_exactly_three_members(self):
        assert len(SchemaMode) == 3

    # -- strips_optional_fields --

    def test_strips_optional_fields_full(self):
        assert SchemaMode.FULL.strips_optional_fields is False

    def test_strips_optional_fields_standard(self):
        assert SchemaMode.STANDARD.strips_optional_fields is False

    def test_strips_optional_fields_minimal(self):
        assert SchemaMode.MINIMAL.strips_optional_fields is True

    # -- strips_property_descriptions --

    def test_strips_property_descriptions_full(self):
        assert SchemaMode.FULL.strips_property_descriptions is False

    def test_strips_property_descriptions_standard(self):
        assert SchemaMode.STANDARD.strips_property_descriptions is True

    def test_strips_property_descriptions_minimal(self):
        assert SchemaMode.MINIMAL.strips_property_descriptions is True

    # -- flattens_unions --

    def test_flattens_unions_full(self):
        assert SchemaMode.FULL.flattens_unions is False

    def test_flattens_unions_standard(self):
        assert SchemaMode.STANDARD.flattens_unions is False

    def test_flattens_unions_minimal(self):
        assert SchemaMode.MINIMAL.flattens_unions is True


# ===================================================================
# SlimToolSchema
# ===================================================================


class TestSlimToolSchema:
    """Tests for SlimToolSchema construction and transformation."""

    # -- FULL mode --

    def test_from_full_schema_full_mode_preserves_all_properties(self, sample_schema):
        slim = SlimToolSchema.from_full_schema("execute_step", sample_schema, SchemaMode.FULL)
        assert set(slim.properties.keys()) == {"session_id", "keyword", "args"}

    def test_from_full_schema_full_mode_preserves_descriptions(self, sample_schema):
        slim = SlimToolSchema.from_full_schema("execute_step", sample_schema, SchemaMode.FULL)
        assert slim.properties["session_id"]["description"] == "The session ID"
        assert slim.properties["keyword"]["description"] == "RF keyword to execute"
        assert slim.properties["args"]["description"] == "Arguments"

    def test_from_full_schema_full_mode_preserves_unions(self, sample_schema):
        slim = SlimToolSchema.from_full_schema("execute_step", sample_schema, SchemaMode.FULL)
        assert "anyOf" in slim.properties["args"]

    # -- STANDARD mode --

    def test_from_full_schema_standard_mode_keeps_optional_props(self, sample_schema):
        slim = SlimToolSchema.from_full_schema("execute_step", sample_schema, SchemaMode.STANDARD)
        assert "args" in slim.properties

    def test_from_full_schema_standard_mode_strips_descriptions(self, sample_schema):
        slim = SlimToolSchema.from_full_schema("execute_step", sample_schema, SchemaMode.STANDARD)
        for prop in slim.properties.values():
            assert "description" not in prop

    def test_from_full_schema_standard_mode_preserves_unions(self, sample_schema):
        slim = SlimToolSchema.from_full_schema("execute_step", sample_schema, SchemaMode.STANDARD)
        assert "anyOf" in slim.properties["args"]

    # -- MINIMAL mode --

    def test_from_full_schema_minimal_mode_strips_optional_props(self, sample_schema):
        slim = SlimToolSchema.from_full_schema("execute_step", sample_schema, SchemaMode.MINIMAL)
        assert "args" not in slim.properties
        assert set(slim.properties.keys()) == {"session_id", "keyword"}

    def test_from_full_schema_minimal_mode_strips_descriptions(self, sample_schema):
        slim = SlimToolSchema.from_full_schema("execute_step", sample_schema, SchemaMode.MINIMAL)
        for prop in slim.properties.values():
            assert "description" not in prop

    def test_from_full_schema_minimal_mode_flattens_anyof(self):
        schema = {
            "properties": {
                "value": {
                    "anyOf": [{"type": "string"}, {"type": "integer"}],
                    "description": "A value",
                },
            },
            "required": ["value"],
        }
        slim = SlimToolSchema.from_full_schema("tool", schema, SchemaMode.MINIMAL)
        # anyOf should be flattened to first variant
        assert slim.properties["value"] == {"type": "string"}

    def test_from_full_schema_minimal_mode_flattens_oneof(self):
        schema = {
            "properties": {
                "value": {
                    "oneOf": [{"type": "integer"}, {"type": "null"}],
                    "description": "A value",
                },
            },
            "required": ["value"],
        }
        slim = SlimToolSchema.from_full_schema("tool", schema, SchemaMode.MINIMAL)
        assert slim.properties["value"] == {"type": "integer"}

    # -- Safety net: empty properties after strip --

    def test_empty_properties_after_strip_keeps_first(self):
        schema = {
            "properties": {
                "optional_a": {"type": "string", "description": "A"},
                "optional_b": {"type": "integer", "description": "B"},
            },
            "required": [],
        }
        slim = SlimToolSchema.from_full_schema("tool", schema, SchemaMode.MINIMAL)
        # All props were optional and stripped, safety net keeps first
        assert len(slim.properties) == 1
        assert "optional_a" in slim.properties

    # -- to_schema --

    def test_to_schema_returns_valid_dict(self, sample_schema):
        slim = SlimToolSchema.from_full_schema("execute_step", sample_schema, SchemaMode.FULL)
        result = slim.to_schema()
        assert result["type"] == "object"
        assert "properties" in result
        assert "required" in result
        assert isinstance(result["required"], list)

    def test_to_schema_required_matches_input(self, sample_schema):
        slim = SlimToolSchema.from_full_schema("execute_step", sample_schema, SchemaMode.FULL)
        result = slim.to_schema()
        assert set(result["required"]) == {"session_id", "keyword"}

    # -- Validation: empty properties --

    def test_empty_properties_raises_value_error(self):
        with pytest.raises(ValueError, match="at least one property"):
            SlimToolSchema(
                tool_name="tool",
                properties={},
                required=(),
                schema_mode=SchemaMode.FULL,
                token_estimate=0,
            )

    # -- token_estimate --

    def test_token_estimate_approximately_correct(self, sample_schema):
        slim = SlimToolSchema.from_full_schema("execute_step", sample_schema, SchemaMode.FULL)
        serialized = json.dumps({
            "type": "object",
            "properties": slim.properties,
            "required": list(slim.required),
        })
        expected = len(serialized) // 4
        assert slim.token_estimate == expected

    # -- schema_mode stored correctly --

    def test_schema_mode_stored(self, sample_schema):
        for mode in SchemaMode:
            slim = SlimToolSchema.from_full_schema("tool", sample_schema, mode)
            assert slim.schema_mode == mode

    # -- tool_name stored correctly --

    def test_tool_name_stored(self, sample_schema):
        slim = SlimToolSchema.from_full_schema("my_tool", sample_schema, SchemaMode.FULL)
        assert slim.tool_name == "my_tool"


# ===================================================================
# AutoProfileSelection
# ===================================================================


class TestAutoProfileSelection:
    """Tests for AutoProfileSelection creation and serialization."""

    def _make(self, **overrides):
        defaults = {
            "model_tier": ModelTier.SMALL_7B,
            "task_domain": "browser",
            "recommended_profile": "browser_exec",
            "recommended_schema_mode": SchemaMode.MINIMAL,
            "recommended_instruction_template": "slim_browser",
            "confidence": 0.85,
            "rationale": "Small model detected, browser domain",
        }
        defaults.update(overrides)
        return AutoProfileSelection(**defaults)

    def test_valid_creation_all_tiers(self):
        for tier in ModelTier:
            sel = self._make(model_tier=tier)
            assert sel.model_tier == tier

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            self._make(confidence=-0.1)

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            self._make(confidence=1.01)

    def test_confidence_boundary_zero(self):
        sel = self._make(confidence=0.0)
        assert sel.confidence == 0.0

    def test_confidence_boundary_one(self):
        sel = self._make(confidence=1.0)
        assert sel.confidence == 1.0

    def test_to_dict_serialization(self):
        sel = self._make()
        d = sel.to_dict()
        assert d["tier"] == "small_7b"
        assert d["domain"] == "browser"
        assert d["profile"] == "browser_exec"
        assert d["schema_mode"] == "minimal"
        assert d["template"] == "slim_browser"
        assert d["confidence"] == 0.85
        assert d["rationale"] == "Small model detected, browser domain"

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        expected_keys = {"tier", "domain", "profile", "schema_mode", "template", "confidence", "rationale"}
        assert set(d.keys()) == expected_keys

    @pytest.mark.parametrize(
        "tier, domain",
        [
            (ModelTier.SMALL_7B, "browser"),
            (ModelTier.SMALL_CONTEXT, "api"),
            (ModelTier.MEDIUM_13B, "desktop"),
            (ModelTier.STANDARD, "browser"),
            (ModelTier.LARGE_CONTEXT, "api"),
            (ModelTier.HOSTED, "desktop"),
        ],
    )
    def test_various_tier_domain_combinations(self, tier, domain):
        sel = self._make(model_tier=tier, task_domain=domain)
        assert sel.model_tier == tier
        assert sel.task_domain == domain
        d = sel.to_dict()
        assert d["tier"] == tier.value
        assert d["domain"] == domain

    def test_frozen_immutability(self):
        sel = self._make()
        with pytest.raises(AttributeError):
            sel.confidence = 0.5  # type: ignore[misc]

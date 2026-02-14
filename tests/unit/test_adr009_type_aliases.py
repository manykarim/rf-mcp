"""Unit tests for ADR-009 type-constrained tool parameter aliases.

Validates that each of the 16 type aliases defined in
robotmcp.domains.shared.kernel:
  1. Produces correct JSON Schema with "enum" values
  2. Normalizes mixed-case / whitespace input
  3. Rejects invalid values with ValidationError
  4. Works correctly with Optional[] wrappers

Run with: uv run pytest tests/unit/test_adr009_type_aliases.py -v
"""

from __future__ import annotations

__test__ = True

from typing import Optional

import pytest
from pydantic import TypeAdapter, ValidationError

from robotmcp.domains.shared.kernel import (
    AttachAction,
    AutomationContext,
    DetailLevel,
    ExecutionMode,
    FilteringLevel,
    FlowStructure,
    IntentVerb,
    KeywordStrategy,
    ModelTierLiteral,
    PluginAction,
    RecommendMode,
    SessionAction,
    SuiteRunMode,
    TestStatus,
    ToolProfileName,
    ValidationLevel,
)

# ============================================================
# Registry of all 16 type aliases with their expected enum values
# ============================================================

_TYPE_ALIASES = {
    "SessionAction": (
        SessionAction,
        [
            "init", "import_library", "import_resource",
            "set_variables", "import_variables",
            "start_test", "end_test", "list_tests",
            "set_suite_setup", "set_suite_teardown",
            "set_tool_profile",
        ],
    ),
    "TestStatus": (
        TestStatus,
        ["pass", "fail"],
    ),
    "ToolProfileName": (
        ToolProfileName,
        ["browser_exec", "api_exec", "discovery", "minimal_exec", "full"],
    ),
    "ModelTierLiteral": (
        ModelTierLiteral,
        ["small_context", "standard", "large_context"],
    ),
    "PluginAction": (
        PluginAction,
        ["list", "reload", "diagnose"],
    ),
    "AttachAction": (
        AttachAction,
        ["status", "stop", "cleanup", "reset", "disconnect_all"],
    ),
    "IntentVerb": (
        IntentVerb,
        [
            "navigate", "click", "fill", "hover",
            "select", "assert_visible", "extract_text", "wait_for",
        ],
    ),
    "KeywordStrategy": (
        KeywordStrategy,
        ["semantic", "pattern", "catalog", "session"],
    ),
    "AutomationContext": (
        AutomationContext,
        ["web", "mobile", "api", "desktop"],
    ),
    "RecommendMode": (
        RecommendMode,
        ["direct", "sampling_prompt", "merge_samples"],
    ),
    "FlowStructure": (
        FlowStructure,
        ["if", "for", "try"],
    ),
    "ExecutionMode": (
        ExecutionMode,
        ["keyword", "evaluate"],
    ),
    "DetailLevel": (
        DetailLevel,
        ["minimal", "standard", "full"],
    ),
    "FilteringLevel": (
        FilteringLevel,
        ["standard", "aggressive"],
    ),
    "SuiteRunMode": (
        SuiteRunMode,
        ["dry", "validate", "full"],
    ),
    "ValidationLevel": (
        ValidationLevel,
        ["minimal", "standard", "strict"],
    ),
}


# ============================================================
# 1. Schema Generation Tests
# ============================================================


class TestSchemaGeneration:
    """Verify each type alias generates JSON Schema with correct enum values."""

    @pytest.mark.parametrize(
        "alias_name, alias_type, expected_values",
        [
            pytest.param(name, info[0], info[1], id=name)
            for name, info in _TYPE_ALIASES.items()
        ],
    )
    def test_schema_has_enum(self, alias_name, alias_type, expected_values):
        """JSON Schema must contain 'enum' with the exact allowed values."""
        ta = TypeAdapter(alias_type)
        schema = ta.json_schema()
        assert "enum" in schema, (
            f"{alias_name} schema is missing 'enum' key: {schema}"
        )
        assert sorted(schema["enum"]) == sorted(expected_values), (
            f"{alias_name} enum mismatch: got {schema['enum']}, "
            f"expected {expected_values}"
        )

    @pytest.mark.parametrize(
        "alias_name, alias_type, expected_values",
        [
            pytest.param(name, info[0], info[1], id=name)
            for name, info in _TYPE_ALIASES.items()
        ],
    )
    def test_schema_type_is_string(self, alias_name, alias_type, expected_values):
        """JSON Schema should declare type as string."""
        ta = TypeAdapter(alias_type)
        schema = ta.json_schema()
        assert schema.get("type") == "string", (
            f"{alias_name} schema type should be 'string', got {schema.get('type')}"
        )

    @pytest.mark.parametrize(
        "alias_name, alias_type, expected_values",
        [
            pytest.param(name, info[0], info[1], id=name)
            for name, info in _TYPE_ALIASES.items()
        ],
    )
    def test_schema_no_anyof(self, alias_name, alias_type, expected_values):
        """Schema should be flat enum, not wrapped in anyOf."""
        ta = TypeAdapter(alias_type)
        schema = ta.json_schema()
        assert "anyOf" not in schema, (
            f"{alias_name} schema should not use anyOf: {schema}"
        )


# ============================================================
# 2. Case Normalization Tests
# ============================================================


class TestCaseNormalization:
    """Verify BeforeValidator normalizes case for all aliases."""

    @pytest.mark.parametrize(
        "alias_name, alias_type, expected_values",
        [
            pytest.param(name, info[0], info[1], id=name)
            for name, info in _TYPE_ALIASES.items()
        ],
    )
    def test_accepts_uppercase(self, alias_name, alias_type, expected_values):
        """UPPERCASE input should be normalized to lowercase."""
        ta = TypeAdapter(alias_type)
        for val in expected_values:
            result = ta.validate_python(val.upper())
            assert result == val, (
                f"{alias_name}: '{val.upper()}' should normalize to '{val}', "
                f"got '{result}'"
            )

    @pytest.mark.parametrize(
        "alias_name, alias_type, expected_values",
        [
            pytest.param(name, info[0], info[1], id=name)
            for name, info in _TYPE_ALIASES.items()
        ],
    )
    def test_accepts_mixed_case(self, alias_name, alias_type, expected_values):
        """MiXeD cAsE input should be normalized to lowercase."""
        ta = TypeAdapter(alias_type)
        for val in expected_values:
            # Convert to alternating case: iNiT, pAsS, etc.
            mixed = "".join(
                c.upper() if i % 2 else c.lower()
                for i, c in enumerate(val)
            )
            result = ta.validate_python(mixed)
            assert result == val, (
                f"{alias_name}: '{mixed}' should normalize to '{val}', "
                f"got '{result}'"
            )

    @pytest.mark.parametrize(
        "alias_name, alias_type, expected_values",
        [
            pytest.param(name, info[0], info[1], id=name)
            for name, info in _TYPE_ALIASES.items()
        ],
    )
    def test_accepts_title_case(self, alias_name, alias_type, expected_values):
        """Title Case input should be normalized to lowercase."""
        ta = TypeAdapter(alias_type)
        first_val = expected_values[0]
        result = ta.validate_python(first_val.title())
        assert result == first_val


# ============================================================
# 3. Whitespace Stripping Tests
# ============================================================


class TestWhitespaceStripping:
    """Verify BeforeValidator strips leading/trailing whitespace."""

    @pytest.mark.parametrize(
        "alias_name, alias_type, expected_values",
        [
            pytest.param(name, info[0], info[1], id=name)
            for name, info in _TYPE_ALIASES.items()
        ],
    )
    def test_strips_leading_whitespace(self, alias_name, alias_type, expected_values):
        """Leading whitespace should be stripped before validation."""
        ta = TypeAdapter(alias_type)
        first_val = expected_values[0]
        result = ta.validate_python(f"  {first_val}")
        assert result == first_val

    @pytest.mark.parametrize(
        "alias_name, alias_type, expected_values",
        [
            pytest.param(name, info[0], info[1], id=name)
            for name, info in _TYPE_ALIASES.items()
        ],
    )
    def test_strips_trailing_whitespace(self, alias_name, alias_type, expected_values):
        """Trailing whitespace should be stripped before validation."""
        ta = TypeAdapter(alias_type)
        first_val = expected_values[0]
        result = ta.validate_python(f"{first_val}  ")
        assert result == first_val

    @pytest.mark.parametrize(
        "alias_name, alias_type, expected_values",
        [
            pytest.param(name, info[0], info[1], id=name)
            for name, info in _TYPE_ALIASES.items()
        ],
    )
    def test_strips_surrounding_whitespace(self, alias_name, alias_type, expected_values):
        """Both leading and trailing whitespace should be stripped."""
        ta = TypeAdapter(alias_type)
        first_val = expected_values[0]
        result = ta.validate_python(f"  {first_val}  ")
        assert result == first_val

    @pytest.mark.parametrize(
        "alias_name, alias_type, expected_values",
        [
            pytest.param(name, info[0], info[1], id=name)
            for name, info in _TYPE_ALIASES.items()
        ],
    )
    def test_strips_tabs_and_newlines(self, alias_name, alias_type, expected_values):
        """Tab and newline characters should be stripped."""
        ta = TypeAdapter(alias_type)
        first_val = expected_values[0]
        result = ta.validate_python(f"\t{first_val}\n")
        assert result == first_val


# ============================================================
# 4. Rejection Tests
# ============================================================


class TestRejection:
    """Verify invalid values are rejected with ValidationError."""

    @pytest.mark.parametrize(
        "alias_name, alias_type, expected_values",
        [
            pytest.param(name, info[0], info[1], id=name)
            for name, info in _TYPE_ALIASES.items()
        ],
    )
    def test_rejects_invalid_string(self, alias_name, alias_type, expected_values):
        """Completely invalid string values should raise ValidationError."""
        ta = TypeAdapter(alias_type)
        with pytest.raises(ValidationError):
            ta.validate_python("bogus_invalid_value_xyz")

    @pytest.mark.parametrize(
        "alias_name, alias_type, expected_values",
        [
            pytest.param(name, info[0], info[1], id=name)
            for name, info in _TYPE_ALIASES.items()
        ],
    )
    def test_rejects_empty_string(self, alias_name, alias_type, expected_values):
        """Empty string should raise ValidationError."""
        ta = TypeAdapter(alias_type)
        with pytest.raises(ValidationError):
            ta.validate_python("")

    @pytest.mark.parametrize(
        "alias_name, alias_type, expected_values",
        [
            pytest.param(name, info[0], info[1], id=name)
            for name, info in _TYPE_ALIASES.items()
        ],
    )
    def test_rejects_whitespace_only(self, alias_name, alias_type, expected_values):
        """Whitespace-only string should raise ValidationError."""
        ta = TypeAdapter(alias_type)
        with pytest.raises(ValidationError):
            ta.validate_python("   ")

    @pytest.mark.parametrize(
        "alias_name, alias_type, expected_values",
        [
            pytest.param(name, info[0], info[1], id=name)
            for name, info in _TYPE_ALIASES.items()
        ],
    )
    def test_rejects_numeric_value(self, alias_name, alias_type, expected_values):
        """Numeric input should raise ValidationError."""
        ta = TypeAdapter(alias_type)
        with pytest.raises(ValidationError):
            ta.validate_python(42)

    @pytest.mark.parametrize(
        "alias_name, alias_type, expected_values",
        [
            pytest.param(name, info[0], info[1], id=name)
            for name, info in _TYPE_ALIASES.items()
        ],
    )
    def test_rejects_typo_value(self, alias_name, alias_type, expected_values):
        """Misspelled value should raise ValidationError."""
        ta = TypeAdapter(alias_type)
        first_val = expected_values[0]
        typo = first_val + "_typo"
        with pytest.raises(ValidationError):
            ta.validate_python(typo)

    @pytest.mark.parametrize(
        "alias_name, alias_type, expected_values",
        [
            pytest.param(name, info[0], info[1], id=name)
            for name, info in _TYPE_ALIASES.items()
        ],
    )
    def test_rejects_none_on_required(self, alias_name, alias_type, expected_values):
        """None should raise ValidationError for non-optional aliases."""
        ta = TypeAdapter(alias_type)
        with pytest.raises(ValidationError):
            ta.validate_python(None)


# ============================================================
# 5. Optional Variant Tests
# ============================================================


class TestOptionalVariants:
    """Verify Optional[TypeAlias] accepts None and normalizes strings."""

    @pytest.mark.parametrize(
        "alias_name, alias_type, expected_values",
        [
            pytest.param(name, info[0], info[1], id=name)
            for name, info in _TYPE_ALIASES.items()
        ],
    )
    def test_optional_accepts_none(self, alias_name, alias_type, expected_values):
        """Optional wrapper should accept None."""
        ta = TypeAdapter(Optional[alias_type])
        result = ta.validate_python(None)
        assert result is None

    @pytest.mark.parametrize(
        "alias_name, alias_type, expected_values",
        [
            pytest.param(name, info[0], info[1], id=name)
            for name, info in _TYPE_ALIASES.items()
        ],
    )
    def test_optional_normalizes_valid_string(self, alias_name, alias_type, expected_values):
        """Optional wrapper should still normalize valid string values."""
        ta = TypeAdapter(Optional[alias_type])
        first_val = expected_values[0]
        result = ta.validate_python(first_val.upper())
        assert result == first_val

    @pytest.mark.parametrize(
        "alias_name, alias_type, expected_values",
        [
            pytest.param(name, info[0], info[1], id=name)
            for name, info in _TYPE_ALIASES.items()
        ],
    )
    def test_optional_rejects_invalid_string(self, alias_name, alias_type, expected_values):
        """Optional wrapper should still reject invalid strings."""
        ta = TypeAdapter(Optional[alias_type])
        with pytest.raises(ValidationError):
            ta.validate_python("bogus_invalid_value_xyz")

    @pytest.mark.parametrize(
        "alias_name, alias_type, expected_values",
        [
            pytest.param(name, info[0], info[1], id=name)
            for name, info in _TYPE_ALIASES.items()
        ],
    )
    def test_optional_schema_has_anyof_with_enum(self, alias_name, alias_type, expected_values):
        """Optional schema should use anyOf with enum and null."""
        ta = TypeAdapter(Optional[alias_type])
        schema = ta.json_schema()
        # Pydantic can express Optional as anyOf[{enum...}, {null}]
        # or with a top-level enum + nullable flag. Accept both.
        if "anyOf" in schema:
            enum_variants = [v for v in schema["anyOf"] if "enum" in v]
            assert len(enum_variants) >= 1, (
                f"{alias_name} Optional schema has anyOf but no enum variant: {schema}"
            )
            enum_values = enum_variants[0]["enum"]
            assert sorted(enum_values) == sorted(expected_values)
        elif "enum" in schema:
            # Pydantic may include None in enum for Optional
            non_null = [v for v in schema["enum"] if v is not None]
            assert sorted(non_null) == sorted(expected_values)
        else:
            pytest.fail(
                f"{alias_name} Optional schema has neither 'enum' nor 'anyOf': {schema}"
            )


# ============================================================
# 6. Specific Alias Targeted Tests
# ============================================================


class TestSessionActionSpecifics:
    """Targeted tests for SessionAction (most values of any alias)."""

    def test_all_11_actions_accepted(self):
        ta = TypeAdapter(SessionAction)
        actions = [
            "init", "import_library", "import_resource",
            "set_variables", "import_variables",
            "start_test", "end_test", "list_tests",
            "set_suite_setup", "set_suite_teardown",
            "set_tool_profile",
        ]
        for action in actions:
            assert ta.validate_python(action) == action

    def test_session_action_count(self):
        """SessionAction should have exactly 11 allowed values."""
        ta = TypeAdapter(SessionAction)
        schema = ta.json_schema()
        assert len(schema["enum"]) == 11

    def test_session_action_init_case_variants(self):
        ta = TypeAdapter(SessionAction)
        for variant in ["Init", "INIT", "iNiT", " init ", "\tinit\n"]:
            assert ta.validate_python(variant) == "init"


class TestIntentVerbSpecifics:
    """Targeted tests for IntentVerb (used by intent_action tool)."""

    def test_all_8_verbs_accepted(self):
        ta = TypeAdapter(IntentVerb)
        verbs = [
            "navigate", "click", "fill", "hover",
            "select", "assert_visible", "extract_text", "wait_for",
        ]
        for verb in verbs:
            assert ta.validate_python(verb) == verb

    def test_intent_verb_count(self):
        ta = TypeAdapter(IntentVerb)
        schema = ta.json_schema()
        assert len(schema["enum"]) == 8

    def test_navigate_case_variants(self):
        ta = TypeAdapter(IntentVerb)
        for variant in ["Navigate", "NAVIGATE", "  navigate  "]:
            assert ta.validate_python(variant) == "navigate"

    def test_assert_visible_with_underscore(self):
        """Underscore values must be preserved (only case/whitespace normalized)."""
        ta = TypeAdapter(IntentVerb)
        assert ta.validate_python("ASSERT_VISIBLE") == "assert_visible"
        assert ta.validate_python("Assert_Visible") == "assert_visible"


class TestFlowStructureSpecifics:
    """Targeted tests for FlowStructure (only 3 values)."""

    def test_all_3_structures(self):
        ta = TypeAdapter(FlowStructure)
        for val in ["if", "for", "try"]:
            assert ta.validate_python(val) == val

    def test_rejects_while(self):
        ta = TypeAdapter(FlowStructure)
        with pytest.raises(ValidationError):
            ta.validate_python("while")

    def test_rejects_switch(self):
        ta = TypeAdapter(FlowStructure)
        with pytest.raises(ValidationError):
            ta.validate_python("switch")


class TestToolProfileNameSpecifics:
    """Targeted tests for ToolProfileName (ADR-006 profiles)."""

    def test_all_5_profiles(self):
        ta = TypeAdapter(ToolProfileName)
        profiles = ["browser_exec", "api_exec", "discovery", "minimal_exec", "full"]
        for p in profiles:
            assert ta.validate_python(p) == p

    def test_browser_exec_case(self):
        ta = TypeAdapter(ToolProfileName)
        assert ta.validate_python("BROWSER_EXEC") == "browser_exec"
        assert ta.validate_python("Browser_Exec") == "browser_exec"

    def test_optional_tool_profile_none(self):
        ta = TypeAdapter(Optional[ToolProfileName])
        assert ta.validate_python(None) is None

    def test_optional_tool_profile_valid(self):
        ta = TypeAdapter(Optional[ToolProfileName])
        assert ta.validate_python("BROWSER_EXEC") == "browser_exec"


class TestModelTierLiteralSpecifics:
    """Targeted tests for ModelTierLiteral."""

    def test_all_3_tiers(self):
        ta = TypeAdapter(ModelTierLiteral)
        for tier in ["small_context", "standard", "large_context"]:
            assert ta.validate_python(tier) == tier

    def test_optional_model_tier_none(self):
        ta = TypeAdapter(Optional[ModelTierLiteral])
        assert ta.validate_python(None) is None

    def test_small_context_upper(self):
        ta = TypeAdapter(ModelTierLiteral)
        assert ta.validate_python("SMALL_CONTEXT") == "small_context"


class TestTestStatusSpecifics:
    """Targeted tests for TestStatus (only 2 values)."""

    def test_pass_and_fail(self):
        ta = TypeAdapter(TestStatus)
        assert ta.validate_python("pass") == "pass"
        assert ta.validate_python("fail") == "fail"

    def test_rejects_skip(self):
        ta = TypeAdapter(TestStatus)
        with pytest.raises(ValidationError):
            ta.validate_python("skip")

    def test_rejects_error(self):
        ta = TypeAdapter(TestStatus)
        with pytest.raises(ValidationError):
            ta.validate_python("error")


# ============================================================
# 7. Cross-Cutting Concerns
# ============================================================


class TestNormalizeStrFunction:
    """Tests for the _normalize_str helper used by all aliases."""

    def test_non_string_passthrough(self):
        """Non-string values should pass through unchanged (Pydantic rejects later)."""
        from robotmcp.domains.shared.kernel import _normalize_str

        assert _normalize_str(42) == 42
        assert _normalize_str(None) is None
        assert _normalize_str([]) == []
        assert _normalize_str(True) is True

    def test_string_lowered_and_stripped(self):
        from robotmcp.domains.shared.kernel import _normalize_str

        assert _normalize_str("  FOO  ") == "foo"
        assert _normalize_str("BAR") == "bar"
        assert _normalize_str("\tBaz\n") == "baz"

    def test_already_normalized(self):
        from robotmcp.domains.shared.kernel import _normalize_str

        assert _normalize_str("init") == "init"
        assert _normalize_str("pass") == "pass"


class TestAllAliasesExported:
    """Verify all 16 type aliases are importable from kernel module."""

    def test_all_16_aliases_importable(self):
        from robotmcp.domains.shared import kernel

        aliases = [
            "SessionAction", "TestStatus", "ToolProfileName",
            "ModelTierLiteral", "PluginAction", "AttachAction",
            "IntentVerb", "KeywordStrategy", "AutomationContext",
            "RecommendMode", "FlowStructure", "ExecutionMode",
            "DetailLevel", "FilteringLevel", "SuiteRunMode",
            "ValidationLevel",
        ]
        for name in aliases:
            assert hasattr(kernel, name), f"kernel module missing '{name}'"

    def test_exactly_16_aliases_in_registry(self):
        """Our test registry should cover all 16 aliases."""
        assert len(_TYPE_ALIASES) == 16

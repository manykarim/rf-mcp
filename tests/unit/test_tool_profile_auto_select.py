"""Unit tests for ADR-016 auto-selection: AutoProfileSelection VO and
ToolProfileManager.auto_select_profile / generate_slim_schema methods.

Run with: uv run pytest tests/unit/test_tool_profile_auto_select.py -v
"""

__test__ = True

import pytest

from robotmcp.domains.tool_profile.aggregates import ProfilePresets, ToolProfile
from robotmcp.domains.tool_profile.entities import ToolDescriptor
from robotmcp.domains.tool_profile.services import ToolProfileManager
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


class MockToolManagerPort:
    """Mock implementation of ToolManagerPort for testing."""

    def __init__(self):
        self.visible = frozenset(ProfilePresets.ALL_TOOLS)
        self.removed = []
        self.added = []
        self.swapped = []

    async def remove_tool(self, name):
        self.removed.append(name)
        self.visible = self.visible - {name}

    async def add_tool_with_description(self, name, desc, schema):
        self.added.append(name)
        self.visible = self.visible | {name}

    async def get_visible_tool_names(self):
        return self.visible

    async def swap_tool_description(self, name, desc, schema):
        self.swapped.append(name)


@pytest.fixture
def mock_tool_manager():
    return MockToolManagerPort()


@pytest.fixture
def sample_descriptors():
    """Descriptor registry for all tools with realistic schemas."""
    descriptors = {}
    for tool_name in ProfilePresets.ALL_TOOLS:
        descriptors[tool_name] = ToolDescriptor(
            tool_name=tool_name,
            tags=frozenset({ToolTag.CORE}),
            description_full=f"Full {tool_name}",
            description_compact=f"Compact {tool_name}",
            description_minimal=f"Min {tool_name}",
            schema_full={
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": f"Keyword for {tool_name}"},
                    "args": {"type": "array", "description": "Arguments"},
                    "timeout": {"type": "integer", "description": "Timeout"},
                },
                "required": ["keyword"],
            },
            token_estimate_full=400,
            token_estimate_compact=100,
            token_estimate_minimal=50,
        )
    return descriptors


@pytest.fixture
def event_log():
    return []


@pytest.fixture
def manager(mock_tool_manager, sample_descriptors, event_log):
    return ToolProfileManager(
        tool_manager=mock_tool_manager,
        descriptor_registry=sample_descriptors,
        event_publisher=lambda evt: event_log.append(evt),
    )


# =============================================================================
# AutoProfileSelection Value Object
# =============================================================================


class TestAutoProfileSelectionConstruction:
    """Test AutoProfileSelection construction and invariants."""

    def test_valid_construction(self):
        aps = AutoProfileSelection(
            model_tier=ModelTier.SMALL_7B,
            task_domain="browser",
            recommended_profile="slim_exec",
            recommended_schema_mode=SchemaMode.MINIMAL,
            recommended_instruction_template="slim_7b",
            confidence=0.95,
            rationale="7B model needs slim profile",
        )
        assert aps.model_tier == ModelTier.SMALL_7B
        assert aps.task_domain == "browser"
        assert aps.recommended_profile == "slim_exec"

    def test_confidence_at_zero(self):
        aps = AutoProfileSelection(
            model_tier=ModelTier.STANDARD,
            task_domain="browser",
            recommended_profile="full",
            recommended_schema_mode=SchemaMode.FULL,
            recommended_instruction_template="full",
            confidence=0.0,
            rationale="No confidence",
        )
        assert aps.confidence == 0.0

    def test_confidence_at_one(self):
        aps = AutoProfileSelection(
            model_tier=ModelTier.SMALL_7B,
            task_domain="browser",
            recommended_profile="slim_exec",
            recommended_schema_mode=SchemaMode.MINIMAL,
            recommended_instruction_template="slim_7b",
            confidence=1.0,
            rationale="Maximum confidence",
        )
        assert aps.confidence == 1.0

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValueError, match="confidence must be 0.0-1.0"):
            AutoProfileSelection(
                model_tier=ModelTier.STANDARD,
                task_domain="browser",
                recommended_profile="full",
                recommended_schema_mode=SchemaMode.FULL,
                recommended_instruction_template="full",
                confidence=-0.1,
                rationale="Invalid",
            )

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValueError, match="confidence must be 0.0-1.0"):
            AutoProfileSelection(
                model_tier=ModelTier.STANDARD,
                task_domain="browser",
                recommended_profile="full",
                recommended_schema_mode=SchemaMode.FULL,
                recommended_instruction_template="full",
                confidence=1.1,
                rationale="Invalid",
            )

    def test_is_frozen(self):
        aps = AutoProfileSelection(
            model_tier=ModelTier.SMALL_7B,
            task_domain="browser",
            recommended_profile="slim_exec",
            recommended_schema_mode=SchemaMode.MINIMAL,
            recommended_instruction_template="slim_7b",
            confidence=0.95,
            rationale="Frozen",
        )
        with pytest.raises(AttributeError):
            aps.confidence = 0.5


class TestAutoProfileSelectionToDict:
    """Test AutoProfileSelection.to_dict serialization."""

    def test_to_dict_keys(self):
        aps = AutoProfileSelection(
            model_tier=ModelTier.SMALL_7B,
            task_domain="browser",
            recommended_profile="slim_exec",
            recommended_schema_mode=SchemaMode.MINIMAL,
            recommended_instruction_template="slim_7b",
            confidence=0.95,
            rationale="Test",
        )
        d = aps.to_dict()
        assert set(d.keys()) == {
            "tier", "domain", "profile", "schema_mode",
            "template", "confidence", "rationale",
        }

    def test_to_dict_values(self):
        aps = AutoProfileSelection(
            model_tier=ModelTier.SMALL_7B,
            task_domain="api",
            recommended_profile="slim_exec",
            recommended_schema_mode=SchemaMode.MINIMAL,
            recommended_instruction_template="slim_7b",
            confidence=0.9,
            rationale="API domain",
        )
        d = aps.to_dict()
        assert d["tier"] == "small_7b"
        assert d["domain"] == "api"
        assert d["profile"] == "slim_exec"
        assert d["schema_mode"] == "minimal"
        assert d["template"] == "slim_7b"
        assert d["confidence"] == 0.9
        assert d["rationale"] == "API domain"

    def test_to_dict_schema_mode_full(self):
        aps = AutoProfileSelection(
            model_tier=ModelTier.HOSTED,
            task_domain="browser",
            recommended_profile="full",
            recommended_schema_mode=SchemaMode.FULL,
            recommended_instruction_template="full",
            confidence=0.9,
            rationale="Hosted model",
        )
        d = aps.to_dict()
        assert d["schema_mode"] == "full"


# =============================================================================
# auto_select_profile - SMALL_7B
# =============================================================================


class TestAutoSelectSmall7B:
    """Test auto_select_profile for SMALL_7B tier."""

    def test_browser_domain(self, manager):
        result = manager.auto_select_profile(ModelTier.SMALL_7B, "browser")
        assert result.recommended_profile == "slim_exec"
        assert result.recommended_schema_mode == SchemaMode.MINIMAL

    def test_api_domain(self, manager):
        result = manager.auto_select_profile(ModelTier.SMALL_7B, "api")
        assert result.recommended_profile == "slim_exec"

    def test_desktop_domain(self, manager):
        result = manager.auto_select_profile(ModelTier.SMALL_7B, "desktop")
        assert result.recommended_profile == "slim_exec"

    def test_unknown_domain(self, manager):
        result = manager.auto_select_profile(ModelTier.SMALL_7B, "whatever")
        assert result.recommended_profile == "slim_exec"

    def test_high_confidence(self, manager):
        result = manager.auto_select_profile(ModelTier.SMALL_7B, "browser")
        assert result.confidence >= 0.9

    def test_instruction_template(self, manager):
        result = manager.auto_select_profile(ModelTier.SMALL_7B, "browser")
        assert result.recommended_instruction_template == "slim_7b"

    def test_rationale_not_empty(self, manager):
        result = manager.auto_select_profile(ModelTier.SMALL_7B, "browser")
        assert len(result.rationale) > 0


# =============================================================================
# auto_select_profile - SMALL_CONTEXT
# =============================================================================


class TestAutoSelectSmallContext:
    """Test auto_select_profile for SMALL_CONTEXT tier."""

    def test_browser_domain(self, manager):
        result = manager.auto_select_profile(ModelTier.SMALL_CONTEXT, "browser")
        assert result.recommended_profile == "browser_exec"

    def test_api_domain(self, manager):
        result = manager.auto_select_profile(ModelTier.SMALL_CONTEXT, "api")
        assert result.recommended_profile == "api_exec"

    def test_desktop_domain(self, manager):
        result = manager.auto_select_profile(ModelTier.SMALL_CONTEXT, "desktop")
        assert result.recommended_profile == "desktop_exec"

    def test_unknown_domain_defaults_to_browser(self, manager):
        result = manager.auto_select_profile(ModelTier.SMALL_CONTEXT, "generic")
        assert result.recommended_profile == "browser_exec"

    def test_schema_mode_standard(self, manager):
        result = manager.auto_select_profile(ModelTier.SMALL_CONTEXT, "browser")
        assert result.recommended_schema_mode == SchemaMode.STANDARD

    def test_instruction_template(self, manager):
        result = manager.auto_select_profile(ModelTier.SMALL_CONTEXT, "browser")
        assert result.recommended_instruction_template == "small_context"


# =============================================================================
# auto_select_profile - MEDIUM_13B
# =============================================================================


class TestAutoSelectMedium13B:
    """Test auto_select_profile for MEDIUM_13B tier."""

    def test_browser_domain(self, manager):
        result = manager.auto_select_profile(ModelTier.MEDIUM_13B, "browser")
        assert result.recommended_profile == "browser_exec"

    def test_api_domain(self, manager):
        result = manager.auto_select_profile(ModelTier.MEDIUM_13B, "api")
        assert result.recommended_profile == "api_exec"

    def test_desktop_domain(self, manager):
        result = manager.auto_select_profile(ModelTier.MEDIUM_13B, "desktop")
        assert result.recommended_profile == "desktop_exec"

    def test_schema_mode_standard(self, manager):
        result = manager.auto_select_profile(ModelTier.MEDIUM_13B, "browser")
        assert result.recommended_schema_mode == SchemaMode.STANDARD

    def test_instruction_template(self, manager):
        result = manager.auto_select_profile(ModelTier.MEDIUM_13B, "browser")
        assert result.recommended_instruction_template == "medium_13b"


# =============================================================================
# auto_select_profile - STANDARD / LARGE_CONTEXT / HOSTED
# =============================================================================


class TestAutoSelectLargerTiers:
    """Test auto_select_profile for STANDARD, LARGE_CONTEXT, HOSTED tiers."""

    def test_standard_returns_full(self, manager):
        result = manager.auto_select_profile(ModelTier.STANDARD, "browser")
        assert result.recommended_profile == "full"
        assert result.recommended_schema_mode == SchemaMode.FULL

    def test_large_context_returns_full(self, manager):
        result = manager.auto_select_profile(ModelTier.LARGE_CONTEXT, "browser")
        assert result.recommended_profile == "full"
        assert result.recommended_schema_mode == SchemaMode.FULL

    def test_hosted_returns_full(self, manager):
        result = manager.auto_select_profile(ModelTier.HOSTED, "api")
        assert result.recommended_profile == "full"
        assert result.recommended_schema_mode == SchemaMode.FULL

    def test_standard_instruction_template(self, manager):
        result = manager.auto_select_profile(ModelTier.STANDARD, "browser")
        assert result.recommended_instruction_template == "full"

    def test_large_context_confidence(self, manager):
        result = manager.auto_select_profile(ModelTier.LARGE_CONTEXT, "browser")
        assert result.confidence >= 0.8

    def test_hosted_confidence(self, manager):
        result = manager.auto_select_profile(ModelTier.HOSTED, "browser")
        assert result.confidence >= 0.8


# =============================================================================
# auto_select_profile - domain normalization
# =============================================================================


class TestAutoSelectDomainNormalization:
    """Test that auto_select_profile normalizes domain strings."""

    def test_uppercase_domain(self, manager):
        result = manager.auto_select_profile(ModelTier.SMALL_CONTEXT, "API")
        assert result.recommended_profile == "api_exec"

    def test_mixed_case_domain(self, manager):
        result = manager.auto_select_profile(ModelTier.SMALL_CONTEXT, "Browser")
        assert result.recommended_profile == "browser_exec"

    def test_whitespace_domain(self, manager):
        result = manager.auto_select_profile(ModelTier.SMALL_CONTEXT, "  api  ")
        assert result.recommended_profile == "api_exec"

    def test_task_domain_in_result(self, manager):
        result = manager.auto_select_profile(ModelTier.SMALL_CONTEXT, "  API  ")
        assert result.task_domain == "api"


# =============================================================================
# generate_slim_schema
# =============================================================================


class TestGenerateSlimSchema:
    """Test ToolProfileManager.generate_slim_schema."""

    def test_returns_slim_schema_for_known_tool(self, manager):
        result = manager.generate_slim_schema("manage_session")
        assert result is not None
        assert isinstance(result, SlimToolSchema)
        assert result.tool_name == "manage_session"

    def test_returns_none_for_unknown_tool(self, manager):
        result = manager.generate_slim_schema("nonexistent_tool")
        assert result is None

    def test_schema_mode_is_minimal(self, manager):
        result = manager.generate_slim_schema("manage_session")
        assert result.schema_mode == SchemaMode.MINIMAL

    def test_only_required_properties_kept(self, manager):
        result = manager.generate_slim_schema("manage_session")
        assert "keyword" in result.properties
        # args and timeout are not required, should be removed
        assert "args" not in result.properties
        assert "timeout" not in result.properties

    def test_descriptions_stripped(self, manager):
        result = manager.generate_slim_schema("manage_session")
        for prop_def in result.properties.values():
            if isinstance(prop_def, dict):
                assert "description" not in prop_def

    def test_returns_none_for_tool_with_no_properties(self, mock_tool_manager, event_log):
        """Tool with empty schema (no properties) returns None."""
        descriptors = {
            "empty_tool": ToolDescriptor(
                tool_name="empty_tool",
                tags=frozenset({ToolTag.CORE}),
                description_full="Full",
                description_compact="Compact",
                description_minimal="Min",
                schema_full={"type": "object"},
            ),
        }
        mgr = ToolProfileManager(
            tool_manager=mock_tool_manager,
            descriptor_registry=descriptors,
            event_publisher=lambda evt: event_log.append(evt),
        )
        result = mgr.generate_slim_schema("empty_tool")
        assert result is None

    def test_token_estimate_positive(self, manager):
        result = manager.generate_slim_schema("execute_step")
        assert result.token_estimate > 0

    def test_to_schema_produces_valid_object(self, manager):
        result = manager.generate_slim_schema("execute_step")
        schema = result.to_schema()
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

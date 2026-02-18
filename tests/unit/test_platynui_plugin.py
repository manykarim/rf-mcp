"""Comprehensive unit tests for PlatynUIPlugin (ADR-012).

Tests cover: metadata, capabilities, keyword maps, keyword sets,
validate_keyword_for_session, generate_failure_hints,
get_keyword_alternatives, state provider, and plugin registration.

Run with: uv run pytest tests/unit/test_platynui_plugin.py -v
"""

__test__ = True

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock

import pytest

from robotmcp.plugins.builtin.platynui_plugin import (
    PlatynUIPlugin,
    PlatynUIStateProvider,
)
from robotmcp.plugins.builtin import generate_builtin_plugins


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def plugin() -> PlatynUIPlugin:
    return PlatynUIPlugin()


@pytest.fixture
def state_provider() -> PlatynUIStateProvider:
    return PlatynUIStateProvider()


def _make_mock_session(
    *,
    session_id: str = "test-desktop",
    session_type: str = "desktop_testing",
    imported_libraries: Optional[List[str]] = None,
    explicit_library_preference: Optional[str] = None,
) -> MagicMock:
    """Create a lightweight mock session for plugin validation tests."""
    session = MagicMock()
    session.session_id = session_id
    session.session_type = session_type
    session.imported_libraries = imported_libraries or ["PlatynUI.BareMetal"]
    session.explicit_library_preference = (
        explicit_library_preference or "PlatynUI"
    )
    return session


# ===========================================================================
# Metadata tests
# ===========================================================================

class TestPlatynUIMetadata:
    """Tests for plugin metadata fields."""

    def test_plugin_name(self, plugin):
        assert plugin.metadata.name == "PlatynUI"

    def test_package_name(self, plugin):
        assert plugin.metadata.package_name == "robotframework-platynui"

    def test_import_path(self, plugin):
        assert plugin.metadata.import_path == "PlatynUI.BareMetal"

    def test_library_type(self, plugin):
        assert plugin.metadata.library_type == "external"

    def test_contexts_include_desktop(self, plugin):
        assert "desktop" in plugin.metadata.contexts

    def test_categories_include_desktop(self, plugin):
        assert "desktop" in plugin.metadata.categories

    def test_categories_include_gui(self, plugin):
        assert "gui" in plugin.metadata.categories

    def test_categories_include_accessibility(self, plugin):
        assert "accessibility" in plugin.metadata.categories

    def test_supports_async_is_false(self, plugin):
        assert plugin.metadata.supports_async is False

    def test_requires_type_conversion_is_true(self, plugin):
        assert plugin.metadata.requires_type_conversion is True

    def test_class_has_test_false(self):
        assert PlatynUIPlugin.__test__ is False

    def test_description_nonempty(self, plugin):
        assert len(plugin.metadata.description) > 10

    def test_installation_command(self, plugin):
        assert "pip install" in plugin.metadata.installation_command

    def test_dependencies_include_native(self, plugin):
        assert "platynui_native" in plugin.metadata.dependencies

    def test_load_priority(self, plugin):
        assert plugin.metadata.load_priority == 55

    def test_default_enabled(self, plugin):
        assert plugin.metadata.default_enabled is True


# ===========================================================================
# Capabilities tests
# ===========================================================================

class TestPlatynUICapabilities:
    """Tests for plugin capabilities."""

    def test_contexts(self, plugin):
        assert plugin.capabilities.contexts == ["desktop"]

    def test_supports_page_source_false(self, plugin):
        assert plugin.capabilities.supports_page_source is False

    def test_supports_application_state_true(self, plugin):
        assert plugin.capabilities.supports_application_state is True

    def test_features_element_inspection(self, plugin):
        assert "element_inspection" in plugin.capabilities.features

    def test_features_keyboard_input(self, plugin):
        assert "keyboard_input" in plugin.capabilities.features

    def test_features_pointer_input(self, plugin):
        assert "pointer_input" in plugin.capabilities.features

    def test_features_window_management(self, plugin):
        assert "window_management" in plugin.capabilities.features

    def test_features_screenshot(self, plugin):
        assert "screenshot" in plugin.capabilities.features

    def test_features_accessibility_tree(self, plugin):
        assert "accessibility_tree" in plugin.capabilities.features

    def test_technology_uia(self, plugin):
        assert "uia" in plugin.capabilities.technology

    def test_technology_atspi(self, plugin):
        assert "atspi" in plugin.capabilities.technology

    def test_technology_ax(self, plugin):
        assert "ax" in plugin.capabilities.technology

    def test_technology_pyo3(self, plugin):
        assert "pyo3" in plugin.capabilities.technology


# ===========================================================================
# Keyword map tests
# ===========================================================================

class TestKeywordLibraryMap:
    """Tests for get_keyword_library_map()."""

    def test_keyword_map_has_19_entries(self, plugin):
        kw_map = plugin.get_keyword_library_map()
        assert len(kw_map) == 19

    def test_all_keywords_map_to_platynui(self, plugin):
        kw_map = plugin.get_keyword_library_map()
        for kw, lib in kw_map.items():
            assert lib == "PlatynUI", f"Keyword '{kw}' maps to '{lib}', expected 'PlatynUI'"

    @pytest.mark.parametrize("keyword", [
        "query",
        "pointer click",
        "pointer multi click",
        "keyboard type",
        "focus",
        "activate",
        "maximize",
        "minimize",
        "close",
        "restore",
        "get attribute",
        "take screenshot",
        "highlight",
        "pointer move to",
        "get pointer position",
        "pointer press",
        "pointer release",
        "keyboard press",
        "keyboard release",
    ])
    def test_specific_keyword_exists(self, plugin, keyword):
        kw_map = plugin.get_keyword_library_map()
        assert keyword in kw_map, f"Expected keyword '{keyword}' not found in map"


# ===========================================================================
# Keyword sets tests
# ===========================================================================

class TestKeywordSets:
    """Tests for internal keyword set constants."""

    def test_all_keywords_count(self):
        assert len(PlatynUIPlugin._ALL_KEYWORDS) == 19

    def test_browser_shared_keywords_count(self):
        assert len(PlatynUIPlugin._BROWSER_SHARED_KEYWORDS) == 3

    def test_unique_keywords_count(self):
        assert len(PlatynUIPlugin._UNIQUE_KEYWORDS) == 16

    def test_unique_is_all_minus_shared(self):
        assert PlatynUIPlugin._UNIQUE_KEYWORDS == (
            PlatynUIPlugin._ALL_KEYWORDS - PlatynUIPlugin._BROWSER_SHARED_KEYWORDS
        )

    def test_shared_keywords_contents(self):
        assert PlatynUIPlugin._BROWSER_SHARED_KEYWORDS == frozenset({
            "focus", "get attribute", "take screenshot",
        })

    def test_sets_are_frozensets(self):
        assert isinstance(PlatynUIPlugin._ALL_KEYWORDS, frozenset)
        assert isinstance(PlatynUIPlugin._BROWSER_SHARED_KEYWORDS, frozenset)
        assert isinstance(PlatynUIPlugin._UNIQUE_KEYWORDS, frozenset)


# ===========================================================================
# validate_keyword_for_session tests
# ===========================================================================

class TestValidateKeywordForSession:
    """Tests for validate_keyword_for_session()."""

    def test_platynui_keyword_in_desktop_session_is_valid(self, plugin):
        session = _make_mock_session()
        result = plugin.validate_keyword_for_session(session, "Pointer Click", "PlatynUI")
        assert result is None

    def test_web_keyword_in_desktop_session_returns_error(self, plugin):
        session = _make_mock_session()
        result = plugin.validate_keyword_for_session(
            session, "click element", "SeleniumLibrary"
        )
        assert result is not None
        assert result["success"] is False
        assert "alternative" in result

    def test_web_keyword_fill_text_in_desktop_session_returns_error(self, plugin):
        session = _make_mock_session()
        result = plugin.validate_keyword_for_session(
            session, "fill text", "Browser"
        )
        assert result is not None
        assert result["success"] is False

    def test_web_keyword_allowed_when_web_library_also_imported(self, plugin):
        session = _make_mock_session(
            imported_libraries=["PlatynUI.BareMetal", "Browser"]
        )
        result = plugin.validate_keyword_for_session(
            session, "click element", "Browser"
        )
        assert result is None

    def test_web_keyword_allowed_when_selenium_also_imported(self, plugin):
        session = _make_mock_session(
            imported_libraries=["PlatynUI.BareMetal", "SeleniumLibrary"]
        )
        result = plugin.validate_keyword_for_session(
            session, "click element", "SeleniumLibrary"
        )
        assert result is None

    def test_shared_keyword_focus_is_valid(self, plugin):
        session = _make_mock_session()
        result = plugin.validate_keyword_for_session(
            session, "focus", "Browser"
        )
        assert result is None

    def test_shared_keyword_get_attribute_is_valid(self, plugin):
        session = _make_mock_session()
        result = plugin.validate_keyword_for_session(
            session, "get attribute", "Browser"
        )
        assert result is None

    def test_shared_keyword_take_screenshot_is_valid(self, plugin):
        session = _make_mock_session()
        result = plugin.validate_keyword_for_session(
            session, "take screenshot", "Browser"
        )
        assert result is None

    def test_non_platynui_session_always_valid(self, plugin):
        session = _make_mock_session(
            imported_libraries=["Browser"],
            explicit_library_preference="Browser",
        )
        result = plugin.validate_keyword_for_session(
            session, "click element", "Browser"
        )
        assert result is None

    def test_error_dict_has_expected_keys(self, plugin):
        session = _make_mock_session()
        result = plugin.validate_keyword_for_session(
            session, "click element", "SeleniumLibrary"
        )
        assert result is not None
        assert "error" in result
        assert "keyword" in result
        assert "keyword_library" in result
        assert "session_library" in result
        assert "alternative" in result
        assert "hints" in result

    def test_error_suggests_pointer_click_for_click_element(self, plugin):
        session = _make_mock_session()
        result = plugin.validate_keyword_for_session(
            session, "click element", "SeleniumLibrary"
        )
        assert result is not None
        assert result["alternative"] == "Pointer Click"

    def test_platynui_in_imported_triggers_validation(self, plugin):
        session = _make_mock_session(
            imported_libraries=["PlatynUI"],
            explicit_library_preference="",
        )
        result = plugin.validate_keyword_for_session(
            session, "click element", "Browser"
        )
        assert result is not None
        assert result["success"] is False

    def test_no_source_library_returns_none(self, plugin):
        session = _make_mock_session()
        result = plugin.validate_keyword_for_session(
            session, "some keyword", None
        )
        assert result is None


# ===========================================================================
# generate_failure_hints tests
# ===========================================================================

class TestGenerateFailureHints:
    """Tests for generate_failure_hints()."""

    def test_no_results_error_gives_element_not_found_hint(self, plugin):
        session = _make_mock_session()
        hints = plugin.generate_failure_hints(
            session, "Query", ["/Window/Button"], "Query returned no results"
        )
        assert len(hints) == 1
        assert "Element not found" in hints[0]["title"]

    def test_no_module_error_gives_native_backend_hint(self, plugin):
        session = _make_mock_session()
        hints = plugin.generate_failure_hints(
            session, "Pointer Click", ["/Window/Button"],
            "No module named 'platynui_native'"
        )
        assert len(hints) == 1
        assert "Native backend" in hints[0]["title"]

    def test_attribute_not_found_error_gives_hint(self, plugin):
        session = _make_mock_session()
        hints = plugin.generate_failure_hints(
            session, "Get Attribute", ["/Window/Text", "FooBar"],
            "AttributeNotFoundError"
        )
        assert len(hints) == 1
        assert "Attribute not found" in hints[0]["title"]
        assert "FooBar" in hints[0]["message"]

    def test_unknown_error_gives_empty_hints(self, plugin):
        session = _make_mock_session()
        hints = plugin.generate_failure_hints(
            session, "Pointer Click", ["/Window/Button"],
            "Some unrecognized error message"
        )
        assert hints == []

    def test_element_not_found_hint_has_examples(self, plugin):
        session = _make_mock_session()
        hints = plugin.generate_failure_hints(
            session, "Query", ["/Window/Button"],
            "No results found"
        )
        assert "examples" in hints[0]
        assert len(hints[0]["examples"]) > 0

    def test_attribute_hint_with_missing_second_arg(self, plugin):
        session = _make_mock_session()
        hints = plugin.generate_failure_hints(
            session, "Get Attribute", ["/Window/Text"],
            "attributenotfounderror"
        )
        assert len(hints) == 1
        assert "<unknown>" in hints[0]["message"]

    def test_empty_error_text_gives_empty_hints(self, plugin):
        session = _make_mock_session()
        hints = plugin.generate_failure_hints(
            session, "Query", ["/Window"], ""
        )
        assert hints == []

    def test_none_error_text_gives_empty_hints(self, plugin):
        session = _make_mock_session()
        hints = plugin.generate_failure_hints(
            session, "Query", ["/Window"], None
        )
        assert hints == []


# ===========================================================================
# get_keyword_alternatives tests
# ===========================================================================

class TestGetKeywordAlternatives:
    """Tests for get_keyword_alternatives()."""

    def test_returns_dict(self, plugin):
        alts = plugin.get_keyword_alternatives()
        assert isinstance(alts, dict)

    def test_click_element_maps_to_pointer_click(self, plugin):
        alts = plugin.get_keyword_alternatives()
        assert alts["click element"]["alternative"] == "Pointer Click"

    def test_input_text_maps_to_keyboard_type(self, plugin):
        alts = plugin.get_keyword_alternatives()
        assert alts["input text"]["alternative"] == "Keyboard Type"

    def test_open_browser_maps_to_activate(self, plugin):
        alts = plugin.get_keyword_alternatives()
        assert alts["open browser"]["alternative"] == "Activate"

    def test_get_page_source_maps_to_query(self, plugin):
        alts = plugin.get_keyword_alternatives()
        assert alts["get page source"]["alternative"] == "Query"

    @pytest.mark.parametrize("web_keyword", [
        "click element",
        "click",
        "input text",
        "fill text",
        "open browser",
        "get page source",
        "get text",
        "mouse over",
        "set focus to element",
        "capture page screenshot",
        "close browser",
    ])
    def test_each_alternative_has_required_keys(self, plugin, web_keyword):
        alts = plugin.get_keyword_alternatives()
        assert web_keyword in alts, f"Missing alternative for '{web_keyword}'"
        entry = alts[web_keyword]
        assert "alternative" in entry
        assert "explanation" in entry
        assert "example" in entry


# ===========================================================================
# State provider tests
# ===========================================================================

class TestPlatynUIStateProvider:
    """Tests for PlatynUIStateProvider."""

    @pytest.mark.asyncio
    async def test_get_page_source_returns_none(self, state_provider):
        session = _make_mock_session()
        result = await state_provider.get_page_source(session)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_application_state_returns_dict(self, state_provider):
        session = _make_mock_session()
        result = await state_provider.get_application_state(session)
        assert isinstance(result, dict)
        assert result["provider"] == "PlatynUI"

    @pytest.mark.asyncio
    async def test_get_ui_tree_returns_stub(self, state_provider):
        session = _make_mock_session()
        result = await state_provider.get_ui_tree(session)
        assert result["success"] is False
        assert "native backend" in result["note"].lower()

    @pytest.mark.asyncio
    async def test_plugin_get_state_provider(self, plugin):
        provider = plugin.get_state_provider()
        assert isinstance(provider, PlatynUIStateProvider)

    @pytest.mark.asyncio
    async def test_get_page_source_via_plugin_provider(self, plugin):
        provider = plugin.get_state_provider()
        session = _make_mock_session()
        result = await provider.get_page_source(session)
        assert result is None


# ===========================================================================
# Plugin registration tests
# ===========================================================================

class TestPluginRegistration:
    """Tests for PlatynUI in the global plugin registry."""

    def test_platynui_in_generate_builtin_plugins(self):
        plugins = generate_builtin_plugins()
        names = [p.metadata.name for p in plugins]
        assert "PlatynUI" in names

    def test_can_find_plugin_by_name(self):
        plugins = generate_builtin_plugins()
        platynui_plugins = [p for p in plugins if p.metadata.name == "PlatynUI"]
        assert len(platynui_plugins) == 1
        assert isinstance(platynui_plugins[0], PlatynUIPlugin)

    def test_get_incompatible_libraries_empty(self, plugin):
        assert plugin.get_incompatible_libraries() == []

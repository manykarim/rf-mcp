"""Unit tests for PlatynUI locator guidance (ADR-025).

Covers RobotFrameworkNativeConverter.get_platynui_locator_guidance:
- always returns namespaces/performance_rules/platform_notes keys.
- 'element not found' adds element_not_found_suggestions.
- 'timed out' adds timeout_suggestions.

Run with: uv run pytest tests/unit/test_platynui_newcore_guidance.py -q
"""

__test__ = True

import pytest

from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter


@pytest.fixture
def converter():
    return RobotFrameworkNativeConverter()


# =============================================================================
# Base structure
# =============================================================================


class TestBaseStructure:
    def test_has_required_keys(self, converter):
        guidance = converter.get_platynui_locator_guidance()
        for key in ("namespaces", "performance_rules", "platform_notes"):
            assert key in guidance

    def test_namespaces_include_core(self, converter):
        guidance = converter.get_platynui_locator_guidance()
        ns = guidance["namespaces"]
        for n in ("app", "control", "item", "native"):
            assert n in ns

    def test_performance_rules_is_nonempty_list(self, converter):
        guidance = converter.get_platynui_locator_guidance()
        assert isinstance(guidance["performance_rules"], list)
        assert guidance["performance_rules"]

    def test_platform_notes_linux_mentions_frame(self, converter):
        guidance = converter.get_platynui_locator_guidance()
        assert "Frame" in guidance["platform_notes"]["linux"]

    def test_no_error_means_no_suggestions(self, converter):
        guidance = converter.get_platynui_locator_guidance()
        assert "element_not_found_suggestions" not in guidance
        assert "timeout_suggestions" not in guidance


# =============================================================================
# Error-message specific suggestions
# =============================================================================


class TestErrorSpecific:
    def test_element_not_found_adds_suggestions(self, converter):
        guidance = converter.get_platynui_locator_guidance(
            error_message="ElementNotFoundError: element not found"
        )
        assert "element_not_found_suggestions" in guidance
        assert guidance["element_not_found_suggestions"]

    def test_no_nodes_adds_suggestions(self, converter):
        guidance = converter.get_platynui_locator_guidance(
            error_message="Query returned no nodes"
        )
        assert "element_not_found_suggestions" in guidance

    def test_timed_out_adds_timeout_suggestions(self, converter):
        guidance = converter.get_platynui_locator_guidance(
            error_message="Operation timed out after 30s"
        )
        assert "timeout_suggestions" in guidance
        assert guidance["timeout_suggestions"]

    def test_timeout_keyword_adds_timeout_suggestions(self, converter):
        guidance = converter.get_platynui_locator_guidance(
            error_message="timeout while resolving descriptor"
        )
        assert "timeout_suggestions" in guidance

    def test_import_error_adds_installation_guidance(self, converter):
        guidance = converter.get_platynui_locator_guidance(
            error_message="ImportError: cannot import WindowSurface"
        )
        assert "installation_guidance" in guidance

    def test_unrelated_error_keeps_base_only(self, converter):
        guidance = converter.get_platynui_locator_guidance(
            error_message="some random failure"
        )
        assert "element_not_found_suggestions" not in guidance
        assert "timeout_suggestions" not in guidance
        # base keys still present
        assert "namespaces" in guidance

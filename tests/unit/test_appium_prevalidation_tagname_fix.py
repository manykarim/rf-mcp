"""Regression tests for Appium pre-validation tag_name comparison bug.

Issue: docs/issues/robotmcp-prevalidation-appium-analysis.md

`element.tag_name` on Android returns the full Java class name (e.g.
``android.widget.EditText``), and on iOS it returns the XCUI element type
(e.g. ``XCUIElementTypeTextField``). The previous implementation compared
the lower-cased ``tag_name`` against the tuple
``("edittext", "textfield", "input", "textarea")`` using exact membership
(``in`` on a tuple), which never matched these realistic mobile values.

The result: ``Input Text`` / ``Input Value`` / ``Clear Text`` failed
pre-validation with "missing required state: editable", even though the
underlying Appium element was perfectly editable.

These tests pin the new substring-based detection so the bug cannot regress.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from robotmcp.components.execution.keyword_executor import KeywordExecutor
from robotmcp.models.config_models import ExecutionConfig


@pytest.fixture
def executor() -> KeywordExecutor:
    """Create a KeywordExecutor with test configuration."""
    config = ExecutionConfig()
    return KeywordExecutor(config=config)


def _make_element(tag_name: str, *, displayed: bool = True, enabled: bool = True,
                  class_attr: str | None = None) -> MagicMock:
    """Build a mock Appium WebElement with the given tag/class metadata."""
    element = MagicMock()
    element.tag_name = tag_name
    element.is_displayed.return_value = displayed
    element.is_enabled.return_value = enabled

    def get_attribute(name: str) -> str | None:
        # Appium's get_attribute is invoked as a fallback path when tag_name
        # alone cannot identify an editable element (e.g. React Native wraps).
        if name == "className":
            return class_attr
        return None

    if class_attr is not None:
        element.get_attribute.side_effect = get_attribute
    else:
        # No className attribute available; simulate Appium returning None.
        element.get_attribute.return_value = None
    return element


class TestAppiumEditableDetectionRealistic:
    """Verify editable detection for realistic Android/iOS ``tag_name`` values."""

    @pytest.mark.parametrize(
        "tag_name",
        [
            # Android: full Java class names returned by Appium for editable widgets.
            "android.widget.EditText",
            "android.widget.AutoCompleteTextView",
            "androidx.appcompat.widget.AppCompatEditText",
            # iOS: XCUI element types from XCUITest driver.
            "XCUIElementTypeTextField",
            "XCUIElementTypeSecureTextField",
            "XCUIElementTypeSearchField",
            # Web fallback / WebView contexts.
            "input",
            "textarea",
            # Lower-cased forms (Robot mobile drivers sometimes lowercase).
            "edittext",
            "textfield",
        ],
    )
    def test_editable_detected_for_mobile_tag_names(self, executor, tag_name):
        """Editable state must be inferred from realistic mobile ``tag_name`` values."""
        with patch("robotmcp.components.execution.keyword_executor.BuiltIn") as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            element = _make_element(tag_name)
            mock_builtin.run_keyword.return_value = [element]

            result = executor._run_appium_state_check(
                "accessibility_id=test-Username",
                {"editable", "enabled", "visible"},
                500,
            )

            assert result["valid"] is True, (
                f"tag_name={tag_name!r} should be classified as editable; "
                f"got missing={result['missing']}, error={result['error']}"
            )
            assert "editable" in result["states"]
            assert "enabled" in result["states"]
            assert "visible" in result["states"]

    def test_non_editable_tag_does_not_get_editable_state(self, executor):
        """A button-like element must NOT be tagged as editable."""
        with patch("robotmcp.components.execution.keyword_executor.BuiltIn") as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            element = _make_element("android.widget.Button")
            mock_builtin.run_keyword.return_value = [element]

            result = executor._run_appium_state_check(
                "accessibility_id=test-LOGIN", {"editable"}, 500,
            )

            assert result["valid"] is False
            assert "editable" in result["missing"]

    def test_disabled_edittext_does_not_get_editable_state(self, executor):
        """An EditText that is not enabled must not be reported as editable."""
        with patch("robotmcp.components.execution.keyword_executor.BuiltIn") as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            element = _make_element("android.widget.EditText", enabled=False)
            mock_builtin.run_keyword.return_value = [element]

            result = executor._run_appium_state_check(
                "accessibility_id=test-Username", {"editable"}, 500,
            )

            assert result["valid"] is False
            assert "editable" in result["missing"]

    def test_classname_attribute_fallback_for_react_native_wrapper(self, executor):
        """When ``tag_name`` is a custom wrapper, fall back to ``className`` attr."""
        with patch("robotmcp.components.execution.keyword_executor.BuiltIn") as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            # React Native often surfaces a generic wrapper as the leaf tag
            # while exposing the underlying Android class via the className
            # attribute. The pre-validator must consult that attribute.
            element = _make_element(
                "androidx.compose.ui.platform.ComposeView",
                class_attr="android.widget.EditText",
            )
            mock_builtin.run_keyword.return_value = [element]

            result = executor._run_appium_state_check(
                "accessibility_id=test-Username",
                {"editable", "enabled", "visible"},
                500,
            )

            assert result["valid"] is True, (
                f"Expected className fallback to detect editable; got {result}"
            )
            assert "editable" in result["states"]

    def test_classname_attribute_unavailable_does_not_crash(self, executor):
        """If get_attribute raises, pre-validation must degrade gracefully."""
        with patch("robotmcp.components.execution.keyword_executor.BuiltIn") as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            element = MagicMock()
            element.tag_name = "android.view.ViewGroup"
            element.is_displayed.return_value = True
            element.is_enabled.return_value = True
            element.get_attribute.side_effect = RuntimeError("Stale element reference")

            mock_builtin.run_keyword.return_value = [element]

            result = executor._run_appium_state_check(
                "accessibility_id=test-Username", {"editable"}, 500,
            )

            # ViewGroup is not editable; missing 'editable' must be reported,
            # not a crash from the raised exception in get_attribute.
            assert result["valid"] is False
            assert "editable" in result["missing"]
            assert "error" in result and "Pre-validation error" not in (result["error"] or "")


class TestAppiumNonTextControlTypesAudit:
    """Audit non-text Appium control types to confirm no analogous bug exists.

    Pre-validation injects only ``attached``/``visible``/``enabled``/``editable``
    into the state set. ``visible`` and ``enabled`` are derived from
    ``element.is_displayed()`` / ``element.is_enabled()`` — driver-evaluated per
    element, with no tag-name introspection. So click/tap/check/select on
    buttons, image views, view groups, check boxes, switches, list items, etc.
    must always pass pre-validation when the element is displayed and enabled,
    regardless of the platform-specific tag_name string.
    """

    @pytest.fixture
    def executor(self) -> KeywordExecutor:
        config = ExecutionConfig()
        return KeywordExecutor(config=config)

    @pytest.mark.parametrize(
        "tag_name",
        [
            # Android control types commonly targeted by Click/Tap/Check
            "android.widget.Button",
            "android.widget.ImageButton",
            "android.widget.ImageView",
            "android.widget.CheckBox",
            "android.widget.RadioButton",
            "android.widget.Switch",
            "android.widget.ToggleButton",
            "android.widget.Spinner",
            "android.widget.TextView",        # often clickable
            "android.view.ViewGroup",         # SauceLabsDemo: test-ADD TO CART
            "android.view.View",
            "androidx.recyclerview.widget.RecyclerView",
            # iOS XCUI element types commonly targeted by Click/Tap
            "XCUIElementTypeButton",
            "XCUIElementTypeStaticText",
            "XCUIElementTypeImage",
            "XCUIElementTypeSwitch",
            "XCUIElementTypeCell",
            "XCUIElementTypeNavigationBar",
            "XCUIElementTypeTabBar",
            "XCUIElementTypePicker",
            "XCUIElementTypePickerWheel",
        ],
    )
    def test_non_text_controls_pass_click_prevalidation(self, executor, tag_name):
        """visible+enabled must be derivable for any control type without
        the tag_name participating in the decision."""
        with patch("robotmcp.components.execution.keyword_executor.BuiltIn") as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            element = _make_element(tag_name)
            mock_builtin.run_keyword.return_value = [element]

            result = executor._run_appium_state_check(
                f"class={tag_name}", {"visible", "enabled"}, 500,
            )

            assert result["valid"] is True, (
                f"Click pre-validation must pass for {tag_name!r}; got {result}"
            )
            assert "visible" in result["states"]
            assert "enabled" in result["states"]

    @pytest.mark.parametrize(
        "tag_name",
        [
            "android.widget.Button",
            "android.widget.ImageButton",
            "android.widget.CheckBox",
            "android.widget.RadioButton",
            "android.widget.Switch",
            "android.widget.Spinner",
            "android.view.ViewGroup",
            "android.view.View",
            "XCUIElementTypeButton",
            "XCUIElementTypeStaticText",
            "XCUIElementTypeImage",
            "XCUIElementTypeSwitch",
            "XCUIElementTypeCell",
            "XCUIElementTypePicker",
            "XCUIElementTypePickerWheel",
        ],
    )
    def test_non_text_controls_not_promoted_to_editable(self, executor, tag_name):
        """Defensive: non-text control types must NOT be flagged editable.

        Otherwise ``Input Text`` would silently target a Button or PickerWheel
        and produce confusing failures further down the stack.
        """
        with patch("robotmcp.components.execution.keyword_executor.BuiltIn") as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            element = _make_element(tag_name)
            mock_builtin.run_keyword.return_value = [element]

            result = executor._run_appium_state_check(
                f"class={tag_name}", {"editable"}, 500,
            )

            assert result["valid"] is False, (
                f"{tag_name!r} should NOT be classified as editable; got {result}"
            )
            assert "editable" in result["missing"]


class TestAppiumAdditionalEditableTypes:
    """Cover additional editable widget types beyond the basic EditText set.

    These are real classes seen in the wild that should also be recognised:

    * Android: ``MultiAutoCompleteTextView`` (matched via ``textview``),
      ``EditTextPreference`` (matched via ``edittext``).
    * iOS: ``XCUIElementTypeTextView`` (multi-line, matched via ``textview``).
    """

    @pytest.fixture
    def executor(self) -> KeywordExecutor:
        config = ExecutionConfig()
        return KeywordExecutor(config=config)

    @pytest.mark.parametrize(
        "tag_name",
        [
            "android.widget.MultiAutoCompleteTextView",
            "androidx.preference.EditTextPreference",
            "XCUIElementTypeTextView",  # iOS multi-line text view
        ],
    )
    def test_additional_editable_widget_types(self, executor, tag_name):
        with patch("robotmcp.components.execution.keyword_executor.BuiltIn") as MockBuiltIn:
            mock_builtin = MagicMock()
            MockBuiltIn.return_value = mock_builtin

            element = _make_element(tag_name)
            mock_builtin.run_keyword.return_value = [element]

            result = executor._run_appium_state_check(
                f"class={tag_name}", {"editable", "enabled", "visible"}, 500,
            )

            assert result["valid"] is True, (
                f"Editable promotion missing for {tag_name!r}; got {result}"
            )
            assert "editable" in result["states"]

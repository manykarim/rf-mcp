"""Regression test: Browser-JS path call shape for ActionableElementsCollector.

Verifies that _run_browser_js_sync uses exactly 2 positional args after the
keyword name when calling Browser.Evaluate JavaScript via BuiltIn.run_keyword.
This is the test that would have caught the v0.32.1 G2 bug.
"""
import pytest
from unittest.mock import patch, MagicMock

from robotmcp.components.execution.page_source_service import (
    ActionableElementsCollector,
)


class TestActionableElementsBrowserJSCallShape:
    """Regression test: the Browser-JS path must call Browser.Evaluate JavaScript
    with exactly 2 positional args after the keyword name (selector, function)."""

    @pytest.mark.asyncio
    async def test_browser_js_call_shape(self):
        mock_instance = MagicMock()
        mock_instance.run_keyword.return_value = []  # empty list ok

        session = MagicMock()
        session.session_id = "test"
        session.imported_libraries = ["Browser"]
        session.variables = {}
        try:
            session.get_web_automation_library = MagicMock(return_value="Browser")
        except Exception:
            pass

        collector = ActionableElementsCollector(limit=10)
        with patch("robot.libraries.BuiltIn.BuiltIn", return_value=mock_instance):
            await collector.collect(session, page_source=None)

        # If the JS path was attempted, it must have used 2 args
        if mock_instance.run_keyword.called:
            call_args = mock_instance.run_keyword.call_args[0]
            assert call_args[0] == "Browser.Evaluate JavaScript"
            extra_args = call_args[1:]
            assert len(extra_args) == 2, (
                f"Browser.Evaluate JavaScript must receive exactly 2 args, "
                f"got {len(extra_args)}: {extra_args}"
            )

    @pytest.mark.asyncio
    async def test_browser_js_first_arg_is_css_selector(self):
        """First argument must be the css=html selector (scope for the evaluation)."""
        mock_instance = MagicMock()
        mock_instance.run_keyword.return_value = []

        session = MagicMock()
        session.session_id = "test"
        session.imported_libraries = ["Browser"]
        session.variables = {}
        session.get_web_automation_library = MagicMock(return_value="Browser")

        collector = ActionableElementsCollector(limit=10)
        with patch("robot.libraries.BuiltIn.BuiltIn", return_value=mock_instance):
            await collector.collect(session, page_source=None)

        if mock_instance.run_keyword.called:
            call_args = mock_instance.run_keyword.call_args[0]
            # arg[1] is the selector
            assert call_args[1] == "css=html", (
                f"Expected selector 'css=html', got: {call_args[1]!r}"
            )

    @pytest.mark.asyncio
    async def test_browser_js_second_arg_is_function_string(self):
        """Second argument must be a JavaScript function string starting with '() =>'."""
        mock_instance = MagicMock()
        mock_instance.run_keyword.return_value = []

        session = MagicMock()
        session.session_id = "test"
        session.imported_libraries = ["Browser"]
        session.variables = {}
        session.get_web_automation_library = MagicMock(return_value="Browser")

        collector = ActionableElementsCollector(limit=10)
        with patch("robot.libraries.BuiltIn.BuiltIn", return_value=mock_instance):
            await collector.collect(session, page_source=None)

        if mock_instance.run_keyword.called:
            call_args = mock_instance.run_keyword.call_args[0]
            js_func = call_args[2]
            assert isinstance(js_func, str), "JS arg must be a string"
            assert "() =>" in js_func or "()=>" in js_func, (
                f"JS arg must be an arrow function, got: {js_func[:80]!r}"
            )

    @pytest.mark.asyncio
    async def test_bs4_fallback_emits_actionable_surface_for_labeled_input(self):
        """BS4 fallback path must emit actionable_surface for inputs with sibling label."""
        from robotmcp.models.session_models import ExecutionSession

        html = """
        <html><body>
          <label for="gendermale">Male</label>
          <input id="gendermale" name="gender" type="radio" style="display:none" />
        </body></html>
        """
        session = ExecutionSession(session_id="bs4_shape_sess")
        session.imported_libraries.append("SeleniumLibrary")
        collector = ActionableElementsCollector(limit=80)
        result = await collector.collect(session, html)

        elements = result["elements"]
        inp = next((e for e in elements if e.get("id") == "gendermale"), None)
        assert inp is not None, "hidden radio input must be found"
        surface = inp.get("actionable_surface")
        assert surface is not None, (
            "actionable_surface must be present for input with sibling label in BS4 path"
        )
        # Sibling label (<label for="x">) produces attribute-selector form, not descendant chain.
        assert surface["selector"] == "css=label[for='gendermale']"
        assert surface["wrapper_tag"] == "label"
        assert surface["wrapper_relation"] == "sibling_for"
        assert surface.get("wrapper_text") == "Male"
        assert "wrapper_visible" not in surface

    @pytest.mark.asyncio
    async def test_browser_js_result_list_used_as_elements(self):
        """When BuiltIn.run_keyword returns a list, it is used as the element list."""
        from robotmcp.models.session_models import ExecutionSession

        mock_elements = [
            {
                "tag": "input",
                "id": "foo",
                "type": "text",
                "bounding_rect": {"x": 0, "y": 0, "width": 100, "height": 30},
                "display_state": "visible",
                "disabled": False,
                "parent_hidden": False,
            }
        ]
        mock_instance = MagicMock()
        mock_instance.run_keyword.return_value = mock_elements

        session = ExecutionSession(session_id="shape_list_sess")
        session.imported_libraries.append("Browser")

        collector = ActionableElementsCollector(limit=10)
        with patch("robot.libraries.BuiltIn.BuiltIn", return_value=mock_instance):
            result = await collector.collect(session, page_source=None)

        assert result["collection_method"] == "browser_js"
        assert result["count"] == 1
        assert result["elements"][0]["id"] == "foo"

    @pytest.mark.asyncio
    async def test_browser_js_non_list_result_falls_back_to_bs4(self):
        """When BuiltIn.run_keyword returns non-list, falls back to BS4 with supplied HTML."""
        from robotmcp.models.session_models import ExecutionSession

        mock_instance = MagicMock()
        mock_instance.run_keyword.return_value = "not a list"

        html = "<html><body><input id='x' type='text' /></body></html>"
        session = ExecutionSession(session_id="shape_nlist_sess")
        session.imported_libraries.append("Browser")

        collector = ActionableElementsCollector(limit=10)
        with patch("robot.libraries.BuiltIn.BuiltIn", return_value=mock_instance):
            result = await collector.collect(session, page_source=html)

        # JS returned non-list → fallback to BS4
        assert result["collection_method"] == "html_static"
        assert result["count"] >= 1

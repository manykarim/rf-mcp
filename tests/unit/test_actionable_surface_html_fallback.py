"""G2: actionable_surface - BS4 static HTML fallback path tests.

These tests exercise ActionableElementsCollector._collect_from_html which
runs without a live browser. Geometry (bounding_rect) and wrapper_visible
are not available on this path; actionable_surface is best-effort via
<label for="id"> or wrapping <label> ancestor.
"""

import json

import pytest

from robotmcp.components.execution.page_source_service import ActionableElementsCollector
from robotmcp.models.session_models import ExecutionSession


async def _collect_html(html: str, limit: int = 80) -> list:
    """Run the static HTML collector and return the cleaned element list."""
    # Use a SeleniumLibrary session so the Browser JS path is not attempted.
    session = ExecutionSession(session_id="html_fb_sess")
    session.imported_libraries.append("SeleniumLibrary")
    collector = ActionableElementsCollector(limit=limit)
    result = await collector.collect(session, html)
    return result["elements"]


class TestActionableSurfaceHtmlFallback:

    @pytest.mark.asyncio
    async def test_sibling_label_for_hidden_input_produces_sibling_selector(self):
        """<label for="x"> sibling pattern (Bootstrap form-check style)
        must produce the attribute-selector form, NOT the descendant chain.
        """
        html = """
        <html><body>
          <label for="gendermale">Male</label>
          <input id="gendermale" name="gender" type="radio" style="display:none" />
        </body></html>
        """
        elements = await _collect_html(html)
        inp = next((e for e in elements if e.get("id") == "gendermale"), None)
        assert inp is not None
        assert inp["display_state"] == "display:none"
        surface = inp.get("actionable_surface")
        assert surface is not None, "actionable_surface must exist for hidden input with label"
        assert surface["selector"] == "css=label[for='gendermale']"
        assert surface["wrapper_tag"] == "label"
        assert surface["wrapper_relation"] == "sibling_for"
        assert surface.get("wrapper_text") == "Male"
        # wrapper_visible is NOT set on the static path (no geometry)
        assert "wrapper_visible" not in surface

    @pytest.mark.asyncio
    async def test_wrapping_label_ancestor_produces_descendant_selector(self):
        """Input nested INSIDE a <label> element produces the descendant chain."""
        html = """
        <html><body>
          <label>
            Female
            <input id="genderfemale" name="gender" type="radio" style="visibility:hidden" />
          </label>
        </body></html>
        """
        elements = await _collect_html(html)
        inp = next((e for e in elements if e.get("id") == "genderfemale"), None)
        assert inp is not None
        assert inp["display_state"] == "display:none"
        surface = inp.get("actionable_surface")
        assert surface is not None
        assert surface["selector"] == "*css=label >> id=genderfemale"
        assert surface["wrapper_tag"] == "label"
        assert surface["wrapper_relation"] == "ancestor"

    @pytest.mark.asyncio
    async def test_visible_input_with_sibling_label_gets_sibling_selector(self):
        """Visible inputs with sibling <label> get the attribute-form selector.

        BS4 cannot detect computed CSS visibility, so an element appearing 'visible'
        due to lack of inline style may still be CSS-hidden at runtime.
        The wrapper-locator pattern is correct in either case.
        """
        html = """
        <html><body>
          <label for="email">Email</label>
          <input id="email" type="text" name="email" />
        </body></html>
        """
        elements = await _collect_html(html)
        inp = next((e for e in elements if e.get("id") == "email"), None)
        assert inp is not None
        assert inp["display_state"] == "visible"
        surface = inp.get("actionable_surface")
        assert surface is not None
        assert surface["selector"] == "css=label[for='email']"
        assert surface["wrapper_tag"] == "label"
        assert surface["wrapper_relation"] == "sibling_for"

    @pytest.mark.asyncio
    async def test_hidden_input_without_label_no_surface(self):
        """Hidden input with no label should produce no actionable_surface."""
        html = """
        <html><body>
          <input id="csrf" name="csrf_token" type="hidden" style="display:none" />
        </body></html>
        """
        elements = await _collect_html(html)
        inp = next((e for e in elements if e.get("id") == "csrf"), None)
        assert inp is not None
        assert inp["display_state"] == "display:none"
        assert "actionable_surface" not in inp

    @pytest.mark.asyncio
    async def test_hidden_input_without_id_no_surface(self):
        """Hidden input with no id cannot form a scoped selector; no surface expected."""
        html = """
        <html><body>
          <label for="">Pick one</label>
          <input name="choice" type="radio" style="display:none" />
        </body></html>
        """
        elements = await _collect_html(html)
        radio = next((e for e in elements if e.get("name") == "choice"), None)
        assert radio is not None
        assert "actionable_surface" not in radio

    @pytest.mark.asyncio
    async def test_multiple_radio_inputs_each_get_their_label(self):
        """Each hidden radio in an idealForms-style group should map to its own label."""
        html = """
        <html><body>
          <label for="opt_a">Option A</label>
          <input id="opt_a" name="choice" type="radio" style="display:none" />
          <label for="opt_b">Option B</label>
          <input id="opt_b" name="choice" type="radio" style="display:none" />
        </body></html>
        """
        elements = await _collect_html(html)
        opt_a = next((e for e in elements if e.get("id") == "opt_a"), None)
        opt_b = next((e for e in elements if e.get("id") == "opt_b"), None)
        assert opt_a is not None and opt_b is not None

        # These inputs use the sibling-label pattern (<label for="x"> + <input id="x">).
        assert opt_a["actionable_surface"]["selector"] == "css=label[for='opt_a']"
        assert opt_a["actionable_surface"]["wrapper_relation"] == "sibling_for"
        assert opt_a["actionable_surface"]["wrapper_text"] == "Option A"
        assert opt_b["actionable_surface"]["selector"] == "css=label[for='opt_b']"
        assert opt_b["actionable_surface"]["wrapper_relation"] == "sibling_for"
        assert opt_b["actionable_surface"]["wrapper_text"] == "Option B"

    @pytest.mark.asyncio
    async def test_token_budget_30_hidden_inputs_with_labels(self):
        """30 hidden inputs each with a label should stay under 2500 tokens."""
        inputs_html = "".join(
            f'<label for="f{i}">Option {i}</label>'
            f'<input id="f{i}" name="choice" type="radio" style="display:none" />'
            for i in range(30)
        )
        html = f"<html><body>{inputs_html}</body></html>"
        elements = await _collect_html(html, limit=80)
        hidden = [e for e in elements if e.get("display_state") == "display:none"]
        assert len(hidden) == 30

        payload = {
            "success": True,
            "session_id": "demo",
            "data": {"actionable_elements": {"elements": elements, "count": len(elements)}},
        }
        token_estimate = max(1, len(json.dumps(payload)) // 4)
        assert token_estimate < 2500, (
            f"30-element HTML fallback payload too large: {token_estimate} tokens"
        )

    @pytest.mark.asyncio
    async def test_collection_method_is_html_static(self):
        """Ensure the static path is correctly reported in collection_method."""
        html = "<html><body><input id='x' type='text' /></body></html>"
        session = ExecutionSession(session_id="method_sess")
        session.imported_libraries.append("SeleniumLibrary")
        collector = ActionableElementsCollector(limit=80)
        result = await collector.collect(session, html)
        assert result["collection_method"] == "html_static"
        assert "note" in result

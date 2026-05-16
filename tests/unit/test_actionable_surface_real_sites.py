"""Smoke tests for ActionableElementsCollector against real public-site fixtures.

These tests pin behaviour against three real-world checkbox patterns so future
edits to the BS4 fallback path cannot silently regress cross-site correctness:

  1. expandtesting_checkboxes.html — Bootstrap form-check: <input> + <label for=>.
     The label is a SIBLING, not an ancestor. Correct selector:
        css=label[for='<id>']
     Wrong (Tricentis-style descendant chain) would be:
        *css=label >> id=<id>

  2. herokuapp_checkboxes.html — bare <input> with no id and no label.
     No actionable_surface should be emitted.

  3. tricentis_wrapper_label.html — input WRAPPED inside <label>.
     Correct selector is the descendant chain "*css=label >> id=<id>".
"""

__test__ = True

from pathlib import Path

import pytest

from robotmcp.components.execution.page_source_service import (
    ActionableElementsCollector,
)


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _read(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


class TestSiblingLabelPattern:
    """Bootstrap form-check pattern: <input id=foo>+<label for=foo>."""

    def test_emits_sibling_for_selector(self) -> None:
        html = _read("expandtesting_checkboxes.html")
        elements = ActionableElementsCollector()._collect_from_html(html)
        checkbox1 = next(e for e in elements if e.get("id") == "checkbox1")
        surface = checkbox1.get("actionable_surface")
        assert surface is not None, "Expected actionable_surface for sibling-label input"
        assert surface["selector"] == "css=label[for='checkbox1']"
        assert surface["wrapper_tag"] == "label"
        assert surface["wrapper_relation"] == "sibling_for"
        assert surface["wrapper_text"] == "Checkbox 1"

    def test_does_not_emit_descendant_chain_for_sibling(self) -> None:
        """Regression: the sibling pattern must NOT produce the wrapping selector."""
        html = _read("expandtesting_checkboxes.html")
        elements = ActionableElementsCollector()._collect_from_html(html)
        for el in elements:
            surface = el.get("actionable_surface")
            if surface is None:
                continue
            assert ">> id=" not in surface["selector"], (
                f"Sibling-label pattern produced wrapping descendant selector: "
                f"{surface['selector']}"
            )


class TestBareInputPattern:
    """Bare <input> followed by text, no <label> at all."""

    def test_no_surface_emitted(self) -> None:
        html = _read("herokuapp_checkboxes.html")
        elements = ActionableElementsCollector()._collect_from_html(html)
        # Both inputs lack id and lack any label association.
        assert len(elements) == 2
        for el in elements:
            assert "actionable_surface" not in el, (
                "Bare input must not produce an actionable_surface"
            )


class TestWrappingLabelPattern:
    """Tricentis-style: <input> wrapped INSIDE a <label>."""

    def test_emits_descendant_chain_selector(self) -> None:
        html = _read("tricentis_wrapper_label.html")
        elements = ActionableElementsCollector()._collect_from_html(html)
        male = next(e for e in elements if e.get("id") == "gendermale")
        surface = male.get("actionable_surface")
        assert surface is not None
        assert surface["selector"] == "*css=label >> id=gendermale"
        assert surface["wrapper_tag"] == "label"
        assert surface["wrapper_relation"] == "ancestor"
        assert "Male" in (surface.get("wrapper_text") or "")

    def test_relation_field_differentiates_from_sibling(self) -> None:
        """Both patterns target labels, but wrapper_relation must distinguish them."""
        wrap_html = _read("tricentis_wrapper_label.html")
        sib_html = _read("expandtesting_checkboxes.html")

        wrap_surface = next(
            e["actionable_surface"]
            for e in ActionableElementsCollector()._collect_from_html(wrap_html)
            if e.get("id") == "gendermale"
        )
        sib_surface = next(
            e["actionable_surface"]
            for e in ActionableElementsCollector()._collect_from_html(sib_html)
            if e.get("id") == "checkbox1"
        )
        assert wrap_surface["wrapper_relation"] == "ancestor"
        assert sib_surface["wrapper_relation"] == "sibling_for"
        assert wrap_surface["selector"] != sib_surface["selector"]

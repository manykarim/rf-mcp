"""Proposal-B: auto-curation of `Evaluate JavaScript` / `Execute Javascript`.

F-N12 covered read-only keywords by name. The Tricentis multi-agent
diagnostic showed Evaluate JavaScript probes (getBoundingClientRect,
getComputedStyle, .innerHTML reads, etc.) dominate suite bloat — up to
130 probes in a single 162-step generated suite. This module's tests
pin the classifier behaviour against representative real-world JS bodies
extracted from the diagnostic artefacts.

Classifier contract:
- Mutation patterns (.value = X, .click(), dispatchEvent, etc.) -> recorded
- Read-only patterns (getBoundingClientRect, .innerHTML read, etc.) -> NOT recorded
- Unclassified -> recorded (conservative default)
"""

from __future__ import annotations

__test__ = True

import pytest

from robotmcp.components.execution.keyword_executor import (
    _classify_evaluate_javascript,
)


# ---------------------------------------------------------------------------
# Read-only JS bodies — collected from real artefacts
# ---------------------------------------------------------------------------

READONLY_JS_BODIES = [
    # From art_81870423004a.txt — geometry probes
    "elem => JSON.stringify(elem.getBoundingClientRect())",
    "elem => { const style = window.getComputedStyle(elem); return {display: style.display, visibility: style.visibility, opacity: style.opacity, boundingRect: JSON.stringify(elem.getBoundingClientRect())} }",
    "() => { return {innerWidth: window.innerWidth, innerHeight: window.innerHeight, bodyScrollWidth: document.body.scrollWidth, bodyScrollHeight: document.body.scrollHeight} }",
    "elem => elem.innerHTML.substring(0, 500)",
    "elem => { const sections = elem.querySelectorAll('.idealsteps-step'); return Array.from(sections).map((s,i) => ({index: i, display: s.style.display})); }",
    "() => document.title",
    "el => el.getAttribute('class')",
    "el => el.value",
    "() => document.readyState",
    "el => el.querySelector('input').checked",
    "() => window.location.href",
]


# ---------------------------------------------------------------------------
# Mutation JS bodies — collected from real artefacts and ADR-022
# ---------------------------------------------------------------------------

MUTATION_JS_BODIES = [
    # From art_81870423004a.txt — agent injecting values
    "elem => { elem.value = '25000'; elem.dispatchEvent(new Event('change', {bubbles: true})); elem.dispatchEvent(new Event('input', {bubbles: true})); return elem.value; }",
    "document.querySelector('#hiddenBtn').click()",
    "el => { el.checked = true; el.dispatchEvent(new Event('change')); }",
    "el => el.classList.add('selected')",
    "el => el.setAttribute('data-confirmed', 'yes')",
    # idealForms commit
    "() => $('#form').idealforms('validate', '#fieldId')",
    "() => $('#form').valid()",
    # navigation
    "() => location.assign('/next-page')",
    "() => history.pushState({}, '', '/path')",
    "() => document.querySelector('#submitBtn').focus()",
    "el => el.scrollIntoView()",
    "() => document.cookie = 'session=abc'",
]


# ---------------------------------------------------------------------------
# Boundary / ambiguous cases
# ---------------------------------------------------------------------------


class TestReadOnlyClassification:
    @pytest.mark.parametrize("js", READONLY_JS_BODIES)
    def test_readonly_js_classifies_as_inspection(self, js):
        # Browser library form: ["selector", "js_body"]
        result = _classify_evaluate_javascript(["css=html", js])
        assert result is False, f"Expected read-only False, got {result} for: {js[:60]}"

    @pytest.mark.parametrize("js", READONLY_JS_BODIES)
    def test_readonly_js_classifies_with_selenium_arg_order(self, js):
        # Selenium-style: ["js_body"] or ["js_body_with_return", "args..."]
        result = _classify_evaluate_javascript([f"return ({js})(arguments[0]);"])
        assert result is False


class TestMutationClassification:
    @pytest.mark.parametrize("js", MUTATION_JS_BODIES)
    def test_mutation_js_classifies_as_recordable(self, js):
        result = _classify_evaluate_javascript(["css=html", js])
        assert result is True, f"Expected mutation True, got {result} for: {js[:60]}"


class TestArgumentShapeHandling:
    def test_locator_only_returns_none(self):
        """Arg list of just a CSS locator has no JS body -> unclassified."""
        result = _classify_evaluate_javascript(["css=html"])
        # css=html does not look like JS; falls back to longest-arg heuristic,
        # which still picks 'css=html'. Neither read-only nor mutation patterns
        # match -> None (default to recorded by caller).
        assert result is None

    def test_no_args_returns_none(self):
        assert _classify_evaluate_javascript([]) is None
        assert _classify_evaluate_javascript(None) is None

    def test_non_string_args_skipped(self):
        result = _classify_evaluate_javascript([1, 2.0, None])
        assert result is None

    def test_mutation_wins_over_readonly_when_both_present(self):
        """A JS body that BOTH reads and writes (e.g. reads input.value then
        sets it) must classify as a mutation."""
        body = "el => { const v = el.value; el.value = v + 'x'; return el.value; }"
        result = _classify_evaluate_javascript(["css=html", body])
        assert result is True


class TestUnclassifiedDefaultBehaviour:
    def test_arbitrary_expression_returns_none(self):
        """A JS body with no recognised mutation or read marker is unclassified."""
        result = _classify_evaluate_javascript(["css=html", "() => 42"])
        assert result is None

    def test_string_concat_only_is_unclassified(self):
        result = _classify_evaluate_javascript(["css=html", "() => 'hello'"])
        assert result is None

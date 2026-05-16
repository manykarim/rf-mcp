"""v0.32.5 P2 — close the document.title=X false-negative in JS classification.

Regression: `document.title = 'X'` was being classified as inspection-only
(``recorded:false``) because the read-only ``document.(title|URL|...)``
pattern matched before any mutation-specific pattern.  Fix: add explicit
``document.{title,domain,location}\\s*=`` to ``_EVAL_JS_MUTATION_PATTERNS``
AND add a ``(?!\\s*=)`` negative lookahead to the read-only pattern.

These tests pin both behaviours and the related document-cookie / domain
cases that are easy to regress.
"""

from __future__ import annotations

import pytest

from robotmcp.components.execution.keyword_executor import (
    _classify_evaluate_javascript,
    _EVAL_JS_MUTATION_PATTERNS,
    _EVAL_JS_READONLY_PATTERNS,
)


class TestDocumentMutationsRecordTrue:
    """document.<prop> = X must classify as load-bearing (record=True)."""

    @pytest.mark.parametrize("js_body,description", [
        ("document.title = 'New Title'", "document.title assignment"),
        ("document.title='New Title'", "document.title assignment no space"),
        ("document.domain = 'example.com'", "document.domain assignment"),
        ("document.location = '/login'", "document.location assignment"),
        ("document.cookie = 'session=abc'", "document.cookie assignment"),
        (
            "() => { document.title = 'Loaded'; return document.readyState; }",
            "mixed mutation+read body still classifies as mutation",
        ),
    ])
    def test_document_property_assignment_records_true(self, js_body, description):
        result = _classify_evaluate_javascript(["css=html", js_body])
        assert result is True, f"{description}: expected True (record), got {result}"


class TestDocumentReadsRecordFalse:
    """Read-only document.<prop> access must classify as inspection (record=False)."""

    @pytest.mark.parametrize("js_body,description", [
        ("() => document.title", "bare document.title read"),
        ("() => document.URL", "document.URL read"),
        ("() => document.readyState", "document.readyState read"),
        ("() => document.documentElement.lang", "documentElement chained read"),
        ("() => document.body.innerHTML", "body.innerHTML read"),
        (
            "() => { const t = document.title; return t.length; }",
            "document.title assigned to local variable (read)",
        ),
    ])
    def test_document_property_read_records_false(self, js_body, description):
        result = _classify_evaluate_javascript(["css=html", js_body])
        assert result is False, f"{description}: expected False (skip), got {result}"


class TestMutationPatternsExplicitCoverage:
    """Confirm the new patterns are actually in the mutation set."""

    def test_document_title_assignment_pattern_present(self):
        import re
        matches = [p.pattern for p in _EVAL_JS_MUTATION_PATTERNS
                   if re.search(p.pattern, "document.title = 'x'")]
        assert matches, "document.title=... pattern missing from mutation set"

    def test_document_domain_assignment_pattern_present(self):
        import re
        matches = [p.pattern for p in _EVAL_JS_MUTATION_PATTERNS
                   if re.search(p.pattern, "document.domain = 'x'")]
        assert matches, "document.domain=... pattern missing from mutation set"

    def test_document_location_assignment_pattern_present(self):
        import re
        matches = [p.pattern for p in _EVAL_JS_MUTATION_PATTERNS
                   if re.search(p.pattern, "document.location = '/x'")]
        assert matches, "document.location=... pattern missing from mutation set"


class TestReadOnlyDocumentPatternNegativeLookahead:
    """The read-only document.<prop> pattern must NOT match against an
    assignment site — otherwise mutations can still leak through if the
    mutation patterns are extended unevenly in the future."""

    @pytest.mark.parametrize("js_body", [
        "document.title = 'x'",
        "document.URL = 'x'",
        "document.readyState = 'x'",   # nonsensical but defensive
        "document.body = null",
    ])
    def test_readonly_document_pattern_does_not_match_assignment(self, js_body):
        # Test the regex directly: it should NOT match when followed by =
        for p in _EVAL_JS_READONLY_PATTERNS:
            if "document" not in p.pattern:
                continue
            assert not p.search(js_body), (
                f"Read-only pattern {p.pattern!r} unexpectedly matched assignment: {js_body!r}"
            )


class TestRegressionsFromPriorRound:
    """Cases reported by the v0.32.x validation agents."""

    def test_sonnet_document_title_assignment_now_records_true(self):
        """Sonnet v0.32.x post-commit run reported `() => { document.title = 'Test'; ... }`
        was incorrectly classified as recorded:false."""
        js_body = "() => { document.title = 'Test'; return document.title; }"
        assert _classify_evaluate_javascript(["css=html", js_body]) is True

    def test_pure_getBoundingClientRect_still_records_false(self):
        """Smoke test: the dominant inspection pattern still works."""
        js_body = "() => document.querySelector('#foo').getBoundingClientRect()"
        assert _classify_evaluate_javascript(["css=html", js_body]) is False

    def test_value_assignment_still_records_true(self):
        """Smoke test: the most common DOM mutation still works."""
        js_body = "() => { document.querySelector('#in').value = 'X'; }"
        assert _classify_evaluate_javascript(["css=html", js_body]) is True

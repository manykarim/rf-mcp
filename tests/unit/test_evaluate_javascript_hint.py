"""OBS-03 — ``Evaluate JavaScript`` argument-shape hint.

Sonnet's transcript from the 2026-05-17 Tricentis Obstacle 3 included a
12-call recovery cycle on the cryptic error:

    Unsupported token '{' while parsing css selector

The Browser library's ``Evaluate JavaScript`` keyword takes
``(selector, expression)``. Calling it with only the JS body causes
that single argument to be parsed as a CSS selector, which then
fails because ``{`` is unambiguously not CSS. The error text says
"parsing css selector" but the real problem is the agent passed
the wrong number of arguments.

These tests pin two layers:
(1) ``_arg_looks_like_javascript`` heuristic — true for JS shapes, false
    for plausible locator strings;
(2) ``_check_evaluate_javascript_misuse`` checker — fires only when the
    keyword is one of the documented evaluate-js aliases AND args length
    is exactly 1 AND that single arg passes the JS-shape heuristic.

Plus a regression check: the broadened ``_check_invalid_selector``
pattern now catches both "Unexpected token" and "Unsupported token"
(Sonnet's actual error wording from Playwright).
"""

from __future__ import annotations

import pytest

from robotmcp.utils.hints import (
    HintContext,
    _arg_looks_like_javascript,
    _check_evaluate_javascript_misuse,
    _check_invalid_selector,
    generate_hints,
)


# ---------------------------------------------------------------------------
# Layer 1: _arg_looks_like_javascript heuristic
# ---------------------------------------------------------------------------


class TestArgLooksLikeJavaScript:
    """The heuristic that decides whether the single argument looks like
    a JS expression. False positives are the primary concern — locator
    strings must NOT be flagged as JS."""

    @pytest.mark.parametrize("js_shape", [
        # Arrow function — bare
        "() => document.title",
        "() => { return 42; }",
        # Arrow function — with parameter
        "(el) => el.value",
        "(node) => node.innerHTML",
        # Function expression
        "function() { return doc.title; }",
        "function(node) { return node.value; }",
        # Body shapes (no leading arrow / function — body contains {)
        "{ return document.title; }",
        "const x = 1; () => x",
        # Body with arrow operator only (no `{`)
        "x => x + 1",
        # Multi-line is fine — only first 200 chars are scanned.
        "const value = document.querySelector('input').value;\n() => value",
    ])
    def test_javascript_shapes_detected(self, js_shape):
        assert _arg_looks_like_javascript(js_shape) is True, (
            f"expected JS-like detection for {js_shape!r}"
        )

    @pytest.mark.parametrize("locator", [
        # Bare CSS / locator prefixes — all should be CLEAN (no JS markers).
        "css=body",
        "css=button.primary",
        "id=submit",
        "name=username",
        "text=Login",
        "xpath=//input[@type='submit']",
        "link=Click here",
        "partial link=Click",
        # Bare CSS shortcuts
        "#submit",
        ".classname",
        # CSS attribute selectors use [ ], NOT { }
        "[data-testid='order-id']",
        # Playwright cascaded — uses >> not { or =>
        "id=item >> nth=2",
        "css=button.primary >> visible=true",
        # Playwright text-inside-CSS
        "button:text('Save')",
        # Empty / None / non-string defensive
        "",
    ])
    def test_locator_strings_not_flagged(self, locator):
        assert _arg_looks_like_javascript(locator) is False, (
            f"false-positive JS detection for legitimate locator {locator!r}"
        )

    @pytest.mark.parametrize("non_string", [None, 42, 3.14, ["a"], {"k": "v"}, True])
    def test_non_string_inputs_return_false(self, non_string):
        # Defensive — must not crash on non-string args. The Browser
        # library normalises arg values upstream, but the executor's
        # hint construction sees arguments as they were passed in.
        assert _arg_looks_like_javascript(non_string) is False


# ---------------------------------------------------------------------------
# Layer 2: _check_evaluate_javascript_misuse checker
# ---------------------------------------------------------------------------


def _ctx(keyword: str, args, error_text: str = "Unsupported token '{' while parsing css selector") -> HintContext:
    return HintContext(
        session_id="t",
        keyword=keyword,
        arguments=list(args),
        error_text=error_text,
    )


class TestCheckEvaluateJavaScriptMisuse:
    """Positive cases: the checker fires for every documented evaluate-js
    alias when the call shape matches the misuse pattern."""

    @pytest.mark.parametrize("keyword_alias", [
        "Evaluate JavaScript",
        "evaluate javascript",
        "EVALUATE JAVASCRIPT",
        "Execute Javascript",      # alternate Browser library name
        "execute javascript",
        "Browser.Evaluate JavaScript",
        "browser.execute javascript",
    ])
    def test_fires_for_every_evaluate_alias(self, keyword_alias):
        ctx = _ctx(keyword_alias, ["() => document.title"])
        hints = _check_evaluate_javascript_misuse(ctx, ctx.error_text)
        assert len(hints) == 1
        # The hint must name BOTH the expected signature and a concrete
        # one-line fix example.
        assert "(selector, expression)" in hints[0].message
        assert any("None" in ex["rf"] for ex in hints[0].examples)

    def test_relevance_outranks_invalid_selector(self):
        # When the same failure trips both checkers, the more-specific
        # JS-shape hint wins on relevance (95 > 90).
        js_hints = _check_evaluate_javascript_misuse(
            _ctx("Evaluate JavaScript", ["() => 1"]),
            "Unsupported token '{' while parsing css selector",
        )
        selector_hints = _check_invalid_selector(
            _ctx("Evaluate JavaScript", ["() => 1"]),
            "Unsupported token '{' while parsing css selector",
        )
        assert js_hints[0].relevance > selector_hints[0].relevance


class TestCheckEvaluateJavaScriptMisuseNegatives:
    """Negative cases: the checker MUST NOT fire when the shape is
    actually correct or the keyword is something else entirely."""

    def test_no_hint_for_two_arg_call(self):
        # Correct shape — selector + expression. No misuse.
        ctx = _ctx("Evaluate JavaScript", ["css=body", "() => document.title"])
        assert _check_evaluate_javascript_misuse(ctx, ctx.error_text) == []

    def test_no_hint_for_zero_args(self):
        ctx = _ctx("Evaluate JavaScript", [])
        assert _check_evaluate_javascript_misuse(ctx, ctx.error_text) == []

    def test_no_hint_when_single_arg_is_a_locator(self):
        # Single arg that IS a plausible locator — the user is passing
        # the selector and forgetting the expression. The error is
        # genuine; we don't have evidence of a JS-shape mismatch.
        ctx = _ctx("Evaluate JavaScript", ["css=body"])
        assert _check_evaluate_javascript_misuse(ctx, ctx.error_text) == []

    @pytest.mark.parametrize("other_keyword", [
        "Click",
        "Get Text",
        "Fill Text",
        "Go To",
        "New Page",
        # Same-shape error but DIFFERENT keyword — must not poison
        # unrelated failures with an evaluate-js hint.
        "Get Element Count",
    ])
    def test_no_hint_for_unrelated_keywords(self, other_keyword):
        ctx = _ctx(other_keyword, ["() => 1"])
        assert _check_evaluate_javascript_misuse(ctx, ctx.error_text) == []


# ---------------------------------------------------------------------------
# Layer 3: end-to-end — generate_hints picks up the new checker
# ---------------------------------------------------------------------------


class TestGenerateHintsEndToEnd:
    """The full ``generate_hints`` pipeline surfaces the new hint as a
    dict in its output. This is what the executor's failure path
    returns to the LLM."""

    def test_misuse_hint_surfaces_in_response(self):
        ctx = HintContext(
            session_id="t",
            keyword="Evaluate JavaScript",
            arguments=["() => document.title"],
            error_text="Unsupported token '{' while parsing css selector",
        )
        hints = generate_hints(ctx)
        # The new hint must be in the top-3-by-relevance cap.
        titles = [h["title"] for h in hints]
        assert any("argument-shape mismatch" in t.lower() for t in titles), (
            f"Expected JS argument-shape hint in {titles!r}"
        )

    def test_normal_locator_failure_does_not_include_js_hint(self):
        # Sanity: a different evaluate-js call (correct shape, real
        # selector failure) does NOT get the JS hint. Confirms the
        # heuristic gates on call shape, not error text.
        ctx = HintContext(
            session_id="t",
            keyword="Evaluate JavaScript",
            arguments=["css=#missing", "() => document.title"],
            error_text="Locator resolved to 0 elements",
        )
        hints = generate_hints(ctx)
        titles = [h["title"] for h in hints]
        assert not any("argument-shape mismatch" in t.lower() for t in titles)


# ---------------------------------------------------------------------------
# Layer 4: _check_invalid_selector broadening — "Unsupported token" too
# ---------------------------------------------------------------------------


class TestInvalidSelectorBroadenedPattern:
    """The original regex matched "Unexpected token" but missed
    Playwright's actual "Unsupported token" wording (OBS-03 root cause)."""

    @pytest.mark.parametrize("err_text", [
        "Unsupported token '{' while parsing css selector",
        "invalid selector: unknown engine 'foo'",
        "Element locator not a valid XPath expression",
        "Unexpected token at column 5",
        "selector syntax error near '#'",
    ])
    def test_fires_for_known_selector_syntax_errors(self, err_text):
        ctx = HintContext(
            session_id="t",
            keyword="Click",
            arguments=["css=foo"],
            error_text=err_text,
        )
        hints = _check_invalid_selector(ctx, err_text)
        assert len(hints) == 1
        assert hints[0].title.lower().startswith("invalid selector")

    def test_does_not_fire_for_unrelated_errors(self):
        ctx = HintContext(
            session_id="t",
            keyword="Click",
            arguments=["css=foo"],
            error_text="Timeout 5000ms exceeded",
        )
        assert _check_invalid_selector(ctx, ctx.error_text) == []

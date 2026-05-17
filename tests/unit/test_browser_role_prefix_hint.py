"""OBS-12 — ``button=LABEL`` / role-prefix-against-Browser-library hint.

The 2026-05-17 post-OBS validation benchmark surfaced a friction point on
Obstacle 8 (Wait a moment): Sonnet's first attempt was
``intent_action(intent="click", target="button=Calculate")``. The ARIA
snapshot rendered the element as ``button "Calculate"``, which makes
``button=Calculate`` look like a plausible locator — and it IS valid
SeleniumLibrary syntax. But Browser library doesn't support role-prefix
locators; pre-validation rejected with a generic "element not visible"
message that gave no syntax clue.

Cost in the benchmark: 1 wasted call. Low-impact, but a specific hint
closes the gap entirely (Sonnet's recovery on Obstacle 8 cost 9 calls
total; with the hint, ~6 would suffice).

These tests pin the hint surface:
(1) Browser library + role-prefix locator → hint fires (5 prefixes)
(2) Browser library + valid locator prefix (id=, css=, xpath=, text=,
    name=) → NO hint
(3) SeleniumLibrary + role-prefix locator → NO hint (valid SL syntax)
(4) Hint message names the role + value AND points at TWO working
    alternatives: text= and css=<tag>:text-is('<value>').
(5) Relevance ordering: OBS-12 hint outranks generic visibility hint
    but underranks OBS-03's JS-shape-mismatch.
(6) End-to-end through ``generate_hints``: the hint surfaces in the
    top-3 response output.
"""

from __future__ import annotations

import pytest

from robotmcp.utils.hints import (
    HintContext,
    _BROWSER_ROLE_PREFIX_PATTERN,
    _check_browser_role_prefix_misuse,
    _check_evaluate_javascript_misuse,
    _check_invalid_selector,
    generate_hints,
)


# ---------------------------------------------------------------------------
# Layer 1: pattern correctness (independent of HintContext wiring)
# ---------------------------------------------------------------------------


class TestRolePrefixPattern:
    """The compiled regex pattern recognises every documented role
    prefix and rejects everything else."""

    @pytest.mark.parametrize("locator", [
        "button=Calculate",
        "Button=Calculate",          # case-insensitive prefix
        "BUTTON=Calculate",
        "link=Click here",
        "Link=Click here",
        "input=username",
        "select=Country",
        "textarea=Comment",
        # Whitespace tolerance (matches OBS-01 normaliser's permissiveness)
        " button = Calculate ",
        "button= Calculate",
        "button =Calculate",
    ])
    def test_role_prefix_recognised(self, locator):
        m = _BROWSER_ROLE_PREFIX_PATTERN.match(locator)
        assert m is not None, f"{locator!r} should match role-prefix pattern"

    @pytest.mark.parametrize("locator", [
        # Valid Browser-library prefixes — must NOT match the role pattern
        "id=submit",
        "css=button.primary",
        "xpath=//button[@id='submit']",
        "text=Calculate",
        "name=username",
        # CSS shortcut forms
        "#submit",
        ".classname",
        "[name='foo']",
        # Playwright text-inside-CSS — superficially similar but uses
        # ``:text-is(...)``, not ``=``. Must NOT match.
        "button:text-is('Calculate')",
        "button:has(svg)",
        # Cascaded forms
        "id=foo >> nth=0",
        # XPath
        "//button[text()='Calculate']",
        # Selenium prefixes that aren't in the role set
        "tag=button",
        "class=primary",
        "data-testid=submit",
        "link:Click here",          # SL colon-separated; not role-prefix
        # Empty / non-string-like edge inputs
        "",
        "button",                   # no equals sign at all
        "button=",                  # equals with no value
        "button = ",                # equals with only whitespace value
    ])
    def test_non_role_prefixes_pass_through(self, locator):
        assert _BROWSER_ROLE_PREFIX_PATTERN.match(locator) is None, (
            f"{locator!r} should NOT match the role-prefix pattern"
        )


# ---------------------------------------------------------------------------
# Layer 2: checker behaviour against HintContext
# ---------------------------------------------------------------------------


def _ctx(*, library: str, args, keyword: str = "Click", err: str = "element not visible") -> HintContext:
    return HintContext(
        session_id="t",
        keyword=keyword,
        arguments=list(args),
        error_text=err,
        session_search_order=[library],
    )


class TestCheckBrowserRolePrefixFires:
    """Positive cases: hint fires for every role prefix against Browser."""

    @pytest.mark.parametrize("locator,role,value", [
        ("button=Calculate", "button", "Calculate"),
        ("link=Click here", "link", "Click here"),
        ("input=username", "input", "username"),
        ("select=Country", "select", "Country"),
        ("textarea=Comment", "textarea", "Comment"),
    ])
    def test_fires_for_each_role_prefix(self, locator, role, value):
        hints = _check_browser_role_prefix_misuse(
            _ctx(library="Browser", args=[locator]),
            "element not visible/enabled",
        )
        assert len(hints) == 1
        # Title names the rejected role prefix.
        assert role in hints[0].title.lower()
        assert f"`{role}=`" in hints[0].title.lower() or f"`{role}=" in hints[0].title

    def test_hint_message_names_role_and_value(self):
        hints = _check_browser_role_prefix_misuse(
            _ctx(library="Browser", args=["button=Calculate"]),
            "element not visible",
        )
        msg = hints[0].message
        assert "button=Calculate" in msg
        assert "SeleniumLibrary" in msg
        # The documented Browser-library prefixes are surfaced so the
        # agent knows the explicit-strategy table.
        assert "id=" in msg or "css=" in msg or "text=" in msg

    def test_hint_includes_two_working_alternatives(self):
        hints = _check_browser_role_prefix_misuse(
            _ctx(library="Browser", args=["button=Calculate"]),
            "element not visible",
        )
        examples = hints[0].examples
        # One example uses text= (Browser text engine)
        text_example = next(
            (e for e in examples if "text=Calculate" in str(e)), None,
        )
        assert text_example is not None, (
            f"expected an example with text=Calculate; got {examples!r}"
        )
        # One example uses css=<tag>:text-is(...) (Playwright text-inside-CSS)
        css_example = next(
            (e for e in examples if ":text-is(" in str(e)), None,
        )
        assert css_example is not None, (
            f"expected an example with :text-is(); got {examples!r}"
        )
        # Both examples are RF-keyword-shape strings (so the LLM can
        # copy-paste).
        assert all(
            "intent_action" in e["rf"] or "Click" in e.get("rf", "")
            for e in examples
        ), examples


class TestCheckBrowserRolePrefixSilent:
    """Negative cases: checker MUST NOT fire when it shouldn't."""

    def test_silent_for_selenium_library(self):
        # button=X is valid SeleniumLibrary syntax — hint must NOT fire.
        hints = _check_browser_role_prefix_misuse(
            _ctx(library="SeleniumLibrary", args=["button=Calculate"]),
            "element not visible",
        )
        assert hints == []

    def test_silent_for_appium_library(self):
        # AppiumLibrary uses its own locator forms; the role-prefix
        # rejection is Browser-specific.
        hints = _check_browser_role_prefix_misuse(
            _ctx(library="AppiumLibrary", args=["button=Calculate"]),
            "element not visible",
        )
        assert hints == []

    @pytest.mark.parametrize("locator", [
        "id=submit",
        "css=button.primary",
        "xpath=//button",
        "text=Calculate",
        "name=username",
        "button:text-is('Calculate')",
        "#submit",
        "//button[text()='Save']",
    ])
    def test_silent_for_valid_browser_locators(self, locator):
        hints = _check_browser_role_prefix_misuse(
            _ctx(library="Browser", args=[locator]),
            "element not visible",
        )
        assert hints == []

    def test_silent_when_no_arguments(self):
        hints = _check_browser_role_prefix_misuse(
            _ctx(library="Browser", args=[]),
            "element not visible",
        )
        assert hints == []

    def test_silent_when_first_arg_is_non_string(self):
        hints = _check_browser_role_prefix_misuse(
            _ctx(library="Browser", args=[42]),
            "element not visible",
        )
        assert hints == []

    def test_silent_when_no_library_context(self):
        # Without session_search_order, _detect_library returns
        # "unknown"; the checker must not fire.
        ctx = HintContext(
            session_id="t",
            keyword="Click",
            arguments=["button=Calculate"],
            error_text="element not visible",
            session_search_order=None,
        )
        assert _check_browser_role_prefix_misuse(ctx, ctx.error_text) == []


# ---------------------------------------------------------------------------
# Layer 3: relevance ordering — OBS-12 vs OBS-03 vs invalid-selector
# ---------------------------------------------------------------------------


class TestRelevanceOrdering:
    """When multiple checkers fire on the same failure, the more-
    specific diagnosis must win the top-3-by-relevance cap."""

    def test_outranks_invalid_selector(self):
        # Same failure could trigger _check_invalid_selector if the
        # error text mentions "Unsupported token"; the more-specific
        # role-prefix diagnosis is the primary cause and must outrank.
        role_hints = _check_browser_role_prefix_misuse(
            _ctx(library="Browser", args=["button=Calculate"]),
            "Unsupported token while parsing css selector",
        )
        sel_hints = _check_invalid_selector(
            _ctx(library="Browser", args=["button=Calculate"]),
            "Unsupported token while parsing css selector",
        )
        assert role_hints[0].relevance > sel_hints[0].relevance

    def test_underranks_evaluate_javascript_misuse(self):
        # OBS-03's JS-shape diagnosis is the most specific possible
        # for evaluate-js calls; OBS-12 should not crowd it out when
        # both fire (rare but possible if a JS body happens to start
        # with a role-prefix-shaped string).
        role_hints = _check_browser_role_prefix_misuse(
            _ctx(library="Browser", args=["button=Calculate"]),
            "Unsupported token",
        )
        js_hints = _check_evaluate_javascript_misuse(
            _ctx(library="Browser", keyword="Evaluate JavaScript",
                 args=["() => button.click()"]),
            "Unsupported token",
        )
        assert role_hints[0].relevance < js_hints[0].relevance


# ---------------------------------------------------------------------------
# Layer 4: end-to-end via generate_hints
# ---------------------------------------------------------------------------


class TestGenerateHintsEndToEnd:
    """The full ``generate_hints`` pipeline surfaces the role-prefix
    hint as a dict in its output."""

    def test_hint_appears_in_response(self):
        ctx = HintContext(
            session_id="t",
            keyword="Click",
            arguments=["button=Calculate"],
            error_text="element not visible/enabled",
            session_search_order=["Browser"],
        )
        hints = generate_hints(ctx)
        titles = [h["title"] for h in hints]
        assert any("role prefix" in t.lower() for t in titles), (
            f"role-prefix hint missing from response: {titles!r}"
        )

    def test_no_hint_in_response_for_selenium_session(self):
        ctx = HintContext(
            session_id="t",
            keyword="Click Element",
            arguments=["button=Calculate"],
            error_text="element not visible",
            session_search_order=["SeleniumLibrary"],
        )
        hints = generate_hints(ctx)
        titles = [h["title"] for h in hints]
        assert not any("role prefix" in t.lower() for t in titles), (
            "role-prefix hint must not surface for SeleniumLibrary sessions"
        )

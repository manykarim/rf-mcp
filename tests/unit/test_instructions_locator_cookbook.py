"""Locator cookbook is rendered in the default ``discovery_first`` template.

An LLM driving rf-mcp against a non-trivial page needs a concise,
library-aware locator pattern reference at the time it picks a locator.
The cheap, robust way to deliver this is to append it to the default
MCP instructions text — every tool-using LLM reads instructions when
it loads the server, before its first call.

This test pins the cookbook so a future edit can't silently regress
the most important parts (Browser ``role=``, Selenium ``link=``,
the nth filter syntax).
"""

from __future__ import annotations

import pytest

from robotmcp.domains.instruction.value_objects import InstructionTemplate


@pytest.fixture
def rendered_default():
    """Render the discovery_first template with a token that proves
    placeholder substitution still works after the cookbook insert."""
    tmpl = InstructionTemplate.discovery_first()
    content = tmpl.render({"available_tools": "TOOLS_SENTINEL"})
    return content.value


class TestCookbookPresence:
    """The cookbook section is rendered into the default template."""

    def test_section_header_present(self, rendered_default):
        assert "COMMON LOCATOR PATTERNS" in rendered_default

    def test_get_locator_guidance_reference_present(self, rendered_default):
        # Pointer to the full reference must survive — the cookbook is
        # intentionally short; the LLM falls back to the tool for depth.
        assert "get_locator_guidance" in rendered_default


class TestBrowserPatterns:
    """Browser library (Playwright) locator examples — only the prefixes
    documented in the Browser library's own explicit-strategies table:
    css, xpath, text, id. (No `role=` string engine — Playwright exposes
    role via API, not the locator string.)"""

    @pytest.mark.parametrize("snippet", [
        "id=submit",                  # canonical id locator
        "css=button.primary",         # css prefix
        "text=",                      # text prefix
        "xpath=",                     # xpath fallback
        ">> nth=",                    # Playwright cascaded nth filter
    ])
    def test_browser_pattern_present(self, rendered_default, snippet):
        assert snippet in rendered_default, f"missing Browser pattern: {snippet!r}"

    def test_no_invalid_role_string_engine(self, rendered_default):
        # The Browser library does NOT expose a `role=` string locator
        # engine. Listing it tricks the LLM into shipping locators the
        # library cannot resolve. If you want ARIA-role lookup, use the
        # `Get Element By Role` keyword instead.
        assert "role=button[name=" not in rendered_default
        assert "role=link[name=" not in rendered_default


class TestSeleniumPatterns:
    """SeleniumLibrary-specific locator strategies that have NO Browser
    equivalent — link/partial link and nth-of-type."""

    @pytest.mark.parametrize("snippet", [
        "name=username",       # Selenium-specific name= strategy
        "link=",               # full link text — Selenium-only
        "partial link=",       # partial link text — Selenium-only
        "nth-of-type",         # 1-based CSS nth filter
    ])
    def test_selenium_pattern_present(self, rendered_default, snippet):
        assert snippet in rendered_default, (
            f"missing SeleniumLibrary pattern: {snippet!r}"
        )


class TestPickingGuidance:
    """The terse advice section that an LLM scans when undecided."""

    @pytest.mark.parametrize("phrase", [
        "Prefer id= when available",
        "Fall back to css= or text=",
        "intent_action(..., nth=N)",
    ])
    def test_picking_guidance_present(self, rendered_default, phrase):
        assert phrase in rendered_default, f"missing guidance: {phrase!r}"


class TestPlaceholderStillWorks:
    """Cookbook insertion must not break the existing
    ``{available_tools}`` substitution that callers depend on."""

    def test_available_tools_substituted(self, rendered_default):
        assert "TOOLS_SENTINEL" in rendered_default

    def test_no_unrendered_placeholder(self, rendered_default):
        # The literal placeholder must not survive into the rendered output.
        assert "{available_tools}" not in rendered_default


class TestPreValidationRecoveryRecipe:
    """OBS-05 — 'When pre-validation rejects your locator' cookbook.

    Five of six Haiku-tier failures in the 2026-05-17 Tricentis obstacle
    course benchmark traced to the same pattern: pre-validation rejected
    a locator → the LLM abandoned the obstacle. Sonnet-tier recovered by
    switching locator strategy. This recipe makes that knowledge default
    so Haiku-tier models reach for it without having to guess.

    Pinned here so a future cookbook edit can't silently strip the recipe
    or replace the recommendations with non-existent parameters (e.g. an
    earlier draft referenced a `pre_validate=False` parameter that does
    not actually exist on `execute_step`)."""

    def test_section_header_present(self, rendered_default):
        # Header keyed on "PRE-VALIDATION" so it's discoverable in a
        # full-text scan of the instructions text.
        assert "WHEN PRE-VALIDATION REJECTS YOUR LOCATOR" in rendered_default

    def test_four_numbered_steps_present(self, rendered_default):
        # The recipe is a four-step recovery (OBS-05 added steps 1+2+4,
        # OBS-02 inserted step 3 for the pre_validate_timeout_ms knob).
        # Each step must be marked so the LLM can follow them sequentially.
        # Anchor on the imperative-verb-after-step-number rather than the
        # full sentence so cosmetic edits to wording don't break this.
        assert "1. Try the CSS-prefix equivalent" in rendered_default
        # Step 2 — re-inspect via get_session_state. Tolerant of "the DOM"
        # being present or not in the phrasing.
        assert "2. Re-inspect" in rendered_default
        assert "get_session_state" in rendered_default
        # Step 3 — extend the pre-validation gate (OBS-02).
        assert "3. Extend the gate" in rendered_default
        # Step 4 — last-resort escape hatches.
        assert "4. Last resort" in rendered_default

    @pytest.mark.parametrize("substitution_example", [
        # Step 1 substitutions — both halves of each pair must be present.
        "id=submit",       # also covered by the existing cookbook
        "css=#submit",     # the substitution target
        "name=username",
        "css=[name='username']",
    ])
    def test_step1_substitutions_present(self, rendered_default, substitution_example):
        assert substitution_example in rendered_default, (
            f"missing pre-validation recovery substitution: {substitution_example!r}"
        )

    @pytest.mark.parametrize("browser_substitution", [
        "button:text('Save')",      # Playwright text engine inside CSS
        ":text('Login')",            # bare text engine inside CSS
    ])
    def test_step1_browser_only_substitutions(self, rendered_default, browser_substitution):
        # These are the Playwright-engine-inside-CSS forms that Sonnet
        # discovered worked on obstacle 8 where the Selenium-style
        # `button[name='X']` was rejected by pre-validation.
        assert browser_substitution in rendered_default, (
            f"missing Browser-only recovery example: {browser_substitution!r}"
        )

    def test_step1_selenium_xpath_fallback_present(self, rendered_default):
        # For SeleniumLibrary the canonical fallback is xpath= (no
        # text-engine-inside-CSS like Playwright has). Pin the example.
        assert "xpath=//button[text()='Save']" in rendered_default

    def test_step2_recommends_aria_snapshot(self, rendered_default):
        # The DOM re-inspection step must direct the LLM to the
        # accessibility tree (ARIA snapshot) — that exposes attributes
        # like data-testid / aria-label / role that aren't always in raw
        # HTML, and that are typically MORE stable than the original
        # locator the agent tried.
        assert "get_session_state" in rendered_default
        assert "include_reduced_dom" in rendered_default
        assert "ARIA snapshot" in rendered_default
        # Stable-attribute hint is the operative guidance:
        assert "data-testid" in rendered_default

    def test_step3_force_true_documented_as_acceptable_escape(self, rendered_default):
        # OBS-07 (separate story) will reframe the global force=True
        # docstring; here we just need the recovery recipe to point at
        # it for the click-blocked-by-overlay case.
        assert 'intent_action(intent="click"' in rendered_default
        assert "force=True" in rendered_default
        assert "ACCEPTABLE" in rendered_default
        # Browser-only constraint must be explicit so agents don't try
        # force=True against SeleniumLibrary (no such param there).
        assert "Browser only" in rendered_default

    def test_step3_timeout_ms_zero_escape_present(self, rendered_default):
        # The genuine "skip pre-validation for one call" knob is
        # `timeout_ms=0` (verified at keyword_executor.py:1240, branch
        # `if timeout_ms <= 0: skip_preval = True`). An earlier draft
        # of OBS-05 referenced a `pre_validate=False` parameter that
        # does not exist — this test guards against that draft creeping
        # back into the cookbook.
        assert "timeout_ms=0" in rendered_default
        assert "pre_validate=False" not in rendered_default, (
            "pre_validate=False is not a real parameter on execute_step; "
            "use timeout_ms=0 to skip the gate for a single call"
        )

    def test_pre_validate_timeout_ms_parameter_documented(self, rendered_default):
        # OBS-02 acceptance #3: the new `pre_validate_timeout_ms`
        # parameter on execute_step must be documented in the
        # discovery_first template (not only in the docstring), so the
        # LLM sees it without having to inspect tool schemas.
        assert "pre_validate_timeout_ms" in rendered_default
        # Concrete suggested value — having a number anchors the LLM:
        assert "pre_validate_timeout_ms=2000" in rendered_default
        # The auto-retry behaviour should also be mentioned so the LLM
        # knows the gate already gives it one free retry.
        assert "auto-retries" in rendered_default or "retries" in rendered_default

    def test_anti_pattern_caveat_present(self, rendered_default):
        # The recipe must explicitly call out that force=True is NOT a
        # license to make hidden elements visible via DOM mutation.
        # This is the same "no DOM-mutating JS" rule from CLAUDE.md /
        # the natural-locator ADR; we just re-state it where the LLM
        # is most likely to be tempted to violate it.
        text = rendered_default
        assert "NOT for making hidden elements visible" in text or \
               "NOT for making hidden elements visible via DOM mutation" in text
        assert "anti-pattern" in text

    def test_recipe_appears_before_available_tools_line(self, rendered_default):
        # The new section must sit before the trailing
        # ``Available discovery tools: …`` line so that the closing line
        # remains the final entry — keeps the template's overall shape
        # the same for any tooling that anchors on the trailer.
        recipe_idx = rendered_default.index("WHEN PRE-VALIDATION REJECTS")
        trailer_idx = rendered_default.index("Available discovery tools")
        assert recipe_idx < trailer_idx, (
            "pre-validation recovery section must come before the "
            "'Available discovery tools' trailer"
        )

    def test_recipe_under_token_budget(self, rendered_default):
        # OBS-05 / OBS-02 / OBS-07 acceptance: section must stay terse —
        # the instructions are loaded into every MCP session, so token
        # cost is per-session. Budget history:
        #   OBS-05  : 1500 chars (initial 3-step recipe)
        #   OBS-02  : 1800 chars (+pre_validate_timeout_ms step)
        #   OBS-07  : 2000 chars (+commit=True cross-reference)
        # 2000 chars ≈ 500 tokens ≈ 6% of an 8K-token context. Each bump
        # has bought concrete agent behaviour (Haiku-tier abandonment
        # avoidance, SPA-form-fix discovery); further growth should be
        # justified by similar evidence.
        start = rendered_default.index("WHEN PRE-VALIDATION REJECTS")
        end = rendered_default.index("Available discovery tools")
        section_length = end - start
        assert section_length < 2000, (
            f"Pre-validation recovery section is {section_length} chars "
            f"(>=2000 char budget). Tighten it — the instructions are loaded "
            f"into every session."
        )

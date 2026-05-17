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
    """Browser library (Playwright) locator examples — the patterns most
    likely to be misremembered as plain CSS."""

    @pytest.mark.parametrize("snippet", [
        "id=submit",                  # bare prefix
        "css=button.primary",         # css prefix
        "text=",                      # text prefix
        "role=",                      # role prefix (Browser-specific, often missed)
        "xpath=",                     # xpath fallback
        ">> nth=",                    # Playwright nth filter syntax
    ])
    def test_browser_pattern_present(self, rendered_default, snippet):
        assert snippet in rendered_default, f"missing Browser pattern: {snippet!r}"


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
        "prefer role=",
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

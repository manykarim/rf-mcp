"""intent_action(nth=N) disambiguates when multiple elements match the
same locator.

Browser library (Playwright) supports a native ``>> nth=<n>`` filter that
slices into the matched set. SeleniumLibrary CSS locators support
``:nth-of-type(<n+1>)`` (1-based). Other Selenium locator strategies
(xpath, id, text, link, partial link) have no nth filter — the adapter
returns the locator unchanged and logs a debug-level note.
"""

from __future__ import annotations

import logging

import pytest

from robotmcp.domains.intent.adapters.mcp_tool import _apply_nth_to_locator
from robotmcp.domains.intent.value_objects import IntentTarget


class TestApplyNthToLocatorBrowser:
    """Browser library uses Playwright's ``>> nth=<n>`` filter."""

    @pytest.mark.parametrize("locator,nth,expected", [
        ("id=nav_automobile", 0, "id=nav_automobile >> nth=0"),
        ("css=button.submit", 2, "css=button.submit >> nth=2"),
        ("text=Next", 1, "text=Next >> nth=1"),
        ("xpath=//button[@type='submit']", 0,
         "xpath=//button[@type='submit'] >> nth=0"),
    ])
    def test_browser_appends_playwright_filter(self, locator, nth, expected):
        assert _apply_nth_to_locator(locator, nth, "Browser") == expected


class TestApplyNthToLocatorSelenium:
    """SeleniumLibrary supports nth-of-type for CSS only."""

    def test_css_prefix_gains_nth_of_type(self):
        result = _apply_nth_to_locator("css=button.submit", 0, "SeleniumLibrary")
        assert result == "css=button.submit:nth-of-type(1)"

    def test_css_alt_prefix_also_works(self):
        result = _apply_nth_to_locator("css:button.submit", 1, "SeleniumLibrary")
        assert result == "css:button.submit:nth-of-type(2)"

    def test_id_locator_unchanged_logs_warning(self, caplog):
        with caplog.at_level(logging.DEBUG):
            result = _apply_nth_to_locator("id=submit", 0, "SeleniumLibrary")
        assert result == "id=submit", "non-CSS Selenium locator must pass through"
        assert any("nth" in r.message.lower() for r in caplog.records), (
            "expected a debug-level note about nth being ignored for non-CSS locators"
        )

    def test_xpath_locator_unchanged(self):
        result = _apply_nth_to_locator(
            "xpath://button[@type='submit']", 0, "SeleniumLibrary",
        )
        assert result == "xpath://button[@type='submit']"


class TestApplyNthToLocatorOther:
    """Other libraries (AppiumLibrary, custom) default to Playwright syntax."""

    def test_default_is_playwright_syntax(self):
        assert _apply_nth_to_locator(
            "id=login", 0, "AppiumLibrary"
        ) == "id=login >> nth=0"


class TestIntentTargetNthField:
    """IntentTarget carries an optional nth field with a non-negative invariant."""

    def test_default_nth_is_none(self):
        t = IntentTarget(locator="id=foo")
        assert t.nth is None

    def test_nth_zero_is_valid(self):
        t = IntentTarget(locator="id=foo", nth=0)
        assert t.nth == 0

    def test_negative_nth_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            IntentTarget(locator="id=foo", nth=-1)

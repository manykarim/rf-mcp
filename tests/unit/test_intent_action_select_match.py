"""intent_action(intent="select", match=...) routes to the correct
underlying RF keyword/attribute combination.

Browser library uses a single ``Select Options By <selector> <attr> <value>``
keyword where ``<attr>`` is one of ``label/value/index/text``.
SeleniumLibrary uses dedicated keywords:
``Select From List By Label/Value/Index``.

The ``match=`` parameter on ``intent_action`` controls both:
- The attribute embedded in args for Browser.
- The dispatched keyword for SeleniumLibrary.

Default is ``label`` (matches RF semantics: "select by visible text").
``auto`` is opt-in and uses a numeric-string heuristic — flagged as
risky in the design because numeric visible labels (years, amounts) mis-route.
"""

from __future__ import annotations

import pytest

from robotmcp.domains.intent.aggregates import (
    _get_selenium_select_keyword,
    _resolve_select_match,
    _select_browser_transformer,
)


class TestResolveSelectMatch:
    """Pure helper: resolve match_option + value heuristic to attr string."""

    @pytest.mark.parametrize("match_opt", ["label", "value", "index", "text"])
    def test_explicit_strategy_passes_through(self, match_opt):
        assert _resolve_select_match(match_opt, "anything") == match_opt

    def test_auto_numeric_value_resolves_to_value(self):
        assert _resolve_select_match("auto", "5000000") == "value"

    def test_auto_negative_integer_resolves_to_value(self):
        assert _resolve_select_match("auto", "-1") == "value"

    def test_auto_alphabetic_resolves_to_label(self):
        assert _resolve_select_match("auto", "Audi") == "label"

    def test_auto_european_format_resolves_to_label(self):
        # "7.000.000,00" is not isdigit() -> label (correct for Tricentis)
        assert _resolve_select_match("auto", "7.000.000,00") == "label"

    def test_unknown_strategy_falls_back_to_label(self):
        assert _resolve_select_match("frobnicate", "Audi") == "label"

    def test_no_value_returns_label(self):
        assert _resolve_select_match("auto", None) == "label"


class TestBrowserSelectTransformer:
    """Browser's Select Options By <selector> <attr> <value>: the attr
    is driven by options["match"]."""

    def _stub_target(self, locator: str):
        class _T:
            pass
        t = _T()
        t.locator = locator
        return t

    def test_default_match_is_label(self):
        args = _select_browser_transformer(
            self._stub_target("id=country"), "Germany", None, options=None,
        )
        assert args == ["id=country", "label", "Germany"]

    def test_explicit_value_match(self):
        args = _select_browser_transformer(
            self._stub_target("id=insurancesum"), "5000000", None,
            options={"match": "value"},
        )
        assert args == ["id=insurancesum", "value", "5000000"]

    def test_explicit_index_match(self):
        args = _select_browser_transformer(
            self._stub_target("id=meritrating"), "1", None,
            options={"match": "index"},
        )
        assert args == ["id=meritrating", "index", "1"]

    def test_auto_numeric_routes_to_value(self):
        args = _select_browser_transformer(
            self._stub_target("id=insurancesum"), "5000000", None,
            options={"match": "auto"},
        )
        assert args[1] == "value"


class TestSeleniumSelectKeywordPicker:
    """SeleniumLibrary's keyword name depends on match strategy."""

    @pytest.mark.parametrize("match_opt,expected", [
        ("label", "Select From List By Label"),
        ("value", "Select From List By Value"),
        ("index", "Select From List By Index"),
        ("text", "Select From List By Label"),  # text -> label alias
    ])
    def test_explicit_strategy(self, match_opt, expected):
        assert _get_selenium_select_keyword(match_opt, "anything") == expected

    def test_auto_numeric_picks_by_value(self):
        assert (
            _get_selenium_select_keyword("auto", "100")
            == "Select From List By Value"
        )

    def test_auto_text_picks_by_label(self):
        assert (
            _get_selenium_select_keyword("auto", "Germany")
            == "Select From List By Label"
        )

    def test_unknown_falls_back_to_by_label(self):
        assert (
            _get_selenium_select_keyword("frobnicate", "x")
            == "Select From List By Label"
        )

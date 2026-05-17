"""OBS-06 — ``intent_action(intent="extract", ...)`` verb.

Two of the ten obstacles in the 2026-05-17 Tricentis benchmark needed
to read DOM state for use in a subsequent step (the order ID on
Obstacle 3; the autocomplete-result count on Obstacle 7). Sonnet
escaped via handwritten ``Evaluate JavaScript`` traversals; Haiku had
no clean primitive and abandoned both obstacles.

This story closes the gap with a new ``intent="extract"`` verb that
dispatches to a library-native getter based on ``mode``:

    text       → Browser.Get Text          / SeleniumLibrary.Get Text
    attribute  → Browser.Get Attribute     / SeleniumLibrary.Get Element Attribute
    count      → Browser.Get Element Count / SeleniumLibrary.Get Element Count
    value      → Browser.Get Property      / SeleniumLibrary.Get Value
    url        → Browser.Get Url           / SeleniumLibrary.Get Location
    title      → Browser.Get Title         / SeleniumLibrary.Get Title

These tests pin (1) the mode → keyword routers, (2) the per-mode
argument transformer's shape per library, (3) the IntentVerb registry
exposing the new EXTRACT verb, (4) the adapter wiring (mode +
attribute_name folded into options, keyword swap based on
library + mode), and (5) the server-level extract-specific behaviours
(extract_mode surfaced in resolution, count mode triggers
pre_validate_timeout_ms=0).
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from robotmcp.domains.intent.aggregates import (
    EXTRACT_MULTI_MATCH_MODES,
    IntentRegistry,
    _builtin_appium_mappings,
    _builtin_browser_mappings,
    _builtin_selenium_mappings,
    _extract_browser_transformer,
    _extract_selenium_transformer,
    _get_appium_extract_keyword,
    _get_browser_extract_keyword,
    _get_selenium_extract_keyword,
)
from robotmcp.domains.intent.value_objects import IntentTarget, IntentVerb


# ---------------------------------------------------------------------------
# Layer 1: IntentVerb enum + Literal alias both contain EXTRACT
# ---------------------------------------------------------------------------


class TestIntentVerbExposesExtract:
    """Both the domain enum and the MCP-facing Literal alias must accept
    ``"extract"``."""

    def test_intent_verb_enum_has_extract(self):
        assert IntentVerb.EXTRACT.value == "extract"

    def test_intent_verb_literal_alias_accepts_extract(self):
        # The Annotated[Literal[...]] form in kernel.py — runtime is just
        # the inner Literal. We can introspect __args__ on the Annotated
        # alias to confirm "extract" is one of the allowed strings.
        from robotmcp.domains.shared.kernel import IntentVerb as IntentVerbLit
        # The Annotated wraps Literal[...] as the first __args__ entry.
        literal_args = IntentVerbLit.__args__[0].__args__
        assert "extract" in literal_args


# ---------------------------------------------------------------------------
# Layer 2: mode → keyword routers
# ---------------------------------------------------------------------------


class TestBrowserExtractKeywordRouter:
    """Mode → Browser library keyword name."""

    @pytest.mark.parametrize("mode,expected_keyword", [
        ("text",      "Get Text"),
        ("attribute", "Get Attribute"),
        ("count",     "Get Element Count"),
        ("value",     "Get Property"),
        ("url",       "Get Url"),
        ("title",     "Get Title"),
    ])
    def test_mode_dispatches_to_expected_keyword(self, mode, expected_keyword):
        assert _get_browser_extract_keyword(mode) == expected_keyword

    def test_unknown_mode_falls_back_to_get_text(self):
        # Defensive — Literal validation upstream should prevent this,
        # but the router still has a safe fallback.
        assert _get_browser_extract_keyword("bogus") == "Get Text"

    def test_case_insensitive(self):
        assert _get_browser_extract_keyword("Attribute") == "Get Attribute"
        assert _get_browser_extract_keyword("URL") == "Get Url"


class TestSeleniumExtractKeywordRouter:
    """Mode → SeleniumLibrary keyword name."""

    @pytest.mark.parametrize("mode,expected_keyword", [
        ("text",      "Get Text"),
        # Selenium calls the keyword "Get Element Attribute" (not "Get
        # Attribute" like Browser does); regression-guard the difference.
        ("attribute", "Get Element Attribute"),
        ("count",     "Get Element Count"),
        # Selenium has a dedicated "Get Value" keyword; Browser uses
        # "Get Property" with attribute="value". These names diverge by
        # design.
        ("value",     "Get Value"),
        # Selenium calls page-URL "Get Location", Browser calls it "Get Url".
        ("url",       "Get Location"),
        ("title",     "Get Title"),
    ])
    def test_mode_dispatches_to_expected_keyword(self, mode, expected_keyword):
        assert _get_selenium_extract_keyword(mode) == expected_keyword


class TestAppiumExtractKeywordRouter:
    """Mode → AppiumLibrary keyword name. value/url/title aren't
    first-class in mobile context; they fall back to Get Text."""

    @pytest.mark.parametrize("mode,expected_keyword", [
        ("text",      "Get Text"),
        ("attribute", "Get Element Attribute"),
        ("count",     "Get Matching Xpath Count"),
    ])
    def test_mode_dispatches_to_expected_keyword(self, mode, expected_keyword):
        assert _get_appium_extract_keyword(mode) == expected_keyword

    @pytest.mark.parametrize("unsupported_mode", ["value", "url", "title"])
    def test_unsupported_modes_fall_back_to_get_text(self, unsupported_mode):
        assert _get_appium_extract_keyword(unsupported_mode) == "Get Text"


# ---------------------------------------------------------------------------
# Layer 3: argument transformers (per library, per mode)
# ---------------------------------------------------------------------------


def _target(locator: str) -> IntentTarget:
    return IntentTarget(locator=locator)


class TestExtractBrowserTransformerArgShapes:
    """The Browser transformer's args list must match each underlying
    keyword's signature."""

    def test_text_mode_one_arg_locator(self):
        args = _extract_browser_transformer(
            _target("id=foo"), None, None, {"mode": "text"},
        )
        assert args == ["id=foo"]

    def test_attribute_mode_two_args_locator_and_attribute(self):
        args = _extract_browser_transformer(
            _target("id=foo"), None, None,
            {"mode": "attribute", "attribute_name": "data-testid"},
        )
        assert args == ["id=foo", "data-testid"]

    def test_count_mode_one_arg_locator(self):
        args = _extract_browser_transformer(
            _target("css=.item"), None, None, {"mode": "count"},
        )
        assert args == ["css=.item"]

    def test_value_mode_browser_uses_get_property_with_value_attr(self):
        # Browser library has no Get Value keyword; the canonical way to
        # read an input's value is Get Property(selector, "value").
        # The transformer hard-wires the attribute name.
        args = _extract_browser_transformer(
            _target("id=input"), None, None, {"mode": "value"},
        )
        assert args == ["id=input", "value"]

    @pytest.mark.parametrize("mode", ["url", "title"])
    def test_url_and_title_modes_take_no_args(self, mode):
        args = _extract_browser_transformer(
            None, None, None, {"mode": mode},
        )
        assert args == []


class TestExtractBrowserTransformerValidation:
    """The transformer raises on invalid mode/target combinations
    BEFORE the keyword is dispatched, so the user sees a clear error
    instead of a confusing RF parse failure downstream."""

    def test_text_mode_without_target_raises(self):
        with pytest.raises(ValueError, match="requires a target"):
            _extract_browser_transformer(
                None, None, None, {"mode": "text"},
            )

    def test_attribute_mode_without_attribute_name_raises(self):
        with pytest.raises(ValueError, match="attribute_name"):
            _extract_browser_transformer(
                _target("id=foo"), None, None,
                {"mode": "attribute"},  # missing attribute_name
            )

    def test_attribute_mode_with_empty_attribute_name_raises(self):
        with pytest.raises(ValueError, match="attribute_name"):
            _extract_browser_transformer(
                _target("id=foo"), None, None,
                {"mode": "attribute", "attribute_name": ""},
            )


class TestExtractSeleniumTransformerArgShapes:
    """SeleniumLibrary transformer is similar to Browser but mode=value
    takes only the locator (Selenium's Get Value(locator) signature, no
    attribute name like Browser's Get Property)."""

    def test_text_mode_one_arg(self):
        args = _extract_selenium_transformer(
            _target("id=foo"), None, None, {"mode": "text"},
        )
        assert args == ["id=foo"]

    def test_attribute_mode_two_args(self):
        args = _extract_selenium_transformer(
            _target("id=foo"), None, None,
            {"mode": "attribute", "attribute_name": "value"},
        )
        assert args == ["id=foo", "value"]

    def test_value_mode_selenium_takes_only_locator(self):
        # CRITICAL: differs from Browser's value mode shape.
        # Selenium's Get Value(locator) does not take an attribute name.
        args = _extract_selenium_transformer(
            _target("id=input"), None, None, {"mode": "value"},
        )
        assert args == ["id=input"]

    @pytest.mark.parametrize("mode", ["url", "title"])
    def test_url_and_title_modes_take_no_args(self, mode):
        assert _extract_selenium_transformer(
            None, None, None, {"mode": mode},
        ) == []


# ---------------------------------------------------------------------------
# Layer 4: built-in mappings registered for all three libraries
# ---------------------------------------------------------------------------


class TestExtractMappingsRegistered:
    """Each library's built-in mappings must include an EXTRACT entry."""

    @pytest.mark.parametrize("mappings,library", [
        (_builtin_browser_mappings(),  "Browser"),
        (_builtin_selenium_mappings(), "SeleniumLibrary"),
        (_builtin_appium_mappings(),   "AppiumLibrary"),
    ])
    def test_library_has_extract_mapping(self, mappings, library):
        extract = next(
            (m for m in mappings if m.intent_verb == IntentVerb.EXTRACT),
            None,
        )
        assert extract is not None, (
            f"{library} mapping list missing EXTRACT entry"
        )
        # requires_target=False so url/title (no-locator modes) can be
        # invoked without target; the transformer validates per-mode.
        assert extract.requires_target is False
        assert extract.argument_transformer is not None

    def test_registry_with_builtins_resolves_extract_for_browser(self):
        registry = IntentRegistry.with_builtins()
        mapping = registry.resolve(IntentVerb.EXTRACT, "Browser")
        assert mapping is not None
        # The default keyword on the mapping is Get Text — adapter swaps
        # for other modes. The registry-level resolution doesn't know
        # about mode.
        assert mapping.keyword == "Get Text"


# ---------------------------------------------------------------------------
# Layer 5: multi-match-mode constant
# ---------------------------------------------------------------------------


class TestMultiMatchModeConstant:
    """``EXTRACT_MULTI_MATCH_MODES`` is the contract for which modes
    accept multi-match locators (count). The server uses this constant
    to know when to bypass pre-validation."""

    def test_count_is_multi_match(self):
        assert "count" in EXTRACT_MULTI_MATCH_MODES

    @pytest.mark.parametrize("single_match_mode", [
        "text", "attribute", "value", "url", "title",
    ])
    def test_single_match_modes_excluded(self, single_match_mode):
        # Pre-validation MUST run on these — they're regular reads /
        # implicit-existence assertions.
        assert single_match_mode not in EXTRACT_MULTI_MATCH_MODES


# ---------------------------------------------------------------------------
# Layer 6: server-level signature wiring
# ---------------------------------------------------------------------------


class TestServerLevelSignatureWiring:
    """The new parameters must be reachable from the top-level MCP tool.
    A signature check is enough; the behavioural integration is exercised
    by the integration test in tests/integration/test_real_browser_*.py."""

    def test_intent_action_accepts_mode_parameter(self):
        from robotmcp.server import intent_action
        fn = getattr(intent_action, "fn", intent_action)
        sig = inspect.signature(fn)
        assert "mode" in sig.parameters
        assert sig.parameters["mode"].default == "text"

    def test_intent_action_accepts_attribute_name_parameter(self):
        from robotmcp.server import intent_action
        fn = getattr(intent_action, "fn", intent_action)
        sig = inspect.signature(fn)
        assert "attribute_name" in sig.parameters
        assert sig.parameters["attribute_name"].default is None

    def test_resolve_intent_accepts_mode_and_attribute_name(self):
        from robotmcp.domains.intent.adapters.mcp_tool import IntentActionAdapter
        sig = inspect.signature(IntentActionAdapter.resolve_intent)
        assert "mode" in sig.parameters
        assert sig.parameters["mode"].default == "text"
        assert "attribute_name" in sig.parameters
        assert sig.parameters["attribute_name"].default is None


# ---------------------------------------------------------------------------
# Layer 7: adapter routing — mode + library → dispatched keyword
# ---------------------------------------------------------------------------


class TestAdapterDispatchedKeyword:
    """The adapter swaps the registry's default keyword based on the
    requested mode + resolved library. This is the load-bearing wire-up
    that takes the per-mode router output and produces the keyword the
    server-level layer dispatches via execute_step."""

    def _adapter_with_mock_resolver(self, resolved_library: str):
        """Build an IntentActionAdapter with a mocked resolver that
        returns a predictable Resolution. Avoids needing a real RF
        context."""
        from robotmcp.domains.intent.adapters.mcp_tool import IntentActionAdapter
        from robotmcp.domains.intent.value_objects import ResolvedIntent

        resolver = MagicMock()
        registry = MagicMock()
        # The registry lookup for force_keyword — return a mapping that
        # has no force_keyword (extract has no force-able dispatch).
        registry.resolve = MagicMock(
            return_value=MagicMock(force_keyword=None)
        )
        resolver.registry = registry

        def _fake_resolve(*, intent_verb, target, value, session_id, options, assign_to):
            return ResolvedIntent(
                intent_verb=intent_verb,
                library=resolved_library,
                keyword="Get Text",   # default keyword from mapping
                arguments=["id=foo"],
                normalized_locator=None,
            )

        resolver.resolve = _fake_resolve
        return IntentActionAdapter(resolver=resolver)

    @pytest.mark.parametrize("mode,expected_keyword", [
        ("text",      "Get Text"),
        ("attribute", "Get Attribute"),
        ("count",     "Get Element Count"),
        ("value",     "Get Property"),
        ("url",       "Get Url"),
        ("title",     "Get Title"),
    ])
    def test_browser_dispatched_keyword_per_mode(self, mode, expected_keyword):
        adapter = self._adapter_with_mock_resolver("Browser")
        result = adapter.resolve_intent(
            intent="extract",
            target="id=foo",
            mode=mode,
            attribute_name="data-testid" if mode == "attribute" else None,
        )
        assert result["keyword"] == expected_keyword
        assert result["extract_mode"] == mode

    @pytest.mark.parametrize("mode,expected_keyword", [
        ("text",      "Get Text"),
        ("attribute", "Get Element Attribute"),
        ("count",     "Get Element Count"),
        ("value",     "Get Value"),
        ("url",       "Get Location"),
        ("title",     "Get Title"),
    ])
    def test_selenium_dispatched_keyword_per_mode(self, mode, expected_keyword):
        adapter = self._adapter_with_mock_resolver("SeleniumLibrary")
        result = adapter.resolve_intent(
            intent="extract",
            target="id=foo",
            mode=mode,
            attribute_name="value" if mode == "attribute" else None,
        )
        assert result["keyword"] == expected_keyword

    def test_non_extract_intent_leaves_extract_mode_None(self):
        # Sanity: a click intent's resolution should NOT carry an
        # extract_mode field. Otherwise the server would try to apply
        # extract-specific behaviour (count → skip pre-validation) to
        # unrelated intents.
        adapter = self._adapter_with_mock_resolver("Browser")
        result = adapter.resolve_intent(intent="click", target="id=foo")
        assert result["extract_mode"] is None

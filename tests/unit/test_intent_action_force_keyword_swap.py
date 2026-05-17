"""intent_action(force=True) must swap the dispatched keyword when the
underlying RF keyword doesn't accept ``force=`` natively.

The Browser library's ``Click(selector, button)`` keyword takes no
``force`` argument — its second positional is ``button``. Pre-v0.32,
``intent_action(click, force=True)`` appended the literal string
``"force=True"`` to the args list, which RF then tried to parse as a
``MouseButton`` value, producing::

    ValueError: Argument 'button' got value 'force=True' that cannot be
    converted to MouseButton

The fix is declarative: each ``IntentMapping`` can declare a
``force_keyword`` — when set, intent_action substitutes it before
appending ``force=True``. Browser CLICK declares
``force_keyword="Click With Options"``.

These tests pin both halves: the declarative field on the mapping,
and the adapter exposing it in the resolution dict.
"""

from __future__ import annotations

from robotmcp.domains.intent.aggregates import (
    IntentRegistry,
    _builtin_browser_mappings,
)
from robotmcp.domains.intent.entities import IntentMapping
from robotmcp.domains.intent.value_objects import IntentVerb


# ---------------------------------------------------------------------------
# Layer 1: entity field
# ---------------------------------------------------------------------------


class TestIntentMappingForceKeywordField:
    """The IntentMapping dataclass carries an optional force_keyword."""

    def test_default_force_keyword_is_none(self):
        m = IntentMapping(
            intent_verb=IntentVerb.CLICK,
            library="Browser",
            keyword="Click",
        )
        assert m.force_keyword is None

    def test_force_keyword_set_explicitly(self):
        m = IntentMapping(
            intent_verb=IntentVerb.CLICK,
            library="Browser",
            keyword="Click",
            force_keyword="Click With Options",
        )
        assert m.force_keyword == "Click With Options"


# ---------------------------------------------------------------------------
# Layer 2: built-in Browser CLICK mapping declares the swap
# ---------------------------------------------------------------------------


class TestBuiltinBrowserClickDeclaresForceSwap:
    """The shipped Browser CLICK mapping must declare Click With Options
    as its force_keyword."""

    def test_browser_click_force_keyword_is_click_with_options(self):
        mappings = _builtin_browser_mappings()
        browser_click = next(
            (m for m in mappings
             if m.intent_verb == IntentVerb.CLICK and m.library == "Browser"),
            None,
        )
        assert browser_click is not None, "Browser CLICK mapping is missing"
        assert browser_click.force_keyword == "Click With Options", (
            "Browser CLICK must declare force_keyword='Click With Options' "
            "so the documented escape hatch actually dispatches the right "
            "RF keyword"
        )

    def test_browser_fill_text_has_no_force_keyword(self):
        """Fill Text accepts force= natively — no swap needed."""
        mappings = _builtin_browser_mappings()
        browser_fill = next(
            (m for m in mappings
             if m.intent_verb == IntentVerb.FILL and m.library == "Browser"),
            None,
        )
        assert browser_fill is not None
        assert browser_fill.force_keyword is None


# ---------------------------------------------------------------------------
# Layer 3: registry exposes the field via resolve()
# ---------------------------------------------------------------------------


class TestRegistryExposesForceKeyword:
    """IntentRegistry.resolve() returns the IntentMapping including the
    force_keyword field — consumed by the MCP-tool adapter."""

    def test_resolve_browser_click_carries_force_keyword(self):
        registry = IntentRegistry.with_builtins()
        mapping = registry.resolve(IntentVerb.CLICK, "Browser")
        assert mapping is not None
        assert mapping.force_keyword == "Click With Options"

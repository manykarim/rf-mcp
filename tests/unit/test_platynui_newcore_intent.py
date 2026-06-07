"""Unit tests for PlatynUI intent-domain integration (ADR-025).

Covers:
- IntentRegistry.with_builtins() PlatynUI.BareMetal mappings.
- _extract_platynui_transformer + _get_platynui_extract_keyword.
- LocatorStrategy.PLATYNUI_XPATH + LocatorNormalizerAdapter pass-through.
- SessionLookupAdapter.get_platform_type -> 'desktop'.
- IntentResolver._determine_library -> PlatynUI.BareMetal for desktop platform.

Run with: uv run pytest tests/unit/test_platynui_newcore_intent.py -q
"""

__test__ = True

from typing import List, Optional

import pytest

from robotmcp.domains.intent.adapters.locator_normalizer_adapter import (
    LocatorNormalizerAdapter,
)
from robotmcp.domains.intent.adapters.session_lookup_adapter import (
    SessionLookupAdapter,
)
from robotmcp.domains.intent.aggregates import (
    IntentRegistry,
    _extract_platynui_transformer,
    _get_platynui_extract_keyword,
)
from robotmcp.domains.intent.services import IntentResolver
from robotmcp.domains.intent.value_objects import (
    IntentTarget,
    IntentVerb,
    LocatorStrategy,
)

PLATYNUI = "PlatynUI.BareMetal"


@pytest.fixture
def registry():
    return IntentRegistry.with_builtins()


# =============================================================================
# Registry mappings
# =============================================================================


class TestRegistryMappings:
    @pytest.mark.parametrize(
        "verb,keyword",
        [
            (IntentVerb.CLICK, "Pointer Click"),
            (IntentVerb.FILL, "Keyboard Type"),
            (IntentVerb.HOVER, "Pointer Move To"),
            (IntentVerb.ASSERT_VISIBLE, "Query"),
            (IntentVerb.EXTRACT_TEXT, "Get Attribute"),
            (IntentVerb.EXTRACT, "Get Attribute"),
            (IntentVerb.WAIT_FOR, "Query"),
        ],
    )
    def test_mapping_resolves(self, registry, verb, keyword):
        mapping = registry.resolve(verb, PLATYNUI)
        assert mapping is not None
        assert mapping.keyword == keyword

    def test_navigate_has_no_mapping(self, registry):
        assert registry.resolve(IntentVerb.NAVIGATE, PLATYNUI) is None
        assert not registry.has_mapping(IntentVerb.NAVIGATE, PLATYNUI)

    def test_select_has_no_mapping(self, registry):
        assert registry.resolve(IntentVerb.SELECT, PLATYNUI) is None
        assert not registry.has_mapping(IntentVerb.SELECT, PLATYNUI)

    def test_platynui_in_supported_libraries(self, registry):
        assert PLATYNUI in registry.get_supported_libraries()


# =============================================================================
# Extract transformer
# =============================================================================


class TestExtractTransformer:
    def test_get_platynui_extract_keyword_always_get_attribute(self):
        for mode in ("text", "attribute", "value", "url", "title", "count", "weird"):
            assert _get_platynui_extract_keyword(mode) == "Get Attribute"

    def test_text_mode_appends_name(self):
        target = IntentTarget(locator="/app:*//control:Button[@Name='OK']")
        args = _extract_platynui_transformer(
            target, None, None, {"mode": "text"}
        )
        assert args == ["/app:*//control:Button[@Name='OK']", "Name"]

    def test_default_mode_is_text(self):
        target = IntentTarget(locator="/app:*//control:Text")
        args = _extract_platynui_transformer(target, None, None, {})
        assert args == ["/app:*//control:Text", "Name"]

    def test_attribute_mode_with_attribute_name(self):
        target = IntentTarget(locator="/app:*//control:Frame")
        args = _extract_platynui_transformer(
            target, None, None, {"mode": "attribute", "attribute_name": "Bounds"}
        )
        assert args == ["/app:*//control:Frame", "Bounds"]

    def test_attribute_mode_without_attribute_name_raises(self):
        target = IntentTarget(locator="/app:*//control:Frame")
        with pytest.raises(ValueError):
            _extract_platynui_transformer(
                target, None, None, {"mode": "attribute"}
            )

    def test_no_target_raises(self):
        with pytest.raises(ValueError):
            _extract_platynui_transformer(None, None, None, {"mode": "text"})

    def test_normalized_locator_takes_precedence(self):
        from robotmcp.domains.intent.value_objects import NormalizedLocator

        target = IntentTarget(locator="bare")
        norm = NormalizedLocator(
            value="/app:*//control:Button",
            source_locator="bare",
            target_library=PLATYNUI,
            strategy_applied="platynui_xpath_pass_through",
            was_transformed=False,
        )
        args = _extract_platynui_transformer(target, None, norm, {"mode": "text"})
        assert args == ["/app:*//control:Button", "Name"]


# =============================================================================
# LocatorStrategy + normalizer pass-through
# =============================================================================


class TestLocatorNormalization:
    def test_platynui_xpath_strategy_exists(self):
        assert LocatorStrategy.PLATYNUI_XPATH.value == "platynui_xpath"

    def test_xpath_descriptor_passes_through_unchanged(self):
        adapter = LocatorNormalizerAdapter()
        loc = "/app:*[@Name='x']//control:Button"
        result = adapter.normalize(IntentTarget(locator=loc), PLATYNUI)
        assert result.value == loc
        assert result.was_transformed is False
        assert result.strategy_applied == "platynui_xpath_pass_through"

    def test_bare_text_also_passes_through_for_platynui(self):
        adapter = LocatorNormalizerAdapter()
        result = adapter.normalize(IntentTarget(locator="OK"), PLATYNUI)
        # No web css/text prefixing for desktop descriptors
        assert result.value == "OK"
        assert result.was_transformed is False
        assert result.strategy_applied == "platynui_xpath_pass_through"


# =============================================================================
# SessionLookupAdapter.get_platform_type
# =============================================================================


class _StubSession:
    def __init__(self, imported):
        self.imported_libraries = imported

    def get_web_automation_library(self):
        return None


class _StubSessionManager:
    def __init__(self, session):
        self._session = session

    def get_session(self, session_id):
        return self._session


class TestSessionLookupPlatform:
    def test_desktop_when_platynui_imported(self):
        sm = _StubSessionManager(_StubSession(["PlatynUI.BareMetal", "BuiltIn"]))
        adapter = SessionLookupAdapter(sm)
        assert adapter.get_platform_type("s1") == "desktop"

    def test_desktop_when_platynui_alias_imported(self):
        sm = _StubSessionManager(_StubSession(["platynui"]))
        adapter = SessionLookupAdapter(sm)
        assert adapter.get_platform_type("s1") == "desktop"

    def test_web_when_browser_imported(self):
        sm = _StubSessionManager(_StubSession(["Browser"]))
        adapter = SessionLookupAdapter(sm)
        assert adapter.get_platform_type("s1") == "web"

    def test_web_when_no_session(self):
        class NoSession:
            def get_session(self, sid):
                return None

        adapter = SessionLookupAdapter(NoSession())
        assert adapter.get_platform_type("s1") == "web"


# =============================================================================
# IntentResolver._determine_library for desktop platform
# =============================================================================


class _DesktopLookup:
    """SessionLookup stub reporting a desktop session with PlatynUI imported."""

    def get_active_web_library(self, session_id: str) -> Optional[str]:
        return None

    def get_imported_libraries(self, session_id: str) -> List[str]:
        return ["PlatynUI.BareMetal", "BuiltIn"]

    def get_platform_type(self, session_id: str) -> str:
        return "desktop"


class _PassThroughNormalizer:
    def normalize(self, target, target_library):
        from robotmcp.domains.intent.value_objects import NormalizedLocator

        return NormalizedLocator(
            value=target.locator,
            source_locator=target.locator,
            target_library=target_library,
            strategy_applied="platynui_xpath_pass_through",
            was_transformed=False,
        )


class TestDetermineLibraryDesktop:
    def _resolver(self, registry):
        return IntentResolver(
            registry=registry,
            session_lookup=_DesktopLookup(),
            normalizer=_PassThroughNormalizer(),
            event_publisher=None,
        )

    def test_determine_library_returns_platynui(self, registry):
        resolver = self._resolver(registry)
        assert resolver._determine_library("s1", IntentVerb.CLICK) == PLATYNUI

    def test_resolve_click_uses_pointer_click(self, registry):
        resolver = self._resolver(registry)
        result = resolver.resolve(
            IntentVerb.CLICK,
            target=IntentTarget(locator="/app:*//control:Button[@Name='OK']"),
            value=None,
            session_id="s1",
        )
        assert result.library == PLATYNUI
        assert result.keyword == "Pointer Click"
        assert result.arguments == ["/app:*//control:Button[@Name='OK']"]

    def test_resolve_fill_uses_keyboard_type(self, registry):
        resolver = self._resolver(registry)
        result = resolver.resolve(
            IntentVerb.FILL,
            target=IntentTarget(locator="/app:*//control:Text"),
            value="Hello",
            session_id="s1",
        )
        assert result.keyword == "Keyboard Type"
        assert result.library == PLATYNUI

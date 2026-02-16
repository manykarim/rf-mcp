"""Comprehensive unit tests for IntentResolver domain service (ADR-007).

Tests cover: full resolution pipeline, library determination, error handling,
event publishing, locator normalization integration.

Run with: uv run pytest tests/unit/test_intent_resolver.py -v
"""

__test__ = True

from typing import Dict, List, Optional

import pytest

from robotmcp.domains.intent.aggregates import IntentRegistry
from robotmcp.domains.intent.events import (
    IntentResolved,
    LocatorNormalized,
    UnmappedIntentRequested,
)
from robotmcp.domains.intent.services import IntentResolutionError, IntentResolver
from robotmcp.domains.intent.value_objects import (
    IntentTarget,
    IntentVerb,
    NormalizedLocator,
)


# =============================================================================
# Mock SessionLookup
# =============================================================================


class MockSessionLookup:
    """Mock implementation of the SessionLookup protocol."""

    def __init__(self, library="Browser", platform="web", imported=None):
        self.library = library
        self.platform = platform
        self.imported = imported or [library]

    def get_active_web_library(self, session_id: str) -> Optional[str]:
        return self.library

    def get_imported_libraries(self, session_id: str) -> List[str]:
        return self.imported

    def get_platform_type(self, session_id: str) -> str:
        return self.platform


# =============================================================================
# Mock LocatorNormalizer
# =============================================================================


class MockLocatorNormalizer:
    """Mock normalizer that returns a pass-through NormalizedLocator."""

    def __init__(self, transform: bool = False, prefix: str = ""):
        self._transform = transform
        self._prefix = prefix

    def normalize(
        self, target: IntentTarget, target_library: str,
    ) -> NormalizedLocator:
        if self._transform and self._prefix:
            new_value = f"{self._prefix}{target.locator}"
            return NormalizedLocator(
                value=new_value,
                source_locator=target.locator,
                target_library=target_library,
                strategy_applied="test_transform",
                was_transformed=True,
            )
        return NormalizedLocator(
            value=target.locator,
            source_locator=target.locator,
            target_library=target_library,
            strategy_applied="pass_through",
            was_transformed=False,
        )


# =============================================================================
# Mock EventPublisher
# =============================================================================


class MockEventPublisher:
    """Collects published events."""

    def __init__(self):
        self.events: list = []

    def publish(self, event: object) -> None:
        self.events.append(event)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def registry():
    return IntentRegistry.with_builtins()


@pytest.fixture
def event_publisher():
    return MockEventPublisher()


def make_resolver(
    registry, library="Browser", platform="web",
    transform=False, prefix="", event_publisher=None,
):
    """Helper to create a resolver with given configuration."""
    return IntentResolver(
        registry=registry,
        session_lookup=MockSessionLookup(library=library, platform=platform),
        normalizer=MockLocatorNormalizer(transform=transform, prefix=prefix),
        event_publisher=event_publisher,
    )


# =============================================================================
# Basic resolution: Browser Library
# =============================================================================


class TestBrowserResolution:
    """Test resolution for Browser Library."""

    def test_click_browser(self, registry):
        resolver = make_resolver(registry, library="Browser")
        result = resolver.resolve(
            IntentVerb.CLICK,
            target=IntentTarget(locator="text=Login"),
            value=None,
            session_id="s1",
        )
        assert result.keyword == "Click"
        assert result.library == "Browser"
        assert result.arguments == ["text=Login"]

    def test_fill_browser(self, registry):
        resolver = make_resolver(registry, library="Browser")
        result = resolver.resolve(
            IntentVerb.FILL,
            target=IntentTarget(locator="#username"),
            value="alice",
            session_id="s1",
        )
        assert result.keyword == "Fill Text"
        assert result.arguments == ["#username", "alice"]

    def test_navigate_browser(self, registry):
        resolver = make_resolver(registry, library="Browser")
        result = resolver.resolve(
            IntentVerb.NAVIGATE,
            target=IntentTarget(locator="https://example.com"),
            value=None,
            session_id="s1",
        )
        assert result.keyword == "Go To"
        assert result.arguments == ["https://example.com"]

    def test_hover_browser(self, registry):
        resolver = make_resolver(registry, library="Browser")
        result = resolver.resolve(
            IntentVerb.HOVER,
            target=IntentTarget(locator="text=Menu"),
            value=None,
            session_id="s1",
        )
        assert result.keyword == "Hover"

    def test_select_browser(self, registry):
        resolver = make_resolver(registry, library="Browser")
        result = resolver.resolve(
            IntentVerb.SELECT,
            target=IntentTarget(locator="#dropdown"),
            value="Option A",
            session_id="s1",
        )
        assert result.keyword == "Select Options By"
        assert "label" in result.arguments  # label injected by transformer
        assert "Option A" in result.arguments

    def test_assert_visible_browser(self, registry):
        resolver = make_resolver(registry, library="Browser")
        result = resolver.resolve(
            IntentVerb.ASSERT_VISIBLE,
            target=IntentTarget(locator="text=Welcome"),
            value=None,
            session_id="s1",
        )
        assert result.keyword == "Get Element States"
        assert "visible" in result.arguments
        assert "==" in result.arguments
        assert "True" in result.arguments

    def test_wait_for_browser(self, registry):
        resolver = make_resolver(registry, library="Browser")
        result = resolver.resolve(
            IntentVerb.WAIT_FOR,
            target=IntentTarget(locator="text=Loading"),
            value=None,
            session_id="s1",
        )
        assert result.keyword == "Wait For Elements State"
        assert "visible" in result.arguments
        assert any("timeout=" in a for a in result.arguments)

    def test_extract_text_browser(self, registry):
        resolver = make_resolver(registry, library="Browser")
        result = resolver.resolve(
            IntentVerb.EXTRACT_TEXT,
            target=IntentTarget(locator="#price"),
            value=None,
            session_id="s1",
        )
        assert result.keyword == "Get Text"


# =============================================================================
# Basic resolution: SeleniumLibrary
# =============================================================================


class TestSeleniumResolution:
    """Test resolution for SeleniumLibrary."""

    def test_click_selenium(self, registry):
        resolver = make_resolver(registry, library="SeleniumLibrary")
        result = resolver.resolve(
            IntentVerb.CLICK,
            target=IntentTarget(locator="id=submit"),
            value=None,
            session_id="s1",
        )
        assert result.keyword == "Click Element"
        assert result.library == "SeleniumLibrary"

    def test_fill_selenium(self, registry):
        resolver = make_resolver(registry, library="SeleniumLibrary")
        result = resolver.resolve(
            IntentVerb.FILL,
            target=IntentTarget(locator="id=username"),
            value="bob",
            session_id="s1",
        )
        assert result.keyword == "Input Text"
        assert result.arguments == ["id=username", "bob"]

    def test_hover_selenium(self, registry):
        resolver = make_resolver(registry, library="SeleniumLibrary")
        result = resolver.resolve(
            IntentVerb.HOVER,
            target=IntentTarget(locator="id=menu"),
            value=None,
            session_id="s1",
        )
        assert result.keyword == "Mouse Over"

    def test_navigate_selenium(self, registry):
        resolver = make_resolver(registry, library="SeleniumLibrary")
        result = resolver.resolve(
            IntentVerb.NAVIGATE,
            target=IntentTarget(locator="https://test.com"),
            value=None,
            session_id="s1",
        )
        assert result.keyword == "Go To"

    def test_wait_for_selenium(self, registry):
        resolver = make_resolver(registry, library="SeleniumLibrary")
        result = resolver.resolve(
            IntentVerb.WAIT_FOR,
            target=IntentTarget(locator="id=spinner"),
            value=None,
            session_id="s1",
            options={"timeout": "15s"},
        )
        assert result.keyword == "Wait Until Element Is Visible"
        assert "15s" in result.arguments


# =============================================================================
# Basic resolution: AppiumLibrary
# =============================================================================


class TestAppiumResolution:
    """Test resolution for AppiumLibrary."""

    def test_click_appium(self, registry):
        resolver = make_resolver(
            registry, library="AppiumLibrary", platform="mobile",
        )
        result = resolver.resolve(
            IntentVerb.CLICK,
            target=IntentTarget(locator="accessibility_id=Login"),
            value=None,
            session_id="s1",
        )
        assert result.keyword == "Click Element"
        assert result.library == "AppiumLibrary"

    def test_hover_appium_raises(self, registry):
        """AppiumLibrary has no HOVER mapping -- _determine_library cannot find a library."""
        resolver = make_resolver(
            registry, library="AppiumLibrary", platform="mobile",
        )
        with pytest.raises(IntentResolutionError, match="Cannot determine target library"):
            resolver.resolve(
                IntentVerb.HOVER,
                target=IntentTarget(locator="some_locator"),
                value=None,
                session_id="s1",
            )

    def test_select_appium_raises(self, registry):
        """AppiumLibrary has no SELECT mapping -- _determine_library cannot find a library."""
        resolver = make_resolver(
            registry, library="AppiumLibrary", platform="mobile",
        )
        with pytest.raises(IntentResolutionError, match="Cannot determine target library"):
            resolver.resolve(
                IntentVerb.SELECT,
                target=IntentTarget(locator="dropdown"),
                value="opt",
                session_id="s1",
            )


# =============================================================================
# Error handling
# =============================================================================


class TestErrorHandling:
    """Test error cases in resolution."""

    def test_missing_target_when_required(self, registry):
        resolver = make_resolver(registry, library="Browser")
        with pytest.raises(IntentResolutionError, match="requires a target"):
            resolver.resolve(
                IntentVerb.CLICK,
                target=None,
                value=None,
                session_id="s1",
            )

    def test_missing_value_when_required(self, registry):
        resolver = make_resolver(registry, library="Browser")
        with pytest.raises(IntentResolutionError, match="requires a value"):
            resolver.resolve(
                IntentVerb.FILL,
                target=IntentTarget(locator="#field"),
                value=None,
                session_id="s1",
            )

    def test_no_library_determined_raises(self, registry):
        """When session has no web library and no imported libraries with mappings."""
        resolver = IntentResolver(
            registry=registry,
            session_lookup=MockSessionLookup(library=None, imported=[]),
            normalizer=MockLocatorNormalizer(),
        )
        with pytest.raises(IntentResolutionError, match="Cannot determine target library"):
            resolver.resolve(
                IntentVerb.CLICK,
                target=IntentTarget(locator="x"),
                value=None,
                session_id="s1",
            )


# =============================================================================
# Event publishing
# =============================================================================


class TestEventPublishing:
    """Test that domain events are published correctly."""

    def test_intent_resolved_event_published(self, registry, event_publisher):
        resolver = make_resolver(
            registry, library="Browser", event_publisher=event_publisher,
        )
        resolver.resolve(
            IntentVerb.CLICK,
            target=IntentTarget(locator="text=Login"),
            value=None,
            session_id="s1",
        )
        resolved_events = [e for e in event_publisher.events if isinstance(e, IntentResolved)]
        assert len(resolved_events) == 1
        assert resolved_events[0].intent_verb == "click"
        assert resolved_events[0].keyword == "Click"
        assert resolved_events[0].library == "Browser"

    def test_locator_normalized_event_when_transformed(self, registry, event_publisher):
        """LocatorNormalized is emitted when normalizer transforms the locator."""
        resolver = make_resolver(
            registry, library="Browser",
            transform=True, prefix="text=",
            event_publisher=event_publisher,
        )
        resolver.resolve(
            IntentVerb.CLICK,
            target=IntentTarget(locator="Login"),
            value=None,
            session_id="s1",
        )
        norm_events = [e for e in event_publisher.events if isinstance(e, LocatorNormalized)]
        assert len(norm_events) == 1
        assert norm_events[0].original == "Login"
        assert norm_events[0].normalized == "text=Login"

    def test_no_locator_normalized_when_not_transformed(self, registry, event_publisher):
        resolver = make_resolver(
            registry, library="Browser",
            transform=False,
            event_publisher=event_publisher,
        )
        resolver.resolve(
            IntentVerb.CLICK,
            target=IntentTarget(locator="text=Login"),
            value=None,
            session_id="s1",
        )
        norm_events = [e for e in event_publisher.events if isinstance(e, LocatorNormalized)]
        assert len(norm_events) == 0

    def test_unmapped_intent_event_not_published_when_library_undetermined(
        self, registry, event_publisher,
    ):
        """UnmappedIntentRequested is NOT emitted when _determine_library fails.

        In the current implementation, _determine_library checks has_mapping()
        before returning a library. If no library has a mapping for the verb,
        _determine_library raises IntentResolutionError before the code reaches
        the "No mapping" path that publishes UnmappedIntentRequested. This
        means the event is effectively unreachable for verbs that no library
        supports. This test documents that behavior.
        """
        resolver = make_resolver(
            registry, library="AppiumLibrary", platform="mobile",
            event_publisher=event_publisher,
        )
        with pytest.raises(IntentResolutionError, match="Cannot determine target library"):
            resolver.resolve(
                IntentVerb.HOVER,
                target=IntentTarget(locator="x"),
                value=None,
                session_id="s1",
            )
        # No UnmappedIntentRequested because _determine_library fails first
        unmapped_events = [e for e in event_publisher.events if isinstance(e, UnmappedIntentRequested)]
        assert len(unmapped_events) == 0

    def test_no_events_without_publisher(self, registry):
        """Resolution works fine without an event publisher."""
        resolver = make_resolver(registry, library="Browser")
        result = resolver.resolve(
            IntentVerb.CLICK,
            target=IntentTarget(locator="text=Login"),
            value=None,
            session_id="s1",
        )
        assert result.keyword == "Click"


# =============================================================================
# Library determination
# =============================================================================


class TestLibraryDetermination:
    """Test _determine_library logic."""

    def test_prefers_active_web_library(self, registry):
        resolver = IntentResolver(
            registry=registry,
            session_lookup=MockSessionLookup(
                library="SeleniumLibrary",
                imported=["Browser", "SeleniumLibrary"],
            ),
            normalizer=MockLocatorNormalizer(),
        )
        result = resolver.resolve(
            IntentVerb.CLICK,
            target=IntentTarget(locator="id=btn"),
            value=None,
            session_id="s1",
        )
        assert result.library == "SeleniumLibrary"

    def test_falls_back_to_imported_libraries(self, registry):
        """When no active web library, falls back to first imported with a mapping."""
        resolver = IntentResolver(
            registry=registry,
            session_lookup=MockSessionLookup(
                library=None,
                imported=["Browser"],
            ),
            normalizer=MockLocatorNormalizer(),
        )
        result = resolver.resolve(
            IntentVerb.CLICK,
            target=IntentTarget(locator="x"),
            value=None,
            session_id="s1",
        )
        assert result.library == "Browser"

    def test_mobile_platform_uses_appium(self, registry):
        """Mobile platform should pick AppiumLibrary."""
        resolver = IntentResolver(
            registry=registry,
            session_lookup=MockSessionLookup(
                library=None,
                platform="mobile",
                imported=["AppiumLibrary"],
            ),
            normalizer=MockLocatorNormalizer(),
        )
        result = resolver.resolve(
            IntentVerb.CLICK,
            target=IntentTarget(locator="x"),
            value=None,
            session_id="s1",
        )
        assert result.library == "AppiumLibrary"


# =============================================================================
# Metadata / assign_to
# =============================================================================


class TestMetadata:
    """Test metadata and assign_to propagation."""

    def test_timeout_category_in_metadata(self, registry):
        resolver = make_resolver(registry, library="Browser")
        result = resolver.resolve(
            IntentVerb.NAVIGATE,
            target=IntentTarget(locator="https://example.com"),
            value=None,
            session_id="s1",
        )
        assert result.metadata["timeout_category"] == "navigation"

    def test_assign_to_in_metadata(self, registry):
        resolver = make_resolver(registry, library="Browser")
        result = resolver.resolve(
            IntentVerb.EXTRACT_TEXT,
            target=IntentTarget(locator="#price"),
            value=None,
            session_id="s1",
            assign_to="${PRICE}",
        )
        assert result.metadata.get("assign_to") == "${PRICE}"

    def test_no_assign_to_omitted_from_metadata(self, registry):
        resolver = make_resolver(registry, library="Browser")
        result = resolver.resolve(
            IntentVerb.CLICK,
            target=IntentTarget(locator="text=Login"),
            value=None,
            session_id="s1",
        )
        assert "assign_to" not in result.metadata

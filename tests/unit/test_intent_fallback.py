"""Unit tests for navigate intent fallback (FallbackStep, NavigateFallbackSequence,
IntentRegistry fallbacks, IntentResolver.get_navigate_fallback).

Run with: uv run pytest tests/unit/test_intent_fallback.py -v
"""

__test__ = True

from typing import List, Optional

import pytest

from robotmcp.domains.intent.aggregates import IntentRegistry
from robotmcp.domains.intent.events import IntentFallbackUsed
from robotmcp.domains.intent.services import IntentResolver
from robotmcp.domains.intent.value_objects import (
    FallbackStep,
    IntentTarget,
    IntentVerb,
    NavigateFallbackSequence,
    NormalizedLocator,
)


# =============================================================================
# Mock helpers (reuse patterns from test_intent_resolver.py)
# =============================================================================


class MockSessionLookup:
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


class MockLocatorNormalizer:
    def normalize(self, target: IntentTarget, target_library: str) -> NormalizedLocator:
        return NormalizedLocator(
            value=target.locator,
            source_locator=target.locator,
            target_library=target_library,
            strategy_applied="pass_through",
            was_transformed=False,
        )


class MockEventPublisher:
    def __init__(self):
        self.events: list = []

    def publish(self, event: object) -> None:
        self.events.append(event)


# =============================================================================
# TestFallbackStep
# =============================================================================


class TestFallbackStep:
    """Tests for the FallbackStep frozen dataclass."""

    def test_fields_accessible(self):
        step = FallbackStep(
            keyword="New Browser",
            arguments=("headless=False",),
            reason="Open browser first",
        )
        assert step.keyword == "New Browser"
        assert step.arguments == ("headless=False",)
        assert step.reason == "Open browser first"

    def test_frozen_immutability(self):
        step = FallbackStep(keyword="New Page", arguments=(), reason="test")
        with pytest.raises(AttributeError):
            step.keyword = "Other"  # type: ignore[misc]

    def test_equality_by_value(self):
        a = FallbackStep(keyword="New Page", arguments=(), reason="r")
        b = FallbackStep(keyword="New Page", arguments=(), reason="r")
        assert a == b


# =============================================================================
# TestNavigateFallbackSequence
# =============================================================================


class TestNavigateFallbackSequence:
    """Tests for NavigateFallbackSequence value object."""

    def _make_seq(self, pattern="no browser"):
        return NavigateFallbackSequence(
            library="Browser",
            error_pattern=pattern,
            steps=(
                FallbackStep(keyword="New Browser", arguments=("headless=False",), reason="r"),
            ),
            description="test",
        )

    def test_matches_error_true(self):
        seq = self._make_seq(r"no browser|browser.*not.*open")
        assert seq.matches_error("Error: no browser found") is True

    def test_matches_error_false(self):
        seq = self._make_seq(r"no browser")
        assert seq.matches_error("element not found") is False

    def test_case_insensitive(self):
        seq = self._make_seq(r"No Browser")
        assert seq.matches_error("NO BROWSER available") is True

    def test_alternation_patterns(self):
        seq = self._make_seq(r"no browser|browser.*not.*open|no open browser")
        assert seq.matches_error("browser is not open yet") is True
        assert seq.matches_error("no open browser detected") is True
        assert seq.matches_error("timeout waiting for element") is False

    def test_empty_error_string_returns_false(self):
        seq = self._make_seq(r"no browser")
        assert seq.matches_error("") is False

    def test_description_and_library_fields(self):
        seq = NavigateFallbackSequence(
            library="SeleniumLibrary",
            error_pattern="test",
            steps=(),
            description="Open browser before navigating",
        )
        assert seq.library == "SeleniumLibrary"
        assert seq.description == "Open browser before navigating"


# =============================================================================
# TestIntentRegistryFallbacks
# =============================================================================


class TestIntentRegistryFallbacks:
    """Tests for IntentRegistry navigate fallback methods."""

    def test_register_and_get_fallback(self):
        registry = IntentRegistry()
        seq = NavigateFallbackSequence(
            library="Browser",
            error_pattern=r"no browser",
            steps=(FallbackStep(keyword="New Browser", arguments=(), reason="r"),),
            description="test",
        )
        registry.register_navigate_fallback(seq)
        result = registry.get_navigate_fallback("Browser", "no browser found")
        assert result is seq

    def test_get_fallback_no_match(self):
        registry = IntentRegistry()
        seq = NavigateFallbackSequence(
            library="Browser",
            error_pattern=r"no browser",
            steps=(FallbackStep(keyword="New Browser", arguments=(), reason="r"),),
            description="test",
        )
        registry.register_navigate_fallback(seq)
        assert registry.get_navigate_fallback("Browser", "element not found") is None

    def test_get_fallback_unknown_library(self):
        registry = IntentRegistry()
        assert registry.get_navigate_fallback("UnknownLib", "no browser") is None

    def test_builtin_browser_no_browser(self):
        registry = IntentRegistry.with_builtins()
        result = registry.get_navigate_fallback("Browser", "no browser")
        assert result is not None
        assert len(result.steps) == 2
        assert result.steps[0].keyword == "New Browser"
        assert result.steps[1].keyword == "New Page"

    def test_builtin_browser_page_closed(self):
        registry = IntentRegistry.with_builtins()
        result = registry.get_navigate_fallback("Browser", "target closed")
        assert result is not None
        assert len(result.steps) == 1
        assert result.steps[0].keyword == "New Page"

    def test_builtin_selenium_no_browser(self):
        registry = IntentRegistry.with_builtins()
        result = registry.get_navigate_fallback(
            "SeleniumLibrary", "No browser is open"
        )
        assert result is not None
        assert len(result.steps) == 1
        assert result.steps[0].keyword == "Open Browser"

    def test_no_appium_fallbacks(self):
        registry = IntentRegistry.with_builtins()
        assert registry.get_navigate_fallback("AppiumLibrary", "any error") is None

    def test_first_matching_sequence_wins(self):
        """More specific pattern registered first should match first."""
        registry = IntentRegistry.with_builtins()
        # "no browser" matches the first Browser sequence (2 steps)
        result = registry.get_navigate_fallback("Browser", "no browser")
        assert result is not None
        assert len(result.steps) == 2  # New Browser + New Page


# =============================================================================
# TestIntentResolverFallback
# =============================================================================


class TestIntentResolverFallback:
    """Tests for IntentResolver.get_navigate_fallback() method."""

    def _make_resolver(self, library="Browser", publisher=None):
        return IntentResolver(
            registry=IntentRegistry.with_builtins(),
            session_lookup=MockSessionLookup(library=library),
            normalizer=MockLocatorNormalizer(),
            event_publisher=publisher,
        )

    def test_returns_steps_for_matching_error(self):
        resolver = self._make_resolver()
        result = resolver.get_navigate_fallback(
            library="Browser",
            error_message="no browser",
            session_id="test-1",
        )
        assert result is not None
        assert len(result) == 2
        assert result[0]["keyword"] == "New Browser"
        assert result[1]["keyword"] == "New Page"

    def test_returns_none_for_no_match(self):
        resolver = self._make_resolver()
        result = resolver.get_navigate_fallback(
            library="Browser",
            error_message="timeout waiting for element",
            session_id="test-1",
        )
        assert result is None

    def test_emits_fallback_event(self):
        pub = MockEventPublisher()
        resolver = self._make_resolver(publisher=pub)
        resolver.get_navigate_fallback(
            library="Browser",
            error_message="no browser",
            session_id="test-1",
        )
        assert len(pub.events) == 1
        evt = pub.events[0]
        assert isinstance(evt, IntentFallbackUsed)
        assert evt.intent_verb == "navigate"
        assert evt.fallback_keyword == "New Browser"
        assert evt.library == "Browser"
        assert evt.session_id == "test-1"

    def test_no_event_when_no_match(self):
        pub = MockEventPublisher()
        resolver = self._make_resolver(publisher=pub)
        resolver.get_navigate_fallback(
            library="Browser",
            error_message="element not found",
            session_id="test-1",
        )
        assert len(pub.events) == 0

    def test_browser_no_browser_returns_two_steps(self):
        resolver = self._make_resolver()
        result = resolver.get_navigate_fallback(
            library="Browser", error_message="no browser", session_id="s",
        )
        assert len(result) == 2
        assert result[0]["keyword"] == "New Browser"
        assert result[0]["arguments"] == ["headless=False"]
        assert result[1]["keyword"] == "New Page"
        assert result[1]["arguments"] == []

    def test_browser_page_closed_returns_one_step(self):
        resolver = self._make_resolver()
        result = resolver.get_navigate_fallback(
            library="Browser", error_message="page closed unexpectedly", session_id="s",
        )
        assert len(result) == 1
        assert result[0]["keyword"] == "New Page"

    def test_selenium_no_browser_returns_one_step(self):
        resolver = self._make_resolver(library="SeleniumLibrary")
        result = resolver.get_navigate_fallback(
            library="SeleniumLibrary",
            error_message="No browser is open",
            session_id="s",
        )
        assert len(result) == 1
        assert result[0]["keyword"] == "Open Browser"
        assert result[0]["arguments"] == ["about:blank", "chrome"]

    def test_returns_none_for_unknown_library(self):
        resolver = self._make_resolver()
        result = resolver.get_navigate_fallback(
            library="UnknownLib", error_message="no browser", session_id="s",
        )
        assert result is None


# =============================================================================
# TestBuiltinFallbackPatterns
# =============================================================================


class TestBuiltinFallbackPatterns:
    """Verify built-in error patterns cover real Playwright/WebDriver errors."""

    @pytest.fixture
    def registry(self):
        return IntentRegistry.with_builtins()

    @pytest.mark.parametrize("error_msg", [
        "no browser",
        "Browser is not open",
        "browser was not open",
        "no open browser",
    ])
    def test_browser_no_browser_patterns(self, registry, error_msg):
        result = registry.get_navigate_fallback("Browser", error_msg)
        assert result is not None
        assert result.steps[0].keyword == "New Browser"

    @pytest.mark.parametrize("error_msg", [
        "target closed",
        "page was closed",
        "no page",
        "no context",
    ])
    def test_browser_page_closed_patterns(self, registry, error_msg):
        result = registry.get_navigate_fallback("Browser", error_msg)
        assert result is not None
        assert result.steps[0].keyword == "New Page"

    @pytest.mark.parametrize("error_msg", [
        "No browser is open",
        "invalid session id",
        "Session was not found",
    ])
    def test_selenium_patterns(self, registry, error_msg):
        result = registry.get_navigate_fallback("SeleniumLibrary", error_msg)
        assert result is not None
        assert result.steps[0].keyword == "Open Browser"

    @pytest.mark.parametrize("error_msg", [
        "invalid URL",
        "ERR_NAME_NOT_RESOLVED",
        "timeout waiting for element",
        "element not found",
    ])
    def test_non_browser_errors_do_not_match(self, registry, error_msg):
        """URL syntax and timeout errors should NOT trigger fallback."""
        assert registry.get_navigate_fallback("Browser", error_msg) is None
        assert registry.get_navigate_fallback("SeleniumLibrary", error_msg) is None

    def test_browser_new_browser_has_headless_arg(self, registry):
        result = registry.get_navigate_fallback("Browser", "no browser")
        assert "headless=False" in result.steps[0].arguments

    def test_selenium_open_browser_has_correct_args(self, registry):
        result = registry.get_navigate_fallback(
            "SeleniumLibrary", "No browser is open"
        )
        assert result.steps[0].arguments == ("about:blank", "chrome")

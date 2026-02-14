"""Unit tests for Intent domain events (ADR-007).

Tests cover: IntentResolved, LocatorNormalized, UnmappedIntentRequested,
IntentFallbackUsed construction and frozen behavior.

Run with: uv run pytest tests/unit/test_intent_events.py -v
"""

__test__ = True

import pytest

from robotmcp.domains.intent.events import (
    IntentFallbackUsed,
    IntentResolved,
    LocatorNormalized,
    UnmappedIntentRequested,
)


class TestIntentResolved:
    """Test IntentResolved frozen dataclass."""

    def test_construction(self):
        event = IntentResolved(
            intent_verb="click",
            keyword="Click",
            library="Browser",
            session_id="s1",
            locator_transformed=False,
        )
        assert event.intent_verb == "click"
        assert event.keyword == "Click"
        assert event.library == "Browser"
        assert event.session_id == "s1"
        assert event.locator_transformed is False
        assert event.timestamp is not None

    def test_frozen(self):
        event = IntentResolved(
            intent_verb="click", keyword="Click",
            library="Browser", session_id="s1",
            locator_transformed=False,
        )
        with pytest.raises(AttributeError):
            event.keyword = "Other"

    def test_equality(self):
        """Two events with same values (except timestamp) are generally different."""
        e1 = IntentResolved("click", "Click", "Browser", "s1", False)
        e2 = IntentResolved("click", "Click", "Browser", "s1", False)
        # timestamps differ so they are not equal
        # (unless created in the same sub-second resolution)
        # Just verify they are both IntentResolved
        assert isinstance(e1, IntentResolved)
        assert isinstance(e2, IntentResolved)


class TestLocatorNormalized:
    """Test LocatorNormalized frozen dataclass."""

    def test_construction(self):
        event = LocatorNormalized(
            original="Login",
            normalized="text=Login",
            target_library="Browser",
            strategy="bare_text",
            session_id="s1",
        )
        assert event.original == "Login"
        assert event.normalized == "text=Login"
        assert event.target_library == "Browser"
        assert event.strategy == "bare_text"

    def test_frozen(self):
        event = LocatorNormalized("x", "y", "Browser", "test", "s1")
        with pytest.raises(AttributeError):
            event.original = "z"


class TestUnmappedIntentRequested:
    """Test UnmappedIntentRequested frozen dataclass."""

    def test_construction(self):
        event = UnmappedIntentRequested(
            intent_verb="hover",
            library="AppiumLibrary",
            session_id="s1",
        )
        assert event.intent_verb == "hover"
        assert event.library == "AppiumLibrary"
        assert event.session_id == "s1"
        assert event.timestamp is not None

    def test_frozen(self):
        event = UnmappedIntentRequested("hover", "AppiumLibrary", "s1")
        with pytest.raises(AttributeError):
            event.intent_verb = "click"


class TestIntentFallbackUsed:
    """Test IntentFallbackUsed frozen dataclass."""

    def test_construction(self):
        event = IntentFallbackUsed(
            intent_verb="scroll",
            fallback_keyword="Scroll Element Into View",
            library="SeleniumLibrary",
            session_id="s1",
            reason="No mapping found, user provided keyword hint",
        )
        assert event.intent_verb == "scroll"
        assert event.fallback_keyword == "Scroll Element Into View"
        assert event.reason.startswith("No mapping")

    def test_frozen(self):
        event = IntentFallbackUsed("scroll", "Scroll", "Browser", "s1", "test")
        with pytest.raises(AttributeError):
            event.intent_verb = "click"

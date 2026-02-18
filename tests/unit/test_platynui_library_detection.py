"""Unit tests for PlatynUI library detection and session type inference (ADR-012).

Tests cover: LibraryDetector patterns, session type detection,
preference routing, fallback detection, and is_desktop_session.

Run with: uv run pytest tests/unit/test_platynui_library_detection.py -v
"""

__test__ = True

import pytest

from robotmcp.utils.library_detection import LibraryDetector
from robotmcp.models.session_models import (
    ExecutionSession,
    PlatformType,
    SessionType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector() -> LibraryDetector:
    return LibraryDetector()


def _fresh_session(session_id: str = "det-test") -> ExecutionSession:
    """Create a fresh session with default state."""
    return ExecutionSession(session_id=session_id)


# ===========================================================================
# LibraryDetector tests
# ===========================================================================

class TestLibraryDetectorPlatynUIPatterns:
    """Tests for PlatynUI detection in LibraryDetector."""

    @pytest.mark.parametrize("text,expected", [
        ("PlatynUI test for calculator", "PlatynUI"),
        ("using platynui for desktop automation", "PlatynUI"),
        ("desktop ui automation testing", "PlatynUI"),
        ("BareMetal keyword execution", "PlatynUI"),
        ("accessibility tree inspection", "PlatynUI"),
    ])
    def test_direct_platynui_detection(self, detector, text, expected):
        result = detector.detect(text)
        assert result == expected, f"Text '{text}' detected as '{result}', expected '{expected}'"

    @pytest.mark.parametrize("text", [
        "test native app with AT-SPI",
        "UI Automation testing on Windows",
        "pointer click on button",
        "keyboard type in edit field",
        "activate window Calculator",
    ])
    def test_platynui_detection_via_related_terms(self, detector, text):
        # These should detect PlatynUI due to related patterns.
        # Some may have competing scores but PlatynUI should win or be detected.
        scores = detector.get_scores(text)
        assert scores.get("PlatynUI", 0) > 0, (
            f"Expected PlatynUI score > 0 for '{text}', got scores: {scores}"
        )

    def test_selenium_text_does_not_detect_platynui(self, detector):
        result = detector.detect("use Selenium for web testing")
        assert result != "PlatynUI"

    def test_browser_library_text_does_not_detect_platynui(self, detector):
        result = detector.detect("Browser Library playwright test")
        assert result != "PlatynUI"

    def test_api_text_does_not_detect_platynui(self, detector):
        result = detector.detect("API testing with requests")
        assert result != "PlatynUI"

    def test_platynui_not_in_conflict_groups(self, detector):
        # PlatynUI is a desktop library, not a web library.
        # It should never appear in the web_automation conflict group.
        conflicts = detector.get_conflicting_detections(
            "PlatynUI desktop test with Browser Library web test"
        )
        for group_name, libs in conflicts.items():
            assert "PlatynUI" not in libs, (
                f"PlatynUI should not be in conflict group '{group_name}'"
            )

    def test_platynui_score_higher_for_desktop_text(self, detector):
        desktop_text = "desktop ui automation testing with PlatynUI"
        scores = detector.get_scores(desktop_text)
        platynui_score = scores.get("PlatynUI", 0)
        browser_score = scores.get("Browser", 0)
        selenium_score = scores.get("SeleniumLibrary", 0)
        assert platynui_score > browser_score, (
            f"PlatynUI score ({platynui_score}) should exceed Browser ({browser_score})"
        )
        assert platynui_score > selenium_score, (
            f"PlatynUI score ({platynui_score}) should exceed SeleniumLibrary ({selenium_score})"
        )

    def test_platynui_in_compiled_patterns(self, detector):
        assert "PlatynUI" in detector._compiled_patterns

    def test_detect_all_returns_platynui(self, detector):
        detected = detector.detect_all("PlatynUI test for calculator")
        libs = [lib for lib, _ in detected]
        assert "PlatynUI" in libs

    def test_empty_text_returns_none(self, detector):
        assert detector.detect("") is None

    def test_none_text_returns_none(self, detector):
        assert detector.detect(None) is None

    def test_at_spi_scores(self, detector):
        scores = detector.get_scores("AT-SPI desktop testing on Linux")
        assert scores.get("PlatynUI", 0) >= 5

    def test_uia_pattern(self, detector):
        scores = detector.get_scores("UIA testing on Windows desktop")
        assert scores.get("PlatynUI", 0) > 0


# ===========================================================================
# Session type detection tests
# ===========================================================================

class TestSessionTypeDetection:
    """Tests for detect_session_type_from_scenario with PlatynUI."""

    @pytest.mark.parametrize("text,expected", [
        ("PlatynUI test for calculator", SessionType.DESKTOP_TESTING),
        ("Desktop UI automation", SessionType.DESKTOP_TESTING),
        ("desktop ui automation testing with native app", SessionType.DESKTOP_TESTING),
    ])
    def test_desktop_scenario_detection(self, text, expected):
        session = _fresh_session()
        result = session.detect_session_type_from_scenario(text)
        assert result == expected, f"Text '{text}' detected as {result.value}"

    def test_baremetal_scenario_detects_desktop(self):
        session = _fresh_session()
        result = session.detect_session_type_from_scenario(
            "using baremetal for native app testing"
        )
        assert result == SessionType.DESKTOP_TESTING

    def test_atspi_scenario_detects_desktop(self):
        session = _fresh_session()
        result = session.detect_session_type_from_scenario(
            "AT-SPI desktop testing on Linux"
        )
        assert result == SessionType.DESKTOP_TESTING

    def test_web_scenario_not_desktop(self):
        session = _fresh_session()
        result = session.detect_session_type_from_scenario(
            "Web automation with browser"
        )
        assert result != SessionType.DESKTOP_TESTING

    def test_selenium_scenario_is_web(self):
        session = _fresh_session()
        result = session.detect_session_type_from_scenario(
            "Selenium test for e-commerce"
        )
        assert result == SessionType.WEB_AUTOMATION

    def test_desktop_testing_profile_exists(self):
        profiles = ExecutionSession._get_session_profiles()
        assert SessionType.DESKTOP_TESTING in profiles

    def test_desktop_testing_profile_has_platynui(self):
        profiles = ExecutionSession._get_session_profiles()
        profile = profiles[SessionType.DESKTOP_TESTING]
        assert "PlatynUI.BareMetal" in profile.core_libraries

    def test_desktop_testing_profile_search_order(self):
        profiles = ExecutionSession._get_session_profiles()
        profile = profiles[SessionType.DESKTOP_TESTING]
        assert profile.search_order[0] == "PlatynUI.BareMetal"


# ===========================================================================
# Preference routing tests
# ===========================================================================

class TestPreferenceRouting:
    """Tests for _get_profile_for_preferences with PlatynUI."""

    def test_platynui_preference_sets_desktop_testing(self):
        session = _fresh_session()
        session.explicit_library_preference = "PlatynUI"
        profiles = ExecutionSession._get_session_profiles()
        profile = session._get_profile_for_preferences(profiles)
        assert session.session_type == SessionType.DESKTOP_TESTING
        assert profile is not None

    def test_platynui_baremetal_preference_sets_desktop_testing(self):
        session = _fresh_session()
        session.explicit_library_preference = "PlatynUI.BareMetal"
        profiles = ExecutionSession._get_session_profiles()
        profile = session._get_profile_for_preferences(profiles)
        assert session.session_type == SessionType.DESKTOP_TESTING
        assert profile is not None

    def test_platynui_profile_core_libraries_include_baremetal(self):
        session = _fresh_session()
        session.explicit_library_preference = "PlatynUI"
        profiles = ExecutionSession._get_session_profiles()
        profile = session._get_profile_for_preferences(profiles)
        assert "PlatynUI.BareMetal" in profile.core_libraries

    def test_platynui_profile_search_order_starts_with_baremetal(self):
        session = _fresh_session()
        session.explicit_library_preference = "PlatynUI.BareMetal"
        profiles = ExecutionSession._get_session_profiles()
        profile = session._get_profile_for_preferences(profiles)
        assert profile.search_order[0] == "PlatynUI.BareMetal"


# ===========================================================================
# Fallback detection tests
# ===========================================================================

class TestFallbackDetection:
    """Tests for _fallback_detect_library with PlatynUI terms."""

    def test_platynui_text_detected(self):
        session = _fresh_session()
        result = session._fallback_detect_library("platynui test for calculator")
        assert result == "PlatynUI"

    def test_desktop_automation_detected(self):
        session = _fresh_session()
        result = session._fallback_detect_library("desktop automation for forms")
        assert result == "PlatynUI"

    def test_baremetal_detected(self):
        session = _fresh_session()
        result = session._fallback_detect_library("baremetal keywords for testing")
        assert result == "PlatynUI"

    def test_unrelated_text_returns_none(self):
        session = _fresh_session()
        result = session._fallback_detect_library("just a simple text scenario")
        assert result is None


# ===========================================================================
# is_desktop_session tests
# ===========================================================================

class TestIsDesktopSession:
    """Tests for ExecutionSession.is_desktop_session()."""

    def test_desktop_testing_type(self):
        session = _fresh_session()
        session.session_type = SessionType.DESKTOP_TESTING
        assert session.is_desktop_session() is True

    def test_platynui_baremetal_imported(self):
        session = _fresh_session()
        session.imported_libraries.append("PlatynUI.BareMetal")
        assert session.is_desktop_session() is True

    def test_platynui_imported(self):
        session = _fresh_session()
        session.imported_libraries.append("PlatynUI")
        assert session.is_desktop_session() is True

    def test_desktop_platform_type(self):
        session = _fresh_session()
        session.platform_type = PlatformType.DESKTOP
        assert session.is_desktop_session() is True

    def test_web_session_is_not_desktop(self):
        session = _fresh_session()
        session.session_type = SessionType.WEB_AUTOMATION
        session.imported_libraries.append("Browser")
        assert session.is_desktop_session() is False

    def test_default_session_is_not_desktop(self):
        session = _fresh_session()
        assert session.is_desktop_session() is False

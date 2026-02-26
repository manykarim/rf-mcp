"""Unit tests for desktop platform detection (Fix 1 + Fix 2).

Fix 1: detect_platform_from_scenario desktop support -- ensures desktop keywords
       (platynui, gnome, baremetal, gtk, atspi, qt, ui automation, etc.) are
       detected correctly and that the ``context`` parameter boosts scores.
Fix 2: server.py context passing -- ensures the ``context`` parameter flows
       through to detect_platform_from_scenario so that callers (e.g. the MCP
       server's manage_session handler) can hint the platform.

Run with: uv run pytest tests/unit/test_platform_detection_desktop.py -v
"""

import pytest
from robotmcp.components.execution.session_manager import SessionManager
from robotmcp.models.session_models import PlatformType


class TestDesktopPlatformDetection:
    """Tests for Fix 1: detect_platform_from_scenario desktop support."""

    @pytest.fixture
    def manager(self):
        return SessionManager()

    # ----- Desktop keyword detection (no context) -----

    def test_desktop_keyword_platynui(self, manager):
        result = manager.detect_platform_from_scenario("Test GNOME Calculator with PlatynUI")
        assert result == PlatformType.DESKTOP

    def test_desktop_keyword_gnome(self, manager):
        result = manager.detect_platform_from_scenario("Automate GNOME desktop application")
        assert result == PlatformType.DESKTOP

    def test_desktop_keyword_baremetal(self, manager):
        result = manager.detect_platform_from_scenario("Use BareMetal library for UI testing")
        assert result == PlatformType.DESKTOP

    def test_desktop_keyword_bare_metal_with_space(self, manager):
        result = manager.detect_platform_from_scenario("Use bare metal library for testing")
        assert result == PlatformType.DESKTOP

    def test_desktop_keyword_gtk(self, manager):
        result = manager.detect_platform_from_scenario("Test GTK dialog window")
        assert result == PlatformType.DESKTOP

    def test_desktop_keyword_atspi(self, manager):
        result = manager.detect_platform_from_scenario("Access AT-SPI accessibility tree")
        assert result == PlatformType.DESKTOP

    def test_desktop_keyword_qt(self, manager):
        result = manager.detect_platform_from_scenario("Automate Qt desktop application")
        assert result == PlatformType.DESKTOP

    def test_desktop_keyword_ui_automation(self, manager):
        result = manager.detect_platform_from_scenario("Use UI Automation for native gui testing")
        assert result == PlatformType.DESKTOP

    def test_desktop_keyword_window_management(self, manager):
        result = manager.detect_platform_from_scenario("Test window management features")
        assert result == PlatformType.DESKTOP

    def test_desktop_keyword_accessibility_tree(self, manager):
        result = manager.detect_platform_from_scenario("Navigate the accessibility tree")
        assert result == PlatformType.DESKTOP

    def test_desktop_keyword_native_gui(self, manager):
        result = manager.detect_platform_from_scenario("Automate native gui controls")
        assert result == PlatformType.DESKTOP

    def test_desktop_keyword_uia(self, manager):
        result = manager.detect_platform_from_scenario("Inspect uia elements on the dialog")
        assert result == PlatformType.DESKTOP

    def test_desktop_keyword_wxwidgets(self, manager):
        result = manager.detect_platform_from_scenario("Test wxwidgets dialog controls")
        assert result == PlatformType.DESKTOP

    # ----- Context parameter boost -----

    def test_desktop_context_boost(self, manager):
        """Context='desktop' should boost desktop score even with no desktop keywords."""
        result = manager.detect_platform_from_scenario("Test the application", context="desktop")
        assert result == PlatformType.DESKTOP

    def test_mobile_context_boost(self, manager):
        """Context='mobile' should boost mobile score."""
        result = manager.detect_platform_from_scenario("Test the application", context="mobile")
        assert result == PlatformType.MOBILE

    def test_web_context_boost(self, manager):
        """Context='web' should stay as web."""
        result = manager.detect_platform_from_scenario("Test the application", context="web")
        assert result == PlatformType.WEB

    def test_api_context_boost(self, manager):
        """Context='api' should boost api score."""
        result = manager.detect_platform_from_scenario("Test the application", context="api")
        assert result == PlatformType.API

    # ----- Desktop vs mobile disambiguation (the confirmed bug) -----

    def test_desktop_beats_mobile_with_app_and_context(self, manager):
        """'desktop app' should be DESKTOP, not MOBILE (the confirmed bug)."""
        result = manager.detect_platform_from_scenario(
            "Test GNOME Calculator desktop app with PlatynUI", context="desktop"
        )
        assert result == PlatformType.DESKTOP

    def test_desktop_beats_mobile_app_keyword(self, manager):
        """Even 'app' alone shouldn't override desktop context."""
        result = manager.detect_platform_from_scenario(
            "Automate desktop app", context="desktop"
        )
        assert result == PlatformType.DESKTOP

    def test_desktop_keyword_plus_app_no_context(self, manager):
        """'app' triggers mobile but 'desktop' triggers desktop -- desktop should win."""
        # 'desktop' adds 1 to desktop_score, 'app' adds 1 to mobile_score.
        # When tied, desktop is checked first (desktop > mobile), so with equal scores
        # the condition desktop_score > mobile_score fails -- desktop won't win with a tie.
        # With 'platynui' we get desktop_score=2, mobile_score=1.
        result = manager.detect_platform_from_scenario(
            "Test desktop app with platynui"
        )
        assert result == PlatformType.DESKTOP

    # ----- Backward compatibility -----

    def test_pure_mobile_scenario_stays_mobile(self, manager):
        """Pure mobile scenario without desktop context stays mobile."""
        result = manager.detect_platform_from_scenario("Test Android app on emulator")
        assert result == PlatformType.MOBILE

    def test_no_context_defaults_to_web(self, manager):
        """No context and no keywords defaults to WEB."""
        result = manager.detect_platform_from_scenario("Run some test")
        assert result == PlatformType.WEB

    def test_backward_compatible_no_context(self, manager):
        """Existing callers passing no context should still work."""
        result = manager.detect_platform_from_scenario("Click the button on the web page")
        assert result == PlatformType.WEB

    def test_backward_compatible_mobile(self, manager):
        """Mobile detection still works without context."""
        result = manager.detect_platform_from_scenario("Swipe on mobile device")
        assert result == PlatformType.MOBILE

    def test_backward_compatible_api(self, manager):
        """API detection still works without context."""
        result = manager.detect_platform_from_scenario("Send REST API request to endpoint")
        assert result == PlatformType.API

    # ----- Edge cases -----

    def test_empty_scenario_defaults_to_web(self, manager):
        """Empty scenario string with no context defaults to WEB."""
        result = manager.detect_platform_from_scenario("")
        assert result == PlatformType.WEB

    def test_empty_scenario_with_desktop_context(self, manager):
        """Empty scenario with desktop context should be DESKTOP."""
        result = manager.detect_platform_from_scenario("", context="desktop")
        assert result == PlatformType.DESKTOP

    def test_case_insensitive_detection(self, manager):
        """Keywords should match case-insensitively."""
        result = manager.detect_platform_from_scenario("Test PLATYNUI on GNOME Desktop")
        assert result == PlatformType.DESKTOP

    def test_multiple_desktop_keywords_accumulate(self, manager):
        """Multiple desktop keywords should all count toward the score."""
        result = manager.detect_platform_from_scenario(
            "Test PlatynUI with GTK on GNOME desktop using AT-SPI accessibility tree"
        )
        assert result == PlatformType.DESKTOP

    def test_context_none_is_default(self, manager):
        """Passing context=None should behave the same as not passing context."""
        result_none = manager.detect_platform_from_scenario("Test the thing", context=None)
        result_default = manager.detect_platform_from_scenario("Test the thing")
        assert result_none == result_default

    def test_unknown_context_has_no_effect(self, manager):
        """An unrecognized context string should not boost any score."""
        result = manager.detect_platform_from_scenario("Test the thing", context="unknown")
        # With no keywords and no valid context boost, should default to WEB
        assert result == PlatformType.WEB

    def test_mixed_signals_desktop_context_wins(self, manager):
        """When scenario has web keywords but context is desktop, context should help."""
        result = manager.detect_platform_from_scenario(
            "Open the browser page for desktop platynui application",
            context="desktop"
        )
        # desktop keywords: 'desktop', 'platynui' (2) + context boost (3) = 5
        # web keywords: 'browser', 'page' (2)
        assert result == PlatformType.DESKTOP

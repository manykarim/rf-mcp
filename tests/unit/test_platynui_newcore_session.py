"""Unit tests for PlatynUI desktop session support in session_models (ADR-025).

Covers:
- SessionType.DESKTOP_TESTING existence + profile (search_order).
- detect_session_type_from_scenario desktop vs web routing.
- ExecutionSession.is_desktop_session().
- _get_profile_for_preferences PlatynUI handling.

Run with: uv run pytest tests/unit/test_platynui_newcore_session.py -q
"""

__test__ = True

import pytest

from robotmcp.models.session_models import (
    ExecutionSession,
    PlatformType,
    SessionType,
)


# =============================================================================
# SessionType + profile
# =============================================================================


class TestDesktopProfile:
    def test_desktop_testing_enum_exists(self):
        assert SessionType.DESKTOP_TESTING.value == "desktop_testing"

    def test_profile_registered(self):
        profiles = ExecutionSession._get_session_profiles()
        assert SessionType.DESKTOP_TESTING in profiles

    def test_platynui_first_in_search_order(self):
        profiles = ExecutionSession._get_session_profiles()
        profile = profiles[SessionType.DESKTOP_TESTING]
        assert profile.search_order[0] == "PlatynUI.BareMetal"

    def test_platynui_in_core_libraries(self):
        profiles = ExecutionSession._get_session_profiles()
        profile = profiles[SessionType.DESKTOP_TESTING]
        assert "PlatynUI.BareMetal" in profile.core_libraries
        assert "BuiltIn" in profile.core_libraries


# =============================================================================
# detect_session_type_from_scenario
# =============================================================================


class TestDetectSessionType:
    @pytest.fixture
    def session(self):
        return ExecutionSession(session_id="s1")

    def test_gnome_calculator_platynui_scenario_is_desktop(self, session):
        scenario = (
            "Automate the gnome calculator desktop application using PlatynUI"
        )
        assert (
            session.detect_session_type_from_scenario(scenario)
            == SessionType.DESKTOP_TESTING
        )

    def test_native_desktop_automation_is_desktop(self, session):
        scenario = "Native desktop UI automation testing of a notepad app"
        assert (
            session.detect_session_type_from_scenario(scenario)
            == SessionType.DESKTOP_TESTING
        )

    def test_window_management_is_desktop(self, session):
        scenario = "Activate window then maximize window using PlatynUI BareMetal"
        assert (
            session.detect_session_type_from_scenario(scenario)
            == SessionType.DESKTOP_TESTING
        )

    def test_browser_scenario_still_web(self, session):
        scenario = (
            "Open browser, navigate to the login page, fill the form and click submit "
            "using Playwright"
        )
        assert (
            session.detect_session_type_from_scenario(scenario)
            == SessionType.WEB_AUTOMATION
        )

    def test_empty_scenario_unknown(self, session):
        assert (
            session.detect_session_type_from_scenario("") == SessionType.UNKNOWN
        )


# =============================================================================
# is_desktop_session
# =============================================================================


class TestIsDesktopSession:
    def test_via_platform_type(self):
        s = ExecutionSession(session_id="s1", platform_type=PlatformType.DESKTOP)
        assert s.is_desktop_session() is True

    def test_via_session_type(self):
        s = ExecutionSession(
            session_id="s1", session_type=SessionType.DESKTOP_TESTING
        )
        assert s.is_desktop_session() is True

    def test_via_imported_platynui_library(self):
        s = ExecutionSession(session_id="s1")
        s.imported_libraries.append("PlatynUI.BareMetal")
        assert s.is_desktop_session() is True

    def test_via_imported_platynui_alias_case_insensitive(self):
        s = ExecutionSession(session_id="s1")
        s.imported_libraries.append("platynui")
        assert s.is_desktop_session() is True

    def test_web_session_not_desktop(self):
        s = ExecutionSession(session_id="s1", platform_type=PlatformType.WEB)
        s.imported_libraries.append("Browser")
        assert s.is_desktop_session() is False


# =============================================================================
# _get_profile_for_preferences
# =============================================================================


class TestProfileForPreferences:
    def _profiles(self):
        return ExecutionSession._get_session_profiles()

    def test_platynui_baremetal_sets_desktop(self):
        s = ExecutionSession(session_id="s1")
        s.explicit_library_preference = "PlatynUI.BareMetal"
        profile = s._get_profile_for_preferences(self._profiles())
        assert s.session_type == SessionType.DESKTOP_TESTING
        assert s.platform_type == PlatformType.DESKTOP
        assert profile.session_type == SessionType.DESKTOP_TESTING

    def test_platynui_alias_sets_desktop(self):
        s = ExecutionSession(session_id="s1")
        s.explicit_library_preference = "PlatynUI"
        profile = s._get_profile_for_preferences(self._profiles())
        assert s.session_type == SessionType.DESKTOP_TESTING
        assert s.platform_type == PlatformType.DESKTOP
        assert profile is self._profiles()[SessionType.DESKTOP_TESTING] or (
            profile.session_type == SessionType.DESKTOP_TESTING
        )

    def test_browser_preference_does_not_set_desktop(self):
        s = ExecutionSession(session_id="s1")
        s.explicit_library_preference = "Browser"
        s._get_profile_for_preferences(self._profiles())
        assert s.session_type == SessionType.WEB_AUTOMATION
        assert s.platform_type != PlatformType.DESKTOP

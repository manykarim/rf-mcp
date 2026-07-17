"""Unit tests: desktop scenario classification precedence
(change: platynui-desktop-safety-isolation, tasks 1.1/2.5)."""

import pytest

from robotmcp.models.session_models import ExecutionSession, SessionType


@pytest.fixture
def session():
    return ExecutionSession(session_id="cls-test")


@pytest.mark.parametrize("scenario,expected", [
    # Desktop signals, no explicit mobile signal -> desktop
    ("Test the calculator app", SessionType.DESKTOP_TESTING),
    ("Open GNOME Calculator and perform calculations", SessionType.DESKTOP_TESTING),
    ("Automate the gnome text editor", SessionType.DESKTOP_TESTING),
    ("Test the calculator app on windows", SessionType.DESKTOP_TESTING),
    ("desktop automation of a native application", SessionType.DESKTOP_TESTING),
    ("launch myapp.exe and click buttons", SessionType.DESKTOP_TESTING),
    # Explicit mobile signal wins even with a desktop noun
    ("Test the calculator app on android", SessionType.MOBILE_TESTING),
    ("Open the iOS calculator app", SessionType.MOBILE_TESTING),
    ("mobile app in emulator", SessionType.MOBILE_TESTING),
    ("tap and swipe on the device", SessionType.MOBILE_TESTING),
    ("run the appium test", SessionType.MOBILE_TESTING),
])
def test_classification_precedence(session, scenario, expected):
    assert session.detect_session_type_from_scenario(scenario) == expected


def test_bare_app_with_desktop_does_not_force_mobile(session):
    # "calculator" is a desktop signal; bare "app" must not win for mobile.
    assert session.detect_session_type_from_scenario(
        "Test the calculator app"
    ) == SessionType.DESKTOP_TESTING


def test_no_desktop_no_mobile_falls_through(session):
    # Neither signal -> precedence returns None, weighted scoring decides.
    result = session._desktop_vs_mobile_precedence("run a sql database query")
    assert result is None


def test_desktop_session_allows_process_and_platynui(session):
    session.session_type = SessionType.DESKTOP_TESTING
    allowed = session._get_allowed_libraries_for_session_type()
    assert "Process" in allowed
    assert "PlatynUI.BareMetal" in allowed
    assert "BuiltIn" in allowed

"""Desktop replay-environment preamble
(change: desktop-suite-replay-environment).

The 2026-06-12 standalone replay failed from a plain Wayland shell
("ProviderError: Wayland screenshot provider: not yet implemented"):
generated desktop suites carried no display/backend pinning. They now emit
a `Prepare Desktop Display Environment` keyword wired as Suite Setup.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from robotmcp.components.test_builder import TestBuilder
from robotmcp.models.execution_models import ExecutionStep
from robotmcp.models.session_models import ExecutionSession

KW = "Prepare Desktop Display Environment"


@pytest.fixture(autouse=True)
def _force_posix(monkeypatch):
    """Windows-CI Linux-model guard.

    These tests validate the Linux/X11 replay preamble. On a Windows host
    ``desktop_display_safety.classify_bound_display_detailed`` short-circuits
    to ``display=None`` (change: fix-platynui-windows-runtime, F4), so the
    DISPLAY pin is dropped and the generated preamble differs. Force the
    non-Windows classification path (inverse of the ``os.name="nt"`` idiom in
    tests/unit/test_platynui_windows_runtime.py) so the Linux-model assertions
    exercise the intended path on any host. Patch the narrow ``_is_windows``
    helper rather than ``os.name`` so ``pathlib`` still uses native paths.
    """
    from robotmcp.components.execution import desktop_display_safety as dds

    monkeypatch.setattr(dds, "_is_windows", lambda: False)


def _desktop_session(sid, suite_setup=None):
    sess = ExecutionSession(session_id=sid)
    sess.configure_from_scenario(
        "Open LibreOffice Writer desktop application", context="desktop"
    )
    if suite_setup:
        sess.suite_setup = suite_setup
    sess.test_registry.start_test("T")
    for kw, args in [
        ("Start Process", ["soffice", "--writer"]),
        ("PlatynUI.BareMetal.Keyboard Type", ["${None}", "hello"]),
    ]:
        st = ExecutionStep(step_id=kw, keyword=kw, arguments=args)
        st.mark_success()
        sess.test_registry.tests["T"].steps.append(st)
    sess.test_registry.end_test(status="pass")
    return sess


async def _build(sess, **kwargs):
    engine = MagicMock()
    engine.sessions = {sess.session_id: sess}
    builder = TestBuilder(execution_engine=engine)
    return await builder.build_suite(
        session_id=sess.session_id, test_name="", **kwargs
    )


@pytest.mark.asyncio
class TestReplayEnvironmentPreamble:
    async def test_desktop_suite_is_self_sufficient(self, monkeypatch):
        monkeypatch.setenv("DISPLAY", ":100")
        monkeypatch.setenv("ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY", ":100")
        sess = _desktop_session("replay-1")
        result = await _build(sess)
        assert result["success"] is True
        assert result["replay_environment"] == "wired"
        rf = result["rf_text"]
        assert "Library         OperatingSystem" in rf
        assert f"Suite Setup     {KW}" in rf
        assert "*** Keywords ***" in rf
        assert KW in rf
        for line in (
            "Set Environment Variable    DISPLAY    :100",
            "Set Environment Variable    XDG_SESSION_TYPE    x11",
            "Set Environment Variable    GDK_BACKEND    x11",
            "Set Environment Variable    QT_QPA_PLATFORM    xcb",
            "Set Environment Variable    GTK_A11Y    atspi",
            "Remove Environment Variable    WAYLAND_DISPLAY",
        ):
            assert line in rf, f"missing: {line}"

    async def test_gtk_a11y_pins_atspi_backend_not_1(self, monkeypatch):
        # Harness finding (change: desktop-a11y-atspi-backend): modern GTK
        # rejects GTK_A11Y=1 and then exposes NO AT-SPI tree; the pin MUST be
        # the backend name "atspi" so a freshly launched GTK AUT is inspectable
        # when the suite replays on a bare env.
        monkeypatch.setenv("DISPLAY", ":100")
        sess = _desktop_session("replay-a11y")
        rf = (await _build(sess))["rf_text"]
        assert "Set Environment Variable    GTK_A11Y    atspi" in rf
        assert "Set Environment Variable    GTK_A11Y    1" not in rf

    async def test_unknown_display_omits_only_display_pin(self, monkeypatch):
        monkeypatch.delenv("DISPLAY", raising=False)
        sess = _desktop_session("replay-2")
        result = await _build(sess)
        rf = result["rf_text"]
        assert "Set Environment Variable    DISPLAY" not in rf
        assert "Set Environment Variable    GDK_BACKEND    x11" in rf

    async def test_web_suite_untouched(self):
        sess = ExecutionSession(session_id="replay-web")
        sess.test_registry.start_test("W")
        st = ExecutionStep(step_id="s", keyword="Click", arguments=["id=x"])
        st.mark_success()
        sess.test_registry.tests["W"].steps.append(st)
        sess.test_registry.end_test(status="pass")
        result = await _build(sess)
        rf = result["rf_text"]
        assert KW not in rf
        assert result.get("replay_environment") is None

    async def test_user_suite_setup_preserved_with_hint(self, monkeypatch):
        monkeypatch.setenv("DISPLAY", ":100")
        sess = _desktop_session(
            "replay-3", suite_setup={"keyword": "My Setup", "arguments": []}
        )
        result = await _build(sess)
        rf = result["rf_text"]
        assert "Suite Setup     My Setup" in rf
        assert KW in rf  # keyword still emitted, callable manually
        assert result["replay_environment"] == "keyword_only"
        assert "NOT wired" in (result.get("replay_environment_hint") or "")

    async def test_composes_with_bdd_style(self, monkeypatch):
        monkeypatch.setenv("DISPLAY", ":100")
        sess = _desktop_session("replay-4")
        result = await _build(sess, bdd_style=True)
        assert result["success"] is True
        rf = result["rf_text"]
        assert f"Suite Setup     {KW}" in rf
        assert rf.count("*** Keywords ***") == 1  # one merged section


@pytest.mark.asyncio
class TestRemoveEnvVarSemantics:
    async def test_remove_missing_env_var_succeeds(self):
        # The preamble removes WAYLAND_DISPLAY unconditionally — RF's
        # OperatingSystem keyword must tolerate an absent variable.
        from robotmcp.components.execution.execution_coordinator import (
            ExecutionCoordinator,
        )

        engine = ExecutionCoordinator()
        sid = "replay-rmenv"
        sess = engine.session_manager.get_or_create_session(sid)
        sess.search_order = ["BuiltIn", "OperatingSystem"]
        r = await engine.execute_step(
            "Remove Environment Variable",
            ["SURELY_NOT_SET_ROBOTMCP_XYZ"],
            sid,
            use_context=True,
        )
        assert r["success"] is True, r.get("error")


class TestReplayRenderingFixes:
    """Two defects found by the replay smoke (change: desktop-suite-
    replay-environment): unescaped dash-args misparse as named arguments,
    and first-dot prefix removal mangled dotted library names."""

    def test_dash_arg_with_equals_escaped_at_render(self):
        builder = TestBuilder.__new__(TestBuilder)
        out = builder._escape_robot_argument(
            "-env:UserInstallation=file:///tmp/p"
        )
        assert out == "-env:UserInstallation\\=file:///tmp/p"

    def test_already_escaped_dash_arg_untouched(self):
        builder = TestBuilder.__new__(TestBuilder)
        arg = "-env:UserInstallation\\=file:///tmp/p"
        assert builder._escape_robot_argument(arg) == arg

    def test_named_argument_not_escaped(self):
        builder = TestBuilder.__new__(TestBuilder)
        assert builder._escape_robot_argument("alias=smokeproc") == "alias=smokeproc"

    def test_dotted_library_prefix_removed_fully(self):
        builder = TestBuilder.__new__(TestBuilder)
        assert builder._remove_library_prefix(
            "PlatynUI.BareMetal.Take Screenshot"
        ) == "Take Screenshot"
        assert builder._remove_library_prefix("Browser.Click") == "Click"
        assert builder._remove_library_prefix("Click") == "Click"

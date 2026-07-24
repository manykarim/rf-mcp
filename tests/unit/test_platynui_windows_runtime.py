"""Unit tests for the PlatynUI Windows runtime fixes (change:
fix-platynui-windows-runtime).

Covers:
- F16: stuck-key release safety net (release_all_modifiers, on_session_end,
  atexit registration, keyword-failure release, keyboard-keyword detection).
- F3: short default desktop query timeout via ${PLATYNUI_QUERY_SETTINGS},
  honoring an explicit timeout_ms.
- F14: desktop pre-dispatch native queries are offloaded off the event loop.
- F4: platform-aware safety guard (Windows classification + allow-by-default +
  strict opt-in + no nested-X-server recipe; Linux unchanged).

Run with: uv run pytest tests/unit/test_platynui_windows_runtime.py -q
"""

__test__ = True

import asyncio
import inspect
import os
from unittest.mock import MagicMock, patch

import pytest

from robotmcp.components.execution import desktop_display_safety as dds
from robotmcp.components.execution.keyword_executor import KeywordExecutor
from robotmcp.models.config_models import ExecutionConfig
from robotmcp.plugins.builtin import platynui_plugin as pnp


@pytest.fixture
def executor():
    return KeywordExecutor(config=ExecutionConfig())


class _DesktopSession:
    session_id = "desk-1"
    imported_libraries = ["PlatynUI.BareMetal", "BuiltIn"]
    variables: dict = {}

    def is_desktop_session(self):
        return True


class _WebSession:
    session_id = "web-1"
    imported_libraries = ["Browser", "BuiltIn"]
    variables: dict = {}

    def is_desktop_session(self):
        return False


# =============================================================================
# F16 — stuck-key release safety net
# =============================================================================


class TestReleaseAllModifiers:
    def test_noop_when_runtime_not_open(self):
        with patch.object(pnp, "_RUNTIME", None), patch.object(
            pnp, "_RUNTIME_STATE", "new"
        ):
            assert pnp.release_all_modifiers() is False

    def test_noop_when_disposed(self):
        rt = MagicMock()
        with patch.object(pnp, "_RUNTIME", rt), patch.object(
            pnp, "_RUNTIME_STATE", "disposed"
        ):
            assert pnp.release_all_modifiers() is False
            rt.keyboard_release.assert_not_called()

    def test_dispatches_release_sequence_when_open(self):
        rt = MagicMock()
        with patch.object(pnp, "_RUNTIME", rt), patch.object(
            pnp, "_RUNTIME_STATE", "open"
        ):
            assert pnp.release_all_modifiers() is True
            rt.keyboard_release.assert_called_once_with(pnp._RELEASE_ALL_SEQUENCE)

    def test_release_sequence_covers_all_modifiers(self):
        seq = pnp._RELEASE_ALL_SEQUENCE.lower()
        for mod in ("lctrl", "rctrl", "lalt", "ralt", "lshift", "rshift", "lwin", "rwin"):
            assert mod in seq

    def test_never_raises_on_backend_error(self):
        rt = MagicMock()
        rt.keyboard_release.side_effect = RuntimeError("boom")
        with patch.object(pnp, "_RUNTIME", rt), patch.object(
            pnp, "_RUNTIME_STATE", "open"
        ):
            assert pnp.release_all_modifiers() is False


class TestReleaseHandlerRegistration:
    def test_registers_atexit_once(self):
        with patch.object(pnp, "_RELEASE_HANDLERS_REGISTERED", False), patch(
            "atexit.register"
        ) as reg, patch("signal.signal"), patch(
            "signal.getsignal", return_value=None
        ):
            pnp._register_release_handlers_once()
            pnp._register_release_handlers_once()
            reg.assert_called_once_with(pnp.release_tracked_keys)

    def test_on_session_start_arms_handlers_on_main_thread(self):
        """The F14 to_thread offload can push the first runtime bind onto a
        worker thread (where signal.signal fails), so on_session_start must arm
        the handlers on the main thread for the SIGTERM net to be reliable."""

        class _S:
            imported_libraries = ["PlatynUI.BareMetal"]
            search_order: list = []
            explicit_library_preference = ""

        plugin = pnp.PlatynUILibraryPlugin()
        with patch.object(pnp, "_register_release_handlers_once") as reg, patch.object(
            pnp, "ensure_x11_session_env"
        ), patch.object(pnp, "release_all_modifiers"):
            plugin.on_session_start(_S())
            reg.assert_called_once()

    def test_on_session_start_skips_handlers_for_non_platynui(self):
        class _S:
            imported_libraries = ["Browser"]
            search_order: list = []
            explicit_library_preference = ""

        plugin = pnp.PlatynUILibraryPlugin()
        with patch.object(pnp, "_register_release_handlers_once") as reg, patch.object(
            pnp, "ensure_x11_session_env"
        ), patch.object(pnp, "release_all_modifiers"):
            plugin.on_session_start(_S())
            reg.assert_not_called()


class TestPluginSessionEnd:
    def test_on_session_end_releases_modifiers(self):
        plugin = pnp.PlatynUILibraryPlugin()
        with patch.object(pnp, "release_tracked_keys") as rel:
            plugin.on_session_end(_DesktopSession())
            rel.assert_called_once()

    def test_on_session_end_never_raises(self):
        plugin = pnp.PlatynUILibraryPlugin()
        with patch.object(pnp, "release_tracked_keys", side_effect=RuntimeError):
            plugin.on_session_end(_DesktopSession())  # must not raise


class TestDesktopKeyboardDetection:
    @pytest.mark.parametrize(
        "kw",
        ["Keyboard Press", "keyboard type", "PlatynUI.BareMetal.Keyboard Release"],
    )
    def test_desktop_keyboard_keyword_true(self, executor, kw):
        assert executor._is_desktop_keyboard_keyword(_DesktopSession(), kw) is True

    def test_non_keyboard_desktop_keyword_false(self, executor):
        assert (
            executor._is_desktop_keyboard_keyword(_DesktopSession(), "Pointer Click")
            is False
        )

    def test_web_keyboard_keyword_false(self, executor):
        # Not a desktop session -> never a desktop keyboard keyword.
        assert (
            executor._is_desktop_keyboard_keyword(_WebSession(), "Keyboard Type")
            is False
        )


class TestKeywordFailureRelease:
    """The _execute_keyword wrapper must release held modifiers when a desktop
    keyboard keyword fails or raises, but NOT on success."""

    def _run(self, executor, result_or_exc):
        async def _drive():
            return await executor.execute_keyword(
                _DesktopSession(), "Keyboard Type", ["<Ctrl+A>"], None
            )

        if isinstance(result_or_exc, Exception):

            async def _serialized(*a, **k):
                raise result_or_exc

        else:

            async def _serialized(*a, **k):
                return result_or_exc

        with patch.object(
            executor, "_execute_keyword_serialized", _serialized
        ), patch.object(executor, "_release_desktop_keys") as rel:
            loop = asyncio.new_event_loop()
            try:
                if isinstance(result_or_exc, Exception):
                    with pytest.raises(type(result_or_exc)):
                        loop.run_until_complete(_drive())
                else:
                    loop.run_until_complete(_drive())
            finally:
                loop.close()
            return rel

    def test_release_on_failure_result(self, executor):
        rel = self._run(executor, {"success": False, "error": "nope"})
        rel.assert_called_once()

    def test_release_on_exception(self, executor):
        rel = self._run(executor, RuntimeError("killed mid-chord"))
        rel.assert_called_once()

    def test_no_release_on_success(self, executor):
        rel = self._run(executor, {"success": True, "output": None})
        rel.assert_not_called()

    def test_no_release_on_steering_downgrade(self, executor):
        # F2: steering-confidence downgraded a Keyboard Press whose native
        # press actually succeeded (key held) — must NOT release the modifier,
        # or a deliberate Press/Release chord is corrupted.
        rel = self._run(
            executor,
            {"success": False, "error": "not verified", "steering_confidence": "contradicted"},
        )
        rel.assert_not_called()

    def test_release_on_genuine_failure_with_no_steering(self, executor):
        # A genuine native failure (no steering marker) still releases.
        rel = self._run(executor, {"success": False, "error": "descriptor not found"})
        rel.assert_called_once()

    def test_failure_result_gets_steering_hint(self, executor):
        async def _serialized(*a, **k):
            return {"success": False, "error": "nope"}

        async def _drive():
            return await executor.execute_keyword(
                _DesktopSession(), "Keyboard Press", ["<Ctrl>"], None
            )

        with patch.object(
            executor, "_execute_keyword_serialized", _serialized
        ), patch.object(executor, "_release_desktop_keys"):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(_drive())
            finally:
                loop.close()
        hints = result.get("hints", [])
        assert any(
            h.get("type") == "platynui_keyboard_release_safety" for h in hints
        )


# =============================================================================
# F3 — short default desktop query timeout
# =============================================================================


def _bare(name):
    return name[2:-1] if isinstance(name, str) and name.startswith("${") and name.endswith("}") else name


class _FakeVars:
    """Mimics RF VariableScopes for the bits F3 uses: set_suite(children=True)
    writes a value that a bare-name membership check + decorated getitem see
    (the library's exact access pattern)."""

    def __init__(self):
        self._store = {}  # keyed by BARE name (as RF __contains__ resolves)
        self.set_suite_calls = []

    def set_suite(self, name, value, top=False, children=False):
        self.set_suite_calls.append({"name": name, "value": value, "children": children})
        self._store[_bare(name)] = value

    def __contains__(self, name):  # library uses the BARE name
        return _bare(name) in self._store

    def __getitem__(self, name):  # library reads with the decorated name
        return self._store[_bare(name)]


class _FakeCtx:
    def __init__(self):
        self.variables = _FakeVars()


class TestDesktopQueryTimeout:
    def _apply(self, executor, session, ctx=None):
        pytest.importorskip("PlatynUI.BareMetal")
        ctx = ctx or _FakeCtx()
        fake_ec = MagicMock()
        fake_ec.current = ctx
        with patch("robot.running.context.EXECUTION_CONTEXTS", fake_ec):
            executor._ensure_desktop_query_timeout(session)
        return ctx

    def _applied(self, ctx):
        from PlatynUI.BareMetal import PLATYNUI_QUERY_SETTINGS

        name = f"${{{PLATYNUI_QUERY_SETTINGS}}}"
        return ctx.variables[name]

    def test_sets_short_default_via_set_suite_children(self, executor):
        pytest.importorskip("PlatynUI.BareMetal")
        from PlatynUI.BareMetal import QuerySettings

        ctx = self._apply(executor, _DesktopSession())
        # F1: must use set_suite with children=True (visible at current + persists)
        calls = ctx.variables.set_suite_calls
        assert len(calls) == 1 and calls[0]["children"] is True
        settings = self._applied(ctx)
        assert isinstance(settings, QuerySettings)
        assert settings.timeout == pytest.approx(1.5)

    def test_honors_explicit_timeout_ms(self, executor):
        session = _DesktopSession()
        session._platynui_query_timeout_ms = 8000
        ctx = self._apply(executor, session)
        assert self._applied(ctx).timeout == pytest.approx(8.0)

    def test_env_override(self, executor):
        with patch.dict(os.environ, {"ROBOTMCP_PLATYNUI_QUERY_TIMEOUT_MS": "500"}):
            ctx = self._apply(executor, _DesktopSession())
        assert self._applied(ctx).timeout == pytest.approx(0.5)

    def test_f6_preserves_existing_non_timeout_fields(self, executor):
        # A prior QuerySettings with custom retry_interval/ignore_exceptions must
        # survive — only the timeout is ours to change.
        pytest.importorskip("PlatynUI.BareMetal")
        from PlatynUI.BareMetal import PLATYNUI_QUERY_SETTINGS, QuerySettings

        ctx = _FakeCtx()
        prior = QuerySettings(timeout=30.0, retry_interval=0.5, ignore_exceptions=True)
        ctx.variables.set_suite(f"${{{PLATYNUI_QUERY_SETTINGS}}}", prior, children=True)
        ctx.variables.set_suite_calls.clear()
        ctx = self._apply(executor, _DesktopSession(), ctx=ctx)
        applied = self._applied(ctx)
        assert applied.timeout == pytest.approx(1.5)
        assert applied.retry_interval == pytest.approx(0.5)
        assert applied.ignore_exceptions is True

    def test_leak_default_restored_after_explicit(self, executor):
        # F3-leak: an explicit long timeout must not stick — once the per-call
        # stash is cleared, the short default is re-applied.
        session = _DesktopSession()
        session._platynui_query_timeout_ms = 120000
        ctx = self._apply(executor, session)
        assert self._applied(ctx).timeout == pytest.approx(120.0)
        # Next step: no explicit timeout (stash reset to None by the executor)
        session._platynui_query_timeout_ms = None
        ctx = self._apply(executor, session, ctx=ctx)
        assert self._applied(ctx).timeout == pytest.approx(1.5)

    def test_noop_for_web_session(self, executor):
        with patch("robot.running.context.EXECUTION_CONTEXTS") as ec:
            executor._ensure_desktop_query_timeout(_WebSession())
            ec.current.__bool__.assert_not_called() if hasattr(
                ec.current, "__bool__"
            ) else None

    def test_idempotent_when_value_unchanged(self, executor):
        pytest.importorskip("PlatynUI.BareMetal")
        session = _DesktopSession()
        session._platynui_query_timeout_applied_ms = 1500
        ctx = _FakeCtx()
        fake_ec = MagicMock()
        fake_ec.current = ctx
        with patch("robot.running.context.EXECUTION_CONTEXTS", fake_ec):
            executor._ensure_desktop_query_timeout(session)
        # applied flag already matches default -> nothing written
        assert ctx.variables.set_suite_calls == []

    def test_visible_to_library_with_real_rf_scopes(self, executor):
        """F1 regression guard against REAL RF internals: rf-mcp holds a live
        TEST scope copied from suite BEFORE the write, and PlatynUI resolves the
        variable via the current (test) scope with a BARE-name membership check.
        A plain suite-store write is invisible; set_suite(children=True) is not.
        Verifies with the library's exact access pattern."""
        pytest.importorskip("PlatynUI.BareMetal")
        from robot.conf.settings import RobotSettings
        from robot.variables.scopes import VariableScopes
        from PlatynUI.BareMetal import PLATYNUI_QUERY_SETTINGS

        vs = VariableScopes(RobotSettings())
        vs.start_suite()
        vs.start_test()  # rf-mcp's real state: test copied from suite
        ctx = MagicMock()
        ctx.variables = vs
        fake_ec = MagicMock()
        fake_ec.current = ctx
        with patch("robot.running.context.EXECUTION_CONTEXTS", fake_ec):
            executor._ensure_desktop_query_timeout(_DesktopSession())
        # The library's own read: bare-name membership + decorated getitem.
        assert PLATYNUI_QUERY_SETTINGS in vs
        assert vs[f"${{{PLATYNUI_QUERY_SETTINGS}}}"].timeout == pytest.approx(1.5)


# =============================================================================
# F14 — desktop pre-dispatch native queries offloaded off the event loop
# =============================================================================


class TestEventLoopOffload:
    def test_focus_and_text_count_wrapped_in_to_thread(self):
        """Regression guard: the two synchronous native pre-dispatch queries
        are dispatched via asyncio.to_thread (whitespace-tolerant)."""
        import re

        src = inspect.getsource(KeywordExecutor._execute_keyword_serialized)
        assert re.search(
            r"to_thread\(\s*self\._platynui_focus_before_act", src
        ), "focus query must be offloaded via asyncio.to_thread"
        assert re.search(
            r"to_thread\(\s*self\._desktop_text_count_before", src
        ), "text-count query must be offloaded via asyncio.to_thread"

    def test_slow_focus_does_not_block_event_loop(self, executor):
        """Behavioural proof: a slow (blocking) focus query must NOT stall the
        event loop — a concurrent coroutine keeps ticking while focus runs off
        the loop. If the offload were removed, ticks_during would be 0."""
        import threading
        import time

        order = []

        def slow_focus(session, keyword, arguments):
            order.append(("focus_start", time.monotonic()))
            time.sleep(0.2)  # blocking native query, off the loop
            order.append(("focus_end", time.monotonic()))
            return None

        async def ticker():
            for _ in range(40):
                await asyncio.sleep(0.005)
                order.append(("tick", time.monotonic()))

        async def fake_ctx_exec(*a, **k):
            return {"success": True, "output": None}

        session = _DesktopSession()
        session.platynui_allow_active_desktop = True
        session.desktop_tree_dirty = False

        with patch(
            "robotmcp.plugins.builtin.platynui_plugin.ensure_x11_session_env"
        ), patch.object(
            executor, "_platynui_focus_before_act", slow_focus
        ), patch.object(
            executor, "_desktop_text_count_before", return_value=None
        ), patch.object(
            executor, "_ensure_library_registration", return_value=None
        ), patch.object(
            executor, "_execute_keyword_with_context", fake_ctx_exec
        ):

            async def run():
                t = asyncio.create_task(ticker())
                res = await executor.execute_keyword(
                    session, "Keyboard Type", ["<Ctrl+A>"], None
                )
                await t
                return res

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(run())
            finally:
                loop.close()

        starts = [t for (n, t) in order if n == "focus_start"]
        ends = [t for (n, t) in order if n == "focus_end"]
        ticks = [t for (n, t) in order if n == "tick"]
        assert starts and ends, "focus was not invoked"
        ticks_during = [t for t in ticks if starts[0] < t < ends[0]]
        # ~0.2s of blocking focus / 5ms ticks -> tens of ticks if the loop is
        # free; require a solid margin above 0 to prove the offload.
        assert len(ticks_during) >= 5, (
            f"event loop was blocked during focus "
            f"(ticks_during={len(ticks_during)})"
        )


# =============================================================================
# F4 — platform-aware safety guard
# =============================================================================


class _GuardSession:
    session_id = "g-1"


@pytest.fixture
def _windows():
    with patch.object(os, "name", "nt"):
        yield


class TestWindowsClassification:
    def test_classify_windows(self, _windows):
        det = dds.classify_bound_display_detailed({})
        assert det["isolation"] == dds.WINDOWS
        assert det["isolation_source"] == "windows_console"

    def test_classify_bound_display_returns_windows(self, _windows):
        assert dds.classify_bound_display({}) == dds.WINDOWS


class TestWindowsEvaluateSafety:
    def test_allowed_by_default(self, _windows):
        ev = dds.evaluate_safety(_GuardSession(), {})
        assert ev["allowed"] is True
        assert ev["bypassed"] is False
        assert ev["enforcing"] is True
        assert ev["classification"] == dds.WINDOWS

    def test_no_nested_x_server_recipe(self, _windows):
        ev = dds.evaluate_safety(_GuardSession(), {})
        assert "isolation_recipe" not in ev
        blob = (ev.get("reason") or "") + (ev.get("windows_note") or "")
        assert "ephyr" not in blob and "Xvfb" not in blob and "xvfb" not in blob
        assert "RDP" in blob or "dedicated" in blob

    def test_strict_opt_in_refuses(self, _windows):
        ev = dds.evaluate_safety(
            _GuardSession(), {"ROBOTMCP_PLATYNUI_REQUIRE_ISOLATED": "1"}
        )
        assert ev["allowed"] is False
        assert "isolation_recipe" not in ev


class TestLinuxUnchanged:
    def test_active_desktop_still_refused(self):
        # os.name is 'posix' here; an active EWMH desktop with no marker refuses.
        with patch.object(
            dds, "classify_bound_display", return_value=dds.ACTIVE
        ):
            ev = dds.evaluate_safety(_GuardSession(), {})
        assert ev["allowed"] is False
        assert ev["classification"] == dds.ACTIVE
        assert "isolation_recipe" in ev  # Linux keeps the recipe

    def test_isolated_still_allowed(self):
        with patch.object(
            dds, "classify_bound_display", return_value=dds.ISOLATED
        ):
            ev = dds.evaluate_safety(_GuardSession(), {})
        assert ev["allowed"] is True
        assert ev["bypassed"] is False


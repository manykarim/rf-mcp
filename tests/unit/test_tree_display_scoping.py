"""ui_tree display scoping (change: desktop-evidence-and-display-scoping, D4).

The AT-SPI bus is session-global, not display-scoped: on the 2026-06-11
rerun an isolation-marked session's tree contained gnome-shell, Chrome and
keepassxc, and a gnome-shell dash icon labeled "LibreOffice Writer" misled
the agent. Isolated sessions must list only applications whose process owns
a window on the bound display — fail-open for PID-less apps, degrade to
unfiltered when the probe is unavailable.
"""

from __future__ import annotations

import pytest

import robotmcp.components.execution.ui_tree_service as u
import robotmcp.plugins.builtin.platynui_plugin as plugin


class FakeApp:
    def __init__(self, name, pid=None, role="Application"):
        self.name = name
        self.role = role
        self._pid = pid

    def attribute(self, attr):
        if attr == "ProcessId" and self._pid is not None:
            return self._pid
        raise KeyError(attr)

    def children(self):
        return iter(())


class FakeRuntime:
    def __init__(self, apps):
        self._apps = apps

    def evaluate(self, expr):
        return list(self._apps)

    def desktop_info(self):
        return {}

    def clear_cache(self):
        pass


@pytest.fixture()
def isolated_env(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":100")
    monkeypatch.setenv("ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY", ":100")
    # Strict guard: an isolated display requires ownership corroboration
    # (change: desktop-isolation-marker-hardening). This fixture represents a
    # legitimately rf-mcp-owned display, so corroborate the marker.
    import robotmcp.components.execution.desktop_display_safety as _dds
    monkeypatch.setattr(_dds, "_marker_ownership_status", lambda e, d: "verified")
    u.clear_display_pid_cache()
    yield
    u.clear_display_pid_cache()


def _collect(monkeypatch, apps, scoped_pids, aut_pid=None):
    monkeypatch.setattr(plugin, "get_runtime", lambda: FakeRuntime(apps))
    monkeypatch.setattr(u, "_display_scoped_pids", lambda: scoped_pids)
    return u._collect_ui_tree_sync(None, 3, 20, 200, aut_pid=aut_pid)


class TestDisplayScoping:
    def test_host_apps_filtered(self, monkeypatch, isolated_env):
        apps = [
            FakeApp("gnome-shell", pid=2000),
            FakeApp("Google Chrome", pid=2001),
            FakeApp("soffice", pid=5000),
        ]
        res = _collect(monkeypatch, apps, frozenset({5000}))
        assert res["success"] is True
        names = [a["name"] for a in res["applications"]]
        assert names == ["soffice"]
        assert res["host_apps_filtered"] == 2

    def test_pidless_app_kept_and_annotated(self, monkeypatch, isolated_env):
        apps = [FakeApp("mystery-app", pid=None), FakeApp("soffice", pid=5000)]
        res = _collect(monkeypatch, apps, frozenset({5000}))
        names = {a["name"]: a for a in res["applications"]}
        assert "mystery-app" in names
        assert names["mystery-app"]["display_scoped"] is False
        assert "display_scoped" not in names["soffice"]
        assert res["host_apps_filtered"] == 0

    def test_probe_unavailable_degrades_to_unfiltered(
        self, monkeypatch, isolated_env
    ):
        apps = [FakeApp("gnome-shell", pid=2000), FakeApp("soffice", pid=5000)]
        res = _collect(monkeypatch, apps, None)
        assert len(res["applications"]) == 2
        assert res["display_scoping"] == "unavailable"
        assert "host_apps_filtered" not in res

    def test_active_display_not_filtered(self, monkeypatch):
        # No isolation marker -> classification is not marker-isolated.
        monkeypatch.setenv("DISPLAY", ":0")
        monkeypatch.delenv("ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY", raising=False)
        from robotmcp.components.execution import desktop_display_safety as dds

        monkeypatch.setattr(
            dds, "_ewmh_wm_present", lambda display: True
        )  # active desktop
        apps = [FakeApp("gnome-shell", pid=2000), FakeApp("app", pid=3000)]
        monkeypatch.setattr(plugin, "get_runtime", lambda: FakeRuntime(apps))
        called = []
        monkeypatch.setattr(
            u, "_display_scoped_pids", lambda: called.append(1) or frozenset()
        )
        res = u._collect_ui_tree_sync(None, 3, 20, 200)
        assert len(res["applications"]) == 2
        assert called == []  # probe never invoked
        assert "host_apps_filtered" not in res

    def test_aut_pid_always_in_scope(self, monkeypatch, isolated_env):
        # Post-launch the AUT's window may not be mapped yet — the launched
        # PID must never be filtered as a host app.
        apps = [FakeApp("soffice", pid=5000), FakeApp("gnome-shell", pid=2000)]
        res = _collect(monkeypatch, apps, frozenset(), aut_pid=5000)
        names = [a["name"] for a in res["applications"]]
        assert names == ["soffice"]
        assert res["host_apps_filtered"] == 1


class TestPidCache:
    def test_probe_result_cached_per_display(self, monkeypatch, isolated_env):
        calls = []

        def fake_probe():
            calls.append(1)
            return frozenset({1, 2})

        import robotmcp.components.execution.platynui_focus as focus_mod

        monkeypatch.setattr(focus_mod, "x11_display_pids", fake_probe)
        assert u._display_scoped_pids() == frozenset({1, 2})
        assert u._display_scoped_pids() == frozenset({1, 2})
        assert len(calls) == 1  # second call served from cache

    def test_clear_cache_forces_reprobe(self, monkeypatch, isolated_env):
        calls = []
        import robotmcp.components.execution.platynui_focus as focus_mod

        monkeypatch.setattr(
            focus_mod, "x11_display_pids",
            lambda: calls.append(1) or frozenset(),
        )
        u._display_scoped_pids()
        u.clear_display_pid_cache()
        u._display_scoped_pids()
        assert len(calls) == 2

    def test_failed_probe_not_cached(self, monkeypatch, isolated_env):
        calls = []
        import robotmcp.components.execution.platynui_focus as focus_mod

        monkeypatch.setattr(
            focus_mod, "x11_display_pids", lambda: calls.append(1) or None
        )
        assert u._display_scoped_pids() is None
        assert u._display_scoped_pids() is None
        assert len(calls) == 2  # None results retry next call

"""Empty-display exposure diagnostic
(change: desktop-test-scoping-and-close-lifecycle, D6).

Pre-launch on run 3, an empty isolated display was diagnosed as
"(X11 probe unavailable)" — the name-based presence probe returns
"unknown" by contract when given nothing to match. The batched PID probe
distinguishes empty-and-reachable from probe failure.
"""

from __future__ import annotations

import pytest

import robotmcp.components.execution.ui_tree_service as u
import robotmcp.plugins.builtin.platynui_plugin as plugin


@pytest.fixture(autouse=True)
def _stub_env(monkeypatch):
    import robotmcp.components.execution.platynui_focus as focus_mod

    monkeypatch.setattr(plugin, "native_providers", lambda: [])
    # Name-based presence probe returns "unknown" (nothing to match).
    monkeypatch.setattr(focus_mod, "x11_window_present", lambda **k: "unknown")
    u.clear_display_pid_cache()
    yield
    u.clear_display_pid_cache()


def _patch_probe(monkeypatch, value):
    import robotmcp.components.execution.platynui_focus as focus_mod

    monkeypatch.setattr(focus_mod, "x11_display_pids", lambda: value)


class TestEmptyDisplayDiagnostic:
    def test_reachable_empty_display_diagnosed_as_empty(self, monkeypatch):
        _patch_probe(monkeypatch, frozenset())
        diag = u._build_exposure_diagnostic(None)
        assert diag is not None
        assert diag["type"] == "display_empty"
        assert diag["window_present"] is False
        assert "no application windows" in diag["message"]
        assert "X11 probe unavailable" not in diag["message"]

    def test_probe_failure_keeps_undetermined_wording(self, monkeypatch):
        _patch_probe(monkeypatch, None)
        diag = u._build_exposure_diagnostic(None)
        assert diag["type"] == "accessibility_exposure_undetermined"
        assert "X11 probe unavailable" in diag["message"]

    def test_windows_present_keeps_undetermined_path(self, monkeypatch):
        # Client windows exist but nothing resolved by name — the existing
        # undetermined diagnostic applies unchanged.
        _patch_probe(monkeypatch, frozenset({123}))
        diag = u._build_exposure_diagnostic(None)
        assert diag["type"] == "accessibility_exposure_undetermined"

    def test_app_filters_path_untouched(self, monkeypatch):
        # With filters given, "unknown" means the probe really couldn't
        # answer — the PID probe must not reroute this case.
        called = []
        import robotmcp.components.execution.platynui_focus as focus_mod

        monkeypatch.setattr(
            focus_mod, "x11_display_pids",
            lambda: called.append(1) or frozenset(),
        )
        diag = u._build_exposure_diagnostic(["soffice"])
        assert diag["type"] == "accessibility_exposure_undetermined"
        assert called == []


    def test_stale_scoping_cache_does_not_mask_empty_display(self, monkeypatch):
        # Run-4 regression: the AUT exited via Ctrl+Q; the scoping cache
        # still held the dead pid, and the diagnostic (using the cache)
        # reported "(X11 probe unavailable)". The diagnostic must use a
        # FRESH probe.
        u._DISPLAY_PIDS_CACHE[":100"] = frozenset({99999})  # stale entry
        _patch_probe(monkeypatch, frozenset())  # fresh probe: empty now
        diag = u._build_exposure_diagnostic(None)
        assert diag["type"] == "display_empty"

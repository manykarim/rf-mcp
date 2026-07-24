"""Unit tests: isolation-marker ownership hardening
(change: desktop-isolation-marker-hardening)."""

import os

import pytest

import robotmcp.components.execution.desktop_display_safety as ds
from robotmcp.components.execution.desktop_display_safety import (
    ISOLATION_MARKER_ENV,
    ISOLATION_XPID_ENV,
    _marker_ownership_status,
    classify_bound_display_detailed,
    ISOLATED,
    ACTIVE,
    UNKNOWN,
)


# Windows-CI Linux-model guard: these tests validate the X11/EWMH/marker
# isolation model, but classify_bound_display_detailed short-circuits to the
# "windows" classification when _is_windows() is True (change:
# fix-platynui-windows-runtime, F4). Force the non-Windows path so the Linux
# model runs on any host, including Windows CI.
@pytest.fixture(autouse=True)
def _force_posix(monkeypatch):
    monkeypatch.setattr(ds, "_is_windows", lambda: False)


# ── _marker_ownership_status ───────────────────────────────────────────────
def test_ownership_absent_when_no_xpid():
    assert _marker_ownership_status({}, ":99") == "absent"


def test_ownership_invalid_bad_xpid():
    assert _marker_ownership_status({ISOLATION_XPID_ENV: "not-a-pid"}, ":99") == "invalid"
    assert _marker_ownership_status({ISOLATION_XPID_ENV: "0"}, ":99") == "invalid"


def test_ownership_invalid_dead_pid():
    # a PID that (almost certainly) does not exist
    assert _marker_ownership_status({ISOLATION_XPID_ENV: "2147480000"}, ":99") == "invalid"


def test_ownership_invalid_real_pid_not_xserver():
    # our own process is real but is not an X server bound to :99
    assert _marker_ownership_status({ISOLATION_XPID_ENV: str(os.getpid())}, ":99") == "invalid"


def test_ownership_absent_no_display():
    assert _marker_ownership_status({ISOLATION_XPID_ENV: "123"}, None) == "absent"


# ── classify_bound_display_detailed branches ───────────────────────────────
def _classify(monkeypatch, *, status, wm):
    monkeypatch.setattr(ds, "_marker_ownership_status", lambda env, disp: status)
    monkeypatch.setattr(ds, "_ewmh_wm_present", lambda disp: wm)
    env = {"DISPLAY": ":99", ISOLATION_MARKER_ENV: ":99"}
    return classify_bound_display_detailed(env)


def test_verified_marker_is_isolated(monkeypatch):
    r = _classify(monkeypatch, status="verified", wm=True)  # WM present but owned
    assert r["isolation"] == ISOLATED and r["isolation_source"] == "marker"


def test_invalid_marker_active_wm_fails_closed(monkeypatch):
    r = _classify(monkeypatch, status="invalid", wm=True)
    assert r["isolation"] == UNKNOWN
    assert r["isolation_source"] == "marker_over_active_wm"


def test_invalid_marker_no_wm_fails_closed(monkeypatch):
    r = _classify(monkeypatch, status="invalid", wm=False)
    assert r["isolation"] == UNKNOWN
    assert r["isolation_source"] == "marker_invalid"


def test_absent_marker_no_wm_fails_closed(monkeypatch):
    # STRICT: no ownership proof -> refused, even with no WM.
    r = _classify(monkeypatch, status="absent", wm=None)
    assert r["isolation"] == UNKNOWN and r["isolation_source"] == "marker_unverified"


def test_absent_marker_active_wm_fails_closed_flagged(monkeypatch):
    # STRICT: no ownership proof + active WM -> refused, conflict surfaced.
    r = _classify(monkeypatch, status="absent", wm=True)
    assert r["isolation"] == UNKNOWN
    assert r["isolation_source"] == "marker_over_active_wm"


def test_no_marker_active_wm_is_active(monkeypatch):
    monkeypatch.setattr(ds, "_ewmh_wm_present", lambda disp: True)
    r = classify_bound_display_detailed({"DISPLAY": ":0"})
    assert r["isolation"] == ACTIVE and r["isolation_source"] == "ewmh"


def test_no_display_is_unknown():
    r = classify_bound_display_detailed({})
    assert r["isolation"] == UNKNOWN and r["isolation_source"] == "none"

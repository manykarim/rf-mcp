"""Unit tests: active-desktop safety guard
(change: platynui-desktop-safety-isolation, tasks 4.1/4.5)."""

import pytest

import robotmcp.components.execution.desktop_display_safety as dds
from robotmcp.components.execution.desktop_display_safety import (
    ACTIVE,
    ISOLATED,
    UNKNOWN,
    classify_bound_display,
    evaluate_safety,
)


class _Session:
    platynui_allow_active_desktop = False


def _env(**kw):
    base = {"DISPLAY": ":0"}
    base.update(kw)
    return base


def test_marker_yields_isolated(monkeypatch):
    # Marker present + ownership-corroborated -> isolated, even if a WM is
    # present (the legitimate isolated Xvfb+fluxbox case). change:
    # desktop-isolation-marker-hardening.
    monkeypatch.setattr(dds, "_ewmh_wm_present", lambda d: True)  # even if WM present
    monkeypatch.setattr(dds, "_marker_ownership_status", lambda e, d: "verified")
    env = {"DISPLAY": ":99", dds.ISOLATION_MARKER_ENV: ":99"}
    assert classify_bound_display(env) == ISOLATED


def test_marker_without_ownership_is_unknown(monkeypatch):
    # STRICT fail-closed: a marker with no ownership proof is refused.
    monkeypatch.setattr(dds, "_ewmh_wm_present", lambda d: None)
    env = {"DISPLAY": ":99", dds.ISOLATION_MARKER_ENV: ":99"}  # no XPID
    assert classify_bound_display(env) == UNKNOWN


def test_ewmh_wm_present_yields_active(monkeypatch):
    monkeypatch.setattr(dds, "_ewmh_wm_present", lambda d: True)
    assert classify_bound_display(_env()) == ACTIVE


def test_non_ewmh_wm_is_unknown_not_isolated(monkeypatch):
    # The dangerous false-allow: a real desktop with a non-EWMH WM (probe says
    # "no WM") must classify UNKNOWN (refused), NOT isolated.
    monkeypatch.setattr(dds, "_ewmh_wm_present", lambda d: False)
    assert classify_bound_display(_env()) == UNKNOWN


def test_probe_failure_is_unknown(monkeypatch):
    monkeypatch.setattr(dds, "_ewmh_wm_present", lambda d: None)
    assert classify_bound_display(_env()) == UNKNOWN


def test_no_display_is_unknown():
    assert classify_bound_display({}) == UNKNOWN


def test_evaluate_refuses_active(monkeypatch):
    monkeypatch.setattr(dds, "_ewmh_wm_present", lambda d: True)
    out = evaluate_safety(_Session(), _env())
    assert out["classification"] == ACTIVE
    assert out["allowed"] is False
    assert out["enforcing"] is True


def test_evaluate_refuses_unknown(monkeypatch):
    monkeypatch.setattr(dds, "_ewmh_wm_present", lambda d: None)
    out = evaluate_safety(_Session(), _env())
    assert out["classification"] == UNKNOWN
    assert out["allowed"] is False


def test_evaluate_allows_isolated(monkeypatch):
    # Corroborated marker -> isolated + allowed.
    monkeypatch.setattr(dds, "_marker_ownership_status", lambda e, d: "verified")
    out = evaluate_safety(_Session(), {"DISPLAY": ":99", dds.ISOLATION_MARKER_ENV: ":99"})
    assert out["classification"] == ISOLATED
    assert out["allowed"] is True
    assert out["bypassed"] is False


def test_evaluate_refuses_uncorroborated_marker():
    # STRICT: a marker with no XPID ownership proof is refused (fail-closed).
    out = evaluate_safety(_Session(), {"DISPLAY": ":99", dds.ISOLATION_MARKER_ENV: ":99"})
    assert out["classification"] == UNKNOWN
    assert out["allowed"] is False


def test_env_opt_in_bypasses_and_flags(monkeypatch):
    monkeypatch.setattr(dds, "_ewmh_wm_present", lambda d: True)
    out = evaluate_safety(_Session(), _env(ROBOTMCP_PLATYNUI_ALLOW_ACTIVE_DESKTOP="1"))
    assert out["allowed"] is True
    assert out["bypassed"] is True


def test_session_opt_in_bypasses(monkeypatch):
    monkeypatch.setattr(dds, "_ewmh_wm_present", lambda d: True)
    s = _Session()
    s.platynui_allow_active_desktop = True
    out = evaluate_safety(s, _env())
    assert out["allowed"] is True
    assert out["bypassed"] is True


def test_warn_mode_allows_without_bypass_flag(monkeypatch):
    monkeypatch.setattr(dds, "_ewmh_wm_present", lambda d: True)
    out = evaluate_safety(_Session(), _env(ROBOTMCP_PLATYNUI_SAFETY_GUARD="warn"))
    assert out["allowed"] is True
    assert out["enforcing"] is False
    assert out["bypassed"] is False

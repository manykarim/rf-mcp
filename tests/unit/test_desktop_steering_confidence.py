"""Unit tests: desktop steering-confidence verdict
(change: desktop-steering-confidence-gate)."""

from robotmcp.components.execution.desktop_execution_signals import (
    steering_confidence,
    steering_confidence_mode,
    SC_CONFIRMED,
    SC_UNCONFIRMED,
    SC_CONTRADICTED,
)


def _sc(**kw):
    base = dict(
        keyword="Keyboard Type", success=True, verified_focus=None,
        state_before=None, state_after=None, wayland_risk=False,
    )
    base.update(kw)
    return steering_confidence(**base)


def test_non_interaction_keyword_no_verdict():
    assert _sc(keyword="Query") is None
    assert _sc(keyword="Get Attribute") is None


def test_non_success_no_verdict():
    assert _sc(success=False) is None


def test_confirmed_via_verified_focus():
    v = _sc(verified_focus=True)
    assert v["verdict"] == SC_CONFIRMED


def test_confirmed_via_observed_effect():
    v = _sc(verified_focus=False, state_before=3, state_after=5)
    assert v["verdict"] == SC_CONFIRMED
    assert v["signals"]["effect_observed"] is True


def test_contradicted_unverified_focus_no_effect():
    v = _sc(verified_focus=False, state_before=5, state_after=5)
    assert v["verdict"] == SC_CONTRADICTED
    assert v["signals"]["effect_absent"] is True


def test_contradicted_via_wayland_risk_unverified():
    v = _sc(verified_focus=False, wayland_risk=True)
    assert v["verdict"] == SC_CONTRADICTED


def test_verified_focus_wins_over_wayland_risk():
    # focus verified => confirmed even under wayland risk
    v = _sc(verified_focus=True, wayland_risk=True, state_before=5, state_after=5)
    assert v["verdict"] == SC_CONFIRMED


def test_unconfirmed_no_evidence():
    v = _sc(verified_focus=None, state_before=None, state_after=None)
    assert v["verdict"] == SC_UNCONFIRMED


def test_mode_default_enforce(monkeypatch):
    monkeypatch.delenv("ROBOTMCP_PLATYNUI_STEERING_CONFIDENCE", raising=False)
    assert steering_confidence_mode() == "enforce"


def test_mode_warn_optout(monkeypatch):
    monkeypatch.setenv("ROBOTMCP_PLATYNUI_STEERING_CONFIDENCE", "warn")
    assert steering_confidence_mode() == "warn"
    monkeypatch.setenv("ROBOTMCP_PLATYNUI_STEERING_CONFIDENCE", "WARN")
    assert steering_confidence_mode() == "warn"


def test_verdict_shape_stable():
    v = _sc(verified_focus=True)
    assert v["type"] == "desktop_steering_confidence"
    assert set(v.keys()) == {"type", "verdict", "message", "signals"}
    assert set(v["signals"].keys()) == {
        "verified_focus", "effect_observed", "effect_absent", "wayland_risk"
    }


# ── _read_native_character_count (docker-probe-confirmed read path) ─────────
from robotmcp.components.execution.keyword_executor import _read_native_character_count


class _FakeAttr:
    def __init__(self, namespace, name, val):
        self.namespace, self.name, self._v = namespace, name, val
    def value(self):
        return self._v


class _FakeNode:
    """Mirrors PlatynUI new-core: the bare string accessor returns None; the
    working read is attributes() + .value()."""
    def __init__(self, count):
        self._attrs = [
            _FakeAttr("control", "Name", "text"),
            _FakeAttr("native", "Text.CharacterCount", count),
        ]
    def attribute(self, name):
        return None  # new-core behavior for the colon-string form
    def attributes(self):
        return self._attrs


def test_read_char_count_via_attributes_iteration():
    assert _read_native_character_count(_FakeNode(5)) == 5
    assert _read_native_character_count(_FakeNode(0)) == 0


def test_read_char_count_string_accessor_forward_compat():
    class _Node:
        def attribute(self, name):
            return _FakeAttr("native", "Text.CharacterCount", 7)
        def attributes(self):
            return []
    assert _read_native_character_count(_Node()) == 7


def test_read_char_count_none_when_absent():
    class _Node:
        def attribute(self, name):
            return None
        def attributes(self):
            return [_FakeAttr("control", "Name", "x")]
    assert _read_native_character_count(_Node()) is None

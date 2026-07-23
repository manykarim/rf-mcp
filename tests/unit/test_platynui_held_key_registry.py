"""Unit tests for the held-key registry stuck-key hardening.

change: harden-platynui-stuck-key-release. Covers the non-modifier release gap
and the hard-kill persisted-state recovery on top of the F16 net.
"""

import json
import os

import pytest

import robotmcp.plugins.builtin.platynui_plugin as p


class FakeRuntime:
    """Captures keyboard_release dispatches (the OS key-UP calls)."""

    def __init__(self):
        self.releases = []

    def keyboard_release(self, sequence, **kwargs):
        self.releases.append(sequence)


@pytest.fixture
def open_runtime(monkeypatch, tmp_path):
    """Bind a fake open runtime, isolate the state dir, and reset the registry."""
    fake = FakeRuntime()
    monkeypatch.setattr(p, "_RUNTIME", fake)
    monkeypatch.setattr(p, "_RUNTIME_STATE", "open")
    monkeypatch.setattr(p, "_held_keys_dir", lambda: str(tmp_path))
    with p._HELD_KEYS_LOCK:
        p._HELD_KEYS.clear()
    yield fake, tmp_path
    with p._HELD_KEYS_LOCK:
        p._HELD_KEYS.clear()


# ---- normalizer -------------------------------------------------------------


def test_normalizer_parses_chords_keys_and_literals():
    assert p._normalize_key_tokens("<Ctrl+A>") == ["Ctrl", "A"]
    assert p._normalize_key_tokens("<F12>") == ["F12"]
    assert p._normalize_key_tokens("<Escape>") == ["Escape"]
    assert p._normalize_key_tokens("A") == ["A"]
    assert p._normalize_key_tokens("hi") == ["h", "i"]
    assert p._normalize_key_tokens("") == []
    assert p._normalize_key_tokens(None) == []
    assert p._release_sequence_for(["A", "Ctrl"]) == "<A><Ctrl>"


# ---- 5.1 non-modifier gap: session-end releases exactly the held keys -------


def test_bare_nonmodifier_press_is_released_on_teardown(open_runtime):
    fake, tmp = open_runtime
    for seq in ("A", "<F12>", "<Escape>"):
        p.record_pressed_keys(seq)
    assert p._HELD_KEYS == {"A", "F12", "Escape"}
    # state file written with owning pid
    state = tmp / f"held_{os.getpid()}.json"
    assert state.exists()

    assert p.release_tracked_keys() is True
    # every held key (incl. non-modifiers) appears in a dispatched key-UP
    joined = "".join(fake.releases)
    for tok in ("<A>", "<F12>", "<Escape>"):
        assert tok in joined
    # registry + state file cleared
    assert p._HELD_KEYS == set()
    assert not state.exists()


# ---- 5.2 deliberate chord preserved; steering-downgrade helper --------------


def test_deliberate_press_release_chord_preserved(open_runtime):
    fake, _ = open_runtime
    p.record_pressed_keys("<Shift>")
    assert p._HELD_KEYS == {"Shift"}
    # No premature release happens at record time (only teardown/failure does).
    assert fake.releases == []
    # Explicit release clears exactly that key.
    p.record_released_keys("<Shift>")
    assert p._HELD_KEYS == set()


def test_steering_downgrade_helper_detects_contradiction():
    from robotmcp.components.execution.keyword_executor import KeywordExecutor

    assert KeywordExecutor._is_steering_downgrade(
        {"steering_confidence": "contradicted"}
    ) is True
    assert KeywordExecutor._is_steering_downgrade({"success": False}) is False


def test_desktop_keyboard_op_classifies_and_extracts_sequence():
    from robotmcp.components.execution.keyword_executor import KeywordExecutor

    assert KeywordExecutor._desktop_keyboard_op(
        "Keyboard Press", ["/app:*//control:Window", "<Ctrl+A>"]
    ) == ("press", "<Ctrl+A>")
    assert KeywordExecutor._desktop_keyboard_op(
        "PlatynUI.BareMetal.Keyboard Type", ["desc", "text=hello"]
    ) == ("type", "hello")
    assert KeywordExecutor._desktop_keyboard_op(
        "Keyboard Release", ["desc", "<Shift>"]
    ) == ("release", "<Shift>")
    assert KeywordExecutor._desktop_keyboard_op("Pointer Click", ["x"]) == (None, None)


# ---- 5.3 hard-kill recovery -------------------------------------------------


def test_recover_replays_dead_pid_and_leaves_live_pid(open_runtime, monkeypatch):
    fake, tmp = open_runtime
    dead_pid = 999_999  # not alive
    live_pid = os.getpid()
    (tmp / f"held_{dead_pid}.json").write_text(
        json.dumps({"pid": dead_pid, "keys": ["A", "LShift"]}), encoding="utf-8"
    )
    (tmp / f"held_{live_pid}.json").write_text(
        json.dumps({"pid": live_pid, "keys": ["B"]}), encoding="utf-8"
    )

    recovered = p.recover_orphaned_held_keys()
    assert recovered == 1
    # dead-pid keys replayed as UPs, its file deleted
    joined = "".join(fake.releases)
    assert "<A>" in joined and "<LShift>" in joined
    assert not (tmp / f"held_{dead_pid}.json").exists()
    # live-pid file untouched
    assert (tmp / f"held_{live_pid}.json").exists()


# ---- 5.4 clean shutdown / stale replay safety -------------------------------


def test_clean_release_leaves_no_stale_state(open_runtime):
    fake, tmp = open_runtime
    p.record_pressed_keys("<Ctrl>")
    assert (tmp / f"held_{os.getpid()}.json").exists()
    p.release_tracked_keys()
    assert not (tmp / f"held_{os.getpid()}.json").exists()


def test_stale_replay_of_not_held_keys_is_noop(open_runtime):
    fake, tmp = open_runtime
    (tmp / "held_1.json").write_text(
        json.dumps({"pid": 999_998, "keys": ["Z"]}), encoding="utf-8"
    )
    # Not raising and file removed even though 'Z' is not really held.
    assert p.recover_orphaned_held_keys() == 1
    assert not (tmp / "held_1.json").exists()


def test_unreadable_state_file_is_removed(open_runtime):
    fake, tmp = open_runtime
    (tmp / "held_2.json").write_text("{ not json", encoding="utf-8")
    p.recover_orphaned_held_keys()
    assert not (tmp / "held_2.json").exists()


# ---- 5.5 never resurrects the runtime ---------------------------------------


def test_release_and_recover_noop_when_runtime_not_open(monkeypatch, tmp_path):
    monkeypatch.setattr(p, "_RUNTIME", None)
    monkeypatch.setattr(p, "_RUNTIME_STATE", "new")
    monkeypatch.setattr(p, "_held_keys_dir", lambda: str(tmp_path))
    with p._HELD_KEYS_LOCK:
        p._HELD_KEYS.clear()
    # record still tracks (in-memory + file) but release/recover do not dispatch
    p.record_pressed_keys("<Ctrl+A>")
    assert p.release_tracked_keys() is False
    assert p.recover_orphaned_held_keys() == 0
    with p._HELD_KEYS_LOCK:
        p._HELD_KEYS.clear()


def test_pid_alive_self_and_dead():
    assert p._pid_alive(os.getpid()) is True
    assert p._pid_alive(999_999) is False
    assert p._pid_alive(-1) is False

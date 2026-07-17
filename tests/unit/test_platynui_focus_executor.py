"""Unit tests for KeywordExecutor._platynui_focus_before_act
(change: platynui-focused-execution).

We avoid building a real runtime by monkeypatching ``PlatynUIFocusManager``
in the ``platynui_focus`` module with a fake that records the ``ensure_focused``
call and returns a controllable FocusOutcome.
"""

from __future__ import annotations

import types

import pytest

import robotmcp.components.execution.platynui_focus as focus_mod
from robotmcp.components.execution.keyword_executor import KeywordExecutor
from robotmcp.components.execution.platynui_focus import FocusOutcome


class FakeFocusManager:
    """Drop-in for PlatynUIFocusManager that records ensure_focused calls."""

    instances = []

    def __init__(self):
        self.calls = []
        FakeFocusManager.instances.append(self)

    def ensure_focused(self, keyword, arguments, **kwargs):
        self.calls.append({"keyword": keyword, "arguments": arguments, "kwargs": kwargs})
        oc = FocusOutcome()
        if kwargs.get("focus") is False:
            oc.bypassed = True
        else:
            oc.attempted = True
            oc.focused = True
        return oc


def _make_session(
    *, no_focus=False, fail_on_hidden=False, strict_scope=False
):
    return types.SimpleNamespace(
        platynui_no_focus=no_focus,
        platynui_fail_on_hidden=fail_on_hidden,
        platynui_strict_scope=strict_scope,
    )


@pytest.fixture
def executor(monkeypatch):
    monkeypatch.setattr(focus_mod, "PlatynUIFocusManager", FakeFocusManager)
    FakeFocusManager.instances = []
    return KeywordExecutor()


def test_non_interaction_keyword_returns_none(executor):
    session = _make_session()
    result = executor._platynui_focus_before_act(
        session, "Activate Window", ["/app:*//control:Frame"]
    )
    assert result is None
    assert FakeFocusManager.instances == []  # never instantiated


def test_default_session_invokes_ensure_focused_with_focus_true(executor):
    session = _make_session()
    outcome = executor._platynui_focus_before_act(
        session, "Pointer Click", ["/app:*//control:Button"]
    )
    assert outcome is not None
    assert outcome.attempted is True
    assert len(FakeFocusManager.instances) == 1
    call = FakeFocusManager.instances[0].calls[0]
    assert call["kwargs"]["focus"] is True
    assert call["kwargs"]["fail_on_hidden"] is False
    assert call["kwargs"]["strict_scope"] is False


def test_escape_hatch_session_bypasses(executor):
    session = _make_session(no_focus=True)
    outcome = executor._platynui_focus_before_act(
        session, "Pointer Click", ["/app:*//control:Button"]
    )
    assert outcome is not None
    assert outcome.bypassed is True
    call = FakeFocusManager.instances[0].calls[0]
    assert call["kwargs"]["focus"] is False


def test_policy_flags_forwarded(executor):
    session = _make_session(fail_on_hidden=True, strict_scope=True)
    executor._platynui_focus_before_act(
        session, "Keyboard Type", ["/app:*//control:Edit", "hi"]
    )
    call = FakeFocusManager.instances[0].calls[0]
    assert call["kwargs"]["fail_on_hidden"] is True
    assert call["kwargs"]["strict_scope"] is True
    assert call["kwargs"]["focus"] is True


def test_manager_reused_across_calls(executor):
    session = _make_session()
    executor._platynui_focus_before_act(session, "Pointer Click", ["/app:*//c:B"])
    executor._platynui_focus_before_act(session, "Keyboard Press", ["/app:*//c:B"])
    # Only one manager instance created (lazy + cached).
    assert len(FakeFocusManager.instances) == 1


def test_missing_session_attrs_default_to_focus_on(executor):
    # Session without the platynui_* attributes at all.
    session = types.SimpleNamespace()
    outcome = executor._platynui_focus_before_act(
        session, "Pointer Click", ["/app:*//control:Button"]
    )
    assert outcome is not None
    call = FakeFocusManager.instances[0].calls[0]
    assert call["kwargs"]["focus"] is True
    assert call["kwargs"]["fail_on_hidden"] is False
    assert call["kwargs"]["strict_scope"] is False

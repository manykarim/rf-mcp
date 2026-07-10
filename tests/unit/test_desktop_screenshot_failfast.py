"""Desktop screenshot / control:Window fail-fast guards
(change: desktop-screenshot-failfast).

Two pre-dispatch guards mirroring the unscoped-locator guardrail, each turning a
silent 30s AT-SPI hang into a fast, actionable refusal:
- Take Screenshot with a bare image path in the DESCRIPTOR slot.
- A Linux desktop tree-resolving keyword using control:Window (windows are
  control:Frame on AT-SPI).
"""

from __future__ import annotations

import sys
import types

import pytest

from robotmcp.components.execution.desktop_execution_signals import (
    screenshot_path_in_descriptor_slot,
    control_window_locator,
)
from robotmcp.components.execution.keyword_executor import KeywordExecutor


def _desktop_session(**attrs):
    s = types.SimpleNamespace(
        is_desktop_session=lambda: True,
        platynui_allow_path_descriptor=False,
        desktop_path_descriptor_warned=False,
        platynui_allow_control_window=False,
        desktop_control_window_warned=False,
    )
    for k, v in attrs.items():
        setattr(s, k, v)
    return s


class TestScreenshotDetection:
    @pytest.mark.parametrize("args,expected", [
        (["/artifacts/x.png"], "/artifacts/x.png"),          # trap
        (["descriptor=/x.png"], "/x.png"),                   # explicit descriptor=path
        (["a_descriptor", "/x.png"], None),                  # correct 2-positional
        (["filename=/x.png"], None),                         # correct named form
        (["EMBED"], None),
        (["run-{index}.png"], None),                         # templated name (not a positional path)
        (["a_descriptor"], None),                            # descriptor only, no shot
    ])
    def test_descriptor_slot_path(self, args, expected):
        assert screenshot_path_in_descriptor_slot("Take Screenshot", args) == expected

    def test_non_screenshot_keyword_ignored(self):
        assert screenshot_path_in_descriptor_slot("Query", ["/x.png"]) is None


class TestControlWindowDetection:
    @pytest.mark.parametrize("kw", ["Query", "Evaluate", "Set Root", "Get Attribute", "Pointer Click"])
    def test_control_window_detected_on_tree_keywords(self, kw):
        loc = "/app:*[@Name='x']//control:Window"
        assert control_window_locator(kw, [loc]) == loc

    def test_control_frame_ignored(self):
        assert control_window_locator("Query", ["/app:*//control:Frame"]) is None

    def test_non_tree_keyword_ignored(self):
        assert control_window_locator("Take Screenshot", ["//control:Window"]) is None


class TestScreenshotGuard:
    def setup_method(self):
        self.ex = KeywordExecutor.__new__(KeywordExecutor)

    def test_path_descriptor_refused_with_signature_hint(self):
        r = self.ex._screenshot_signature_guard(
            _desktop_session(), "Take Screenshot", ["/artifacts/x.png"]
        )
        assert r is not None and r["success"] is False
        hint = r["hints"][0]["message"]
        assert "descriptor, filename, rect" in hint and "filename=/artifacts/x.png" in hint

    def test_correct_forms_not_refused(self):
        s = _desktop_session()
        assert self.ex._screenshot_signature_guard(s, "Take Screenshot", ["filename=/x.png"]) is None
        assert self.ex._screenshot_signature_guard(s, "Take Screenshot", ["desc", "/x.png"]) is None

    def test_optout_downgrades_to_one_time_warning(self):
        s = _desktop_session(platynui_allow_path_descriptor=True)
        assert self.ex._screenshot_signature_guard(s, "Take Screenshot", ["/x.png"]) is None
        assert s.desktop_path_descriptor_warned is True  # flipped once
        # second call: still proceeds, no error, warned stays set (no dup)
        assert self.ex._screenshot_signature_guard(s, "Take Screenshot", ["/y.png"]) is None

    def test_non_desktop_session_unaffected(self):
        web = types.SimpleNamespace(is_desktop_session=lambda: False)
        assert self.ex._screenshot_signature_guard(web, "Take Screenshot", ["/x.png"]) is None


@pytest.mark.skipif(sys.platform != "linux", reason="control:Window guard is Linux-only")
class TestControlWindowGuard:
    def setup_method(self):
        self.ex = KeywordExecutor.__new__(KeywordExecutor)

    def test_control_window_refused_with_frame_rewrite(self):
        loc = "/app:*[@Name='calc']//control:Window"
        r = self.ex._control_window_guard(_desktop_session(), "Query", [loc])
        assert r is not None and r["success"] is False
        hint = r["hints"][0]["message"]
        assert "control:Frame" in hint
        assert "/app:*[@Name='calc']//control:Frame" in hint  # exact rewrite

    def test_control_frame_not_refused(self):
        assert self.ex._control_window_guard(
            _desktop_session(), "Query", ["/app:*//control:Frame"]
        ) is None

    def test_optout_one_time_warning(self):
        s = _desktop_session(platynui_allow_control_window=True)
        assert self.ex._control_window_guard(s, "Query", ["//control:Window"]) is None
        assert s.desktop_control_window_warned is True

    def test_non_desktop_unaffected(self):
        web = types.SimpleNamespace(is_desktop_session=lambda: False)
        assert self.ex._control_window_guard(web, "Query", ["//control:Window"]) is None

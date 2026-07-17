"""Close liveness hint
(change: desktop-test-scoping-and-close-lifecycle, D5).

Run 3: LibreOffice's document window closed but the process survived as a
start-center frame with no signal; the agent looped on Alt+F4 retries.
"""

from __future__ import annotations

from robotmcp.components.execution.desktop_execution_signals import (
    close_liveness_hint,
    is_close_keyword,
)


class TestCloseKeywordDetection:
    def test_close_window_detected(self):
        assert is_close_keyword("Close Window") is True
        assert is_close_keyword("PlatynUI.BareMetal.Close Window") is True

    def test_other_keywords_not_detected(self):
        for kw in ("Pointer Click", "Terminate Process", "Keyboard Type"):
            assert is_close_keyword(kw) is False


class TestCloseLivenessHint:
    def test_alive_process_hints(self):
        hint = close_liveness_hint(True)
        assert hint is not None
        assert hint["type"] == "desktop_close_liveness"
        assert "still" in hint["message"]
        assert "Terminate Process" in hint["message"]

    def test_dead_process_silent(self):
        assert close_liveness_hint(False) is None

    def test_unknown_liveness_silent(self):
        assert close_liveness_hint(None) is None

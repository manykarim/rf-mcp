"""Screenshot evidence integrity
(change: desktop-evidence-and-display-scoping, D2/D3/D8).

The 2026-06-11 LibreOffice rerun produced five "successful" screenshots
that exist nowhere on disk, one PlatynUI screenshot that WAS written but
whose keyword failed on an upstream log-link quirk, and a near-empty suite
that did not warn because evidence steps counted as substance.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

# The path guard validates POSIX absolute roots (/tmp, /etc); on macOS /tmp is a
# symlink to /private/tmp and on Windows these paths are meaningless.
_linux_path_semantics = pytest.mark.skipif(
    sys.platform != "linux",
    reason="POSIX absolute-root path semantics — Linux only",
)

from robotmcp.components.execution.desktop_execution_signals import (
    evidence_missing_hint,
    is_screenshot_keyword,
    screenshot_request_path,
)
from robotmcp.components.execution.keyword_executor import KeywordExecutor


class TestScreenshotPathExtraction:
    def test_named_filename_form(self):
        assert screenshot_request_path(
            "Take Screenshot", ["filename=/tmp/run/shot.png"]
        ) == "/tmp/run/shot.png"

    def test_positional_path(self):
        assert screenshot_request_path(
            "PlatynUI.BareMetal.Take Screenshot", ["/tmp/run/shot.jpg"]
        ) == "/tmp/run/shot.jpg"

    def test_descriptor_then_path(self):
        assert screenshot_request_path(
            "Take Screenshot", ["//control:Window[@Name='X']", "/tmp/a.png"]
        ) == "/tmp/a.png"

    def test_embed_and_template_yield_none(self):
        assert screenshot_request_path("Take Screenshot", ["EMBED"]) is None
        assert screenshot_request_path(
            "Take Screenshot", ["shot-{index}.png"]
        ) is None

    def test_non_screenshot_keyword_none(self):
        assert screenshot_request_path("Pointer Click", ["/tmp/a.png"]) is None
        assert is_screenshot_keyword("Pointer Click") is False
        assert is_screenshot_keyword("Screenshot.Take Screenshot") is True


class TestEvidenceMissingHint:
    def test_ghost_screenshot_warns(self):
        hint = evidence_missing_hint(
            "Take Screenshot",
            ["/tmp/run/after-typing.jpg"],
            "/tmp/run/after-typing.jpg",
            _isfile=lambda p: False,
        )
        assert hint is not None
        assert hint["type"] == "evidence_missing"
        assert "/tmp/run/after-typing.jpg" in hint["message"]

    def test_real_file_silent(self):
        hint = evidence_missing_hint(
            "Take Screenshot",
            ["/tmp/run/shot.png"],
            "/tmp/run/shot.png",
            _isfile=lambda p: True,
        )
        assert hint is None

    def test_return_value_preferred_over_argument(self):
        # RF Screenshot returns the actually-saved absolute path; verify THAT.
        seen = []

        def isfile(p):
            seen.append(p)
            return False

        evidence_missing_hint(
            "Take Screenshot",
            ["/tmp/requested.jpg"],
            "/tmp/actually-saved.jpg",
            _isfile=isfile,
        )
        assert seen == ["/tmp/actually-saved.jpg"]

    def test_relative_paths_not_verifiable(self):
        assert evidence_missing_hint(
            "Take Screenshot", ["shot.png"], "shot.png", _isfile=lambda p: False
        ) is None

    def test_non_screenshot_keyword_none(self):
        assert evidence_missing_hint(
            "Click", ["/tmp/x.png"], None, _isfile=lambda p: False
        ) is None


class TestScreenshotPathGuard:
    def setup_method(self):
        self.executor = KeywordExecutor.__new__(KeywordExecutor)

    @_linux_path_semantics
    def test_tmp_path_allowed(self):
        assert self.executor._screenshot_path_guard(
            "Take Screenshot", ["/tmp/run/shots/a.png"]
        ) is None

    @_linux_path_semantics
    def test_disallowed_path_refused_with_roots_hint(self):
        out = self.executor._screenshot_path_guard(
            "Take Screenshot", ["/etc/shot.png"]
        )
        assert out is not None
        assert out["success"] is False
        assert out["hints"][0]["type"] == "screenshot_path_refused"
        assert "/tmp" in out["hints"][0]["message"]

    def test_extra_root_via_env(self, monkeypatch, tmp_path):
        extra = tmp_path / "artifacts"
        extra.mkdir()
        monkeypatch.setenv("ROBOTMCP_SCREENSHOT_DIR", str(extra))
        assert self.executor._screenshot_path_guard(
            "Take Screenshot", [f"{extra}/a.png"]
        ) is None

    def test_relative_path_untouched(self):
        assert self.executor._screenshot_path_guard(
            "Take Screenshot", ["shot.png"]
        ) is None


class TestScreenshotRecovery:
    def _desktop_session(self):
        s = types.SimpleNamespace()
        s.is_desktop_session = lambda: True
        return s

    def test_subpath_failure_with_real_file_recovers(self, tmp_path):
        shot = tmp_path / "after-launch.png"
        shot.write_bytes(b"\x89PNG fake")
        result = {
            "success": False,
            "error": (
                f"ValueError: '{shot}' is not in the subpath of "
                "'/tmp/rf_mcp_xyz'"
            ),
        }
        KeywordExecutor._maybe_recover_screenshot_result(
            self._desktop_session(), "Take Screenshot", [str(shot)], result
        )
        assert result["success"] is True
        assert result["error"] is None
        assert result["result"] == str(shot)
        assert result["hints"][0]["type"] == "screenshot_path_recovered"

    def test_missing_file_stays_failed(self, tmp_path):
        result = {
            "success": False,
            "error": "'x' is not in the subpath of '/tmp/rf_mcp_xyz'",
        }
        KeywordExecutor._maybe_recover_screenshot_result(
            self._desktop_session(),
            "Take Screenshot",
            [str(tmp_path / "never-written.png")],
            result,
        )
        assert result["success"] is False

    def test_other_errors_untouched(self, tmp_path):
        shot = tmp_path / "a.png"
        shot.write_bytes(b"x")
        result = {"success": False, "error": "element not found"}
        KeywordExecutor._maybe_recover_screenshot_result(
            self._desktop_session(), "Take Screenshot", [str(shot)], result
        )
        assert result["success"] is False

    def test_web_session_untouched(self, tmp_path):
        shot = tmp_path / "a.png"
        shot.write_bytes(b"x")
        web = types.SimpleNamespace(is_desktop_session=lambda: False)
        result = {
            "success": False,
            "error": "'a' is not in the subpath of '/tmp/rf_mcp_x'",
        }
        KeywordExecutor._maybe_recover_screenshot_result(
            web, "Take Screenshot", [str(shot)], result
        )
        assert result["success"] is False


@pytest.mark.asyncio
class TestRun3SuiteShapeWarns:
    async def test_launch_plus_evidence_with_failures_warns(self):
        from robotmcp.components.test_builder import TestBuilder
        from robotmcp.models.execution_models import ExecutionStep
        from robotmcp.models.session_models import ExecutionSession

        sess = ExecutionSession(session_id="run3")
        sess.configure_from_scenario(
            "Open LibreOffice Writer desktop application", context="desktop"
        )
        sess.executed_step_count = 100
        sess.failed_step_count = 41
        sess.test_registry.start_test("Writer Validation", tags=["desktop"])
        for kw, args in [
            ("Create Directory", ["/tmp/run/testdir"]),
            ("Start Process", ["soffice", "--writer"]),
            ("Sleep", ["6s"]),
            ("Is Process Running", ["aut"]),
            ("Get Process Id", ["aut"]),
            ("Take Screenshot", ["/tmp/run/shots/a.jpg"]),
            ("Terminate Process", ["aut"]),
        ]:
            st = ExecutionStep(step_id=kw, keyword=kw, arguments=args)
            st.mark_success()
            sess.test_registry.tests["Writer Validation"].steps.append(st)
        sess.test_registry.end_test(status="pass")

        engine = MagicMock()
        engine.sessions = {"run3": sess}
        builder = TestBuilder(execution_engine=engine)
        result = await builder.build_suite(session_id="run3", test_name="Run3")
        assert result["success"] is True
        assert result.get("warning"), (
            "launch + evidence + 41 failed interactions must warn"
        )

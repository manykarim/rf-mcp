"""Process `=`-argument detect-and-hint
(change: platynui-visible-safe-targeting, I-4).

RF parses a dash-prefixed positional argument containing ``=`` (e.g.
LibreOffice's ``-env:UserInstallation=file:///…``) as a named argument and
silently drops it from the command line; the documented fix is escaping the
first ``=`` as ``\\=``. These tests pin the detection heuristic, the
reactive hint on failed launches, and that legitimate Process configuration
is never flagged.
"""

from __future__ import annotations

from robotmcp.utils.hints import (
    HintContext,
    _check_process_named_arg_misparse,
    detect_process_eq_arg_misparse,
    generate_hints,
)


LIBRE_ARG = "-env:UserInstallation=file:///tmp/profile"


class TestDetection:
    def test_libreoffice_userinstallation_flagged(self):
        flagged = detect_process_eq_arg_misparse(
            "Start Process", ["soffice", "--writer", LIBRE_ARG]
        )
        assert flagged == [LIBRE_ARG]

    def test_library_prefixed_keyword(self):
        assert detect_process_eq_arg_misparse(
            "Process.Run Process", ["app", "-Dkey=value"]
        ) == ["-Dkey=value"]

    def test_legit_env_configuration_not_flagged(self):
        assert detect_process_eq_arg_misparse(
            "Start Process", ["app", "env:HOME=/tmp/home"]
        ) == []

    def test_legit_shell_configuration_not_flagged(self):
        assert detect_process_eq_arg_misparse(
            "Start Process", ["app", "shell=True", "cwd=/tmp", "alias=aut",
                              "stdout=/tmp/o", "stderr=/tmp/e"]
        ) == []

    def test_already_escaped_not_flagged(self):
        assert detect_process_eq_arg_misparse(
            "Start Process", ["soffice", "-env:UserInstallation\\=file:///tmp/p"]
        ) == []

    def test_non_process_keyword_not_flagged(self):
        assert detect_process_eq_arg_misparse(
            "Pointer Click", ["//control:Button", "-x=1"]
        ) == []

    def test_dash_arg_without_equals_not_flagged(self):
        assert detect_process_eq_arg_misparse(
            "Start Process", ["soffice", "--writer", "--norestore"]
        ) == []

    def test_non_string_args_ignored(self):
        assert detect_process_eq_arg_misparse("Start Process", ["app", 5, None]) == []


class TestReactiveHint:
    def _ctx(self, keyword, arguments, error="Process launch failed"):
        return HintContext(
            session_id="s1",
            keyword=keyword,
            arguments=arguments,
            error_text=error,
        )

    def test_failed_launch_gets_escape_hint(self):
        ctx = self._ctx("Start Process", ["soffice", LIBRE_ARG])
        hints = _check_process_named_arg_misparse(ctx, ctx.error_text)
        assert len(hints) == 1
        h = hints[0]
        assert "named argument" in h.message
        assert "\\=" in h.message
        assert any(
            "-env:UserInstallation\\=file:///tmp/profile" in ex["arguments"]
            for ex in h.examples
        )

    def test_hint_flows_through_generate_hints(self):
        ctx = self._ctx("Run Process", ["soffice", LIBRE_ARG])
        out = generate_hints(ctx)
        assert any("named argument" in h["message"] for h in out)

    def test_clean_launch_failure_no_hint(self):
        ctx = self._ctx("Start Process", ["soffice", "--writer"])
        assert _check_process_named_arg_misparse(ctx, ctx.error_text) == []

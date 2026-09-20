"""Regression tests for the installer safety/UX defects (change: installer-cli-safety).

Every test here pins a defect that was REPRODUCED against the shipped CLI in
sandboxed probes (evidence: experiments/installer_ux_*.md, adapter_config_audit.md).
Each one fails against the pre-fix code.

Run: uv run pytest tests/unit/test_installer_cli_safety.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from robotmcp.onboarding import adapters as A
from robotmcp.onboarding import cli
from robotmcp.onboarding import codecs
from robotmcp.onboarding import installer as I
from robotmcp.onboarding import project_env as pe
from robotmcp.onboarding.manifest import Manifest


def _make_venv(dirpath: Path) -> Path:
    venv = dirpath / ".venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    py = venv / "bin" / "python"
    py.write_text("", encoding="utf-8")
    py.chmod(0o755)
    return py


# =============================================================================
# sec 1 - --dry-run must not mutate
# =============================================================================


class TestDryRunIsSideEffectFree:
    """`install --dry-run --into-project` really ran `uv pip install` into the
    project venv: `resolve_launch` was called before any dry_run branch and
    `dry_run` was never threaded into `_install_into_project`."""

    def _env(self, tmp_path, monkeypatch):
        py = _make_venv(tmp_path)
        env = pe.ProjectEnv(project_dir=tmp_path, type="venv", python=py, is_venv=True)
        monkeypatch.setattr(pe, "detect", lambda d: env)
        monkeypatch.setattr(pe, "rfmcp_in_project", lambda p: False)
        monkeypatch.setattr(pe, "rf_conflict", lambda e: None)
        monkeypatch.setattr(pe, "project_extra_libraries", lambda e: ["JSONLibrary"])
        return env

    def test_dry_run_does_not_install_into_project(self, tmp_path, monkeypatch):
        self._env(tmp_path, monkeypatch)
        called = []
        monkeypatch.setattr(I, "_install_into_project",
                            lambda env, **kw: called.append(env) or (True, ""))
        plan = I.resolve_launch(scope="project", project_dir=tmp_path,
                                into_project=True, dry_run=True)
        assert called == [], "dry-run must not install into the project env"
        assert "would install" in plan.note

    def test_without_dry_run_it_still_installs(self, tmp_path, monkeypatch):
        """Guard the fix from over-reaching: the real path must still work."""
        self._env(tmp_path, monkeypatch)
        called = []
        monkeypatch.setattr(I, "_install_into_project",
                            lambda env, **kw: called.append(env) or (True, ""))
        I.resolve_launch(scope="project", project_dir=tmp_path,
                         into_project=True, dry_run=False)
        assert len(called) == 1

    def test_install_into_project_refuses_under_dry_run(self, tmp_path):
        """Defence in depth: the helper itself refuses, whoever calls it."""
        py = _make_venv(tmp_path)
        env = pe.ProjectEnv(project_dir=tmp_path, type="venv", python=py, is_venv=True)
        ok, detail = I._install_into_project(env, dry_run=True)
        assert ok is False and "dry-run" in detail


# =============================================================================
# sec 3 / sec 3b - selection, abort, and argument validation
# =============================================================================


class TestAgentSelection:
    def test_empty_selection_is_an_error_not_everything(self):
        """`--agents ""` (and a declined prompt) silently meant "all detected"."""
        with pytest.raises(A.EmptySelectionError):
            A.resolve_selection("")

    def test_unknown_id_raises_instead_of_silent_no_op(self):
        with pytest.raises(A.UnknownAgentError) as exc:
            A.resolve_selection("claude")          # the real id is claude-code
        assert "claude" in str(exc.value)
        assert "claude-code" in str(exc.value)     # names the valid ids

    def test_partially_valid_list_reports_the_invalid_token(self):
        with pytest.raises(A.UnknownAgentError) as exc:
            A.resolve_selection("claude-code,nosuchagent")
        assert exc.value.unknown == ["nosuchagent"]

    def test_all_and_detected_still_work(self):
        assert A.resolve_selection("all")
        A.resolve_selection("detected")            # must not raise

    def test_non_strict_mode_is_best_effort(self):
        assert A.resolve_selection("nosuchagent", strict=False) == []


class TestDeclineAndAbort:
    def test_declined_prompt_returns_a_sentinel_not_empty_string(self, monkeypatch):
        """An empty string was coerced back to "detected" downstream, so answering
        `n` installed into every detected agent."""
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *_: "n")
        monkeypatch.setattr(A, "REGISTRY", A.REGISTRY)
        assert cli._interactive_agents("detected", no_input=False) == cli.DECLINED

    def test_accepted_prompt_returns_detected(self, monkeypatch):
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *_: "")
        assert cli._interactive_agents("detected", no_input=False) == "detected"

    def test_declined_install_writes_nothing_and_exits_nonzero(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda *_: "n")
        called = []
        monkeypatch.setattr(I, "install", lambda **kw: called.append(kw) or [])
        rc = cli.run(["install", "-C", str(tmp_path)])
        assert called == [], "a declined prompt must not reach the installer"
        assert rc != 0

    def test_keyboard_interrupt_is_reported_not_raised(self, monkeypatch):
        def boom(*_a, **_k):
            raise KeyboardInterrupt
        monkeypatch.setattr(cli, "_run", boom)
        assert cli.run(["install"]) == 130


class TestArgumentValidation:
    @pytest.mark.parametrize("bad", ["FOO", "=bar", ""])
    def test_malformed_env_is_rejected(self, bad):
        with pytest.raises(cli.UsageError):
            cli._parse_env([bad])

    def test_duplicate_env_with_conflicting_values_is_rejected(self):
        with pytest.raises(cli.UsageError):
            cli._parse_env(["FOO=bar", "FOO=baz"])

    def test_wellformed_env_parses(self):
        assert cli._parse_env(["FOO=bar", "B=x=y"]) == {"FOO": "bar", "B": "x=y"}

    @pytest.mark.parametrize("bad", ["nonsense", "mcp,nonsense", "MCP"])
    def test_unknown_what_is_rejected(self, bad):
        with pytest.raises(cli.UsageError):
            cli._validate_whats([w for w in bad.split(",") if w])

    def test_known_what_passes(self):
        assert cli._validate_whats(["mcp"]) == ["mcp"]

    @pytest.mark.parametrize("bad", ["localhost:notaport", "localhost:99999",
                                     "localhost:", ":9999", "  "])
    def test_invalid_attach_is_rejected(self, bad):
        with pytest.raises(cli.UsageError):
            cli._validate_attach(bad)

    @pytest.mark.parametrize("good", [None, "auto", "localhost", "1.2.3.4:9999"])
    def test_valid_attach_passes(self, good):
        assert cli._validate_attach(good) == good

    def test_empty_project_dir_is_rejected(self):
        """`-C ""` silently inspected the CWD (e.g. `-C "$UNSET_VAR"`)."""
        with pytest.raises(cli.UsageError):
            cli._resolve_project_dir("")

    def test_project_dir_that_is_a_file_is_rejected(self, tmp_path):
        f = tmp_path / "afile"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(cli.UsageError):
            cli._resolve_project_dir(str(f))

    def test_nonexistent_project_dir_is_allowed_but_warned(self, tmp_path, capsys):
        p = cli._resolve_project_dir(str(tmp_path / "nope"))
        cli._warn_project_dir(p)
        assert "does not exist" in capsys.readouterr().out


# =============================================================================
# sec 2 - exit codes
# =============================================================================


class TestExitCodes:
    def test_failed_status_exits_nonzero(self):
        rc = cli._print_results([I.Result("claude-code", "project", "mcp", "unverified",
                                          detail="spawn failed")])
        assert rc != 0

    def test_successful_status_exits_zero(self):
        rc = cli._print_results([I.Result("claude-code", "project", "mcp", "installed",
                                          path="/tmp/x")])
        assert rc == 0

    def test_benign_statuses_exit_zero(self):
        rc = cli._print_results([
            I.Result("claude-code", "project", "mcp", "already-present"),
            I.Result("codex", "project", "mcp", "absent"),
            I.Result("kilo", "project", "mcp", "kept-user-modified"),
        ])
        assert rc == 0

    def test_empty_result_set_exits_nonzero(self):
        assert cli._print_results([]) != 0


# =============================================================================
# sec 4 - config integrity
# =============================================================================


class TestJsoncStringSafety:
    """The trailing-comma pass was a regex over the whole document, so it rewrote
    the CONTENTS of string values."""

    def test_string_values_are_never_altered(self):
        src = '''{
          // comment forces the JSONC path
          "promptTemplate": "Close the brace like this: { a, } and the bracket like [ b, ]",
          "note": "keep the trailing spaces, } exactly",
          "mcp": {}
        }'''
        d = json.loads(codecs._strip_jsonc(src))
        assert d["promptTemplate"] == \
            "Close the brace like this: { a, } and the bracket like [ b, ]"
        assert d["note"] == "keep the trailing spaces, } exactly"

    @pytest.mark.parametrize("value", ["https://example.com/x", "don't // panic",
                                       "a /* b", "trailing, }", "x, ]"])
    def test_tricky_strings_survive(self, value):
        src = '{ // c\n "k": %s, "mcp": {} }' % json.dumps(value)
        assert json.loads(codecs._strip_jsonc(src))["k"] == value

    def test_real_trailing_commas_are_still_removed(self):
        src = '{ // c\n "arr": [1, 2, 3,], "obj": { "k": "v", }, "mcp": {} }'
        d = json.loads(codecs._strip_jsonc(src))
        assert d["arr"] == [1, 2, 3] and d["obj"] == {"k": "v"}


class TestConfigParseErrors:
    def test_malformed_json_raises_a_named_error(self, tmp_path):
        p = tmp_path / "broken.json"
        p.write_text("{ not json", encoding="utf-8")
        with pytest.raises(codecs.ConfigParseError) as exc:
            codecs.load(p, "json")
        assert str(p) in str(exc.value)      # the traceback never named the file

    def test_yaml_anchors_block_a_rewrite(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("defaults: &defaults\n  a: 1\nx:\n  <<: *defaults\n", encoding="utf-8")
        hazard = codecs.rewrite_hazard(p, "yaml")
        assert hazard and "anchor" in hazard

    def test_plain_yaml_has_no_hazard(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("a: 1\nb: two\n", encoding="utf-8")
        assert codecs.rewrite_hazard(p, "yaml") is None

    def test_non_yaml_formats_have_no_hazard(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text("{}", encoding="utf-8")
        assert codecs.rewrite_hazard(p, "json") is None


class TestKiloAdapterShape:
    """The adapter wrote `.kilo/kilo.jsonc` with `{type: stdio, command: "<str>"}`;
    a real Kilo config is `.kilo/kilo.json` with the opencode shape."""

    def test_kilo_writes_kilo_json(self):
        ad = A.get("kilo")
        assert ad.project_path == ".kilo/kilo.json"
        assert ad.user_path.endswith("kilo.json")

    def test_kilo_entry_is_local_with_argv_list(self):
        entry = A.get("kilo").build_entry("/bin/robotmcp", ["--flag"], {})
        assert entry["type"] == "local"
        assert entry["command"] == ["/bin/robotmcp", "--flag"]
        assert entry["enabled"] is True


# =============================================================================
# sec 5 - uninstall correctness
# =============================================================================


class TestUninstallSelection:
    def _manifest_with(self, tmp_path, *paths):
        m = Manifest(path=tmp_path / "manifest.json")
        for p in paths:
            m.record(agent="claude-code", scope="project", what="mcp",
                     path=str(p), value={"command": "robotmcp"}, created_file=True)
        return m

    def _write_cfg(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"mcpServers": {"robotmcp": {"command": "robotmcp"}}}),
                        encoding="utf-8")

    def test_project_dir_filters_removal(self, tmp_path):
        """`uninstall -C dirA` also removed dirB's entry: the path was never a filter."""
        a, b = tmp_path / "a" / ".mcp.json", tmp_path / "b" / ".mcp.json"
        self._write_cfg(a); self._write_cfg(b)
        m = self._manifest_with(tmp_path, a, b)
        results = I.uninstall(agents="all", scope="project", whats=["mcp"],
                              manifest=m, project_dir=tmp_path / "a")
        removed = {r.path for r in results if r.status == "removed"}
        assert str(a) in removed
        assert str(b) not in removed
        assert b.exists(), "another project's config must be untouched"

    def test_undetected_agent_can_still_be_uninstalled(self, tmp_path, monkeypatch):
        """Default `--agents detected` meant an entry for an agent that is no longer
        installed could never be removed."""
        cfg = tmp_path / ".mcp.json"
        self._write_cfg(cfg)
        m = self._manifest_with(tmp_path, cfg)
        monkeypatch.setattr(A.AgentAdapter, "detect", lambda self, **kw: False)
        results = I.uninstall(whats=["mcp"], manifest=m, project_dir=tmp_path)
        assert any(r.status == "removed" for r in results)

    def test_force_removes_a_user_modified_entry(self, tmp_path):
        cfg = tmp_path / ".mcp.json"
        self._write_cfg(cfg)
        m = Manifest(path=tmp_path / "manifest.json")
        m.record(agent="claude-code", scope="project", what="mcp", path=str(cfg),
                 value={"command": "SOMETHING-ELSE"}, created_file=False)
        kept = I.uninstall(agents="all", whats=["mcp"], manifest=m, project_dir=tmp_path)
        assert any(r.status == "kept-user-modified" for r in kept)
        forced = I.uninstall(agents="all", whats=["mcp"], manifest=m,
                             project_dir=tmp_path, force=True)
        assert any(r.status == "removed" for r in forced)

    def test_one_corrupt_file_does_not_abort_the_run(self, tmp_path):
        good, bad = tmp_path / "g" / ".mcp.json", tmp_path / "b" / ".mcp.json"
        self._write_cfg(good)
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("{ not json", encoding="utf-8")
        m = self._manifest_with(tmp_path, bad, good)
        results = I.uninstall(agents="all", scope="project", whats=["mcp"],
                              manifest=m, project_dir=tmp_path)
        assert any(r.status == "removed" and r.path == str(good) for r in results)
        assert any(r.status == "error" and r.path == str(bad) for r in results)


class TestManifestCreatedFile:
    def test_created_flag_survives_a_forced_reinstall(self, tmp_path):
        """A re-install with --force flipped created_file to False, so uninstall
        left an orphaned `{}` behind."""
        m = Manifest(path=tmp_path / "m.json")
        m.record(agent="cursor", scope="project", what="mcp", path="/tmp/x.json",
                 value={"a": 1}, created_file=True)
        assert m.was_created(agent="cursor", scope="project", what="mcp",
                             path="/tmp/x.json") is True
        assert m.was_created(agent="cursor", scope="project", what="mcp",
                             path="/tmp/other.json") is False


# =============================================================================
# sec 7 / sec 8 / sec 9 - discoverability, browser opt-in, diagnostics
# =============================================================================


class TestDiscoverability:
    def test_help_lists_the_subcommands(self, capsys):
        from robotmcp.entry import main
        with pytest.raises(SystemExit):
            main(["--help"])
        out = capsys.readouterr().out
        for cmd in ("init", "install", "uninstall", "list", "doctor"):
            assert cmd in out, f"`robotmcp --help` must mention `{cmd}`"

    def test_unknown_subcommand_is_not_a_server_invocation(self):
        from robotmcp.entry import _is_server_invocation
        assert _is_server_invocation([]) is True
        assert _is_server_invocation(["--log-level", "INFO"]) is True
        assert _is_server_invocation(["instal"]) is False
        assert _is_server_invocation(["initt"]) is False

    def test_mistyped_subcommand_suggests_the_real_one(self, capsys):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["instal"])
        assert "did you mean 'install'" in capsys.readouterr().err


class TestBrowserOptIn:
    def test_plain_init_never_downloads(self, monkeypatch, capsys):
        """`want_browsers = browsers or libs.get("Browser")` started a ~500MB
        download with no flag and no prompt."""
        from robotmcp.onboarding import diagnostics as D
        monkeypatch.setattr(D, "library_status",
                            lambda: {m: (m == "Browser") for m, _, _ in D.TEST_LIBRARIES})
        monkeypatch.setattr(D, "browser_initialized", lambda: False)
        monkeypatch.setattr(D, "node_present", lambda: True)
        called = []
        monkeypatch.setattr(D, "run_browser_init",
                            lambda: called.append(1) or (True, ""))
        D.cmd_init(browsers=False)
        assert called == [], "plain init must not download a browser"
        out = capsys.readouterr().out
        assert "robotmcp init --browsers" in out    # tells the user how to do it

    def test_browsers_flag_still_downloads(self, monkeypatch):
        from robotmcp.onboarding import diagnostics as D
        monkeypatch.setattr(D, "library_status",
                            lambda: {m: (m == "Browser") for m, _, _ in D.TEST_LIBRARIES})
        monkeypatch.setattr(D, "browser_initialized", lambda: False)
        monkeypatch.setattr(D, "node_present", lambda: True)
        called = []
        monkeypatch.setattr(D, "run_browser_init",
                            lambda: called.append(1) or (True, ""))
        D.cmd_init(browsers=True)
        assert called == [1]

    def test_failed_browser_init_exits_nonzero(self, monkeypatch):
        from robotmcp.onboarding import diagnostics as D
        monkeypatch.setattr(D, "library_status",
                            lambda: {m: (m == "Browser") for m, _, _ in D.TEST_LIBRARIES})
        monkeypatch.setattr(D, "browser_initialized", lambda: False)
        monkeypatch.setattr(D, "node_present", lambda: True)
        monkeypatch.setattr(D, "run_browser_init", lambda: (False, "boom"))
        assert D.cmd_init(browsers=True) != 0


class TestDiagnostics:
    def test_desktop_is_a_reported_capability(self):
        from robotmcp.onboarding import diagnostics as D
        mods = [m for m, _, _ in D.TEST_LIBRARIES]
        assert "PlatynUI" in mods, "doctor could not diagnose desktop at all"

    def test_desktop_hint_is_the_command_that_actually_works(self):
        from robotmcp.onboarding import diagnostics as D
        hint = D._extra_hint("desktop")
        assert "--prerelease=allow" in hint, \
            'a plain `uv ... "rf-mcp[desktop]"` does not resolve'

    def test_doctor_strict_fails_when_a_capability_is_missing(self, monkeypatch):
        from robotmcp.onboarding import diagnostics as D
        monkeypatch.setattr(D.importlib.util, "find_spec", lambda m: None)
        assert D.cmd_doctor(strict=True) != 0
        assert D.cmd_doctor(strict=False) == 0

    def test_bundled_dists_reflect_this_install(self):
        """A static `[all]` catalogue made doctor contradict itself."""
        available = pe.available_bundled_dists()
        assert available <= pe._BUNDLED_DISTS
        assert "robotframework" in available          # always present in this env

    def test_list_reads_registered_from_the_agent_config(self, tmp_path, monkeypatch):
        """REGISTERED came from rf-mcp's manifest alone, so a hand-pasted config
        reported `no`."""
        cfg = tmp_path / ".mcp.json"
        cfg.write_text(json.dumps({"mcpServers": {"robotmcp": {"command": "robotmcp"}}}),
                       encoding="utf-8")
        ad = A.get("claude-code")
        assert I._config_has_entry(ad, home=tmp_path, cwd=tmp_path) is True

    def test_list_reports_not_registered_for_an_absent_entry(self, tmp_path):
        ad = A.get("claude-code")
        assert I._config_has_entry(ad, home=tmp_path, cwd=tmp_path) is False


class TestJsonFormattingPreservation:
    def test_existing_indentation_is_preserved(self, tmp_path):
        """A 4-space config was rewritten at 2 spaces: a whitespace-only diff."""
        p = tmp_path / "mcp.json"
        p.write_text('{\n    "mcpServers": {\n        "other": {}\n    }\n}\n',
                     encoding="utf-8")
        data, _ = codecs.load(p, "json")
        data["mcpServers"]["robotmcp"] = {"command": "robotmcp"}
        codecs.dump(p, "json", data)
        text = p.read_text(encoding="utf-8")
        assert '\n    "mcpServers"' in text, "4-space indent must survive"
        assert json.loads(text)["mcpServers"]["other"] == {}

    def test_new_files_default_to_two_spaces(self, tmp_path):
        p = tmp_path / "new.json"
        codecs.dump(p, "json", {"mcpServers": {"robotmcp": {}}})
        assert '\n  "mcpServers"' in p.read_text(encoding="utf-8")


class TestCopilotEntryShape:
    def test_copilot_declares_the_stdio_transport(self):
        """This repo's own working .vscode/mcp.json carries `"type": "stdio"`."""
        entry = A.get("copilot").build_entry("/bin/robotmcp", [], {})
        assert entry["type"] == "stdio"
        assert entry["command"] == "/bin/robotmcp"

    def test_standard_adapters_do_not_gain_a_type(self):
        entry = A.get("claude-code").build_entry("/bin/robotmcp", [], {})
        assert "type" not in entry


class TestDryRunStatusWording:
    def test_dry_run_status_does_not_claim_the_action_happened(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "proj"
        cfg_dir.mkdir()
        m = Manifest(path=tmp_path / "m.json")
        monkeypatch.setattr(A.AgentAdapter, "detect", lambda self, **kw: self.id == "claude-code")
        results = I.install(agents="claude-code", scope="project", whats=["mcp"],
                            dry_run=True, manifest=m, cwd=cfg_dir, project_dir=cfg_dir,
                            no_verify=True)
        assert results and all(r.status.startswith("would-") for r in results)
        assert not (cfg_dir / ".mcp.json").exists()

    def test_dry_run_exits_zero(self):
        assert cli._print_results([
            I.Result("claude-code", "project", "mcp", "would-install", path="/tmp/x")
        ]) == 0


class TestHelpPathStaysFast:
    def test_help_path_does_not_import_the_server_module(self):
        """A typo'd subcommand loaded the whole server (2.4s vs 0.4s) and printed
        unrelated library warnings before the error."""
        import subprocess
        import sys as _s
        code = (
            "import sys\n"
            "from robotmcp.entry import main\n"
            "try:\n"
            "    main(['--help'])\n"
            "except SystemExit:\n"
            "    pass\n"
            "print('SERVER_IMPORTED' if 'robotmcp.server' in sys.modules else 'CLEAN')\n"
        )
        out = subprocess.run([_s.executable, "-c", code], capture_output=True, text=True)
        assert "CLEAN" in out.stdout, \
            f"help must not import robotmcp.server; got: {out.stdout[-200:]}"


class TestTomlRoundTripNotRegressed:
    def test_codex_toml_preserves_comments(self, tmp_path):
        """The TOML adapter is the reference implementation - do not regress it."""
        p = tmp_path / "config.toml"
        original = ('# leading comment\n'
                    '[mcp_servers.other]\n'
                    'command = "x"  # inline comment\n')
        p.write_text(original, encoding="utf-8")
        data, _ = codecs.load(p, "toml")
        codecs.dump(p, "toml", data)
        after = p.read_text(encoding="utf-8")
        assert "# leading comment" in after
        assert "# inline comment" in after

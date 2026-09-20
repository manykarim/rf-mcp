"""Regression tests for launch/report environment fidelity (change: launch-env-fidelity).

Each test pins a defect measured against a real project fixture through an MCP client
(evidence: experiments/tool_install_env_coupling_evidence.md). The common theme: rf-mcp
reported the state of the environment it runs in, not the one the tests run in.

Run: uv run pytest tests/unit/test_launch_env_fidelity.py -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from robotmcp.components.execution import suite_execution_service as SES
from robotmcp.onboarding import installer as I
from robotmcp.onboarding import project_env as pe


def _patch_direct_url(monkeypatch, payload):
    class _Dist:
        def read_text(self, name):
            return payload

    import importlib.metadata as md
    monkeypatch.setattr(md, "distribution", lambda name: _Dist())


# =============================================================================
# D1 - the overlay carries the installation's extras
# =============================================================================


class TestOverlayCarriesExtras:
    """`uv tool install "rf-mcp[all]"` + `robotmcp install` produced an overlay with
    NO Browser and NO SeleniumLibrary: the spec was a bare `rf-mcp==<ver>`."""

    def test_extras_are_derived_by_import_probe(self, monkeypatch):
        import importlib.util as u
        real = u.find_spec

        def fake(name, *a, **k):
            if name in ("Browser", "SeleniumLibrary", "RequestsLibrary"):
                return object()
            if name in I._MODULE_TO_EXTRA:
                return None
            return real(name, *a, **k)

        monkeypatch.setattr(u, "find_spec", fake)
        assert I.installed_extras() == ["api", "web"]

    def test_bare_install_yields_no_extras(self, monkeypatch):
        import importlib.util as u
        monkeypatch.setattr(u, "find_spec",
                            lambda name, *a, **k: None if name in I._MODULE_TO_EXTRA else object())
        assert I.installed_extras() == []
        assert I._extras_suffix([]) == ""

    def test_suffix_shape(self):
        assert I._extras_suffix(["web", "api"]) == "[web,api]"
        assert I._extras_suffix([]) == ""

    def test_desktop_is_never_overlaid(self):
        """Requesting [desktop] in an overlay would make uv refuse the whole command:
        the PlatynUI pins are pre-releases and the reference is transitive."""
        assert "desktop" not in I._MODULE_TO_EXTRA.values()
        assert "PlatynUI" not in I._MODULE_TO_EXTRA

    def test_published_spec_includes_extras(self, monkeypatch):
        monkeypatch.setattr(I, "_own_version", lambda: "1.2.3")
        monkeypatch.setattr(I, "installed_extras", lambda: ["api", "database", "web"])
        _patch_direct_url(monkeypatch, None)
        args, known = I._rfmcp_with_args()
        assert args == ["--with", "rf-mcp[api,database,web]==1.2.3"]
        assert known is True


# =============================================================================
# D2 - editable only for a directory
# =============================================================================


class TestEditableOnlyForDirectory:
    """`--with-editable <wheel>` is rejected by uv:
    'Editable must refer to a local directory, not an archive'."""

    def test_wheel_source_uses_with_not_with_editable(self, tmp_path, monkeypatch):
        wheel = tmp_path / "rf_mcp-1.2.3-py3-none-any.whl"
        wheel.write_bytes(b"not really a wheel")
        monkeypatch.setattr(I, "installed_extras", lambda: ["web"])
        _patch_direct_url(monkeypatch,
                          json.dumps({"url": wheel.as_uri(), "dir_info": {}}))
        args, known = I._rfmcp_with_args()
        assert args[0] == "--with", "an archive must not use --with-editable"
        assert args[1].endswith("[web]")
        assert str(wheel) in args[1]

    def test_directory_source_still_uses_with_editable(self, tmp_path, monkeypatch):
        src = tmp_path / "checkout"
        src.mkdir()
        _patch_direct_url(monkeypatch,
                          json.dumps({"url": src.as_uri(), "dir_info": {"editable": True}}))
        args, _ = I._rfmcp_with_args()
        assert args[0] == "--with-editable"
        assert Path(args[1]) == src

    def test_editable_reference_takes_no_extras_suffix(self, tmp_path, monkeypatch):
        src = tmp_path / "checkout"
        src.mkdir()
        monkeypatch.setattr(I, "installed_extras", lambda: ["web", "api"])
        _patch_direct_url(monkeypatch,
                          json.dumps({"url": src.as_uri(), "dir_info": {"editable": True}}))
        args, _ = I._rfmcp_with_args()
        assert "[" not in args[1], "an editable dir reference takes no extras suffix"


# =============================================================================
# D3 - the Robot Framework that will execute is reported
# =============================================================================


def _fake_env(tmp_path, python=None):
    venv = tmp_path / ".venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    py = venv / "bin" / "python"
    py.write_text("", encoding="utf-8")
    return pe.ProjectEnv(project_dir=tmp_path, type="venv", python=python or py, is_venv=True)


class TestRfVersionReporting:
    """The overlay's RF shadows the project's; `rf_conflict` only guards MAJOR."""

    def test_full_version_is_captured(self, monkeypatch):
        monkeypatch.setattr(pe, "_run", lambda *a, **k: "7.2.1|3.12")
        info = pe.project_rf_info(Path("/fake/python"))
        assert info["robot_version"] == "7.2.1"
        assert info["robot_major"] == 7

    def test_differing_minor_version_is_reported(self, tmp_path, monkeypatch):
        env = _fake_env(tmp_path)
        monkeypatch.setattr(pe, "project_rf_info",
                            lambda py: {"robot_version": "7.2.1", "robot_major": 7,
                                        "py_version": (3, 12)})
        note = pe.rf_version_note(env)
        assert note and "7.2.1" in note
        import robot
        assert robot.version.VERSION in note

    def test_matching_version_is_silent(self, tmp_path, monkeypatch):
        import robot
        env = _fake_env(tmp_path)
        monkeypatch.setattr(pe, "project_rf_info",
                            lambda py: {"robot_version": robot.version.VERSION,
                                        "robot_major": 7, "py_version": (3, 12)})
        assert pe.rf_version_note(env) is None

    def test_no_project_rf_is_silent(self, tmp_path, monkeypatch):
        env = _fake_env(tmp_path)
        monkeypatch.setattr(pe, "project_rf_info",
                            lambda py: {"robot_version": None, "robot_major": None,
                                        "py_version": (3, 12)})
        assert pe.rf_version_note(env) is None

    def test_major_conflict_still_routes_to_attach(self, tmp_path, monkeypatch):
        """Reporting must not change rf_conflict's routing - a major mismatch is the
        only case where the overlay is genuinely unsafe."""
        env = _fake_env(tmp_path)
        monkeypatch.setattr(pe, "project_rf_info",
                            lambda py: {"robot_version": "6.1.1", "robot_major": 6,
                                        "py_version": (3, 12)})
        conflict = pe.rf_conflict(env)
        assert conflict and "6.x" in conflict


# =============================================================================
# D4 - validation status reflects error-severity issues
# =============================================================================


class TestValidationStatusGate:
    """A dry run whose imports FAILED reported success:true / validation_status:passed
    because `return_code == 0` short-circuited the gate."""

    def _parse(self, stdout="", stderr="", return_code=0):
        from robotmcp.models.config_models import ExecutionConfig

        svc = SES.SuiteExecutionService(ExecutionConfig())
        return svc._parse_dry_run_output(stdout, stderr, return_code, {})

    def test_failed_imports_fail_the_validation(self):
        out = "[ ERROR ] Importing library 'AcmeLibrary' failed: ModuleNotFoundError"
        r = self._parse(stdout=out, return_code=0)
        assert r["validation_status"] == "failed"
        assert r["success"] is False
        assert r["validation_results"]["imports_valid"] is False

    def test_missing_keyword_fails_the_validation(self):
        out = "No keyword with name 'Acme Add' found."
        r = self._parse(stdout=out, return_code=0)
        assert r["validation_status"] == "failed"
        assert r["success"] is False

    def test_clean_validation_passes(self):
        r = self._parse(stdout="1 test, 1 passed, 0 failed", return_code=0)
        assert r["validation_status"] == "passed"
        assert r["success"] is True

    def test_warnings_alone_do_not_fail(self):
        out = "Keyword 'Old Thing' is deprecated"
        r = self._parse(stdout=out, return_code=0)
        assert r["validation_status"] != "failed"
        assert r["success"] is True

    def test_nonzero_return_code_still_fails(self):
        r = self._parse(stdout="", return_code=1)
        assert r["validation_status"] == "failed"
        assert r["success"] is False

    def test_success_follows_validation_status(self):
        """They were computed independently, which is how success:true coexisted with
        two error-severity import failures."""
        out = "[ ERROR ] Importing resource 'x.resource' failed"
        r = self._parse(stdout=out, return_code=0)
        assert r["success"] == (r["validation_status"] != "failed")

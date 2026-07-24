"""Unit tests for project-aware launch resolution (change:
installer-project-aware-launch).

Covers the detector (env types, precedence, conflict, extra-libs), the uv-first
resolver (each decision-tree branch), and the verification gate (a command that
fails verification is not written). Launch-heavy end-to-end paths are exercised
lightly / mocked so the suite stays fast and needs no network.

Run: uv run pytest tests/unit/test_installer_project_aware.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from robotmcp.onboarding import installer as I
from robotmcp.onboarding import project_env as pe


# --- fixtures ---------------------------------------------------------------

def _make_venv(dirpath: Path) -> Path:
    """A minimal fake virtualenv: pyvenv.cfg + bin/python."""
    venv = dirpath / ".venv"
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    (venv / "pyvenv.cfg").write_text("home = /usr\n", encoding="utf-8")
    py = venv / "bin" / "python"
    py.write_text("", encoding="utf-8")
    py.chmod(0o755)
    return py


# =============================================================================
# Detector: env type + precedence
# =============================================================================


class TestDetect:
    def test_uv_project(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
        (tmp_path / "uv.lock").write_text("", encoding="utf-8")
        _make_venv(tmp_path)
        e = pe.detect(tmp_path)
        assert e.type == "uv" and e.is_venv and e.python is not None

    def test_plain_venv(self, tmp_path):
        _make_venv(tmp_path)
        e = pe.detect(tmp_path)
        assert e.type == "venv" and e.is_venv

    def test_poetry_wins_over_venv(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[tool.poetry]\nname='x'\n", encoding="utf-8")
        (tmp_path / "poetry.lock").write_text("", encoding="utf-8")
        _make_venv(tmp_path)
        e = pe.detect(tmp_path)
        assert e.type == "poetry"

    def test_hatchling_build_backend_is_not_hatch_env(self, tmp_path):
        # hatchling as BUILD backend must NOT be misread as a hatch-managed env.
        (tmp_path / "pyproject.toml").write_text(
            "[build-system]\nrequires=['hatchling']\n"
            "[tool.hatch.build.targets.wheel]\npackages=['x']\n"
            "[project]\nname='x'\n", encoding="utf-8")
        (tmp_path / "uv.lock").write_text("", encoding="utf-8")
        _make_venv(tmp_path)
        assert pe.detect(tmp_path).type == "uv"

    def test_hatch_envs_is_hatch(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            "[tool.hatch.envs.default]\ndependencies=[]\n", encoding="utf-8")
        _make_venv(tmp_path)
        assert pe.detect(tmp_path).type == "hatch"

    def test_bare_dir_is_none_global_case(self, tmp_path):
        # No markers → global-tool case (uvx/uv tool stays easy).
        assert pe.detect(tmp_path).type == "none"
        assert pe.detect(tmp_path).has_env is False

    def test_looks_like_project(self, tmp_path):
        assert pe.looks_like_project(tmp_path) is False
        (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
        assert pe.looks_like_project(tmp_path) is True


# =============================================================================
# Detector: conflict + extra libraries
# =============================================================================


class TestConflictAndExtras:
    def test_conflict_on_old_robotframework(self, tmp_path, monkeypatch):
        e = pe.ProjectEnv(project_dir=tmp_path, type="venv",
                          python=Path("/x/py"), is_venv=True)
        monkeypatch.setattr(pe, "project_rf_info",
                            lambda py: {"robot_major": 6, "py_version": (3, 12)})
        reason = pe.rf_conflict(e)
        assert reason and "6" in reason

    def test_no_conflict_on_rf7(self, tmp_path, monkeypatch):
        e = pe.ProjectEnv(project_dir=tmp_path, type="venv",
                          python=Path("/x/py"), is_venv=True)
        monkeypatch.setattr(pe, "project_rf_info",
                            lambda py: {"robot_major": 7, "py_version": (3, 12)})
        assert pe.rf_conflict(e) is None

    def test_extra_libs_from_robot_files(self, tmp_path, monkeypatch):
        (tmp_path / "t.robot").write_text(
            "*** Settings ***\n"
            "Library    SeleniumLibrary\n"       # bundled -> dropped
            "Library    AcmeLibrary\n"           # extra -> kept
            "Resource   pages/login.resource\n", # extra resource -> 'login'
            encoding="utf-8")
        e = pe.ProjectEnv(project_dir=tmp_path, type="venv", python=None, is_venv=True)
        monkeypatch.setattr(pe, "_installed_extra_libraries", lambda py: [])
        libs = pe.project_extra_libraries(e)
        assert "AcmeLibrary" in libs and "login" in libs
        assert "SeleniumLibrary" not in libs


# =============================================================================
# Resolver: each decision-tree branch
# =============================================================================


def _fake_env(tmp_path, *, type="venv", is_venv=True):
    py = _make_venv(tmp_path)
    return pe.ProjectEnv(project_dir=tmp_path, type=type, python=py, is_venv=is_venv)


class TestResolver:
    def test_user_scope_is_own_shim(self):
        plan = I.resolve_launch(scope="user")
        assert plan.strategy == "own-shim" and plan.args == []

    def test_no_project_is_own_shim(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pe, "detect",
                            lambda d: pe.ProjectEnv(project_dir=Path(d or "."), type="none"))
        plan = I.resolve_launch(scope="project", project_dir=tmp_path)
        assert plan.strategy == "own-shim"

    def test_rfmcp_in_project_runs_in_project(self, tmp_path, monkeypatch):
        env = _fake_env(tmp_path)
        monkeypatch.setattr(pe, "detect", lambda d: env)
        monkeypatch.setattr(pe, "rfmcp_in_project", lambda py: True)
        monkeypatch.setattr(pe, "project_extra_libraries", lambda e: ["AcmeLibrary"])
        plan = I.resolve_launch(scope="project", project_dir=tmp_path)
        assert plan.strategy == "in-project"
        assert str(env.python) in (plan.command + " ".join(plan.args))

    def test_conflict_routes_to_attach(self, tmp_path, monkeypatch):
        env = _fake_env(tmp_path)
        monkeypatch.setattr(pe, "detect", lambda d: env)
        monkeypatch.setattr(pe, "rfmcp_in_project", lambda py: False)
        monkeypatch.setattr(pe, "rf_conflict", lambda e: "RF6 conflict")
        monkeypatch.setattr(pe, "project_extra_libraries", lambda e: [])
        plan = I.resolve_launch(scope="project", project_dir=tmp_path)
        assert plan.strategy == "attach"
        assert plan.env.get("ROBOTMCP_ATTACH_HOST") == "127.0.0.1"

    def test_generic_project_is_own_shim(self, tmp_path, monkeypatch):
        env = _fake_env(tmp_path)
        monkeypatch.setattr(pe, "detect", lambda d: env)
        monkeypatch.setattr(pe, "rfmcp_in_project", lambda py: False)
        monkeypatch.setattr(pe, "rf_conflict", lambda e: None)
        monkeypatch.setattr(pe, "project_extra_libraries", lambda e: [])  # nothing extra
        plan = I.resolve_launch(scope="project", project_dir=tmp_path)
        assert plan.strategy == "own-shim"

    def test_extra_libs_venv_uv_gives_overlay(self, tmp_path, monkeypatch):
        env = _fake_env(tmp_path, type="venv")
        monkeypatch.setattr(pe, "detect", lambda d: env)
        monkeypatch.setattr(pe, "rfmcp_in_project", lambda py: False)
        monkeypatch.setattr(pe, "rf_conflict", lambda e: None)
        monkeypatch.setattr(pe, "project_extra_libraries", lambda e: ["JSONLibrary"])
        monkeypatch.setattr(I.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
        plan = I.resolve_launch(scope="project", project_dir=tmp_path)
        assert plan.strategy == "uv-overlay"
        assert plan.command == "uv"
        assert "--no-project" in plan.args and "--python" in plan.args
        assert str(env.python) in plan.args
        assert plan.args[-1] == "robotmcp"
        assert plan.verify_lib == "JSONLibrary"

    def test_extra_libs_no_uv_is_fallback(self, tmp_path, monkeypatch):
        env = _fake_env(tmp_path, type="venv")
        monkeypatch.setattr(pe, "detect", lambda d: env)
        monkeypatch.setattr(pe, "rfmcp_in_project", lambda py: False)
        monkeypatch.setattr(pe, "rf_conflict", lambda e: None)
        monkeypatch.setattr(pe, "project_extra_libraries", lambda e: ["JSONLibrary"])
        monkeypatch.setattr(I.shutil, "which", lambda name: None)  # no uv
        plan = I.resolve_launch(scope="project", project_dir=tmp_path)
        assert plan.strategy == "fallback"
        assert plan.verify_lib == "JSONLibrary"  # so it is refused unless overridden

    def test_command_override_wins(self):
        plan = I.resolve_launch(scope="project", command_override="/opt/robotmcp")
        assert plan.command == "/opt/robotmcp" and plan.strategy == "override"

    def test_attach_flag_wins(self, tmp_path):
        plan = I.resolve_launch(scope="project", project_dir=tmp_path, attach="1.2.3.4:9999")
        assert plan.strategy == "attach"
        assert plan.env["ROBOTMCP_ATTACH_HOST"] == "1.2.3.4"
        assert plan.env["ROBOTMCP_ATTACH_PORT"] == "9999"


# =============================================================================
# Verification gate: a command that fails verification is not written
# =============================================================================


class TestVerificationGate:
    def _install(self, tmp_path, monkeypatch, *, verified, force=False):
        # force a uv-overlay plan, then control verification
        env = _fake_env(tmp_path, type="venv")
        monkeypatch.setattr(pe, "detect", lambda d: env)
        monkeypatch.setattr(pe, "rfmcp_in_project", lambda py: False)
        monkeypatch.setattr(pe, "rf_conflict", lambda e: None)
        monkeypatch.setattr(pe, "project_extra_libraries", lambda e: ["JSONLibrary"])
        monkeypatch.setattr(I.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)
        monkeypatch.setattr(I, "verify_launch",
                            lambda plan, **k: (verified, "ok" if verified else "lib not reachable"))
        from robotmcp.onboarding.manifest import Manifest
        man = Manifest(path=tmp_path / "manifest.json")
        return I.install(agents="claude-code", scope="project", project_dir=tmp_path,
                         cwd=tmp_path, force=force, manifest=man)

    def test_unverified_command_is_not_written(self, tmp_path, monkeypatch):
        results = self._install(tmp_path, monkeypatch, verified=False)
        r = next(x for x in results if x.agent == "claude-code")
        assert r.status == "unverified"
        assert not (tmp_path / ".mcp.json").exists()

    def test_verified_command_is_written(self, tmp_path, monkeypatch):
        results = self._install(tmp_path, monkeypatch, verified=True)
        r = next(x for x in results if x.agent == "claude-code")
        assert r.status == "installed"
        assert (tmp_path / ".mcp.json").exists()

    def test_force_writes_despite_failure(self, tmp_path, monkeypatch):
        results = self._install(tmp_path, monkeypatch, verified=False, force=True)
        r = next(x for x in results if x.agent == "claude-code")
        assert r.status == "installed"
        assert (tmp_path / ".mcp.json").exists()


class TestWindowsConsoleSafe:
    """Onboarding CLI output must survive the Windows default console (cp1252):
    a stray non-ASCII glyph raises UnicodeEncodeError and crashes init/install."""

    def test_onboarding_sources_are_ascii(self):
        import glob
        offenders = []
        for f in glob.glob("src/robotmcp/onboarding/*.py"):
            for i, line in enumerate(open(f, encoding="utf-8"), 1):
                for c in line:
                    if ord(c) > 127:
                        offenders.append(f"{f}:{i}: {c!r}")
        assert not offenders, "non-ASCII in onboarding output would crash cp1252 consoles:\n" + "\n".join(offenders)

    def test_result_and_doctor_output_encodes_cp1252(self, capsys):
        from robotmcp.onboarding import diagnostics
        from robotmcp.onboarding.cli import _print_results
        from robotmcp.onboarding.installer import Result
        _print_results([
            Result("claude-code", "project", "mcp", "installed", path="/x/.mcp.json",
                   detail="[uv-overlay] runs against the project's venv - sees JSONLibrary"),
        ])
        diagnostics.cmd_doctor()
        out = capsys.readouterr().out
        out.encode("cp1252")  # must not raise UnicodeEncodeError
        assert "->" in out or "installed" in out


# =============================================================================
# The overlay references the SAME rf-mcp cross-platform
# (change: fix-installer-windows-file-url) - a local/unpublished source install
# must overlay --with-editable <src> on every platform, incl. Windows drive
# letters, never silently degrading to an unresolvable version pin.
# =============================================================================

class _FakeDist:
    """Stand-in for importlib.metadata.distribution('rf-mcp')."""
    def __init__(self, direct_url_json):
        self._j = direct_url_json

    def read_text(self, name):
        return self._j if name == "direct_url.json" else None


def _patch_direct_url(monkeypatch, direct_url_json):
    import importlib.metadata as im
    monkeypatch.setattr(im, "distribution", lambda name: _FakeDist(direct_url_json))


class TestOverlayRfmcpReference:
    def test_local_source_url_overlays_with_editable_posix(self, tmp_path, monkeypatch):
        # 2.1 a file:// URL pointing at a real dir -> --with-editable <dir>
        src = tmp_path / "rf-mcp-src"
        src.mkdir()
        _patch_direct_url(monkeypatch,
                          json.dumps({"url": src.as_uri(), "dir_info": {"editable": True}}))
        result = I._rfmcp_with_args()
        assert result[0] == "--with-editable"
        assert Path(result[1]) == src

    def test_file_url_to_path_windows_drive_letter(self, monkeypatch):
        # 2.2 the drive-letter conversion, exercised cross-platform via nturl2path
        import urllib.request
        import nturl2path
        monkeypatch.setattr(urllib.request, "url2pathname", nturl2path.url2pathname)
        assert I._file_url_to_path("file:///C:/work/rf-mcp") == r"C:\work\rf-mcp"
        # the old naive slice produced an invalid '/C:/work/rf-mcp':
        assert "file:///C:/work/rf-mcp"[7:] == "/C:/work/rf-mcp"

    def test_no_direct_url_uses_version_pin(self, monkeypatch):
        # 2.3 published install (no direct_url.json) -> --with rf-mcp==<ver>
        monkeypatch.setattr(I, "_own_version", lambda: "9.9.9")
        _patch_direct_url(monkeypatch, None)
        assert I._rfmcp_with_args() == ["--with", "rf-mcp==9.9.9"]

    def test_file_url_missing_dir_falls_back_to_pin(self, tmp_path, monkeypatch):
        # 2.3 a file:// URL whose dir does NOT exist -> version pin (guard preserved)
        monkeypatch.setattr(I, "_own_version", lambda: "9.9.9")
        missing = tmp_path / "does-not-exist"
        _patch_direct_url(monkeypatch,
                          json.dumps({"url": missing.as_uri(), "dir_info": {}}))
        assert I._rfmcp_with_args() == ["--with", "rf-mcp==9.9.9"]

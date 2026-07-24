"""Unit tests for project-aware launch resolution (change:
installer-project-aware-launch).

Covers the detector (env types, precedence, conflict, extra-libs), the uv-first
resolver (each decision-tree branch), and the verification gate (a command that
fails verification is not written). Launch-heavy end-to-end paths are exercised
lightly / mocked so the suite stays fast and needs no network.

Run: uv run pytest tests/unit/test_installer_project_aware.py -q
"""
from __future__ import annotations

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

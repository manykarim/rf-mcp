"""Environment resolution per project root (project-independent-server, tasks 2.1-2.3)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from robotmcp.runner.environment import EnvironmentResolver, venv_python

PIN = "2.9.0.dev115"


def make_venv(path: Path, *, layout: str = "nt" if os.name == "nt" else "posix", robotcode_exe: bool = False) -> Path:
    """A fake venv: pyvenv.cfg, an interpreter file and a site-packages folder."""
    scripts = path / ("Scripts" if layout == "nt" else "bin")
    scripts.mkdir(parents=True)
    python = scripts / ("python.exe" if layout == "nt" else "python")
    python.write_text("", encoding="utf-8")
    (path / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")
    (path / ("Lib/site-packages" if layout == "nt" else "lib/python3.12/site-packages")).mkdir(parents=True)
    if robotcode_exe:
        exe = scripts / ("robotcode.exe" if layout == "nt" else "robotcode")
        exe.write_text("", encoding="utf-8")
        exe.chmod(0o755)  # shutil.which needs the executable bit on POSIX
    return python


class Probes:
    def __init__(self, versions: dict[Path, str | None] | None = None, layer_ok: bool = True) -> None:
        self.versions = {k.resolve(): v for k, v in (versions or {}).items()}
        self.layer_ok = layer_ok
        self.version_calls: list[Path] = []
        self.command_calls: list[tuple[str, ...]] = []

    def version(self, interpreter: Path) -> str | None:
        self.version_calls.append(interpreter)
        return self.versions.get(interpreter.resolve())

    def command(self, prefix: tuple[str, ...]) -> tuple[bool, str]:
        self.command_calls.append(prefix)
        if prefix[0] == "uv":
            return (True, f"robotcode, version {PIN}") if self.layer_ok else (False, "No solution found: Python 3.8")
        return True, "robotcode, version 2.9.0-dev.120"


@pytest.fixture
def empty_path(tmp_path: Path) -> dict[str, str]:
    (tmp_path / "_empty_path").mkdir()
    return {"PATH": str(tmp_path / "_empty_path")}


def resolver(probes: Probes, environ: dict[str, str], *, own: Path | None = None, uv: str | bool = "uv"):
    return EnvironmentResolver(
        own_prefix=own or Path("/nonexistent/rf-mcp-tool-env"), uv=uv, pin=PIN,  # type: ignore[arg-type]
        version_probe=probes.version, command_probe=probes.command, environ=environ,
    )


def test_project_install_uses_its_own_environment_without_probing(tmp_path: Path, empty_path) -> None:
    make_venv(tmp_path / ".venv")
    probes = Probes()
    env = resolver(probes, empty_path, own=tmp_path / ".venv").resolve(tmp_path)
    assert env.source == "own" and not env.layered and env.warning is None
    assert probes.version_calls == [] and probes.command_calls == []


def test_a_configured_interpreter_wins_over_the_venv(tmp_path: Path, empty_path) -> None:
    make_venv(tmp_path / ".venv")
    configured = make_venv(tmp_path / "envs" / "rf", layout="posix", robotcode_exe=True)
    (tmp_path / ".robotframework").mkdir()
    (tmp_path / ".robotframework" / "rf-mcp.toml").write_text('python = "envs/rf/bin/python"\n', encoding="utf-8")
    probes = Probes({configured: "2.9.0.dev186"})
    env = resolver(probes, empty_path).resolve(tmp_path)
    assert env.source == "configured" and env.interpreter == tmp_path / "envs/rf/bin/python"
    assert env.command_prefix == (str(configured.parent / "robotcode"),)


def test_a_missing_configured_interpreter_is_reported_and_skipped(tmp_path: Path, empty_path) -> None:
    python = make_venv(tmp_path / ".venv", robotcode_exe=True)
    (tmp_path / ".robotframework").mkdir()
    (tmp_path / ".robotframework" / "rf-mcp.toml").write_text('python = "gone/python"\n', encoding="utf-8")
    env = resolver(Probes({python: "2.9.0.dev186"}), empty_path).resolve(tmp_path)
    assert env.source == "venv"
    assert any("gone/python" in note for note in env.notes)


def test_a_supported_project_robotcode_runs_without_uv(tmp_path: Path, empty_path) -> None:
    python = make_venv(tmp_path / ".venv", robotcode_exe=True)
    probes = Probes({python: "2.9.0.dev186"})
    env = resolver(probes, empty_path).resolve(tmp_path)
    assert env.source == "venv" and not env.layered and env.robotcode_version == "2.9.0.dev186"
    assert env.command_prefix[0].endswith(("robotcode.exe", "robotcode")) and probes.command_calls == []


@pytest.mark.parametrize("project_version", [None, "1.5.0"])
def test_missing_or_old_robotcode_gets_rf_mcps_version_layered(tmp_path: Path, empty_path, project_version) -> None:
    python = make_venv(tmp_path / ".venv")
    probes = Probes({python: project_version})
    env = resolver(probes, empty_path).resolve(tmp_path)
    assert env.source == "venv" and env.layered and env.robotcode_version == PIN
    assert env.command_prefix == ("uv", "run", "--quiet", "--no-project", "--python", str(python),
                                  "--with", f"robotcode[all]=={PIN}", "robotcode")
    assert probes.command_calls == [env.command_prefix]
    assert env.warning is None


def test_without_uv_nothing_is_layered_and_both_remedies_are_named(tmp_path: Path, empty_path) -> None:
    python = make_venv(tmp_path / ".venv")
    probes = Probes({python: None})
    env = resolver(probes, empty_path, uv=False).resolve(tmp_path)
    assert env.source == "fallback" and not env.libraries_visible
    assert probes.command_calls == []
    assert "install uv" in env.warning and 'uv add --dev "robotcode[all]' in env.warning


def test_a_failing_layer_falls_back_with_uvs_error(tmp_path: Path, empty_path) -> None:
    python = make_venv(tmp_path / ".venv")
    env = resolver(Probes({python: None}, layer_ok=False), empty_path).resolve(tmp_path)
    assert env.source == "fallback" and "No solution found" in env.warning


def test_the_activated_environment_is_used_when_the_project_has_no_venv(tmp_path: Path, empty_path) -> None:
    active = make_venv(tmp_path / "poetry-env", robotcode_exe=True)
    project = tmp_path / "project"
    project.mkdir()
    env = resolver(Probes({active: "2.9.0.dev150"}), {**empty_path, "VIRTUAL_ENV": str(tmp_path / "poetry-env")}).resolve(project)
    assert env.source == "activated" and env.interpreter == active


def test_robotcode_on_path_from_rf_mcps_own_environment_is_own(tmp_path: Path) -> None:
    own = tmp_path / "app-venv"
    make_venv(own, robotcode_exe=True)
    scripts = own / ("Scripts" if os.name == "nt" else "bin")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    probes = Probes()
    env = resolver(probes, {"PATH": str(scripts)}, own=own).resolve(workspace)
    assert env.source == "own" and env.warning is None and probes.command_calls == []


def test_a_supported_robotcode_on_path_is_used(tmp_path: Path) -> None:
    other = tmp_path / "global"
    make_venv(other, robotcode_exe=True)
    project = tmp_path / "project"
    project.mkdir()
    env = resolver(Probes(), {"PATH": str(other / ("Scripts" if os.name == "nt" else "bin"))}).resolve(project)
    assert env.source == "path" and env.robotcode_version == "2.9.0-dev.120"


def test_nothing_usable_falls_back_to_rf_mcps_robotcode_with_a_warning(tmp_path: Path, empty_path) -> None:
    env = resolver(Probes(), empty_path).resolve(tmp_path)
    assert env.source == "fallback"
    assert "project's libraries are not visible" in env.warning and "rf-mcp doctor" in env.warning


@pytest.mark.parametrize("layout", ["nt", "posix"])
def test_windows_and_posix_venv_layouts_are_found(tmp_path: Path, layout: str) -> None:
    python = make_venv(tmp_path / "v", layout=layout)
    assert venv_python(tmp_path / "v") == python


def test_resolutions_are_cached_until_the_environment_changes(tmp_path: Path, empty_path) -> None:
    python = make_venv(tmp_path / ".venv")
    probes = Probes({python: None})
    res = resolver(probes, empty_path)
    first = res.resolve(tmp_path)
    assert res.resolve(tmp_path) is first
    assert len(probes.version_calls) == 1 and len(probes.command_calls) == 1

    # Installing a package touches site-packages; the next call resolves again.
    probes.versions[python.resolve()] = "2.9.0.dev186"
    (python.parent / ("robotcode.exe" if os.name == "nt" else "robotcode")).write_text("", encoding="utf-8")
    site = next((tmp_path / ".venv").glob("Lib/site-packages")) if os.name == "nt" else next((tmp_path / ".venv").glob("lib/python*/site-packages"))
    later = site.stat().st_mtime + 10
    os.utime(site, (later, later))
    second = res.resolve(tmp_path)
    assert second.source == "venv" and not second.layered and len(probes.version_calls) == 2


def test_the_documented_rf_mcp_toml_example_resolves(tmp_path: Path, empty_path) -> None:
    """docs/CONFIGURATION.md, "Which Python runs your tests": the example line is used as written."""
    doc = (Path(__file__).resolve().parents[1] / "docs" / "CONFIGURATION.md").read_text(encoding="utf-8")
    line = next(ln.strip() for ln in doc.splitlines() if ln.strip().startswith('python = "'))
    rel = line.split('"')[1]
    python = tmp_path / rel
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    (python.parent / "robotcode").write_text("", encoding="utf-8")
    (tmp_path / ".robotframework").mkdir()
    (tmp_path / ".robotframework" / "rf-mcp.toml").write_text(line + "\n", encoding="utf-8")
    env = resolver(Probes({python: "2.9.0.dev186"}), empty_path).resolve(tmp_path)
    assert env.source == "configured" and env.interpreter == python


def test_a_failed_layer_does_not_hide_behind_rf_mcps_own_activated_environment(tmp_path: Path) -> None:
    """Found on Linux with uv 0.9: the layer failed and `uv run`'s VIRTUAL_ENV (rf-mcp's own) ran silently."""
    own = tmp_path / "rf-mcp-venv"
    make_venv(own, robotcode_exe=True)
    project = tmp_path / "shop"
    python = make_venv(project / ".venv")
    scripts = own / ("Scripts" if os.name == "nt" else "bin")
    env = resolver(Probes({python: None}, layer_ok=False), {"PATH": str(scripts), "VIRTUAL_ENV": str(own)},
                   own=own).resolve(project)
    assert env.source == "fallback" and "project's libraries are not visible" in env.warning
    assert "No solution found" in env.warning


def test_rf_mcps_own_environment_on_path_is_fine_for_a_root_without_an_environment(tmp_path: Path) -> None:
    """Docker: /workspace has no .venv and the image's environment holds the libraries."""
    own = tmp_path / "app-venv"
    make_venv(own, robotcode_exe=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    scripts = own / ("Scripts" if os.name == "nt" else "bin")
    env = resolver(Probes(), {"PATH": str(scripts), "VIRTUAL_ENV": str(own)}, own=own).resolve(workspace)
    assert env.source == "own" and env.warning is None


def test_the_default_layer_pins_every_robotcode_sub_package() -> None:
    from robotmcp.runner.environment import own_robotcode_requirements, own_robotcode_version

    reqs = own_robotcode_requirements()
    version = own_robotcode_version()
    assert reqs[0] == f"robotcode[all]=={version}"
    assert f"robotcode-core=={version}" in reqs and f"robotcode-robot=={version}" in reqs
    assert all(r.split("==")[1] == version for r in reqs)
    layered = EnvironmentResolver(uv="uv", version_probe=lambda _p: None,
                                  command_probe=lambda _p: (True, "")).requirements
    assert layered == reqs


# library-aware-skill-retrieval 4.1: the packages installed where a root's robotcode runs.


class Distributions:
    def __init__(self, packages: dict[str, str] | None) -> None:
        self.packages = packages
        self.calls: list[Path] = []

    def __call__(self, interpreter: Path) -> dict[str, str] | None:
        self.calls.append(interpreter)
        return None if self.packages is None else dict(self.packages)


def inventory_resolver(probes: Probes, environ: dict[str, str], dists: Distributions) -> EnvironmentResolver:
    return EnvironmentResolver(
        own_prefix=Path("/nonexistent/rf-mcp-tool-env"), uv="uv", pin=PIN,
        version_probe=probes.version, command_probe=probes.command, distributions_probe=dists, environ=environ,
    )


def test_the_project_venvs_packages_are_listed(tmp_path: Path, empty_path) -> None:
    python = make_venv(tmp_path / ".venv", robotcode_exe=True)
    dists = Distributions({"robotframework-browser": "20.1.0", "robotframework_PlatynUI": "0.12.0"})
    res = inventory_resolver(Probes({python: "2.9.0.dev186"}), empty_path, dists)
    inventory = res.installed_distributions(tmp_path)
    assert inventory == {"robotframework-browser": "20.1.0", "robotframework-platynui": "0.12.0"}
    assert dists.calls == [python]
    assert res.installed_distributions(tmp_path) == inventory and len(dists.calls) == 1  # cached


def test_a_package_installed_later_is_listed(tmp_path: Path, empty_path) -> None:
    python = make_venv(tmp_path / ".venv", robotcode_exe=True)
    dists = Distributions({"robotframework-browser": "20.1.0"})
    res = inventory_resolver(Probes({python: "2.9.0.dev186"}), empty_path, dists)
    assert "robotframework-requests" not in (res.installed_distributions(tmp_path) or {})
    dists.packages = {"robotframework-browser": "20.1.0", "robotframework-requests": "0.9.7"}
    site = next(p for p in (tmp_path / ".venv").rglob("site-packages"))
    later = site.stat().st_mtime + 10
    os.utime(site, (later, later))
    assert res.installed_distributions(tmp_path)["robotframework-requests"] == "0.9.7"


def test_no_inventory_when_libraries_are_not_visible(tmp_path: Path, empty_path) -> None:
    dists = Distributions({"robotcode": "2.9.0"})
    res = inventory_resolver(Probes(), empty_path, dists)
    assert res.resolve(tmp_path).source == "fallback"
    assert res.installed_distributions(tmp_path) is None and dists.calls == []


def test_a_failed_listing_gives_no_inventory(tmp_path: Path, empty_path) -> None:
    python = make_venv(tmp_path / ".venv", robotcode_exe=True)
    res = inventory_resolver(Probes({python: "2.9.0.dev186"}), empty_path, Distributions(None))
    assert res.installed_distributions(tmp_path) is None


def test_robotcode_on_path_lists_the_python_next_to_it(tmp_path: Path) -> None:
    other = tmp_path / "global"
    python = make_venv(other, robotcode_exe=True)
    project = tmp_path / "project"
    project.mkdir()
    dists = Distributions({"robotframework": "7.3"})
    res = inventory_resolver(Probes(), {"PATH": str(python.parent)}, dists)
    assert res.installed_distributions(project) == {"robotframework": "7.3"}
    assert dists.calls == [python]


def test_the_real_probe_lists_this_environments_packages() -> None:
    import sys

    from robotmcp.runner.environment import probe_distributions

    found = probe_distributions(Path(sys.executable))
    assert found is not None and "robotcode" in {k.lower() for k in found}

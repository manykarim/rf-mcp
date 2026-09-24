"""Project Python-environment detection for the project-aware installer
(change: installer-project-aware-launch).

rf-mcp imports Robot Framework libraries in its OWN process ``sys.path``, so to let
the running server see a project's libraries/keywords/resources the installer must
know the project's environment and its interpreter. This module answers, best-effort
and never raising:

* what kind of environment the project uses (uv / poetry / pdm / pipenv / rye / hatch
  / plain venv / conda / bare global), by most-specific marker,
* the path to that environment's Python interpreter,
* whether that interpreter is a real virtualenv (uv can overlay only virtualenvs),
* whether rf-mcp is already importable there,
* whether the project's pinned Robot Framework / Python conflict with rf-mcp,
* which Robot Framework libraries the project references that rf-mcp does not bundle.

Global tool installs (``uvx`` / ``uv tool``) are the easy default and need none of
this - they resolve to ``type == "none"`` when run outside a project.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Libraries rf-mcp[all] already bundles - a project needing only these is served fine
# by rf-mcp's own environment (no project-aware resolution required).
BUNDLED_LIBRARIES = frozenset({
    "Browser", "SeleniumLibrary", "AppiumLibrary", "RequestsLibrary",
    "DatabaseLibrary", "PlatynUI", "PlatynUI.BareMetal",
})
# Robot Framework standard libraries (always available; never "extra").
RF_STDLIB = frozenset({
    "BuiltIn", "Collections", "DateTime", "Dialogs", "OperatingSystem",
    "Process", "Screenshot", "String", "Telnet", "XML", "Remote", "Reserved",
})

# rf-mcp's own floors (kept in sync with pyproject; used only for conflict hints).
RF_MIN_MAJOR = 7          # robotframework>=7.0
PY_MIN = (3, 10)          # requires-python >=3.10

_CMD_TIMEOUT = 8          # best-effort external tool calls (poetry/pdm/...)


@dataclass
class ProjectEnv:
    project_dir: Path
    type: str = "none"                 # uv|poetry|pdm|pipenv|rye|hatch|venv|conda|global|none
    python: Optional[Path] = None      # interpreter path, or None
    is_venv: bool = False              # True when python lives in a real virtualenv (pyvenv.cfg)
    markers: List[str] = field(default_factory=list)

    @property
    def has_env(self) -> bool:
        return self.type != "none" and self.python is not None


# --- interpreter helpers ----------------------------------------------------

def _venv_python(venv_dir: Path) -> Optional[Path]:
    """Return the interpreter inside a venv dir (POSIX ``bin`` or Windows
    ``Scripts``), or None if absent."""
    for rel in ("bin/python", "bin/python3", "Scripts/python.exe"):
        p = venv_dir / rel
        if p.exists():
            return p
    return None


def _is_virtualenv(python: Optional[Path]) -> bool:
    """A real virtualenv has a ``pyvenv.cfg`` next to its ``bin``/``Scripts`` dir.
    uv's ``--with`` overlay only layers the target's site-packages for these."""
    if python is None:
        return False
    try:
        return (python.parent.parent / "pyvenv.cfg").exists()
    except Exception:
        return False


def _run(cmd: List[str], cwd: Optional[Path] = None) -> Optional[str]:
    """Best-effort external command -> stripped stdout, or None. Never raises."""
    try:
        r = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True,
            timeout=_CMD_TIMEOUT, stdin=subprocess.DEVNULL,
        )
        if r.returncode == 0:
            return (r.stdout or "").strip()
    except Exception:
        pass
    return None


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


# --- detection --------------------------------------------------------------

def looks_like_project(project_dir: Path) -> bool:
    """Heuristic: does this directory look like a software project the user meant to
    target (so an explicit ``-C`` pointing at a bare/wrong directory can be flagged)?"""
    d = Path(project_dir)
    if any((d / m).exists() for m in (
            "pyproject.toml", ".git", ".venv", "venv", "requirements.txt",
            "setup.py", "setup.cfg", "Pipfile", "poetry.lock", "uv.lock")):
        return True
    try:
        return any(True for _ in d.glob("*.robot")) or any(True for _ in d.glob("*.resource"))
    except Exception:
        return False


def detect(project_dir: Optional[Path] = None) -> ProjectEnv:
    """Detect the project's Python environment by most-specific marker. Precedence:
    poetry > pdm > hatch > rye > pipenv > uv/PEP621 > plain-.venv > conda(active) >
    VIRTUAL_ENV(active) > bare-global. Returns ``type == "none"`` when nothing is
    found (the global-tool case)."""
    d = Path(project_dir or Path.cwd()).resolve()
    pyproject = _read(d / "pyproject.toml") if (d / "pyproject.toml").exists() else ""
    inproj_venv = _venv_python(d / ".venv") or _venv_python(d / "venv")

    def _env(t: str, py: Optional[Path], *markers: str) -> ProjectEnv:
        return ProjectEnv(project_dir=d, type=t, python=py,
                          is_venv=_is_virtualenv(py), markers=[m for m in markers if m])

    # poetry
    if "[tool.poetry]" in pyproject or (d / "poetry.lock").exists():
        py = inproj_venv
        if py is None:
            path = _run(["poetry", "env", "info", "--path"], cwd=d)
            if path:
                py = _venv_python(Path(path))
        return _env("poetry", py, "[tool.poetry]" if "[tool.poetry]" in pyproject else "poetry.lock")

    # pdm
    if "[tool.pdm]" in pyproject or (d / "pdm.lock").exists() or (d / ".pdm-python").exists():
        py = None
        if (d / ".pdm-python").exists():
            raw = _read(d / ".pdm-python").strip()
            if raw:
                py = Path(raw)
        py = py if (py and py.exists()) else inproj_venv
        if py is None:
            out = _run(["pdm", "info", "--python"], cwd=d)
            if out:
                py = Path(out)
        return _env("pdm", py, "pdm.lock")

    # hatch - as an ENV MANAGER (``[tool.hatch.envs...]``), not merely hatchling as
    # the build backend (``[tool.hatch.build...]``), which countless projects use.
    if "[tool.hatch.envs" in pyproject:
        py = inproj_venv
        if py is None:
            path = _run(["hatch", "env", "find"], cwd=d)
            if path:
                py = _venv_python(Path(path))
        return _env("hatch", py, "[tool.hatch.envs]")

    # rye
    if "[tool.rye]" in pyproject or (d / "requirements.lock").exists():
        return _env("rye", inproj_venv, "[tool.rye]")

    # pipenv
    if (d / "Pipfile").exists():
        py = inproj_venv
        if py is None:
            path = _run(["pipenv", "--venv"], cwd=d)
            if path:
                py = _venv_python(Path(path))
        return _env("pipenv", py, "Pipfile")

    # uv project (uv.lock, or PEP-621 pyproject with an in-project .venv)
    if (d / "uv.lock").exists() or ((d / "pyproject.toml").exists() and inproj_venv):
        return _env("uv", inproj_venv,
                    "uv.lock" if (d / "uv.lock").exists() else "pyproject+.venv")

    # plain virtualenv in the project
    if inproj_venv is not None:
        return _env("venv", inproj_venv, ".venv")

    # conda - only when the project declares one (environment.yml) AND a conda env is
    # active; project-anchored so an activated conda env can't hijack the global case.
    if (d / "environment.yml").exists() or (d / "environment.yaml").exists():
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            py = Path(conda_prefix) / "bin" / "python"
            if py.exists():
                return _env("conda", py, "environment.yml")

    # Nothing project-specific in this directory -> the global-tool case. We do NOT
    # fall back to an active VIRTUAL_ENV/CONDA_PREFIX: that is not anchored to the
    # project directory and would hijack the easy `uvx`/`uv tool` global install
    # (e.g. picking up rf-mcp's own activated venv). A properly set-up project is
    # identified by its own markers above.
    return ProjectEnv(project_dir=d, type="none", python=None, is_venv=False, markers=[])


# --- probes -----------------------------------------------------------------

def rfmcp_in_project(python: Optional[Path]) -> bool:
    """True when rf-mcp is importable in the project interpreter."""
    if python is None:
        return False
    out = _run([str(python), "-c", "import robotmcp; print('ok')"])
    return out == "ok"


def project_rf_info(python: Optional[Path]) -> dict:
    """Return ``{robot_major, robot_version, py_version}`` for the project interpreter.

    ``robot_version`` is the FULL version string (e.g. ``"7.2.1"``), added so callers
    can report a minor/patch difference between the project's Robot Framework and the
    one that will actually execute the tests. ``robot_major`` keeps its meaning and
    remains what `rf_conflict` routes on (change: launch-env-fidelity sec 3).
    """
    info = {"robot_major": None, "robot_version": None, "py_version": None}
    if python is None:
        return info
    out = _run([str(python), "-c",
                "import sys;"
                "v=None;\n"
                "try:\n import robot; v=robot.version.VERSION\nexcept Exception: pass\n"
                "print((v or '') + '|' + '.'.join(map(str, sys.version_info[:2])))"])
    if not out or "|" not in out:
        return info
    rv, pv = out.split("|", 1)
    if rv:
        info["robot_version"] = rv.strip()
        m = re.match(r"(\d+)", rv)
        if m:
            info["robot_major"] = int(m.group(1))
    if pv:
        parts = pv.split(".")
        try:
            info["py_version"] = (int(parts[0]), int(parts[1]))
        except Exception:
            pass
    return info


def rf_conflict(env: ProjectEnv) -> Optional[str]:
    """Return a human reason when the project's pins are irreconcilable with rf-mcp
    (Robot Framework older than rf-mcp supports, or Python older than its baseline),
    else None. These are the cases that must route to the attach bridge."""
    if not env.has_env:
        return None
    info = project_rf_info(env.python)
    rm = info["robot_major"]
    pv = info["py_version"]
    if rm is not None and rm < RF_MIN_MAJOR:
        return (f"project pins Robot Framework {rm}.x but rf-mcp needs "
                f">={RF_MIN_MAJOR}; overlaying would test a different RF version")
    if pv is not None and pv < PY_MIN:
        return (f"project Python {pv[0]}.{pv[1]} is older than rf-mcp's baseline "
                f"{PY_MIN[0]}.{PY_MIN[1]}")
    return None


def rf_version_note(env: ProjectEnv) -> Optional[str]:
    """Report which Robot Framework will actually execute the tests, when it differs
    from the project's own.

    This is REPORTING, not refusal. `rf_conflict` deliberately routes only MAJOR
    mismatches to the attach bridge, because that is where an overlay is genuinely
    unsafe; a minor difference is usually harmless and refusing it would block
    working setups. But the user is told the overlay lets rf-mcp "see" their
    libraries - they are not told it also replaces their Robot Framework
    (change: launch-env-fidelity sec 3).

    Comparison basis is stated explicitly rather than implied: the project's
    installed RF versus the RF THIS rf-mcp has, which is what an overlay layers on.
    """
    if not env.has_env:
        return None
    project_rf = project_rf_info(env.python).get("robot_version")
    if not project_rf:
        return None
    try:
        import robot
        own_rf = robot.version.VERSION
    except Exception:
        return None
    if project_rf == own_rf:
        return None
    return (f"project's Robot Framework is {project_rf}, but an overlay would execute "
            f"tests with rf-mcp's {own_rf}")


_LIB_RE = re.compile(r"^\s*(Library|Resource)\s{2,}(\S+)", re.MULTILINE)

# Distributions rf-mcp[all] already provides - any OTHER robotframework-* dist in the
# project env is an "extra" library rf-mcp's own environment would be blind to.
_BUNDLED_DISTS = frozenset({
    "robotframework", "robotframework-browser", "robotframework-seleniumlibrary",
    "robotframework-appiumlibrary", "robotframework-requests",
    "robotframework-databaselibrary", "robotframework-platynui",
})


def available_bundled_dists() -> frozenset:
    """The subset of `_BUNDLED_DISTS` THIS rf-mcp installation actually provides.

    `_BUNDLED_DISTS` is a catalogue of what `rf-mcp[all]` *would* ship. Using it
    directly made `doctor` contradict itself: it could report
    `[ ] API (RequestsLibrary)` and, twelve lines later, "extra project libraries:
    none (rf-mcp[all]'s bundle suffices)" for a project whose only dependency was
    RequestsLibrary. A user who installed `rf-mcp[api]` - or, since
    install-extra-resolvability, anyone at all for `robotframework-platynui`, which
    `[all]` no longer includes - does not have the whole catalogue
    (change: installer-cli-safety sec 9).
    """
    import importlib.metadata as _m

    present = set()
    for dist in _BUNDLED_DISTS:
        try:
            _m.version(dist)
            present.add(dist)
        except Exception:
            continue
    return frozenset(present)


def _installed_probe(bundled: frozenset) -> str:
    return (
    "import importlib.metadata as m\n"
    "B=" + repr(set(bundled)) + "\n"
    "out=[]\n"
    "for d in m.distributions():\n"
    "    n=(d.metadata.get('Name') or '').lower()\n"
    "    if n.startswith('robotframework-') and n not in B:\n"
    "        tl=[]\n"
    "        try: tl=[x for x in (d.read_text('top_level.txt') or '').split() if x]\n"
    "        except Exception: tl=[]\n"
    "        out.append(tl[0] if tl else n.replace('robotframework-',''))\n"
    "print('\\n'.join(out))\n"
    )


# Back-compat for callers/tests that referenced the module-level probe string.
_INSTALLED_PROBE = _installed_probe(_BUNDLED_DISTS)


def _installed_extra_libraries(python: Optional[Path]) -> List[str]:
    """Non-bundled ``robotframework-*`` library MODULES installed in the project env
    (e.g. robotframework-jsonlibrary -> JSONLibrary). Best-effort.

    "Non-bundled" is judged against what this rf-mcp ACTUALLY has, not against the
    full `[all]` catalogue - see `available_bundled_dists`.
    """
    if python is None:
        return []
    out = _run([str(python), "-c", _installed_probe(available_bundled_dists())])
    return [ln.strip() for ln in (out or "").splitlines() if ln.strip()] if out else []


def project_extra_libraries(env: ProjectEnv, *, max_files: int = 400) -> List[str]:
    """RF libraries the project needs that rf-mcp does NOT bundle - the signal that the
    project needs project-aware resolution (rf-mcp's own env would be blind to them).
    Combines non-bundled ``robotframework-*`` packages installed in the project env
    with ``Library``/``Resource`` imports scanned from the project's ``.robot`` /
    ``.resource`` files. Best-effort, bounded."""
    found: List[str] = []
    seen = set()
    for mod in _installed_extra_libraries(env.python):
        if mod not in RF_STDLIB and mod not in BUNDLED_LIBRARIES and mod not in seen:
            seen.add(mod); found.append(mod)
    try:
        files = []
        for ext in ("*.robot", "*.resource"):
            files.extend(env.project_dir.rglob(ext))
            if len(files) >= max_files:
                break
        for f in files[:max_files]:
            for _, name in _LIB_RE.findall(_read(f)):
                # normalize: strip path + a known extension for local resources/libs
                base = name.replace("\\", "/").rsplit("/", 1)[-1]
                lib = re.sub(r"\.(resource|robot|py)$", "", base)
                if not lib or lib in RF_STDLIB or lib in BUNDLED_LIBRARIES:
                    continue
                if lib.startswith("${") or lib in seen:  # skip variable-driven imports
                    continue
                seen.add(lib)
                found.append(lib)
    except Exception:
        pass
    return found

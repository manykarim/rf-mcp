"""Choose the environment that runs `robotcode` for a project root.

Order (design D2-D6 of the project-independent-server change):

1. `python` in `<root>/.robotframework/rf-mcp.toml`
2. `<root>/.venv`
3. the activated environment (`VIRTUAL_ENV`), then a `robotcode` on `PATH`
4. rf-mcp's own robotcode, with a warning that the project's libraries are not visible

An environment that is rf-mcp's own runs its robotcode directly. Another one runs its own robotcode when that is at
least `MIN_ROBOTCODE`; otherwise rf-mcp's robotcode version is layered onto its interpreter with
`uv run --no-project --python <interpreter> --with robotcode[all]==<pin>`, which leaves the project unchanged.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

import tomlkit
from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

MIN_ROBOTCODE = "2.9.0.dev115"
CONFIG_FILE = Path(".robotframework") / "rf-mcp.toml"
LAYER_PROBE_TIMEOUT_SECONDS = 300
VERSION_PROBE_TIMEOUT_SECONDS = 30

Source = Literal["own", "configured", "venv", "activated", "path", "fallback"]


@dataclass(frozen=True)
class ProjectEnvironment:
    """The command that runs `robotcode` for one root, and why it was chosen."""

    source: Source
    command_prefix: tuple[str, ...]
    interpreter: Path | None = None
    robotcode_version: str | None = None
    layered: bool = False
    warning: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def libraries_visible(self) -> bool:
        return self.source != "fallback"


def own_robotcode_version() -> str | None:
    """Return the robotcode version installed next to rf-mcp."""
    try:
        return importlib.metadata.version("robotcode")
    except importlib.metadata.PackageNotFoundError:
        return None


def own_robotcode_requirements(pin: str | None = None) -> list[str]:
    """`robotcode[all]==<pin>` plus every robotcode sub-package installed next to rf-mcp, pinned exactly.

    Naming each pre-release sub-package lets uv versions that do not treat a transitive exact pin as explicit
    (uv 0.9 refuses `robotcode-core==2.9.0.dev115` under `robotcode[all]==2.9.0.dev115`) resolve the layer without
    `--prerelease=allow`, which would let pre-releases of unrelated dependencies in.
    """
    pin = pin or own_robotcode_version()
    if not pin:
        return []
    subpackages = set()
    for dist in importlib.metadata.distributions():
        name = (dist.metadata.get("Name") or "").lower().replace("_", "-")
        if name.startswith("robotcode-") and dist.version == pin:
            subpackages.add(f"{name}=={dist.version}")
    return [f"robotcode[all]=={pin}", *sorted(subpackages)]


def own_robotcode_command() -> tuple[str, ...]:
    """Return the robotcode executable of rf-mcp's own environment (`python -m robotcode.cli` without one)."""
    for directory in (Path(sys.executable).parent, Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")):
        for name in ("robotcode.exe", "robotcode"):
            candidate = directory / name
            if candidate.is_file():
                return (str(candidate),)
    return (sys.executable, "-m", "robotcode.cli")


def venv_python(venv: Path) -> Path | None:
    """Return the interpreter of a virtual environment folder (Windows `Scripts` or POSIX `bin`)."""
    for candidate in (venv / "Scripts" / "python.exe", venv / "bin" / "python", venv / "bin" / "python3"):
        if candidate.is_file():
            return candidate
    return None


def _executable_next_to(interpreter: Path, name: str) -> Path | None:
    for candidate in (interpreter.parent / f"{name}.exe", interpreter.parent / name):
        if candidate.is_file():
            return candidate
    return None


def supported(version: str | None) -> bool:
    if not version:
        return False
    try:
        return Version(version) >= Version(MIN_ROBOTCODE)
    except InvalidVersion:
        return False


def probe_robotcode_version(interpreter: Path) -> str | None:
    """Return the robotcode version importable by `interpreter`, or None."""
    try:
        proc = subprocess.run(
            [str(interpreter), "-c", "import importlib.metadata as m; print(m.version('robotcode'))"],
            capture_output=True, text=True, timeout=VERSION_PROBE_TIMEOUT_SECONDS, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (proc.stdout.strip() or None) if proc.returncode == 0 else None


def probe_command(prefix: tuple[str, ...]) -> tuple[bool, str]:
    """Run `<prefix> --version` and return (ok, output). Warms uv's cache for a layered prefix."""
    try:
        proc = subprocess.run(
            [*prefix, "--version"], capture_output=True, text=True, timeout=LAYER_PROBE_TIMEOUT_SECONDS,
            stdin=subprocess.DEVNULL, env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, output


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


class EnvironmentResolver:
    """Resolve and cache the robotcode command for each project root."""

    def __init__(
        self,
        *,
        own_prefix: Path | None = None,
        uv: str | None | Literal[False] = None,
        pin: str | None = None,
        version_probe: Callable[[Path], str | None] = probe_robotcode_version,
        command_probe: Callable[[tuple[str, ...]], tuple[bool, str]] = probe_command,
        environ: dict[str, str] | None = None,
    ) -> None:
        self.own_prefix = (own_prefix or Path(sys.prefix)).resolve()
        self._uv = uv
        self.pin = pin or own_robotcode_version()
        self.requirements = [f"robotcode[all]=={pin}"] if pin else own_robotcode_requirements(self.pin)
        self._version_probe = version_probe
        self._command_probe = command_probe
        self._environ = environ
        self._cache: dict[Path, tuple[tuple, ProjectEnvironment]] = {}
        self.last_layer_seconds: float | None = None

    @property
    def uv(self) -> str | None:
        if self._uv is False:
            return None
        return self._uv or shutil.which("uv")

    def _env(self) -> dict[str, str]:
        return dict(os.environ) if self._environ is None else self._environ

    def is_own(self, interpreter_or_venv: Path) -> bool:
        path = interpreter_or_venv
        venv = path.parent.parent if path.is_file() else path
        return _same_path(venv, self.own_prefix)

    # --- cache --------------------------------------------------------------------------------

    def _fingerprint(self, root: Path, env: ProjectEnvironment | None = None) -> tuple:
        def stamp(p: Path) -> float | None:
            try:
                return p.stat().st_mtime
            except OSError:
                return None

        venvs = [root / ".venv"]
        if env is not None and env.interpreter is not None:
            venvs.append(env.interpreter.parent.parent)
        parts: list[object] = [stamp(root / CONFIG_FILE), self._env().get("VIRTUAL_ENV")]
        for venv in venvs:
            parts += [stamp(venv / "pyvenv.cfg"), *(stamp(sp) for sp in sorted(venv.glob("Lib/site-packages")))]
            parts += [stamp(sp) for sp in sorted(venv.glob("lib/python*/site-packages"))]
        if env is not None:
            parts.append(Path(env.command_prefix[0]).exists())
        return tuple(parts)

    def resolve(self, root: Path) -> ProjectEnvironment:
        root = root.resolve()
        cached = self._cache.get(root)
        if cached is not None and cached[0] == self._fingerprint(root, cached[1]):
            return cached[1]
        env = self._resolve(root)
        self._cache[root] = (self._fingerprint(root, env), env)
        return env

    # --- resolution ---------------------------------------------------------------------------

    def _resolve(self, root: Path) -> ProjectEnvironment:
        notes: list[str] = []

        configured = self._configured_interpreter(root, notes)
        if configured is not None:
            env = self._from_interpreter(configured, "configured", notes)
            if env is not None:
                return env

        venv = root / ".venv"
        python = venv_python(venv) if venv.is_dir() else None
        if python is not None:
            env = self._from_interpreter(python, "venv", notes)
            if env is not None:
                return env
        # The project has an environment of its own that could not run robotcode: landing on rf-mcp's own
        # environment below (through VIRTUAL_ENV or PATH, as under `uv run`) must not look like a normal run.
        project_env = configured is not None or python is not None

        active = self._env().get("VIRTUAL_ENV")
        if active:
            python = venv_python(Path(active))
            if python is not None:
                env = self._from_interpreter(python, "activated", notes)
                if env is not None:
                    return self._fallback(root, notes) if env.source == "own" and project_env else env

        on_path = shutil.which("robotcode", path=self._env().get("PATH"))
        if on_path:
            exe = Path(on_path)
            if self.is_own(exe):
                return self._fallback(root, notes) if project_env else self._own(notes)
            ok, output = self._command_probe((str(exe),))
            version = output.rsplit("version", 1)[-1].strip() if ok else None
            if supported(version):
                return ProjectEnvironment("path", (str(exe),), robotcode_version=version, notes=tuple(notes))
            notes.append(f"robotcode on PATH ({exe}) is {version or 'not runnable'}; rf-mcp needs {MIN_ROBOTCODE}+.")

        return self._fallback(root, notes)

    def _configured_interpreter(self, root: Path, notes: list[str]) -> Path | None:
        config = root / CONFIG_FILE
        if not config.is_file():
            return None
        try:
            raw = tomlkit.parse(config.read_text(encoding="utf-8")).unwrap().get("python")
        except Exception as exc:
            notes.append(f"{CONFIG_FILE.as_posix()} does not parse: {exc}")
            return None
        if not raw:
            return None
        path = Path(str(raw)).expanduser()
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            notes.append(f"{CONFIG_FILE.as_posix()} sets python = {raw!r}, which does not exist ({path}).")
            return None
        return path

    def _own(self, notes: list[str]) -> ProjectEnvironment:
        return ProjectEnvironment(
            "own", own_robotcode_command(), interpreter=Path(sys.executable), robotcode_version=own_robotcode_version(),
            notes=tuple(notes),
        )

    def _from_interpreter(self, python: Path, source: Source, notes: list[str]) -> ProjectEnvironment | None:
        if self.is_own(python):
            return self._own(notes)
        version = self._version_probe(python)
        if supported(version):
            exe = _executable_next_to(python, "robotcode")
            prefix: tuple[str, ...] = (str(exe),) if exe else (str(python), "-m", "robotcode.cli")
            return ProjectEnvironment(source, prefix, interpreter=python, robotcode_version=version, notes=tuple(notes))
        found = f"robotcode {version}" if version else "no robotcode"
        uv = self.uv
        if uv is None:
            notes.append(f"{python} has {found} and uv is not installed, so rf-mcp cannot add robotcode to it.")
            return None
        if not self.pin:
            notes.append("rf-mcp's own robotcode version is unknown, so it cannot be layered onto the project.")
            return None
        withs = [arg for req in self.requirements for arg in ("--with", req)]
        layered: tuple[str, ...] = (uv, "run", "--quiet", "--no-project", "--python", str(python), *withs, "robotcode")
        started = time.monotonic()
        ok, output = self._command_probe(layered)
        self.last_layer_seconds = time.monotonic() - started
        if not ok:
            notes.append(f"Adding robotcode {self.pin} to {python} with uv failed: {output[-600:]}")
            return None
        return ProjectEnvironment(
            source, layered, interpreter=python, robotcode_version=self.pin, layered=True,
            notes=(*notes, f"{python} has {found}; rf-mcp adds robotcode {self.pin} with uv."),
        )

    def _fallback(self, root: Path, notes: list[str]) -> ProjectEnvironment:
        if self.is_own(root / ".venv"):
            return self._own(notes)
        remedies = (
            "install uv (rf-mcp then adds robotcode to the project environment itself)"
            if self.uv is None
            else f'create the project environment ({root / ".venv"}) or set `python` in {CONFIG_FILE.as_posix()}'
        )
        warning = (
            f"No project environment was usable for {root}, so robotcode ran in rf-mcp's own environment and the "
            f"project's libraries are not visible. Fix: {remedies}, or add robotcode to the project environment "
            '(`uv add --dev "robotcode[all]>=2.9.0.dev115"`). Run `rf-mcp doctor` to check.'
        )
        if notes:
            warning += " Details: " + " ".join(notes)[:800]
        return ProjectEnvironment(
            "fallback", own_robotcode_command(), interpreter=Path(sys.executable),
            robotcode_version=own_robotcode_version(), warning=warning, notes=tuple(notes),
        )


def own_robotcode_available() -> bool:
    """True when rf-mcp's own environment holds a robotcode executable."""
    return len(own_robotcode_command()) == 1


def missing_robotcode_instruction(root: Path, env: ProjectEnvironment, resolver: EnvironmentResolver) -> str:
    """Name the fix for a robotcode that could not be started, for the way rf-mcp is installed."""
    add = 'uv add --dev "robotcode[all]>=2.9.0.dev115"'
    started = env.command_prefix[0]
    if resolver.is_own(root / ".venv"):
        fix = f"Add robotcode to the project environment (`{add}`)"
    elif resolver.uv is None:
        fix = f"Install uv so rf-mcp can add robotcode to the project environment itself, or run `{add}` in the project"
    else:
        fix = f"Check the project environment, or run `{add}` in the project"
    return f"`robotcode` could not be started for {root} ({started}). {fix}; run `rf-mcp doctor` to check."

"""Shared fixtures: a fake official skills package and a plugin repository for `--skills-dir`."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
import yaml

from robotmcp.skills import official
from robotmcp.skills.official import load_official_skills


def make_skill(skills_dir: Path, name: str, *references: str, body: str = "") -> Path:
    """Write `<skills_dir>/<name>/SKILL.md` and the given `references/*.md` files."""
    folder = skills_dir / name
    (folder / "references").mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} skill.\n---\n\n# {name}\nFrom {skills_dir.parent.name}. {body}\n",
        encoding="utf-8",
    )
    for ref in references:
        (folder / "references" / ref).write_text(f"# {ref}\n", encoding="utf-8")
    return folder


# The official plugins as robotframework-plugins ships them (trimmed to what the tests need).
OFFICIAL_PLUGINS: dict[str, dict[str, Any]] = {
    "rf-core": {"skills": {"rf-language": ()}, "libraries": []},
    "rf-web": {
        "skills": {"rf-browser": ("locators.md", "iframes-shadow-dom.md"), "rf-selenium": ("locators.md",)},
        "libraries": [
            {"name": "Browser", "package": "robotframework-browser", "versions": ">=20,<21", "skills": ["rf-browser"]},
            {"name": "SeleniumLibrary", "package": "robotframework-seleniumlibrary", "versions": ">=6.9,<7",
             "skills": ["rf-selenium"]},
        ],
    },
    "rf-api": {
        "skills": {"rf-requests": ("request-options.md",)},
        "libraries": [{"name": "RequestsLibrary", "package": "robotframework-requests", "versions": ">=0.9.7"}],
    },
}


def write_plugins(base: Path, plugins: dict[str, dict[str, Any]]) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    """Write `base/plugins/<p>/skills/*` and return the skill folders and catalogue entries."""
    dirs: dict[str, Path] = {}
    catalog: list[dict[str, Any]] = []
    for name, spec in plugins.items():
        skills_dir = base / "plugins" / name / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        for skill, refs in spec["skills"].items():
            make_skill(skills_dir, skill, *refs, body=spec.get("body", ""))
        dirs[name] = skills_dir
        catalog.append({"name": name, "version": "1.0.0", "status": "active", "libraries": spec["libraries"]})
    return dirs, catalog


@dataclass(frozen=True)
class FakeSnapshot:
    """`robotframework_plugins.bundle.Snapshot`: the plugins at one commit, in a folder with `catalog.json`."""

    ref: str
    commit: str
    path: Path
    #: None as from robotframework-plugins 2026.11.0.dev1, which didn't record it (upstream #23).
    catalog_format: Any = None

    def catalog(self) -> list[dict[str, Any]]:
        return list(json.loads((self.path / "catalog.json").read_text(encoding="utf-8"))["plugins"])

    def skills_dirs(self) -> dict[str, Path]:
        dirs = {e["name"]: self.path / "plugins" / e["name"] / "skills" for e in self.catalog()}
        return {name: path for name, path in dirs.items() if path.is_dir()}


class FakeFetchError(RuntimeError):
    """`robotframework_plugins.bundle.FetchError`."""


class FakeRemote:
    """What `bundle.fetch(ref)` finds on GitHub: `publish(ref, commit, plugins)` or `fail(ref)`; counts calls."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.refs: dict[str, tuple[str, dict[str, dict[str, Any]]]] = {}
        self.failing: set[str] = set()
        self.calls: list[str] = []
        #: What fetched snapshots report as `catalog_format`.
        self.snapshot_format: Any = None

    def publish(self, ref: str, commit: str, plugins: dict[str, dict[str, Any]] | None = None) -> str:
        self.refs[ref] = (commit, OFFICIAL_PLUGINS if plugins is None else plugins)
        self.failing.discard(ref)
        return commit

    def fail(self, ref: str) -> None:
        self.failing.add(ref)

    def fetch(self, ref: str = "stable", *, cache_dir: Any = None, timeout: float = 30.0) -> FakeSnapshot:
        self.calls.append(ref)
        if ref in self.failing:
            raise FakeFetchError(f"downloading {ref!r} failed: URLError: offline")
        found = self.refs.get(ref) or next(((c, p) for c, p in self.refs.values() if c == ref), None)
        if found is None:
            raise FakeFetchError(f"ref {ref!r} not found in MarketSquare/robotframework-plugins")
        commit, plugins = found
        folder = Path(cache_dir) if cache_dir is not None else self.base / "snapshots"
        folder = folder / commit
        if not (folder / "catalog.json").is_file():
            _, catalog = write_plugins(folder, plugins)
            (folder / "catalog.json").write_text(json.dumps({"plugins": catalog}), encoding="utf-8")
        return FakeSnapshot(ref=ref, commit=commit, path=folder.resolve(), catalog_format=self.snapshot_format)


def sha(n: int) -> str:
    """A full commit SHA for tests."""
    return f"{n:040x}"


def install_fake_bundle(
    monkeypatch: pytest.MonkeyPatch,
    base: Path,
    plugins: dict[str, dict[str, Any]] | None = None,
    catalog_format: Any = 1,
) -> dict[str, Path]:
    """Put a `robotframework_plugins.bundle` module over a tree under `base` into `sys.modules`.

    Its `fetch()` asks `bundle.remote` (a `FakeRemote`), which knows no ref until a test publishes one.
    """
    plugins = OFFICIAL_PLUGINS if plugins is None else plugins
    dirs, catalog = write_plugins(base, plugins)
    package = types.ModuleType("robotframework_plugins")
    package.__version__ = "2026.11.0"  # type: ignore[attr-defined]
    bundle = types.ModuleType("robotframework_plugins.bundle")
    bundle.CATALOG_FORMAT = catalog_format  # type: ignore[attr-defined]
    bundle.catalog = lambda: [dict(e) for e in catalog]  # type: ignore[attr-defined]
    bundle.skills_dirs = lambda: dict(dirs)  # type: ignore[attr-defined]
    bundle.remote = FakeRemote(base)  # type: ignore[attr-defined]
    bundle.fetch = bundle.remote.fetch  # type: ignore[attr-defined]
    bundle.Snapshot = FakeSnapshot  # type: ignore[attr-defined]
    bundle.FetchError = FakeFetchError  # type: ignore[attr-defined]
    package.bundle = bundle  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "robotframework_plugins", package)
    monkeypatch.setitem(sys.modules, "robotframework_plugins.bundle", bundle)
    load_official_skills.cache_clear()
    return dirs


@pytest.fixture(autouse=True)
def _no_official_skills(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Tests see no official package unless they install the fake one: the real one may be installed.

    rf-mcp's cache folder (the snapshot state) is a fresh temporary folder, never the user's.
    """
    monkeypatch.setitem(sys.modules, "robotframework_plugins", None)
    cache = tmp_path_factory.mktemp("cache")
    monkeypatch.setattr(official, "cache_base", lambda: cache)
    load_official_skills.cache_clear()
    yield
    load_official_skills.cache_clear()


@pytest.fixture
def official_cache() -> Path:
    """rf-mcp's cache folder for this test (see `_no_official_skills`)."""
    return official.cache_base()


@pytest.fixture
def remote(fake_bundle: dict[str, Path]) -> FakeRemote:
    """The fake GitHub behind the fake bundle's `fetch()`."""
    from robotframework_plugins import bundle

    return bundle.remote


@pytest.fixture
def fake_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """The official skills package, faked under `tmp_path/official`: rf-core, rf-web and rf-api."""
    return install_fake_bundle(monkeypatch, tmp_path / "official")


def make_plugin_repo(base: Path, plugins: dict[str, dict[str, Any]]) -> Path:
    """A plugin repository for `--skills-dir`: `plugins/<p>/catalog.yaml` plus `plugins/<p>/skills/*`."""
    for name, spec in plugins.items():
        plugin = base / "plugins" / name
        (plugin / "skills").mkdir(parents=True, exist_ok=True)
        for skill, refs in spec.get("skills", {}).items():
            make_skill(plugin / "skills", skill, *refs)
        manifest = {"name": name, "libraries": spec.get("libraries", [])}
        (plugin / "catalog.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return base


@pytest.fixture
def plugin_repo(tmp_path: Path) -> Callable[..., Path]:
    """Build a `--skills-dir` plugin repository: `plugin_repo({"acme": {"skills": {...}, "libraries": [...]}})`."""
    return lambda plugins, name="acme-skills": make_plugin_repo(tmp_path / name, plugins)

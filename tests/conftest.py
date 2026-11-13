"""Shared fixtures: a fake official skills package and a plugin repository for `--skills-dir`."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
import yaml

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


def install_fake_bundle(
    monkeypatch: pytest.MonkeyPatch,
    base: Path,
    plugins: dict[str, dict[str, Any]] | None = None,
    catalog_format: Any = 1,
) -> dict[str, Path]:
    """Put a `robotframework_plugins.bundle` module over a tree under `base` into `sys.modules`."""
    plugins = OFFICIAL_PLUGINS if plugins is None else plugins
    dirs: dict[str, Path] = {}
    catalog: list[dict[str, Any]] = []
    for name, spec in plugins.items():
        skills_dir = base / "plugins" / name / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        for skill, refs in spec["skills"].items():
            make_skill(skills_dir, skill, *refs)
        dirs[name] = skills_dir
        catalog.append({"name": name, "version": "1.0.0", "status": "active", "libraries": spec["libraries"]})
    package = types.ModuleType("robotframework_plugins")
    package.__version__ = "2026.11.0"  # type: ignore[attr-defined]
    bundle = types.ModuleType("robotframework_plugins.bundle")
    bundle.CATALOG_FORMAT = catalog_format  # type: ignore[attr-defined]
    bundle.catalog = lambda: [dict(e) for e in catalog]  # type: ignore[attr-defined]
    bundle.skills_dirs = lambda: dict(dirs)  # type: ignore[attr-defined]
    package.bundle = bundle  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "robotframework_plugins", package)
    monkeypatch.setitem(sys.modules, "robotframework_plugins.bundle", bundle)
    load_official_skills.cache_clear()
    return dirs


@pytest.fixture(autouse=True)
def _no_official_skills(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Tests see no official package unless they install the fake one: the real one may be installed."""
    monkeypatch.setitem(sys.modules, "robotframework_plugins", None)
    load_official_skills.cache_clear()
    yield
    load_official_skills.cache_clear()


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

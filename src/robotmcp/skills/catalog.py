"""Skill catalogues: which skills cover which Robot Framework libraries, matched against a project's packages.

The catalogue shape is `robotframework-plugins`' `catalog.yaml`: per plugin, `libraries: [{name, package,
versions}]`, optionally with `skills` per library. Three sources are read, in precedence order: plugin
repositories in `--skills-dir` (`plugins/*/catalog.yaml`), the official package, and rf-mcp's built-in catalogue.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import yaml  # type: ignore[import-untyped]
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from robotmcp.config import ServerConfig
from robotmcp.skills.official import load_official_skills

logger = logging.getLogger(__name__)

Layer = Literal["skills_dir", "official", "builtin"]
LAYER_ORDER: dict[Layer, int] = {"skills_dir": 0, "official": 1, "builtin": 2}

BUILTIN_CATALOG = Path(__file__).resolve().parent / "builtin" / "catalog.yaml"


@dataclass(frozen=True)
class LibraryRef:
    name: str
    package: str
    versions: str = ""
    skills: tuple[str, ...] = ()


@dataclass(frozen=True)
class CatalogEntry:
    plugin: str
    layer: Layer
    skills: tuple[str, ...]
    libraries: tuple[LibraryRef, ...]


@dataclass(frozen=True)
class Match:
    library: str
    installed: str
    skill: str
    supported: bool
    versions: str = ""

    @property
    def uri(self) -> str:
        return f"skill://{self.skill}/SKILL.md"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "library": self.library, "installed": self.installed, "skill": self.uri, "supported": self.supported,
        }
        if not self.supported:
            out["versions"] = self.versions
        return out


def _skill_names(skills_dir: Path) -> tuple[str, ...]:
    if not skills_dir.is_dir():
        return ()
    return tuple(sorted(d.name for d in skills_dir.iterdir() if (d / "SKILL.md").is_file()))


def _libraries(raw: Any, source: str) -> tuple[LibraryRef, ...]:
    out: list[LibraryRef] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or not item.get("package"):
            logger.warning("Skipping a library without a package in %s: %r", source, item)
            continue
        skills = item.get("skills") or ()
        out.append(LibraryRef(
            name=str(item.get("name") or item["package"]),
            package=str(item["package"]),
            versions=str(item.get("versions") or ""),
            skills=tuple(str(s) for s in skills) if isinstance(skills, list) else (),
        ))
    return tuple(out)


def entry_from_dict(data: Mapping[str, Any], layer: Layer, skills_dir: Path, source: str) -> CatalogEntry:
    """One plugin's catalogue (`catalog.yaml` / a `bundle.catalog()` item) with the skills in `skills_dir`."""
    return CatalogEntry(
        plugin=str(data.get("name") or skills_dir.parent.name),
        layer=layer,
        skills=_skill_names(skills_dir),
        libraries=_libraries(data.get("libraries"), source),
    )


def _read_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Skipping the skill catalogue %s: %s", path, exc)
        return None


def skills_dir_entries(skills_dirs: Iterable[Path]) -> list[CatalogEntry]:
    """Catalogues of plugin repositories (`plugins/*/catalog.yaml`) in the `--skills-dir` folders."""
    entries: list[CatalogEntry] = []
    for base in skills_dirs:
        plugins = Path(base).expanduser() / "plugins"
        if not plugins.is_dir():
            continue
        for manifest in sorted(plugins.glob("*/catalog.yaml")):
            data = _read_yaml(manifest)
            if isinstance(data, dict) and not data.get("bundle"):
                entries.append(entry_from_dict(data, "skills_dir", manifest.parent / "skills", str(manifest)))
            elif data is not None and not isinstance(data, dict):
                logger.warning("Skipping the skill catalogue %s: not a mapping", manifest)
    return entries


def official_entries() -> list[CatalogEntry]:
    """Catalogues of the official package's plugins that ship skills (bundles such as rf-agentskills excluded)."""
    official = load_official_skills()
    entries: list[CatalogEntry] = []
    for data in official.catalog:
        name = str(data.get("name", ""))
        if data.get("bundle") or name not in official.skills_dirs:
            continue
        entries.append(entry_from_dict(data, "official", official.skills_dirs[name], f"robotframework-plugins {name}"))
    return entries


def builtin_entries(path: Path = BUILTIN_CATALOG) -> list[CatalogEntry]:
    data = _read_yaml(path) if path.is_file() else None
    plugins = data.get("plugins", []) if isinstance(data, dict) else []
    return [entry_from_dict(p, "builtin", path.parent, str(path)) for p in plugins if isinstance(p, dict)]


def catalog_entries(config: ServerConfig) -> list[CatalogEntry]:
    """Every catalogue, in precedence order: `--skills-dir`, official, built-in."""
    return [*skills_dir_entries(config.skills_dirs), *official_entries(), *builtin_entries()]


def _supported(installed: str, versions: str) -> bool:
    if not versions.strip():
        return True
    try:
        return SpecifierSet(versions).contains(Version(installed), prereleases=True)
    except (InvalidSpecifier, InvalidVersion):
        return True  # cannot tell: do not warn


def match(
    inventory: Mapping[str, str],
    entries: Iterable[CatalogEntry],
    served: Iterable[str],
) -> list[Match]:
    """Skills for the installed packages: one match per library and served skill.

    `inventory` maps distribution names to versions. A package matched by an earlier layer is not matched again by
    a later one, so the built-in catalogue only fills in what the skills folders and the official package leave out.
    """
    installed = {canonicalize_name(name): version for name, version in inventory.items()}
    served_names = set(served)
    taken: set[str] = set()
    out: list[Match] = []
    for entry in sorted(entries, key=lambda e: LAYER_ORDER[e.layer]):
        matched_here: set[str] = set()
        for lib in entry.libraries:
            package = canonicalize_name(lib.package)
            version = installed.get(package)
            if version is None or package in taken:
                continue
            for skill in lib.skills or _skills_named_for(lib.name, entry.skills):
                if skill not in served_names or any(m.skill == skill and m.library == lib.name for m in out):
                    continue
                out.append(Match(lib.name, version, skill, _supported(version, lib.versions), lib.versions))
                matched_here.add(package)
        taken |= matched_here
    return _first_skill_per_library_first(out)


def _skills_named_for(library: str, skills: tuple[str, ...]) -> tuple[str, ...]:
    """Without a skill list for the library: the plugin's skills named after it, else all of them.

    `rf-web` ships `rf-browser` and `rf-selenium`; Browser gets `rf-browser`, SeleniumLibrary `rf-selenium`.
    """
    stem = library.lower().removesuffix("library") or library.lower()
    named = tuple(s for s in skills if stem in s.lower())
    return named or skills


def _first_skill_per_library_first(matches: list[Match]) -> list[Match]:
    """Keep the order, but put each library's first skill before any library's second one."""
    seen: set[str] = set()
    first: list[Match] = []
    rest: list[Match] = []
    for m in matches:
        (rest if m.library in seen else first).append(m)
        seen.add(m.library)
    return first + rest

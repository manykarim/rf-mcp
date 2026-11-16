"""The official Robot Framework skills bundled in the `robotframework-plugins` package.

rf-mcp reads them only through the package's public `robotframework_plugins.bundle` API
(MarketSquare/robotframework-plugins#5): `CATALOG_FORMAT`, `catalog()` and `skills_dirs()`.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = frozenset({1})


@dataclass(frozen=True)
class OfficialSkills:
    """What the installed `robotframework-plugins` offers; empty when it is missing or unusable."""

    version: str | None = None
    catalog_format: Any = None
    skills_dirs: dict[str, Path] = field(default_factory=dict)
    catalog: list[dict[str, Any]] = field(default_factory=list)
    warning: str | None = None

    @property
    def installed(self) -> bool:
        return self.version is not None or self.catalog_format is not None

    def skill_count(self) -> int:
        return sum(1 for d in self.skills_dirs.values() for s in d.iterdir() if (s / "SKILL.md").is_file())


@lru_cache(maxsize=1)
def load_official_skills() -> OfficialSkills:
    """Import the bundle once; a missing package or an unsupported catalogue format gives no official skills."""
    try:
        from robotframework_plugins import bundle  # type: ignore[import-not-found]
    except ImportError:
        return OfficialSkills()
    version = getattr(sys.modules.get("robotframework_plugins"), "__version__", None)
    fmt = getattr(bundle, "CATALOG_FORMAT", None)
    if fmt not in SUPPORTED_FORMATS:
        warning = (
            f"robotframework-plugins {version or '?'} has catalogue format {fmt!r}; rf-mcp supports "
            f"{', '.join(str(f) for f in sorted(SUPPORTED_FORMATS))}. Its skills are not served."
        )
        logger.warning(warning)
        return OfficialSkills(version, fmt, warning=warning)
    try:
        dirs = {str(name): Path(path).resolve() for name, path in bundle.skills_dirs().items()}
        catalog = [dict(entry) for entry in bundle.catalog()]
    except Exception as exc:  # a broken install must not take the server down
        warning = f"robotframework-plugins {version or '?'} could not be read: {exc}. Its skills are not served."
        logger.warning(warning)
        return OfficialSkills(version, fmt, warning=warning)
    return OfficialSkills(version, fmt, {n: p for n, p in dirs.items() if p.is_dir()}, catalog)

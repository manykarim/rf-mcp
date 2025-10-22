"""Helpers for optional dependency test permutations."""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pytest

# Mapping of optional extras to their underlying Robot Framework libraries.
EXTRA_LIBRARIES: Dict[str, Set[str]] = {
    "slim": set(),
    "web": {"Browser", "SeleniumLibrary"},
    "api": {"RequestsLibrary"},
    "mobile": {"AppiumLibrary"},
    "database": {"DatabaseLibrary"},
}

# Canonical combinations exercised by the smoke suite and CI.
EXTRA_COMBINATIONS: Dict[str, Tuple[str, ...]] = {
    "slim": ("slim",),
    "web": ("web",),
    "api": ("api",),
    "mobile": ("mobile",),
    "database": ("database",),
    "web+api": ("web", "api"),
    "web+mobile": ("web", "mobile"),
    "api+database": ("api", "database"),
    "all": ("web", "api", "mobile", "database"),
}


@lru_cache(maxsize=None)
def _is_library_available(library: str) -> bool:
    """Return True if the given Robot Framework library can be imported."""
    try:
        importlib.import_module(library)
        return True
    except ImportError:
        return False


def _flatten_extras(extras: Iterable[str]) -> Set[str]:
    """Expand extras to the set of Robot Framework libraries they require."""
    resolved: Set[str] = set()
    for extra in extras:
        libs = EXTRA_LIBRARIES.get(extra)
        if libs is None:
            raise KeyError(f"Unknown extra '{extra}'")
        resolved.update(libs)
    return resolved


def extras_available(extras: Sequence[str]) -> bool:
    """Return True if all libraries for the provided extras are installed."""
    required_libraries = _flatten_extras(extras)
    return all(_is_library_available(library) for library in required_libraries)


def missing_libraries(extras: Sequence[str]) -> List[str]:
    """Return the list of libraries that are missing for the extras."""
    return [library for library in sorted(_flatten_extras(extras)) if not _is_library_available(library)]


def combination_available(combination: str) -> bool:
    """Return True if every extra in the named combination is installed."""
    extras = EXTRA_COMBINATIONS.get(combination)
    if extras is None:
        raise KeyError(f"Unknown combination '{combination}'")
    return extras_available(extras)


def requires_extras(*extras: str, reason: str | None = None) -> pytest.MarkDecorator:
    """Skip a test when any of the requested extras is missing."""

    extras = tuple(extras)
    available = extras_available(extras)
    missing = missing_libraries(extras) if extras else []
    skip_reason = reason or (
        "Missing optional dependencies: " + ", ".join(missing)
        if missing
        else "Optional dependency check failed"
    )

    return pytest.mark.skipif(not available, reason=skip_reason)


def requires_combination(name: str, reason: str | None = None) -> pytest.MarkDecorator:
    """Skip a test when the named combination is not satisfied."""

    extras = EXTRA_COMBINATIONS.get(name)
    if extras is None:
        raise KeyError(f"Unknown combination '{name}'")
    available = extras_available(extras)
    missing = missing_libraries(extras)
    skip_reason = reason or (
        f"Missing optional dependencies for combination '{name}': " + ", ".join(missing)
        if missing
        else f"Optional dependency combination '{name}' unavailable"
    )
    return pytest.mark.skipif(not available, reason=skip_reason)


def install_matrix() -> Dict[str, bool]:
    """Report availability for each combination in EXTRA_COMBINATIONS."""
    return {name: combination_available(name) for name in EXTRA_COMBINATIONS}


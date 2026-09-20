"""Offline guards for the `[all]` extra's resolvability (change: install-extra-resolvability).

rf-mcp 0.34.0/0.35.0 shipped with `rf-mcp[desktop]` folded into `[all]`, which made
`uv tool install "rf-mcp[all]"` fail for every user on every OS: uv grants its
pre-release allowance only to FIRST-PARTY requirements, so rf-mcp's pinned PlatynUI
pre-release was refused once it became a transitive dependency of a consumer.

Nothing in CI could see it — every job installs rf-mcp as a local path (`-e .`) or from
the lock file, both of which keep the pins first-party. These tests are the cheap,
offline, always-on half of the guard (the `uv tool install` half lives in
tests/unit/test_tool_install_packaging_smoke.py and the CI installability job).

They deliberately assert the GENERAL property — "`[all]` contains nothing that only
exists as a pre-release" — not just "no PlatynUI", so the next pre-release dependency
someone adds to `[all]` is caught too.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

# `tomllib` is stdlib only from 3.11, and rf-mcp supports 3.10 (requires-python
# >=3.10). tomlkit is a hard rf-mcp dependency, so it is always available.
try:  # Python 3.11+
    from tomllib import loads as _toml_loads
except ModuleNotFoundError:  # Python 3.10
    from tomlkit import parse as _toml_loads

REPO = Path(__file__).resolve().parents[2]
PYPROJECT = REPO / "pyproject.toml"


def _extras() -> dict[str, list[str]]:
    data = _toml_loads(PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def test_all_extra_has_no_platynui_dependency():
    """`[all]` must not pull PlatynUI, directly or via the `rf-mcp[desktop]` self-extra.

    PlatynUI is published only as a pre-release AND only for a subset of platforms
    (no sdist), so any reference here re-breaks `[all]` for every uv/pipx user and for
    every macOS-Intel / musl / glibc<2.34 user.
    """
    offenders = [
        req for req in _extras()["all"]
        if "platynui" in req.lower() or "rf-mcp[" in req.lower().replace(" ", "")
    ]
    assert not offenders, (
        "The `all` extra must not reference PlatynUI or the desktop self-extra; "
        f"found {offenders}. Desktop is an explicit opt-in — see "
        "openspec/changes/install-extra-resolvability."
    )


def test_all_extra_admits_a_stable_release_for_every_requirement():
    """Every requirement in `[all]` must allow at least one non-pre-release version.

    A specifier that admits only pre-releases (e.g. `==1.0.0.dev2`, or `>=1.0.0.dev0`
    when no stable release exists in range) makes the whole extra unresolvable under
    uv's default policy once rf-mcp is a third-party dependency.
    """
    pre_only = []
    for raw in _extras()["all"]:
        spec = Requirement(raw).specifier
        pinned_pre = [
            s for s in spec
            if s.operator in ("==", "===") and Version(s.version).is_prerelease
        ]
        lower_pre = [
            s for s in spec
            if s.operator in (">=", ">") and Version(s.version).is_prerelease
        ]
        if pinned_pre or lower_pre:
            pre_only.append(raw)

    assert not pre_only, (
        "These `all` requirements are pinned to / bounded below by a pre-release, which "
        f"makes `rf-mcp[all]` unresolvable for uv and pipx users: {pre_only}. "
        "Move them to a dedicated opt-in extra instead."
    )


def test_desktop_extra_still_declares_platynui():
    """Guard the other direction: the fix must not have deleted desktop support."""
    desktop = " ".join(_extras()["desktop"]).lower()
    assert "robotframework-platynui" in desktop
    assert "platynui-cli" in desktop


@pytest.mark.parametrize("extra", ["web", "api", "mobile", "database", "frontend",
                                   "memory", "tokens", "all"])
def test_generally_installable_extras_are_pre_release_free(extra):
    """The same property for every extra advertised as installable without a flag.

    `desktop` and `semantic` are intentionally excluded: `desktop` is the documented
    pre-release opt-in, and `semantic` is a heavyweight optional backend.
    """
    offenders = [
        raw for raw in _extras()[extra]
        if any(
            s.operator in ("==", "===", ">=", ">") and Version(s.version).is_prerelease
            for s in Requirement(raw).specifier
        )
    ]
    assert not offenders, f"extra `{extra}` carries a pre-release requirement: {offenders}"

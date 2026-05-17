"""build_test_suite must be in every execution profile so the documented
end-of-workflow step (`init -> execute_step* -> build_test_suite`) works
without manual profile escalation.

Pre-v0.32 only the `full` profile included build_test_suite, so a session
auto-bound to `browser_exec` / `api_exec` / `minimal_exec` / `desktop_exec`
(e.g. because of a small-context model hint) would fail at the final
build step with "Unknown tool: build_test_suite". Closing that surprise.

`discovery` (planning-only) and `slim_exec` (4-tool ultra-slim 7B
profile, ADR-016) intentionally omit build_test_suite — the test pins
which profiles include it and which don't.
"""

from __future__ import annotations

import pytest

from robotmcp.domains.tool_profile.aggregates import ProfilePresets
from robotmcp.domains.tool_profile.value_objects import ToolTag


@pytest.mark.parametrize("preset_name", [
    "browser_exec", "api_exec", "minimal_exec", "desktop_exec",
])
def test_execution_profile_includes_build_test_suite(preset_name: str):
    """Every execution profile must include build_test_suite (v0.32)."""
    p = getattr(ProfilePresets, preset_name)()
    assert "build_test_suite" in p.tool_names, (
        f"{preset_name} must include build_test_suite for end-of-workflow "
        f"suite generation; got tool_names={sorted(p.tool_names)}"
    )


@pytest.mark.parametrize("preset_name", [
    "browser_exec", "api_exec", "minimal_exec", "desktop_exec",
])
def test_execution_profile_has_reporting_tag(preset_name: str):
    """ToolTag.REPORTING surfaces the suite-generation capability for tag-
    based profile filtering."""
    p = getattr(ProfilePresets, preset_name)()
    assert ToolTag.REPORTING in p.tags, (
        f"{preset_name} must carry ToolTag.REPORTING when build_test_suite "
        f"is in the tool set"
    )


@pytest.mark.parametrize("preset_name", ["discovery", "slim_exec"])
def test_non_execution_profile_excludes_build_test_suite(preset_name: str):
    """discovery is planning-only (no execution); slim_exec is the
    intentional 4-tool ultra-slim 7B profile. Both correctly OMIT
    build_test_suite — pin this so future changes don't accidentally
    add it back."""
    p = getattr(ProfilePresets, preset_name)()
    assert "build_test_suite" not in p.tool_names, (
        f"{preset_name} should NOT include build_test_suite "
        f"({preset_name} is planning-only / ultra-slim)"
    )


def test_full_profile_includes_build_test_suite():
    """Sanity: the full profile keeps build_test_suite (it always had it)."""
    p = ProfilePresets.full()
    assert "build_test_suite" in p.tool_names

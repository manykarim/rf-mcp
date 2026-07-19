## Why

The FastMCP 3.x upgrade (`fastmcp3-failed-step-warning`) regressed the tool-profile
feature (ADR-006/016) — the mechanism that shrinks the exposed MCP tool set for
small-context models. CI caught it on all 9 OS×Python combos with a single failing
integration test:

```
test_tools_reduced_after_small_profile:  assert 19 < 19  (no reduction)
WARNING fastmcp_adapter.py:114  No original tool found for 'manage_session', cannot add
```

Root cause, reproduced locally on fastmcp 3.4.4: `ToolManagerCompat.get_tools()`
returns an **empty dict** on v3. FastMCP 3.x has no `_tool_manager` attribute and
no `mcp.get_tools()` method, so the compat falls through both branches to
`return {}`. With no tools enumerated, `ToolManagerAdapter.initialize()` snapshots
zero tools into `_original_tools`, so switching to a `small_context` profile can
neither disable non-profile tools nor re-enable profile ones — the tool count
never changes and small-context models still see the full surface.

The hide/show primitives already work on v3: `mcp.disable(names={…})` /
`mcp.enable(names={…})` correctly drop and restore tools from `list_tools()`
(verified 19→18→19), and the compat's `remove_tool`/`add_tool` already fall through
to them. The only broken link is enumeration.

This regression was missed because the change's verification gate ran `tests/unit/`
only; the failing test lives in `tests/integration/`.

## What Changes

- **Fix `ToolManagerCompat.get_tools()` on v3**: enumerate via `await
  server.list_tools()` (the v3-native async enumerator) and return a
  `{name: tool}` dict, instead of falling through to an empty result. This
  repopulates `_original_tools` so the disable/enable-based profile switch works.
- Confirm `remove_tool`/`add_tool` map to `disable`/`enable` on v3 (they already
  fall through correctly now that `_tool_manager` is absent) and tidy the dead
  `_tool_manager` probes for v3.
- **Degrade the profile machinery to name-based enable/disable on v3.** Turning
  enumeration back on exposed a second bug: the v2 clone-and-re-register path plus
  `swap_tool_description`'s disable→re-enable churn corrupt FastMCP 3.x's
  tool-provider chain into unbounded `list_tools()` recursion under repeated
  profile switches (`maximum recursion depth exceeded` — one poisoned test cascades
  to 145 failures). On v3: `add_tool_with_description` re-enables by name only (no
  clone), `swap_tool_description` is a no-op (descriptions are immutable per profile
  there), and `restore_all` bulk-enables by name. Per-profile descriptions remain a
  documented non-goal on v3; the count reduction is intact.
- **Verification gate**: run `tests/integration/` (not just `tests/unit/`) for this
  change, and re-confirm `test_tools_reduced_after_small_profile` passes on v3.

Non-goals: per-profile tool *description* customization on v3 (FastMCP 3.x has no
add-with-new-description via enable; tools keep their original description when
re-enabled) — the token-saving *count* reduction is what matters and is restored.
Not touching MCP tool docstrings or server instructions. Not addressing the
pre-existing macOS benchmark flake (separate follow-up).

## Capabilities

### New Capabilities
- `tool-profile-fastmcp3`: tool-profile activation (the small-context tool-set
  reduction) works on FastMCP 3.x — enumeration via the v3 async `list_tools()` and
  hide/show via native `enable`/`disable`, not the v2 remove/add snapshot that
  silently no-ops on v3.

### Modified Capabilities
<!-- none -->

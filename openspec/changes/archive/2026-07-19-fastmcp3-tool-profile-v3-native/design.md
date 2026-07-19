## Context

Empirically established on fastmcp 3.4.4 (local repro of the CI failure):
- `mcp._tool_manager` — **absent** on v3 (was the v2 tool registry).
- `mcp.get_tools()` — **does not exist** on v3.
- `await mcp.list_tools()` — the v3 async enumerator; returns the 19 `mcp.types.Tool`
  objects (`.name`, `.description`, `.inputSchema`).
- `mcp.disable(names={n})` / `mcp.enable(names={n})` — work; a disabled tool drops
  out of `list_tools()` (19→18) and re-enabling restores it.
- `ToolManagerCompat.get_tools()` v3 path: `hasattr(server,"get_tools")` is False,
  `hasattr(server,"_tool_manager")` is False → falls through to `return {}`. This
  empty result is the whole bug: `ToolManagerAdapter.initialize()` snapshots 0
  tools, so the profile switch can neither disable nor re-enable anything.

## Goals / Non-Goals

**Goals:** the `small_context` profile reduces the exposed tool count on FastMCP
3.x; `test_tools_reduced_after_small_profile` passes; the fix is contained to the
compat enumeration.

**Non-Goals:** per-profile tool *description* rewriting on v3 (no FastMCP 3.x path
to enable-with-a-new-description); restoring exact v2 behavior; the macOS benchmark
flake.

## Decisions

1. **Enumerate via `list_tools()` on v3.** Add a v3 branch to
   `ToolManagerCompat.get_tools()`: if the server exposes an (async) `list_tools`,
   await it and return `{t.name: t for t in tools}`. Order the checks so the working
   path wins: existing-`get_tools` → `list_tools` → `_tool_manager` → `{}`. This
   repopulates `_original_tools` and everything downstream (`has_tool`,
   `get_visible_tool_names`, the profile switch) starts working.

2. **Keep remove/add mapping to disable/enable.** `remove_tool` and `add_tool`
   already fall through to `server.disable(names=…)` / `server.enable(names=…)` on
   v3 now that `_tool_manager` is absent. Tidy the dead `_tool_manager` probe so the
   intent is explicit, but the behavior is already correct.

3. **Description-swap degrades to NAME-BASED enable/disable on v3 — no churn.**
   Enumeration alone was not enough: activating it exposed a second, worse bug.
   The v2 adapter re-adds cloned `FunctionTool` objects (`copy.copy` + register).
   On v3 that path clones `mcp.types.Tool` protocol objects and re-registers via
   `enable`, and — critically — `swap_tool_description` does `disable` then
   `enable` on every description-mode change. Repeated across profile switches
   (`browser_exec`/`discovery`/`minimal_exec` → `full`, each a different
   `description_mode`) this churn corrupts FastMCP 3.x's tool-**provider** chain
   (`Provider.get_tool`) into unbounded recursion — `list_tools()` and every
   subsequent tool call then fail with `maximum recursion depth exceeded`
   (reproduced: 145 failures once one test poisons the shared server process).

   Fix: on v3 the adapter toggles visibility purely by NAME.
   - `add_tool_with_description` → `enable_tool(name)` only; never clones. Works
     even for tools with no snapshot/descriptor (e.g. `execute_batch`).
   - `swap_tool_description` → **no-op** (v3 descriptions are immutable per profile
     and the tool is already visible — nothing to do, and disable+enable is the
     exact churn that recurses).
   - `restore_all` → one bulk `enable(names={all})`, no remove-then-add.
   The mode-specific *description* is still not applied on v3 (documented
   non-goal); the count reduction — the token-saving win — is intact.

4. **Verify with integration tests.** Run `tests/integration/` for this change (the
   gap that let the regression through), asserting the profile-switch test passes,
   plus the `test_fastmcp_compat` suite.

## Risks / Trade-offs

- **`list_tools()` returns protocol Tool objects, not FastMCP FunctionTools.** The
  adapter only needs names (for enable/disable) and description (for the snapshot);
  `mcp.types.Tool` has both. `getattr(tool, "enabled", True)` defaults True (no such
  attr on the protocol object), so `initialize()` snapshots all currently-listed
  tools — correct, since it runs before any disabling.
- **Snapshot timing.** `initialize()` must run while all tools are enabled (it does
  — at adapter construction, before any profile activation). If a profile were
  active at snapshot time, disabled tools would be missed; not the case in the flow.
- **v2 path untouched.** The v2 branch (`_tool_manager.get_tools()`) is unchanged,
  so a 2.x runtime (should one occur) behaves as before.

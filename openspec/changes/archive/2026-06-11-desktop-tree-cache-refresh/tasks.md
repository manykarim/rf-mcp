## 1. Reproduction

- [x] 1.1 Confirm in code: `get_ui_tree` clears the runtime cache
  (`runtime.clear_cache()`) but PlatynUI keyword executions (`Query`/`Get
  Attribute`/locator resolution) do not — so a launch followed by name queries
  hits a stale snapshot
- [x] 1.2 Capture the live-run evidence (second GNOME Calculator run): name
  locators unresolved after launch, coordinate-click fallback, 0 recorded
  assertions in the built suite

## 2. Shared cache-clear helper (D3)

- [x] 2.1 Factor the `getattr(runtime, "clear_cache", None)` best-effort clear
  into one helper (`clear_runtime_tree_cache`) reused by `get_ui_tree`, D1, D2
- [x] 2.2 Unit test the helper with a mock runtime (clears when available;
  no-op/guarded when absent or raising)

## 3. Clear on desktop launch (D1)

- [x] 3.1 In the post-success desktop block of `_execute_keyword`, after a
  desktop GUI `Start Process`/`Run Process` succeeds, call the helper and set a
  session `desktop_tree_dirty` flag (best-effort, desktop-gated, never fails)
- [x] 3.2 Add the `desktop_tree_dirty` flag to the session model
- [x] 3.3 Unit tests: desktop launch clears the cache + sets the flag;
  non-desktop launch and non-launch desktop keywords do not

## 4. Refresh before first post-launch query (D2)

- [x] 4.1 Before a desktop tree-resolving keyword (`Query`/`Get Attribute`, and
  interaction-keyword locator resolution), if `desktop_tree_dirty` is set, call
  the helper and clear the flag
- [x] 4.2 Unit tests: first post-launch `Query`/`Get Attribute` refreshes and
  clears the flag; steady-state queries (flag clear) do NOT re-snapshot;
  non-desktop sessions untouched

## 5. Guidance (D4)

- [x] 5.1 PlatynUI guidance: the desktop tree is cached; the keyword path
  auto-refreshes after a launch; `get_session_state(sections=['ui_tree'])`
  forces a refresh when a locator still does not resolve (recover instead of
  coordinate clicks/OCR)
- [x] 5.2 Unit test: guidance documents the refresh recovery path

## 6. Validation + docs

- [x] 6.1 Full unit suite green; confirm web/api/mobile + existing desktop flows
  unaffected; steady-state desktop query cost unchanged
- [x] 6.2 ADR with the second-run evidence (unresolved name locators →
  coordinate clicks → assertion-less suite) and the cache-refresh fix
- [x] 6.3 Release note

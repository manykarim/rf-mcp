## Context

Source: a second live run where an external Codex agent drove robotmcp against
GNOME Calculator on an isolated Xvfb display. It confirmed the `desktop-stepwise-
followups` fixes (unqualified `Start Process` worked; guidance led with history
Labels), but the PlatynUI accessibility tree was largely unresolvable by name
this run — `//app:*[@Name='gnome-calculator']` had no usable root, button and
history-Label name queries returned empty / `ElementNotFoundError`. The agent
fell back to coordinate clicks + OCR and produced a **clicks-only suite with
zero recorded assertions**.

Reproduced in code:
- `ui_tree_service.get_ui_tree` calls `runtime.clear_cache()` before evaluating
  (it always sees live AT-SPI).
- The e2e `_wait_for_app` helper clears the cache every round, with the comment
  "the runtime caches the desktop tree … newly launched applications never
  become visible to this long-lived probe."
- PlatynUI KEYWORD executions (`Query`, `Get Attribute`, locator resolution for
  `Pointer Click`/keyboard) run against the shared runtime's CACHED tree and
  never clear it.

So a desktop `Start Process` followed by name queries resolves against a stale
snapshot taken before the app registered in AT-SPI. This is the root cause of
the run-to-run flakiness and the assertion-less suite.

## Goals / Non-Goals

**Goals:**
- After launching the AUT, desktop keyword queries see the freshly-launched app
  (name-based locators/assertions resolve reliably).
- Do this without re-snapshotting the whole desktop tree on every desktop
  keyword (bounded cost).
- Give agents a documented recovery path when a locator still does not resolve.

**Non-Goals:**
- Fixing AT-SPI registration latency itself (upstream); we re-read live state
  rather than serve a stale cache, but the app must still register.
- Changing web/api/mobile sessions or the `get_ui_tree` path (which already
  refreshes).
- Eliminating the coordinate-click fallback (it remains a valid last resort).

## Decisions

### D1: Clear the runtime tree cache after a desktop GUI launch
In the post-success desktop block of `_execute_keyword` (where the launch-
liveness hint already runs for `is_launch_keyword`), after a desktop GUI
`Start Process`/`Run Process` succeeds, call a shared
`clear_runtime_tree_cache()` helper and set a session flag
`desktop_tree_dirty = True`. Best-effort (guarded; never fails the step),
desktop-gated. This is the exact reproduced trigger ("app started after the
snapshot is invisible").

### D2: Refresh before the first post-launch tree-resolving keyword
Because the app may not be registered in AT-SPI at the instant the launch
returns, clearing only at launch can re-cache an app-less tree. So ALSO refresh
on the read side: before a desktop tree-resolving keyword (`Query` /
`Get Attribute`, and locator resolution for interaction keywords) runs, if
`session.desktop_tree_dirty` is set, call `clear_runtime_tree_cache()` and clear
the flag. This guarantees the first query AFTER the agent's own settle/wait sees
live state, while steady-state queries (flag already cleared) use the warm cache
— no per-call re-snapshot. Mirrors the proven e2e/`get_ui_tree` pattern without
its per-round cost.

### D3: A shared clear_runtime_tree_cache helper
Factor the `getattr(runtime, "clear_cache", None)` dance (currently inline in
`ui_tree_service`) into one best-effort helper (e.g. in `platynui_plugin` or
`ui_tree_service`) used by `get_ui_tree`, D1, and D2, so the cache-clear logic
lives in one place and is unit-testable via a mock runtime.

### D4: Guidance documents the refresh path
Add to the PlatynUI guidance: the desktop tree is cached; the keyword path
auto-refreshes after a launch; and `get_session_state(sections=['ui_tree'])`
forces a refresh — so an agent whose name locator does not resolve right after
launch refreshes and retries instead of dropping to coordinate clicks/OCR.

## Risks / Trade-offs

- **Refreshing re-snapshots the desktop tree (latency)** → Gate strictly: clear
  only after a launch and before the FIRST post-launch resolving keyword (flag
  cleared afterward), not on every keyword. Steady-state cost is unchanged.
- **clear_cache may interact with PlatynUI's lazy descriptor retry** → We clear
  BEFORE resolution begins, so the resolution (and its retries) build from live
  state; we do not clear mid-retry. Best-effort and guarded.
- **App still not registered when the first query runs** → The refresh re-reads
  live state; if the app is genuinely not yet in AT-SPI, the query still fails,
  but the agent can re-query (flag is consumed once; a subsequent
  `get_session_state(ui_tree)` or another launch re-arms it). Guidance covers
  this recovery.
- **Can't validate against live AT-SPI in CI** → Unit-test the wiring with a
  mock runtime: launch sets the flag + calls the helper; first resolving keyword
  refreshes + clears the flag; non-desktop/non-launch paths untouched.

## Migration Plan

1. D3 shared `clear_runtime_tree_cache` helper + unit test (mock runtime).
2. D1 clear-on-launch + `desktop_tree_dirty` flag + test.
3. D2 refresh-before-first-post-launch-query + flag-clear + test; non-desktop
   regression.
4. D4 guidance + test.
5. ADR with the second-run evidence; release note.
6. Rollback: all additive and desktop-gated/best-effort; revert independently.

## Open Questions

- Should D2 also refresh before interaction keywords (Pointer Click) that
  resolve a locator, or only Query/Get Attribute? (Lean: include interaction
  locator resolution, since a stale click target is just as broken — but a
  pre-resolved node reference needs no refresh.)
- Land after `desktop-stepwise-followups` (ADR-030).

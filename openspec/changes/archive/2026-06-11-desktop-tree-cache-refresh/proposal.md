## Why

A second live end-to-end run (external Codex agent driving robotmcp against
GNOME Calculator on an isolated display) confirmed the `desktop-stepwise-
followups` fixes work — unqualified `Start Process` launched the app, the
display-state guidance led with history Labels — but exposed a deeper,
run-to-run reliability problem: the PlatynUI accessibility tree was **largely
unresolvable by name** this run.

- `//app:*[@Name='gnome-calculator']` had no usable root
- `//control:Button[@Name='7']` returned empty
- `//control:Label[@Name='56']` (a visible history value) raised
  `ElementNotFoundError`

The agent had to fall back to absolute-coordinate clicks + OCR. The previous run
the same name-based queries worked. The agent's own diagnosis named the cause:
"apps started after the first desktop snapshot may never appear."

Reproduced in code: the PlatynUI runtime **caches the desktop accessibility
tree**. `ui_tree_service.get_ui_tree` clears that cache before evaluating
(`runtime.clear_cache()`), and the e2e helper clears it every round —
explicitly because "the runtime caches the desktop tree … newly launched
applications never become visible". BUT the PlatynUI **keyword executions**
(`Query`, `Get Attribute`, and the locator resolution inside `Pointer Click` /
keyboard keywords) run against the shared runtime's CACHED tree and never clear
it. So a desktop `Start Process` followed by name-based queries resolves against
a stale snapshot taken before the app registered in AT-SPI → not found.

The consequences are severe for the user's stated goal: the generated suite
ended up **clicks-only with zero recorded assertions** (`Should Be*` count = 0),
because the name-based Label assertions never resolved and verification was done
ad-hoc via OCR — a brittle, non-re-runnable suite.

## What Changes

- **A desktop GUI launch refreshes the PlatynUI tree cache.** After a desktop
  session's `Start Process` / `Run Process` of a GUI binary succeeds, the shared
  runtime's tree cache is cleared so the newly-launched application becomes
  visible to subsequent keyword queries (matching what `get_ui_tree` and the
  e2e already do). Best-effort, desktop-gated, never fails the step.
- **The first desktop tree-resolving keyword after a launch sees a fresh tree.**
  A desktop session flagged "tree may be stale" (set on launch) refreshes the
  runtime cache before the next tree-resolving keyword (`Query` / `Get
  Attribute`) so name-based locators and assertions resolve against live AT-SPI
  instead of a pre-launch snapshot. The flag clears after the refresh so steady-
  state queries are not re-snapshotted on every call.
- **Guidance: when a name locator fails right after launch, refresh the tree.**
  The PlatynUI guidance documents that the desktop tree is cached, that the
  keyword path now auto-refreshes after a launch, and that
  `get_session_state(sections=['ui_tree'])` forces a refresh if a locator still
  does not resolve — so agents recover deterministically instead of falling back
  to coordinate clicks + OCR.

## Capabilities

### New Capabilities
- `desktop-tree-cache-refresh`: PlatynUI keyword executions see a fresh
  accessibility tree after a desktop app launch — the runtime cache is cleared
  on launch and refreshed before the first post-launch tree-resolving keyword —
  so name-based locators and assertions resolve reliably instead of run-to-run
  flaky; plus guidance on the refresh path.

### Modified Capabilities
<!-- PlatynUI desktop execution lives in not-yet-archived changes
     (desktop-stepwise-execution-fidelity / -followups); this is a new spec that
     composes with that work. -->

## Impact

- **Code**: `components/execution/keyword_executor.py` (clear the runtime tree
  cache after a desktop GUI launch; refresh before the first post-launch
  tree-resolving keyword via a session stale-flag); `models/session_models.py`
  (a `desktop_tree_dirty`-style session flag); `plugins/builtin/platynui_plugin`
  / `components/execution/ui_tree_service` (a shared `clear_runtime_tree_cache`
  helper, reused by both paths); `utils/rf_native_type_converter.py` (guidance).
- **Behavior**: name-based desktop locators resolve reliably after launching the
  AUT; generated suites can carry real recorded assertions instead of
  coordinate-only clicks. No change to web/api/mobile sessions; steady-state
  desktop queries are not re-snapshotted on every call (only after a launch).
- **Tests**: unit tests that a desktop launch sets the stale-flag and triggers a
  cache clear; that the first post-launch `Query`/`Get Attribute` refreshes and
  the flag clears; that non-desktop sessions and non-launch keywords are
  untouched; guidance documents the refresh path. The cache helper is mocked
  (no live AT-SPI in CI).
- **Dependencies/env**: builds on `desktop-stepwise-followups` (ADR-030),
  `desktop-stepwise-execution-fidelity` (ADR-029). No new dependency. Underlying
  AT-SPI registration latency is upstream; this makes robotmcp re-read live state
  rather than serve a stale snapshot.
- **Docs**: an ADR with the live-run evidence (the second GNOME Calculator run:
  unresolved name locators, coordinate-click fallback, assertion-less suite).

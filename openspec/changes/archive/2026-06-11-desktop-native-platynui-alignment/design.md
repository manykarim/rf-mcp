## Context

Two drivers, one principle (native PlatynUI, not custom robotmcp code):

1. Third live run: an agent launched GNOME Calculator, followed the ADR-031
   refresh path, but the controls never appeared in the AT-SPI tree
   (`application_count: 0`) while the X11 window existed. robotmcp gave no signal
   distinguishing "didn't start" from "started, no AT-SPI tree".
2. robotmcp carries hand-rolled ctypes X11/EWMH code that PlatynUI new-core
   already implements natively.

Researched on the PlatynUI new_core branch (`robotframework-PlatynUI`):
- Native `Runtime` API: `providers()`, `desktop_info()`, `desktop_node()`,
  `top_level_window_for(node)`, `evaluate(xpath, node)`, `focus(node)`,
  `highlight`, `screenshot`, pointer/keyboard. (`packages/native/python/.../_native.pyi`)
- `platynui-cli`: `list-providers`, `info --format json`, `query`, `window`,
  `focus`, `snapshot`, `highlight`, `screenshot`. (`packages/cli/README.md`)
- Linux has an EWMH WindowManager (`_NET_CLIENT_LIST`/`_NET_WM_PID`) — but it is
  used INTERNALLY for XID resolution + WindowSurface actions on already-exposed
  AT-SPI nodes; it is NOT a separate tree provider and NOT surfaced as a
  queryable native API/CLI. (`docs/platform-linux.md §3`,
  `crates/platform-linux-x11/src/window_manager.rs`)

**Spike conclusion (decisive, pre-apply):** the native API/CLI cannot detect a
window independent of the AT-SPI tree, nor expose live-WM-on-display state:
- `runtime.evaluate`/`desktop_node`/`desktop_info` all read the AT-SPI tree.
- `platynui-cli window --list` = `runtime.evaluate(None, "//control:Window")`
  (`crates/cli/src/commands/window.rs:75`, `DEFAULT_WINDOW_QUERY`).
- `_NET_CLIENT_LIST`/`_NET_SUPPORTING_WM_CHECK` are internal-only (no public API).
So "window present but no AT-SPI tree" and "WM-active-on-display" have NO native
equivalent. Decision (recorded): native-first WHERE PlatynUI provides it; a
guarded, documented custom EWMH probe ONLY for these two gaps.

robotmcp's custom code in scope: `platynui_focus._x11_raise_by_pid` (ctypes
`XQueryTree`/`XRaiseWindow` — refactor to native-first, ctypes as fallback) and
`desktop_display_safety._ewmh_wm_present` (ctypes `_NET_SUPPORTING_WM_CHECK` —
KEEP as documented fallback; no native equivalent).

## Goals / Non-Goals

**Goals:**
- Diagnose "window present but no AT-SPI tree" using PlatynUI's NATIVE window
  detection (EWMH WindowManager via the Runtime API / CLI), not custom ctypes.
- Refactor the existing ctypes EWMH/X11 workarounds to native-first.
- Establish native-first as the standing rule for desktop platform needs.

**Non-Goals:**
- Fixing the upstream GTK4/AT-SPI exposure flakiness (diagnose, not fix).
- Removing custom fallbacks that cover environments the native API genuinely
  does not support (WM-less Xvfb) — they stay, behind a native-first attempt.
- Web/api/mobile sessions.

## Decisions

### D1: Guarded EWMH window-presence helper (documented gap) + native providers
Provide a `native_providers()` helper (native `runtime.providers()`) AND a
guarded window-presence probe (ctypes `XQueryTree` + `_NET_WM_PID`/`WM_CLASS`/
`_NET_WM_NAME` matching). Tri-state (present/absent/unknown), best-effort, never
raises. Annotated inline with the spike finding (PlatynUI exposes no native
window list independent of AT-SPI). User-approved guarded custom probe, scoped
to this gap only.
APPLY REFINEMENT (segfault found + fixed): the ctypes probe runs in an ISOLATED
SUBPROCESS. A second in-process Xlib connection conflicts with the PlatynUI
runtime's own Xlib connection (Xlib is not thread-safe) and segfaulted the
process when the native runtime was live. The subprocess gives the probe its own
connection; a crash/timeout there yields "unknown" and cannot take down the
server.

### D2: get_ui_tree emits the exposure diagnostic
In `get_ui_tree`, on the no-matching-app path, call the D1 probe +
`runtime.providers()`. Add `accessibility_not_exposed` (window present, providers
listed, remediation), `app_window_absent` (no window), or "undetermined"
(unknown). Only on the no-matching-app path; the happy path is untouched.

### D3: Keep `_ewmh_wm_present` as the documented WM-active fallback
The spike found no native API exposing live-WM-on-display state, and the
active-desktop safety classification depends on that signal. KEEP the existing
ctypes `_NET_SUPPORTING_WM_CHECK` probe; re-annotate it with the spike finding
(why no native path exists). `runtime.providers()` may enrich the reported state
but MUST NOT replace the security-relevant WM-active signal. Preserve EXACTLY the
isolation-marker precedence, fail-closed-on-unknown, and isolated/active outcomes
(no safety change). This is NOT a removal — it is a documented native-gap.

### D4: Refactor X11 raise to native-first
In `platynui_focus.focus_window`, attempt the native `Runtime.focus()` /
WindowSurface `activate()` FIRST; reach the ctypes `_x11_raise_by_pid` only when
the native pattern is unavailable (the existing WM-less-Xvfb case). Mark the
ctypes path explicitly as a documented last-resort fallback. Behavior on a real
desktop becomes native; headless keeps working.

### D5: Guidance + standing rule
Guidance references the exposure diagnostic and states the native-first rule.

## Risks / Trade-offs

- **No native window list exists (spike-confirmed)** → window presence uses the
  guarded EWMH probe factored from existing ctypes; reuse, do not add new ctypes
  surface. Re-check on PlatynUI upgrades — if a native window list is added
  later, swap the probe for it.
- **Refactoring the safety EWMH probe could regress isolation classification**
  (security-sensitive) → Preserve marker precedence + fail-closed; keep the
  existing probe as a fallback; re-run ALL existing safety scenarios unchanged.
- **Refactoring the focus raise could break headless e2e** → Native-first with
  the ctypes raise retained as fallback; the WM-less path is unchanged in effect.
- **`platynui-cli` may be absent** → Use it only when present; native Runtime API
  is the primary; never hard-depend on the CLI.
- **No live AT-SPI/X11 in CI** → Mock the native runtime/providers and the
  window helper; assert diagnostics and the preserved safety outcomes.

## Migration Plan

1. D1 native window-presence/providers helper (+ apply-time confirmation of the
   exact native call) + unit tests (mock runtime/CLI).
2. D2 exposure diagnostic in get_ui_tree + tests.
3. D3 refactor `_ewmh_wm_present` native-first; re-run all safety scenarios.
4. D4 refactor focus raise native-first; keep headless fallback; tests.
5. D5 guidance + native-first rule.
6. Full suite green; ADR + release note.
7. Rollback: each refactor keeps its prior probe as a fallback, so revert is
   additive/independent.

## Open Questions

- RESOLVED by the spike: the new-core `Runtime`/CLI exposes no standalone window
  list independent of AT-SPI (`platynui-cli window` → `//control:Window` via the
  AT-SPI tree; `_NET_CLIENT_LIST` internal-only). Hence the guarded EWMH probe.
- Worth an upstream PlatynUI feature request: a native window-list / WM-active
  API would let robotmcp drop the custom probe later. (Track, not block.)
- Should the exposure diagnostic also fire from a desktop `Query` that returns
  empty post-launch, or only `get_ui_tree`? (Lean: get_ui_tree only.)
- Land after `desktop-tree-cache-refresh` (ADR-031).

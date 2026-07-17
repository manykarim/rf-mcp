## Why

Two drivers, both pointing at the same principle — **use PlatynUI's native CLI
and API instead of custom robotmcp solutions**:

1. **A diagnostic gap (third live run).** An external agent launched GNOME
   Calculator via robotmcp, followed the tree-refresh path (ADR-031), but the
   controls never appeared in the AT-SPI tree (`application_count: 0`) even
   though the X11 window existed. robotmcp returned only an empty tree with no
   signal distinguishing "app did not start" from "app started but exposes no
   accessibility tree". The agent diagnosed it manually via `ps`/`gsettings`/
   `xwininfo`.

2. **Custom workarounds duplicate native PlatynUI capabilities.** robotmcp
   carries hand-rolled `ctypes` X11/EWMH code that PlatynUI already implements
   natively: `desktop_display_safety._ewmh_wm_present`
   (`_NET_SUPPORTING_WM_CHECK` via ctypes) and `platynui_focus._x11_raise_by_pid`
   (`XQueryTree`/`XRaiseWindow` via ctypes). PlatynUI's new-core exposes a native
   `Runtime` API — `providers()`, `desktop_info()`, `top_level_window_for()`,
   `focus()`, `evaluate()`, WindowSurface patterns — and a `platynui-cli` with
   `list-providers`, `info`, `query`, `window`, `focus`, `snapshot`. On Linux it
   ships a dedicated **EWMH WindowManager** (`_NET_CLIENT_LIST` + `_NET_WM_PID` +
   `_NET_WM_NAME`) that resolves windows independently of the AT-SPI2 provider —
   exactly the capability the custom ctypes code reimplements, and exactly what
   the exposure diagnostic needs.

The upstream GTK4→AT-SPI exposure flakiness is not robotmcp's to fix, but
robotmcp should DIAGNOSE it — and should do so (and everything else
window/desktop related) through PlatynUI's native providers, not parallel
ctypes.

## Spike finding (decisive — pre-apply)

A spike against the PlatynUI new_core branch established that the native API/CLI
**cannot detect a window independent of the AT-SPI control tree**, and exposes
**no native WM-active-on-display state**:
- `runtime.evaluate(xpath)`, `runtime.desktop_node()`/`desktop_info()` all read
  the AT-SPI-built tree (Linux `DesktopInfoProvider` = AT-SPI2).
- `platynui-cli window --list` calls `runtime.evaluate(None, "//control:Window")`
  (`crates/cli/src/commands/window.rs:75`, `DEFAULT_WINDOW_QUERY="//control:Window"`)
  — the SAME AT-SPI tree; returns empty when AT-SPI is empty.
- The EWMH `_NET_CLIENT_LIST` / `_NET_SUPPORTING_WM_CHECK` logic exists only
  INTERNALLY (`crates/platform-linux-x11/src/window_manager.rs`) for XID
  resolution + WindowSurface actions on already-exposed nodes; it is NOT surfaced
  as a queryable native API or CLI command.

Therefore the precise "window present but no AT-SPI tree" distinction and the
"is a live WM on this display" signal have **no native equivalent**. Per the
decision recorded here, robotmcp uses native PlatynUI wherever it provides the
capability, and a documented, guarded **custom EWMH probe** (reusing robotmcp's
existing ctypes) ONLY for these two gaps — each annotated with the spike finding.

## What Changes

- **Accessibility-exposure diagnostic: native providers + a guarded EWMH window
  probe.** When a desktop `get_ui_tree` inspection finds no application matching
  the requested filter, robotmcp reports the active providers from the NATIVE
  `runtime.providers()` API, and uses a guarded custom EWMH window-presence probe
  (factored from robotmcp's existing ctypes `_NET_CLIENT_LIST`/`_NET_WM_PID`
  enumeration — PlatynUI has no native window list independent of AT-SPI) to
  decide: window present but control tree empty → `accessibility_not_exposed`;
  no window → `app_window_absent`; probe unavailable → undetermined. The
  diagnostic carries actionable remediation. The probe is documented as the
  fallback for a missing native capability (spike-cited).
- **Refactor window focus/raise to native-first.** `platynui_focus` uses the
  native focus/activate path (`runtime.focus()` / WindowSurface `activate()`)
  FIRST; the custom ctypes `XRaiseWindow` raise is retained ONLY as a
  clearly-marked last-resort fallback for environments where the native
  WindowSurface pattern is genuinely unavailable (WM-less Xvfb), reached only
  after the native attempt. This part IS a genuine native-first refactor (the
  native capability exists).
- **EWMH WM-active probe stays as a documented fallback.**
  `desktop_display_safety._ewmh_wm_present` (ctypes `_NET_SUPPORTING_WM_CHECK`)
  is KEPT — the spike found no native API exposing live-WM-on-display state, and
  the active-desktop safety classification depends on that signal. It is
  re-annotated with the spike finding (why no native path exists) rather than
  removed; `runtime.providers()` is consulted first only to enrich, never to
  replace the security-relevant signal.
- **Native-first as the standing rule (with documented gaps).** New desktop
  needs use the native `Runtime` API / `platynui-cli` WHERE PlatynUI provides
  the capability; a guarded, documented custom probe is permitted ONLY where the
  spike (or a future equivalent check) shows no native equivalent exists.

## Capabilities

### New Capabilities
- `desktop-accessibility-exposure-diagnostic`: distinguish "app window present
  but no AT-SPI tree" from "app window absent", surfaced as an
  `accessibility_not_exposed` / `app_window_absent` diagnostic on `get_ui_tree`
  with the native `providers()` and remediation. Window presence comes from a
  guarded, documented EWMH probe (PlatynUI exposes no native window list
  independent of AT-SPI — spike-cited).
- `native-platynui-desktop-alignment`: robotmcp's desktop window operations use
  PlatynUI's native `Runtime` API WHERE it provides the capability (window
  focus/raise → native-first); custom ctypes is retained, native-first or as a
  documented fallback, only where the spike shows no native equivalent (EWMH
  window-presence + WM-active).

### Modified Capabilities
<!-- The desktop safety guard and focus-before-act live in not-yet-archived
     changes (platynui-desktop-safety-isolation, platynui-focused-execution);
     these are new specs that compose with — and refactor — that work. -->

## Impact

- **Code**: `ui_tree_service.py` (exposure diagnostic on the no-matching-app
  path: native `providers()` + guarded EWMH window probe); `platynui_focus.py`
  (native focus/activate first; ctypes raise demoted to a documented fallback) +
  a factored `_x11_window_present` helper from its existing ctypes enumeration;
  `desktop_display_safety.py` (`_ewmh_wm_present` kept + re-annotated with the
  spike finding; `providers()` consulted only to enrich); `platynui_plugin.py`
  (a small native `providers()` helper on the shared runtime);
  `utils/rf_native_type_converter.py` (guidance references the diagnostic and
  the native-first-with-documented-gaps rule).
- **Behavior**: a launched-but-unexposed desktop app yields a precise diagnostic
  instead of an opaque empty tree; window/WM detection routes through native
  PlatynUI. The safety guard keeps its fail-closed + marker-precedence
  semantics; the focus fallback keeps working on WM-less headless displays.
- **Tests**: unit tests with the native `Runtime`/probe mocked — exposure
  diagnostic (window-present→`accessibility_not_exposed`, absent→
  `app_window_absent`, app-in-tree→none, native-unavailable→graceful); the
  refactored EWMH/active classification preserves the existing safety scenarios
  (isolated/active/marker precedence/fail-closed); the focus path prefers native
  and falls back only when the native pattern is unavailable. Native calls are
  mocked (no live AT-SPI/X11 in CI).
- **Dependencies/env**: builds on `desktop-tree-cache-refresh` (ADR-031) and the
  safety/focus changes (ADR-026/027). Uses the already-required
  `platynui-native` Runtime; `platynui-cli` is optional (used when present,
  native API otherwise). No new hard dependency.
- **Docs**: an ADR mapping the third-run evidence + the native-first refactor;
  reference the PlatynUI new-core `Runtime` API and `platynui-cli` README.

## ADDED Requirements

### Requirement: WM-active detection — native where exposed, documented EWMH fallback otherwise

The system SHALL prefer PlatynUI's native `Runtime` state when reporting desktop
window-manager / active-desktop status. Because the spike found NO native API
exposing live-WM-on-display state (`_NET_SUPPORTING_WM_CHECK` is internal-only),
the existing ctypes EWMH probe (`_ewmh_wm_present`) is RETAINED as the documented
fallback that supplies the security-relevant signal; `runtime.providers()` MAY be
consulted to enrich the report but MUST NOT replace that signal. The probe MUST
carry an inline note citing the missing native capability.

#### Scenario: native consulted, EWMH probe supplies the WM-active signal
- **WHEN** the active-desktop safety classification needs to know whether a live
  WM owns the bound display
- **THEN** the EWMH probe supplies that signal (no native equivalent exists),
  optionally enriched by `runtime.providers()`, annotated as a documented gap

#### Scenario: safety semantics are preserved
- **WHEN** the classification runs
- **THEN** the isolation-marker precedence, the fail-closed-on-unknown behavior,
  and the isolated/active outcomes are unchanged from the prior safety guard

### Requirement: Window focus/raise uses the native focus path first

The system SHALL focus/raise an application window via PlatynUI's native path
(`Runtime.focus()` / WindowSurface `activate()`) before any custom ctypes
`XRaiseWindow` fallback. The custom raise is retained ONLY as a clearly-marked
last resort for environments where the native WindowSurface pattern is genuinely
unavailable (e.g. a WM-less Xvfb), and is reached only after the native attempt.

#### Scenario: native focus attempted first
- **WHEN** robotmcp needs to focus/raise the AUT window
- **THEN** it tries the native focus/activate path first; the ctypes raise runs
  only if the native pattern is unavailable

#### Scenario: headless fallback still works
- **WHEN** the native WindowSurface pattern is unavailable (WM-less display)
- **THEN** the documented ctypes fallback still raises the window so headless
  flows keep working

### Requirement: Native-first is the standing rule for desktop platform needs

The system SHALL implement new desktop window / accessibility / provider needs
through the native PlatynUI `Runtime` API or `platynui-cli`; custom platform
code is permitted only as a documented fallback when a native capability is
missing on the supported PlatynUI new-core branch.

#### Scenario: a native capability exists → use it
- **WHEN** a desktop need is covered by the native Runtime API / CLI
- **THEN** robotmcp uses the native capability rather than reimplementing it

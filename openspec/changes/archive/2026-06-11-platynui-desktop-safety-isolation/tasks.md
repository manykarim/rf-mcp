## 1. Reproduction harness (findings A–D)

- [x] 1.1 Repro A (classification): unit test asserting current
  `detect_session_type_from_scenario("Test the calculator app")` →
  `mobile_testing` (red), to be flipped green by the classification fix
- [x] 1.2 Repro B (snap launch): unit test for the env sanitizer using a
  synthetic snap-rooted `LD_LIBRARY_PATH`/`GTK_PATH`/`GIO_MODULE_DIR` parent
  env — assert the sanitized child env drops the `/snap/`-rooted entries
- [x] 1.3 Repro C (runtime init): document/script the diff between the working
  pytest isolation path and the failing Robot/MCP path; capture the
  `ProviderError ... not available after shutdown` trigger
- [x] 1.4 Repro D (fresh-isolated input): confirm under the isolation
  bootstrap (with ADR-025 coordinate patch + ADR-026 focus) that clicks mutate
  the calculator display (the report's no-mutation case)

## 2. Desktop scenario classification fix

- [x] 2.1 In `detect_session_type_from_scenario`, codify the precedence: an
  explicit mobile signal (android/ios/appium/apk/bundle-id/emulator/device/
  uiautomator2/xcuitest/tap/swipe) → mobile; else a desktop signal (desktop-app
  name or toolkit gnome/gtk/qt/kde/win32/wpf/x11/wayland/exe/native-window/
  platynui) → desktop; the bare token "app" alone never yields mobile when a
  desktop signal is present and no explicit mobile signal is present
- [x] 2.2 Negative-case coverage: "calculator app on android" → mobile;
  "iOS calculator app" → mobile; "calculator app on windows" → desktop;
  "calculator app" → desktop
- [x] 2.3 Desktop-typed session library allow-list includes Process/BuiltIn/
  PlatynUI.BareMetal so `Process` import is not rejected
- [x] 2.4 `analyze_scenario` / `recommend_libraries` return the desktop set
  (not Appium) for these scenarios
- [x] 2.5 Unit tests: classification matrix (calculator/editor/desktop-app vs
  genuine mobile), desktop session accepts Process, recommendations correct

## 3. Desktop launch-env sanitization

- [x] 3.1 Add a child-env sanitizer that filters `/snap/`-rooted **path
  segments** (not whole vars) from `LD_LIBRARY_PATH`, `LD_PRELOAD`, `GTK_PATH`,
  `GTK_EXE_PREFIX`, `GIO_MODULE_DIR`, `GIO_EXTRA_MODULES`, `GSETTINGS_SCHEMA_DIR`,
  `QT_PLUGIN_PATH`, `XDG_DATA_DIRS`, `FONTCONFIG_FILE`, `FONTCONFIG_PATH`,
  `LOCPATH`; preserve non-snap segments; then add the display vars (DISPLAY,
  XDG_SESSION_TYPE=x11, GDK_BACKEND=x11, WAYLAND_DISPLAY unset)
- [x] 3.1a Snap-AUT preservation: when the launched binary is itself a
  snap-confined app, do NOT strip that snap's roots; expose a `--no-sanitize`
  escape hatch
- [x] 3.2 Wire it into desktop-session `Start Process` launches for known GUI
  binaries via RF `env:`-style per-var overrides (do not replace the whole env
  with `env=`); leave non-GUI / non-desktop launches unchanged
- [x] 3.3 Detect immediate-exit GUI launches and surface captured stderr + the
  env-contamination cause/mitigation in the step result
- [x] 3.4 Unit tests: sanitizer drops only snap-rooted entries; clean env
  passthrough; immediate-exit diagnostic

## 4. Active-desktop safety guard

- [x] 4.1 Tri-state bound-display classifier returning `isolated`/`active`/
  `unknown`: `isolated` ONLY on a positive rf-mcp isolation marker (marker
  beats the EWMH probe); `active` when `_NET_SUPPORTING_WM_CHECK` resolves to a
  live window; `unknown` otherwise (incl. probe failure). Absence of a WM is
  NOT isolated
- [x] 4.2 At the desktop chokepoint (before focus/dispatch), fail closed:
  refuse pointer/keyboard ops on `active` AND `unknown` unless opt-in is set;
  allow only on `isolated`; clear actionable error
- [x] 4.3 Auditable opt-in: `ROBOTMCP_PLATYNUI_ALLOW_ACTIVE_DESKTOP=1` +
  per-session attribute; flag bypassed runs in the result payload + log WARNING.
  Also implement the one-release `ROBOTMCP_PLATYNUI_SAFETY_GUARD=warn` transition
  mode (logs, does not block)
- [x] 4.4 Surface bound-display state (isolated/active/unknown + enforcing) in
  session init and `get_session_state`, computed via the runtime broker (no
  throwaway runtime)
- [x] 4.5 Unit tests: classifier decisions (mock the EWMH probe + the marker),
  marker-beats-probe, probe-failure→unknown→refuse, non-EWMH WM on :0 →
  unknown→refuse (regression for the dangerous false-allow), opt-in bypass +
  audit flag, state surfaced

## 5. Runtime-binding stability

- [x] 5.1 Introduce a runtime broker `get_runtime()` (next to
  `ensure_x11_session_env`) with states `open`/`shutting_down`/`closed`,
  lock-protected lazy first bind, thread-affine, session/display-scoped;
  refuse re-init after `closed` with a clear "restart the process" error
- [x] 5.1a FIX the proven proximate cause: `ui_tree_service._collect_ui_tree_sync`
  currently creates a `pn.Runtime()` and calls `runtime.shutdown()` per call —
  switch it to the broker (no per-call create/shutdown). Repoint
  `platynui_focus._get_runtime` (and any BareMetal/attach site) to the broker
- [x] 5.2 Guarantee display-env settle (`ensure_x11_session_env` + in-session/
  bootstrap display mutation, incl. the server process's own DISPLAY/XAUTHORITY)
  precedes the first bind on the MCP/Robot path
- [x] 5.3 Eliminate the "not available after shutdown or failed connect" path;
  reuse the broker runtime, never re-initialize the platform module
- [x] 5.4 Unit tests: broker reused across two `_collect_ui_tree_sync` calls
  (no new Runtime), state transitions enforced, concurrent first-use binds once
  under the lock; document any residual upstream PlatynUI limitation (file
  upstream with a reproducer if purely upstream)

## 6. Isolation bootstrap + validated suite

- [x] 6.1 Provide a supported isolation bootstrap (script and/or session
  option) that prepares Xvfb + documented env + session bus and runs the
  confined session; codify the pytest-reference recipe
- [x] 6.1a Add a VISIBLE isolation mode: nested X server (Xephyr) rendering the
  isolated display as a host window for stepwise observation, with a VNC-backed
  display fallback; visible = preferred default for interactive/stepwise use,
  headless Xvfb = default for CI; verify the safety guard still treats the
  visible nested/VNC display as isolated
- [x] 6.2 Replace the investigation artifact
  `tests/e2e/gnome_calculator_mcp_stepwise.robot` with a validated stepwise
  calculator suite (per-entry + result assertions) that runs green isolated,
  reusing the proven `_raise_x11_window`/`_editor_type` patterns from
  `tests/integration/test_platynui_gnome_apps_e2e.py` so finding #5's WM-less
  input path is actually exercised
- [x] 6.2a Finding #5 live mutation acceptance (must-pass): a stepwise GNOME
  Calculator flow in isolated mode where the display character count provably
  changes after each click and the result matches; a resolve-but-no-mutate is
  tracked as an explicit investigate/fix item, not reported as success
- [x] 6.3 Live isolated e2e: the validated suite + a Text Editor scenario run
  green under the bootstrap (extends the existing GNOME e2e); add a visible-mode
  (Xephyr) smoke confirming the safety guard still classifies `:N` as isolated
  via the marker

## 7. Validation, docs, wrap-up

- [x] 7.1 Full unit suite + new tests green; confirm web/mobile/API sessions
  unaffected by the guard / sanitizer / classification changes
- [x] 7.2 Run the reproduction harness: each finding (1–6) reproduced pre-fix
  and passing post-fix, with **live MCP-driven** verification (not only unit/
  spec/suite-generation) for the safety guard, classification, launch, and the
  finding-#5 mutation acceptance
- [x] 7.3 Author an ADR capturing the desktop safety model + the reproduction
  findings; cross-reference the investigation report
- [x] 7.4 Release notes for the breaking refuse-by-default safety guard and the
  opt-in / sanitization / classification changes
- [x] 7.5 Update the investigation report (or a follow-up note) mapping each
  finding to its fix + status

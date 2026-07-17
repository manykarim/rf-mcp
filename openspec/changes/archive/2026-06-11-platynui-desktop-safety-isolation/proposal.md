## Why

A real VS Code MCP session driving GNOME Calculator through rf-mcp produced a
stack of failures (documented in
`docs/gnome-calculator-mcp-findings-2026-06-08.md`), the worst of which is a
**safety blocker**: desktop pointer/keyboard actions leaked onto the user's
**active IDE desktop** (`DISPLAY=:0`) instead of being confined to the AUT —
because nothing in rf-mcp stops a desktop session from binding to and driving
the user's live session. Around that sit four more issues that make
MCP-driven desktop automation unreliable: GNOME Calculator scenarios are
misclassified as `mobile_testing` (auto-loading AppiumLibrary), `Start
Process` launches of GNOME apps fail under the VS Code snap environment
(`__libc_pthread_init ... GLIBC_PRIVATE` from `/snap/core20`), the PlatynUI
runtime fails to bind on the Robot/MCP path even when an isolated Xvfb is
healthy (`ProviderError ... x11 connection: not available after shutdown or
failed connect`), and clicks resolve nodes but do not mutate display state.
This change is investigation-driven: reproduce each finding, then ship the
fixes — safety and correct session classification first.

## What Changes

- **Active-desktop safety guard (headline). BREAKING (behavioral).** A
  DESKTOP_TESTING session classifies its bound display as **isolated /
  active / unknown** and **fails closed**: it dispatches desktop pointer/
  keyboard operations only when the display is **provably isolated** (an
  rf-mcp-owned isolation marker is present), and **refuses by default** on both
  `active` and `unknown` (so a display we cannot prove safe is treated as
  unsafe). Isolation is established by **positive proof** — an rf-mcp-created X
  server with a dedicated `XAUTHORITY`/socket and a bootstrap marker — not by
  the *absence* of an EWMH window manager (which would false-allow non-EWMH
  real desktops). The user may explicitly opt in
  (`ROBOTMCP_PLATYNUI_ALLOW_ACTIVE_DESKTOP=1`); opt-in runs are flagged in the
  result payload and logged at WARNING for auditability. This prevents clicks/
  typing from leaking onto the user's session.
- **Desktop scenario classification fix.** GNOME / native-desktop scenarios
  (calculator, text editor, "desktop app", GTK/Qt/Win32) never default to
  `mobile_testing`/Appium; `analyze_scenario` resolves them to a
  desktop/PlatynUI session, and a desktop-typed session stops rejecting
  `Process`/PlatynUI libraries.
- **Desktop launch-env sanitization.** When launching a desktop AUT (via
  `Process.Start Process` for a known GUI app under a desktop session),
  rf-mcp sanitizes the **child** environment so package-rooted loader/module
  variables from a VS Code snap shell do not break the GNOME app — stripping
  `/snap/`-rooted segments from `LD_LIBRARY_PATH`, `LD_PRELOAD`, `GTK_PATH`,
  `GTK_EXE_PREFIX`, `GIO_MODULE_DIR`, `GIO_EXTRA_MODULES`,
  `GSETTINGS_SCHEMA_DIR`, `QT_PLUGIN_PATH`, `XDG_DATA_DIRS`, `FONTCONFIG_FILE`/
  `FONTCONFIG_PATH`, and `LOCPATH` — while **preserving the vars a snap-
  confined AUT needs for its own snap** (do not strip the matching snap's
  roots when the AUT is itself that snap). The AUT also inherits the bound
  isolated display.
- **Runtime-binding stability.** A single **runtime broker** scoped to the
  bound desktop session/display owns the PlatynUI native runtime: it binds
  **once, lazily, after the display env is settled**, with explicit states
  (`open` / `shutting_down` / `closed`), and all PlatynUI call sites use it.
  Critically, the `ui_tree` path stops creating and shutting down its own
  per-call runtime (the proximate cause of the `ProviderError ... not available
  after shutdown` on the Robot/MCP path) and instead reuses the broker. No path
  re-initializes the process-global platform module after shutdown.
- **First-class desktop isolation bootstrap, with a VISIBLE mode.** A
  supported way to start an isolated display + bootstrap a confined desktop
  session (so users do not hand-roll Xvfb in ad-hoc Robot suite setup), with
  the documented environment baked in. Crucially, the bootstrap offers a
  **visible/observable** isolated display (a nested X server such as Xephyr —
  a window on the host showing the isolated display — or a VNC-backed display)
  in addition to the headless Xvfb mode, so that during **stepwise execution
  the user can watch the actual desktop application and its interactions**
  while the session stays confined and off the user's active session. Visible
  mode is the preferred default for interactive/stepwise use; headless Xvfb
  remains the default for CI/automated runs.
- **Reproduction harness + curated suite + live mutation acceptance.**
  Scripts/tests that reproduce each finding (A–D from the report) and a
  validated stepwise calculator suite that replaces the investigation
  artifact. Finding #5 ("frame resolves but clicks don't mutate") is closed by
  an explicit **must-pass live acceptance criterion**: in isolated mode, a
  stepwise GNOME Calculator flow where each click provably changes the AUT's
  observable state (display character count). If a residual input-delivery
  cause remains, it is tracked as an explicit investigate/fix item rather than
  implied-solved.

## Capabilities

### New Capabilities
- `desktop-session-safety`: detect when a desktop session would drive the
  user's active display and refuse-by-default with an opt-in escape hatch;
  surface the bound-display safety state.
- `desktop-scenario-classification`: route GNOME/native-desktop scenarios to a
  desktop/PlatynUI session (never mobile/Appium); keep desktop-typed sessions
  from rejecting Process/PlatynUI.
- `desktop-launch-sanitization`: sanitize the child environment when launching
  a desktop AUT so snap/IDE env contamination does not break GUI app startup
  and the AUT inherits the isolated display.
- `desktop-runtime-binding`: stable, once-and-lazy PlatynUI runtime binding on
  the MCP/Robot path; no "not available after shutdown" failures; reliable
  input delivery once the AUT frame is resolvable.
- `desktop-isolation-bootstrap`: a supported isolated-display bootstrap for
  confined desktop sessions with the documented environment, including a
  **visible/observable** mode (nested X server / VNC) so stepwise interactions
  can be watched without using the user's active session, alongside a headless
  Xvfb mode for CI.

### Modified Capabilities
<!-- The PlatynUI focus/visibility behavior (change platynui-focused-execution,
     ADR-026) is not yet an archived openspec capability, so these are all new
     specs; this change composes with — and depends on — that focus work. -->

## Impact

- **Code**: `utils/library_detection.py` + `models/session_models.py`
  (classification + desktop-typed library allow-list);
  `components/execution/keyword_executor.py` /
  `plugins/builtin/platynui_plugin.py` (active-desktop guard, launch
  sanitization hook); `components/execution/platynui_focus.py` /
  `ui_tree_service.py` (bound-display + binding lifecycle); `server.py` (init
  ordering, optional isolation-bootstrap surface).
- **Behavior**: desktop sessions refuse to drive the active desktop by
  default (opt-out available); GNOME scenarios classify as desktop; GUI app
  launches are sanitized. No change to web/mobile/API sessions.
- **Tests**: reproduction scripts (findings A–D), unit tests
  (classification, guard decision, env sanitization, binding lifecycle), and a
  validated isolated stepwise calculator suite replacing
  `tests/e2e/gnome_calculator_mcp_stepwise.robot`.
- **Dependencies/env**: relies on the ADR-026 focus/X11-raise work and the
  ADR-025 `resolve_extents` patch; documents the VS Code snap contamination
  and the required isolated-display environment. No new Python dependency.
- **Docs**: a new ADR capturing the safety model + the reproduction findings;
  the investigation report referenced as the source of record.

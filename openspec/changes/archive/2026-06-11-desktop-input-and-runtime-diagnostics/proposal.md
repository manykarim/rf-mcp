## Why

An 8-run live investigation (external Codex agent driving robotmcp against GNOME
desktop apps) validated the desktop read/locate/suite-build path end-to-end, but
surfaced three robotmcp robustness gaps that silently turned hard failures into
confusing "successes":

1. **Input-with-no-effect is silently reported as `OK`.** On the gnome-text-editor
   run, `Pointer Click`/`Keyboard Type` returned success, but the text never
   entered (`native:Text.CharacterCount` stayed `0`, the saved file stayed
   empty). robotmcp gave no signal — the agent only noticed via its own
   assertions. A pure decision helper for this already exists
   (`desktop_execution_signals.input_effect_hint`, ADR-029 D2b) but is NOT wired
   into execution.
2. **Synthetic X11 input is silently blocked on Wayland.** robotmcp forces the
   X11 backend (`ensure_x11_session_env`) and injects input via XTest. On a real
   Wayland session the compositor (mutter) blocks synthetic X input, so every
   click/keystroke "succeeds" but does nothing. `ensure_x11_session_env` already
   DETECTS the Wayland origin but does not warn callers.
3. **A PlatynUI runtime bind/connect failure reports as "not installed".** When
   the MCP server's launch env lacked `XAUTHORITY`, the runtime broker failed to
   connect to `:0` (`ProviderError ... x11 connection: not available after
   shutdown or failed connect`). `get_session_state(ui_tree)` then returned the
   generic `platynui-native not installed or runtime unavailable` — misleading,
   since the package IS installed; the real problem was display/auth. The native
   module is ONE-SHOT (no re-init after a failed/disposed bind), so the broker
   stays dead until the process restarts — but the agent was never told that.

Each of these turned a clear, fixable condition into a silent or misleading one.
This change makes them loud and actionable.

## What Changes

- **Auto-detect input-with-no-effect (best-effort, non-reentrant).** For a
  desktop keyboard/typing interaction targeting a resolvable text node, robotmcp
  snapshots the target's `native:Text.CharacterCount` BEFORE and AFTER the
  keyword using the shared native `Runtime` directly (no re-entrant RF
  execution under the lock), and when the keyword succeeds but the count did not
  change, attaches a `desktop_input_no_effect` warning (via the existing
  `input_effect_hint`). Best-effort, soft, never fails the step; skipped when no
  text-state snapshot is determinable.
- **Warn when forced-X11 input runs on a Wayland session.** `ensure_x11_session_env`
  records whether the ORIGINAL session was Wayland. On the first desktop
  interaction keyword of such a session, robotmcp attaches a
  `wayland_x11_input_blocked_risk` warning: synthetic X11 (XTest) input is likely
  blocked by the Wayland compositor; remediation — run on a real X11 session, or
  use PlatynUI's Wayland input backend when available. Read/query operations
  (AT-SPI over D-Bus) are unaffected and the warning says so.
- **Classify runtime bind/connect failures with an actionable diagnostic.** The
  runtime broker records the last bind/connect failure reason and classifies it:
  `not_installed` (import failed) vs `display_connect_failed` (DISPLAY/XAUTHORITY/
  XDG_RUNTIME_DIR) vs `disposed` (one-shot module — restart required). A
  `runtime_unavailable_reason()` accessor exposes it. `get_session_state(ui_tree)`
  (and other runtime-None paths) return this structured diagnostic — "failed to
  connect to display :0 — check DISPLAY/XAUTHORITY; the native runtime is
  one-shot, restart the MCP server" — instead of the bare "not installed".
  (Recovery itself is bounded by the one-shot native module; this makes the
  failure diagnosable and the restart-need explicit, not silent.)

## Capabilities

### New Capabilities
- `desktop-input-effect-detection`: a successful desktop interaction that did not
  change the target's accessible state is flagged (auto-wired, non-reentrant,
  best-effort).
- `desktop-wayland-input-warning`: a forced-X11 desktop session that originated
  as Wayland warns that synthetic X11 input may be blocked by the compositor.
- `desktop-runtime-failure-diagnostics`: PlatynUI runtime bind/connect failures
  are classified and surfaced as an actionable diagnostic (display/auth/restart)
  rather than a generic "not installed".

### Modified Capabilities
<!-- The desktop execution / safety / ui_tree paths live in not-yet-archived
     changes (ADR-026..032); these are new specs that compose with that work. -->

## Impact

- **Code**: `keyword_executor.py` (wire input-effect snapshot + Wayland warning
  on the desktop interaction path); `desktop_execution_signals.py` (helpers
  already present; add the Wayland-warning helper); `platynui_plugin.py`
  (`ensure_x11_session_env` records Wayland origin; broker records + classifies
  the last bind failure; `runtime_unavailable_reason()`);
  `ui_tree_service.py` (runtime-None branch returns the classified diagnostic).
- **Behavior**: a no-effect click/keystroke, a Wayland-blocked input session, and
  a runtime display/auth failure now each yield a clear, actionable signal
  instead of a silent "OK" or a misleading "not installed". No change to the
  happy path, to read/query operations, or to web/api/mobile sessions.
- **Tests**: unit tests with the native runtime / signals mocked — input-effect
  (success+unchanged → warning; changed → none; no-snapshot → none); Wayland
  warning fires only on a was-Wayland session's first interaction; runtime-reason
  classification (not_installed / display_connect_failed / disposed) and the
  ui_tree diagnostic. No live display/AT-SPI in CI.
- **Dependencies/env**: builds on ADR-029 (input_effect_hint), ADR-032 (exposure
  diagnostic), and the safety/broker work. No new dependency. Does not attempt to
  make Wayland input work or to re-init the one-shot runtime — it diagnoses both.
- **Docs**: an ADR mapping the 8-run findings (input-no-effect, Wayland-blocked
  input, XAUTHORITY/runtime-connect failure) to these diagnostics.

## 1. Reproduction / grounding

- [x] 1.1 Confirm in code: `input_effect_hint` exists but is unwired; the desktop
  interaction path returns `OK` with no state-change check (finding #1)
- [x] 1.2 Confirm `ensure_x11_session_env` detects Wayland origin but does not
  surface it; forced-X11 XTest input is blocked on Wayland (finding #2)
- [x] 1.3 Confirm `get_runtime` returns None on bind failure and `get_ui_tree`
  reports generic "not installed"; native module is one-shot (finding #3/#4)

## 2. Runtime failure classification + diagnostic (D3)

- [x] 2.1 Broker records the last bind exception; add `runtime_unavailable_reason()`
  classifying `not_installed` / `display_connect_failed` / `disposed` / None
  (heuristic on the error; never raises)
- [x] 2.2 `ui_tree_service` runtime-None branch returns a structured diagnostic
  from the reason (display/auth + one-shot-restart for connect failures; install
  hint for not_installed)
- [x] 2.3 Unit tests: each reason classified; ui_tree diagnostic content per
  reason; never raises

## 3. Wayland-input warning (D2)

- [x] 3.1 `ensure_x11_session_env` records the ORIGINAL session type; add a
  `was_wayland_session()` / origin accessor
- [x] 3.2 `desktop_execution_signals.wayland_input_warning(keyword, origin)`
  returns the `wayland_x11_input_blocked_risk` hint for an interaction keyword on
  a was-Wayland session (with remediation + read-unaffected note)
- [x] 3.3 Wire it on the FIRST desktop interaction keyword per session (session
  flag, fires once); read/query keywords do not warn
- [x] 3.4 Unit tests: was-Wayland interaction warns once; X11-origin no warn;
  read/query no warn

## 4. Input-effect auto-detection (D1)

- [x] 4.1 In the desktop branch of `_execute_keyword`, for a keyboard/typing
  interaction with a resolvable text node, snapshot `native:Text.CharacterCount`
  via the native `Runtime` DIRECTLY (non-reentrant) before/after; pass to
  `input_effect_hint`; append `desktop_input_no_effect` on success+unchanged
- [x] 4.2 Best-effort: skip when no before snapshot / unreadable / any error;
  never fail the step; never re-enter RF execution under the lock
- [x] 4.3 Unit tests: success+unchanged → warning; changed → none; no-snapshot →
  none; non-interaction keyword → none

## 5. Validation + docs

- [x] 5.1 Full unit suite green; happy path + read/query + web/mobile unaffected
- [x] 5.2 ADR mapping the 8-run findings (input-no-effect, Wayland-blocked input,
  XAUTHORITY/runtime-connect failure) to these diagnostics
- [x] 5.3 Release note

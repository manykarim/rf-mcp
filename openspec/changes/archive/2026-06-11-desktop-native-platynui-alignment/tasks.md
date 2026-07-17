## 1. Spike (DONE) / reproduction

- [x] 1.1 SPIKE: confirmed the native API/CLI exposes NO window list independent
  of AT-SPI (`platynui-cli window` → `//control:Window` via the AT-SPI tree;
  `_NET_CLIENT_LIST`/`_NET_SUPPORTING_WM_CHECK` internal-only). Recorded in
  design.md; basis for the guarded EWMH probe
- [x] 1.2 Confirm `runtime.providers()` output shape for reporting active
  providers in the diagnostic
- [x] 1.3 Capture the third-run evidence (window present, AT-SPI tree absent
  after refresh, `application_count: 0`, no calculations)

## 2. Guarded EWMH window-presence + native providers helper (D1)

- [x] 2.1 Add a `native_providers()` helper (native `runtime.providers()`) AND a
  guarded window-presence probe factored from the EXISTING
  `platynui_focus._x11_raise_by_pid` ctypes enumeration (match by
  `_NET_WM_PID`/`WM_CLASS`/`_NET_WM_NAME`); tri-state present/absent/unknown;
  never raises; annotated inline with the spike finding (no native equivalent)
- [x] 2.2 Reuse the existing ctypes surface — do NOT add a new ctypes module;
  swap for a native call if PlatynUI later exposes a window list
- [x] 2.3 Unit tests with the probe + providers mocked (present/absent/unknown)

## 3. Accessibility-exposure diagnostic (D2)

- [x] 3.1 In `get_ui_tree`, on the no-matching-app path, call the helper +
  `providers()`; add `accessibility_not_exposed` (window present) /
  `app_window_absent` (absent) / undetermined (unknown)
- [x] 3.2 Diagnostic payload: active providers, remediation, framed as
  accessibility/environment not locator
- [x] 3.3 Unit tests: window-present→exposed diagnostic; absent→window-absent;
  app-in-tree→no diagnostic; unknown→undetermined

## 4. EWMH WM-active probe — keep + annotate (D3)

- [x] 4.1 KEEP `desktop_display_safety._ewmh_wm_present` (no native equivalent
  per the spike); re-annotate it inline citing the spike finding; optionally
  consult `runtime.providers()` to ENRICH the report — never to replace the
  WM-active signal
- [x] 4.2 Preserve isolation-marker precedence, fail-closed-on-unknown, and the
  isolated/active outcomes (security-sensitive — no behavior change)
- [x] 4.3 Re-run ALL existing safety-classification tests unchanged

## 5. Refactor window raise to native-first (D4)

- [x] 5.1 In `platynui_focus.focus_window`, try native `Runtime.focus()` /
  WindowSurface `activate()` FIRST; reach the ctypes `_x11_raise_by_pid` only
  when the native pattern is unavailable; mark it a documented fallback
- [x] 5.2 Unit tests: native attempted first; ctypes fallback only when native
  unavailable; headless WM-less path still raises

## 6. Guidance + native-first rule (D5)

- [x] 6.1 PlatynUI guidance references the exposure diagnostic and states the
  native-first rule (use native Runtime/CLI; custom code only as documented
  fallback)
- [x] 6.2 Unit test: guidance documents the diagnostic + native-first rule

## 7. Validation + docs

- [x] 7.1 Full unit suite green; existing desktop + safety + focus flows
  unaffected; web/api/mobile untouched
- [x] 7.2 ADR mapping the third-run evidence + the native-first refactor;
  reference the PlatynUI new-core Runtime API + platynui-cli README
- [x] 7.3 Release note

## 1. Gate browser state on reality
- [x] 1.1 `bridge._merge_application_state`: only fetch/merge DOM+browser state when
  `session and session.is_browser_session()`; browser-less sessions surface no browser_type/current_url.

## 2. Honest platform
- [x] 2.1 Add `bridge._derive_platform(session)` (browser->web, Requests/REST->api, PlatynUI->desktop,
  Appium/mobile->mobile, else generic); set it on the serialized detail.

## 3. Crash + hardcode
- [x] 3.1 Fix the `None.lower()` crash in state_manager (`(url or "").lower()`) so no laundered
  about:blank / "Error Page" and no error-log spam.

## 4. Frontend
- [x] 4.1 Render the Browser / Current URL summary cells only when a real value is present.

## 5. Real end-to-end validation (different technologies)
- [x] 5.1 Drive real sessions and assert the bridge output is honest:
  - Browser (headless chromium): New Browser + New Page -> is_browser_session, real browser_type + current_url.
  - Selenium (headlesschrome, separate process): Open Browser -> real browser state.
  - Requests (RequestsLibrary): Create Session + GET -> platform=api, no browser fields.
  - BuiltIn/generic: Set Variable -> platform=generic, no browser fields.
  - RESTInstance / PlatynUI: skipped (not installed here) - recorded.
- [x] 5.2 `openspec validate --strict` passes. Real e2e PASSED across 4 technologies via equivalent
  driver scripts (BuiltIn->generic/no-browser, Requests->api/no-browser, Browser->web/chromium/real-url,
  Selenium->web/headlesschrome/real-url); committed as tests/integration/test_dashboard_browser_state_fidelity.py
  (runs in CI; the in-process pytest is resource-killed in THIS sandbox, so validation of record here is the
  e2e scripts). RESTInstance + PlatynUI: not installed in this env -> skipped/recorded.

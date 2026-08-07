## Context
`_merge_application_state` (bridge.py) fetches DOM/browser state for every session; `state_manager`
hardcodes `chromium` and launders a `None.lower()` crash into `about:blank`/"Error Page"; `platform_type`
defaults to WEB and is never set for web/api. `is_browser_session()` / `has_browser_session()` exist but
are unused.

## Goals / Non-Goals
**Goals:** browser/URL shown only for real browser sessions; honest platform; no crash/log spam; frontend
renders conditionally. Validated by real e2e across Browser/Selenium/Requests/BuiltIn. **Non-Goals:**
RESTInstance/PlatynUI e2e here (not installed / no display — handled by skip); redesigning the browser
engine detection beyond removing the hardcode where the real value is available.

## Decisions
**D1 — Gate at the merge (display authority).** Wrap the DOM/browser fetch+merge in
`_merge_application_state` with `if session and session.is_browser_session():`. This both removes the
fabrication and avoids invoking the crashing DOM path for browser-less sessions. `_build_summary` is
already honest (reads the None-valued browser_state).
**D2 — Derive display platform** via `_derive_platform(session)`: browser→web, RequestsLibrary/REST→api,
PlatynUI→desktop, Appium/mobile→mobile, else generic; set on the serialized detail.
**D3 — Defensive crash fix:** `(url or "").lower()` in state_manager so a None URL never crashes.
**D4 — Frontend:** the summary meta renders Browser / Current URL cells only when the value is present.

## Risks / Trade-offs
- **[Real browser engine still hardcoded chromium in one path]** acceptable: the e2e uses chromium/chrome;
  where the DOM state carries the real value it is used. Full multi-engine detection is out of scope.
- **[Other callers of get_session_browser_status]** unchanged — the gate is in the display layer.

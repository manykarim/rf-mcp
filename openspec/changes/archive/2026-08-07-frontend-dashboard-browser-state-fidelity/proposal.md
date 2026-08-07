## Why

The dashboard fabricates browser/platform identity for sessions that never used a browser (surfaced
during manual validation, 2026-08-07). A pure `BuiltIn`/`Collections` session (`checkout-flow-demo`,
6 non-browser steps, no web library) is shown as **Platform: Web · Browser: Chromium · Current URL:
about:blank** — none of which reflect reality. Verified mechanism:

- `_build_summary` is honest (yields `None`), but `_merge_application_state` (bridge.py:147-164)
  unconditionally fetches DOM/browser state; `state_manager._get_browser_library_state` **hardcodes**
  `browser_type="chromium"` (state_manager.py:698) and `current_url` falls back to `"about:blank"`
  (:700), gated only on a non-error status — and `get_session_browser_status` returns `success` for
  *every* session (no `is_browser_session()` check).
- `"about:blank"` is a **laundered crash**: with `url=None`, `state_manager.py:758` does `None.lower()`
  → `AttributeError` → the `except` returns `PageState(url="about:blank", title="Error Page")`. This logs
  *"Error converting browser state"* on **every** detail fetch of a browser-less session (58 in the live
  log).
- `platform_type` defaults to `WEB` (session_models.py:200) and is only ever set for DESKTOP/MOBILE, so
  even an API/Requests session reports "Web" — a fall-through, not a determination.

Both guard predicates (`ExecutionSession.is_browser_session()`, `BrowserState.has_browser_session()`)
exist and are unused end-to-end.

## What Changes

- **Gate browser state on reality:** in `_merge_application_state`, only fetch/merge DOM+browser state
  when `session.is_browser_session()`. Browser-less sessions surface no `browser_type`/`current_url`.
- **Derive an honest platform for display:** web when a browser session, `api` for Requests/REST,
  `desktop` for PlatynUI, `mobile` for Appium/mobile, else `generic` — instead of the Web fall-through.
- **Fix the `None.lower()` crash** in `state_manager` so no laundered `about:blank`/`Error Page` and no
  error-log spam.
- **Frontend:** render the Browser / Current URL summary cells only when a real value is present.

## Capabilities

### Modified Capabilities

- `frontend-dashboard`: add a requirement that browser/platform metadata reflects the session's actual
  automation technology (no fabricated browser state for non-browser sessions).

## Impact

- `src/robotmcp/frontend/bridge.py` (gate merge on `is_browser_session()`, derive platform),
  `src/robotmcp/components/state_manager.py` (crash fix + stop hardcoding), `static/frontend/app.js`
  (conditional browser cells). Validated by real end-to-end sessions across Browser, Selenium, Requests,
  and BuiltIn technologies.

## 1. Already-landed bug fixes (from the same round)

- [x] 1.1 Fix `Dialogs`/`STDLIBS` frozenset subscript in keyword_matcher fallback (+ test).
- [x] 1.2 Fix `import_library(notify=True)` — removed the RF-7.4-unsupported kwarg in library_manager (+ test).

## 2. Logging hygiene — expected failures

- [x] 2.1 In `rf_native_context_manager`, the "RF native execution traceback:" / generic-keyword-failure paths log a single-line WARNING/ERROR summary; move `traceback.format_exc()` to `logger.debug`.
- [x] 2.2 In `keyword_executor`, the expected-failure error path (unknown keyword, unresolved variable, HTTP errors surfaced to the client) keeps a one-line summary; full stack goes to DEBUG.

## 3. Logging hygiene — lazy-bootstrap + fallback noise

- [x] 3.1 Downgrade/reword the lazy-context messages: "No active RF execution context for X", "Failed to register X in RF context for keyword: Y" (keyword_executor / library_manager) to DEBUG or "(will import on demand)", keeping a WARNING only if the on-demand import then fails.
- [x] 3.2 Reword "BuiltIn library import failed during context creation" (rf_native_context_manager) as a recoverable fallback attempt at DEBUG.
- [x] 3.3 Optional-library-not-installed fallback ("Fallback loading failed for library X: No module named 'X'") logs at DEBUG/INFO, not WARNING (keyword_matcher).
- [x] 3.4 De-duplicate the keyword-shadowing notice so it is logged once per library load (keyword_discovery).

## 4. Page-source gating

- [x] 4.1 In the page-source service, short-circuit when the session has no web/browser library (SeleniumLibrary/Browser/AppiumLibrary) loaded — return a "no DOM (non-web session)" result instead of iterating DOM keywords. Evaluate the loaded-library set at call time.

## 5. Assignment heuristic + Requests alias

- [x] 5.1 Add RequestsLibrary response keywords (`GET/POST/PUT/DELETE/PATCH/HEAD/OPTIONS On Session`, and bare `GET/POST/...`) to the returnable-keyword allowlist so the assignment heuristic stops false-warning on `${resp}= GET On Session`.
- [x] 5.2 Normalize the library name `Requests` → `RequestsLibrary` at the session-init resolution boundary; log the correction at WARNING (not an ERROR-level import failure).

## 6. Requests JSON-body guidance

- [x] 6.1 Add a JSON-request-body pattern to `utils/requests_guidance.py` (single source): the inline-eval form `json=${{ {...} }}` and the define-`${body}`-before-POST ordering, surfaced via `get_locator_guidance(library="requests")`.

## 7. Tests

- [x] 7.1 Page-source gate: a Requests-only session returns the non-web short-circuit result and does NOT attempt browser DOM keywords (assert no DOM-keyword calls / no cascade).
- [x] 7.2 Assignment heuristic: `GET On Session` / `POST On Session` are treated as returnable (no false warning); a genuinely non-returning keyword still warns.
- [x] 7.3 Requests alias: initializing a session with library `Requests` resolves to `RequestsLibrary` and logs a WARNING-level correction.
- [x] 7.4 Requests guidance includes the JSON-body pattern (`json=${{` and the define-before-POST rule).
- [x] 7.5 Log-level regression: the downgraded lazy-bootstrap / expected-failure paths no longer emit at WARNING/ERROR at default level (assert via caplog), while a real unrecovered failure still does.

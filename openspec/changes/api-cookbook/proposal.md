# Proposal: api-cookbook

## Why

API/RequestsLibrary testing is the single worst turn-economy path in rf-mcp, and
the cause is a missing **proactive** guidance surface — the exact thing
`get_locator_guidance` provides for web.

- **Reconfirmed, quantified.** In the 2026-07-17 restful-booker eval
  (`experiments/EVAL_WEB_API_2026-07-17.md`) the agent achieved **6/6 CRUD
  assertions** (auth → create booking 3138 → GET → PUT → DELETE 201 → 404) but
  **hit max-turns (51) before writing its result.json / suite** — burned by
  **178 `Evaluate` calls + 43 `resp.json()`** routing every assertion through
  ad-hoc `Evaluate` because it never settled RequestsLibrary's response access.
  It also spent **5 `find_keywords`** (vs **0** on both web runs the same day).
  This is the *same* finding `experiments/EVAL_M27.md` flagged as an open
  follow-up ("~25 api calls", "a cookbook would roughly halve api turns") — now
  reproduced on a different API with harder evidence.

- **The guidance content already exists, but only REACTIVELY.** `utils/hints.py`
  already carries the right material — the `$var`-not-`${var}`-inside-Evaluate
  rule and `${resp.json()}` for method calls (`hints.py:947`), Create Session +
  Get/Post On Session shapes (`:993-1074`), named-args-as `name=value`
  (`:1067`), and POST/PUT/PATCH payload/header fixes on 400/415 (`:1087-1144`).
  But these fire **only after a failing step** — so a weak model must *trigger
  each error* to learn the pattern, one wasted turn at a time. Web does not work
  this way: `get_locator_guidance` (`server.py:6590`) lets an agent fetch
  locator guidance **upfront**, before interacting.

- **The web analog is a clean, established pattern.**
  `get_locator_guidance(library, error_message, keyword_name)` dispatches by
  library string to `RobotFrameworkNativeConverter.get_<lib>_locator_guidance()`
  (`rf_native_type_converter.py:1267/1318/1682`) for Browser / Selenium /
  Appium / PlatynUI. There is **no** `requests`/`api` branch — the one library
  where proactive guidance would help most.

Net: the API path is guidance-starved and error-reactive, so the model burns
its whole turn budget discovering what the codebase already knows.

## What Changes

- **Expose RequestsLibrary guidance PROACTIVELY through the existing
  `get_locator_guidance` tool.** Add a `requests` / `requestslibrary` / `api`
  branch to `get_locator_guidance` (`server.py:6590-6644`) that dispatches to a
  new `RobotFrameworkNativeConverter.get_requests_guidance(error_message,
  keyword_name)`, returning the same `{success, library, tips, warnings,
  examples}` shape the other guidance methods return. Reuse this tool rather
  than adding a new one (mirrors the web mechanism; no new tool surface).
- **Curate the cookbook content** from the material already in `hints.py` plus
  the pieces the evals proved missing — a small, ordered set of recipes (see
  `design.md`): session setup; response-object access (`${resp.json()["f"]}`,
  `${resp.status_code}`); the `$resp`-in-Evaluate vs `${resp.json()}`-elsewhere
  rule; **`Status Should Be`** as the native status assertion (instead of
  Evaluate); JSON body via `json=` + headers via `headers=`; the auth **`Cookie:
  token=<t>`** header pattern; and `expected_status=` for asserting non-2xx
  (404/error paths) without raising.
- **Steer API sessions toward it.** Update the `get_locator_guidance` docstring
  to advertise the `requests`/`api` option, and add a one-line pointer in the
  API-session surface (session-init guidance / recommender note) so an agent
  consults the cookbook before hand-rolling `Evaluate` assertions — the same way
  the web flow is steered to `get_locator_guidance`.

Out of scope: changing the reactive `hints.py` error-hints (kept — they remain
the on-failure safety net); a full separate API test-authoring profile; any
RequestsLibrary version pinning or new library dependency.

## Capabilities

### New Capabilities

- `api-cookbook`: `get_locator_guidance` accepts `library="requests"` (and
  `requestslibrary`/`api`) and returns a proactive RequestsLibrary
  request/response cookbook — session setup, response-field access, Evaluate
  semantics, native status assertions, JSON body/headers, auth token header, and
  non-2xx assertion — so an agent gets the patterns upfront instead of
  rediscovering them through failed `Evaluate` steps.

### Modified Capabilities

- None (guidance content is additive; the existing reactive `hints.py` behavior
  and the other `get_locator_guidance` libraries are unchanged).

## Impact

- `src/robotmcp/server.py:6590-6644` — add the `requests`/`requestslibrary`/`api`
  branch + docstring advertisement.
- `src/robotmcp/utils/rf_native_type_converter.py` — new
  `get_requests_guidance(error_message, keyword_name)` method (mirrors
  `get_browser_locator_guidance` shape); may factor shared recipe text from
  `utils/hints.py`.
- API-session steering surface (session-init guidance / recommender note) — one
  pointer line to the `requests` cookbook.
- Tests: `tests/unit/` — `get_locator_guidance(library="requests")` returns the
  cookbook (asserts the key recipes present: `${resp.json()}` access,
  `Status Should Be`, `Cookie: token=`, `expected_status=`); alias resolution
  (`api`/`requestslibrary`); shape parity with the other guidance libraries;
  unknown library still errors as before.
- Docs/eval: a follow-up restful-booker docker run should complete WITH
  artifacts (result.json + suite) inside the turn budget — the acceptance signal.

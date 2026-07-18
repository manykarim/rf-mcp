## Context

Findings from `experiments/uv-tool-install/FINDINGS_ROUND2.md` (stderr forensics
over 4 agent runs + a deterministic deep probe). Verified facts this design relies
on:

- The stdio JSON-RPC channel stays pure (0 stray) even under tracebacks — so all
  this output is on stderr, and relabeling/trimming it changes only log noise.
- Expected failures (404 HTTPError, unknown keyword `DataError`, unresolved
  `${var}` `VariableError`) are already caught and returned to the client as
  structured error results; the traceback dump to stderr is redundant.
- The page-source service runs even for API/Requests-only sessions, attempting
  browser DOM keywords (Get Page Source/Source/Url/Location/Window Url/Title/
  Window Title) that don't exist there → 7 DataError tracebacks per state read.
- The return-value assignment heuristic's allowlist omits RequestsLibrary
  response keywords, so `${resp}= GET On Session` triggers a false "may not return
  a useful value" WARNING.

## Goals / Non-Goals

**Goals:** quiet expected/recovered paths to their true severity; stop the API
page-source cascade; remove the false assignment warning; smooth the
`Requests`→`RequestsLibrary` mistake; close the JSON-body guidance gap.

**Non-Goals:** never suppress a genuine, unrecovered error; no change to tool
schemas or execution; do not touch FastMCP-owned banner/rich-traceback output
beyond existing env toggles.

## Decisions

1. **Level, don't delete.** For expected keyword-execution failures, keep one
   `logger.warning`/`error` line with the exception's short message and move the
   `traceback.format_exc()` payload behind `logger.debug`. This preserves triage
   info at DEBUG while cutting the default noise. Applied in
   `rf_native_context_manager` (the "RF native execution traceback:" dumps) and
   the keyword_executor error path.

2. **Reword lazy-bootstrap logs.** "No active RF execution context for X",
   "Failed to register X in RF context for keyword: Y", and "BuiltIn library
   import failed during context creation" describe a try→fallback that succeeds.
   Downgrade to `logger.debug` (or reword to "…(will import on demand)"), keeping
   a WARNING only if the fallback itself fails.

3. **Gate page-source on loaded web libraries.** In the page-source service, if
   the session has no web/browser library loaded (SeleniumLibrary/Browser/
   AppiumLibrary), return an empty/"no DOM (non-web session)" result immediately
   instead of iterating DOM keywords. Decide via the session's library set /
   `session_type`, not by trying and catching 7 failures.

4. **Extend the returnable-keyword allowlist** with RequestsLibrary response
   keywords (`GET/POST/PUT/DELETE/PATCH/HEAD/OPTIONS On Session`, plus the
   non-session `GET/POST/...`), so the assignment heuristic recognizes them as
   value-returning. Keep the heuristic advisory.

5. **Alias `Requests` → `RequestsLibrary`** at the library-resolution boundary
   used by session init, and log the correction at WARNING (not an ERROR-level
   ModuleNotFoundError). This is a name normalization, not a new library.

6. **Cookbook JSON-body pattern.** Add to the requests guidance (single source
   `utils/requests_guidance.py`) the working inline-eval body form
   `json=${{ {"k": "v", "n": 1, "b": True} }}` and the define-`${body}`-before-POST
   ordering, with a one-line "build the dict first, then pass json=" rule.

## Risks / Trade-offs

- **Hiding a real error by over-downgrading.** Mitigate: only downgrade paths
  proven to recover (there is a subsequent success log) or that are already
  returned to the client as structured errors; leave the summary line at
  WARNING/ERROR. Never downgrade a path with no recovery.
- **Page-source gate false-negative.** If a session loads a web library late, the
  gate must re-evaluate per call (check current loaded libs at call time, not
  cached at init). Low risk — it reads the live library set.
- **Alias surprise.** Aliasing `Requests`→`RequestsLibrary` could mask a user who
  genuinely has a third-party `Requests` library. Acceptable: RequestsLibrary is
  the RF-ecosystem meaning of "Requests", and the correction is logged.
- **Log-assertion tests.** Some existing tests may assert on the exact WARNING
  strings being downgraded; adjust those to the new level/text.

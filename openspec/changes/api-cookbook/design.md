# Design: api-cookbook

## Integration decision — reuse `get_locator_guidance`, don't add a new tool

Three options were considered:

1. **Extend `get_locator_guidance` with a `requests` branch** (chosen).
2. A new dedicated `get_api_guidance` tool.
3. An init-time API guidance bundle (like `desktop_guidance`) always injected.

**Chosen: (1).** The web flow already trains agents to call `get_locator_guidance`
before interacting; the tool already dispatches by library string
(`server.py:6613-6644`) to per-library converter methods. Adding a `requests`
branch reuses the exact mechanism, adds **zero new tool surface** (important for
small-context models and the tool-profile token budget), and keeps one guidance
entry point. (2) fragments the surface and costs profile tokens; (3) injects
tokens into every API session even when unneeded and duplicates the reactive
`hints.py` net. The cookbook stays **pull, not push** — matching web.

Steering (so agents actually pull it) is via the `get_locator_guidance` docstring
advertising `requests`/`api`, plus a one-line pointer in the API-session surface.

## Content shape (mirrors the other guidance methods)

`get_requests_guidance(error_message, keyword_name)` returns the same dict shape
as `get_browser_locator_guidance`:

```
{ "success": true, "library": "RequestsLibrary",
  "tips": [ ... ordered recipe strings ... ],
  "warnings": [ ... the gotchas ... ],
  "examples": [ {"keyword": "...", "arguments": [...], "note": "..."} ... ] }
```

`error_message`/`keyword_name` may tailor ordering (e.g. surface the 400/415
payload recipe first when the error is a 400) but the full cookbook is always
returned — the point is proactive coverage.

## The cookbook recipes (the 6–8 entries that remove F-API1)

1. **Session setup.** `Create Session   api   https://host` then
   `GET/POST/PUT/PATCH/DELETE On Session   api   /relative/path`. Prefer the
   sessionful form; a bare `GET`/`POST` takes a full URL. (from `hints.py:993-1074`)
2. **Response object access.** On-Session keywords RETURN the response; capture
   with `${resp}=`. Read status via `${resp.status_code}`, JSON body via
   `${resp.json()}`, a field via `${resp.json()["bookingid"]}`.
3. **Evaluate semantics (the tarpit).** Inside `Evaluate` use the **bare** name
   `$resp` (e.g. `Evaluate   $resp.json()["firstname"]`); OUTSIDE Evaluate use
   `${resp.json()}`. Do NOT reach for `Evaluate` to compare status/fields — use
   the native keywords below. (from `hints.py:947`)
4. **Native status assertion.** `Status Should Be   200   ${resp}` — not an
   `Evaluate` equality on `${resp.status_code}`. This is the single biggest
   Evaluate-call remover.
5. **JSON body + headers.** Send a dict body with `json=${body}` (not `data=`)
   and headers with `headers=${headers}`; a 400/415 usually means `data=` was
   used or `Content-Type: application/json` is missing. (from `hints.py:1087-1144`)
6. **Auth / token header.** Capture the token from `${resp.json()["token"]}`,
   then pass it as a cookie header: `headers=${{"Cookie": "token=" + $token}}`
   (or a `&{headers}` dict). Required for PUT/DELETE on restful-booker.
7. **Asserting non-2xx (404/error paths).** Pass `expected_status=404` (or
   `expected_status=anything`) to the On-Session keyword so an error response is
   returned instead of raising — needed for "GET a deleted resource → 404".
8. **Named args.** RequestsLibrary named args are `name=value` positionals
   (e.g. `expected_status=200`, `json=${body}`), not RF `&{dict}` kwargs.
   (from `hints.py:1067`)

## Reuse vs duplication

Recipes 1, 3, 5, 8 have source text in `hints.py` (reactive). The converter
method should factor that shared text (a small module-level constant reused by
both the reactive hint and the proactive cookbook) so the two never drift.
Recipes 4, 6, 7 are the eval-proven additions not currently surfaced anywhere.

## Risks / boundaries

- **Steering is advisory.** The cookbook only helps if the agent pulls it; the
  docstring + one pointer line are the nudge, matching how web already works
  (acceptable — web proves the pull model works).
- **No behavior change to execution.** This is guidance-only; it cannot regress
  any keyword execution or the reactive hints.
- **Acceptance is empirical.** The real proof is a follow-up restful-booker
  docker run completing WITH artifacts inside the turn budget (the "6/6 but
  out of turns" run turning green).

## Naming decision — keep `get_locator_guidance` (do NOT rename)

The cookbook is exposed through `get_locator_guidance(library="requests")`, which
is a misnomer for API (there are no "locators" in RequestsLibrary). This raised a
fair question: does the web-centric NAME stop agents — especially weak models —
from selecting the tool for an API task? The concern was that weaker models pick
tools by NAME (skimming) rather than by DESCRIPTION, so "locator" would steer
them away from the API cookbook — precisely the models that need it most.

**Decision: keep the name. The concern is refuted by evidence.** Two docker spikes
(2026-07-17, neutral prompt with NO mention of the cookbook / Evaluate / native
keywords) measured whether agents self-serve `get_locator_guidance(library=
"requests")` unprompted, and whether the pull is PROACTIVE (before the first
`execute_step`):

| Model | Replicates | Proactive self-serve | Notes |
|---|---|---|---|
| MiniMax-M3 | 3 | **3/3** | all before execution |
| MiniMax-M2.7 | 2 | **2/2** | success + artifacts |
| MiniMax-M2.5 | 2 | **2/2** | 1 max-turns, but still discovered it proactively |
| MiniMax-M2 (weakest) | 2 | **2/2** | success; tool-calls fine (execute_batch-heavy), cheapest |

**9/9 across the entire model ladder (M3 → M2.7 → M2.5 → M2)** pulled the cookbook
proactively. The weakest model (M2) did so in both runs. Conclusion: agents select
this tool by its DESCRIPTION (the docstring already advertises `library="requests"`
/ `api`), not by the "locator" token in its name — so the name is cosmetic, not a
functional barrier. A hard rename would cost an ~78-file ripple (incl. instruction
templates, tool profiles, tests, external prompts, archived specs) for **zero
measured agent-behavior gain**, so it is out of scope.

Residual (optional, cosmetic only — NOT required): a library-agnostic first line
in the docstring and updating the 2–3 instruction templates that still frame the
tool as "for locator syntax" would help HUMAN readers; neither is needed for agent
behavior. Reserve any rename for future evidence that the name actually blocks
tool selection (none exists today).

# Critical Validation — RobotMCP Dashboard ("Command Center")

**Scope:** the optional Django frontend at `src/robotmcp/frontend/` (started via `--with-frontend` /
`ROBOTMCP_ENABLE_FRONTEND=1`, or `python -m robotmcp.frontend.devserver`). A live observability +
control panel for Robot-Framework MCP sessions: session list, per-session summary/variables/steps, a
generated RF suite preview, and a live SSE event feed.

**Date:** 2026-08-07 · **Methodology:** codebase analysis + live browser inspection + multi-agent
adversarial validation. This report is the input to the remediation OpenSpec changes (see §7).

---

## 1. Executive summary

The dashboard is **visually polished and well-structured** — a clean, dark, card-based layout that is
responsive to mobile, has a decent semantic/a11y baseline (landmarks, `lang`, labelled controls), and
renders a correct, syntax-highlighted RF suite. It looks finished.

It is **not** finished. Beneath the surface sit **four critical defects** — two of which mean the
product's headline capability (watching an agent drive a test *live*) **does not work against a real
server at all**, and two of which are **critical, unauthenticated security holes** in the shipped
Docker image. Behind them are ten high-severity correctness/UX/a11y defects in the core flow.

| Severity | Count | One-line theme |
|---|---:|---|
| 🔴 Critical | 4 | Live feed is a frozen snapshot; unauth RCE; DEBUG/secret leak; frontend never binds under HTTP |
| 🟠 High | 10 | Stored XSS, broken empty state, keyboard lockout, focus/scroll destroyed on refresh, wrong labels, silent failures |
| 🟡 Medium | 16 | Session-switch races, mobile clipping, base-path 404, weak error contract, aria-live spam, no SRI/CSP |
| 🔵 Low | 4 | Timestamp inconsistency, list-value chips, custom fonts never load, icon collision |
| ⚪ Polish | 3 | Empty-state guidance, reduced-motion, dead code |
| **Total** | **37** | (deduped from 91 raw findings) |

**Headline verdict:** treat the dashboard as an **unshipped prototype**. Do not expose it on a network
(the Docker default binds `0.0.0.0:8001` with `DEBUG=True`, an unauthenticated arbitrary-keyword
execution endpoint, and `ALLOWED_HOSTS=["*"]`). Its live features need a real fix before the UI can be
trusted to reflect reality.

---

## 2. Methodology (evidence-based, multi-agent)

1. **Codebase analysis** — read the full frontend (`layout.html`, `index.html`, `app.js` 1638 lines,
   `base.css` 612 lines, `api.py`, `bridge.py`, `controller.py`, `config.py`, `django_app.py`,
   `devserver.py`, `urls.py`, `views.py`) plus the `--with-frontend` wiring in `server.py`.
2. **Live inspection** — ran the dashboard (`devserver`, seeded demo session) and drove it with real
   Chrome via Playwright: 4 full-page screenshots (desktop landing, session detail, suite preview,
   mobile), console/network capture, CDN-load capture, a11y probes, and end-to-end journeys
   (`scratchpad/ui_review/`).
3. **Fable 7-dimension critical validation** — parallel subagents (visual, UX, functionality,
   accessibility, security, frontend code quality, backend architecture) → **91 raw findings** → a
   single adversarial **verify + dedupe** pass → **37 verified** (each CONFIRMED / PLAUSIBLE, re-ranked
   by real impact).
4. **Cross-agent validation** — independent `codex` and `opencode` CLI reviews.
   - `codex` (timed out before a final list, having wandered into a GitHub-MCP fetch loop) still
     independently surfaced **stored XSS via `innerHTML`** and **no stale-response guard on session
     loads** — corroborating findings #6 and the session-switch race.
   - `opencode` failed on a model-configuration error (`anthropic/claude-sonnet-4-5` not resolvable);
     no findings produced. *(Tooling friction, honestly reported — not a coverage claim.)*

---

## 3. Critical findings (4)

### C1 — The Live Event Feed cannot work against a real server *(confirmed, backend)*
Three individually-fatal defects on one path (`api.py:126-176`):
1. `events_stream` returns a `StreamingHttpResponse` wrapping a **sync** generator. Under uvicorn/ASGI
   (the only way it is served), Django materializes the whole sync iterator before yielding — an
   infinite `while True` never emits a byte.
2. `EventBus.publish_sync` schedules delivery on the **publisher's** loop, but the SSE worker's
   `asyncio.Queue` lives on a **different** private loop; the cross-loop `put_nowait` is a silent
   no-op (reproduced: 3 events published → `RECEIVED: []`). Every real publisher is on the MCP loop.
3. Each connection leaks 2 threads + a never-discarded subscriber queue; `EventSource` auto-reconnect
   makes an idle tab bleed threads/memory inside the MCP process.

**Impact:** steps/variables/suite/feed are a **frozen one-shot snapshot** taken at page load. A user
watching an agent run sees nothing move and cannot tell that from "nothing is happening." There is no
polling fallback (`scheduleSessionRefresh` is only triggered by SSE messages).
**Fix:** rewrite `events_stream` as an **async** view over an async generator iterating
`event_bus.subscribe()` on the ASGI loop; make `EventBus._deliver` use
`owner_loop.call_soon_threadsafe(...)` for cross-loop delivery; emit `:` ping keepalives; add a
client-side 5–10 s recent-events poll so a dead stream degrades instead of freezing.

### C2 — Unauthenticated arbitrary RF keyword execution (RCE) on `0.0.0.0:8001` *(confirmed, security)*
`POST /api/sessions/<id>/execute/` (`urls.py:29`, `api.py:98`, `bridge.py:667`) runs **any** RF
keyword — `Evaluate` is arbitrary Python; `Run Process`/`OperatingSystem.*` are shells. No auth
anywhere (`django_app.py:37` is the stock middleware stack). `docker/Dockerfile:76` +
`supervisord.conf:37` ship `--frontend-host 0.0.0.0` with `EXPOSE 8001`. **The SPA never calls this
endpoint** (`grep execute app.js` → 0) — it is pure attack surface with zero user benefit. CSRF is not
access control.
**Impact:** full unauthenticated RCE as the container/host user for anyone who can reach the port.
**Fix:** delete the route until it has both a UI and an auth story; if kept, gate all mutating routes
behind a mandatory shared-secret, default-refuse on non-loopback binds, and allow-list keywords.

### C3 — `DEBUG=True` default in the shipped Docker path + `ALLOWED_HOSTS=["*"]` *(confirmed, security)*
`FrontendConfig.debug` defaults `True` (`config.py:35`, and the `_env_bool` fallback `config.py:80`);
the Docker command passes no `--frontend-no-debug`, so **DEBUG=True in production**, bound to
`0.0.0.0`. `ALLOWED_HOSTS=["*"]` (`django_app.py:27`) disables host validation; no CSP anywhere;
`SECRET_KEY` is regenerated per process.
**Impact:** any unhandled exception (several are reachable) returns Django's technical 500 page to an
unauthenticated remote client — **SECRET_KEY, the full settings dict, all environment variables
(incl. `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`), local session data, and source excerpts**. `*` also
enables DNS-rebinding.
**Fix:** default `debug=False` (require explicit opt-in); set `ALLOWED_HOSTS` from the configured host;
add a startup guard that refuses `DEBUG=True` on a non-loopback bind.

### C4 — `--transport http --with-frontend` never binds the frontend port *(confirmed, backend)*
The frontend is attached as a **per-MCP-session FastMCP lifespan**, so with HTTP transport the uvicorn
frontend server never starts until an MCP client connects — and is torn down when that session ends.
Empirically reproduced: `:8011` never listened across two clean starts.
**Impact:** the documented Docker deployment is broken — `docker run -p 8001:8001` is connection-refused
until an agent connects, then dies whenever an agent's session ends (which agents do constantly).
**Fix:** for non-stdio transports, start the controller eagerly in `main()` before `mcp.run()` and stop
it in the existing `finally`; or refcount it so it survives individual session churn.

---

## 4. High findings (10)

| # | Finding | Where | Fix in brief |
|---|---|---|---|
| H1 | **Zero-sessions path throws a swallowed `TypeError`** (`state.sessionPanel` doesn't exist) → dead session pane with live buttons after the last session closes | app.js | Use `elements.sessionPanel`/`sessionActions`; null out session state in the empty-state renderer |
| H2 | **Stored XSS** — variable names/values scraped from the page-under-test are written via `innerHTML`; same-origin as the RCE endpoint + a readable csrftoken = an RCE chain | app.js (multiple `innerHTML`) | Replace interpolating `innerHTML` with DOM construction (`iconLabel()` helper); add CSP |
| H3 | **Variables panel hides every ALL-CAPS RF variable** (`${BASE_URL}` etc.) via a `/^[A-Z_]+$/` filter and shows internal plumbing keys + a wrong count | app.js:166 | Delete the regex branch; have the bridge return namespaced `{user, meta}`; read the live RF namespace |
| H4 | **Suite preview mutates the live shared session** and injects a synthetic session into the running `SessionManager` — from an unauthenticated GET → recorded steps vanish | bridge/TestBuilder | Build the preview against a detached `ExecutionSession`; never register/mutate the live one |
| H5 | **Keyboard lockout** — the session list is not focusable and the stylesheet has **zero focus indicators**; the core flow is impossible without a mouse (WCAG 2.1.1/4.1.2/2.4.7) | index.html + base.css | `role=listbox/option`, `tabindex`, roving focus + Enter/Space; global `:focus-visible` outline |
| H6 | **Every render wipes `innerHTML`**, so an SSE refresh destroys focus, scroll, and any in-flight drag (and can throw mid-drag); pressing ▲ destroys the button pressed | app.js | Guard refresh during drag; harden the placeholder teardown; move to keyed/diff updates |
| H7 | **Blind `.slice(0,8)`/`.slice(0,6)`** on identifiers → wrong session name ("Session frontend" for `frontend-demo`) and five identical step chips; prefix-colliding ids become indistinguishable | app.js:290/1028/1263/1418 | `shortId()` with middle-ellipsis + full `title`; do not truncate the `<h2>` |
| H8 | **Libraries chip reports libraries never imported**, from two contradicting hardcoded heuristic tables (fabricates e.g. SeleniumLibrary → wrong locator dialect) | app.js:389-433, bridge | `imported_libraries` = exactly that, deduped; separate `search_order`; delete the heuristics |
| H9 | **Every failure is silent** — four console-only `catch` blocks, a `Promise.all` that blanks the whole pane, no connection-state indicator: dead state reads as live state | app.js | Connection pill (`onopen`/`onerror`); `Promise.allSettled` + per-panel inline errors |
| H10 | **Suite Preview goes stale** after edit/disable/reorder and **Copy hands over the wrong suite**; overrides discarded on session-switch without warning | app.js | Mark preview out-of-date + disable Copy while pending (or debounced auto-regen); confirm before clearing overrides |

---

## 5. Medium / Low / Polish (23)

**Medium (16):** step editing comma-splits arguments through blocking `prompt()`s (corrupts locators);
session-switch race renders one session's data under another's header; mobile suite header clips the
**Generate** button inside `overflow:hidden`; sticky sidebar overflows its own `max-height` with
`overflow:visible` (strands content); duplicate events + duplicated/leaked `imported_libraries` at the
source; event feed renders in two conflicting orders with a too-weak dedupe key; `--frontend-base-path`
is non-functional (app 404s at its own advertised URL); pass/fail has no text alternative and
non-outcome events are painted success-green; `aria-live` wraps fully-rebuilt containers (re-announces
the entire list on every event); CDN scripts with **no SRI**, unpinned `latest` feather-icons, no
offline fallback, no CSP; `touch-action:none` on the whole step card blocks touch scrolling; API error
contract is unreliable (200 for missing sessions/failed previews, an HTML 404 elsewhere); frontend
asyncio primitives shared across independent loops; content clipped with a keyboard-unreachable scroll
affordance; narrow-viewport source order buries live content ~2 screens down; no JS test infrastructure
and the browser tests self-skip in CI.

**Low (4):** raw microsecond ISO timestamp in the heading + three formats for one value;
assigned-variable chips show the wrong value when a keyword returns a list; declared type identity
(Inter/JetBrains Mono) is named but never `@font-face`-loaded; event-row clock icon collides with the
title on every row.

**Polish (3):** first-run empty state offers no guidance and no `prefers-reduced-motion`; compact mode
stacks the event icon (less dense); dead code (a handler bound to a removed element, unused state).

---

## 6. What's genuinely good (keep it)

- Clean, consistent **dark card design** with clear panels and good use of accent color.
- **Responsive** — the two-column desktop layout collapses to a readable single column on mobile with
  no horizontal overflow.
- **Semantic/a11y baseline** present: `main`/`nav`/`header` landmarks, `lang="en"`, labelled controls,
  `aria-live` regions (even if misapplied), alt text.
- Correct, **syntax-highlighted** RF suite generation (Settings/Test Cases/VAR/teardown).
- Thoughtful affordances exist in intent (step reorder/disable/edit, copy, compact mode) — they just
  need to be made correct, accessible, and non-destructive.

---

## 7. Remediation roadmap → OpenSpec change groups

The 37 findings group into **five coherent changes** (ordered by priority). Each will run
explore → propose → apply → archive.

1. **`frontend-dashboard-security-hardening`** *(C2, C3, H2, + CDN/SRI/CSP)* — flip DEBUG default,
   scope ALLOWED_HOSTS, refuse non-loopback+DEBUG, remove/authn the exec endpoint, escape all
   `innerHTML`, add CSP + SRI (or vendored assets). **Highest priority.**
2. **`frontend-dashboard-live-runtime`** *(C1, C4, + base-path 404, API error contract, shared loops)*
   — make the live feed actually stream (async SSE + cross-loop delivery + keepalive + poll fallback),
   start the frontend eagerly under HTTP, fix base-path, standardize the error contract.
3. **`frontend-dashboard-rendering-correctness`** *(H1, H3, H4, H6, H7, H8, H10, + duplicate
   events/libraries, session-switch race, list-value chips, timestamps)* — the empty-state crash, the
   identifier truncation, non-destructive suite preview, keyed updates, honest labels/variables.
4. **`frontend-dashboard-accessibility`** *(H5, + status text alt, aria-live, touch-action, focus
   indicators, reduced-motion)* — keyboard operability and screen-reader correctness for the core flow.
5. **`frontend-dashboard-visual-ux-polish`** *(H9, + mobile clipping, sidebar overflow, fonts,
   empty-state guidance, connection state, dead code)* — surface failures, fix layout clipping, load
   the declared type, first-run guidance.

Full per-finding evidence and fixes: `scratchpad/ui_review/verified_findings.json`; screenshots:
`scratchpad/ui_review/shots/`.

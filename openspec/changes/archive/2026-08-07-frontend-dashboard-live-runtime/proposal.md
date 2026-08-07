## Why

Two **critical** runtime defects (validation 2026-08-07, `docs/frontend_dashboard_critical_validation.md`)
make the dashboard's headline capability — watching an agent drive a test *live* — non-functional
against a real server:

- **C1 — the Live Event Feed cannot work.** `events_stream` returns a `StreamingHttpResponse` over a
  **sync** generator; under ASGI Django materializes it before yielding, so it never streams. And
  `EventBus.publish_sync` schedules delivery on the *publisher's* loop while each subscriber `Queue`
  lives on a *different* loop — the cross-loop `put_nowait` is a **silent no-op** (reproduced:
  `RECEIVED: []`). Plus each connection leaks threads. Net: steps/variables/suite/feed are a frozen
  one-shot snapshot; there is no client poll fallback.
- **C4 — the frontend never binds under HTTP.** It is attached as a per-MCP-session FastMCP lifespan,
  so `--transport http --with-frontend` never starts the uvicorn frontend until a client connects, and
  tears it down when that session ends. The documented Docker deployment is connection-refused.

Plus two closely-related mediums: `--frontend-base-path` 404s at its own advertised URL, and the API
error contract is inconsistent (200 for missing sessions/failed previews).

## What Changes

- **Fix cross-loop delivery** in `EventBus`: record each subscriber's owning loop and deliver via
  `loop.call_soon_threadsafe(...)`, dropping subscribers whose loop is closed. `publish_sync` fans out
  directly (no wrong-loop `create_task`).
- **Rewrite `events_stream`** as an **async** view over an async generator iterating
  `event_bus.subscribe()` on the ASGI loop, with client-disconnect cleanup and a keepalive; delete the
  thread/queue/stop_flag machinery.
- **Client poll fallback** — poll recent events every ~5 s so a dead stream degrades instead of freezing.
- **Start the frontend eagerly for non-stdio transports** (before `mcp.run()`, stopped in the existing
  `finally`) so it is up independent of MCP-session churn.
- **Base-path + error contract** — serve the app under a configured base path; return proper status
  codes for missing sessions / failed operations.

## Capabilities

### Modified Capabilities

- `frontend-dashboard`: add a live-runtime requirement — the event stream actually streams cross-loop,
  the dashboard binds independent of MCP sessions, and the client degrades gracefully.

## Impact

- `src/robotmcp/core/event_bus.py` (cross-loop delivery), `frontend/api.py` (async SSE view),
  `server.py` (eager frontend start for http/sse), `frontend/static/frontend/app.js` (poll fallback +
  connection state), `frontend/urls.py`/`views.py` (base-path), `frontend/api.py` (error contract).

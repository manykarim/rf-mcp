## Context

`EventBus` (`core/event_bus.py`) fans out `FrontendEvent`s to SSE subscribers. `publish_sync` does
`loop.create_task(self._deliver(...))` on the publisher's loop, but subscriber queues live on the ASGI
loop → cross-loop `put_nowait` never wakes the getter. `api.events_stream` wraps a sync generator (never
streams under ASGI). The frontend is a per-session FastMCP lifespan (never binds under HTTP).

## Goals / Non-Goals

**Goals:** make the stream actually stream cross-loop; bind the frontend independent of sessions; add a
client poll fallback; fix base-path + error contract. **Non-Goals:** auth (done in security change);
rendering correctness / a11y / visual (separate changes); a full pub/sub broker.

## Decisions

**D1 — Owner-loop delivery.** `_subscribers` becomes `{queue: owner_loop}`; `_fanout(event)` schedules
`owner_loop.call_soon_threadsafe(_put, queue, event)` (drop-oldest on full; drop subscriber on closed
loop). `publish` and `publish_sync` both call `_fanout` — no wrong-loop `create_task`. Use a
`threading.Lock` (cross-loop safe) for the tiny replay/registry critical sections.

**D2 — Async SSE view.** `events_stream` becomes `async def` returning `StreamingHttpResponse` over an
async generator. Keepalive via a non-cancelling get-task race (a keepalive tick must not cancel the
subscription). Client disconnect → generator close → `subscribe()` finally discards the queue.

**D3 — Client poll fallback.** app.js polls `api/events/recent/` every ~5 s in addition to SSE, deduped,
so a dead stream still updates. (Reuses the existing recent-events endpoint.)

**D4 — Eager frontend for non-stdio.** In `server.py main()`, for `transport != stdio` start the
controller eagerly (`asyncio.run(controller.start())`, uvicorn runs in its own thread) before
`mcp.run()` and stop it in the existing `finally`; keep the lifespan only for stdio.

## Risks / Trade-offs

- **[Keepalive cancels the subscription]** avoided by racing a persistent get-task, not `wait_for(anext)`.
- **[Poll + SSE double-render]** deduped client-side by a stable event key.
- **[Eager start blocks main briefly]** `controller.start()` only waits for uvicorn `started`; fast.

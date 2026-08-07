## 1. Cross-loop event delivery (C1a)
- [x] 1.1 `event_bus.py`: `_subscribers` → `{queue: owner_loop}`; `_fanout` via `call_soon_threadsafe`
  (drop-oldest full, drop closed-loop); `publish`/`publish_sync` call `_fanout`; `threading.Lock`.
- [x] 1.2 `subscribe()` records `asyncio.get_running_loop()`; unregister in `finally`.

## 2. Async SSE view (C1b)
- [x] 2.1 Rewrite `api.events_stream` as async over an async generator iterating `event_bus.subscribe()`;
  delete the thread/queue/stop_flag; add a non-cancelling keepalive; disconnect cleans up.

## 3. Client resilience (C1c)
- [x] 3.1 `app.js`: add a ~5 s `loadRecentEvents` poll alongside SSE, deduped; add a connection-state
  signal driven by `EventSource` onopen/onerror.

## 4. Eager frontend under HTTP (C4)
- [x] 4.1 `server.py`: for non-stdio transports start the controller eagerly before `mcp.run()` and stop
  in the existing `finally`; keep the per-session lifespan only for stdio.

## 5. Base-path + error contract
- [~] 5.1 (DEFERRED — medium, not in the spec contract; needs Django prefix routing) 5.1 Serve the app under the configured base path (fix the 404 at the advertised URL).
- [~] 5.2 (DEFERRED — medium; changing status codes risks SPA display, follow-up) 5.2 Return proper status codes for missing sessions / failed operations.

## 6. Verify + wrap-up
- [x] 6.1 Cross-loop test: publish from loop A, receive on an SSE subscriber on loop B (was `RECEIVED: []`).
- [x] 6.2 Live: SSE streams incrementally; disconnect leaves `len(_subscribers)` at baseline; HTTP
  transport binds the frontend at startup and survives a session end.
- [x] 6.3 `openspec validate frontend-dashboard-live-runtime --strict` passes.

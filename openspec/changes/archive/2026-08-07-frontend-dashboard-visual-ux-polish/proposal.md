## Why

The dashboard fails silently and clips content (validation 2026-08-07): four console-only catch blocks
and a `Promise.all` that blanks the whole pane on any single failure, with **no connection-state
indicator** — so a dead server reads as live, frozen state. On mobile the suite header clips the primary
**Generate** button inside an `overflow:hidden`; the sticky sidebar overflows its own `max-height` with
`overflow:visible`, stranding content; the empty state offers no guidance; and dead code remains.

## What Changes

- Add a **connection-state pill** (Live / Reconnecting / Offline) driven by the SSE `onopen`/`onerror`.
- Switch the session-detail load to `Promise.allSettled` so one failed request degrades that region
  instead of blanking the pane, and surface failures.
- Fix layout clipping: `suite-header` wraps; the sidebar scrolls (`overflow-y:auto`).
- Add first-run empty-state guidance; remove dead code (a handler bound to a removed element).

Bundling the declared fonts (Inter / JetBrains Mono) is left as a follow-up — it needs font assets and
the current system-font fallback is functional.

## Capabilities

### Modified Capabilities

- `frontend-dashboard`: add a resilience/legibility requirement — failures are surfaced (incl. connection
  state), a single failed request does not blank the view, and core controls are not clipped.

## Impact

- `static/frontend/app.js` (connection state, `allSettled`, empty-state, dead code),
  `templates/frontend/layout.html` (pill), `static/frontend/base.css` (pill, sidebar/suite clipping).

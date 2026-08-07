## Context
`loadSessionDetails` uses `Promise.all` (one rejection throws → the caller's console-only catch blanks
the pane); there is no connection indicator; `.suite-header` clips on mobile; `.sidebar` uses
`overflow:visible` with a `max-height`; a dead `headerBuildSuite` handler references a removed element.

## Goals / Non-Goals
**Goals:** connection state, resilient load, no clipping, empty-state guidance, dead-code removal — all
verifiable. **Non-Goals:** bundling web fonts (needs assets; system fallback works) — follow-up.

## Decisions
**D1 —** a `.conn-pill` in the topbar, `updateConnectionState()` wired to SSE `onopen`/`onerror` and the
poll fallback. **D2 —** `Promise.all` → `Promise.allSettled` with per-request fallbacks + surfaced
rejections. **D3 —** `.suite-header { flex-wrap: wrap }`; `.sidebar { overflow-y: auto }`. **D4 —**
guidance text in the empty state; delete the dead handler.

## Risks / Trade-offs
- **[Fonts deferred]** system fallback is fine; bundling is a separate asset task.
- **[allSettled changes shape]** handled with a `pick(i, fallback)` helper.

## Why

The dashboard mis-renders core data (validation 2026-08-07,
`docs/frontend_dashboard_critical_validation.md`): identifiers are blind-truncated so a session named
`frontend-demo` shows as "Session frontend"; the Variables panel HIDES every ALL-CAPS Robot-Framework
variable (`${BASE_URL}`, `${CITY}`) — the common case — via a `/^[A-Z_]+$/` filter, while showing
internal plumbing; the Libraries chip fabricates libraries the session never imported; events and
`imported_libraries` are duplicated at the source; and the zero-sessions path throws a swallowed
`TypeError` (`state.sessionPanel` doesn't exist), leaving a zombie pane.

## What Changes

- Replace blind `.slice(0,8)`/`.slice(0,6)` with a `shortId()` middle-ellipsis helper; show the session
  title in full.
- Drop the ALL-CAPS variable-hiding branch so user variables are shown; rely on the explicit builtin list.
- Stop fabricating libraries from step heuristics; dedupe `imported_libraries` at the bridge.
- Remove the redundant `session_created` publish in the devserver (the session manager already emits it).
- Fix the empty-state crash (`state.sessionPanel`/`sessionActions` → `elements.*`).

Deeper correctness items (suite preview mutating the live session; keyed DOM updates so an SSE refresh
doesn't wipe focus/scroll; suite-stale-after-edit) are recorded as follow-ups — they need a TestBuilder
overload and a render-diff rewrite, out of scope for this correctness pass.

## Capabilities

### Modified Capabilities

- `frontend-dashboard`: add a data-fidelity requirement — identifiers, variables, and library/event data
  are rendered faithfully (no blind truncation, no convention-based hiding, no fabricated/duplicated data,
  no empty-state crash).

## Impact

- `static/frontend/app.js` (shortId, variable filter, empty-state, library heuristic),
  `frontend/bridge.py` (dedupe), `frontend/devserver.py` (double-publish).

## Context
`app.js` truncates ids with `.slice(0,N)`, hides ALL-CAPS variables with `/^[A-Z_]+$/`, merges a
`STEP_LIBRARY_HINTS` heuristic table, and references a non-existent `state.sessionPanel` on the empty
path. The devserver re-publishes `session_created` (also emitted by `session_manager.create_session`),
and `bridge` returns `imported_libraries` unduplicated.

## Goals / Non-Goals
**Goals:** faithful identifiers/variables/library/event rendering; no empty-state crash — all verifiable
via the live probe. **Non-Goals:** the suite-preview-mutates-live-session data-loss fix (needs a
TestBuilder overload), keyed DOM diffing (needs a render rewrite), suite-stale-after-edit — recorded as
follow-ups.

## Decisions
**D1 — `shortId(id,n)`** middle-ellipsis; full title for the H2 (it has room). **D2 —** delete the
ALL-CAPS hide branch; keep the explicit builtin denylist. **D3 —** replace `inferLibrariesFromSteps()`
with an empty set (stop fabricating); dedupe `imported_libraries` in the bridge. **D4 —** drop the
devserver's redundant publish. **D5 —** `state.sessionPanel/sessionActions` → `elements.*`.

## Risks / Trade-offs
- **[Removing library heuristics may under-report]** accepted — showing only truly-imported libraries is
  correct; the search-order path still contributes real data.
- **[Deep items deferred]** explicitly recorded; this pass is the high-value verifiable subset.

## 1. Faithful identifiers
- [x] 1.1 Add `shortId()`; replace the 4 `.slice(0,N)` sites; show the session title in full. (finding #1 + H7)

## 2. Variables + libraries
- [x] 2.1 Remove the `/^[A-Z_]+$/` hide branch so ALL-CAPS user variables show. (H3)
- [x] 2.2 Stop fabricating libraries (`inferLibrariesFromSteps` → empty); dedupe `imported_libraries` in the bridge. (H8)

## 3. Duplicates + empty state
- [x] 3.1 Remove the devserver's redundant `session_created` publish (source dedup).
- [x] 3.2 Fix the empty-state crash: `state.sessionPanel`/`sessionActions` → `elements.*`. (H1)

## 4. Deferred (recorded follow-ups)
- [~] 4.1 Suite preview mutates the live session (data loss) — needs a TestBuilder detached-session overload. (H4)
- [~] 4.2 Keyed DOM updates so an SSE refresh doesn't wipe focus/scroll/drag. (H6)
- [~] 4.3 Suite-stale-after-edit + Copy-hands-wrong-content signal. (H10)

## 5. Verify + wrap-up
- [x] 5.1 Live probe: title "Session frontend-demo"; CITY/ENV variables visible; 1 session_created event;
  `imported_libraries` deduped. PASS.
- [x] 5.2 `openspec validate frontend-dashboard-rendering-correctness --strict` passes.

## 1. Surface failures + connection state
- [x] 1.1 Add a connection-state pill (Live/Reconnecting/Offline) wired to SSE onopen/onerror + poll fallback. (H9)
- [x] 1.2 `Promise.all` -> `Promise.allSettled` with per-request fallbacks; surface rejections. (H9)

## 2. Layout + guidance + cleanup
- [x] 2.1 `.suite-header` wraps; `.sidebar` scrolls (`overflow-y:auto`) — fix mobile clipping / stranded content.
- [x] 2.2 First-run empty-state guidance text.
- [x] 2.3 Remove dead `headerBuildSuite` handler (element no longer exists).

## 3. Deferred (recorded follow-up)
- [~] 3.1 Bundle the declared fonts (Inter/JetBrains Mono) — needs assets; system fallback works today.

## 4. Verify + wrap-up
- [x] 4.1 Live probe: connection pill shows "Live" (SSE connected); Generate button visible + within the
  mobile viewport; desktop screenshot confirms the polished layout. PASS.
- [x] 4.2 `openspec validate frontend-dashboard-visual-ux-polish --strict` passes.

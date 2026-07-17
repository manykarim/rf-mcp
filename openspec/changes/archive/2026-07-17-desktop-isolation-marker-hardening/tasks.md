# Tasks: desktop-isolation-marker-hardening

## 1. Ownership corroboration
- [x] 1.1 Decide the ownership proof (design.md): rf-mcp-minted marker token matching the launched display, and/or process-tree ownership of the bound Xvfb/nested-X
- [x] 1.2 `_marker_ownership_verified(env, display)` helper; `_has_isolation_marker` remains the "named the display" check, ownership is the new gate on granting ISOLATED

## 2. Classification changes
- [x] 2.1 `classify_bound_display_detailed`: grant ISOLATED from marker only when ownership-corroborated; uncorroborated marker → `unknown` (fail-closed)
- [x] 2.2 When marker names the display AND EWMH probe reports an active-desktop-shaped WM that is not the expected isolated WM → `isolation_source = marker_over_active_wm`, classification `unknown` unless corroborated
- [x] 2.3 No marker → existing EWMH-probe behavior unchanged

## 3. Marker minting (keep the harness green)
- [x] 3.1 Set the corroboratable marker token wherever rf-mcp launches/adopts an isolated display
- [x] 3.2 Update `docker/entrypoint.sh` / `docker/claude_mcp.json` so the reference Xvfb :99 + fluxbox display still classifies ISOLATED

## 4. Tests
- [x] 4.1 Marker names the bound display but is uncorroborated → `unknown` (fail-closed), input refused
- [x] 4.2 Corroborated marker on a WM-owning (fluxbox) display → `isolated` (the legitimate isolated case is preserved)
- [x] 4.3 Marker + active-desktop-WM conflict → `marker_over_active_wm` recorded
- [x] 4.4 No marker → EWMH path unchanged (regression)

## 5. Docs
- [x] 5.1 `docs/desktop_docker_harness.md` isolation checklist gains the corroboration note

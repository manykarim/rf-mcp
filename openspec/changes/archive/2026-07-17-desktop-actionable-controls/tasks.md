# Tasks: desktop-actionable-controls

## 1. Collector — flat role-filtered walk (ui_tree_service.py)
- [x] 1.1 `get_actionable_controls(session, app_filters, roles, max_nodes, max_elements, time_budget_s)` async entry point + sync collector dispatched via asyncio.to_thread
- [x] 1.2 Sync DFS: resolve the anchor application (reuses /app:* listing + D4 display scoping), walk node.children() with node/element/time budgets
- [x] 1.3 Entry shape: role, name, app-anchored index-disambiguated descriptor (nameless controls get a positional descriptor), enabled/visible/bounds/depth
- [x] 1.4 Truncation: partial results + `truncated: {reason}` on any exhausted budget; per-node provider errors skipped, never raised

## 2. Scoping guarantees
- [x] 2.1 Anchor resolution: elements_of_interest filter → session desktop_aut_pid → single display-scoped app; >1 candidate with no filter → application list + hint, NO walk
- [x] 2.2 Reuses `_display_scoped_pids()` D4 probe; host apps never the anchor on isolation-marked displays
- [x] 2.3 Unit test: the collector only walks the anchor subtree (node.children() only), never a desktop-rooted // expression

## 3. Server wiring (server.py)
- [x] 3.1 `get_session_state`: `actionable_controls` section wired next to ui_tree, passing elements_of_interest as app filter; desktop-only via the collector's non-desktop rejection
- [~] 3.2 Per-call knobs: app filter wired via elements_of_interest; role/element-cap params use the collector defaults for now (not surfaced on the tool signature) — follow-up if needed
- [~] 3.3 get_session_state docstring: the ui_tree hint (4.1) advertises the section; full docstring paragraph is a follow-up

## 4. Discoverability
- [x] 4.1 ui_tree hint points to actionable_controls for enumerating interactive controls with ready descriptors
- [ ] 4.2 PlatynUI locator guidance cross-reference — DEFERRED (follow-up)

## 5. Tests + validation
- [x] 5.1 `tests/unit/test_desktop_actionable_controls.py` (fake node tree): depth-5 flat; descriptor shape + index disambiguation + nameless; role filter + default set; anchor precedence + multi-app refusal; node/element budget truncation; per-node error skipped; non-desktop rejection; only-walks-anchor
- [x] 5.2 Hint text assertion (4.1)
- [x] 5.3 Full unit suite green (6924 passed + 1 skipped); openspec validate --strict passes
- [ ] 5.4 (OPTIONAL) Docker lab e2e — DEFERRED

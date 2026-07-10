# Tasks: desktop-actionable-controls

## 1. Collector — flat role-filtered walk (ui_tree_service.py)
- [ ] 1.1 Add `_INTERACTIVE_ROLES` default set and `get_actionable_controls(session, app_filters, roles, max_nodes, max_elements, time_budget_s)` async entry point next to `get_ui_tree` (`ui_tree_service.py:477`), dispatching a sync DFS via `asyncio.to_thread` (design.md D1).
- [ ] 1.2 Implement the sync DFS: resolve the anchor application (reuse the `/app:*` listing + display scoping of `_collect_ui_tree_sync`, `ui_tree_service.py:295`; anchor precedence per design.md D3), walk `node.children()` depth-first with node/element/time budgets, collect entries whose role matches the filter.
- [ ] 1.3 Entry shape: `role`, `name`, `descriptor` (app-anchored, quoted-Name, index-disambiguated per design.md D4; nameless interactive controls get a positional descriptor), `enabled`/`visible` (from the existing `_NODE_ATTRIBUTES` reads), `bounds`, `depth`.
- [ ] 1.4 Truncation semantics: on any exhausted budget return partial results with `truncated: {reason}` and totals where known; per-node provider errors are skipped, never raised (spec req 3).

## 2. Scoping guarantees
- [ ] 2.1 Anchor resolution: `elements_of_interest` filter → session `desktop_aut_pid` (`session_models.py:172`) → single display-scoped app; with no anchor and >1 candidate, return the application list + "pass an application name" hint — no walk (spec req 2).
- [ ] 2.2 Reuse the D4 display-scoping PID probe unchanged (`_display_scoped_pids`, `ui_tree_service.py:42`); host apps never become the anchor on isolation-marked displays.
- [ ] 2.3 Unit-test the guarantee: the collector never evaluates a desktop-rooted `//` expression and never walks more than one application subtree per call.

## 3. Server wiring (server.py)
- [ ] 3.1 `get_session_state`: accept `actionable_controls` in `sections`, wire next to the `ui_tree` block (`server.py:4043`), passing `elements_of_interest` as app filter; desktop-only with a structured non-desktop rejection.
- [ ] 3.2 Expose per-call knobs (role filter, element cap) without widening the tool signature beyond need — see design.md D5 for the parameter mapping.
- [ ] 3.3 Update the `get_session_state` docstring (`server.py:3937`): document the section, the app-filter contract, and the desktop-only constraint.

## 4. Discoverability
- [ ] 4.1 `ui_tree` hint (`ui_tree_service.py:457`): mention `actionable_controls` for enumerating interactive controls with ready descriptors.
- [ ] 4.2 PlatynUI locator guidance (`rf_native_type_converter.py:1682` ff.): cross-reference the view as the alternative to per-element `Query` probing and `//control:*` dumps.

## 5. Tests + validation
- [ ] 5.1 `tests/unit/test_desktop_actionable_controls.py` with a fake node tree (pattern: existing ui_tree unit tests): (a) control nested at depth 5 appears flat; (b) descriptor shape incl. index disambiguation and nameless controls; (c) role filter + default set; (d) anchor precedence and the multi-app refusal; (e) node/element/time budget truncation flags; (f) non-desktop rejection; (g) never-walks-desktop assertion (2.3).
- [ ] 5.2 Hint/docstring text assertions (4.1, 4.2, 3.3).
- [ ] 5.3 Full unit suite green; `openspec validate desktop-actionable-controls --strict` passes.
- [ ] 5.4 (Docker lab, when a container slot is free) e2e sanity: gnome-calculator buttons and mousepad `control:Text` appear in `actionable_controls`, and a returned Button descriptor round-trips through `Pointer Click` — the SPIKE 2 §1.3/§1.4 probing sequences become one call.

# Proposal: desktop-actionable-controls

## Why

SPIKE 2 (`experiments/SPIKE_2_desktop_efficiency.md`, 2026-07-10) measured where
desktop-automation tool calls go: **~60% of every run is discovery/probing, not
actions** (cc-desk-base 59% of 70 calls, desk-calc 61% of 31, desk-edit 61% of
23). A dominant probing sink (spike §2.6, recommendation #6) is structural:

- `ui_tree` expands application subtrees only to `DEFAULT_MAX_DEPTH = 3`
  (`ui_tree_service.py:25`). GTK nests interactive controls 4–6 levels deep, so
  the controls agents actually need are invisible in the snapshot: in desk-edit
  the mousepad `control:Text` did not appear in the ui_tree ("no Text control
  visible at depth 3", spike §1.4); calculator buttons likewise never surfaced.
- Agents compensate with per-element existence probing — desk-calc issued one
  `Query` per button (5 probes: Frame, "7", "×", "=", result Label; spike §1.3)
  and desk-edit fell back to `Query //control:*` + `count(Text)` + `count(Edit)`
  (spike §1.4). cc-desk-base burned **8 calls (11%)** hunting a read-back recipe
  with `string()`/`count()`/`distinct-values()` probes (spike §1.2).
- Raising `max_depth` alone is the wrong fix: a depth-6 nested tree is a large,
  mostly-structural payload (panels, boxes, viewports) where the ~20 interesting
  leaves drown in wrapper nodes, and depth is not a portable bound — GTK nesting
  varies per widget.

The web side already solved this exact problem: P8 `actionable_elements` (ADR-021,
branch `feature/improve_complex_sites_handling`, `page_source_service.py:156`
`ActionableElementsCollector`) returns a **flat, role-filtered list** of
interactive elements with identity, state, geometry, and a ready-to-use selector
— replacing DOM spelunking with one call. Desktop has no analog (spike §2.6:
"There is no desktop equivalent of P8 `actionable_elements`").

One flat, app-scoped "here are all the Buttons/Edits/MenuItems with their Names
and ready descriptors" call should replace the 5–8 probing Queries per run.

## What Changes

- **New `actionable_controls` section in `get_session_state`** (desktop/PlatynUI
  sessions only) — the desktop analog of the web `actionable_elements` view:
  - A **flat list** (no nesting) of interactive controls found inside the
    selected application's accessibility subtree, filtered to an interactive
    role set (Button, Edit, Text, MenuItem, CheckBox, RadioButton, ComboBox,
    ListItem, TabItem, Slider, …; overridable per call).
  - Each entry carries: `role`, `name`, a **ready-to-use app-scoped descriptor**
    (e.g. `/app:*[@Name='gnome-calculator']//control:Button[@Name='7']`) that
    can be pasted directly into `Pointer Click`/`Query`/`Get Attribute`,
    `enabled`/`visible` state, bounds, and tree depth. Duplicate role+Name pairs
    are index-disambiguated so every descriptor resolves to one node.
  - The walk is **depth-unbounded within the AUT subtree** but bounded by node
    budget, element cap, and a wall-clock time budget (see design.md D2) — the
    depth-3 blindness disappears without unbounded AT-SPI cost.
- **AUT scoping is mandatory** — the walk always starts from an application node
  resolved via `elements_of_interest` (same contract as `ui_tree`), the
  session's `desktop_aut_pid` (`session_models.py:172`), or a single
  display-scoped app. With no resolvable anchor and multiple candidate apps, the
  section **refuses with the application list** instead of walking the desktop —
  the same posture as the ADR-027-family unscoped-locator guard
  (`keyword_executor.py:626`, `desktop_execution_signals.py:83`), which this
  view is the sanctioned alternative to. Display scoping (D4 PID probe) is
  reused unchanged.
- **Discoverability** — the `ui_tree` response hint and the `get_session_state`
  docstring name `actionable_controls` as the way to enumerate interactive
  controls (spike lesson §1.2: "Optional guidance ≠ delivered guidance");
  `get_platynui_locator_guidance` (`rf_native_type_converter.py:1682`) gains a
  cross-reference.

Out of scope here, proposed as an optional **sibling follow-on** (spike
recommendation #7, see design.md "Future"): a higher-level `launch` intent
(Start Process + Query Frame + Activate Window in one round-trip) and a
filename-only `screenshot` intent (the SCREENSHOT verb is already reserved,
`domains/intent/value_objects.py:47`). Also out of scope: the init-time keyword
cheat-sheet (spike #1, change `desktop-turn-economy-guidance`), Take Screenshot
fail-fast (#2), and desktop-aware batch (#3/#8).

## Capabilities

### New Capabilities

- `desktop-actionable-controls`: desktop sessions can retrieve a flat,
  role-filtered list of interactive controls with ready-to-use app-scoped
  descriptors in one `get_session_state` call; the walk is always AUT-scoped
  and budget-bounded, never desktop-wide.

## Impact

- `src/robotmcp/components/execution/ui_tree_service.py` — new
  `get_actionable_controls()` collector (DFS via `node.children()`, reusing the
  app-resolution/display-scoping machinery of `_collect_ui_tree_sync`,
  `ui_tree_service.py:295`); shared role/descriptor helpers.
- `src/robotmcp/server.py` — `get_session_state`: wire the new section next to
  `ui_tree` (`server.py:4043`), document it in the tool docstring
  (`server.py:3937`).
- `src/robotmcp/utils/rf_native_type_converter.py` — PlatynUI locator guidance
  cross-reference (`:1682` ff.).
- Tests: `tests/unit/test_desktop_actionable_controls.py` — flat view (deep
  controls found beyond depth 3), descriptor round-trip shape, role filtering,
  scoping refusal, budgets/truncation flags, non-desktop-session rejection;
  integration assertion in the Docker desktop lab (deferred to the harness) that
  mousepad's `control:Text` and calculator buttons appear.

# Design: desktop-actionable-controls

## Context

SPIKE 2 (`experiments/SPIKE_2_desktop_efficiency.md`) quantified the desktop
probing sink this change removes:

| Evidence | Number | Where |
|---|---|---|
| Discovery/probing share of tool calls | 59–61% across 3 runs | spike §1.5 |
| Per-element `Query` existence probes | 5 (desk-calc), one per button | spike §1.3 |
| Read-back recipe hunting (`string()`/`count()`/`distinct-values()`) | 8 calls (11%) | spike §1.2 |
| `control:Text` invisible in depth-3 `ui_tree` → `//control:*` + `count()` fallback | 6 state-probing calls | spike §1.4 |
| Root cause | `DEFAULT_MAX_DEPTH = 3` | `ui_tree_service.py:25`, spike §2.6 |

The web side solved the same problem with P8 `actionable_elements`
(`ActionableElementsCollector`, `page_source_service.py:156` on branch
`feature/improve_complex_sites_handling`): a flat list of interactive elements
with identity, state, geometry, and a ready selector, surfaced as a
`get_session_state` section with an element cap (default 80) and truncation
flags. That shape is proven with agents; the desktop view mirrors it.

## Goals / Non-Goals

**Goals:** one call replaces the 5–8 per-element probes; controls at any GTK
nesting depth are enumerable; every returned descriptor is directly usable by
PlatynUI keywords; the walk can never become the desktop-wide AT-SPI crawl the
unscoped-locator guardrail exists to prevent; cost is bounded and degrades to
partial results, never to a hang.

**Non-Goals:** replacing `ui_tree` (structure/nesting questions still need the
tree); a live "watch" or diffing view; web/mobile sessions (they have
`page_source`/ARIA and, on the sibling branch, `actionable_elements`); the
launch/screenshot intents (sibling follow-on, see Future); raising
`DEFAULT_MAX_DEPTH` for `ui_tree` itself (see D2).

## Decisions

### D1 — One Python DFS over `node.children()`, not per-role native queries
Two candidate mechanics: (a) one depth-first traversal of the anchor subtree in
Python via `node.children()` (the machinery `_expand_subtree`
(`ui_tree_service.py:94`) already uses), filtering by role as we visit; (b) one
native `runtime.evaluate("/app:*[@Name='X']//control:Button")` per role.
Chosen: **(a)**. One traversal touches every node exactly once regardless of how
many roles are requested; budgets (nodes, elements, wall-clock) are enforced
per-visit in Python, which native evaluation cannot interrupt; and it reuses the
existing broker-owned runtime + tree-cache-refresh path
(`clear_runtime_tree_cache`, `ui_tree_service.py:365`) without new native
surface. Per-role native queries would re-walk the subtree N times and each walk
is uninterruptible — on a stalling AT-SPI node (1 s timeout per unresponsive
node, `platynui_plugin.py:84` hint) that multiplies the worst case.

### D2 — Depth-unbounded, budget-bounded; `ui_tree` depth stays at 3
Depth is the wrong bound for this view: GTK nesting varies per widget (3–6+),
so any fixed depth either misses controls (the spike failure) or over-fetches
structure. The collector bounds **cost**, not shape: `max_nodes` visited
(default 1500 — an order above the ui_tree budget of 200, affordable because we
return leaves, not structure), `max_elements` returned (default 80, matching the
web collector's cap), and a wall-clock budget (default 5 s) checked between
visits so a stalling provider yields partial results instead of an MCP-timeout.
`ui_tree`'s `DEFAULT_MAX_DEPTH = 3` is deliberately untouched: it answers
"what's the structure" cheaply; this view answers "what can I act on". A
14-app desktop with one expanded app stays within the existing ui_tree contract;
this view's worst case is one full app subtree — bounded by the budgets, and in
practice small (gnome-calculator ≈ low hundreds of nodes).

### D3 — Mandatory AUT anchor; relationship to the ADR-027-family guards
The unscoped-locator guardrail refuses `//`-rooted `Query`/`Evaluate` because an
absolute walk re-crawls every desktop app and can exceed the MCP transport
timeout (`keyword_executor.py:626`, `desktop_execution_signals.py:62-108`). This
view must be the *sanctioned* alternative, so it can never itself become that
walk: the DFS starts at exactly one application node, resolved by precedence
(1) `elements_of_interest` name filter — same contract as `ui_tree`;
(2) the session's `desktop_aut_pid` (`session_models.py:172`, set on launch);
(3) the single remaining app after display scoping. Anything else → structured
refusal carrying the `/app:*` application list (that listing is the safe
single-level query, `ui_tree_service.py:371`). Display scoping reuses the D4
batched `_NET_WM_PID` probe (`_display_scoped_pids`, `ui_tree_service.py:42`)
verbatim, including its fail-open-for-the-AUT semantics — host password
managers and browsers on the session bus never become an anchor.
*Alternative rejected*: defaulting to "walk all display-scoped apps" when no
filter is given — on an active (non-isolated) desktop that IS the desktop-wide
walk, and refusing-with-the-list costs the agent exactly one cheap round-trip.

### D4 — Descriptor construction
Entries carry a descriptor an agent can paste into `Pointer Click`/`Query`
unmodified: anchored at the resolved application
(`/app:*[@Name='<app>']//control:<Role>[@Name='<name>']`), with the Name
XPath-quoted (apostrophes handled via the standard `concat()` fallback). When
several controls share role+Name within the app, entries get a stable positional
suffix (`[1]`, `[2]` in document order) and a `nth` field so the ambiguity is
visible. Nameless interactive controls (common for `control:Text` — the
mousepad case) are still listed: role-only descriptor when unique in the app
subtree, positional otherwise. Correctness rule: the collector must never emit a
descriptor that resolves to more than one node — that would re-create the
probing loop this change removes.

### D5 — Surfacing: a `get_session_state` section named `actionable_controls`
Mirrors how the web view is surfaced (a section + a cap parameter,
`server.py:3486` on the sibling branch) so agents meet one idiom. The name is
deliberately NOT `actionable_elements`: the web collector lives on
`feature/improve_complex_sites_handling` and both branches will merge — distinct
names keep the desktop (accessibility-tree, descriptor) and web (DOM, selector)
payload shapes separately documented instead of one section with two disjoint
schemas. Desktop sessions already auto-include `ui_tree` (`server.py:3970`);
`actionable_controls` stays opt-in — it is an action-planning payload, not
ambient state. Parameters map onto existing ones where possible
(`elements_of_interest` = app filter, as for `ui_tree`); the role filter and
element cap ride either dedicated small params or a conservative default —
decided at implementation with the constraint that `get_session_state`'s
signature growth stays minimal.

### D6 — Default interactive role set
Button, Edit, Text, MenuItem, Menu, CheckBox, RadioButton, ComboBox, ListItem,
TabItem, Slider, Link, ToggleButton, SpinButton — the AT-SPI roles agents act
on, plus Label kept OUT by default (dozens of static labels per app would eat
the element cap) but selectable via the role filter for read-back scenarios
(the spike's result-Label recipe). The set lives next to `PLATYNUI_KEYWORDS`
conventions as a module constant so guidance and tests share it.

## Risks / Trade-offs

- [AT-SPI stalls: 1 s per unresponsive node] → wall-clock budget checked
  between visits returns partial results with a truncation reason; the walk is
  in a worker thread (`asyncio.to_thread`, same as `get_ui_tree`) so the event
  loop never blocks.
- [Huge apps (IDEs, browsers) blow the budgets] → that is the designed
  behavior: caps + truncation flags + role filter to narrow; the refusal/hint
  text tells the agent to filter rather than paginate (pagination is a
  follow-on if real transcripts demand it).
- [Descriptor drift: tree changes between collection and use] → same exposure
  every discovery view has; descriptors are name/role-based (stable across
  relayout), and the tree cache is cleared before collection
  (`clear_runtime_tree_cache`) so results reflect the live tree.
- [Section name divergence from the web `actionable_elements`] → deliberate
  (D5); the docstring cross-links the concepts so agents transfer the idiom.
- [Nameless-control positional descriptors are brittle] → flagged per-entry
  (`nth`), and the entry still carries bounds so an agent can fall back to
  coordinate interaction knowingly.

## Future (sibling follow-on — spike recommendation #7, NOT part of this change)

A separate `desktop-launch-intent` change may add: (a) an
`intent_action(intent="launch", app=...)` verb collapsing the 3–4-call preamble
(Start Process → Query Frame → Activate Window, optionally an
`actionable_controls` snapshot) into one round-trip — the launch preamble was
4 calls in every spike run; and (b) a filename-only `screenshot` intent that
sidesteps the Take Screenshot descriptor/filename trap (spike §2.3) entirely.
The verb surface is prepared: `SCREENSHOT` is already reserved in
`domains/intent/value_objects.py:47`, and `ENSURE_FOCUSED` (ADR-026) shows the
pattern for desktop-only verbs. This change's collector is a dependency of that
follow-on's optional snapshot, which is why the view ships first.

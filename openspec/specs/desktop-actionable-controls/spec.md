# desktop-actionable-controls Specification

## Purpose
TBD - created by archiving change desktop-actionable-controls. Update Purpose after archive.
## Requirements
### Requirement: Desktop sessions expose a flat, role-filtered actionable-controls view
For a desktop (PlatynUI) session, `get_session_state` SHALL provide an
`actionable_controls` section: a FLAT list (no nesting) of the interactive
controls found in the selected application's accessibility subtree, filtered to
an interactive role set (at minimum Button, Edit, Text, MenuItem, CheckBox,
RadioButton, ComboBox, ListItem, TabItem, Slider; overridable per call). Each
entry SHALL include the control's role, its `@Name` (when present), a
ready-to-use **app-scoped descriptor** accepted verbatim by PlatynUI keywords
(`Pointer Click`, `Query`, `Get Attribute`), enabled/visible state, and bounds
when available. The walk SHALL NOT be limited by the `ui_tree` depth bound
(`DEFAULT_MAX_DEPTH = 3`): controls at any nesting depth within the application
subtree SHALL be found, subject only to the cost budgets. Descriptors SHALL be
unambiguous: when several controls share role and Name, entries SHALL be
disambiguated (e.g. positional index) so each descriptor resolves to exactly one
node; nameless controls of interactive roles SHALL still be listed with a
resolvable descriptor.

#### Scenario: deeply nested GTK control appears in the flat view
- **WHEN** `actionable_controls` is requested for a GTK application whose text area nests deeper than 3 levels (e.g. mousepad's `control:Text`, invisible in the depth-3 `ui_tree` — SPIKE 2 §1.4)
- **THEN** the section lists that control with its role and a ready app-scoped descriptor, without the agent issuing any per-element `Query`/`count()` probe

#### Scenario: returned descriptor is directly actionable
- **WHEN** an agent takes the `descriptor` of a listed Button entry and passes it unchanged as the locator of `Pointer Click` (or `Query`)
- **THEN** the descriptor resolves to that control — no reformatting, re-anchoring, or additional discovery call is required

#### Scenario: role filter narrows the list
- **WHEN** the caller restricts the view to specific roles (e.g. only Button)
- **THEN** only controls of the requested roles are returned, and the default role set applies when no filter is given

#### Scenario: non-desktop sessions are rejected with guidance
- **WHEN** `actionable_controls` is requested for a web/mobile session
- **THEN** the section returns a structured failure explaining it is desktop-only and pointing web sessions at their page-source/element views

### Requirement: The actionable-controls walk is always AUT-scoped
The collector SHALL anchor its walk at a single application node and SHALL NOT
walk the desktop-wide accessibility tree. The anchor SHALL be resolved from, in
order: an explicit application filter (`elements_of_interest`, same contract as
`ui_tree`), the session's launched-AUT identity (`desktop_aut_pid`), or the only
candidate application after display scoping. When no anchor can be resolved and
multiple applications are present, the section SHALL refuse with the application
list and instructions to pass a filter — mirroring the unscoped-locator
guardrail posture (`//`-rooted desktop walks are refused at the keyword level).
Display scoping SHALL behave exactly as in `ui_tree`: on isolation-marked
displays, host applications are filtered by the PID probe and the AUT is never
hidden.

#### Scenario: explicit application filter anchors the walk
- **WHEN** `actionable_controls` is requested with an application name filter matching one application
- **THEN** the walk covers only that application's subtree and the response names the resolved application

#### Scenario: no anchor and multiple applications refuses instead of walking the desktop
- **WHEN** no application filter is given, the session has no recorded AUT pid, and more than one application is in scope
- **THEN** no subtree is walked; the section returns the candidate application list and a hint to pass an application name — never a desktop-wide enumeration

#### Scenario: host applications stay invisible on isolated displays
- **WHEN** the bound display is isolation-marked and the anchor is resolved
- **THEN** applications whose process owns no window on the bound display are never selected as the anchor nor represented in the results

### Requirement: The actionable-controls walk is budget-bounded
The collector SHALL enforce explicit cost bounds independent of tree depth: a
maximum number of visited nodes, a maximum number of returned elements, and a
wall-clock time budget for the walk. On reaching any bound it SHALL return the
elements collected so far with explicit truncation indicators (which bound was
hit, totals where known) rather than failing or blocking. Provider errors on
individual nodes SHALL be skipped without aborting the walk, and the collector
SHALL never raise into the `get_session_state` response.

#### Scenario: node budget truncates gracefully
- **WHEN** the application subtree contains more nodes than the visit budget
- **THEN** the response contains the controls found within budget plus a truncation flag identifying the exhausted bound, and the call still succeeds

#### Scenario: element cap limits payload
- **WHEN** more interactive controls match than the element cap
- **THEN** the list is capped, the response reports the cap was applied, and the caller can narrow via the role filter

#### Scenario: a stalling node does not hang the section
- **WHEN** individual accessibility nodes error or respond slowly during the walk
- **THEN** the walk skips or stops within the time budget and returns partial results with a truncation indicator, instead of hanging the MCP call

### Requirement: The actionable-controls view is discoverable from existing surfaces
Agent-facing text SHALL steer desktop agents to the view without prior
knowledge: the `ui_tree` section's hint SHALL name `actionable_controls` as the
way to enumerate interactive controls, the `get_session_state` docstring SHALL
document the section and its parameters, and the PlatynUI locator guidance SHALL
cross-reference it as the alternative to per-element `Query` probing and
unscoped `//control:*` dumps.

#### Scenario: ui_tree points to the flat view
- **WHEN** an agent reads a `ui_tree` response for a desktop session
- **THEN** its hint mentions `actionable_controls` for enumerating interactive controls with ready descriptors

#### Scenario: locator guidance cross-references the view
- **WHEN** an agent requests PlatynUI locator guidance
- **THEN** the guidance names `actionable_controls` instead of recommending per-element existence probing


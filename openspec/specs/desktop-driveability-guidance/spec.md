# desktop-driveability-guidance Specification

## Purpose
TBD - created by archiving change desktop-stepwise-execution-fidelity. Update Purpose after archive.
## Requirements
### Requirement: PlatynUI guidance documents the UiNode attribute API

The system SHALL document the supported `UiNode` attribute-access surface in the
PlatynUI guidance so agents do not guess a non-existent API (e.g.
`get_attribute`) through `Evaluate`. The guidance MUST name `node.name`,
`node.role`, `node.attribute(<name>)`, and `node.attributes()` + `attr.value()`,
and MUST steer attribute reads to the `Get Attribute` keyword rather than raw
Python introspection.

#### Scenario: guidance names the correct attribute API
- **WHEN** a caller requests PlatynUI locator/usage guidance
- **THEN** the guidance states `UiNode` exposes `attribute(...)`/`attributes()`
  (`attr.value()`), `name`, `role` — and NOT `get_attribute` — and points to the
  `Get Attribute` keyword for reading attributes

### Requirement: Guidance covers duplicate application roots and duplicate controls

The system SHALL document that a single desktop application can present multiple
live application-root nodes and duplicate controls (Wayland/AT-SPI), and give a
first-class disambiguation approach (scope with `Set Root`, prefer the
interactable/in-view node, query within one root) instead of leaving agents to
index raw query results by trial.

#### Scenario: guidance explains duplicate-root disambiguation
- **WHEN** PlatynUI guidance is requested
- **THEN** it explains that `//app:*[@Name='X']` may return multiple roots and
  how to disambiguate (scope to one root, select the interactable node) rather
  than blindly indexing the query result

### Requirement: Guidance covers desktop control naming

The system SHALL document that desktop control names are inconsistent across
roles — operator keys expose symbol `Name`s (e.g. `+`) on a `Button` while the
same operator may read as a word (e.g. `plus`) on a `Label` — and recommend
discovering names via the ui_tree / `Get Attribute` rather than guessing.

#### Scenario: guidance explains symbol-vs-word naming
- **WHEN** PlatynUI guidance is requested
- **THEN** it warns that operator/control names differ by role (symbol on
  Button vs word on Label) and points to ui_tree discovery for exact names

### Requirement: Guidance covers reading the calculator/entry display state

The system SHALL document how to read a desktop application's display state —
which node carries the displayed text and that GTK content is exposed via
`native:Text.CharacterCount` (length proxy) rather than readable text content —
so agents can assert per-entry/result state deterministically.

#### Scenario: guidance explains display-state reading
- **WHEN** PlatynUI guidance is requested
- **THEN** it explains that the entry's text content may not be AT-SPI-readable
  and that `Get Attribute ... native:Text.CharacterCount` is the supported
  length proxy for verifying display changes


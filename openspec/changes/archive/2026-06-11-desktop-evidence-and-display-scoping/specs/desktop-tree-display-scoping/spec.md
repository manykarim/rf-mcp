# Spec: desktop-tree-display-scoping

## ADDED Requirements

### Requirement: Isolated sessions list only applications on the bound display
For desktop sessions whose bound display is isolation-marked, the `ui_tree` section SHALL exclude applications whose process has no X client window on the bound display (determined via a single batched `_NET_WM_PID` probe executed in an isolated subprocess), SHALL report the number excluded as `host_apps_filtered`, and SHALL keep applications whose process id cannot be read (fail-open) annotated as `display_scoped: false`.

#### Scenario: Host desktop apps filtered
- **WHEN** the AT-SPI tree contains gnome-shell, Chrome, and the AUT, and only the AUT's PID owns a window on the marked display `:100`
- **THEN** `ui_tree.applications` contains the AUT but not gnome-shell or Chrome, and `host_apps_filtered` is 2

#### Scenario: PID-less app kept and annotated
- **WHEN** an application node exposes no readable ProcessId
- **THEN** it remains in the list with `display_scoped: false`

#### Scenario: Active-display sessions unfiltered
- **WHEN** the bound display is classified `active` or `unknown`
- **THEN** no display filtering is applied

### Requirement: Probe failure degrades to no filtering
When the display probe cannot complete, the system SHALL return the unfiltered application list with a `display_scoping: "unavailable"` annotation rather than hiding applications.

#### Scenario: Probe unavailable
- **WHEN** the EWMH probe subprocess fails or times out
- **THEN** all applications are listed and the section notes that display scoping was unavailable

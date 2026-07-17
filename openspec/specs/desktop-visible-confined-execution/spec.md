# desktop-visible-confined-execution Specification

## Purpose
TBD - created by archiving change platynui-visible-safe-targeting. Update Purpose after archive.
## Requirements
### Requirement: Target highlighting before interaction
For PlatynUI interaction keywords in desktop sessions, the system SHALL mark the resolved target element on screen using the upstream `Runtime.highlight(rect, duration_ms)` API before dispatching input, so a human watching the (visible) display sees exactly which element receives input. Highlighting SHALL be soft-fail (never blocks or fails the step), enabled by default, and disableable per session (`platynui_highlight: false`) and via `ROBOTMCP_PLATYNUI_HIGHLIGHT=0`.

#### Scenario: Highlight precedes click
- **WHEN** an interaction keyword executes with highlighting enabled and the target's bounds are known
- **THEN** `Runtime.highlight` is invoked with the target bounds before the input is dispatched, and the step result is unaffected by highlight success or failure

#### Scenario: Kill switch respected
- **WHEN** `ROBOTMCP_PLATYNUI_HIGHLIGHT=0` is set
- **THEN** no highlight call is made for any session

### Requirement: Session state proves display identity and confinement
`get_session_state` for desktop sessions SHALL include a `desktop_environment` section reporting the bound display, its isolation classification (`isolated` / `active` / `unknown`) with the classification source, and upstream `Runtime.desktop_info()` data (technology, bounds, monitors, os).

#### Scenario: Isolated visible display reported
- **WHEN** the session is bound to display `:100` listed in `ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY`
- **THEN** `desktop_environment` reports `display: ":100"`, `isolation: "isolated"`, `isolation_source: "marker"`, and the upstream desktop technology and bounds

### Requirement: Isolation recipe recommends the visible mode
`build_isolation_recipe()` and the desktop safety-guard refusal message SHALL present the visible nested-display mode (Xephyr) as the recommended interactive path — application visible on the tester's screen while synthetic input is confined to the nested display — with the headless mode as the CI alternative, and SHALL include upstream `platynui-cli-rs` verification commands (`info`, `window --list`, `highlight`, `snapshot`).

#### Scenario: Recipe leads with visible mode
- **WHEN** the safety guard refuses an active-desktop session and returns the isolation recipe
- **THEN** the recipe's first option is the visible Xephyr mode marked as recommended for interactive testing, and `platynui-cli-rs` verification commands are included

### Requirement: Visible mode provisions an EWMH window manager
The visible-mode bootstrap (script and recipe) SHALL start a minimal EWMH-capable window manager inside the nested display when one is available, because upstream `WindowSurface.activate()` and window resolution require EWMH; when no WM is available the recipe SHALL state that focus verification will degrade to a warning.

#### Scenario: Bootstrap starts a WM in visible mode
- **WHEN** `scripts/platynui_desktop_bootstrap.sh --mode visible` runs on a host with `openbox` installed
- **THEN** an EWMH WM is started inside the Xephyr display and the display remains classified isolated via the marker


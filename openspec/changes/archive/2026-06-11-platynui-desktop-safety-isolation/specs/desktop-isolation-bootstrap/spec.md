## ADDED Requirements

### Requirement: Supported isolated-display bootstrap for desktop sessions

The system SHALL provide a supported way to bootstrap a confined desktop
session on an isolated display with the documented environment
(`DISPLAY` on a dedicated Xvfb, `XDG_SESSION_TYPE=x11`, `GDK_BACKEND=x11`,
`GSK_RENDERER=cairo`, `WAYLAND_DISPLAY` unset, a session bus), so users do not
hand-roll Xvfb setup in ad-hoc Robot suite code.

#### Scenario: Bootstrap yields a confined, usable desktop session
- **WHEN** a user requests an isolated desktop bootstrap
- **THEN** an isolated display is prepared with the documented environment and
  the resulting desktop session is confined to it (the safety guard treats it
  as isolated)

#### Scenario: Bootstrap is reusable across the canonical scenarios
- **WHEN** the canonical Calculator / Text Editor stepwise scenarios run under
  the bootstrap
- **THEN** they execute end-to-end on the isolated display without ad-hoc
  per-suite Xvfb setup

### Requirement: Positive isolation contract for bootstrapped displays

The system SHALL establish isolation by positive proof the bootstrap owns: an
rf-mcp-created X server (Xvfb / nested / VNC-backed) with a dedicated
`XAUTHORITY` and/or private socket, an explicit `DISPLAY` exported to the MCP
server process and to every launched child, and a recorded isolation marker
that the safety guard validates. The marker SHALL take precedence over the
EWMH probe, so a bootstrapped display that happens to run its own window
manager is still classified `isolated`.

#### Scenario: Bootstrap records an isolation marker
- **WHEN** the isolation bootstrap creates a display
- **THEN** it records an rf-mcp isolation marker for that display, sets a
  dedicated `XAUTHORITY`/socket, and exports the `DISPLAY` to the server and
  child launches

#### Scenario: Marker beats the EWMH probe
- **WHEN** a bootstrapped display runs its own window manager (e.g. a nested
  display with an internal WM) and the EWMH probe would otherwise read `active`
- **THEN** the safety guard classifies it `isolated` because the rf-mcp marker
  is present

#### Scenario: Server process binds the bootstrapped display
- **WHEN** the bootstrap prepares an isolated display
- **THEN** the MCP server's own `DISPLAY`/`XDG_SESSION_TYPE`/`WAYLAND_DISPLAY`
  are set to the bootstrapped display before the first PlatynUI runtime binds,
  so the native runtime connects to that display (not the host `:0`)

### Requirement: Visible/observable isolated display for stepwise execution

The system SHALL offer a visible (observable) isolated display mode — a nested
X server (e.g. Xephyr, shown as a window on the host) or a VNC-backed display
— so that during stepwise execution the user can watch the actual desktop
application and its interactions, while the session remains confined and off
the user's active session. Visible mode SHALL be selectable for interactive/
stepwise use, with the headless Xvfb mode available for CI/automated runs.

#### Scenario: Watch interactions during stepwise execution
- **WHEN** a user bootstraps the desktop session in visible mode and runs the
  scenario stepwise
- **THEN** the AUT and each pointer/keyboard interaction are visible on the
  host (via the nested X server / VNC), and the session is still treated as
  isolated by the safety guard (it is not the user's active session)

#### Scenario: Headless mode for CI
- **WHEN** the bootstrap is run in headless mode (e.g. CI, no observer)
- **THEN** the session runs on a headless isolated display with no on-host
  window, and the scenarios still execute end-to-end

#### Scenario: Visible mode does not weaken confinement
- **WHEN** the visible nested-X-server / VNC display is in use
- **THEN** automated input is delivered only to that isolated display, every
  launched AUT process's `DISPLAY` equals the isolated display (verifiable via
  the child's environment), and a window opened on that display does not appear
  on the user's active desktop

#### Scenario: Visible-mode caveats documented
- **WHEN** visible mode is used on a host with a window manager
- **THEN** the documentation notes that host global shortcuts may still
  intercept some key chords (input is confined to the isolated display but the
  visible window is not a fully independent seat), and that VNC binds to
  localhost / a private socket by default

### Requirement: Live mutation acceptance for isolated desktop automation

The system SHALL include a must-pass live acceptance criterion proving that, in
isolated mode, desktop automation actually mutates AUT state — a stepwise GNOME
Calculator flow where each pointer click provably changes the calculator's
observable state (the display character count) — so that finding #5 ("frame
resolves but clicks don't mutate") is verified closed rather than implied by
safer binding or visible mode alone.

#### Scenario: Each click mutates the calculator
- **WHEN** the isolated stepwise calculator flow clicks digits/operators
- **THEN** the display character count changes after each click as expected,
  and the final result matches

#### Scenario: Residual input-delivery failure is tracked, not hidden
- **WHEN** a click resolves and focuses the target but the AUT state does not
  change in isolated mode
- **THEN** the failure is surfaced and tracked as an explicit investigate/fix
  item, not reported as success

### Requirement: Reproduction harness for the documented findings

The system SHALL include a reproduction harness that re-creates the documented
findings (scenario misclassification, snap launch failure, Robot/MCP runtime
init failure, fresh-isolated input non-mutation) so each can be verified fixed.

#### Scenario: Findings are reproducible and then pass after fixes
- **WHEN** the reproduction harness runs against the documented findings
- **THEN** each finding is reproduced on the pre-fix code path and passes once
  the corresponding fix is in place

### Requirement: Validated stepwise calculator suite replaces the artifact

The system SHALL provide a validated isolated stepwise GNOME Calculator suite
(replacing the investigation artifact
`tests/e2e/gnome_calculator_mcp_stepwise.robot`) that asserts each entered
value and the final result and runs green on the isolated bootstrap.

#### Scenario: Validated suite runs green isolated
- **WHEN** the validated stepwise calculator suite runs under the isolation
  bootstrap
- **THEN** every per-entry assertion and the result assertion pass

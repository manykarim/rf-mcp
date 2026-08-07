## ADDED Requirements

### Requirement: The dashboard surfaces failures and does not clip core controls

The dashboard SHALL indicate its live-connection state, SHALL degrade a single failed data request to
that region rather than blanking the whole view, and SHALL NOT clip primary controls (e.g. the suite
Generate button) or strand content behind a non-scrolling overflow.

#### Scenario: connection state is visible
- **WHEN** the live event stream connects, drops, or is unavailable
- **THEN** a connection indicator reflects Live / Reconnecting / Offline rather than silently showing frozen state

#### Scenario: one failed request does not blank the view
- **WHEN** one of the session-detail requests fails
- **THEN** the other regions still render, and the failure is surfaced rather than blanking the whole pane

#### Scenario: core controls are not clipped
- **WHEN** the dashboard is viewed on a narrow (mobile) viewport
- **THEN** the primary suite action remains visible/reachable, and the sidebar scrolls rather than stranding content

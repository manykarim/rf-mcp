## ADDED Requirements

### Requirement: Tri-state bound-display classification by positive isolation proof

The system SHALL classify, for a DESKTOP_TESTING session, the bound display as
exactly one of `isolated`, `active`, or `unknown`, where `isolated` requires
**positive proof** of an rf-mcp-owned isolation (an isolation marker set by the
rf-mcp bootstrap — e.g. a dedicated `XAUTHORITY`/socket and a recorded
display-provenance marker), `active` is asserted when an EWMH window manager is
detected on the bound display (`_NET_SUPPORTING_WM_CHECK` resolves to a live
window), and `unknown` is used in every other case, including when the
detection probe cannot be completed or no proof of isolation exists. The
absence of an EWMH window manager alone MUST NOT be treated as `isolated`.

#### Scenario: Provably isolated display
- **WHEN** the bound display carries the rf-mcp isolation marker from the
  bootstrap
- **THEN** the display is classified `isolated`

#### Scenario: Active user desktop detected
- **WHEN** an EWMH window manager is present on the bound display
  (`_NET_SUPPORTING_WM_CHECK` resolves)
- **THEN** the display is classified `active`

#### Scenario: Unproven display is unknown, not isolated
- **WHEN** the bound display has no EWMH window manager AND no rf-mcp isolation
  marker (e.g. a bare X server, or the probe failed)
- **THEN** the display is classified `unknown` (NOT `isolated`)

### Requirement: Fail closed — refuse on active and unknown by default

The system SHALL refuse, by default, to dispatch desktop pointer/keyboard
operations when the bound display is classified `active` or `unknown`,
returning a clear actionable error, and SHALL allow operations without an
override only when the display is `isolated`.

#### Scenario: Operation blocked on the active desktop
- **WHEN** a desktop pointer/keyboard step would run on an `active` display and
  the opt-in has not been granted
- **THEN** the operation is refused with an error explaining the safety guard
  and how to run on an isolated display or opt in

#### Scenario: Operation blocked on an unknown display
- **WHEN** a desktop pointer/keyboard step would run on an `unknown` display and
  the opt-in has not been granted
- **THEN** the operation is refused (fail closed) rather than risking a leak

#### Scenario: Isolated display proceeds
- **WHEN** a desktop session is bound to a provably `isolated` display
- **THEN** the safety guard does not block operations

### Requirement: Auditable opt-in to bypass the guard

The system SHALL allow an explicit opt-in
(`ROBOTMCP_PLATYNUI_ALLOW_ACTIVE_DESKTOP=1`, and a per-session attribute) to
run on `active`/`unknown` displays, and SHALL make every bypassed run
auditable by flagging it in the step result payload and logging it at WARNING.

#### Scenario: Opt-in allows and is audited
- **WHEN** the operator sets the opt-in and a desktop operation runs on an
  `active`/`unknown` display
- **THEN** the operation proceeds, the result payload is flagged as having
  bypassed the safety guard, and a WARNING is logged

### Requirement: Surface the bound-display safety state

The system SHALL surface the bound-display classification
(`isolated`/`active`/`unknown`) and the guard decision to the agent at session
init and in `get_session_state`, computed without creating a throwaway
PlatynUI runtime (it reuses the runtime broker), so the safety posture is
visible before interaction and the surfacing itself does not destabilize the
runtime binding.

#### Scenario: Safety state reported at init
- **WHEN** a desktop session is initialized
- **THEN** the response indicates the bound-display classification and whether
  the guard is enforcing

#### Scenario: State surfacing does not create a throwaway runtime
- **WHEN** the bound-display state is computed for `get_session_state`
- **THEN** it uses the shared runtime broker and does not create-and-shutdown a
  separate runtime

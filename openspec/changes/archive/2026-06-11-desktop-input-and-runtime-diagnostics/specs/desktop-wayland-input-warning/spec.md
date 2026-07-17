## ADDED Requirements

### Requirement: Forced-X11 input on a Wayland session warns it may be blocked

The system SHALL record whether a desktop session ORIGINATED as a Wayland
session before robotmcp forced the X11 backend, and on the first desktop
interaction keyword (pointer/keyboard) of such a session, attach a
`wayland_x11_input_blocked_risk` warning stating that synthetic X11 (XTest)
input is likely blocked by the Wayland compositor so a "success" may not have
reached the application. The warning MUST name remediation (run on a real X11
session, or use PlatynUI's Wayland input backend when available) and clarify
that read/query (AT-SPI over D-Bus) is unaffected.

#### Scenario: first interaction on a was-Wayland session warns
- **WHEN** a desktop session that originated as Wayland (forced to X11) runs its
  first pointer/keyboard interaction keyword
- **THEN** the response includes a `wayland_x11_input_blocked_risk` warning with
  remediation and the read-still-works clarification

#### Scenario: X11-origin session does not warn
- **WHEN** the session originated as X11 (e.g. an isolated Xvfb)
- **THEN** no Wayland input warning is added

#### Scenario: read/query operations do not warn
- **WHEN** a desktop read/query keyword (e.g. Query / Get Attribute) runs on a
  was-Wayland session
- **THEN** no `wayland_x11_input_blocked_risk` warning is added (only
  input-injecting keywords carry the risk)

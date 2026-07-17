# desktop-runtime-failure-diagnostics Specification

## Purpose
TBD - created by archiving change desktop-input-and-runtime-diagnostics. Update Purpose after archive.
## Requirements
### Requirement: PlatynUI runtime bind/connect failures are classified

The system SHALL record the reason a PlatynUI runtime bind/connect failed and
classify it into at least: `not_installed` (the native module could not be
imported), `display_connect_failed` (the runtime could not connect to the
display — DISPLAY/XAUTHORITY/XDG_RUNTIME_DIR), and `disposed` (the one-shot
native module was disposed and cannot be re-initialized in this process). A
`runtime_unavailable_reason()` accessor SHALL return the current reason (or
`None` when the runtime is available). Classification MUST never raise.

#### Scenario: import failure classified as not_installed
- **WHEN** the native module import fails
- **THEN** the reason is `not_installed`

#### Scenario: connect failure classified as display_connect_failed
- **WHEN** the runtime bind fails with a display/connection error (e.g. "x11
  connection: not available after shutdown or failed connect")
- **THEN** the reason is `display_connect_failed`

#### Scenario: disposed broker classified as disposed
- **WHEN** the broker is in the disposed terminal state
- **THEN** the reason is `disposed`

### Requirement: Runtime-unavailable responses carry an actionable diagnostic

The system SHALL, when a desktop runtime-dependent operation finds the runtime
unavailable, return a structured diagnostic derived from the classified reason
with actionable remediation, rather than a generic "platynui-native not
installed". For `display_connect_failed` it MUST mention checking
DISPLAY/XAUTHORITY/XDG_RUNTIME_DIR and that the native runtime is one-shot so the
MCP server must be RESTARTED after fixing the environment.

#### Scenario: ui_tree on a connect-failed runtime explains the real cause
- **WHEN** `get_session_state(sections=['ui_tree'])` runs while the runtime is
  unavailable due to a display-connect failure
- **THEN** the result reports `display_connect_failed` with remediation (check
  DISPLAY/XAUTHORITY/XDG_RUNTIME_DIR; restart the MCP server), not "not installed"

#### Scenario: genuinely-not-installed still says not installed
- **WHEN** the runtime is unavailable because the native module is not installed
- **THEN** the result reports `not_installed` with the install hint


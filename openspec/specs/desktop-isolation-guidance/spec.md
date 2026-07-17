# desktop-isolation-guidance Specification

## Purpose
TBD - created by archiving change desktop-stepwise-execution-fidelity. Update Purpose after archive.
## Requirements
### Requirement: Active-desktop refusal returns an actionable isolation recipe

The system SHALL, when the active-desktop safety guard refuses desktop input on
a non-isolated display, return guidance that gives a concrete path to an
isolated display (e.g. an Xvfb/`systemd-run` bootstrap recipe and the
`ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY` marker) in addition to naming the
`ROBOTMCP_PLATYNUI_ALLOW_ACTIVE_DESKTOP` bypass — so a desktop scenario in a
normal session has a guided path forward instead of only a bypass switch.

#### Scenario: refusal message includes an isolation recipe
- **WHEN** the safety guard refuses input because the bound display is
  classified `active`
- **THEN** the refusal payload includes an actionable isolation recipe (how to
  run on an isolated display) and notes that the bypass env var is an escape
  hatch that does not guarantee correct input on a shared active desktop

#### Scenario: bypass remains available but is framed as an escape hatch
- **WHEN** the refusal guidance names `ROBOTMCP_PLATYNUI_ALLOW_ACTIVE_DESKTOP`
- **THEN** it is described as an escape hatch (input may still target the wrong
  window on a shared desktop), not as the recommended path


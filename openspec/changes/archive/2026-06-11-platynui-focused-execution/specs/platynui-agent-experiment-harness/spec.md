## ADDED Requirements

### Requirement: Reproducible multi-CLI experiment matrix

The system SHALL provide a documented, reproducible experiment harness that
runs identical stepwise PlatynUI scenarios through each supported coding-agent
CLI — Codex CLI, OpenCode CLI, Kilo CLI, and Claude CLI — against the rf-mcp
MCP server, so that focus/visibility/targeting behavior can be compared and
validated across agents.

#### Scenario: Same scenario runs across all four CLIs
- **WHEN** the experiment harness executes the canonical scenario set
- **THEN** it runs the scenarios via Codex CLI, OpenCode CLI, Kilo CLI, and
  Claude CLI and records a per-CLI result (pass/fail + collected issues)

#### Scenario: An agent CLI is unavailable
- **WHEN** one of the agent CLIs is not installed or cannot connect the MCP
  server
- **THEN** the harness records that CLI as skipped with the reason and
  continues with the others

### Requirement: Canonical Calculator and Text Editor scenarios

The harness SHALL include canonical, assertion-bearing scenarios for GNOME
Calculator and GNOME Text Editor that exercise: launching the app visibly,
stepwise execution, focus before interaction, per-action verification of
entered values, result verification, and final `build_test_suite`.

#### Scenario: Calculator scenario with per-action assertions
- **WHEN** the harness runs the Calculator scenario
- **THEN** the scenario opens the app, performs calculations, asserts each
  entered value and each result, and builds the final suite

#### Scenario: Text Editor scenario with read-back verification
- **WHEN** the harness runs the Text Editor scenario
- **THEN** the scenario types content into the editor, verifies it via an
  out-of-band read-back, and builds the final suite

### Requirement: Environment normalization for agent-spawned MCP servers

The harness SHALL document and apply the environment required for
agent-spawned MCP servers to reach a visible desktop and a working session
bus (display, session type, GDK backend, D-Bus session address, runtime dir,
home), so that apps launched by the agent are shown and addressable rather
than failing to register on the accessibility bus.

#### Scenario: Agent CLI strips the MCP-server environment
- **WHEN** an agent CLI spawns the MCP server with a reduced environment
  that omits the session bus or display
- **THEN** the harness's documented configuration restores the required
  variables so the launched AUT is shown and registers on the bus

### Requirement: Issue-collection report

The harness SHALL collect every focus, visibility, window-targeting, and
agent-environment issue encountered during the runs into a single findings
report, with enough detail (CLI, scenario, symptom, root cause, mitigation)
to drive fixes.

#### Scenario: Findings report aggregates all runs
- **WHEN** the experiment matrix completes
- **THEN** a findings report enumerates each discovered issue with its CLI,
  scenario, symptom, root cause, and mitigation/status

## MODIFIED Requirements

### Requirement: Multi-agent installer registers rf-mcp into coding agents

rf-mcp SHALL provide `robotmcp install`, `robotmcp uninstall`, and `robotmcp list`
commands that register the rf-mcp MCP server into supported coding agents using each
agent's own configuration file and format, and the launch command it writes SHALL be
resolved so that the running server can import the target project's Robot Framework
libraries (see "Project-aware launch resolution"). The registry SHALL cover Claude Code,
Codex, GitHub Copilot, opencode, Gemini CLI, Kilo Code, goose, and Cursor, and MAY
include additional agents whose configuration convention is confirmed.

#### Scenario: install writes each targeted agent's native MCP config
- **WHEN** a user runs `robotmcp install --agents claude-code,codex,gemini --what mcp`
- **THEN** the rf-mcp MCP server is added to each agent's own config file in that agent's format (e.g. Claude Code JSON `mcpServers`, Codex TOML `[mcp_servers]`, Gemini JSON `mcpServers`), and any MCP servers already configured in those files are preserved

#### Scenario: the written command carries resolved args and env
- **WHEN** the resolver selects a command that needs arguments or environment (e.g. a uv overlay or an attach-bridge entry)
- **THEN** the installer writes the full `command` plus `args` (and `env` when required) in the agent's entry shape, rather than only a bare command with an empty argument list

#### Scenario: detection drives interactive selection
- **WHEN** a user runs `robotmcp install` interactively
- **THEN** agents detected on the machine are pre-selected, the user can adjust the selection, and `--agents detected` (non-interactive) targets exactly the detected agents

#### Scenario: dry-run changes nothing
- **WHEN** a user runs `robotmcp install --dry-run`
- **THEN** the planned changes — including the resolved command, args, env, and the verification outcome — are printed and no configuration file is modified

#### Scenario: an unconfirmed agent adapter is surfaced, not silently wrong
- **WHEN** an agent's MCP-config convention has not been confirmed (e.g. `pi`)
- **THEN** `robotmcp list` reports that adapter's status as planned/unconfirmed rather than writing a possibly-incorrect configuration

## ADDED Requirements

### Requirement: Project-aware launch resolution (uv-first)

The installer SHALL resolve the launch command for a project-scoped install so that the running
rf-mcp server can import the Robot Framework libraries present in the target project's Python
environment, preferring uv, and SHALL NOT default to rf-mcp's own environment when the project
provides libraries rf-mcp does not bundle.

Resolution SHALL detect the project's environment type from the project directory (uv project,
plain virtualenv, poetry, pdm, pipenv, rye, hatch, conda, or bare global) and choose, in order:
when rf-mcp is already importable in the project environment, run it there; otherwise, when the
project is a virtualenv and uv is available and there is no dependency conflict, a non-mutating uv
overlay pinned to rf-mcp's installed version; otherwise, when the project needs only libraries
rf-mcp already bundles or has no detectable environment, rf-mcp's own launch command. The project
environment SHALL NOT be mutated unless the user explicitly opts in.

#### Scenario: project-only library is reachable after install
- **WHEN** a project's virtualenv contains a Robot Framework library that rf-mcp does not bundle, and the user runs `robotmcp install --scope project`
- **THEN** the written command launches rf-mcp such that that library is importable by the running server (e.g. a uv overlay against the project interpreter), not rf-mcp's own environment where the library is absent

#### Scenario: generic project keeps rf-mcp's own command
- **WHEN** the project needs only libraries that `rf-mcp[all]` already bundles, or no project environment is detected
- **THEN** the installer writes rf-mcp's own launch command, unchanged from prior behaviour

#### Scenario: the project environment is not mutated by default
- **WHEN** a project-aware install resolves a uv overlay command
- **THEN** the project's dependency tree and lockfile are left unchanged, and rf-mcp is installed into the project environment only when the user passes an explicit opt-in flag

### Requirement: Launch-and-library verification before writing

The installer SHALL launch the resolved command and confirm it starts the MCP server and can reach
the project's Robot Framework libraries before persisting the configuration, and SHALL NOT write an
entry whose command fails to launch or cannot see the project's libraries, unless verification is
explicitly skipped.

#### Scenario: a working command is written after verification
- **WHEN** the resolved command launches the server and a detected project library is confirmed reachable through rf-mcp
- **THEN** the entry is written to the agent config and recorded in the manifest

#### Scenario: a non-working command is refused
- **WHEN** the resolved command fails to start the server or the detected project library is not reachable through it
- **THEN** the installer does not write the entry, and reports which step failed and the next action to take

#### Scenario: verification can be skipped explicitly
- **WHEN** a user passes the skip-verification flag
- **THEN** the installer writes the resolved entry without launching it, and records that verification was skipped

#### Scenario: doctor reports project-library visibility
- **WHEN** a user runs `robotmcp doctor --project` in or against a project directory
- **THEN** it reports, read-only, which of the project's Robot Framework libraries the currently resolvable rf-mcp launch can see, without modifying any file

### Requirement: Attach bridge under irreconcilable dependency conflict

The installer SHALL route to the attach bridge, rather than an overlay that silently changes
versions, when the project's pinned dependencies cannot be reconciled with rf-mcp's own
requirements (for example the project pins Robot Framework older than rf-mcp supports, an
incompatible shared dependency, or a Python older than rf-mcp's baseline).

#### Scenario: conflicting project routes to attach, not a silent upgrade
- **WHEN** the project pins a Robot Framework version incompatible with rf-mcp and the user runs a project-aware install
- **THEN** the installer writes rf-mcp's own command with attach-bridge environment (host/port/token) and reports the conflict and the attach setup, instead of an overlay that would test a different Robot Framework version than the project pins

### Requirement: Project directory targeting and opt-in project-env install

The installer SHALL accept an explicit project-directory flag that sets both where the agent config
is written and which project environment is inspected, and SHALL treat installing rf-mcp into the
project environment as an explicit opt-in that never happens silently.

#### Scenario: explicit project directory is honored and validated
- **WHEN** a user runs `robotmcp install --scope project -C <dir>`
- **THEN** both the config path and the project-environment detection use `<dir>`, and if `<dir>` does not look like a project the installer warns rather than silently writing to the wrong place

#### Scenario: installing into the project env requires opt-in
- **WHEN** rf-mcp is not present in the project environment
- **THEN** the installer installs rf-mcp into that environment only when the user passes the opt-in flag, and otherwise prefers the non-mutating overlay or reports the fallback

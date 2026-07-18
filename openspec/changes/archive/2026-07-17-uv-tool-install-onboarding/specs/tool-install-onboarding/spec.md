## ADDED Requirements

### Requirement: Standalone tool install yields an agent-ready server

rf-mcp SHALL be installable as a standalone CLI tool via `uv tool install
"rf-mcp[<extras>]"` such that a `robotmcp` executable is placed on the user's PATH
and speaks the MCP protocol over stdio without any repository checkout.

#### Scenario: Fresh tool install exposes a working stdio server
- **WHEN** a user runs `uv tool install "rf-mcp[web,api]"` in a clean environment and launches the resulting `robotmcp` command
- **THEN** the command starts an MCP stdio server that answers an `initialize` request and returns the tool list, and the RequestsLibrary, SeleniumLibrary, and Browser libraries are importable in the tool's environment

#### Scenario: API testing needs no post-install step
- **WHEN** a user installs `rf-mcp[api]` and drives the server to run a RequestsLibrary flow against a live HTTP API
- **THEN** the requests execute and assertions pass with no additional setup

### Requirement: `robotmcp init` prepares the environment and prints agent wiring

The `robotmcp init` subcommand SHALL prepare an installed rf-mcp for use without
starting the MCP server, and SHALL be idempotent and non-destructive.

#### Scenario: init reports libraries and prints the MCP config
- **WHEN** a user runs `robotmcp init`
- **THEN** it reports which optional test libraries are installed and prints a ready-to-paste MCP configuration containing `{"command": "robotmcp"}`, and it does not start the server

#### Scenario: init initializes the Playwright browser in the tool's own environment
- **WHEN** the Browser library is installed and the user runs `robotmcp init --browsers`
- **THEN** the Playwright browser is initialized by invoking the bundled browser initializer via the running interpreter (`sys.executable`), so it is installed into the same environment the installed Browser library imports from

#### Scenario: init advises instead of failing when an extra is missing
- **WHEN** browser initialization is requested but the `web` extra is not installed
- **THEN** init prints the exact command to add it (e.g. `uv tool install "rf-mcp[web]"`) and exits without raising

### Requirement: Diagnostic surface

rf-mcp SHALL expose `robotmcp --version` and a read-only `robotmcp doctor` command.

#### Scenario: version reflects the installed distribution
- **WHEN** a user runs `robotmcp --version`
- **THEN** it prints the version of the installed rf-mcp distribution as resolved from package metadata

#### Scenario: doctor reports installation health without mutation
- **WHEN** a user runs `robotmcp doctor`
- **THEN** it reports the version, the import status of each optional test library, whether the Playwright browser has been initialized, and whether Node.js is present, without modifying any state

### Requirement: Command name alias

The installed distribution SHALL provide both `robotmcp` and `rf-mcp` as entry
points to the same server.

#### Scenario: both command names launch the server
- **WHEN** rf-mcp is installed as a tool
- **THEN** both `robotmcp` and `rf-mcp` are available on PATH and start the same MCP server

### Requirement: Multi-agent installer registers rf-mcp into coding agents

rf-mcp SHALL provide `robotmcp install`, `robotmcp uninstall`, and `robotmcp list`
commands that register the rf-mcp MCP server into supported coding agents using each
agent's own configuration file and format. The registry SHALL cover Claude Code,
Codex, GitHub Copilot, opencode, Gemini CLI, Kilo Code, goose, and Cursor, and MAY
include additional agents whose configuration convention is confirmed.

#### Scenario: install writes each targeted agent's native MCP config
- **WHEN** a user runs `robotmcp install --agents claude-code,codex,gemini --what mcp`
- **THEN** the rf-mcp MCP server is added to each agent's own config file in that agent's format (e.g. Claude Code JSON `mcpServers`, Codex TOML `[mcp_servers]`, Gemini JSON `mcpServers`), and any MCP servers already configured in those files are preserved

#### Scenario: detection drives interactive selection
- **WHEN** a user runs `robotmcp install` interactively
- **THEN** agents detected on the machine are pre-selected, the user can adjust the selection, and `--agents detected` (non-interactive) targets exactly the detected agents

#### Scenario: dry-run changes nothing
- **WHEN** a user runs `robotmcp install --dry-run`
- **THEN** the planned changes are printed and no configuration file is modified

#### Scenario: an unconfirmed agent adapter is surfaced, not silently wrong
- **WHEN** an agent's MCP-config convention has not been confirmed (e.g. `pi`)
- **THEN** `robotmcp list` reports that adapter's status as planned/unconfirmed rather than writing a possibly-incorrect configuration

### Requirement: Reversible, non-destructive uninstall via a tracked manifest

The installer SHALL record every configuration change in a manifest that captures,
per agent and scope, which files were touched, whether rf-mcp created the whole file
or inserted a key, and a hash of the value written. `robotmcp uninstall` SHALL revert
only entries whose current content still matches the recorded hash.

#### Scenario: uninstall removes only unmodified rf-mcp entries
- **WHEN** a user runs `robotmcp uninstall` after an install
- **THEN** rf-mcp entries that are unchanged since install are removed, entries a user has since edited are left in place and reported, and no unrelated configuration is altered

### Requirement: Installer is extensible to future skills, subagents, and hooks

The installer SHALL accept a `--what` selector covering `mcp`, `skills`, `agents`,
and `hooks` so that future bundled assets can be installed through the same
mechanism and recorded in the same manifest.

#### Scenario: selectors without bundled assets no-op cleanly
- **WHEN** a user runs `robotmcp install --what skills,agents,hooks` and rf-mcp ships no such assets yet
- **THEN** the command completes successfully, reports that no assets of those kinds are bundled, and writes nothing for them

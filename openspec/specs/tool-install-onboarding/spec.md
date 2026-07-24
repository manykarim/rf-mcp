# tool-install-onboarding Specification

## Purpose
TBD - created by archiving change uv-tool-install-onboarding. Update Purpose after archive.
## Requirements
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

### Requirement: Project-aware overlay references the installed rf-mcp across platforms

The project-aware uv overlay SHALL add the same rf-mcp distribution the user already installed: a
version pin (`--with rf-mcp==<version>`) for a published install, or the local source directory
(`--with-editable <path>`) when rf-mcp was installed from a local or unpublished source recorded as a
`file://` direct URL. The installer SHALL convert a `file://` direct URL to a filesystem path with a
platform-correct routine so a Windows drive-letter URL resolves to the corresponding Windows path and
the source overlay is emitted on Windows, macOS, and Linux alike — never silently degrading a
local-source install to an unresolvable version pin on any platform.

#### Scenario: local source install overlays that source on POSIX
- **WHEN** the installed rf-mcp records a `file://` direct URL such as `file:///home/u/rf-mcp` and that directory exists
- **THEN** the resolved overlay adds `--with-editable /home/u/rf-mcp`, so the overlay uses the same rf-mcp source even when its version is not on PyPI

#### Scenario: local source install overlays that source on Windows
- **WHEN** the installed rf-mcp records a `file://` direct URL such as `file:///C:/work/rf-mcp` on Windows and that directory exists
- **THEN** the resolved overlay adds `--with-editable C:\work\rf-mcp` (the drive-letter URL converted to a valid Windows path), not a `--with rf-mcp==<version>` pin

#### Scenario: published install uses a version pin
- **WHEN** the installed rf-mcp has no `file://` direct URL (a normal PyPI install) and its version is known
- **THEN** the resolved overlay adds `--with rf-mcp==<version>`, unchanged from prior behaviour


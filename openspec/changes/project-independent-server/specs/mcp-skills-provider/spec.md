## MODIFIED Requirements

### Requirement: Multi-agent CLI onboarding and skill synchronization
The system SHALL provide CLI subcommands (`rf-mcp init`, `rf-mcp sync-skills`, `rf-mcp doctor`) that configure MCP server entries and optionally materialize skills into local agent directories for Claude Code, OpenAI Codex, Google Antigravity (`agy`), GitHub Copilot, and OpenCode.

#### Scenario: Synchronize MCP skills into local agent skill directory
- **WHEN** a user runs `rf-mcp sync-skills --agent claude` (or `--target .agents/skills`)
- **THEN** the CLI exports the active skill directories (or downloads skills from a configured remote `rf-mcp` URL) into the target agent skill directory

#### Scenario: sync-skills does not silently overwrite existing skills
- **WHEN** the target directory already contains a skill folder with the same name whose content differs
- **THEN** the CLI skips it and reports the conflict unless `--force` is passed

#### Scenario: init runs rf-mcp from the project environment
- **WHEN** a user runs `rf-mcp init --agent claude` in a uv project where `rf-mcp` is installed into the project environment (including an editable install from a source checkout)
- **THEN** the generated entry runs `uv run rf-mcp` with no project path, no `--directory`, no `-r`, no `--mode` and no `--transport`, never from the `rf-mcp` source checkout, so the same file works in any clone of the project

#### Scenario: init for a project venv without uv
- **WHEN** a user runs `rf-mcp init` in a project whose `rf-mcp` is installed in a venv and the project has no `pyproject.toml`
- **THEN** the generated entry runs that venv's interpreter with `-m robotmcp` and no further arguments

#### Scenario: doctor verifies session support and custom hints
- **WHEN** a user runs `rf-mcp doctor`
- **THEN** it reports the `robotcode` version, whether `session`, `repl`, `analyze`, and `doc` subcommands are available, whether the project `hints.toml` parses, and the active skill roots

#### Scenario: Skill covers the key upstream workflows
- **WHEN** the built-in skill and its references are read
- **THEN** they cover mode routing, results-first triage without parsing `output.xml`, longname selection, analyze-before-run with the exit bitmask, one complete input per `send`, session reuse and announcing open sessions, sandbox honesty, `rf_repl save` with `keep_failed`, the pinned full-run / `rerunfailed` / merge workflow, `profile_show`, and RPA naming

## ADDED Requirements

### Requirement: init registers a global install at user level
`rf-mcp init --scope user` SHALL register an `rf-mcp` stdio entry in each selected agent's user-level configuration, so that every project the agent opens can use it. The entry SHALL contain no project path. It SHALL name the `rf-mcp` executable by its full path when the folder holding it is not on `PATH`.

#### Scenario: Global install registered for Claude Code
- **WHEN** a user who installed rf-mcp with `uv tool install` runs `rf-mcp init --scope user --agent claude --write`
- **THEN** Claude Code's user-level configuration gains an `rf-mcp` stdio entry that starts the installed `rf-mcp` without arguments, and no file in the current folder is written

#### Scenario: Preview without writing
- **WHEN** a user runs `rf-mcp init --scope user --agent all` without `--write`
- **THEN** the CLI prints, for each agent, the file it would change and the entry it would add, and changes nothing

### Requirement: init project scope with a global install
`rf-mcp init --scope project` run from a global install SHALL write project-level entries that start `rf-mcp` by name, without a project path, so the file can be committed for teammates who also install rf-mcp globally.

#### Scenario: Global install shared through the project
- **WHEN** a user with a global rf-mcp runs `rf-mcp init --scope project --agent copilot --write` in a project without rf-mcp in its environment
- **THEN** `.vscode/mcp.json` gains an `rf-mcp` stdio entry whose command is `rf-mcp` with no arguments

### Requirement: init merges into existing agent configuration
When `rf-mcp init --write` targets a configuration file that already exists, the system SHALL add or replace only its own `rf-mcp` entry and SHALL keep every other server and setting in that file. When the file cannot be parsed, it SHALL change nothing and report the file and the parse error.

#### Scenario: Other servers survive
- **WHEN** `.mcp.json` already defines the servers `github` and `rf-mcp` (an old entry with absolute paths) and the user runs `rf-mcp init --agent claude --write`
- **THEN** `.mcp.json` still defines `github` unchanged, and `rf-mcp` now holds the new entry

#### Scenario: Unparsable file is left alone
- **WHEN** `~/.codex/config.toml` contains invalid TOML and the user runs `rf-mcp init --scope user --agent codex --write`
- **THEN** the file is unchanged and the CLI exits non-zero naming the file and the parse error

### Requirement: Project and global registrations share one server name
Entries written by `rf-mcp init` in every scope SHALL use the server name `rf-mcp`, so that an agent that ranks project-level configuration above user-level configuration uses a project's own rf-mcp in that project and the global one elsewhere.

#### Scenario: Both scopes registered
- **WHEN** a user has run `rf-mcp init --scope user` and a project has its own entry from `rf-mcp init --scope project`
- **THEN** both entries are named `rf-mcp`

### Requirement: doctor reports root and environment resolution
`rf-mcp doctor` SHALL report how rf-mcp is installed (project environment or global tool), the project root it resolves and which rule found it, the environment and `robotcode` a call for that root would use and which resolution step chose them, whether `uv` is available, and the project interpreter's Python version. It SHALL exit non-zero when calls for that root would fall back to rf-mcp's own `robotcode` although the root is not rf-mcp's own environment.

#### Scenario: Global install in a project without robotcode
- **WHEN** a user with a global rf-mcp runs `rf-mcp doctor` in a project whose `.venv` holds Robot Framework but no robotcode, with uv installed
- **THEN** the report shows the install kind `global`, the project root and the marker that found it, the environment `<root>/.venv` with robotcode layered on by uv at rf-mcp's pinned version, and exits `0`

#### Scenario: Nothing can run the project
- **WHEN** uv is missing and the project environment holds no robotcode
- **THEN** the report says the project's libraries will not be visible, names both remedies, and exits non-zero

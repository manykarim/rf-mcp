# Spec: installer-cli-safety

## ADDED Requirements

### Requirement: `--dry-run` never mutates any state
A command invoked with `--dry-run` SHALL NOT modify the filesystem, install or remove any
package, or alter any environment — including via helpers reached during launch resolution.
Every mutation the command would perform SHALL be reported as a plan, and no result status
SHALL claim an action was performed. This applies to `--into-project`, which installs
packages into the detected project environment.

#### Scenario: dry-run with --into-project installs nothing
- **WHEN** `robotmcp install --dry-run --yes --into-project -C <project>` runs against a project whose environment lacks rf-mcp
- **THEN** no package is installed into that environment, no `uv pip install` is executed, and the reported status describes the install as planned rather than performed

#### Scenario: dry-run writes no config file
- **WHEN** `robotmcp install --dry-run` targets an agent with no existing config
- **THEN** no file or directory is created, and the plan names the path that would be written

### Requirement: Exit codes reflect the outcome
Onboarding commands SHALL return a non-zero exit code when the requested work did not
succeed, and zero only when it did. A refused or unverified install, a failed browser
initialization, and an unknown argument value SHALL each produce a non-zero status. Commands
SHALL NOT report success for an action that was not performed.

#### Scenario: refused install exits non-zero
- **WHEN** `robotmcp install --command /nonexistent/bin/foo` fails verification and writes no config
- **THEN** the command exits non-zero and names the reason

#### Scenario: failed browser initialization exits non-zero
- **WHEN** `robotmcp init` attempts browser initialization and it fails
- **THEN** the command exits non-zero rather than 0

#### Scenario: a fully successful install exits zero
- **WHEN** every targeted agent is written successfully
- **THEN** the command exits 0

### Requirement: Unknown argument values are rejected with actionable messages
An unrecognised value for `--agents`, `--what`, `--env`, or `--attach` SHALL cause the
command to fail with a message naming the offending token and listing the accepted values,
in the manner `--scope` already does. No unknown value SHALL be silently discarded, and a
value list containing both valid and invalid tokens SHALL report the invalid ones rather
than proceeding with the valid subset only. An empty `--agents` value SHALL NOT be treated
as a request to install into all detected agents.

#### Scenario: an agent-id typo is an error
- **WHEN** a user runs `robotmcp install --agents claude` and no adapter has that id
- **THEN** the command fails, names `claude` as unknown, and lists the valid agent ids

#### Scenario: a partially valid agent list reports the invalid token
- **WHEN** a user runs `robotmcp install --agents claude-code,nosuchagent`
- **THEN** the command reports `nosuchagent` as unknown instead of silently installing only `claude-code`

#### Scenario: malformed --env is rejected
- **WHEN** a user passes `--env FOO` or `--env =bar`
- **THEN** the command fails naming the malformed entry, rather than dropping it and reporting the install as successful

#### Scenario: agent ids are discoverable
- **WHEN** a user runs `robotmcp list`
- **THEN** the output includes the id accepted by `--agents` for each agent, not only its display name

### Requirement: Declining or aborting the installer writes nothing
A user who declines the interactive confirmation, supplies an empty agent selection, or
interrupts the process SHALL end the run with nothing written, a message stating that
nothing was changed, and a non-zero exit code. An empty agent selection SHALL NOT be
reinterpreted as "all detected agents".

#### Scenario: answering no to the confirmation prompt installs nothing
- **WHEN** a user runs `robotmcp install` on a terminal and answers `n` to `Register rf-mcp into these? [Y/n]`
- **THEN** no agent configuration is written, the output says the install was cancelled, and the exit code is non-zero

#### Scenario: an empty --agents value is not "everything"
- **WHEN** a user runs `robotmcp install --agents ""` (for example from a shell variable that expanded to nothing)
- **THEN** the command fails with a message about the empty selection, rather than installing into every detected agent

#### Scenario: interrupting the installer leaves no partial state
- **WHEN** a user presses Ctrl-C during `robotmcp install`
- **THEN** the command reports that it was cancelled and exits non-zero, without a Python traceback

### Requirement: Writing a config never alters unrelated content
Reading and rewriting an agent config SHALL preserve every part of the document the
installer did not intend to change, including the contents of string literals. Trailing-comma
normalization SHALL NOT alter characters inside strings. Where a format's comments or
structural features cannot be preserved, the behaviour SHALL be stated in user-facing
documentation for that format and SHALL NOT be contradicted elsewhere.

#### Scenario: JSONC string contents survive a rewrite
- **WHEN** a JSONC config contains a string value such as `"Close the brace like this: { a, } and the bracket like [ b, ]"` and the installer rewrites the file
- **THEN** that string is byte-identical afterwards

#### Scenario: YAML anchors are not silently flattened
- **WHEN** a goose YAML config uses an anchor and merge key (`defaults: &defaults` with `<<: *defaults`) and the installer rewrites the file
- **THEN** the anchor relationship is preserved, or the installer declines to rewrite the file and explains why

#### Scenario: an unrelated server entry and unrelated keys survive
- **WHEN** a config already contains another MCP server and unrelated top-level keys
- **THEN** both are present and unchanged after install and after uninstall

### Requirement: Each adapter writes the file and entry shape its agent reads
An adapter SHALL write to the path its target agent actually reads, using that agent's entry
shape for the server command. An install SHALL NOT report success when it has written a file
the target agent does not consume.

#### Scenario: kilo config matches the format Kilo reads
- **WHEN** `robotmcp install --agents kilo` runs for a project
- **THEN** the written filename and the server entry's `type` and `command` shape match the configuration Kilo actually loads

### Requirement: `uninstall` removes exactly what was selected
`uninstall` SHALL restrict removal to entries matching the requested scope and project
directory, SHALL NOT require the target agent to be currently detected, and SHALL continue
processing remaining entries when one config file cannot be parsed.

#### Scenario: uninstall honours the project directory
- **WHEN** entries exist for two projects and `robotmcp uninstall -C <projectA>` runs
- **THEN** only projectA's entry is removed and projectB's entry is left intact

#### Scenario: uninstall works for an undetected agent
- **WHEN** an entry was installed for an agent that is not currently detected and bare `robotmcp uninstall` runs
- **THEN** the entry is removed rather than reported as `Nothing to do.`

#### Scenario: one corrupt file does not abort the run
- **WHEN** one targeted config file is unparseable and others are healthy
- **THEN** the healthy entries are still removed and the unparseable file is reported with its path

### Requirement: Malformed input produces a diagnosis, not a traceback
A config file that cannot be parsed SHALL produce a message naming the file path and the
parse problem, and SHALL NOT surface a raw Python traceback. A `--project-dir` that does not
exist or is not a directory SHALL be reported by every subcommand that accepts it, and an
empty value SHALL NOT be silently reinterpreted as the current directory.

#### Scenario: corrupt config is explained
- **WHEN** a targeted config file contains invalid JSON or TOML
- **THEN** the output names the path and the parse error, and the command exits non-zero without a traceback

#### Scenario: doctor validates its project directory
- **WHEN** `robotmcp doctor -C /nonexistent/xyz` or `-C <a regular file>` runs
- **THEN** the invalid path is reported, matching the warning `install` already emits

### Requirement: The onboarding subcommands are discoverable from the CLI itself
`robotmcp --help` and `robotmcp -h` SHALL list the onboarding subcommands. An unrecognised
first argument SHALL be reported by the onboarding parser with a suggestion, without loading
the MCP server module. Running `robotmcp` with no arguments on an interactive terminal SHALL
explain that it is starting the MCP server on stdin and how to reach the subcommands.

#### Scenario: help lists the subcommands
- **WHEN** a user runs `robotmcp --help`
- **THEN** `init`, `install`, `uninstall`, `list` and `doctor` are listed

#### Scenario: a mistyped subcommand is diagnosed quickly
- **WHEN** a user runs `robotmcp instal`
- **THEN** the error names the unknown subcommand and suggests `install`, without printing library-availability warnings from the server module

#### Scenario: bare invocation on a terminal explains itself
- **WHEN** a user runs `robotmcp` with no arguments on a TTY
- **THEN** the output states that the MCP server is reading from stdin and points to `robotmcp --help`

### Requirement: Browser download happens only when requested
`robotmcp init` SHALL NOT download a browser unless `--browsers` was passed. Without the
flag, an uninitialized Playwright installation SHALL be reported together with the command
that initializes it.

#### Scenario: plain init does not download
- **WHEN** `robotmcp init` runs in an environment where `robotframework-browser` is importable but uninitialized
- **THEN** no download is attempted, and the output reports the browser as uninitialized and names the command to initialize it

### Requirement: Diagnostics report desktop support and reflect the running install
`robotmcp doctor` SHALL report desktop (PlatynUI) availability alongside the other test
libraries, with the install command that actually works for the current resolver. Statements
about which project libraries rf-mcp can see SHALL be derived from what this installation
actually provides, not from a static catalogue of what an extra would provide. The
REGISTERED status reported by `robotmcp list` SHALL be derived from the agent's configuration.

#### Scenario: desktop availability is reported
- **WHEN** `robotmcp doctor` runs in an environment without PlatynUI
- **THEN** a desktop row reports it as unavailable and names the command that installs it

#### Scenario: doctor does not contradict itself
- **WHEN** a project depends on a library that this rf-mcp installation does not provide
- **THEN** that library is reported as not reachable, and the output does not simultaneously claim the project needs no extra libraries

#### Scenario: a hand-pasted config reports as registered
- **WHEN** a user pastes `init`'s MCP snippet into an agent config without running `robotmcp install`
- **THEN** `robotmcp list` reports that agent as registered

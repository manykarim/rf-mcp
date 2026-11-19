# Spec Delta

## MODIFIED Requirements

### Requirement: Stateful multi-turn REPL exploration and suite export via robotcode session
The system SHALL expose an `rf_repl` tool (`start`, `send`, `wait`, `status`, `interrupt`, `list`, `save`, `stop`) backed by `robotcode -f json session` (`session start repl`, `session send`, `session wait`, `session status`, `session interrupt`, `session list`, `session stop`) that maintains live library instances, variables (`${_}`), and imports across turns and exports runnable `.robot` files via `.save`.

#### Scenario: Execute keywords across turns and export runnable suite
- **WHEN** an agent starts an `rf_repl` session (`robotcode -r <root> -f json session start repl`), sends `Import Library    Collections` and `${items}=    Create List    a    b` via `session send`, and calls `save` with `test_name="Verify List"` and `path="tests/list.robot"`
- **THEN** the session persists `${items}` across turns and writes `tests/list.robot` via `.save -t "Verify List" tests/list.robot` with `Library    Collections` hoisted into `*** Settings ***` and only passing statements in `*** Test Cases ***`

#### Scenario: Save keeping failed inputs for review
- **WHEN** an agent calls `rf_repl` `save` with `keep_failed=true`
- **THEN** the system sends `.save --keep-failed [-t "Name"] <path>` so failed statements are kept (commented) in the exported file

#### Scenario: Hot-reload modified Python library in REPL session via restart
- **WHEN** an agent edits a project Python library or `.resource` file during an open `rf_repl` session, calls `rf_repl` with `action="stop"` and the `session_id`, then calls `action="start"` with the same start options
- **THEN** the new session loads the edited code, and its `session_id` is the one the agent uses from then on; this stop and start replaces the `restart` action, because robotcode 2.9.0.dev214 removed `session restart`

#### Scenario: Capture native ARIA snapshot in REPL
- **WHEN** an agent sends `Get Aria Snapshot    css=body` inside an `rf_repl` session with `Browser` imported
- **THEN** the session returns the native Playwright ARIA YAML snapshot and assigns it to `${_}`

#### Scenario: Batch several statements in one send
- **WHEN** an agent sends a `statement` of several lines (or a `GROUP    name` ... `END` block)
- **THEN** the statements run in order in one call and stop at the first failure with `status="failed"` and the failing message, and a contract test pins this `robotcode session send` behaviour (its help text describes INPUT as one whole input)

#### Scenario: rf_repl description teaches batching and inspection
- **WHEN** a client lists tools
- **THEN** the `rf_repl` description says that several lines run in one send and stop at the first failure, that FOR/IF/TRY blocks go in one send, and names the inspection inputs `Get Aria Snapshot`, `.kw <text>`, `.doc`, `.vars --user` and `.imports` (2026-11-05 guidance-only experiment: Haiku T1 went from 79 single-line sends / 125 turns / $0.56 to 10 sends / 50 turns / $0.19)

### Requirement: Stateful multi-turn live test debugging via robotcode session debug
The system SHALL expose an `rf_debug` tool (`start`, `send`, `wait`, `status`, `result`, `interrupt`, `list`, `stop`) backed by `robotcode -f json session` (`session start debug`, `session send`, `session wait`, `session interrupt`, `session stop`) that pauses on uncaught failures or breakpoints, defaults to `--output NONE --log NONE --report NONE`, and supports stack/variable inspection, stepping, and live keyword evaluation.

#### Scenario: Pause on uncaught test failure and evaluate fix in paused frame without overwriting output.xml
- **WHEN** an agent starts `rf_debug` targeting a failing test with `by_longname=["MyProject.Auth.Valid Login"]` and sends `.where`, `.vars --user`, and a candidate keyword at the stop
- **THEN** the system starts `robotcode -r <root> -f json session start debug --output NONE --log NONE --report NONE -bl "MyProject.Auth.Valid Login"`, pauses before stack unwinding without overwriting the workspace's existing `output.xml`, returns the live call stack and frame variables, and executes the candidate keyword against the live paused context

#### Scenario: Break-control options map to real robot-debug flags
- **WHEN** an agent calls `rf_debug` `start` with `no_break_on_exception=true`, `break_on_failed_test=true`, or `break_on_failed_suite=true`
- **THEN** the system passes `--no-break-on-exception`, `--break-on-failed-test`, or `--break-on-failed-suite` respectively, and the debug run starts (no `option ... not recognized`, exit `252`)

#### Scenario: Session status reflects the run's exit code
- **WHEN** `robotcode session start debug` returns `{"status":"passed","exit_code":252}` because the run could not start
- **THEN** `rf_debug` reports `status="error"` with `run_exit_code=252` and the robotcode error text, not `status="passed"`

#### Scenario: Batch keyword lines and debugger commands in one send
- **WHEN** an agent sends `"${y}=    Set Variable    ${x}-b
Log    ${y}
.vars --user
.where"` to a paused `rf_debug` session
- **THEN** the system sends the keyword lines as one `session send` (they run in order in the paused frame and stop at the first failure) and each dot-command as its own `session send`, returns the outputs in order in one response, stops at the first failed part, and the `rf_debug` description teaches this batching (2026-11-06: `session send` runs keyword lines in a debug session like the REPL, but a `.vars` line inside a batch fails as `No keyword with name '.vars --user' found`; Sonnet T3 spent 24 turns on single `Evaluate JavaScript` probes)

#### Scenario: Wait on or interrupt a long-running debug or REPL turn
- **WHEN** a `send` or `start` action exceeds `--timeout` and `robotcode session` returns exit code `124` (`status="running"`)
- **THEN** the agent can call `rf_debug` or `rf_repl` with `action="wait"` (`robotcode -f json session wait <id>`) to collect the remaining output or `action="interrupt"` (`robotcode session interrupt <id>`) to pause/cancel execution

## ADDED Requirements

### Requirement: Sessions are reloaded by stop and start
`rf_repl` and `rf_debug` SHALL NOT offer a `restart` action, and SHALL NOT run `robotcode session restart`, which robotcode 2.9.0.dev214 removed. Their `action` descriptions and the `rf-robotcode` skill SHALL say that a session loads Python libraries and resource files once, and that edits take effect after `stop` and a new `start` with the same options. `rf-mcp doctor` SHALL NOT require a `session restart` subcommand.

#### Scenario: Action descriptions teach reloading
- **WHEN** a client lists tools
- **THEN** the `rf_repl` and `rf_debug` `action` descriptions list no `restart`, and say that edited libraries and resources take effect after `stop` and `start`

#### Scenario: A restart call is rejected
- **WHEN** an agent calls `rf_repl` or `rf_debug` with `action="restart"`
- **THEN** the call fails argument validation, the error lists the valid actions, and no `robotcode` command runs

#### Scenario: doctor accepts robotcode without session restart
- **WHEN** a user runs `rf-mcp doctor` with robotcode 2.9.0.dev214, whose `session` command has no `restart` subcommand
- **THEN** the report says robotcode meets the minimum version, and `restart` is not listed as a missing session subcommand

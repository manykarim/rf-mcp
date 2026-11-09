# Spec Delta

## Purpose

Defines the core `robotcode`-backed MCP tool surface for discovering, analyzing, documenting, executing, inspecting, interactively exploring (`session start repl`), debugging (`session start debug`), and diagnosing Robot Framework suites and tasks.

## ADDED Requirements

### Requirement: Suite, metadata, and configuration discovery via robotcode
The system SHALL expose an `rf_discover` tool that queries `robotcode` (`discover`, `config`, and `profiles`) in JSON format to return suites, tests, tasks, tags, test metadata (`metadata --values`), profiles, and resolved project configuration without executing tests.

#### Scenario: Discover tests and longnames in a workspace
- **WHEN** an agent calls `rf_discover` with `action="tests"` and a workspace root
- **THEN** the system invokes `robotcode -r <root> -f json discover tests` (or returns the equivalent client command in `client` mode) and returns the discovered test items including their `longname`, source path, and line number

#### Scenario: Discover test metadata keys and values
- **WHEN** an agent calls `rf_discover` with `action="metadata"` and `include_values=True`
- **THEN** the system invokes `robotcode -r <root> -f json discover metadata --values` and returns the metadata keys and values defined across the suite tree

#### Scenario: Inspect available robot.toml profiles
- **WHEN** an agent calls `rf_discover` with `action="profiles"`
- **THEN** the system returns the list of profiles configured in `robot.toml` / `pyproject.toml` via `robotcode -r <root> -f json profiles list`

#### Scenario: Show the effective configuration of a profile
- **WHEN** an agent calls `rf_discover` with `action="profile_show"` and `profile="ci"`
- **THEN** the system runs `robotcode -r <root> -p ci -f json profiles show` (the effective config, unlike `config show`, which applies no profile)

#### Scenario: Search tests and aggregate tags instead of grepping
- **WHEN** an agent calls `rf_discover` with `action="tests"` and `search="Open Browser"`, or with `action="tags"`, or with `action="files"`
- **THEN** the system passes `--search "Open Browser"` / runs `discover tags` / runs `discover files` respectively, so the agent never needs to grep `.robot` files

#### Scenario: Read several project files in one call
- **WHEN** an agent calls `rf_discover` with `action="read"` and `paths=["robot.toml", "resources/*.resource", "docs/conventions.md"]`
- **THEN** the system returns the text of every matching file under the project root in one response (each file once, keyed by its root-relative path, truncated at the stated output limit with the files left out named), refuses paths outside the root, and runs no robotcode command (2026-11-06 demoshop sweep: tool-only agents spent 9-18 turns per run on one `Read` each, 109 of 327 tool turns, where a shell agent used one `cat`)

### Requirement: Compact, non-duplicated tool responses
The system SHALL return each payload once. When `robotcode` JSON output is parsed into `data`, the raw `stdout` SHALL be omitted. In `server` mode `shell_command` SHALL be omitted. `rf_discover action="tests"|"tasks"` SHALL project each item to `longname`, `relSource:lineno`, and `tags` by default (`verbose=true` returns the full robotcode item). Large text output SHALL be truncated with a stated limit and a pointer to a narrower query.

#### Scenario: Discover tests response size
- **WHEN** an agent calls `rf_discover action="tests"` on a project with 7 tests
- **THEN** the response contains each longname, source:line, and tags once, contains no `stdout` copy of the JSON, contains no `id`/`uri`/`range`/absolute `source` fields, and is no larger than 2x robotcode's text output for the same query

#### Scenario: Server-mode response has no client command
- **WHEN** any tool runs in `server` mode
- **THEN** the response contains `command` (for transparency) but no `shell_command`

### Requirement: Tool descriptions route the agent by intent
Each `rf_*` tool description SHALL begin with a "Use when" sentence naming the intents it serves and the anti-pattern it replaces, and each parameter SHALL carry a one-line description. Agents such as Claude Code defer MCP tool schemas behind tool search, so these descriptions and the server `instructions` decide whether the agent picks an `rf_*` tool.

#### Scenario: Descriptions name the replaced anti-pattern
- **WHEN** a client lists tools
- **THEN** `rf_discover` says to use it instead of grepping `.robot` files, `rf_results` says it is the first step for "why did X fail?" instead of reading `output.xml`, `rf_analyze` says to run it before any run and for checks without running, `rf_debug` says to use it for any real failing test, and `rf_repl` says to use it for exploring with no existing test in play

#### Scenario: Agent chooses rf_results for a failure question
- **WHEN** an agent with `rf-mcp` connected (and Bash blocked) is asked "why did test X fail?" in a project with a recorded `output.xml`
- **THEN** the agent calls `rf_results` before reading project files and does not `Grep`/`Read` `output.xml` (measured by the agent usefulness evaluation for Haiku 5.5 and Sonnet 5.5)

#### Scenario: Descriptions carry common mistakes and stay short
- **WHEN** a client lists tools
- **THEN** each `rf_*` description names at least one common mistake it prevents (for example "`Run Keywords` cannot assign variables - send several lines instead") and is at most 600 characters long

### Requirement: Tolerant arguments for weak models
List parameters SHALL accept a JSON array string, a comma-separated string, or a single string in place of a list, and enumerated parameters SHALL accept values case-insensitively, without changing the published JSON schema. All tools SHALL name their sub-operation selector `action`.

#### Scenario: String in place of a list
- **WHEN** an agent calls `rf_discover` with `include_tags="smoke"`, `include_tags="[\"smoke\"]"`, or `include_tags="smoke,api"`
- **THEN** the call succeeds with `["smoke"]` or `["smoke", "api"]` instead of a `list_type` validation error

#### Scenario: Enum value in another case
- **WHEN** an agent calls `rf_results` with `action="Summary"`
- **THEN** the call runs `results summary` instead of failing with a `literal_error`

#### Scenario: Same selector name on every tool
- **WHEN** a client lists tools
- **THEN** every tool that has sub-operations selects them with a parameter named `action` (no `kind` or `mode`), so an agent that learned one tool does not call another with an unknown argument (Haiku called `rf_doc(kind="keyword")` and got `unexpected_keyword_argument`)

### Requirement: Headless static analysis via robotcode analyze
The system SHALL expose an `rf_analyze` tool that executes `robotcode -r <root> -f json analyze code` on target files or directories and returns structured diagnostics (`file`, `line`, `character`, `severity`, `code`, `message`) and summary counts.

#### Scenario: Detect undefined keyword without running suite
- **WHEN** an agent calls `rf_analyze` on a `.robot` file containing an undefined keyword
- **THEN** the system explicitly passes `-r <root>` to `robotcode` and returns a `KeywordNotFound` diagnostic with exact 0-indexed line/column coordinates and error summary counts

#### Scenario: Analyze exit code is decoded as a bitmask
- **WHEN** `robotcode analyze code` exits with `3`
- **THEN** the response reports `has_errors=true` and `has_warnings=true` (bits 1 error, 2 warning, 4 info, 8 hint) instead of treating the exit code as a command failure, and runtime-oriented hints (for example "variable expanded by shell") are not attached to static diagnostics

#### Scenario: Find unused keywords and variables
- **WHEN** an agent calls `rf_analyze` with `collect_unused=true` and no paths
- **THEN** the system passes `--collect-unused` on the whole project

### Requirement: Project-aware keyword and library documentation via robotcode doc
The system SHALL expose an `rf_doc` tool that delegates to `robotcode -r <root> -f json doc` (`lib`, `keywords`, `keyword`) to resolve documentation for installed libraries, `.resource` files, and custom `.py` libraries using the project's `robot.toml` `pythonpath` and profiles.

#### Scenario: Retrieve exact signature and documentation for a keyword
- **WHEN** an agent calls `rf_doc` with `action="keyword"`, `library="Browser"`, and `pattern="Get Aria Snapshot"`
- **THEN** the system returns the keyword's arguments, types, defaults, return type, and documentation resolved via `robotcode doc keyword`

#### Scenario: Search keywords across a project resource file
- **WHEN** an agent calls `rf_doc` with `action="keywords"`, `library="resources/common.resource"`, and an optional filter pattern
- **THEN** the system returns matching keywords defined or imported in that resource file

#### Scenario: Search keywords by several synonyms
- **WHEN** an agent calls `rf_doc` keywords listing with `pattern=["upload", "file", "attach"]`
- **THEN** the system passes every pattern to `robotcode doc keywords <target> upload file attach` and returns the keywords whose name matches any of them, and the description tells the agent to try synonyms because matching is by name, not meaning

### Requirement: Scoped test and task execution via robotcode robot
The system SHALL expose an `rf_run` tool that executes suites, tests, or tasks via `robotcode robot`, supporting selection by exact longname (`-bl`), tag (`-i`/`-e`), test metadata (`-btm`/`-ebtm`), RPA mode (`--rpa`), profile (`-p`), variables (`-v`), and wrapper (`--wrapper`).

#### Scenario: Execute a single test by longname preserving parent suite setups
- **WHEN** an agent calls `rf_run` with `by_longname=["MyProject.Auth.Valid Login"]` and `profile="ci"`
- **THEN** the system executes `robotcode -r <root> -p ci robot -bl "MyProject.Auth.Valid Login"` so parent `__init__.robot` setups and variables are applied

#### Scenario: Selection guard against single-file runs
- **WHEN** an agent calls `rf_run` or `rf_debug` with `paths` naming a single `.robot` file that lies inside the project's configured `paths`, or combines such a file path with `by_longname`
- **THEN** the response carries a `warning` explaining that the file becomes the top suite (parent `__init__.robot` setup and variables are skipped, and longnames lose their `Tests.` prefix) and names the equivalent `by_longname` selection taken from `rf_discover`

#### Scenario: Pin a full run for multi-failure triage
- **WHEN** an agent calls `rf_run` with `output="full.xml"`, and later `rf_run` with `rerunfailed="results/full.xml"` and `output="rerun1.xml"`, then `rf_results` with `action="merge"` on both files
- **THEN** the system passes `--output`, `--rerunfailed`, and runs `robotcode rebot --merge` so the agent can follow the pinned-run workflow (upstream Workflow E)

### Requirement: Server-side results inspection and diffing via robotcode results
The system SHALL expose an `rf_results` tool that queries `output.xml` or `output.json` via `robotcode -f json results` (`summary`, `show`, `log`, `stats`, `diff`), supporting tag (`-i`/`-e`), longname (`-bl`), and metadata (`-btm`/`-ebtm`) filtering without loading raw XML into the agent context.

#### Scenario: Inspect failure log and extract embedded screenshots
- **WHEN** an agent calls `rf_results` with `action="log"`, `failed_only=True`, and `extract_dir` set
- **THEN** the system runs `robotcode -f json results log --failed --extract <dir>` and returns the structured failure messages and extracted image paths

#### Scenario: Search, sort, and depth-limit results without reading output.xml
- **WHEN** an agent calls `rf_results` with `action="log"`, `search="invalid credentials"`, `max_depth=2`, and `level="WARN"`, or `action="show"` with `sort="elapsed"`, or `action="stats"` with `by=["tag","suite"]`
- **THEN** the system passes `--search`, `--max-depth`, `--level`, `--sort`, and `--by` respectively

#### Scenario: Result file is selected with -o, never positionally
- **WHEN** an agent wants to inspect a specific result file such as `results/output.xml`
- **THEN** `rf_results` accepts it only via `output_path` (emitted as `-o <file>` after the subcommand), exposes no generic positional `paths` list, and a value ending in `.xml`/`.json` passed anywhere else is rejected with a message naming `output_path`

#### Scenario: Diff against an explicit baseline
- **WHEN** an agent calls `rf_results` with `action="diff"`, `baseline="baseline/output.xml"`, and `only="new-failures"`
- **THEN** the system runs `robotcode -f json results diff baseline/output.xml --only new-failures` using the dedicated `baseline` parameter (not the generic `paths` list)

### Requirement: Stateful multi-turn REPL exploration and suite export via robotcode session
The system SHALL expose an `rf_repl` tool (`start`, `send`, `wait`, `status`, `interrupt`, `restart`, `list`, `save`, `stop`) backed by `robotcode -f json session` (`session start repl`, `session send`, `session wait`, `session status`, `session interrupt`, `session restart`, `session list`, `session stop`) that maintains live library instances, variables (`${_}`), and imports across turns and exports runnable `.robot` files via `.save`.

#### Scenario: Execute keywords across turns and export runnable suite
- **WHEN** an agent starts an `rf_repl` session (`robotcode -r <root> -f json session start repl`), sends `Import Library    Collections` and `${items}=    Create List    a    b` via `session send`, and calls `save` with `test_name="Verify List"` and `path="tests/list.robot"`
- **THEN** the session persists `${items}` across turns and writes `tests/list.robot` via `.save -t "Verify List" tests/list.robot` with `Library    Collections` hoisted into `*** Settings ***` and only passing statements in `*** Test Cases ***`

#### Scenario: Save keeping failed inputs for review
- **WHEN** an agent calls `rf_repl` `save` with `keep_failed=true`
- **THEN** the system sends `.save --keep-failed [-t "Name"] <path>` so failed statements are kept (commented) in the exported file

#### Scenario: Hot-reload modified Python library in REPL session via restart
- **WHEN** an agent edits a project Python library or `.resource` file during an open `rf_repl` session and calls `rf_repl` with `action="restart"` and the `session_id`
- **THEN** the system invokes `robotcode -f json session restart <session_id>`, starting a fresh OS process under the same `session_id` and initial start options so the updated code is loaded

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
The system SHALL expose an `rf_debug` tool (`start`, `send`, `wait`, `status`, `result`, `interrupt`, `restart`, `list`, `stop`) backed by `robotcode -f json session` (`session start debug`, `session send`, `session wait`, `session interrupt`, `session restart`, `session stop`) that pauses on uncaught failures or breakpoints, defaults to `--output NONE --log NONE --report NONE`, and supports stack/variable inspection, stepping, and live keyword evaluation.

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

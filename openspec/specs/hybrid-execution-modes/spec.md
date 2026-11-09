# hybrid-execution-modes Specification

## Purpose
Defines the three execution topologies (`server` local mode, `server` remote test runner mode with workspace sync, and zero-daemon `client` mode where the coding agent executes `robotcode` locally) and headless CI pipeline execution.

## Requirements
### Requirement: Local server execution mode over stdio and HTTP
The system SHALL support running in `server` execution mode (`--mode server`, default when `robotcode` is locally available) over `stdio` and Streamable HTTP transports, executing `robotcode` and `robotcode -f json session` subprocesses in the target workspace with `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`, and `ROBOTCODE_FORCE_AI_AGENT=1`, and stripping any leading UTF-8 BOM from received output.

#### Scenario: Local stdio execution on developer machine
- **WHEN** an agent connects to `rf-mcp` via `stdio` in `server` mode and calls `rf_analyze` or `rf_repl`
- **THEN** `rf-mcp` spawns `robotcode` in the local project environment with UTF-8 and AI-agent environment variables set and returns the structured execution output

#### Scenario: Non-ASCII robotcode output on a Windows cp1252 console
- **WHEN** `robotcode results log --failed` emits characters outside cp1252 (for example `❌`) while `rf-mcp` runs on Windows
- **THEN** the subprocess does not crash with `UnicodeEncodeError` because `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` are set, and the output reaches the agent intact

### Requirement: Subprocess timeouts return structured, recoverable results
The system SHALL catch `robotcode` subprocess timeouts and missing-executable errors in `server` mode and return a structured result (`status="timeout"` or `status="error"`, the elapsed time, partial output when available, and an `instruction` naming the parameter to raise or the install command) instead of raising a raw Python exception, and SHALL terminate the whole child process tree on timeout.

#### Scenario: Long suite exceeds rf_run timeout
- **WHEN** an agent calls `rf_run` with `timeout_seconds=3` on a suite that sleeps for 20 seconds
- **THEN** the tool returns `status="timeout"` with an instruction to retry with a larger `timeout_seconds` (or to narrow the selection), and no `robotcode`/`python` child process of that run remains alive afterwards

#### Scenario: robotcode is not installed in the server environment
- **WHEN** the `robotcode` executable cannot be resolved
- **THEN** the tool returns `status="error"` with the exact install command (`uv add --dev --prerelease=allow "robotcode[all]>=2.9.0.dev115"`) instead of a `FileNotFoundError` traceback

### Requirement: Remote test runner mode with workspace file synchronization
The system SHALL support running in `server` mode over Streamable HTTP on a remote host or container where browsers, devices, or desktop environments reside, allowing the client agent to synchronize inline `.robot`, `.resource`, or `robot.toml` files to the remote workspace before executing `robotcode`.

#### Scenario: Local agent runs modified suite on remote browser container
- **WHEN** a local coding agent connects to a remote `rf-mcp` HTTP server and passes inline `files` (`{"tests/login.robot": "..."}`) to `rf_run` or `rf_repl`
- **THEN** the remote `rf-mcp` writes the synchronized files into the workspace root on the remote runner (they persist there for later calls), executes `robotcode`, and returns the execution results, the list of `synced_files`, and any extracted artifacts

#### Scenario: Path traversal in synchronized files is rejected
- **WHEN** an agent passes `files={"../escape.robot": "..."}`
- **THEN** the tool fails with a path-traversal error and writes nothing outside the workspace root

### Requirement: Zero-daemon client execution mode for remote MCP servers
The system SHALL support running in `client` execution mode (`--mode client` or per-call `target="client"`) where a remote `rf-mcp` server does not require any custom client daemon and instead returns shell-safe `robotcode` and `robotcode -f json session` command plans for the coding agent to execute locally via its terminal tool.

#### Scenario: Synthesize PowerShell-safe session and CLI commands protecting RF variables
- **WHEN** an agent on Windows PowerShell calls `rf_repl` (`action="send"`) or `rf_run` against a `client`-mode `rf-mcp` server with Robot Framework variables such as `${items}`
- **THEN** `rf-mcp` returns a PowerShell-safe `robotcode -f json session send <id> '...'` (or `robotcode robot`) command that single-quotes `${items}` to prevent shell expansion and sets `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8`

#### Scenario: Synthesized PowerShell commands execute correctly
- **WHEN** the synthesized PowerShell 7 commands for `session start repl`, a `send` containing `${items}`, a multi-line `FOR ... END` block, and a `.save -t "It's Saved" <path>` are executed in sequence
- **THEN** every command exits `0`, `${items}` reaches `robotcode` unexpanded, the `FOR` block runs as one input, and the saved file contains a test named `It's Saved`

### Requirement: Client-mode paths belong to the client machine
In `client` mode the system SHALL treat `root`, `paths`, and file arguments as paths on the client machine. It SHALL NOT resolve them against the server filesystem, SHALL NOT inspect server-side `robot.toml` / `pyproject.toml` to decide default paths, and SHALL emit `-r <root>` verbatim only when the agent passes `root` (otherwise `robotcode` discovers the project root from the client's working directory).

#### Scenario: Windows server receives a POSIX client root
- **WHEN** a `client`-mode server running on Windows receives `root="/home/alice/proj"`
- **THEN** the synthesized command contains `-r /home/alice/proj` exactly, not `C:\home\alice\proj`

#### Scenario: No root passed in client mode
- **WHEN** an agent calls `rf_run` in `client` mode without `root`
- **THEN** the synthesized command contains no `-r` option and no server-derived default path such as `.` or the server's `--root`

### Requirement: Honest sandbox fallback for client-mode sessions
The system SHALL NOT offer a `fallback_command` for `rf_repl`, because a one-shot REPL cannot reproduce earlier turns. For `rf_debug` `start` it MAY offer a one-shot scripted `robot-debug --plain` fallback marked `one_shot=true`, whose `instruction` forbids escaping the agent sandbox and claiming values that were not observed (aligned with upstream eval `17-session-in-own-sandbox`).

#### Scenario: Synthesize client debug session command with one-shot sandbox fallback
- **WHEN** an agent in `client` mode requests `rf_debug` (`action="start"`) for a failing test longname
- **THEN** `rf-mcp` returns the primary `robotcode -f json session start debug --output NONE --log NONE --report NONE -bl "<longname>"` command plus a `fallback_command` marked `one_shot=true` that runs the same selection with `robot-debug --plain` and the scripted inputs `.where`, `.vars --user`, `.abort`, and the instruction forbids sandbox escapes and unobserved claims

#### Scenario: No misleading REPL fallback
- **WHEN** an agent in `client` mode calls `rf_repl` with `action="send"`
- **THEN** the response has no `fallback_command`

### Requirement: Headless CI pipeline agent compatibility
The system SHALL operate non-interactively inside CI pipelines (GitHub Actions, GitLab CI, Jenkins) when connected to headless coding agents (`claude -p`, `codex exec`, `opencode`, `agy`), honoring `robot.toml` CI profiles (`-p ci`) and display wrappers (`--wrapper "xvfb-run -a"`).

#### Scenario: CI agent triages failing run using CI profile
- **WHEN** a headless coding agent in CI invokes `rf_results`, `rf_debug`, and `rf_run` with `profile="ci"`
- **THEN** `rf-mcp` executes all `robotcode` and `robotcode session` commands with `-p ci` without interactive prompts, ANSI escape codes, or hanging pagers

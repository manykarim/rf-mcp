## MODIFIED Requirements

### Requirement: Local server execution mode over stdio and HTTP
The system SHALL support running in `server` execution mode (`--mode server`, default when `robotcode` is locally available) over `stdio` and Streamable HTTP transports, executing `robotcode` and `robotcode -f json session` subprocesses in the target workspace with `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`, and `ROBOTCODE_FORCE_AI_AGENT=1`, and stripping any leading UTF-8 BOM from received output.

#### Scenario: Local stdio execution on developer machine
- **WHEN** an agent connects to `rf-mcp` via `stdio` in `server` mode and calls `rf_analyze` or `rf_repl`
- **THEN** `rf-mcp` spawns `robotcode` in the environment resolved for the call's project root, with UTF-8 and AI-agent environment variables set, and returns the structured execution output

#### Scenario: Non-ASCII robotcode output on a Windows cp1252 console
- **WHEN** `robotcode results log --failed` emits characters outside cp1252 (for example `❌`) while `rf-mcp` runs on Windows
- **THEN** the subprocess does not crash with `UnicodeEncodeError` because `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` are set, and the output reaches the agent intact

### Requirement: Subprocess timeouts return structured, recoverable results
The system SHALL catch `robotcode` subprocess timeouts and missing-executable errors in `server` mode and return a structured result (`status="timeout"` or `status="error"`, the elapsed time, partial output when available, and an `instruction` naming the parameter to raise or the install command) instead of raising a raw Python exception, and SHALL terminate the whole child process tree on timeout.

#### Scenario: Long suite exceeds rf_run timeout
- **WHEN** an agent calls `rf_run` with `timeout_seconds=3` on a suite that sleeps for 20 seconds
- **THEN** the tool returns `status="timeout"` with an instruction to retry with a larger `timeout_seconds` (or to narrow the selection), and no `robotcode`/`python` child process of that run remains alive afterwards

#### Scenario: robotcode is not installed in the server environment
- **WHEN** no step of the environment resolution yields a runnable `robotcode` for the call's project root
- **THEN** the tool returns `status="error"` instead of a `FileNotFoundError` traceback, with an `instruction` that names the project root and the fix that fits how rf-mcp is installed: installing uv so rf-mcp can provide robotcode itself, or the exact command that adds `robotcode[all]` to the project environment, and `rf-mcp doctor` to check

## ADDED Requirements

### Requirement: Project root resolved per call
In `server` mode the system SHALL resolve each call's project root in this order: the call's `root` argument; the `--root` option or `RF_MCP_ROOT`; the client's MCP roots; the server's working directory. A root from the working directory SHALL be the nearest folder at or above it that contains `.robot.toml`, `robot.toml`, `pyproject.toml`, `.git` or `.hg`, or the working directory itself when there is none.

#### Scenario: The root argument wins
- **WHEN** a server started with `--root /srv/a` receives `rf_run` with `root="/srv/b"`
- **THEN** the run uses `/srv/b`

#### Scenario: A pinned root wins over the client's roots
- **WHEN** a server started with `--root /workspace` receives a call without `root` from a client that reports the MCP root `file:///home/dev/shop`
- **THEN** the call uses `/workspace`

#### Scenario: The client's single MCP root is used
- **WHEN** a server started without `--root` receives a call without `root` from a client that reports exactly one `file://` root
- **THEN** the call uses that folder as the project root

#### Scenario: A server started in a subfolder finds the project
- **WHEN** a server started without `--root` in `<project>/tests/ui`, by a client without MCP roots, receives a call without `root`, and `<project>/robot.toml` exists
- **THEN** the call uses `<project>` as the project root

### Requirement: Several client roots need an explicit choice
When no `root` argument or pinned root applies and the client reports more than one `file://` MCP root, the system SHALL use the root that contains the server's working directory. When none of them contains it, the tool SHALL return `status="error"` listing the reported roots and asking for the `root` argument. Roots that are not `file://` URIs SHALL be ignored.

#### Scenario: Multi-root workspace without a matching working directory
- **WHEN** a client reports the roots `file:///work/api` and `file:///work/web` and the server's working directory is `/home/dev`
- **THEN** the call fails with `status="error"`, names both folders, and tells the agent to pass `root`

#### Scenario: The working directory picks one of several roots
- **WHEN** a client reports the roots `file:///work/api` and `file:///work/web` and the server's working directory is `/work/web`
- **THEN** the call uses `/work/web`

### Requirement: Project environment resolved per root
In `server` mode the system SHALL choose the environment that runs `robotcode` for each project root in this order: an interpreter configured in `<root>/.robotframework/rf-mcp.toml`; `<root>/.venv`; the activated virtual environment (`VIRTUAL_ENV`) or a `robotcode` on `PATH`; rf-mcp's own `robotcode`. Every step SHALL leave the project's files and environment unchanged.

#### Scenario: Project install uses its own environment directly
- **WHEN** rf-mcp runs from `<root>/.venv` (installed into the project) and receives a call for `<root>`
- **THEN** it runs the `robotcode` of that environment directly, without uv

#### Scenario: A configured interpreter wins
- **WHEN** `<root>/.robotframework/rf-mcp.toml` sets `python = "envs/rf/bin/python"` and `<root>/.venv` also exists
- **THEN** robotcode runs with the interpreter `<root>/envs/rf/bin/python`

#### Scenario: Activated environment when the project has no .venv
- **WHEN** a global rf-mcp receives a call for a root without `.venv` and without configuration, and `VIRTUAL_ENV` points to an environment that holds a supported `robotcode`
- **THEN** that environment's `robotcode` runs the call

### Requirement: The project's own robotcode is used when supported
When the chosen project environment is not rf-mcp's own and contains `robotcode` at or above rf-mcp's minimum version with the `session` subcommands, the system SHALL run that `robotcode` directly. Otherwise it SHALL layer the `robotcode[all]` version rf-mcp itself runs onto the project interpreter with `uv run --no-project --python <interpreter> --with`, so that the project's libraries stay importable.

#### Scenario: Project venv without robotcode
- **WHEN** a global rf-mcp receives `rf_analyze` for a root whose `.venv` holds Robot Framework and RequestsLibrary but no `robotcode`
- **THEN** analysis reports no import error for `RequestsLibrary`, and the project's `.venv` still contains no `robotcode` afterwards

#### Scenario: Project pins an older robotcode
- **WHEN** a global rf-mcp receives a call for a root whose `.venv` holds `robotcode` 1.5.0
- **THEN** the call runs rf-mcp's pinned robotcode on the project interpreter, and the project's `.venv` still holds `robotcode` 1.5.0 afterwards

#### Scenario: Project has a supported robotcode
- **WHEN** a global rf-mcp receives a call for a root whose `.venv` holds a supported `robotcode`
- **THEN** that `robotcode` runs the call without uv

### Requirement: Layering requires uv
The system SHALL layer robotcode onto a project interpreter only when the `uv` executable is available. Without uv it SHALL continue with the next steps of the environment resolution, and when it falls back to rf-mcp's own robotcode it SHALL say that uv or a robotcode in the project environment is needed for the project's libraries.

#### Scenario: uv is missing
- **WHEN** `uv` cannot be found and a call targets a root whose `.venv` holds no `robotcode`
- **THEN** the call does not attempt to run uv, and its result names both remedies: installing uv, or adding `robotcode[all]` to the project environment

### Requirement: Fallback to rf-mcp's own robotcode is visible
When the system runs a call with rf-mcp's own `robotcode` for a root that is not rf-mcp's own environment, the result SHALL state that the project's libraries are not visible to this run and how to fix it, and it SHALL NOT present import errors that result from this fallback as defects in the project's tests.

#### Scenario: Global install, project without any environment
- **WHEN** a global rf-mcp receives `rf_analyze` for a root with no `.venv`, no configuration, no activated environment and no `robotcode` on `PATH`
- **THEN** analysis runs with rf-mcp's own robotcode, and the result says that project libraries are not visible and names `rf-mcp doctor`

### Requirement: Sessions work in a layered environment
REPL and debug sessions SHALL work when robotcode is layered onto a project interpreter: a session started by one call SHALL accept `send`, `status`, `result`, `interrupt`, `restart` and `stop` from later calls, each of which runs as its own process.

#### Scenario: REPL across separate calls
- **WHEN** a global rf-mcp starts `rf_repl` for a root whose `.venv` holds RequestsLibrary but no robotcode, then sends `Import Library    RequestsLibrary` and `${r}=    Evaluate    6*7`, then stops the session
- **THEN** both sends pass, the second reports `${r} = 42`, and no session for that root remains open

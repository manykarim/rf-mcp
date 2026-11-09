# Design

## Context

See `proposal.md` — Why for motivation.

`rf-mcp` currently embeds a ~90,210 LOC custom Python runtime (`src/robotmcp/`) that directly manipulates internal Robot Framework classes (`EXECUTION_CONTEXTS.current`, `Namespace`, `VariableScopes`), runs 4 separate keyword/library discovery layers in-process, reconstructs `.robot` files via 4,462 lines of regex/AST post-processing (`test_builder.py`), and parses DOM/ARIA snapshots in Python. Meanwhile, `robotcode` (`>=2.9.0.dev115`) provides native CLI subcommands (`discover`, `analyze code`, `doc`, `robot`, `rebot`, `results`, `session {start|send|wait|status|result|interrupt|restart|list|stop}`, `repl`, `robot-debug`, `config`, `profiles`) that resolve `robot.toml` profiles, `pythonpath`, and project virtualenvs in isolated OS processes.

## Goals / Non-Goals

**Goals:**
- Replace the ~90k LOC custom codebase with a clean ~2,200 LOC architecture built on `FastMCP` 3.x and `robotcode>=2.9.0.dev115` CLI.
- Support all three execution topologies (`Mode 1: Local Server`, `Mode 2: Remote Test Runner`, `Mode 3: Zero-Daemon Remote MCP + Client-Executed robotcode`) and headless CI pipelines using a single unified tool surface.
- Delegate multi-turn `rf_repl` and `rf_debug` sessions natively to `robotcode -f json session` (`start repl`, `start debug`, `send`, `wait`, `status`, `result`, `interrupt`, `restart`, `list`, `stop`) in `server` mode, and synthesize shell-safe `robotcode session` commands (plus per-command sandbox fallback commands) in `client` mode.
- Retain and modernize the Error Hinting system as a `robotcode`-native, declarative, and project-extensible engine (`.robotframework/hints.toml`).
- Serve layered skills over MCP (`SkillsDirectoryProvider`, SEP-2640 alignment, `ResourcesAsTools`) that coexist with `robotcodedev/robotframework-agent-plugins` (`next`) and `manykarim/robotframework-agentskills`.
- Provide built-in OpenTelemetry tracing and metrics extending `fastmcp.telemetry`.

**Non-Goals:**
- No backwards compatibility, shims, or deprecation aliases for the 62 legacy MCP tools.
- No custom background PTY/pipe/socket process pool in `rf-mcp` (`robotcode session` manages background `repl` and `robot-debug` processes natively via `multiprocessing.connection`).
- No custom DOM/ARIA snapshot parser (`Browser.Get Aria Snapshot` is used natively inside `robotcode session` / `repl` / `.robot` suites).
- No Django/Uvicorn web dashboard (`src/robotmcp/frontend/`).
- No in-process `sqlite-vec` / `model2vec` / `sentence-transformers` embedding databases.
- No custom client-side daemon to install on user machines for Mode 3.

## Decisions

### 1. Target Package Layout (`src/robotmcp/`)

```text
src/robotmcp/
├── __init__.py
├── entry.py              # CLI dispatcher: MCP server (`stdio` / `http`) or onboarding subcommands
├── config.py             # ServerConfig (execution_mode, root, profile, skills_dirs, hints_file, otel)
├── server.py             # FastMCP 3.x server wiring 8 `rf_*` tools + SkillsDirectoryProvider + ResourcesAsTools
├── telemetry.py          # OpenTelemetry tracer/meter helpers extending `fastmcp.telemetry`
├── runner/
│   ├── cli_runner.py     # Stateless `robotcode` subprocess executor & command synthesizer (`server` vs `client`)
│   ├── session_runner.py # Stateless `robotcode -f json session` wrapper for `rf_repl` & `rf_debug` + sandbox fallbacks
│   ├── shell_escape.py   # PowerShell & Bash command formatting (`PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`, `${var}` protection)
│   └── workspace_sync.py # Inline file synchronization for Mode 2 Remote Test Runner
├── hints/
│   ├── engine.py         # Pattern & context matcher returning ranked `Hint` objects
│   ├── builtin_rules.py  # Built-in rules for Browser, Selenium, Appium, Requests, PlatynUI, BuiltIn, REPL
│   └── loader.py         # TOML loader for project-custom `<workspace>/.robotframework/hints.toml`
├── skills/
│   ├── provider.py       # Layered `SkillsDirectoryProvider` + server `instructions` catalog builder
│   └── builtin/          # Bundled baseline skill(s) (`rf-robotcode/SKILL.md`)
└── onboarding/
    └── cli.py            # `rf-mcp init`, `rf-mcp sync-skills`, `rf-mcp doctor` for Claude, Codex, AGY, Copilot, OpenCode
```

- **Rationale**: Replaces 14 top-level packages and 12 DDD bounded contexts with 5 focused sub-packages. Because `robotcode>=2.9.0.dev115` includes native `robotcode session` management, `rf-mcp` does not even need a separate `sessions/` process manager package—`runner/session_runner.py` drives `robotcode -f json session` via stateless CLI calls.

### 2. Dual-Execution Architecture (`server` Mode vs. Zero-Daemon `client` Mode)

```text
+-------------------------------------------------------------------------+
|                    Tool Call (`rf_discover`, `rf_repl`, ...)            |
+-------------------------------------------------------------------------+
                                     |
                   +-----------------+-----------------+
                   |                                   |
        [mode == "server"]                  [mode == "client"]
   (Mode 1 Local / Mode 2 Remote Rig)    (Mode 3 Remote MCP, Zero Daemon)
                   |                                   |
                   v                                   v
  1. Optional inline `files` sync       1. Synthesize shell-safe `robotcode`
  2. Execute stateless `robotcode`         CLI or `robotcode session` command
     or `robotcode -f json session`        (+ sandbox fallback) for client OS
  3. If exit_code != 0 or status=="failed": 2. Return command plan + instruction
     run `HintEngine` and attach           to call `rf_diagnose` on failure
     top 3 ranked `hints`
```

- **Why this over a custom client-side daemon for Mode 3**: Coding agents (`claude`, `codex`, `agy`, `copilot`, `opencode`) already run on the client machine with local terminal access and `robotcode` installed. Having the remote `rf-mcp` server synthesize shell-safe `robotcode` and `robotcode session` commands, serve project skills/conventions over MCP, and diagnose errors via `rf_diagnose` requires zero extra client daemons or firewall tunnels.
- **Alternative considered**: MCP sampling or reverse WebSocket daemon on the client. Rejected because most coding agents do not expose client shell execution over MCP sampling, and a reverse daemon violates the zero-daemon requirement.

### 3. Native `robotcode session` Management for `rf_repl` and `rf_debug` (`src/robotmcp/runner/session_runner.py`)

- **How `robotcode session` (`>=2.9.0.dev115`) works**:
  - `robotcode [-p <profile>] [-r <root>] -f json session start {repl|debug} [--idle-timeout <min>] [--timeout <sec>] [ARGS]...` spawns `repl` or `robot-debug` with `--serve-session` in an isolated background OS process, registers a session entry (`repl-<8hex>` or `debug-<8hex>`) in the user's runtime directory (`platformdirs`), and returns `{"id": "...", "output": "...", "status": "passed"|"failed"|"running", "exit_code"?: int}`.
  - `robotcode -f json session send [--timeout <sec>] <ID> '<INPUT>'` sends a statement, a multi-line `FOR`/`IF`/`TRY`...`END` block, or a dot command (`.vars --user`, `.where`, `.list`, `.break`, `.step`, `.next`, `.return`, `.continue`, `.save -t "Name" file.robot`) over a local `multiprocessing.connection` channel and returns `{"output": "...", "status": "passed"|"failed"|"running", "exit_code"?: int}`.
  - Deterministic exit codes: `0` (`passed`), `1` (`failed`—session stays open at the prompt and `rf-mcp` attaches `HintEngine` suggestions), `124` (`running` after `--timeout` seconds—follow up with `session wait <ID>` or `session interrupt <ID>`), `252` (unknown session/error), `253` (aborted debug run).
  - `robotcode -f json session restart <ID>` starts a **new OS process that takes over the same `<ID>`, initial CLI flags, and runtime `.break` list**, picking up any edits made to `.py` libraries, `.resource`, or `.robot` files mid-session.
- **Protecting `output.xml` during `rf_debug`**:
  - By default, `robot-debug` / `session start debug` overwrites `output.xml` in the workspace, clobbering the recorded results of the last full test run (needed for `--rerunfailed` and `rf_results`).
  - `SessionRunner.start_debug()` appends `--output NONE --log NONE --report NONE` by default unless `keep_outputs=True` is explicitly passed.
- **In `client` mode (Mode 3) & Per-Command Sandbox Fallback**:
  - `SessionRunner` synthesizes the primary `robotcode -f json session {start|send|wait|status|result|interrupt|restart|list|stop}` command for the client shell.
  - Because some coding agent harnesses run each shell command in an isolated per-command sandbox or Windows Job Object where background processes cannot outlive the command that started them (`evals/17-session-in-own-sandbox`), `SessionRunner` in `client` mode also returns a `fallback_command` (using non-blocking `-c` / stdin piping into `robotcode repl --plain` or `robotcode robot-debug --plain`) so the agent can seamlessly fall back if `session send` reports `No session '<id>' is open`.

### 4. Shell & OS Safety Invariants (`PYTHONUTF8`, `PYTHONIOENCODING`, Variable Quoting)

- Empirical testing on Windows PowerShell revealed two critical failure modes when running `robotcode`:
  1. Double-quoted CLI strings cause PowerShell and Bash to expand Robot Framework variables (`${x}` -> `""`).
  2. PowerShell pipes (`| robotcode repl`) prepend a UTF-8 BOM (`\ufeff`) to the first line of `stdin`, causing `No keyword with name '\ufeffImport Library' found`, and Windows `cp1252` crashes on `robotcode`'s em-dashes (`0x97`).
- **Decision**: Both `server`-mode subprocess execution and `client`-mode command synthesis enforce:
  - `PYTHONUTF8=1`
  - `PYTHONIOENCODING=utf-8` (BOM-free UTF-8 on `stdin`/`stdout`/`stderr`; the value is `encoding[:errors]`, so the earlier `utf-8-sig:utf-8` named `utf-8` as an error handler and is invalid). A leading BOM in output is stripped by `rf-mcp`, and REPL input never goes through a PowerShell pipe (single-quoted `session send` arguments instead)
  - `ROBOTCODE_FORCE_AI_AGENT=1` (disables ANSI colors, pagers, and prompt-toolkit TUI popups)
  - Single-quoted arguments for `robotcode session send <ID> '<input>'` (and PowerShell single-quoted here-strings `@'\n...\n'@` for multi-line blocks) so `${var}`, `@{list}`, and `&{dict}` are never expanded by the shell and never pass through a BOM-injecting PowerShell pipe.

### 5. Declarative & Project-Customizable Error Hinting Engine (`src/robotmcp/hints/`)

- Instead of 1,512 lines of hardcoded regexes referencing legacy MCP tools, `HintEngine` evaluates `(keyword, arguments, error_text, libraries)` against a declarative rule list and outputs native Robot Framework / `robotcode` fix examples (`Get Aria Snapshot    css=body`, `Evaluate JavaScript    ${None}    () => ...`, `>> nth=0`, `.kw <pattern>`, `Import Library`, `${dict}[key]`).
- Project teams can place `.robotframework/hints.toml` in their repository:
  ```toml
  [[hints]]
  id = "corp-sso-expired"
  title = "Corporate SSO token expired"
  error_pattern = "401 Unauthorized.*SSO"
  keyword_pattern = ".*"
  message = "Refresh the session token using the Auth resource before calling the endpoint."
  fix_examples = ["Import Resource    resources/auth.resource", "Refresh SSO Token"]
  relevance = 99
  ```

### 6. Skills over MCP (`SkillsDirectoryProvider` + SEP-2640 + `ResourcesAsTools`)

- `rf-mcp` mounts FastMCP's `SkillsDirectoryProvider(roots=[...], reload=True)` across three layered roots (highest precedence first):
  1. Project-local `<workspace>/.robotframework/skills/`
  2. External plugin directories passed via `--skills-dir` / `RF_MCP_SKILLS_DIR` (e.g., `robotframework-agent-plugins`, `robotframework-agentskills`)
  3. Built-in `src/robotmcp/skills/builtin/` (`rf-robotcode`, aligned with `robotframework-agent-plugins@next` workflows)
- Because current coding agents consume MCP tools universally while SEP-2640 native client discovery is still rolling out, `rf-mcp` also applies FastMCP's `ResourcesAsTools(mcp)` transform and injects a compact skill catalog table into the MCP server `instructions`.

### 7. OpenTelemetry Observability (`src/robotmcp/telemetry.py`)

- FastMCP 3.4+ already includes `opentelemetry-api` and creates parent `SERVER` spans (`fastmcp.server.telemetry.server_span`) with W3C `traceparent` extraction from HTTP headers and MCP `_meta`.
- `src/robotmcp/telemetry.py` uses `opentelemetry.trace.get_tracer("rf-mcp")` and `opentelemetry.metrics.get_meter("rf-mcp")` to record child spans and metrics (`rf_mcp.commands.total`, `rf_mcp.command.duration_ms`, `rf_mcp.hints.generated`, `rf_mcp.skills.read`).
- When `--otel-endpoint` or `OTEL_EXPORTER_OTLP_ENDPOINT` is set (and the optional `[otel]` extra is installed), `telemetry.configure_otel()` initializes `TracerProvider` and `MeterProvider` with OTLP exporters; otherwise, `opentelemetry-api` acts as a zero-overhead no-op.

## Risks / Trade-offs

- **[Risk] `robotcode` CLI missing or older than `2.9.0.dev115` (`No such command 'session'`)** → **Mitigation**: `rf-mcp doctor` and tool error handlers check `robotcode --version` and subcommand availability (`runner`, `analyze`, `repl` extras) and return exact installation instructions (`uv add --dev --prerelease=allow "robotcode[all]>=2.9.0.dev115"` or `pip install --pre "robotcode[all]>=2.9.0.dev115"`). In `client` mode, `rf-mcp` itself does not need `robotcode` installed on the server host.
- **[Risk] Orphaned `robotcode session` background processes in `server` mode** → **Mitigation**: `robotcode session start` enforces a configurable `--idle-timeout` (default 15 minutes), and `rf-mcp` registers a FastMCP lifespan/shutdown hook that runs `robotcode -r <root> session stop --project` on exit.
- **[Risk] Per-command sandboxed agent shells killing background `robotcode session` processes in `client` mode** → **Mitigation**: `SessionRunner` in `client` mode includes a `fallback_command` (`robotcode repl` / `robotcode robot-debug` with `-c` or piped stdin) alongside the `robotcode session` command.
- **[Risk] Path outside workspace root causes `robotcode analyze code` to report `files: 0`** → **Mitigation**: `rf-mcp` always resolves and passes `-r <workspace_root>` explicitly to `robotcode` commands.

## Migration Plan

1. Remove legacy packages under `src/robotmcp/` (`components`, `domains`, `core`, `plugins`, `adapters`, `optimization`, `frontend`, `attach`, `models`, `compat`, `config`, `utils`, `container.py`) and obsolete unit tests tied to those deleted modules.
2. Simplify `pyproject.toml` dependencies (`robotcode[all]>=2.9.0.dev115` with `prerelease = "allow"` in `[tool.uv]`) and extras (`[otel]`, `[dev]`).
3. Implement `config.py`, `telemetry.py`, `hints/`, `skills/`, `runner/{cli_runner.py,session_runner.py,shell_escape.py,workspace_sync.py}`, `server.py`, and `onboarding/`.
4. Add unit and integration tests covering all 8 `rf_*` tools in both `server` and `client` modes, `robotcode session` lifecycles (`start`, `send`, `wait`, `status`, `result`, `interrupt`, `restart`, `list`, `stop`), the Hinting Engine (built-in + `hints.toml`), Skills over MCP (`skill://` + `ResourcesAsTools`), and OpenTelemetry spans/metrics.

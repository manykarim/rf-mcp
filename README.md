# rf-mcp — Robot Framework Agentic Engineering MCP Server

`rf-mcp` is a lean, customizable Model Context Protocol (MCP) server built on **[RobotCode](https://robotcode.io/) (`>=2.9.0.dev115`) CLI**, **`robotcode session`**, and **FastMCP Skills over MCP (SEP-2640)**.

It empowers coding agents (**Claude Code**, **OpenAI Codex**, **Google Antigravity (`agy`)**, **GitHub Copilot**, **OpenCode**) to discover, analyze, document, author (`repl`), debug (`robot-debug`), execute, and maintain Robot Framework tests and tasks across local machines, remote test runners, zero-daemon remote MCP setups, and CI pipelines.

---

## Key Capabilities

1. **8 Focused `robotcode`-Powered MCP Tools**:
   - `rf_discover`: Project suite, test, task, tag, file, test metadata (`--values`), profile (`profiles show` for a profile's effective settings), and merged `robot.toml` discovery (`robotcode -f json discover|config|profiles`), with `--search` and a compact `longname` / `source:line` / `tags` projection (`verbose=true` for full items).
   - `rf_analyze`: Millisecond headless LSP semantic static analysis (`robotcode -r <root> -f json analyze code`).
   - `rf_doc`: Project-aware library, `.resource`, and keyword documentation (`robotcode -f json doc {lib|keywords|keyword}`).
   - `rf_run`: Scoped test/task execution (`robotcode robot`, supporting `-bl <longname>`, `-btm <metadata>`, `--rpa`, `-p <profile>`, `--wrapper`, and inline file synchronization for remote test runners).
   - `rf_results`: Server-side querying, diffing and merging of `output.xml` / `output.json` (`robotcode -f json results {summary|show|log|stats|diff}` with `--search`, `--sort`, `--level`, `--max-depth`, `--by`, `--extract` screenshot extraction, and `rebot --merge`). Result files are chosen only with `output_path`.
   - `rf_repl`: Stateful multi-turn interactive exploration and test authoring (`start`, `send`, `wait`, `status`, `result`, `interrupt`, `restart`, `list`, `save`, `stop`) backed by `robotcode -f json session` (`session start repl`, `.save -t "Name" file.robot`, hot-reload via `session restart`, and native `Browser.Get Aria Snapshot`).
   - `rf_debug`: Stateful multi-turn live test debugger (`start`, `send`, `wait`, `status`, `result`, `interrupt`, `restart`, `list`, `stop`) backed by `robotcode -f json session` (`session start debug`, pausing on uncaught failures or `--break file.robot:line`, preserving `output.xml` via `--output NONE --log NONE --report NONE`, and inspecting `.where`, `.vars --user`, `.list`).
   - `rf_diagnose`: Ranked, library-aware error recovery hinting (Browser, Selenium, Appium, Requests, PlatynUI, BuiltIn, REPL syntax) extensible via `<workspace>/.robotframework/hints.toml`.

2. **3 Execution Topologies (`server`, `client`, and CI) — Zero Extra Client Daemons**:
   - **Mode 1: Full Local (`--mode server`, `stdio` or `http`)**: `rf-mcp` runs locally alongside your project and executes `robotcode` and `robotcode session` directly.
   - **Mode 2: Remote Test Runner (`--mode server --transport http`)**: `rf-mcp` and `robotcode` run on a remote test container/VM (with browsers/devices/displays), accepting inline `files` synchronization from the coding agent.
   - **Mode 3: Zero-Daemon Remote MCP + Local `robotcode` (`--mode client --transport http`)**: `rf-mcp` runs remotely while the coding agent and `robotcode` run on the developer's machine. `rf-mcp` serves project/ecosystem skills over MCP, synthesizes shell-safe local `robotcode` & `robotcode session` commands for PowerShell and Bash, and diagnoses failures via `rf_diagnose`. The agent's `root` is passed to `robotcode -r` verbatim (it is a path on the client machine); without one, `-r` is omitted. `rf_debug start` also returns a one-shot `fallback_command` for agent shells that kill background processes; `rf_repl` has none, because a one-shot REPL cannot replay earlier turns.

   Client-mode commands are executed by a test in both supported shells: **PowerShell 7+** and **bash**. Windows PowerShell 5.1 is untested (its quoting of native arguments differs). Commands set `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8` and `ROBOTCODE_FORCE_AI_AGENT=1`.

3. **Skills over MCP (SEP-2640 + `ResourcesAsTools`) & Plugin Coexistence**:
   - Serves layered `SKILL.md` directories over MCP (`skill://{name}/SKILL.md` + `list_resources` / `read_resource` tools for tool-only agents):
     1. Project-local `<workspace>/.robotframework/skills/`
     2. External plugin directories via `--skills-dir` / `RF_MCP_SKILLS_DIR` (compatible with [`robotcodedev/robotframework-agent-plugins`](https://github.com/robotcodedev/robotframework-agent-plugins/tree/next) and [`manykarim/robotframework-agentskills`](https://github.com/manykarim/robotframework-agentskills))
     3. Built-in skills: `rf-robotcode` (tool routing, triage, debugging, REPL authoring, results, analysis, configuration; maps the upstream robotcode practices P1-P46), `rf-locators` (Browser, SeleniumLibrary, AppiumLibrary and PlatynUI locators, shadow DOM and iframes, keyword per intent, library chooser) and `rf-requests` (RequestsLibrary for REST, SOAP/XML and GraphQL). Detail lives in `references/` files, e.g. `skill://rf-robotcode/references/triage.md`.
   - The server `instructions` are a short intent routing table plus the authoring loop and its finish condition (done only when `rf_analyze` is clean and `rf_run` passes).

4. **OpenTelemetry Observability**:
   - Built-in child spans (`rf_mcp.discover`, `rf_mcp.analyze`, `rf_mcp.doc`, `rf_mcp.run`, `rf_mcp.results`, `rf_mcp.repl`, `rf_mcp.debug`, `rf_mcp.diagnose`) nested under FastMCP server spans, plus metrics (`rf_mcp.commands.total`, `rf_mcp.command.duration_ms`, `rf_mcp.hints.generated`, `rf_mcp.skills.read`).

---

## Installation & Quick Start

`rf-mcp` requires Python 3.10+ and `robotcode[all]>=2.9.0.dev115` (currently available as a pre-release build):

```bash
# Using uv (recommended)
uv add --dev --prerelease=allow "rf-mcp" "robotcode[all]>=2.9.0.dev115"

# Or with pip
pip install --pre "rf-mcp" "robotcode[all]>=2.9.0.dev115"
```

Verify your environment with `rf-mcp doctor`:

```bash
rf-mcp doctor
```

Generate MCP configuration for your coding agent (`claude`, `codex`, `agy`, `copilot`, `opencode`, or `all`):

```bash
# Write local stdio MCP configs for all supported coding agents
rf-mcp init --agent all --mode server --write

# Or configure for a remote Mode 3 (client-mode) MCP server
rf-mcp init --agent claude --mode client --url http://mcp.internal:8000/mcp --write

# Synchronize skills into a local agent folder (e.g. .claude/skills or .agents/skills)
rf-mcp sync-skills --agent claude
```

`init` starts the server from the project's environment: `uv run --directory <project> rf-mcp` when the project has a `pyproject.toml`, otherwise the Python interpreter that ran `init`. `sync-skills` leaves an existing skill folder alone when its content differs (exit code 1, listed under `skipped`); pass `--force` to overwrite it. `doctor` lists every `robotcode` subcommand rf-mcp needs, the `session` subcommands, and whether `.robotframework/hints.toml` loaded.

---

## Running `rf-mcp`

```bash
# Mode 1: Local stdio MCP server (default)
rf-mcp --mode server -r /path/to/project -p ci

# Mode 2: Remote Test Runner over Streamable HTTP
rf-mcp --mode server --transport http --host 0.0.0.0 --port 8000 -r /workspace

# Mode 3: Zero-Daemon Remote MCP Server (synthesizes client-side robotcode commands)
rf-mcp --mode client --transport http --host 0.0.0.0 --port 8000 \
  --skills-dir /opt/robotframework-agent-plugins \
  --skills-dir /opt/robotframework-agentskills
```

### Docker

The images run Mode 2 with Browser, SeleniumLibrary and RequestsLibrary installed; `rf-mcp-vnc` adds a display you can watch in the browser. Mount the project at `/workspace`:

```bash
docker run --rm --shm-size=1g -p 8000:8000 -v "$PWD:/workspace" ghcr.io/manykarim/rf-mcp
rf-mcp init --agent claude --url http://localhost:8000/mcp --write
```

See [docs/DOCKER.md](docs/DOCKER.md) for the VNC image and for building with other libraries.

---

## Project Customization (Zero Code Required)

### 1. Project Skills (`<workspace>/.robotframework/skills/`)
Create `<workspace>/.robotframework/skills/<skill-name>/SKILL.md` to teach connected agents your project's custom libraries, page object patterns, and naming rules. `rf-mcp` automatically discovers and hot-reloads it at `skill://<skill-name>/SKILL.md`.

### 2. Custom Error Recovery Hints (`<workspace>/.robotframework/hints.toml`)
Define project-specific error recovery hints in `.robotframework/hints.toml`:

```toml
[[hints]]
id = "corp-sso-expired"
title = "Corporate SSO token expired"
error_pattern = "401 Unauthorized.*SSO"
keyword_pattern = ".*"
message = "Refresh the session token using the Auth resource before calling the endpoint."
fix_examples = [
    "Import Resource    resources/auth.resource",
    "Refresh SSO Token",
]
relevance = 99
applies_to = ["runtime", "static"]  # default; "static" = rf_analyze diagnostics
```

A file that does not parse, or a rule with an invalid regular expression, is skipped with a warning naming the file, line and rule id; the valid rules and all built-in rules stay active. `rf-mcp doctor` and the server instructions report the skipped entries.

### 3. Verified cookbook snippets

Every snippet in the built-in `rf-locators` shadow DOM / iframe reference and in `rf-requests` is executed by `tests/test_cookbook.py` (suites in `tests/cookbook/`) against local fixtures: a widget page with open, nested and closed shadow roots and same-origin, nested, cross-origin and shadow-hosted iframes (`tests/fixtures/web/`), and a strict REST/SOAP/GraphQL "ledger" API (`tests/fixtures/ledger/`). The suites need libraries rf-mcp does not depend on and are skipped unless available:

```bash
RF_MCP_COOKBOOK_PYTHON=/path/to/venv-with-requests-browser-selenium/python \
RF_MCP_FIXTURE_PYTHON=/path/to/venv-with-fastapi-uvicorn-strawberry/python \
uv run pytest tests/test_cookbook.py
```

### 4. OpenTelemetry Export
Install `rf-mcp[otel]` and pass `--otel-endpoint http://localhost:4318` (or set `OTEL_EXPORTER_OTLP_ENDPOINT`) to export traces and metrics over OTLP.

---

## Documentation

- [Getting started](docs/GETTING_STARTED.md): install, connect an agent, a first session, troubleshooting
- [Configuration](docs/CONFIGURATION.md): options, environment variables, execution modes, skills, hints, OpenTelemetry
- [MCP tool reference](docs/MCP_TOOLS.md): every tool and parameter, generated from the published schemas
- [Examples](docs/EXAMPLES.md): tool sequences for triage, authoring, debugging, reruns, remote runners and client mode
- [Docker images](docs/DOCKER.md): headless and VNC test-runner images
- [Release notes 2.0.0](docs/RELEASE_NOTES_v2.0.0.md): what changed from 0.x, with a tool migration table
- [Agent evaluation harness](tests/e2e/agent_eval/README.md): how rf-mcp is measured with real agents and models

Documentation for rf-mcp 0.36 and earlier is at the [`v0.36.0` tag](https://github.com/manykarim/rf-mcp/tree/v0.36.0).

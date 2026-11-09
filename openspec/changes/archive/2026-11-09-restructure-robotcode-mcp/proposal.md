# Proposal

## Why

`rf-mcp` has grown to ~90,210 lines of custom Python code (62 registered MCP tools, 12 DDD domains, custom in-process Robot Framework `EXECUTION_CONTEXTS` emulation, 4 overlapping keyword discovery/NLP layers, a 4,462-line heuristic suite rebuilder, custom ARIA/DOM snapshot parsers, and a Django dashboard) that re-implement capabilities now natively and accurately provided by the `robotcode` CLI (`>=2.9.0.dev115`, including the new `robotcode session` background REPL/debugger subsystem). Restructuring `rf-mcp` from the ground up around `robotcode` CLI, FastMCP Skills over MCP (SEP-2640), and a hybrid server/client execution model transforms the server into a lean (~2,200 LOC), 100% stateless CLI-driven foundation for coding agents running locally, against remote test runners, against a zero-daemon remote MCP server, or inside CI pipelines.

## What Changes

- **BREAKING**: Remove all legacy in-process Robot Framework execution, discovery, and UI subsystems (no backwards compatibility or deprecations):
  - Remove `src/robotmcp/components/` (`rf_native_context_manager.py`, `keyword_executor.py`, `execution_coordinator.py`, `suite_execution_service.py`, `test_builder.py`, `page_source_service.py`, `ui_tree_service.py`, `platynui_focus.py`, `keyword_matcher.py`, `nlp_processor.py`, `library_recommender.py`, `variable_resolver.py`).
  - Remove `src/robotmcp/domains/` (12 DDD domains: `snapshot`, `instruction`, `memory`, `intent`, `tool_profile`, `timeout`, `batch_execution`, `recovery`, `artifact_output`, `token_accounting`, `response_optimization`, `keyword_resolution`).
  - Remove `src/robotmcp/core/`, `src/robotmcp/plugins/`, `src/robotmcp/adapters/`, `src/robotmcp/optimization/`, `src/robotmcp/frontend/` (Django dashboard), `src/robotmcp/attach/` (`McpAttach` HTTP polling bridge), `src/robotmcp/models/`, and legacy `src/robotmcp/utils/`.
  - Remove all 62 legacy MCP tools (`execute_step`, `manage_session`, `find_keywords`, `build_test_suite`, `intent_action`, `get_session_state`, etc.) and heavy library/ML extras (`web`, `mobile`, `api`, `database`, `desktop`, `frontend`, `memory`, `semantic`, `tokens`).
- **Introduce a lean `robotcode`-powered MCP tool surface (`robotcode[all]>=2.9.0.dev115`)**:
  - `rf_discover`: Project suite/test/task/tag/metadata discovery (`robotcode -f json discover {tests|tasks|suites|tags|metadata|info|all}`) and merged configuration/profile inspection (`robotcode -f json config` / `profiles`).
  - `rf_analyze`: Headless LSP semantic static analysis (`robotcode -r <root> -f json analyze code`).
  - `rf_doc`: Project-aware library, resource, and keyword documentation (`robotcode -f json doc {lib|keywords|keyword}`).
  - `rf_run`: Suite/test/task execution (`robotcode robot`, supporting `-bl <longname>`, `-btm <metadata>`, `--rpa`, `-p <profile>`, `--wrapper`, and optional inline file sync for remote test runners).
  - `rf_results`: Server-side querying and diffing of `output.xml` / `output.json` (`robotcode -f json results {summary|show|log|stats|diff}`, including `-btm` metadata filters and `--extract` screenshot extraction).
  - `rf_repl`: Multi-turn interactive exploration and test authoring (`start`, `send`, `wait`, `status`, `interrupt`, `restart`, `list`, `save`, `stop`) backed natively by `robotcode -f json session` (`session start repl`, `session send`, `.save`, `session restart` to hot-reload modified Python libraries/resources in a fresh OS process while keeping the session ID, and native `Browser.Get Aria Snapshot` support).
  - `rf_debug`: Multi-turn live test debugging (`start`, `send`, `wait`, `status`, `result`, `interrupt`, `restart`, `list`, `stop`) backed natively by `robotcode -f json session` (`session start debug` with `-bl <longname>`, `--break`, `--stop-on-entry`, default `--output NONE --log NONE --report NONE` to preserve recorded runs, `.where`, `.vars --user`, `.list`, stepping, live keyword evaluation in the paused stack frame, and `session restart` preserving runtime breakpoints).
  - `rf_diagnose`: Actionable error diagnosis and recovery hinting for failed keywords, REPL statements, and test runs.
- **Support 3 Execution Topologies (`server`, `client`, and CI) without any additional client daemon**:
  - **Mode 1 (Full Local — `server` mode via `stdio` or `http`)**: `rf-mcp` executes `robotcode` CLI and `robotcode -f json session` subcommands directly on the developer machine or CI container.
  - **Mode 2 (Remote Test Runner — `server` mode via `http`)**: `rf-mcp` and `robotcode` run on a remote test container/VM/rig (with browsers/devices/display), accepting workspace file sync (`rf_sync` or inline files) from a local or CI coding agent.
  - **Mode 3 (Zero-Daemon Remote MCP + Client-Side `robotcode` — `client` mode via `http`)**: `rf-mcp` runs remotely while the coding agent (`claude`, `codex`, `agy`, `copilot`, `opencode`) and `robotcode` run on the user's local machine. The remote `rf-mcp` serves project/library skills over MCP, synthesizes shell-safe local `robotcode` CLI and `robotcode session` commands (plus per-command sandbox fallback commands) for the client OS/shell (`powershell` / `bash`, including `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`, and single-quoted `${var}` protection), and diagnoses errors via `rf_diagnose`.
- **Implement a `robotcode`-native, customizable Error Hinting Engine**:
  - Provide built-in recovery hints for common library and REPL/CLI errors (Browser/Playwright strict mode, `Evaluate JavaScript` argument shape, intercepted clicks, `RequestsLibrary` named-argument mismatches, Python dict bracket notation `${d}[key]`, REPL section headers, missing keywords).
  - Support project-local custom hint definitions (`.robotframework/hints.toml`) so teams can encode domain-specific error recovery guidance.
  - Attach hints automatically on failures in `server` mode and return them via `rf_diagnose` in `client` mode.
- **Serve Skills over MCP (FastMCP `SkillsDirectoryProvider` + SEP-2640) & Coexist with Ecosystem Plugins**:
  - Expose layered skill directories over MCP (`skill://{name}/SKILL.md` resources, SEP-2640 alignment, server `instructions` catalog signpost, and `ResourcesAsTools` fallback for tool-only agents).
  - Support built-in `robotcode` workflow skills, external skill directories (`--skills-dir` / `RF_MCP_SKILLS_DIR` for `robotcodedev/robotframework-agent-plugins` and `manykarim/robotframework-agentskills`), and zero-code project customizations in `<workspace>/.robotframework/skills/`.
  - Provide streamlined CLI onboarding (`rf-mcp init`, `rf-mcp sync-skills`, `rf-mcp doctor`) for Claude Code, Codex, Antigravity, GitHub Copilot, and OpenCode.
- **Add OpenTelemetry Observability (`telemetry.py`)**:
  - Extend FastMCP's built-in `fastmcp.telemetry` (`opentelemetry-api`) with `rf_mcp.*` child spans (`rf_mcp.command`, `rf_mcp.repl`, `rf_mcp.debug`, `rf_mcp.diagnose`) and metrics (`rf_mcp.commands.total`, `rf_mcp.command.duration_ms`, `rf_mcp.hints.generated`, `rf_mcp.skills.read`), plus an optional `[otel]` extra (`opentelemetry-sdk`, `opentelemetry-exporter-otlp`).

## Capabilities

### New Capabilities
- `robotcode-mcp-tools`: Core MCP tool surface (`rf_discover`, `rf_analyze`, `rf_doc`, `rf_run`, `rf_results`, `rf_repl`, `rf_debug`, `rf_diagnose`) wrapping `robotcode` CLI and `robotcode session` (`repl` / `debug`).
- `hybrid-execution-modes`: Support for `server` mode (local `stdio`/`http` and remote test runner with workspace sync), zero-daemon `client` mode (remote HTTP MCP synthesizing shell-safe client `robotcode` and `robotcode session` commands with sandbox fallbacks), and headless CI agent execution.
- `robotcode-error-hinting`: Declarative, library-aware, and project-customizable (`.robotframework/hints.toml`) error hinting engine for failed Robot Framework keywords, REPL statements, and runs.
- `mcp-skills-provider`: Skills over MCP (`SkillsDirectoryProvider`, SEP-2640 resource/manifest exposure, `ResourcesAsTools` bridge, layered builtin/external/project skill directories, and multi-agent CLI setup/sync).
- `opentelemetry-observability`: OpenTelemetry tracing and metrics instrumentation for MCP tool calls, `robotcode` operations, skill reads, and error hints.

### Modified Capabilities

## Impact

- **Codebase**: Replaces ~90,210 LOC across 224 files in `src/robotmcp/` with a focused ~2,200 LOC architecture (`server.py`, `entry.py`, `config.py`, `telemetry.py`, `runner/`, `hints/`, `skills/`, `onboarding/`).
- **Dependencies (`pyproject.toml`)**:
  - Core dependencies reduced to `fastmcp>=3.0`, `robotcode[all]>=2.9.0.dev115` (or `robotcode[runner,analyze,repl]>=2.9.0.dev115`, optional in pure `client` mode), `pydantic>=2.0`, and `tomlkit>=0.12.0`.
  - Optional extras simplified to `[otel]` (`opentelemetry-sdk`, `opentelemetry-exporter-otlp`) and `[dev]`. Test libraries (`robotframework-browser`, `robotframework-seleniumlibrary`, etc.) live in the target project's environment managed by `robotcode` rather than polluting `rf-mcp`'s own package extras.
- **MCP API**: Breaking replacement of all 62 legacy tools and 2 legacy prompts with the 8 `rf_*` tools and `skill://` MCP resources.

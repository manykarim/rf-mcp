# rf-mcp 2.0.0

rf-mcp 2.0.0 is a rewrite. The server no longer runs Robot Framework in its own process with a tool per step;
it drives the [RobotCode](https://robotcode.io/) CLI (`robotcode >= 2.9.0.dev115`) and its `session` command, and
publishes eight tools plus skills served over MCP. Projects keep their tests and `robot.toml` as they are.

Documentation for 0.36 and earlier is at the [`v0.36.0` tag](https://github.com/manykarim/rf-mcp/tree/v0.36.0/docs).

## What changes for users

- **Tools**: 8 `rf_*` tools replace the about 60 tools of 0.x; see the table below and [MCP_TOOLS.md](MCP_TOOLS.md).
- **Execution**: tests run as normal robotcode runs in the project environment, with the project's `robot.toml`,
  profiles and libraries. There is no in-process session to keep in sync with the files.
- **Execution modes**: `server` (robotcode runs on the MCP host), `client` (rf-mcp returns shell commands for the
  agent to run) and `auto`; see [CONFIGURATION.md](CONFIGURATION.md#execution-modes).
- **Skills over MCP**: built-in skills `rf-robotcode`, `rf-locators` and `rf-requests`, plus project skills from
  `.robotframework/skills/` and `--skills-dir` folders.
- **Hints**: built-in and project (`.robotframework/hints.toml`) recovery hints attached to failures and returned
  by `rf_diagnose`.
- **Onboarding**: `rf-mcp doctor`, `rf-mcp init --agent <agent>` and `rf-mcp sync-skills`.
- **Docker**: both images serve MCP at `http://<host>:8000/mcp` with the project mounted at `/workspace`; see
  [DOCKER.md](DOCKER.md).
- **Python**: 3.10 to 3.12 are tested.

## Removed

- the in-process execution engine, `execute_step` / `execute_batch` / `execute_flow` sessions and
  `build_test_suite`;
- the frontend dashboard (`--with-frontend`, port 8001) and its `ROBOTMCP_FRONTEND_*` variables;
- the `McpAttach` library (`manage_attach`) for attaching to a running Robot Framework process;
- library plugins (`manage_library_plugins` and the entry-point API), instruction templates and the optional memory
  tools (`recall_*`, `store_knowledge`);
- the extras `[slim]`, `[web]`, `[mobile]`, `[api]`, `[database]`, `[desktop]`, `[frontend]`, `[memory]`, `[tokens]`,
  `[semantic]` and `[all]`. Install the Robot Framework libraries your tests use into the project environment
  instead. `[otel]` remains.

## Tool migration

| 0.x tools | 2.0 |
| --- | --- |
| `analyze_scenario`, `recommend_libraries`, `check_library_availability`, `get_loaded_libraries`, `get_library_status` | the agent reads `robot.toml` and the project with `rf_discover` (`action='config'`, `'read'`); libraries come from the project environment |
| `find_keywords`, `discover_keywords`, `search_keywords`, `get_available_keywords` | `rf_doc` with `action='keywords'` and `pattern` |
| `get_keyword_info`, `get_keyword_documentation`, `get_library_documentation` | `rf_doc` with `action='keyword'` or `action='lib'` |
| `manage_session`, `execute_step`, `execute_batch`, `resume_batch`, `execute_flow`, `intent_action`, `evaluate_expression`, `set_variables`, `set_library_search_order`, `initialize_context` | `rf_repl` (`start`, `send`, `stop`): a robotcode REPL session that keeps imports and variables |
| `get_session_state`, `get_session_info`, `get_context_variables`, `get_page_source`, `get_application_state` | `rf_repl` / `rf_debug` dot commands (`.vars --user`, `.imports`) and keywords such as `Get Aria Snapshot` |
| `build_test_suite` | `rf_repl` with `action='save'` |
| `validate_scenario`, `validate_test_readiness`, `run_test_suite_dry` | `rf_analyze`, or `rf_run` with `dryrun=true` |
| `run_test_suite` | `rf_run` |
| `get_session_validation_status`, `fetch_artifact`, `visual_check` | `rf_results` (`summary`, `log` with `extract_dir` for screenshots) |
| `suggest_next_step`, `get_*_locator_guidance` | the `rf-robotcode` and `rf-locators` skills, `rf_diagnose` |
| `manage_library_plugins`, `list_library_plugins`, `diagnose_library_plugin`, `reload_library_plugins`, `manage_attach`, memory tools | removed |
| | new: `rf_debug` (pause a real test at its failure), `rf_results` `diff` / `merge` / `stats`, `rf_discover` tags, metadata and profiles |

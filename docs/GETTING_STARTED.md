# Getting started

This guide installs rf-mcp into a Robot Framework project, connects a coding agent and walks through a first
session. For every option see [CONFIGURATION.md](CONFIGURATION.md); for the tools see [MCP_TOOLS.md](MCP_TOOLS.md).

## 1. Install

rf-mcp needs Python 3.10+ and `robotcode[all]>=2.9.0.dev115`, which is currently a pre-release. Install both into
the environment your tests run in, next to the libraries the project uses: rf-mcp runs `robotcode` from that
environment, so it sees the same libraries and `robot.toml`.

```bash
# uv
uv add --dev --prerelease=allow rf-mcp "robotcode[all]>=2.9.0.dev115"

# pip
pip install --pre rf-mcp "robotcode[all]>=2.9.0.dev115"
```

Add `rf-mcp[otel]` for OpenTelemetry export.

Check the setup from the project folder:

```bash
rf-mcp doctor
```

```text
=== rf-mcp doctor ===
Workspace Root : /path/to/project
Python         : 3.12.8
RobotCode      : robotcode, version 2.9.0-dev.115 (OK)
Active Skills  : rf-locators, rf-requests, rf-robotcode
Custom Hints   : none (using built-in rules) [not present]
```

`doctor` also lists the robotcode subcommands rf-mcp needs and whether `.robotframework/hints.toml` loaded
(`--json` for a machine-readable report).

## 2. Connect your agent

`rf-mcp init` writes the MCP configuration for one agent or all of them:

```bash
rf-mcp init --agent claude --write      # .mcp.json
rf-mcp init --agent codex --write       # .codex/config.toml
rf-mcp init --agent copilot --write     # .vscode/mcp.json
rf-mcp init --agent agy --write         # .gemini/settings.json and .agents/plugins/rf-mcp/
rf-mcp init --agent opencode --write    # opencode.json
rf-mcp init --agent all --write
```

Without `--write` it prints the configuration instead. The server starts from the project's environment:
`uv run --directory <project> rf-mcp` when the project has a `pyproject.toml`, otherwise the Python interpreter
that ran `init`.

To connect to a server that already runs elsewhere, pass its URL:

```bash
rf-mcp init --agent claude --mode client --url http://test-runner:8000/mcp --write
```

Agents that do not read MCP resources well can get the built-in skills as files as well:

```bash
rf-mcp sync-skills --agent claude    # .claude/skills/  (codex, agy, opencode: .agents/skills/, copilot: .github/skills/)
```

`sync-skills` leaves a skill folder alone when its content differs (exit code 1, listed under `skipped`); pass
`--force` to overwrite it.

## 3. A first session

Open the agent in the project folder and ask in plain language. The server's instructions and the
`rf-robotcode` skill route each request to a tool; the agent does not need to know the tool names.

| you ask | what the agent does |
| --- | --- |
| "Which smoke tests do we have?" | `rf_discover` with `action='tags'` or `include_tags=['smoke']` |
| "Run the login tests." | `rf_discover` for the longnames, then `rf_run` with `by_longname` |
| "Why did *Checkout Total* fail?" | `rf_results` (`action='log'`, `failed_only=true`) before it opens any file |
| "Check my new suite without running it." | `rf_analyze` |
| "Which keyword uploads a file in Browser?" | `rf_doc` with `action='keywords'`, `library='Browser'` and synonyms as `pattern` |
| "Write a test for the search page." | `rf_repl` to try the steps live, `save` them as a test, `rf_analyze`, then `rf_run` |
| "*Valid Login* fails on CI, find out why." | `rf_debug` paused at the failure, inspecting variables in the real suite context |

A change counts as done when `rf_analyze` reports no errors and `rf_run` returns `outcome: PASS`. Every `rf_run`
response starts with that verdict, for example:

```json
{"outcome": "FAIL: 3 tests, 2 passed, 1 failed. The tests do not pass yet: fix them and rerun before you report the task as done.",
 "exit_code": 1,
 "failures": ["Checkout Total: Expected 42.00 but was 40.00"]}
```

## 4. Teach the agent your project

Project knowledge lives in the repository, next to the tests:

- `.robotframework/skills/<name>/SKILL.md`: conventions, page objects, custom libraries. The server serves it as
  `skill://<name>/SKILL.md` and picks up changes without a restart.
- `.robotframework/hints.toml`: recovery hints for errors specific to your system, returned by `rf_diagnose` and
  attached to failing results.

See [CONFIGURATION.md](CONFIGURATION.md#skills) for both formats.

## 5. Other setups

- **Remote test runner**: run rf-mcp next to the browsers or devices (for example in the
  [Docker image](DOCKER.md)) with `--transport http`. The agent sends changed files with the `files` argument.
- **Client mode**: rf-mcp does not execute anything and returns `robotcode` commands that the agent runs in its own
  shell. Useful when the MCP server is shared but tests must run on the developer's machine.

Both are described under [Execution modes](CONFIGURATION.md#execution-modes). [EXAMPLES.md](EXAMPLES.md) shows
complete tool sequences for the common tasks.

## Troubleshooting

| symptom | fix |
| --- | --- |
| `` `robotcode` was not found on the MCP server host `` | Install `robotcode[all]>=2.9.0.dev115` in the environment rf-mcp starts from, or start rf-mcp from the project environment (`rf-mcp init` does this). |
| `rf_doc` or `rf_analyze` cannot find a library | The library is missing from the environment rf-mcp runs in, or `robot.toml` does not put its folder on `python-path`. |
| The agent reads `output.xml` instead of using `rf_results` | Sync the skills (`rf-mcp sync-skills`) for agents that ignore MCP resources. |
| Client-mode commands fail to quote correctly | Pass `shell='powershell'` or `shell='bash'`, or start the server with `--shell`. PowerShell 7+ and bash are tested; Windows PowerShell 5.1 is not. |

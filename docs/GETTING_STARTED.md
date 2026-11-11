# Getting started

This guide installs rf-mcp, connects a coding agent and walks through a first session. For every option see
[CONFIGURATION.md](CONFIGURATION.md); for the tools see [MCP_TOOLS.md](MCP_TOOLS.md).

## 1. Install

rf-mcp needs Python 3.10+ and [uv](https://docs.astral.sh/uv/) 0.12 or newer (`uv self update`). It runs
`robotcode[all]>=2.9.0.dev115` (currently a pre-release) in each project's own environment, so robotcode sees the
same libraries and `robot.toml` as your tests. Older uv versions (0.9 is confirmed) refuse robotcode's pre-release
sub-packages during the install unless `--prerelease=allow` is passed. Pick one of two setups.

**One install for all projects (recommended):**

```bash
uv tool install rf-mcp
```

A project then needs only Robot Framework and its libraries in its `.venv`. When that environment has no
robotcode, or an older one, rf-mcp adds its own robotcode version with uv and leaves the project unchanged; see
[Which Python runs your tests](CONFIGURATION.md#which-python-runs-your-tests).

**Installed into the project**, for example to pin the rf-mcp version for a team:

```bash
# uv
uv add --dev rf-mcp "robotcode[all]>=2.9.0.dev115"

# pip
pip install rf-mcp "robotcode[all]>=2.9.0.dev115"
```

Naming robotcode with its pre-release version lets uv and pip install it without `--prerelease=allow` / `--pre`,
which would pull pre-releases of every other dependency too. Add `rf-mcp[otel]` for OpenTelemetry export.

Check the setup from the project folder:

```bash
rf-mcp doctor
```

```text
=== rf-mcp doctor ===
Install        : global (rf-mcp 2.0.0, Python 3.10.22)
Project Root   : /home/dev/shop (found by robot.toml)
Environment    : venv /home/dev/shop/.venv/bin/python (Python 3.12.3)
RobotCode      : robotcode 2.9.0-dev.186 added by uv, ready in 1.3 s (OK)
uv             : uv 0.12.23 (/home/dev/.local/bin/uv)
Active Skills  : rf-locators, rf-requests, rf-robotcode
Custom Hints   : none (using built-in rules) [not present]
```

- **Install:** `project` when rf-mcp runs from the project's `.venv`, else `global`.
- **Project Root:** the folder `doctor` uses, found the way the server finds it (`-r` pins it).
- **Environment:** the interpreter that runs robotcode for this project: `configured`, `venv`, `activated`,
  `path`, `own`, or `fallback`. With `fallback`, `doctor` exits 1 and prints a warning with the fix, because the
  project's libraries are not visible.
- **RobotCode:** the version that runs. "added by uv" means rf-mcp layers its own robotcode onto the project
  environment. `doctor` warms that layer, so the first agent call is fast.

`doctor` also lists missing robotcode subcommands and whether `.robotframework/hints.toml` loaded (`--json` for a
machine-readable report).

## 2. Connect your agent

With the global install, register rf-mcp once for every project:

```bash
rf-mcp init --scope user --agent all --write
```

| agent | how `--scope user` registers `rf-mcp` |
| --- | --- |
| Claude Code | `claude mcp add --scope user rf-mcp -- rf-mcp` |
| Codex | `codex mcp add rf-mcp -- rf-mcp` (`~/.codex/config.toml`) |
| Copilot | `code --add-mcp` (VS Code user profile) and `copilot mcp add` (Copilot CLI) |
| Antigravity | `agy mcp add rf-mcp rf-mcp` |
| OpenCode | `opencode mcp add --global rf-mcp -- rf-mcp` |

An agent whose command is not installed is skipped with the command to run later. The entry names `rf-mcp` by its
full path when `PATH` does not find this version of it: when its folder is not on `PATH` (agents started from a
desktop shortcut often lack `~/.local/bin`), or when an older rf-mcp comes first on `PATH`.

To configure one project instead (and commit the files for the team), run `init` in the project:

```bash
rf-mcp init --agent claude --write      # .mcp.json
rf-mcp init --agent codex --write       # .codex/config.toml
rf-mcp init --agent copilot --write     # .vscode/mcp.json
rf-mcp init --agent agy --write         # .gemini/settings.json and .agents/plugins/rf-mcp/
rf-mcp init --agent opencode --write    # opencode.json
rf-mcp init --agent all --write
```

The entries contain no paths: `uv run rf-mcp` when rf-mcp is installed in a uv project, the project
interpreter's `-m robotmcp` in a plain venv, and `rf-mcp` for a global install. Without `--write`, `init` prints
what it would change. It replaces only its own `rf-mcp` entry, leaves every other server in the file alone, and does
not touch a file with comments or one that does not parse (it prints the entry instead and exits 1). Both scopes
name the server `rf-mcp`, so a project that pins its own rf-mcp can take precedence over the global one.

When both a user entry and a project entry named `rf-mcp` exist, the agents behave like this (measured on
2026-11-11 with probe servers in both places):

| agent | which entry runs |
| --- | --- |
| Claude Code | the project's `.mcp.json` entry |
| Codex | the project's `.codex/config.toml` entry when the project is trusted; otherwise the user entry |
| Copilot CLI | the workspace `.mcp.json` entry when the folder is trusted (for example `--trust-folder`); otherwise the user entry |
| Antigravity | both: the project plugin's server is named separately, so the agent gets two rf-mcp servers. Register it in one place only. |
| OpenCode | the project's `opencode.json` entry |

To connect to a server that already runs elsewhere, pass its URL:

```bash
rf-mcp init --agent claude --url http://test-runner:8000/mcp --write
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
| `` `robotcode` could not be started for <root> `` | The environment chosen for that project has no runnable robotcode. With rf-mcp installed in the project, add `robotcode[all]>=2.9.0.dev115` to it. With a global install, install uv so rf-mcp can add robotcode itself, or add robotcode to the project environment. `rf-mcp doctor` shows which environment was chosen. |
| `warning`: "the project's libraries are not visible" (and analysis import errors marked "not a project defect") | rf-mcp found no environment for the project and ran its own robotcode. Create the project's `.venv`, set `python` in `.robotframework/rf-mcp.toml`, or start the agent in an activated environment; see [Which Python runs your tests](CONFIGURATION.md#which-python-runs-your-tests). |
| `rf_doc` or `rf_analyze` cannot find a library | The library is missing from the project environment, or `robot.toml` does not put its folder on `python-path`. |
| The agent reads `output.xml` instead of using `rf_results` | Sync the skills (`rf-mcp sync-skills`) for agents that ignore MCP resources. |
| Client-mode commands fail to quote correctly | Pass `shell='powershell'` or `shell='bash'`, or start the server with `--shell`. PowerShell 7+ and bash are tested; Windows PowerShell 5.1 is not. |

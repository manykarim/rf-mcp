# Configuration

rf-mcp reads command-line options first, then environment variables, then the defaults below. Project settings
(paths, libraries, profiles) stay in the project's `robot.toml`: rf-mcp passes them on to robotcode and does not
duplicate them.

## Command-line options and environment variables

| option | environment variable | default | meaning |
| --- | --- | --- | --- |
| `--mode {server,client,auto}` | `RF_MCP_MODE` | `auto` | Where robotcode runs; see [Execution modes](#execution-modes). |
| `-r`, `--root` | `RF_MCP_ROOT` | not pinned | Pins the project root (the folder with `robot.toml`) for every call that passes no `root`; see [How the project root is found](#how-the-project-root-is-found). |
| `-p`, `--profile` | `RF_MCP_PROFILE` | none | Default `robot.toml` profile (`-p`). |
| `--skills-dir` (repeatable) | `RF_MCP_SKILLS_DIR` (`;` or the OS path separator) | none | Extra skill folders; see [Skills](#skills). |
| `--hints-file` | `RF_MCP_HINTS_FILE` | `<root>/.robotframework/hints.toml` | Custom recovery hints; see [Hints](#hints). |
| `--shell {powershell,bash}` | `RF_MCP_SHELL` | PowerShell on Windows, bash elsewhere | Syntax of client-mode commands. |
| `--transport {stdio,http,sse}` | `RF_MCP_TRANSPORT` | `stdio` | MCP transport. `http` is streamable HTTP at `/mcp`. |
| `--host` | `RF_MCP_HOST` | `127.0.0.1` | Bind address for `http` and `sse`. |
| `--port` | `RF_MCP_PORT` | `8000` | Port for `http` and `sse`. |
| `--idle-timeout` | `RF_MCP_IDLE_TIMEOUT` | `15` | Minutes after which an unused `rf_repl` / `rf_debug` session ends itself. |
| `--timeout` | `RF_MCP_TIMEOUT` | `60` | Seconds a session call waits before it returns `status: running` (poll with `action='wait'`). |
| `--otel-endpoint` | `RF_MCP_OTEL_ENDPOINT`, `OTEL_EXPORTER_OTLP_ENDPOINT` | none | OTLP collector URL; see [OpenTelemetry](#opentelemetry). |
| `--otel-service-name` | `OTEL_SERVICE_NAME` | `rf-mcp` | `service.name` of exported telemetry. |
| `-V`, `--version` | | | Print the version. |

The subcommands `rf-mcp doctor`, `rf-mcp init` and `rf-mcp sync-skills` are described in
[GETTING_STARTED.md](GETTING_STARTED.md). `python -m robotmcp` is the same as `rf-mcp`.

The server sets `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8` and `ROBOTCODE_FORCE_AI_AGENT=1` for every robotcode
process, and client-mode commands set them too.

## Execution modes

| mode | where robotcode runs | typical setup |
| --- | --- | --- |
| `server` | on the MCP host, started by rf-mcp | Local stdio server in the project environment, or a remote test runner over HTTP |
| `client` | in the agent's own shell | A shared HTTP server that serves skills and hints; tests run on the developer's machine |
| `auto` | `server` when `robotcode` is on the PATH or installed next to rf-mcp, else `client` | Default |

Every tool takes `target='server'` or `target='client'` to override the mode for one call.

### Local (server mode, stdio)

```bash
rf-mcp
```

Started by the agent in the project, rf-mcp finds the project itself (see
[How the project root is found](#how-the-project-root-is-found)). `-r <path>` pins the root and `-p ci` sets a default
profile, for example `rf-mcp -r /path/to/project -p ci`. This is what `rf-mcp init` configures.

### Remote test runner (server mode, HTTP)

```bash
rf-mcp --mode server --transport http --host 0.0.0.0 --port 8000 -r /workspace
```

rf-mcp runs next to the browsers, devices or displays, for example in the [Docker image](DOCKER.md). The agent
edits files locally and sends them with the `files` argument (`{"tests/login.robot": "<content>"}`); the server
writes them below the root before the call. Paths are relative to the root, and absolute paths or `..` are refused.

The HTTP transport has no authentication. Bind it to a private network or put it behind a proxy that
authenticates.

### Client mode

```bash
rf-mcp --mode client --transport http --host 0.0.0.0 --port 8000 --skills-dir /opt/team-skills
```

Tools return a `command` for the agent's shell instead of running it. The agent's `root` is passed to `robotcode -r`
as given, because it is a path on the client machine; without one, `-r` is left out. `rf_debug start` also returns
a `fallback_command` for agent shells that end background processes. `rf_repl` has none, because a one-shot REPL
cannot replay earlier turns. After a local run, the agent can pass the output to `rf_diagnose`.

Client-mode commands are tested in PowerShell 7+ and bash. Windows PowerShell 5.1 is not tested; its quoting of
native arguments differs.

## How the project root is found

In server mode, each tool call works in one project root, chosen in this order:

1. The call's `root` argument.
2. `--root` / `RF_MCP_ROOT`, when set. Docker images and remote runners pin `/workspace` this way.
3. The client's workspace folders (MCP roots), when the client reports them. With one folder, that folder is used.
   With several, rf-mcp uses the one that contains its working directory. If none does, the call fails and asks
   the agent to pass `root`. rf-mcp asks only clients on the older, handshake-era MCP protocol: the 2026-07-28
   revision removed requests from the server to the client.
4. The server's working directory, walked up to the nearest folder with `.robot.toml`, `robot.toml`,
   `pyproject.toml`, `.git` or `.hg`. This is the order robotcode uses to find its own project root. Without any
   of them, the working directory itself is used.

Project skills (`.robotframework/skills/`), the server instructions and the default hints are read once at
startup, from the pinned root or from step 4. Custom hints are read again for each call's root.

How the agents start a stdio server, measured on 2026-11-11 with a probe server that logged its working directory
and the client's `initialize` request:

| agent | version | working directory | MCP protocol | roots |
| --- | --- | --- | --- | --- |
| Claude Code | 2.1.296 | the project | 2025-11-25 | yes: the project folder |
| Codex | 0.166.0 | the project | 2025-06-18 | no |
| Copilot CLI | 1.0.105 | the project | 2025-11-25 | no |
| Antigravity (`agy`) | 1.0.0 | the plugin folder (`.agents/plugins/<name>`) for a project plugin | 2025-11-25 | yes: the project folder |
| Copilot in VS Code, OpenCode | | not measured | | |

All four measured agents give rf-mcp the right root: Claude Code and Antigravity through their roots (and the
working directory), Codex and the Copilot CLI through the working directory.

In client mode, `root` is a path on the client machine and is passed on unchanged; see [Client mode](#client-mode).

## Which Python runs your tests

robotcode imports your libraries, so it has to run in the project's Python environment. For each project root,
rf-mcp picks that environment in this order:

1. `python` in `<root>/.robotframework/rf-mcp.toml`, for environments outside the project folder (Poetry, conda,
   hatch). A relative path is relative to the root:

   ```toml
   python = "envs/robot/bin/python"
   ```

2. `<root>/.venv`.
3. The activated virtual environment (`VIRTUAL_ENV`), then a `robotcode` on `PATH`.
4. rf-mcp's own environment.

What runs in the chosen environment:

- **rf-mcp's own environment** (rf-mcp is installed in the project): its robotcode runs directly.
- **An environment with robotcode `2.9.0.dev115` or newer:** that robotcode runs directly.
- **An environment without robotcode, or with an older one:** rf-mcp adds its own robotcode version on top with
  `uv run --no-project --python <interpreter> --with "robotcode[all]==<version>"`, plus one `--with` per robotcode
  sub-package at the same version, so every pre-release is named and works on older uv too. This needs
  [uv](https://docs.astral.sh/uv/). The project environment is not changed: uv keeps robotcode in its cache. The
  first call per robotcode version takes about ten seconds while uv downloads it; later calls add about 0.1 s.

If step 4 runs for a project that is not rf-mcp's own environment, the project's libraries are not visible. The
result then carries a `warning` that says so and names the fix, and `rf_analyze` marks the resulting import errors
as "not a project defect". `rf-mcp doctor` shows which environment a project gets and why.

## robot.toml

rf-mcp does not have its own project file. Paths, `python-path`, variables, listeners and profiles come from
`robot.toml` (and `pyproject.toml` `[tool.robot]`), exactly as for `robotcode` on the command line. To check what a
profile resolves to, ask for `rf_discover` with `action='profile_show'` and `profile='<name>'`.

## Skills

Skills are folders with a `SKILL.md` and optional `references/*.md`. rf-mcp serves them over MCP as
`skill://<name>/SKILL.md` and `skill://<name>/references/<file>` (SEP-2640), and adds the tools `list_resources`
and `read_resource` for agents that can only call tools. When two sources contain a skill with the same folder
name, the first one wins:

1. project skills in `<root>/.robotframework/skills/`
2. folders given with `--skills-dir` / `RF_MCP_SKILLS_DIR`, in order
3. the built-in skills `rf-robotcode` (tool routing, triage, debugging, REPL authoring, results, analysis,
   configuration), `rf-locators` (Browser, SeleniumLibrary, AppiumLibrary and PlatynUI locators, shadow DOM, iframes)
   and `rf-requests` (RequestsLibrary for REST, SOAP/XML and GraphQL)

Skill collections that work with `--skills-dir` include
[robotcodedev/robotframework-agent-plugins](https://github.com/robotcodedev/robotframework-agent-plugins/tree/next)
and [manykarim/robotframework-agentskills](https://github.com/manykarim/robotframework-agentskills).

A minimal project skill:

```markdown
---
name: shop-conventions
description: Page objects, test data and naming rules of the shop tests. Read before writing or changing a test.
---

- Keywords for pages live in `resources/pages/<page>.resource`; tests never use locators directly.
- Test data comes from `data/*.yaml` via `Load Test Data`, never inline.
- Test names start with the ticket id: `WEB-123 Checkout With Voucher`.
```

`rf-mcp sync-skills` copies the active skills into an agent's skill folder for agents that read skills from disk.

## Hints

`rf_diagnose`, failing `rf_run` results and `rf_analyze` diagnostics carry ranked recovery hints. The built-in rules
cover Browser, SeleniumLibrary, AppiumLibrary, RequestsLibrary, PlatynUI, BuiltIn and robotcode sessions. Projects add
their own in `.robotframework/hints.toml` (or the file given with `--hints-file`):

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
applies_to = ["runtime", "static"]
```

| field | meaning |
| --- | --- |
| `id`, `title` | Identifier and short title shown with the hint. |
| `error_pattern` | Regular expression matched against the error text (alias `pattern`). |
| `keyword_pattern`, `arg_pattern`, `library_pattern` | Optional regular expressions for the keyword, its arguments and the library. |
| `message` | The advice; alternatively `explanation` and `suggestion`. |
| `fix_examples` | Lines the agent can copy (alias `examples`). |
| `relevance` | Ranking against other hints, default 90 (alias `priority`). |
| `applies_to` | `runtime` (failed runs and sessions), `static` (`rf_analyze` diagnostics) or both (default). |

A file that does not parse, or a rule with an invalid regular expression, is skipped with a warning naming the
file, line and rule id; the other rules stay active. `rf-mcp doctor` and the server instructions list skipped
entries.

## OpenTelemetry

Install `rf-mcp[otel]` and pass `--otel-endpoint http://localhost:4318` (or set `OTEL_EXPORTER_OTLP_ENDPOINT`).
rf-mcp exports over OTLP/HTTP:

- spans `rf_mcp.discover`, `rf_mcp.analyze`, `rf_mcp.doc`, `rf_mcp.run`, `rf_mcp.results`, `rf_mcp.repl`,
  `rf_mcp.debug` and `rf_mcp.diagnose`, nested under the FastMCP server spans;
- metrics `rf_mcp.commands.total`, `rf_mcp.command.duration_ms`, `rf_mcp.hints.generated` and `rf_mcp.skills.read`.

Without an endpoint, or without the extra installed, telemetry is off and costs nothing.

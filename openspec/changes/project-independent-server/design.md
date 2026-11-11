# Design

## Context

See proposal.md for the motivation. The current state that shapes the approach:

- `ServerConfig.workspace_root` defaults to `Path.cwd()` at startup (`config.py:176`). Every tool resolves its root
  with `cfg.resolve_root(root)`, so an unset root and a pinned `--root` look the same.
- robotcode is started in two places, `cli_runner.py:756` and `session_runner.py:503`. Both call
  `resolve_robotcode_executable()`, which takes the first `robotcode` on `PATH` or else the one next to
  `sys.executable`. Both already have the call's root at hand.
- Things read once at startup from the startup root: the server instructions, the project skills
  (`mount_skills_provider`) and the default hint engine. Custom hints are already reloaded per call
  (`server.py:127`).
- `rf-mcp init` writes whole files (`onboarding/cli.py:287`), replacing any other servers in them.
- Measured on 2026-11-11 (uv 0.12.23, Windows 11), in a scratch venv with Robot Framework and RequestsLibrary only:
  - `uv run --no-project --python <venv python> --with "robotcode[all]==2.9.0.dev115" robotcode ...` ran
    `analyze code` with no import error, `robot` with 1 passed, and a REPL session across four separate invocations
    (`start`, two `send`, `stop`).
  - The layered robotcode won over robotcode 1.5.0 installed in the venv, and the venv was unchanged afterwards.
  - Timing: 1.28 s per warm call against 1.15 s for a direct call; about 9.5 s for the first call.
- Each invocation runs from a different temporary environment under uv's cache (`builds-v0/.tmpXXXX`), so the
  interpreter path of a layered environment cannot be cached or reused.
- MCP roots:
  - FastMCP 3.4.4 provides `Context.list_roots()`.
  - The Claude Code binary contains its own roots handling.
  - Roots support in Codex, VS Code Copilot, OpenCode and Antigravity is unverified. Measured later (tasks 6.1
    and 6.4): Codex and the Copilot CLI send no roots; VS Code 1.145.2, OpenCode and Antigravity do. VS Code sends
    `file:///c%3A/...` URIs, starts a user entry in the home folder, and runs a user and a workspace entry of the
    same name side by side; Copilot CLI chats inside VS Code start only the workspace entry.

## Goals / Non-Goals

**Goals:**

- The same resolution code serves the project install, the global install and the pinned remote runner.
- A call never modifies the project's files or environment.

**Non-Goals:**

- **Project skills and instructions for roots other than the startup root.** The instructions are sent in the
  `initialize` response, before MCP roots can be requested. Per-root skills would need a dynamic provider. Hints
  already work per root.
- **Using MCP roots in client mode.** Client-mode roots stay verbatim client paths and are passed as `-r` only when
  given. Using the client's roots there is a possible follow-up.
- **Creating a missing `.venv`** (for example `uv sync` for a uv project that was never synced). rf-mcp reports it in
  `doctor`, because creating it writes to the project.
- **Poetry, conda and hatch discovery** beyond the configured interpreter and `VIRTUAL_ENV`.
- **A shared multi-project HTTP server.** It becomes possible with per-call roots but is not designed here.

## Decisions

### D1. An unset root is distinct from a pinned root

`workspace_root` becomes `Path | None`. `None` means the root was not pinned by `--root` or `RF_MCP_ROOT`. A new
`resolve_call_root(root_arg, ctx)` applies the order from the spec:

1. the `root` argument;
2. the pinned root;
3. MCP roots, only when the client declared the `roots` capability;
4. the working directory, walked up.

The walk-up uses robotcode's markers and order (`find_project_root` in `robotcode/robot/config/loader.py`):
`.robot.toml`, `robot.toml`, `pyproject.toml`, `.git`, `.hg`. Each directory is checked for all markers before moving
to its parent. The logic is reimplemented in a few lines rather than imported from robotcode's internals, which are
not public API.

The startup root (pinned, or the working directory walked up) feeds the instructions, the project skills and the
default hints.

MCP roots are asked for only from clients on the older (handshake-era) protocol that declared the `roots`
capability. FastMCP 4 has no `ctx.list_roots()`: the 2026-07-28 protocol revision (SEP-2322) removed requests
from the server to the client, so the request goes through the deprecated `ctx.session.list_roots()`, and the step
is skipped for clients on the new protocol (`Context._is_modern_protocol()`). The lookup runs in an async
middleware that injects `root` before the tool runs, because the tools are synchronous and block while robotcode
runs. Roots are cached per session until `notifications/roots/list_changed` arrives.

- *Alternative:* request roots on every call. Rejected: it adds a client round-trip to every tool call.
- *Alternative:* the new protocol's input requests (the tool returns an input request, the client answers, the
  tool runs again). Deferred: no client is confirmed to support it, and the working directory covers its absence.
- *Alternative:* make the tools async so they can await the roots. Rejected: their robotcode subprocesses block,
  and an async tool would stall the whole server.

### D2. A new environment resolver, cached per root

A new module, `runner/environment.py`, provides `resolve_environment(root) -> ProjectEnvironment`.
`ProjectEnvironment` carries:

- `source`: `own`, `configured`, `venv`, `activated`, `path` or `fallback`;
- `interpreter`, when known;
- `command_prefix`, the argv that replaces the literal `robotcode`;
- `robotcode_version`;
- `layered`, a bool;
- `warning`, when set.

`CliRunner` and `SessionRunner` replace `resolve_robotcode_executable()` with
`resolve_environment(root).command_prefix`.

Results are cached per resolved root for the server's lifetime. An entry is checked again when its interpreter or
`robotcode` executable no longer exists, or when the mtime of `<venv>/pyvenv.cfg` or of the `site-packages` folder
changes, so installing or removing a package in the project is noticed.

- *Alternative:* no cache. Rejected: the version probe costs about one subprocess per call.

### D3. A supported robotcode is decided by version

The project's robotcode counts as supported when its version, read with one
`<interpreter> -c "import importlib.metadata as m; print(m.version('robotcode'))"`, is at least rf-mcp's minimum
(`2.9.0.dev115`, the first version with `session`). `doctor` keeps the full subcommand check for diagnosis.

- *Alternative:* parse `robotcode --help` and `robotcode session --help` on every resolution. Rejected: it costs two
  slow subprocesses for the same answer.

### D4. Layering with `uv run --with`, pinned to rf-mcp's own robotcode

The layered prefix is:

```
uv run --quiet --no-project --python <interpreter> --with "robotcode[all]==<pin>" robotcode
```

`<pin>` is `importlib.metadata.version("robotcode")` of rf-mcp's own environment, the version rf-mcp was resolved and
tested with. An exact pin that names a pre-release is accepted by uv's default pre-release mode, so the command needs
no `--prerelease=allow`, which would also let pre-releases of robotcode's own dependencies in.

On first resolution for a root, rf-mcp runs `robotcode --version` through this prefix. That warms uv's cache and
proves the layer works. When it fails (offline on first use, project Python too old for robotcode, uv error), the
resolver continues with the next step and keeps uv's error text for the warning.

- *Alternative:* a persistent overlay venv per project in rf-mcp's cache. Rejected for now: it saves about 0.1 s per
  warm call at the cost of rf-mcp's own invalidation logic.
- *Alternative:* install robotcode into the project venv. Rejected: it modifies the project.
- *Alternative:* `PYTHONPATH` injection. Rejected: it mixes rf-mcp's and the project's dependency versions in one
  interpreter.

### D5. rf-mcp's own environment is recognised by prefix

The environment counts as rf-mcp's own when `Path(sys.prefix).resolve() == (<root>/.venv).resolve()`, or when the
configured interpreter belongs to `sys.prefix`. rf-mcp then calls its own robotcode directly, with no probe and no
uv. The same test gives the install kind:

- `project`: rf-mcp runs from the root's `.venv` or the configured interpreter;
- `global`: anything else, including a `uv tool` install.

The install kind decides the `init` command and the wording of errors.

### D6. The configured interpreter lives in `.robotframework/rf-mcp.toml`

The file holds a single key, `python = "<path>"`. A relative path resolves against the root. The `.robotframework/`
folder already holds `hints.toml` and `skills/`.

- *Alternative:* `[tool.rf-mcp]` in `pyproject.toml` or `robot.toml`. Rejected: not every project has a
  `pyproject.toml`, and `robot.toml` belongs to robotcode's schema.

The configured interpreter runs code from the repository, but it introduces no new trust: rf-mcp already runs the
project's tests and the project's `.venv` robotcode.

### D7. The fallback warning travels with the result

When `source == "fallback"` and the root is not rf-mcp's own environment, the result's `instruction` (or a `warning`
field where the result type has no instruction) says that project libraries are not visible and names
`rf-mcp doctor`. Import errors in `rf_analyze` diagnostics under the fallback are prefixed with the same note.

Other sources add nothing to the response, which keeps responses compact as `robotcode-mcp-tools` requires.

### D8. `init` scopes, commands and merging

| scope | install kind | command written | where |
| --- | --- | --- | --- |
| project (default) | project, uv project | `uv run rf-mcp` | the project files `init` writes today |
| project | project, plain venv | `<venv python> -m robotmcp` | same |
| project | global | `rf-mcp` | same |
| user | any | `rf-mcp`, or its full path when its folder is not on `PATH` | each agent's user-level configuration |

The entry contains no arguments. Everything is resolved at run time, and `--mode` defaults to `auto`. `--url` keeps
writing an HTTP entry in either scope.

User-level entries go through each agent's own CLI, which knows its file format and locking. Implementation
found that every installed agent has one, so this replaced the planned file merges. Verified on 2026-11-11 against
a redirected home (task 6.2), where each agent's own `mcp list` showed the entry:

| agent | how the entry is written |
| --- | --- |
| Claude Code | `claude mcp remove --scope user rf-mcp`, then `claude mcp add --scope user rf-mcp -e K=V ... -- <cmd>`. The name goes before `-e`, because `-e` takes several values and would swallow it. `~/.claude.json` is never edited directly. |
| Codex | `codex mcp add rf-mcp --env K=V ... -- <cmd>` (`$CODEX_HOME/config.toml`) |
| Copilot | `code --add-mcp '<json>'` (VS Code user profile) and `copilot mcp add --env K=V ... rf-mcp -- <cmd>` (Copilot CLI, `~/.copilot/mcp-config.json`), reported as `copilot` and `copilot-cli` |
| Antigravity | `agy mcp add -e K=V ... rf-mcp <cmd>` (flags before the name, as `agy` requires) |
| OpenCode | `opencode mcp add --global rf-mcp --env K=V ... -- <cmd>` (2.2.7; a second `add` replaces the entry) |

An agent whose CLI is not on `PATH` is reported as `manual` with the command to run. A CLI that fails is
reported as `error` with its output. Both make `init` exit 1.

Merging:

- JSON files are parsed and only the `rf-mcp` key under the agent's server map is replaced; key order is kept. A file
  with JSONC comments (common in VS Code) is not rewritten, because the comments would be lost. `init` changes
  nothing and prints the entry to paste.
- TOML files are merged with tomlkit.
- An unparsable file is never changed.

### D9. `doctor` uses the server's resolver

`doctor` calls `resolve_call_root` (without MCP roots) and `resolve_environment`, so it reports exactly what a tool
call would use. It adds the install kind, the uv version and path, and the interpreter's Python version.

## Risks / Trade-offs

- **The first layered call takes about 10 s** while uv downloads robotcode. → D4 warms the layer during resolution;
  the default per-command timeout (60 s) covers it, and `doctor` warms it too.
- **The first layered call needs network access.** → The resolver falls back with uv's error in the warning; later
  calls work offline from uv's cache.
- **A project on Python older than 3.10 cannot take robotcode 2.9.** → The probe fails and the warning quotes uv's
  resolution error; `doctor` shows the Python version.
- **`--with` behaviour depends on uv's version** (tested with 0.12.23). → `doctor` reports the uv version; an
  integration test runs the layer in CI with the uv that `setup-uv` installs.
- **MCP roots support is unverified for four agents.** → The working-directory fallback covers clients that start
  the server in the project (Claude Code does). Task 6.1 checks each agent.
- **Same-name precedence between user and project entries is agent-specific.** → Task 6.3 checks it, and the docs
  state the behaviour per agent.
- **A `robotcode` on `PATH` can belong to an unrelated environment** in a global install. → It comes after
  `<root>/.venv` and after `VIRTUAL_ENV`, and it must be a supported version.
- **Existing project entries keep their `--directory` and `-r` arguments.** These still work, because `-r` pins the
  root, but they stay non-portable until the user runs `init` again. → The release notes say so.

## Migration Plan

1. Release as a minor version. Old entries keep working: `-r` pins the root, and `uv run --directory` makes rf-mcp's
   own environment the project's, which is resolution step 2a.
2. The docs switch the primary install to `uv tool install rf-mcp` (with whatever robotcode pre-release flag task
   4.4 proves necessary, preferring none) followed by
   `rf-mcp init --scope user`, and keep the project install as the alternative.
3. Rollback: reinstall the previous version. The new entries (`rf-mcp`, `uv run rf-mcp`) also start the previous
   version, which falls back to the working directory as its root.

## Open Questions

- Should `init --scope user` also offer `sync-skills` into a user-level skill folder for Codex, if Codex reads one?
  This depends on task 6.2's findings and does not change the resolution design.

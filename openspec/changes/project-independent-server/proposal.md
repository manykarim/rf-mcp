## Why

rf-mcp only works when it is installed into each project and started with that project's path. `rf-mcp init` writes
`uv run --directory <abs project> rf-mcp --mode server --transport stdio -r <abs project>` into files teams commit
(`.mcp.json`, `.vscode/mcp.json`, `opencode.json`, `.codex/config.toml`), so a clone in another folder gets a broken
configuration. The docs lead with `rf-mcp --mode server -r /path/to/project -p ci`. The path does two jobs: it
names the project, and through `uv run --directory` it chooses the Python environment whose `robotcode` sees the
project's libraries. A server that resolves both per call can be installed once (`uv tool install rf-mcp`) and
registered once per agent for every project.

Measured on 2026-11-11 (uv 0.12.23, Windows): `uv run --no-project --python <project>/.venv/... --with
"robotcode[all]==<pin>" robotcode ...` lets a project venv that holds only Robot Framework and RequestsLibrary pass
`analyze code`, `robot`, and a REPL session (`start`, `send`, `stop`) spread over separate invocations. The pinned
robotcode wins over an older one in the project (1.5.0), and the project venv is left unchanged. A warm call costs
1.28 s against 1.15 s for a direct call; the first call takes about 9.5 s while uv fills its cache.

## What Changes

- **Project root per call:** the `root` argument, else a pinned `--root` / `RF_MCP_ROOT`, else the client's MCP
  roots, else the server's working directory walked upward with robotcode's markers (`.robot.toml`, `robot.toml`,
  `pyproject.toml`, `.git`, `.hg`). Today an unset root means the working directory, with no walk-up.
- **Project environment per root:** an interpreter configured in `<root>/.robotframework/rf-mcp.toml`, else
  `<root>/.venv`. That venv is used directly when it is rf-mcp's own environment or holds a robotcode rf-mcp
  supports; otherwise rf-mcp's pinned robotcode is layered onto it with uv. Next comes the activated environment
  (`VIRTUAL_ENV`) or a `robotcode` on `PATH`, and last rf-mcp's own robotcode, whose results carry a warning that
  project libraries are not visible.
- **uv is required** for the layering step. Without uv, rf-mcp uses the remaining steps and says the project needs
  its own robotcode.
- **`rf-mcp init` gets `--scope project|user`.** Project scope writes path-free entries (`uv run rf-mcp` for a
  project install, `rf-mcp` for a global one). User scope registers `rf-mcp` in each agent's user-level
  configuration. Both merge into existing files instead of replacing them. **BREAKING:** the generated command no
  longer contains `--directory`, `-r`, `--mode` or `--transport`.
- **`rf-mcp doctor`** reports how it was installed (project or global), which root was found and how, which
  environment and robotcode a project gets and why, whether uv is available, and the project's Python version.
- **Missing-robotcode errors** name the fix for each install kind instead of always saying "install robotcode into
  the project".
- **Docs** lead with the global install and keep the project install. `--root` and `-p` stay as overrides, and
  Docker and remote HTTP runners keep `--root /workspace`.
- **Unchanged:** client mode, the HTTP remote runner, the Docker images and every tool's parameters.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `hybrid-execution-modes`: local server mode resolves the project root and the project environment per call and
  supports a global install alongside a project install. The missing-robotcode error changes with the install
  kind.
- `mcp-skills-provider`: `rf-mcp init` gains scopes, writes path-free commands and merges into existing agent
  configuration. `rf-mcp doctor` reports root and environment resolution.

## Impact

- **Code:**
  - `src/robotmcp/config.py`: root resolution, with an unset root no longer meaning the working directory up front.
  - `src/robotmcp/runner/cli_runner.py` and `session_runner.py`: robotcode is resolved per root, replacing
    `resolve_robotcode_executable()`.
  - A new environment resolver module in `src/robotmcp/runner/`.
  - `src/robotmcp/onboarding/cli.py`: `init` scopes and merging, and `doctor`.
- **Dependencies:** none new for rf-mcp. uv becomes a runtime requirement for the layering step only.
- **Docs:** `README.md`, `docs/GETTING_STARTED.md`, `docs/CONFIGURATION.md`, `docs/MCP_TOOLS.md` (regenerated if
  the `root` description changes), and `docs/DOCKER.md` (only to say `--root` stays).
- **Users:**
  - Existing project installs keep working. Re-running `rf-mcp init` replaces the absolute-path entries.
  - A user-level registration and a project-level one named `rf-mcp` can coexist. Which one an agent picks is the
    agent's precedence rule, verified per agent in this change.

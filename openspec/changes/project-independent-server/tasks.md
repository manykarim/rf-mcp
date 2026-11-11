# Tasks

## 1. Project root per call

- [x] 1.1 Make `ServerConfig.workspace_root` optional (`None` = not pinned) and add a walk-up helper with robotcode's markers and order (`.robot.toml`, `robot.toml`, `pyproject.toml`, `.git`, `.hg`). Verify with `tests/test_config.py` cases: pinned root, root argument, start in a subfolder of a `robot.toml` project, no marker (working directory itself).
- [x] 1.2 Add `resolve_call_root(root_arg, ctx)` (design D1). In an async middleware, ask older-protocol clients that declared `roots` through `ctx.session.list_roots()` and skip the step on the new protocol (design D1), cache the roots per session until `notifications/roots/list_changed`, ignore non-`file://` roots, and raise the multi-root error naming the roots. Verify with in-memory FastMCP client tests in `tests/test_server_tools.py`: single root, two roots with and without a matching working directory, a client without the roots capability, and a pinned root that beats client roots.
- [x] 1.3 Route every server-mode tool through `resolve_call_root`, and compute the startup root for instructions, project skills and default hints from the pinned root or the walked-up working directory. Verify that the full suite passes and that a server started in `<project>/tests` serves `<project>/.robotframework/skills`.
- [x] 1.4 Document the root order in `docs/CONFIGURATION.md` (the `-r` / `RF_MCP_ROOT` row and a "How the project root is found" section). Update the `root` parameter description if it changes, and regenerate `docs/MCP_TOOLS.md` with `scripts/generate_tool_reference.py`; `tests/test_tool_reference.py` must pass.

## 2. Project environment per root

- [x] 2.1 Create `src/robotmcp/runner/environment.py` with `ProjectEnvironment` and `resolve_environment(root)` in the order of design D2 to D6: configured `.robotframework/rf-mcp.toml` `python`, `<root>/.venv` (own, supported, layered), `VIRTUAL_ENV`, then `PATH`, then fallback. Verify with `tests/test_environment.py` against fake venvs in `tmp_path` (stub interpreters that print a version): every step, own-environment detection, a relative configured path, and Windows `Scripts` alongside POSIX `bin`.
- [x] 2.2 Implement the version probe (D3) and the layered prefix pinned to rf-mcp's own robotcode (D4), including the warm-up probe and the fall-through that keeps uv's error text. Verify with unit tests for the built argv, uv missing, and a failing probe.
- [x] 2.3 Cache resolutions per root, invalidated by a missing interpreter or executable and by mtime changes of `pyvenv.cfg` or `site-packages` (D2). Verify with unit tests that a second call does not probe again and that installing a stub `robotcode` invalidates the entry.
- [x] 2.4 Add `tests/test_environment_integration.py`, skipped when `uv` is absent. In a temp venv with `robotframework` and `robotframework-requests` but no robotcode, reproduce the 2026-11-11 measurements: `rf_analyze` reports no RequestsLibrary import error; `rf_run` passes; an `rf_repl` start, two sends (`Import Library    RequestsLibrary`, `${r}=    Evaluate    6*7` reports 42) and a stop; a venv holding robotcode 1.5.0 still holds 1.5.0 afterwards. Verify it passes locally on Windows and in the CI matrix.
- [x] 2.5 Document environment resolution in `docs/CONFIGURATION.md` ("Which Python runs your tests": the order, `rf-mcp.toml`, the uv requirement, first-call latency and the fallback warning). Verify that the documented `rf-mcp.toml` example resolves in a unit test.

## 3. Runners use the resolved environment

- [x] 3.1 Replace `resolve_robotcode_executable()` in `cli_runner.py` and `session_runner.py` with `resolve_environment(root).command_prefix`, keeping `cwd`, the environment variables, timeouts and process-tree kill. Verify that `tests/test_runner.py`, `tests/test_session_runner.py` and `tests/test_session_fixes.py` pass and that a new test asserts the layered prefix reaches `run_process`.
- [x] 3.2 Add the fallback warning (D7) to results and the import-error note to `rf_analyze` diagnostics under the fallback, and rewrite the missing-robotcode `instruction` per install kind (spec: "robotcode is not installed in the server environment"). Verify with tests for a global-install fallback, uv missing, and a project install with no robotcode.
- [x] 3.3 Update the troubleshooting table in `docs/GETTING_STARTED.md` (the "`robotcode` was not found" row and a new "project libraries are not visible" row). Verify that the messages in the table match the strings the tests assert.

## 4. `rf-mcp init` scopes and merging

- [x] 4.1 Add `--scope project|user` (default `project`) and the command table of design D8, with no arguments in stdio entries. Verify with `tests/test_onboarding_cli.py`: a project uv install gives `uv run rf-mcp`, a plain venv gives `<python> -m robotmcp`, a global install gives `rf-mcp`, and `--url` still gives an HTTP entry in both scopes.
- [x] 4.2 Merge into existing project files: JSON key replacement under the agent's server map, tomlkit for `.codex/config.toml`, no change plus the printed entry for JSONC with comments, no change and a non-zero exit for unparsable files. Verify with tests in which `github` survives next to a replaced `rf-mcp` in `.mcp.json`, Codex comments survive, and invalid TOML stays byte-identical.
- [x] 4.3 Implement the user-scope targets of D8, using `claude mcp add --scope user` when available and printing the command otherwise, with every home path taken from `Path.home()` and environment variables so tests can redirect them. Verify with tests that set `HOME`, `USERPROFILE`, `APPDATA`, `CODEX_HOME` and `XDG_CONFIG_HOME` to `tmp_path` and stub `claude`, and that nothing is written under the current folder.
- [x] 4.4 Rewrite the install sections of `README.md` and `docs/GETTING_STARTED.md`: global install first (`uv tool install rf-mcp`, adding a pre-release flag only if a dry-run resolution proves it necessary, then `rf-mcp init --scope user`), project install second (`uv add --dev rf-mcp`, `rf-mcp init`), and the bare `rf-mcp` command instead of `rf-mcp --mode server -r /path/to/project -p ci` (keep `-r`/`-p` as override examples). Note in `docs/DOCKER.md` that `--root /workspace` stays pinned. Verify by running each documented command in a scratch project on Windows and checking that the agent config it writes matches the docs.

## 5. `rf-mcp doctor`

- [x] 5.1 Report the install kind, the resolved root with the rule or marker that found it, the environment source, interpreter, robotcode version and whether it is layered, the uv version and path, and the interpreter's Python version, all through the server's resolvers (D9). Exit non-zero on the fallback. Verify with `tests/test_onboarding_cli.py` for a global install over a venv without robotcode (exit 0, layered) and for uv missing with no project robotcode (exit 1, both remedies).
- [x] 5.2 Make `doctor` warm the layered environment and print how long that took. Document the new report fields in `docs/GETTING_STARTED.md`. Verify by running `rf-mcp doctor` in the scratch project from 2.4 and checking that the output matches the documented example.

## 6. Agent behaviour checks (need the user's go-ahead for model calls)

- [x] 6.1 With the probe server from the exploration (it logs the working directory and the client's `initialize` capabilities, then calls `roots/list`), start one minimal session per agent (Claude Code, Codex, Copilot in VS Code, OpenCode, Antigravity) on the cheapest model, after the user approves the calls. Record per agent the working directory and roots in `docs/CONFIGURATION.md`. Verify that the table lists all five agents.
- [x] 6.2 Check each agent's user-level configuration location and format against D8, by reading each agent's documentation or running its own `mcp add`/`mcp list` against a redirected home. Correct `init` and its tests where they differ. Verify that `rf-mcp init --scope user --agent <agent> --write` followed by the agent's `mcp list` shows `rf-mcp` for each available agent.
- [x] 6.3 For each agent from 6.2, register `rf-mcp` at both user and project level with distinguishable commands, and record which one the agent starts in a project. Document the result in `docs/GETTING_STARTED.md`. Verify that the documented precedence matches the observed one for every agent tested.

- [ ] 6.4 Measure Copilot in VS Code (working directory, roots, user-vs-workspace precedence) in the VS Code GUI, which has no headless mode; 6.1 and 6.3 cover Claude Code, Codex, the Copilot CLI, Antigravity and OpenCode. Verify by adding its row to the tables in docs/CONFIGURATION.md and docs/GETTING_STARTED.md.

## 7. Integration

- [ ] 7.1 Run the full suite on Python 3.10 and 3.12 (`uv run pytest -q`) and ruff on `src scripts tasks.py`. Verify both are clean, and that the CI matrix (ubuntu, macos, windows) passes on the branch.
- [x] 7.2 End to end with a global install: in a scratch environment, `uv tool install` the built wheel, run `rf-mcp init --scope user --agent claude --write` against a redirected home, and start the server from a sample project through an in-memory or stdio MCP client. `rf_discover`, `rf_analyze` (Browser and Requests imports) and `rf_run` must pass without any project path in the configuration.

## Workflow follow-up

- Add a release-notes entry for the path-free `init` output and the global install, and bump the minor version when releasing.
- Archive the change after review.

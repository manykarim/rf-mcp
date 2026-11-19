# Proposal

## Why

robotcode 2.9.0.dev214 (2026-11-17) removed `robotcode session restart`. Upstream commit 86c3d79 says: "A session lives in one process from `session start` to its end. To run a REPL or a debug run again, end the session with `session stop` and start a new one with the same options."

rf-mcp requires `robotcode[all]>=2.9.0.dev115`, so every fresh install already gets dev214 or later. On those installs:

- `rf_repl` and `rf_debug` with `action="restart"` fail with robotcode's usage error (exit 252).
- `rf-mcp doctor` reports that robotcode doesn't meet the minimum, because `restart` is missing from `session --help`.
- Five tests fail, so the lock stays at dev115, the version CI tests.

`restart` came into rf-mcp because robotcode had it: the `restructure-robotcode-mcp` change mapped robotcode's session commands one to one. No recorded agent run depends on it. Rebuilding it inside rf-mcp would take a per-process record of start options, guesswork in client mode and an error path for sessions rf-mcp didn't start. It would still be weaker than robotcode's own command, because breakpoints added with `.break` can't be carried over. A stop followed by a start does the same job, and one fewer action is one fewer choice for small models to get wrong.

## What Changes

- **BREAKING (tool interface):** `restart` is removed from the `rf_repl` and `rf_debug` `action` values. A call with `action="restart"` fails validation, and the error lists the remaining actions.
- **Reload guidance.** The `rf_repl` and `rf_debug` action descriptions and the `rf-robotcode` skill say that a session loads Python libraries and resource files once: after editing them, `stop` the session and `start` a new one with the same options.
- **doctor.** `rf-mcp doctor` no longer requires `session restart`: robotcode 2.9 with the other session subcommands meets the minimum.
- **Dependency.** `uv.lock` moves to robotcode 2.9.0.dev230 (the newest dev release), and the `robotcode[all]>=2.9.0.dev115` requirement stays.
- **Docs.** `docs/MCP_TOOLS.md` (generated), the README and the `rf-robotcode` skill references drop `restart`.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- **`robotcode-mcp-tools`:**
  - The `rf_repl` and `rf_debug` requirements no longer list `restart` or `session restart`.
  - The hot-reload scenario becomes "reload edited code with stop and start".
  - A new requirement covers the reload guidance, the rejected `restart` action, and doctor without `session restart`.
- **`hybrid-execution-modes`:** the layered-session requirement no longer lists `restart` among the actions a later call can send.

## Impact

- **Code:**
  - `src/robotmcp/runner/session_runner.py`: `ReplAction`, `DebugAction` and `_common_tail`.
  - `src/robotmcp/server.py`: the `action` field descriptions.
  - `src/robotmcp/onboarding/cli.py`: `REQUIRED_SESSION_COMMANDS`.
- **Tests:** `tests/test_session_runner.py` (the live REPL test restarts a session) and `tests/test_onboarding_cli.py` (the session commands that doctor requires).
- **Docs and skills:** `docs/MCP_TOOLS.md`, `README.md`, and `src/robotmcp/skills/builtin/rf-robotcode/references/` (`debugging.md`, `repl-authoring.md`, `practices.md`).
- **Dependencies:** `uv.lock` (robotcode and its `robotcode-*` sub-packages go from 2.9.0.dev115 to 2.9.0.dev230). `pyproject.toml` stays unchanged.
- **Not affected:** the published rf-mcp 0.36.0, and the other session actions (`start`, `send`, `wait`, `status`, `result`, `interrupt`, `list`, `save`, `stop`), which work unchanged on dev214 and dev230.

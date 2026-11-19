# Tasks

## 1. Remove the action (design D1, D2)

- [x] 1.1 Remove `restart` from `ReplAction` and `DebugAction` in `src/robotmcp/runner/session_runner.py`, and from the `("wait", "restart")` branch in `_common_tail`. Verify with a unit test that `rf_repl` and `rf_debug` with `action="restart"` fail validation, list the valid actions, and run no process (fake `run_process`, as in `tests/test_session_fixes.py`).

## 2. Reload guidance (design D3)

- [x] 2.1 Update the `rf_repl` and `rf_debug` `action` field descriptions in `src/robotmcp/server.py` (lines 467 and 535): no `restart`, plus the sentence that edited libraries and resources load after `stop` and a new `start`. Regenerate `docs/MCP_TOOLS.md` with `scripts/generate_tool_reference.py`. Verify with `test_tool_reference_is_current` and a test that both descriptions contain `stop` and `start` and not `restart`.
- [x] 2.2 Replace the `restart` lines in `src/robotmcp/skills/builtin/rf-robotcode/references/debugging.md` and `repl-authoring.md` with the stop-and-start guidance (breakpoints added with `.break` are not kept), and drop `restart` from the P29 row of `practices.md`. Update the `rf_repl`/`rf_debug` lines in `README.md` (37–38). Verify with `grep -rn -i "restart" src/robotmcp/skills README.md docs/MCP_TOOLS.md`, which finds no session restart.

## 3. doctor and dependency (design D4)

- [x] 3.1 Remove `restart` from `REQUIRED_SESSION_COMMANDS` in `src/robotmcp/onboarding/cli.py` and from the expected set in `tests/test_onboarding_cli.py:109`. Verify that `test_doctor_reports_subcommands_and_hints_status`, `test_run_doctor_verifies_robotcode_2_9_and_session_extra` and `test_sync_skills_copies_builtin_and_project_skills` pass with robotcode dev230.
- [x] 3.2 Run `uv lock --upgrade-package robotcode`, so robotcode and its `robotcode-*` packages are at 2.9.0.dev230 (released 2026-11-19, after this plan; dev214 was planned), and keep `robotcode[all]>=2.9.0.dev115` in `pyproject.toml`. Verify with `uv lock --check`, and with a `git diff pyproject.toml` that shows no change.

## 4. Tests

- [x] 4.1 Rename `test_session_runner_live_repl_multi_turn_restart_and_save` and replace its restart step with a stop and a new start: the old id is no longer listed, a variable set before the stop is gone in the new session, and the new session is stopped in `finally`. Verify by running it with robotcode dev230.
- [x] 4.2 Run the full suite against the dev230 lock (`uv run pytest`) and confirm the five tests that failed on dev214 before the change (`test_doctor_reports_the_layered_environment`, the three doctor and sync-skills tests in `test_onboarding_cli.py`, and the live REPL test) now pass, and that nothing else fails apart from the three local-only Browser cookbook tests. Run `ruff` and `mypy` on `src/`.

## Workflow follow-up

- Open a PR. Merge only after CI passes and the user asks.
- Archive the change with `/opsx:archive` (it syncs the `robotcode-mcp-tools` and `hybrid-execution-modes` deltas into the main specs).

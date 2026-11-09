# Proposal

## Why

The 2026-11-07 Codex CLI sweep (GPT-5.6-Luna at its default window and capped at 32k, GPT-6-Luna at 32k; 148
graded runs) showed failure modes that the Claude runs did not show. Single rf-mcp responses of 81-83k
characters filled most of a 32k window, one-line REPL sends stretched a 27-step flow over 27 calls, a mixed
`.kw`/keyword batch was refused by robotcode, `rf_results(path=...)` failed and was abandoned, and failing tests
were rerun again and again instead of debugged. Codex also hides MCP server instructions and tool descriptions
from the model, so without synced skills the agent ignored rf-mcp, and under `approval_policy="never"` it
rejected every tool not annotated read-only. These are cheap fixes that make rf-mcp usable for small models and
small context windows.

## What Changes

- Successful server-mode responses no longer echo the robotcode argv (`command`); failures keep it.
- REPL and debug output, and any single dominant string in structured data, is capped (20k characters by
  default). A cut `.doc` points to `.kw <name>` or `rf_doc action='keywords'`. A merged batch is capped as a whole.
- `rf_results action='log'` drops log noise: start times, timestamps, `isHtml: false`, NOT RUN branches, and
  absolute `source` when `relSource` exists.
- `rf_repl send` with several lines sends `.`-commands as separate parts, like `rf_debug` does, and stops at the
  first failed part.
- After 3 one-line REPL sends in a row (then every 10th), the response asks for batched sends.
- After the same test fails in 3 `rf_run` calls in a row (then every 3rd), the response points to `rf_debug`.
- `rf_results` accepts `path`, `result_file` and `output_file` as aliases of `output_path`.
- `rf_analyze severity` reports that severity and worse, as documented (robotcode's `--severity` reports only the
  listed ones, so `severity='info'` hid every error from small OpenRouter models on 2026-11-09).
- `rf_run` responses start with `outcome` (`PASS: 3 tests, 3 passed, 0 failed` / `FAIL: ...` with "fix and rerun
  before you report the task as done"), and a clean `rf_analyze` says it ran nothing: on 2026-11-09 Qwen3-Coder
  reported "successfully created" suites after a failed run or after analysis only.
- An empty argument (`""`, `"[]"`, `"{}"`, `"null"`, `[]`, `{}`) means unset for every tool, and list arguments
  accept Python list text such as `"['Tests.A']"` (models behind Claude Code on OpenRouter send every optional
  argument, and Qwen3-Coder's stringified `by_longname` selected no tests).
- `rf-mcp init --agent codex` explains `rf-mcp sync-skills --agent codex` and the opt-in
  `default_tools_approval_mode = "approve"` in the generated config.
- The agent-eval harness drives Codex CLI (`cx-<model>[@<N>k][~<effort>]`, local `cx-oss-<model>`, OpenRouter
  `cx-or-<vendor>.<model>`) and Claude Code on OpenRouter (`or-<vendor>.<model>`), with the `mcp+sync` setup,
  and maps Codex `--json` events onto the existing graders. A new process flag `unverified` marks runs that wrote
  a suite and had no passing run after the last write.

## Capabilities

### New Capabilities
- `agent-context-budget`: how rf-mcp keeps tool responses within a small model's context window and steers
  it out of call-wasting loops (response size, log compaction, batching and rerun nudges, batch splitting).
- `coding-agent-eval-harness`: the `tests/e2e/agent_eval` harness that drives coding-agent CLIs on Robot
  Framework scenarios and grades outcome and process; this change adds the Codex CLI driver.

### Modified Capabilities
- `mcp-skills-provider` (introduced by `restructure-robotcode-mcp`): adds a requirement that the Codex wiring
  from `rf-mcp init` states how to make rf-mcp usable from Codex (skill sync, tool approval).

## Impact

- `src/robotmcp/runner/cli_runner.py` (`to_response`, `truncate_data`, `compact_result_log`),
  `src/robotmcp/runner/session_runner.py` (`split_debug_batch`, `_send_batch`, `_batching_nudge`, output cap),
  `src/robotmcp/args.py` (`ArgumentAliases` middleware, `coerce_list`), `src/robotmcp/server.py` (`FailureStreaks`),
  `src/robotmcp/onboarding/cli.py` (Codex notes).
- `tests/e2e/agent_eval/` (env, runner, graders, report, README).
- No new dependencies. Response shapes stay the same except that `command` is gone from passed server-mode
  calls; clients that read it on success must use `shell_command` in client mode or the failure response.
- Depends on `restructure-robotcode-mcp` (the rf_* tools); archive that change first.

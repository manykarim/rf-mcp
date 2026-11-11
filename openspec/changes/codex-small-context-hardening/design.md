# Design

## Context

Every change comes from a transcript of the 2026-11-07 Codex sweep (see proposal.md - Why, and the "Codex CLI
with small models" section of `tests/e2e/agent_eval/README.md`). The rf_* tools already return a structured
`to_response()` dict with an `instruction` field for hints, cap structured data with `truncate_data`, and split
`rf_debug` sends with `split_debug_batch`. Codex gives the model nothing from the MCP server except the tool
names, so tool descriptions and server instructions don't reach it. Only response text and synced skills do.

## Goals / Non-Goals

**Goals:**
- Keep each response well below a 32k-token window and remove fields that carry nothing on success.
- Put loop-breaking guidance into the responses themselves, because that is the only rf-mcp text Codex shows.

**Non-Goals:**
- Changing tool names, schemas or defaults for Claude-style clients. The aliases are not published in the schema.
- Making `default_tools_approval_mode = "approve"` the generated default.
- Detecting edit-and-rerun loops outside `rf_run`, for example plain `robot` in the shell.

## Decisions

- **Nudges go in `instruction`, not in an error.** The call result is unchanged and the agent can ignore the
  nudge. Thresholds are 3, then every 10th for sends and every 3rd for reruns, so that a long legitimate session
  is not flooded. The alternative, refusing the 4th one-line send, would break agents that send one line on purpose.
- **The failure streak is keyed by test name** (the text before `": "` in the failure line), held per server
  process. A server is one agent session, so no cross-session state is needed. Keying on the full message would
  reset the count whenever the asserted value changes, which is exactly what happens while the agent tries fixes.
- **Argument aliases are middleware.** Renaming in a FastMCP middleware before validation keeps the published
  schema plain. Adding the aliases as schema parameters would be the alternative, but every extra parameter is
  text a small model has to read.
- **Cut the dominant string before shortening lists.** A list cut loses whole items, such as keywords or tests,
  while one 80k-character markdown blob is rarely needed in full.
- **REPL split only on dot-lines.** robotcode refuses dot-commands mixed with keywords, but keyword lines run
  fine as one block. Splitting every line would cost a round-trip per keyword, which is the waste B is meant to
  stop.
- **`command` is dropped only for passed server-mode calls.** On failure, the argv helps the agent see a wrong
  selection. In client mode, `shell_command` is the payload.
- **Codex harness writes a project `.codex/config.toml`.** `-c mcp_servers...` overrides did not take effect in
  Codex 0.166. A project file also mirrors what a user gets from `rf-mcp init --agent codex`.

## Risks / Trade-offs

- [Clients reading `command` from successful responses] → It is still present on failure and in client mode. The
  response shape is documented in the tool tests.
- [Nudge text adds tokens] → It is one sentence, and it fires at most every 10th send or every 3rd run.
- [Cap hides needed output] → The note says how much was cut, and for `.doc` it names the narrower command.
- [One run per cell can't show an effect on turns or tokens] → Changes are justified by the transcript evidence
  and by unit tests. Measuring them needs repeated runs, which is follow-up work.

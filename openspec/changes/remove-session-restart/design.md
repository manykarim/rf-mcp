# Design

## Context

- **How `restart` works today.** `ReplAction` and `DebugAction` in `src/robotmcp/runner/session_runner.py` include `restart`. `_common_tail` handles it like `wait` and builds `session restart --timeout N <id>`. The tools in `src/robotmcp/server.py` take `action` as `tolerant_literal(ReplAction)` / `tolerant_literal(DebugAction)`, so the allowed values come from those Literals, and so do the schema and `docs/MCP_TOOLS.md`.
- **What robotcode dev214 offers** (checked locally): no `session restart`. `session start` can't choose an id, and `session list` doesn't report a session's start options.
- **Where agents learn about `restart`:**
  - the `action` enum and descriptions (`server.py:467` says "restart (reload edited libraries)");
  - `rf-robotcode` references: `debugging.md:41`, `repl-authoring.md:27` and the P29 row in `practices.md`;
  - `README.md` lines 37–38.
- **Who checks it:** `REQUIRED_SESSION_COMMANDS` in `src/robotmcp/onboarding/cli.py:34` (doctor), `tests/test_onboarding_cli.py:109`, and the live test `test_session_runner_live_repl_multi_turn_restart_and_save`.

## Goals / Non-Goals

**Goals:**
- rf-mcp works with robotcode dev214, and the lock can move to it.
- Agents still know how to pick up edited libraries and resources.

**Non-Goals:**
- **Emulating `restart`.** See D1.
- **Changing the other session actions.**
- **Helping clients that still send `restart`** beyond the validation error (D2).

## Decisions

### D1. Remove `restart`; don't emulate it

`restart` came in only because robotcode had it, and no recorded agent run relies on it.

**Alternative:** rf-mcp remembers each session's start options and runs stop + start. Rejected:
- it needs a per-process record, guesses in client mode, and an error for sessions rf-mcp didn't start;
- it can't keep breakpoints added with `.break`, so it's weaker than the command it replaces;
- the session id changes anyway, so agents have to handle a new id either way;
- every extra action is one more choice small models can get wrong.

### D2. A `restart` call fails validation

With `restart` gone from the Literals, `tolerant_literal` rejects it, and the validation error lists the valid values. That's the same treatment as any other unknown action. A special "use stop and start" message isn't worth a code path: the action descriptions already say how to reload.

### D3. Teach reloading where agents look

- `rf_repl` `action` description: "start, send, wait, status, result, interrupt, list, save, stop. Edited libraries and resources load after stop and a new start."
- `rf_debug` `action` description: the same sentence.
- `debugging.md` and `repl-authoring.md`: replace the `restart` lines with "A session loads Python libraries and resource files once. After editing them, `stop` the session and `start` a new one with the same options; breakpoints added with `.break` are not kept."
- `practices.md` P29: drop `restart` from the action list.

### D4. doctor and lock

- `REQUIRED_SESSION_COMMANDS` drops `restart`, so dev214 meets the minimum. A robotcode that still has `restart` passes too, because doctor checks only for required commands.
- `uv lock --upgrade-package robotcode` moves robotcode and its `robotcode-*` packages to the newest dev release, 2.9.0.dev230 (2026-11-19; dev214 was the newest when this was planned). `pyproject.toml` keeps `robotcode[all]>=2.9.0.dev115`.

## Risks / Trade-offs

- **[An agent or saved prompt still calls `restart`]** → It gets a validation error listing the valid actions, and the descriptions say to use stop and start.
- **[A reload now costs two calls and re-entering the start options]** → Accepted. A debug session with many breakpoints and a test selection costs the most to redo.
- **[An agent edits a library and doesn't reload]** → The same risk as before. The descriptions and the skill now say plainly that edits need a new session.

## Migration Plan

There's no data migration. Agents switch to `stop` + `start`. Rollback is a revert, but `restart` would then fail again on robotcode dev214.

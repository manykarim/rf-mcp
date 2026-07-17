# Proposal: agent-ergonomics-fixes

## Why

Two Docker/agent capability spikes (2026-07-10, MiniMax-M3 via opencode then
Claude Code, across 14 scenarios) showed agents can drive rf-mcp well but **lose
turns to unhelpful failures, not missing capability**. Three cheap, high-leverage
gaps recurred (Cluster 1 of the spike findings):

- **F2 — `execute_batch` leaks a bare `'keyword'` error.** A step dict missing the
  `keyword` key hits a raw `s["keyword"]` access (`aggregates.py:111`) →
  `KeyError('keyword')`, surfaced to the agent as just `'keyword'`. In the
  file-proc scenario the agent tried ~10 payload shapes before abandoning
  `execute_batch` for `execute_step`. The error names nothing actionable.
- **F5 — session profiles block standard utility libraries.** An `api_testing`
  session rejected `OperatingSystem` ("Allowed libraries: BuiltIn, Collections,
  DateTime, RequestsLibrary, String, XML") because the allowlist is exactly
  `profile.core + optional + BuiltIn` (`session_models.py:1036`). Writing a result
  file is a near-universal need; agents had to detour through
  `manage_session(action="import_library")`.
- **F4 — `execute_batch` silently lacks `bdd_group`/`bdd_intent`.** Agents inferred
  they must use per-step `execute_step` for BDD grouping, but only by trial. The
  docstring doesn't say so.

Plus a documentation carry-over from the just-shipped `build-suite-safe-persist`
fix: the **`output_path` persistence lesson lives only in the tool docstring**, not
in the server-provided WORKFLOW GUIDE that every agent reads first.

## What Changes

- **F2** — `BatchExecution.create` validates each step is a dict with a non-empty
  `keyword` before construction; on violation it raises an actionable `ValueError`
  ("Step i: missing required 'keyword' … each step is `{keyword, arguments?, …}`")
  instead of leaking `KeyError`. It also rejects a non-list `arguments`/`args`
  (dict or string) with a step-indexed "must be a list" error — previously
  `list(dict)` silently yielded the dict's keys, running the batch with garbage
  arguments (spike §3.2: a dict arg burned 93s in desktop descriptor-retry).
- **F5** — the session library allowlist always includes the domain-agnostic RF
  standard libraries (`OperatingSystem, Collections, String, DateTime, Process`
  alongside the existing `BuiltIn`), so utility keywords aren't blocked in any
  session type. Web libraries (`Browser`/`Selenium`/`Appium`) remain governed by
  the profile.
- **F4** — the `execute_batch` docstring documents that batch steps do NOT support
  `bdd_group`/`bdd_intent`; use per-step `execute_step` for BDD grouping.
- **Docs** — the WORKFLOW GUIDE gains a short "persist generated suites via
  `build_test_suite(output_path=…)`, never via `Create File`" note.

Out of scope (other spike clusters, handled separately): collection-through-string
args (F3, needs a spike), desktop turn-economy + `Take Screenshot` fail-fast
(Cluster 2 spike), MCP fast-handshake (D1 spike).

## Capabilities

### New Capabilities

- `agent-ergonomics-fixes`: malformed `execute_batch` steps produce an actionable
  error; standard utility libraries are allowed in every session type; the batch
  BDD limitation and safe suite-persistence are documented in agent-facing text.

## Impact

- `src/robotmcp/domains/batch_execution/aggregates.py` — step validation in `create`.
- `src/robotmcp/models/session_models.py` — utility libs in `_get_allowed_libraries_for_session_type`.
- `src/robotmcp/server.py` — `execute_batch` docstring (no bdd_group).
- `src/robotmcp/domains/instruction/value_objects.py` (+ `templates/detailed.txt`) — WORKFLOW GUIDE suite-persistence note.
- Tests: `tests/unit/test_agent_ergonomics_fixes.py` — batch error message; utility-lib allow across contexts (incl. api_testing + OperatingSystem); guide/docstring text assertions.

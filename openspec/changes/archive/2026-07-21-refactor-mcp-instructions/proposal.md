## Why

An instruction-quality investigation (~140 live agent runs across 6 instruction sets, 4
scenarios, and 4 models — glm-4.7-flash, qwen3-coder-30b, MiniMax-M3, Haiku 4.5 — plus a
3-agent qualitative review) established that rf-mcp's MCP instruction surface is
**incoherent and over-weight**, and that the current default actively *reduces*
tool-calling quality:

- **The 2800-char `standard` default is consistently among the WORST.** N=5 tie-break:
  suite-exec completion 0.40, XML success 0.48 (lowest of all); worst API earlier (0.61).
  It never wins on any scenario.
- **Shorter wins.** `minimal` (248 chars) is the consistent best-or-near-best on success,
  completion, and right-order with the *fewest turns*; `off` (docstrings only) is
  competitive. Verbose templates (`detailed` 2700, `discovery_first` 6081) have the
  *worst* completion and the *most* turns — more preamble → more meandering.
- **The templates disagree on the first tool.** Three competing "first calls"
  (`find_keywords` / `manage_session init` / `analyze_scenario`); `standard` even
  self-contradicts (step 1 says discover, its RULES say init-first). The tool the schema
  designates as the mandatory front door — `analyze_scenario` ("your FIRST tool call") —
  is named by only 1 of 7 templates, buried at step 4.
- **Dual session entry causes churn.** Both `analyze_scenario` (which *creates* the
  session) and `manage_session(init)` present themselves as "the start." Haiku
  demonstrably churned (redundant `manage_session` calls); one explicit sentence
  eliminated the churn — proving instruction clarity drives behavior. **Validated on the
  real Claude Code CLI (2026-07-20):** once the cold-start stdio bug that hung the client
  on its first keyword was fixed (commit a467c5a), an n=8+n=20 Haiku sweep — candidates
  differing ONLY in server `instructions`, neutral prompt — reached 100% completion, and
  the candidate carrying the explicit "analyze_scenario creates the session; NEVER call
  `manage_session(init)`" line (`checklist`) had **0.0 average session churn** (n=5; also
  fewest avg calls and fastest), tied by `example`, vs 0.2 (`minimal`) / 0.4 (`terse`).
  This is the concrete lean-spine wording adopted (design.md Decision 1).
- **~40% of the long templates is docstring-echo** ("Use find_keywords to search
  keywords") — the tool schemas already carry it; the templates should carry only the
  non-discoverable gotchas.
- **API guidance is unreachable first-try.** The non-obvious RequestsLibrary knowledge
  (`${resp.json()}`, `json=` not `data=`, `… On Session`, native asserts) lives only in
  `get_locator_guidance(library="requests")`, but no template points to it, and
  `manage_session(init)` injects `desktop_guidance` for desktop yet nothing for API —
  the guaranteed-delivery channel (the init response the agent always reads) is unused.
  restful_booker is the turn-sink as a direct result.
- **Dead template files.** `templates/*.txt` are not loaded (only the classmethods are)
  and already disagree with production on the `analyze_scenario` ordering — a latent trap.

Now: the agentic e2e instruction-quality gate (capability `agentic-e2e-instruction-quality`)
exists to catch regressions, so it is safe to refactor the instructions and *measure* the
result rather than fear it.

## What Changes

- **Replace the default template with a lean, order-explicit spine (~250–400 chars).**
  One canonical order aligned to the docstrings: `analyze_scenario` FIRST (it *is* the
  session — do not also call `manage_session`) → discover-if-unknown
  (`find_keywords`/`recommend_libraries`) → `get_locator_guidance` for locators/API →
  `execute_step` → `build_test_suite`. Plus "never guess — discover". Retire `standard`
  as the default.
- **Retire / shrink the verbose templates.** Remove `discovery_first` (6081 chars) as an
  option or collapse it into the lean spine + a one-line pointer to the recovery ladder;
  trim `detailed` toward the lean spine's shape.
- **Guaranteed API-guidance delivery**: inject an `api_guidance` bundle into
  `manage_session(action="init")` when RequestsLibrary is present (mirroring the existing
  `desktop_guidance` injection) so the RequestsLibrary rules land in the init response the
  agent always reads — not conditional on the agent discovering `get_locator_guidance`.
- **Tighten the dense tool docstrings** whose load-bearing lines are buried
  (`get_keyword_info`, `execute_step`, `find_keywords`) — since the docstrings are the
  real driver of tool-calling AND they bloat small-model prompt budgets. **BREAKING for
  agent behaviour** — every docstring/template change is validated by the
  `agentic-e2e-instruction-quality` gate before landing (no regression vs the reference
  baseline).
- **Delete the dead `templates/*.txt`** files (or wire them as the single source of
  truth) so config cannot drift from the live classmethods.
- **Unify the session entry point**: `analyze_scenario` is documented as the single front
  door in both the template and its docstring; `manage_session(init)` docstring points to
  `analyze_scenario` as the normal entry.

## Capabilities

### New Capabilities
- `mcp-instruction-set`: how rf-mcp delivers agent-facing guidance for correct
  tool-calling — the lean order-explicit default template, the single canonical tool
  order + unified session entry, guaranteed init-response guidance injection (API mirrors
  desktop), docstring conciseness requirements, and retirement of the redundant/dead
  templates.

### Modified Capabilities
<!-- none — the instruction templates were never captured as an OpenSpec capability;
     this introduces mcp-instruction-set. Behaviour is guarded by the existing
     agentic-e2e-instruction-quality capability (the gate), which this change relies on. -->

## Impact

- **Instruction templates**: `src/robotmcp/domains/instruction/value_objects.py`
  (lean default, retire/shrink verbose), `src/robotmcp/domains/instruction/templates/*.txt`
  (delete dead files), the resolver (`server.py::_resolve_server_instructions`).
- **Init-guidance injection**: `src/robotmcp/server.py` `manage_session(init)` (add
  `api_guidance` alongside `desktop_guidance`), reusing `utils/requests_guidance.py`.
- **Docstrings**: `analyze_scenario`, `manage_session`, `execute_step`, `get_keyword_info`,
  `find_keywords`, `get_locator_guidance` in `server.py` (tighten; unify session-entry
  wording).
- **Validation**: the `agentic-e2e-instruction-quality` gate (reference model over the
  validated scenario set) must not regress; recapture the reference baseline after the
  intended change (it is the "no decrease" ratchet, so a metric change is a reviewed diff).
- **Non-goals**: changing tool *names* or *signatures*; per-context auto-selection of
  templates (still an env var); lifting Haiku above its capability floor (rf-mcp targets
  medium models — glm-4.7-flash / qwen3-coder-30b — for reliable driving).

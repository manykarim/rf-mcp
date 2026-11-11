# Agent evaluation harness

Runs real agents (`claude -p`) on Robot Framework tasks and grades both the outcome (does the suite pass,
are injected bugs caught) and the process (did the agent use rf-mcp the intended way). It compares

- **`mcp`**: rf-mcp tools (`mcp__rf-mcp__*`) plus Read/Write/Edit/Glob/Grep, Bash/PowerShell disallowed,
  skills served over MCP (`skill://...`);
- **`cli`**: Bash/PowerShell plus the upstream robotcode agent skill in `.claude/skills/robotcode`, no MCP;
- **`mcp:<label>`**: like `mcp`, but rf-mcp starts with `--skills-dir <dir>` (skill A/B: only the skill text differs).
  `--skills-dir` takes precedence over the built-in skills folder by folder name (`rf-robotcode`, `rf-locators`, `rf-requests`).

Every run gets its own project copy (and its own demoshop `X-Workshop-Space` / ledger user), its transcript
(`<run>.jsonl`, stream-json), its MCP config and a `<run>.result.json`; `report.md` summarizes a results folder.

**Real runs call paid models.** Use `--dry-run` to check seeding and commands without starting `claude`.

## Suites

| suite | scenarios | default models x conditions | needs |
| --- | --- | --- | --- |
| `baseline` | s1 triage+fix, s2 smoke by tag, s3 analyze without running, s4 XML authoring, s5 restful-booker CRUD, s6 SauceDemo (Browser), s7 live debug inspection | haiku, sonnet x mcp, cli (28 runs) | RequestsLibrary, Browser, internet |
| `demoshop` | t1 checkout WEB-006 authoring (graded under presets clean/stage4/buggy), t2 API+UI hybrid cart (X-Session-ID), t3 drift repair vs defect, t6 triage of 3 injected test bugs | haiku, sonnet x mcp, cli (16 runs) | clone of github.com/manykarim/ai-engineering-robotframework (main, without `.claude/`), demoshop.makrocode.de |
| `fixtures` | fx_browser / fx_selenium (shadow DOM + iframe widgets on tests/fixtures/web), fx_api (demoshop cart API) | haiku x mcp | Browser + `rfbrowser init`, SeleniumLibrary + Chrome, RequestsLibrary |
| `api` | api_ledger (REST), api_booker (restful-booker), api_soap, api_graphql (tests/fixtures/ledger + one bug server per injected bug) | haiku x mcp | RequestsLibrary; fastapi stack for the fixture |

## Running

```
uv run python -m tests.e2e.agent_eval baseline --models haiku,sonnet --conditions mcp,cli --reps 1 --parallel 6 --out out/baseline
uv run python -m tests.e2e.agent_eval demoshop --models haiku,sonnet --conditions mcp,cli --parallel 4 --out out/demoshop
uv run python -m tests.e2e.agent_eval fixtures --conditions mcp:v1,mcp:v2 --skills v1=path/skills_v1 --skills v2=path/skills_v2 --reps 2 --out out/fx
uv run python -m tests.e2e.agent_eval api --conditions mcp:v1,mcp:v2 --skills v1=... --skills v2=... --reps 3 --out out/api
uv run python -m tests.e2e.agent_eval baseline --dry-run --out out/dry        # seed + print commands, no claude
uv run python -m tests.e2e.agent_eval report out/baseline                     # (re)write out/baseline/report.md
uv run python -m tests.e2e.agent_eval trim run.jsonl tests/e2e/agent_eval/recorded/run.jsonl
```

Options: `--scenarios s1_triage_fix,s3_analyze`, `--max-turns`, `--timeout` (seconds per run), `--cli-skill <dir>`.

`.github/workflows/e2e-weekly.yml` runs the baseline suite every Sunday on Linux with `or-z-ai.glm-5.3-flash`
(condition `mcp`, needs the `OPENROUTER_API_KEY` secret) and publishes `report.md` as the job summary; manual runs
take `models`, `scenarios` and `reps`. Pull requests that touch the harness get a `--dry-run` of the same command.
Model names `haiku` = `claude-haiku-5-5`, `sonnet` = `claude-sonnet-5-5`; other values are passed through as ids.

### Codex CLI

A model named `cx-<model>[@<N>k][~<effort>]` runs through `codex exec --json` instead of `claude -p`
(`cx-gpt-5.6-luna`, `cx-gpt-5.6-luna@32k`, `cx-gpt-6-luna@32k~medium`; effort defaults to `low`). `@32k` sets
`model_context_window` (auto-compaction at 80%), so a large-window model stands in for a small-context one.
`cx-oss-<model>` (`cx-oss-qwen3_14b` = `qwen3:14b`) runs a local Ollama model with `model_provider="ollama"`
(`--oss --local-provider ollama` fails in Codex 0.166 with "your current provider is openai"). Codex has no
metadata for Ollama models, so the harness sets their window to `OLLAMA_CONTEXT_LENGTH` (default 32768). Start
the Ollama server with that variable set: its own default (4k-8k) is below Codex's base prompt, and Ollama cuts
longer requests without an error. Tested with Ollama 0.40.15; with an Ollama provider Codex calls MCP tools
directly instead of through its code-mode `exec`.

- Runs use Codex's `workspace-write` sandbox with network access and `approval_policy="never"`; the user's Codex
  apps, plugins, browser/computer use, image generation, multi-agent and goals are disabled. On Windows the
  sandbox refuses the Store `pwsh.exe`, so the harness drops `WindowsApps` from the agent's PATH and Codex uses
  System32 `powershell.exe` (runs before 2026-11-07 17:30 lost one or more shell calls to that: GPT models fell
  back to `cmd.exe`, qwen3:8b gave up). The sandbox also refuses Chromium (`spawn EPERM`): in the `cli`
  condition every Browser test fails for that reason, while rf-mcp runs outside the sandbox.
- `mcp` writes `.codex/config.toml` with the rf-mcp server and `default_tools_approval_mode = "approve"` (the
  counterpart of `--allowedTools mcp__rf-mcp`; without it Codex rejects every tool not annotated read-only).
  Shell stays available. `mcp+sync` also runs `rf-mcp sync-skills` into `.agents/skills`.
- `cli` copies the robotcode skill to `.agents/skills/robotcode`.
- The transcript is mapped onto Claude-style calls (`mcp__rf-mcp__rf_run`, `Bash`/`PowerShell` with the shell
  wrapper removed, `Write`/`Edit` per changed file), so every grader works. `num_turns` is tool calls + 1 (codex
  exec reports no request count) and there is no cost; compare input tokens instead.

### OpenRouter models

`or-<vendor>.<model>` runs Claude Code on OpenRouter's Anthropic-compatible API (`or-z-ai.glm-5.3-flash` =
`z-ai/glm-5.3-flash`; the first `.` stands for the `/` that run ids cannot contain). Every Claude Code model role
(background calls, subagents) is set to that model, so no request goes to an Anthropic model; the reported
`total_cost_usd` is Claude Code's estimate at Anthropic prices, not what OpenRouter charged. `cx-or-<vendor>.<model>`
runs Codex on OpenRouter's Responses API (`wire_api="responses"`, a 128k window unless `@<N>k`) with the same
model catalog entry as Ollama models, but with `apply_patch` kept: OpenRouter passes the freeform tool through.
`OPENROUTER_API_KEY` (and optionally `OPENROUTER_BASE_URL`) come from the environment or the git-ignored `.env`
in the repo root; only the agent process gets the key, not the MCP server or graders.
The fixtures are started once per suite run on free ports (`fixtures`: web pages; `api`: ledger + bug servers).

## Environment variables

| variable | meaning |
| --- | --- |
| `RF_EVAL_BASELINE_PYTHON`, `RF_EVAL_DEMOSHOP_PYTHON`, `RF_EVAL_FIXTURES_PYTHON`, `RF_EVAL_API_PYTHON` | Python of the environment the project runs in (robotcode + libraries). If rf-mcp is installed there (`uv pip install -e <repo>`), the MCP server runs from it as `python -m robotmcp` and sees the same libraries; otherwise the config uses `uv run --directory <repo> rf-mcp`. Grader runs use `robotcode` next to that Python. |
| `RF_MCP_COOKBOOK_PYTHON` | fallback for `fixtures` and `api` (as in tests/test_cookbook.py) |
| `RF_EVAL_PYTHON` | fallback for all suites (default: the harness interpreter) |
| `RF_MCP_FIXTURE_PYTHON` | Python with fastapi, uvicorn, python-multipart, strawberry-graphql for the ledger fixture |
| `RF_EVAL_CLI_SKILL_DIR` | upstream robotcode skill folder (`SKILL.md` + `references/`) for the `cli` condition |
| `RF_EVAL_DEMOSHOP_SRC` | local checkout of the workshop repo instead of cloning (`RF_EVAL_DEMOSHOP_REPO` overrides the URL) |
| `RF_EVAL_CACHE`, `RF_EVAL_RUNS_DIR` | clone cache (default `%TEMP%/rf-mcp-agent-eval/cache`), project copies (default `<out>/projects`) |
| `RF_EVAL_SHOP_URL`, `RF_EVAL_CLAUDE` | demoshop URL, claude executable |
| `PLAYWRIGHT_BROWSERS_PATH` | passed to the MCP server (Browser library browsers, e.g. `%LOCALAPPDATA%\ms-playwright`) |
| `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL` | OpenRouter access for `or-` and `cx-or-` models (else read from `.env`; base URL default `https://openrouter.ai/api/v1`) |

All runs of a suite share one project environment. Shell agents (`cli`) therefore get `uv` / `pip` shims first on
PATH (`<out>/_guard/`) that refuse `uv sync|add|remove|lock|venv`, `uv pip install|uninstall|sync` and
`pip install|uninstall`; everything else is forwarded. The harness also fingerprints the environment (`uv pip freeze`)
when the sweep starts: a run that leaves it changed gets `env_changed: true`, and later runs are skipped
(`exit: "env_changed"`) instead of being paid for in a broken environment. (2026-11-06: one `uv sync` by a Haiku
CLI run synced the shared demoshop venv to the workshop lockfile and uninstalled rf-mcp mid-sweep.)

Reference machine: `RF_EVAL_BASELINE_PYTHON=C:\rflab\base\.venv\Scripts\python.exe`,
`RF_EVAL_DEMOSHOP_PYTHON=C:\rflab\ws\venv\Scripts\python.exe`, `RF_MCP_COOKBOOK_PYTHON=C:\rflab\fx\venv\Scripts\python.exe`,
`RF_MCP_FIXTURE_PYTHON=C:\rflab\api\venv\Scripts\python.exe`.

## Graders

Outcome (`graders.grade_*`, run robotcode in the project env): s1 test passes with unchanged expectations and a fixed
library; s2 ran by tag and names the 3 smoke tests; s3 found both problems without running; s4-s6 suite exists and passes
(`-bl <longname>`); s7 no file changes, the answer names the cart values, the debugger was used; demoshop per task
(presets, drift repaired / defects kept failing, triage written, test bugs fixed); fixtures/api: tests passed/total,
rerun passes, `bugs_caught` against each bug server, `isolated_pass` (each test alone with `-t` and a fresh ledger user),
static pitfalls in the suite. Every run also records `written_analyze_errors` (robotcode analyze on the files it wrote).

Process (`graders.process_grades`, pure functions over the transcript; `flags` in the result and the report):

| flag | anti-pattern |
| --- | --- |
| `output_xml` | Read/Grep/Glob/Bash touching a result XML (output.xml), or REPL/debug statements parsing it |
| `file_selection` | `rf_run`/`rf_debug` with `paths` naming one .robot file, or `robotcode robot`/`robot` with one .robot path |
| `run_before_analyze` | a run selecting .robot/.resource files written or edited after the last `rf_analyze` covering them |
| `open_session` | `rf_repl`/`rf_debug` start without a matching stop while the final answer does not mention an open session |
| `results_late` | "why did X fail?" tasks (s1, t3, t6): project files read before `rf_results` / `robotcode results` |

Counters: REPL sends, multi-line sends, failed sends, pitfalls in failed results (keyword not imported, JSON as text,
frame/shadow not found, ...), `expected_status=any` probes, token as JSON, skill reads; static counts in the produced
suite: `json={`, `Sleep`, `force=True`, `blocked` tags, `expected_status`, `Set Suite/Global Variable` chaining.

`test_graders.py` checks the graders offline on synthetic transcripts and on trimmed recordings in `recorded/`
(2026-11-05/06 runs: both s1 MCP runs grep/read output.xml, the s1 Sonnet CLI run is clean).

## Cost and time (reference runs)

| suite | runs | cost | wall per run (median / max) |
| --- | --- | --- | --- |
| baseline 2026-11-05 | 28 | $1.76 | 34 s / 106 s |
| demoshop | 16 | about $4.90 (Haiku + Sonnet x MCP + CLI) | 175 s / 14 min |
| fixtures (2 reps, 2 skill versions) | 12 | $0.28 | 116 s / 32 min |
| api 2026-11-06 (Haiku, 3 runs per cell) | 33 | $0.87 | 103 s / 147 s |

## Reference results to reproduce

**baseline** (2026-11-05): 28/28 pass; MCP took 1.6-1.9x more turns than CLI at equal cost; both s1 MCP runs
grepped/read output.xml.

**demoshop**: 16/16 pass.

| model | turns MCP | turns CLI | cost MCP | cost CLI |
| --- | --- | --- | --- | --- |
| Haiku | 279 | 166 | $0.82 | $0.63 |
| Sonnet | 175 | 67 | $2.05 | $1.36 |

Acceptance bar after the fixes: MCP turns <= 1.3x CLI, 0 output.xml reads, no run ends with rf_analyze errors in
files it wrote (report: "Totals by model and condition", `written_analyze_errors`).

**fixtures**, skill v1 -> v2 (Haiku, 2 runs per cell; tests / mean turns / mean failed sends):

| task | v1 | v2 |
| --- | --- | --- |
| Browser widgets | 14/16, 23, 3.0 | 16/16, 18, 0.5 |
| Selenium widgets | 15/16, 28, 4.0 | 16/16, 22, 1.5 |
| cart API | 8/8, 17 | 8/8, 14 |

**api** (2026-11-06, Haiku, 3 runs per cell): REST ledger and booker v1 = v2, all pass.

| task | turns v1/v2/v3 | failed sends v1/v2/v3 | bugs |
| --- | --- | --- | --- |
| SOAP | 30 / 24 / 21 | 3.0 / 0.3 / 0.0 | 9/9 |
| GraphQL | 24 / 21 / 19 | 1.0 / 0.7 / 0.3 | |

Isolation (`isolated_pass`): without the "every test creates its own data" rule SOAP 9/15 tests (0/3 runs fully
isolated), booker 14/18 (2/3); with the rule 15/15 (3/3) and 18/18 (3/3).

## Results after the restructure (2026-11-06, rf-mcp 2.0.0 from this branch)

**baseline** (28 runs, $1.79): 28/28 pass. Turns MCP/CLI: Haiku 94/81 (1.16x), Sonnet 78/56 (1.39x), down from
1.6-1.9x. output.xml: 1 MCP hit (Haiku s1: an unscoped `Grep` for `CalcLib` matched lines of `results/output.xml`),
2 CLI.

**demoshop** (16 runs, $4.15): MCP 8/8, CLI 7/8 (Sonnet T6 read output.xml, which the grader fails).

| model | turns MCP | turns CLI | ratio | cost MCP | cost CLI |
| --- | --- | --- | --- | --- | --- |
| Haiku | 173 | 117 | 1.48x (was 1.68x) | $0.56 | $0.22 |
| Sonnet | 128 | 67 | 1.91x (was 2.61x) | $2.06 | $1.32 |

Acceptance bar: 0 output.xml reads (met), no `rf_analyze` errors in written files (met, 0 in every run), turns
<= 1.3x CLI (not met). Most of the gap is structural: a shell agent reads several files with one `cat a; cat b`,
a tool-only agent needs one `Read` each plus a `ToolSearch` turn for the deferred rf-mcp tools; Haiku T1 also had
9 failed of 21 REPL sends.

**fixtures** (Haiku, 2 runs per cell, $0.46; tests v1 / v2 / shipped): Browser 15/16, 15/16, 14/16; Selenium 14/16,
15/16, 14/16; cart API 8/8 in every cell. The 2026-11-05 v1 -> v2 gain did not reproduce. Every failure is the
closed-shadow-root counter tagged `blocked`: the prompt says "Do not modify the page under test", and forcing the
root open is such a modification, so the outcome is a judgement call per run. Only 5 of 12 widget runs read any
rf-locators file, 1 read `references/shadow-dom-iframes.md`.

**api** (Haiku, 3 runs per cell, 48 runs, $1.50): every run passes, catches every injected bug (REST 6/6, SOAP
3/3, GraphQL 3/3 per run) and every test passes alone. Turns / failed sends, v1 / v2 / v3 / shipped: SOAP
25.3 / 26.7 / 20.0 / 21.7 and 1.3 / 1.0 / 0.7 / 0.0; GraphQL 23.0 / 20.3 / 20.0 / 22.0 and 1.3 / 0.3 / 0.0 / 0.3.
The isolation contrast (without vs with the 14.10 rule) can no longer be measured with `--skills-dir`: the rule is
also in the server instructions, which every condition gets, and all conditions now isolate 100%.

## After the turn reduction (tasks 15.1-15.3, 2026-11-06)

`rf_discover action="read"`, batched `rf_debug` sends, closed-root recipe in the rf-locators SKILL.md.

**demoshop** (16 runs, $4.02): 16/16 pass, 0 output.xml reads, 0 `rf_analyze` errors in written files.

| model | turns MCP | turns CLI | ratio | before | cost MCP | cost CLI |
| --- | --- | --- | --- | --- | --- | --- |
| Haiku | 162 | 103 | 1.57x | 1.48x | $0.60 | $0.23 |
| Sonnet | 111 | 67 | 1.66x | 1.91x | $1.90 | $1.29 |

Sonnet adopted both features (`action="read"` in 4/4 runs; T3 sent 8 multi-line debug batches, 44 -> 24 turns).
Haiku used `action="read"` in 1 of 4 runs and still spent 13-17 single-`Read` turns in T1-T3, so its ratio did
not move (one run per cell: +-10% is noise).

**fixtures** (18 runs, $0.48; tests v1 / v2 / shipped): Browser 14/16 in every cell, Selenium 15/16, 15/16, 14/16,
cart API 8/8. Again every failure is the closed-root counter tagged `blocked`, and only 1 of 12 widget runs read
any rf-locators file, so the recipe added to SKILL.md was not seen.

## After the routing changes (task 15.5, 2026-11-07)

The multi-file read is step 1 of the server-mode authoring loop, a routing row sends UI steps to
`skill://rf-locators/SKILL.md`, and the widget prompts say that JavaScript run by the test is allowed (the prompt
and the routing changed together, so the fixtures gain cannot be split between them).

**demoshop** (MCP only, 2 runs per cell, $4.24; CLI pooled from the two guarded 2026-11-06 sweeps, 8 runs per
model): 16/16 pass, 0 `rf_analyze` errors in written files, 1 output.xml read (a Haiku T3
run grepped the whole project and matched output.xml).

| model | turns/run MCP | turns/run CLI | ratio | 15.4 | 2026-11-06 |
| --- | --- | --- | --- | --- | --- |
| Haiku | 36.5 | 27.5 | 1.33x | 1.57x | 1.48x |
| Sonnet | 23.9 | 16.8 | 1.43x | 1.66x | 1.91x |

`action="read"` was used in 8/8 Sonnet runs and 6/8 Haiku runs; the two Haiku runs without it spent 11 and 14
single-`Read` turns.

**fixtures** (18 runs, $0.45; skills v1 / v2 / shipped): all 12 widget runs read `skill://rf-locators/SKILL.md`
(15.4: 1 of 12). Browser 15/16, 16/16, 16/16; Selenium 15/16, 16/16, 16/16; cart API 8/8. The two failures are
both v1 runs, the version without the closed-root recipe, which tagged the counter `blocked`; v2 and the shipped
skill forced the root open in every run.

## Codex CLI with small models (2026-11-07)

Codex CLI 0.166, GPT-5.6-Luna at its default window and capped at 32k (`@32k`), and GPT-6-Luna at 32k, reasoning
effort low, 1 run per cell per iteration. 119 baseline and 32 demoshop runs (the last demoshop sweep was
stopped after 5 of 8 runs for low memory; the other 3 were rerun). Codex reports no cost; tokens are input
tokens per run.

**What Codex does with an MCP server.** It loads MCP tools lazily behind its code-mode `exec` tool and shows the
model neither the server `instructions` nor the tool descriptions. With rf-mcp configured but no skills, Luna
made 0-1 rf-mcp calls per run and worked through the shell (`rg`, `type`, plain `robot`), with output.xml
reads and single-file selections. With `rf-mcp sync-skills --agent codex` the skills in `.agents/skills`
route the work to rf-mcp (9-14 rf-mcp calls per run) and the process flags disappear. Under
`approval_policy="never"` Codex rejected `rf_run`, `rf_repl` and `rf_debug` until
`default_tools_approval_mode = "approve"` was set.

| suite | setup | GPT-5.6-Luna | Luna @32k | GPT-6-Luna @32k |
| --- | --- | --- | --- | --- |
| baseline (7) | cli + robotcode skill | 12/14, 334k tok | 6/7, 257k | 1/7 (gave up after the sandbox denied pwsh) |
| baseline | rf-mcp, no skills | 12/14, 156k, 0.3 rf-mcp calls | 6/7, 137k | - |
| baseline | rf-mcp + synced skills, iterations 2-5 | 27/28, 345k | 27/28, 256k | 6/7, 303k |
| demoshop (4) | cli + robotcode skill | 1/4 (Chromium blocked by the sandbox) | 1/4 | - |
| demoshop | rf-mcp + synced skills, iterations 3-5 | 10/12 | 12/12 | - |

The t3 rows were regraded after a grader fix: three runs had repaired all four drifted locators, left both
shop defects failing and reported them by test id and amounts ("UI renders `$90.85`"), but failed
`reports_defects`, which looked only for the word "price". It now also accepts a test id named as a defect or bug.

On the 32k window rf-mcp + skills used fewer input tokens than at the default window (256k vs 345k per baseline
run) at the same pass rate. The CLI rows are not a clean comparison on Windows: the sandbox blocks the Store
`pwsh.exe` and Chromium, and rf-mcp runs outside it.

**Changes made from these runs** (each with a unit or contract test):

| | change | evidence |
| --- | --- | --- |
| A | a passed server-mode call no longer echoes `command` (failures keep it) | ~300 of ~340 characters of a one-line REPL send; 9% of all rf-mcp output |
| B | after 3 passed one-line REPL sends in a row, the response says to batch | Luna sent a 27-step web flow one line per call |
| C | `rf_results` accepts `path` / `result_file` / `output_file` for `output_path` | `path=` failed and the agent gave up reading the run |
| D | `rf-mcp init --agent codex` says to sync the skills and how to approve the tools | 0 rf-mcp calls without skills; rejected tools under `never` |
| E | a REPL send splits `.kw` / `.doc` / `.vars` lines into their own parts (like `rf_debug`, 15.2) | `.doc` + 4 `.kw` in one send failed ("Dot-commands can only be sent on their own"), then 11 one-line sends |
| F | REPL and debug output is capped at `max_output_chars`, a cut `.doc` points to `.kw` / `rf_doc` | one `.doc XML` returned 81k characters (~20k tokens, most of a 32k window) |
| G | `rf_results log` drops per-node timestamps, `isHtml: false`, NOT RUN branches and absolute sources | 12 failed-test logs: 156k -> 104k characters |
| H | the output cap also cuts one dominant long string (`rf_doc action=lib`) | `rf_doc` returned the 83k-character XML library markdown uncut |
| I | a test failing in 3 `rf_run` calls in a row gets "use rf_debug" | Luna edited and reran one test 14 times; after the nudge it switched to `rf_debug` in 1 of 3 cases |

Not measurable with one run per cell: the effect of A-I on turns or tokens (iteration-to-iteration spread is
larger than the change); the largest response fell from 81-83k to 21-30k characters, and the merged output of a
split REPL batch is now capped too (the 29k case after F).

### Local models through Codex (2026-11-07, partial)

Ollama 0.40.15 with `OLLAMA_CONTEXT_LENGTH=32768`, rf-mcp + synced skills, s1 only: the sweep of 21 runs was
stopped by Claude Code's low-memory guard after 3. With one run at a time, the job order still loads a different
model for every run and Ollama keeps each loaded for 5 minutes, so qwen3:8b, qwen3:14b and gpt-oss:20b were
resident together (~27 GB on a 16 GB GPU). Set `OLLAMA_MAX_LOADED_MODELS=1`, or run one model per sweep.

| model | s1 | what happened |
| --- | --- | --- |
| qwen3:8b | fail, 3 calls | searched for a file named after the test, got nothing, then asked the user what to do |
| qwen3:14b | pass, 16 calls | shell only: found the bug by listing and reading files, fixed it, never ran the test |
| gpt-oss:20b | fail, 11 calls | `rf_run` with the bare test name found the failure and the project hint; then failed to write the fix with PowerShell `Set-Content` quoting and only printed it |

A second sweep, one model at a time with `OLLAMA_MAX_LOADED_MODELS=1`, was also stopped by the memory guard,
after 5 qwen3:14b runs: at a 32k window qwen3:14b takes 14.4 GB, and `llama-server` held 11 GB of system RAM.

| qwen3:14b | result | calls | rf-mcp calls |
| --- | --- | --- | --- |
| s1 triage+fix | timeout (1200 s) | 66 | 22: after 18 identical `Get-ChildItem *test_*.py`, 10 `rf_discover`/`rf_doc` and 10 one-line `rf_repl` sends that never ran the test |
| s2 smoke by tag | pass | 10 | 0: typed `rf_run(include_tags=["smoke"])` into PowerShell, then ran plain `robot --include smoke` |
| s3 analyze | fail | 3 | 0: `robot --dryrun` on one file |
| s4 XML authoring | fail | 16 | 0: 14 attempts to write an XML file through PowerShell quoting |
| s5 API authoring | fail | 8 | 0: could not write the suite through PowerShell |

Two Codex-side causes, both from the model metadata. Codex has no metadata for Ollama models ("Model metadata for
`qwen3:8b` not found. Defaulting to fallback metadata"), so it offers no file-editing tool: every file is
written through PowerShell quoting, which the local models failed at in 3 of 5 runs. And the cached OpenAI
models are `tool_mode: code_mode_only` with `include_skills_usage_instructions: false`, which is why GPT-5.6-Luna
saw rf-mcp only behind `exec`. Since 2026-11-07 the harness writes a `model_catalog_json` entry for `cx-oss-*`
models: `tool_mode: direct`, skills usage instructions on, the real window, and no `apply_patch`. Codex only
offers `apply_patch` as a freeform tool, and through Ollama the call failed with "tool apply_patch invoked with
incompatible payload". The runs above predate that catalog.

## Small models on OpenRouter (2026-11-09)

Baseline suite (7 scenarios), rf-mcp + synced skills, 1 run per cell, models from the cheap, tool-capable end of
the OpenRouter rankings, each through Claude Code (`or-`) and Codex (`cx-or-`, effort low, 128k window).
About 130 runs including reruns and smoke tests cost $3.85 in OpenRouter usage. The `$` column is list price
times tokens without cache discounts, so it overstates the charge; it is there to compare models.

| model ($/M in, out) | Claude Code | Codex | rf-mcp calls/run (CC / Codex) | $ list (CC / Codex) |
| --- | --- | --- | --- | --- |
| z-ai/glm-5.3-flash (0.03, 0.08) | 7/7 | 7/7 | 16 / 9 | 0.11 / 0.08 |
| deepseek/deepseek-v4.1-flash (0.05, 0.57) | 7/7 | 7/7 | 14 / 11 | 0.20 / 0.24 |
| openai/gpt-6-luna (0.10, 0.50) | 6/7, s3 7/7 after fix 1 | 5/7 | 23 / 10 | 0.54 / 0.18 |
| qwen/qwen3-coder-30b-a3b-instruct (0.15, 0.60) | 2/7 (+s3 after fix 1) | 4/7 | 10 / 11 | 0.67 / 0.80 |
| mistralai/ministral-8b-2512 (0.15, 0.15) | 2/7 (+s3 after fix 1) | 3/7, 2 timeouts | 25 / 40 | 1.20 / 1.39 |
| openai/gpt-oss-20b (0.02, 0.09) | 0/7, retry 2/7 | 2/7 | 1-3 / 1 | 0.04 / 0.02 |

What separated the models:

- GLM-5.3-Flash and DeepSeek-V4.1-Flash passed everything through both harnesses, at 9-16 rf-mcp calls per run;
  GLM was the cheapest model in the sweep.
- The authoring scenarios (s4-s6) split the rest: qwen3-coder, GPT-6-Luna and ministral wrote suites that fail and
  then reported success ("I have successfully created a complete Robot Framework test suite"). GPT-6-Luna
  (Codex, s5) asserted the first name of a public restful-booker booking that other users change; its own run
  passed, the grader's run read another name.
- Claude Code on OpenRouter breaks some models at the API translation: qwen3-coder printed its first tool call as
  text (`<function=mcp__rf-mcp...`) and ended the run, gpt-oss-20b leaked its harmony format
  (`<|start|>assistant<|channel|>commentary to=functions.Read`) or got "upstream returned 502" / "The model's
  tool call could not be parsed", and ministral-8b printed `mcp__rf-mcp__list_resources{}` as its answer and
  once ended with "automatic compaction failed: summarization produced empty response". Through Codex (Responses
  API) gpt-oss-20b called tools but stopped after 1-4 calls, often without an answer.
- Models behind Claude Code send every optional argument, unset ones as `""` or `{}`.

**Changes made from these runs:**

| | change | evidence |
| --- | --- | --- |
| 1 | `rf_analyze severity` reports that severity and worse (`info` -> `--severity error,warning,info`) | robotcode's `--severity` reports only the listed severities: `severity="info"` returned 0 diagnostics for a file with 3 errors, GPT-6-Luna reported "no broken keyword references" (s3 fail), GLM needed 15 calls to find them by reading. After the fix s3 passed for 5 of 6 models through Claude Code (GLM in 4 calls) |
| 2 | `""` for `target`, `shell`, `stop_scope` means unset | `rf_analyze(shell="")` failed "Input should be 'powershell' or 'bash'" |
| 3 | Codex project trust is passed as one inline table | Codex splits `projects.'<path>'.trust_level` at dots inside the quoted path; run ids with `.` left the project untrusted ("ignoring 2 unrecognized configuration settings"). Untrusted, gpt-oss-20b made 0 rf-mcp calls (0/7); trusted 2/7 |

The lab env runs rf-mcp from this repo (editable), so Codex runs started after 08:48 already had changes 1-2.
The `or-untrusted-*` Codex runs (before change 3) are kept for comparison: GLM 6/7, DeepSeek 7/7, gpt-oss 0/7.

### Repeats with the run verdict (2026-11-09)

Two more runs per cell for the mid-range models (`or-reps-*`), with changes 1-2, the `rf_run` verdict (change 4) and
the s5 prompt that asserts only data the test created (change 5). "old r1" is the single run above, before
changes 1, 4 and 5. P pass, f fail, T timeout; `unverified` counts failed runs that wrote a suite and had no passing
run after the last write (change 6, regraded for the old runs).

| harness | model | code | s1 | s2 | s3 | s4 | s5 | s6 | s7 | pass | unverified fails |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Claude Code | gpt-6-luna | old r1 | P | P | f | P | P | P | P | 6/7 | 0 |
| Claude Code | gpt-6-luna | new r1+r2 | PP | PP | PP | PP | PP | PP | PP | 14/14 | 0 |
| Codex | gpt-6-luna | old r1 | P | P | P | P | f | f | P | 5/7 | 0 |
| Codex | gpt-6-luna | new r1+r2 | PP | PP | PP | PP | PP | fP | PP | 13/14 | 0 |
| Claude Code | qwen3-coder-30b | old r1 | f | P | f | f | P | f | f | 2/7 | 2 |
| Claude Code | qwen3-coder-30b | new r1+r2 | PP | PP | PP | ff | ff | ff | fP | 7/14 | 6 |
| Codex | qwen3-coder-30b | old r1 | P | P | P | f | f | f | P | 4/7 | 3 |
| Codex | qwen3-coder-30b | new r1+r2 | Pf | PP | PP | ff | ff | ff | PP | 7/14 | 6 |
| Claude Code | ministral-8b | old r1 | P | P | f | f | f | f | f | 2/7 | 1 |
| Claude Code | ministral-8b | new r1+r2 | ff | ff | PP | TT | ff | ff | fP | 3/14 | 3 |
| Codex | ministral-8b | old r1 | f | P | P | T | f | T | P | 3/7 | 3 |
| Codex | ministral-8b | new r1+r2 | PP | PP | PP | TT | TT | TT | PP | 8/14 | 6 |

The new s5 prompt passed for GLM and DeepSeek through both harnesses (4/4, `or-s5v2-*`).

- GPT-6-Luna is reliable once s3 (change 1) and s5 (change 5) stop failing for reasons outside the model: 27/28.
  The one miss is a Codex s6 run whose own run passed and the grader's rerun failed.
- Qwen3-Coder passes the triage, smoke, analysis and debug scenarios and fails every authoring run (0/12). All 12
  are `unverified`: in 5 it never called `rf_run`, in 2 every `rf_run` failed validation (`files=""`,
  `output=[]`), and in 5 it saw `FAIL: ...` or `PASS (dry run: ...)`. It still answered "I have successfully
  created ..." in most of them. The verdict line does not change this model's answer.
  Part of the failure was argument shape: `by_longname="['Tests.Web.Saucedemo.Login']"` (a Python list in a
  string) became one `-bl` with brackets, and `rerunfailed="[]"` became `--rerunfailed []` (change 7).
- Ministral-8B through Claude Code mostly ends on OpenRouter API errors; through Codex it passes the
  non-authoring scenarios and times out (20 min) on every authoring run after 270-450 tool calls.
- Key usage for this round was $10.65. It includes duplicate runs: a stopped sweep loop kept going and ran the
  luna, qwen and ministral sweeps a second time into the same folders, and the later runs overwrote the results.
  Per-model key-usage deltas in a sweep log are wrong whenever two sweeps share the key at the same time.

| | change | evidence |
| --- | --- | --- |
| 4 | `rf_run` responses start with `outcome`: `PASS: 3 tests, 3 passed, 0 failed`, `PASS (dry run: ...)`, or `FAIL: ...` with "fix them and rerun before you report the task as done"; a clean `rf_analyze` adds a `note` that it ran nothing | Qwen3-Coder answered "successfully created" after `exit_code: 1` and a `failures` list that came after the echoed argv |
| 5 | s5 creates a booking, reads it back by id and asserts only data the test created | the old prompt "Read a booking" let GPT-6-Luna assert public booking 2 ("Eric Smith"); other users had changed it before the grader's run |
| 6 | process flag `unverified`: a `.robot`/`.resource` file was written and no run passed after the last write (dry runs and `files` on `rf_analyze` do not count) | flags 12/12 Qwen authoring failures; the final text is no reliable signal |
| 7 | list arguments accept Python list text (`"['A.A']"`); an empty value (`""`, `"[]"`, `"{}"`, `"null"`, `[]`, `{}`) means unset for every argument (replaces change 2) | Qwen: 5 `by_longname` as Python list text, `rerunfailed="[]"` 3 times, `files=""` and `output=[]` failing validation; DeepSeek: `profile`, `root`, `output`, `rerunfailed` as `"[]"`. Made after these runs, not yet measured |

## Not ported

- The experiment-only "guided" server (`guided_server*.py`): rf-mcp now carries those tool descriptions; skill
  variants go through `--skills-dir` instead.
- Hard-coded lab paths, fixed ports and the hand-started web server: interpreters come from env vars, fixtures start
  on free ports.
- One-off analysis scripts (table printers, fail miners): replaced by `report.py`, the `process` block of each result
  and the `trim` command.

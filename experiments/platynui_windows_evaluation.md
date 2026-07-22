# PlatynUI on Windows 11 — rf-mcp Autonomous Desktop Automation Evaluation

**Purpose.** Gather evidence of how well an autonomous LLM agent can drive **popular, always-installed Windows applications** (Calculator, Notepad, WordPad, Paint, File Explorer, Settings, and MS Office Word/Excel) through **rf-mcp + PlatynUI (new Rust core)** — and surface the rf-mcp/PlatynUI issues that get in the way.

**Two goals frame every scenario and metric:**
1. **Agent autonomy** — the agent completes the task with **few failed tool calls** and **little user input** (ideally zero clarifying questions), using the built-in desktop guidance rather than trial-and-error.
2. **Low latency** — a **wrongly chosen keyword or locator must fail FAST** (a small wait), not hang for seconds; total wall-clock stays low.

**Build under test.** Windows 11, Python 3.12+, `rf-mcp >= 0.34.0.dev3` installed with the desktop extra (`uv pip install --pre "rf-mcp[desktop]"` → PlatynUI `0.13.0.dev2`). This build already contains the Windows dry-run stdin-deadlock fix and the generated-suite Windows-path escaping fix.

**How to run each scenario (to measure *true* autonomy):**
- Give the agent **only** the scenario's *"Agent task"* prompt — nothing else (no locators, no keyword hints).
- Capture the full **MCP tool-call transcript** (every call, its arguments, its result/error, and per-call timing). The rf-mcp CLI/stream-json or your MCP host's log is the source for all metrics.
- Record the metrics defined in **§1 (Metrics Framework)** into the per-scenario results block.
- Compare the agent's **actual** tool+keyword flow to the scenario's **"Expected flow"** (that is the reference, not a script the agent must follow).

**⚠ Accuracy / verification note.** Concrete locators, `@AutomationId`s and control `@Name`s below are the authors' best Windows-accurate guesses; any marked **⚠ ASSUMED** (and any locale-dependent `@Name`, English-UI assumed unless flagged) **must be verified on the box** — the fastest way is to dump the real accessibility tree with `get_session_state(session_id, sections=["ui_tree"], elements_of_interest=["<app>"])` and correct the locator. A locator that turns out wrong is itself **evidence** (record it under the scenario's "Suspected issue").

---

## Contents
1. **Setup & Evaluation Framework** — prerequisites, the exact metrics to gather, the per-scenario results template, scoring rubric, and aggregate roll-up.
2. **Standard always-installed apps** — Calculator, Notepad, WordPad, Paint, File Explorer, Settings, Task Manager, Snipping Tool (+ deliberate failure-mode probes).
3. **Microsoft Word** — launch, type, format, find, save, tables (+ read-back).
4. **Microsoft Excel** — cells, navigation, formulas, ranges, read-back, save.
5. **Cross-cutting capability matrix + Windows risk register + fail-fast probes.**

---

# 1. Setup & Evaluation Framework

This section defines the fixed harness the human operator uses to run every scenario and record results. It is app-agnostic: each scenario elsewhere in this document supplies its own `Agent task` prompt and expected flow; this section supplies the machine setup, the exact metrics, the fill-in template, the scoring rubric, and the roll-up. Run each scenario **once cold** (fresh session, no prior context) and record against the template below. Everything here is extractable from the tool-call transcript plus one in-app verification read/screenshot.

---

### 1. Setup & Prerequisites

#### 1.1 Machine & OS
- **Native Windows 11** (23H2 or 24H2), signed in at a **local interactive console** — NOT over RDP and NOT on a locked/screensaver session. RDP and lock-screen suspend UIAutomation focus and pointer synthesis, which will masquerade as PlatynUI "hangs"/failures and pollute latency numbers.
- If a VM is used, keep it **foregrounded with a real desktop resolution** (≥1280×800) and disable "pause when minimized". Record VM-vs-bare-metal in the run header — it materially affects the fail-fast and pointer numbers.
- **UI language = English (United States)** for the baseline runs. Many Windows `@Name`/`@AutomationId` values are locale-dependent (see the flags in §1.4). Record the actual display language in the run header; if non-English, every `@Name='...'` assumption is void.
- **Display scaling** noted (100% vs 150%) — pointer `x,y` fallbacks and `Get Element At Point` depend on it.

#### 1.2 Install rf-mcp[desktop]
```powershell
# Python 3.12+ required
py -3.12 --version
uv --version

# Clean venv, install the pre-release desktop bundle
uv venv --python 3.12
uv pip install --pre "rf-mcp[desktop]"

# Confirm the exact build under test (record both in the run header)
uv pip show rf-mcp            # expect >= 0.34.0.dev3
uv pip show PlatynUI          # expect 0.13.0.dev2
# Native + CLI PlatynUI wheels (pin what actually resolved)
uv pip show platynui-native platynui-cli
```
- **Record the exact resolved versions** of `rf-mcp`, `PlatynUI`, `platynui-native`, `platynui-cli`, `robotframework`, and Python in the run header. The two recent Windows fixes under test (dry-run no-hang; `C:\...`→forward-slash path rewrite in generated `.robot`) are version-gated on `rf-mcp >= 0.34.0.dev3` — a silently older resolve invalidates the Quality metrics.
- Smoke-test the server starts and libdoc loads without downloading models:
```powershell
$env:HF_HUB_OFFLINE = "1"       # semantic keyword embeddings are lazy/opt-in; keep off for eval
uv run python -m robotmcp.server --help   # or the client-launched stdio command in §1.3
```

#### 1.3 MCP client config (agent-under-test)
The agent-under-test drives rf-mcp over **stdio**. Minimal `.mcp.json`:
```json
{
  "mcpServers": {
    "robotmcp": {
      "command": "uv",
      "args": ["run", "python", "-m", "robotmcp.server"],
      "cwd": "C:/eval/rf-mcp",
      "env": { "HF_HUB_OFFLINE": "1" }
    }
  }
}
```
- Give the agent access to the rf-mcp tool set **only** (`analyze_scenario`, `manage_session`, `execute_step`, `execute_batch`, `get_session_state`, `build_test_suite`, `run_test_suite`, plus the read helpers). Do **not** expose a shell, a browser, or file tools — those let the agent route around PlatynUI and corrupt the autonomy signal.
- Record the **agent model + client** (e.g. Claude Code / MiniMax-M3 / etc.) in the run header; autonomy is a joint property of the model and rf-mcp.

#### 1.4 Apps to install / confirm
Confirm each target app launches interactively **before** the run. Process names are what `Process.Start Process` should launch and what appears as `/app:*[@Name='<here>']`. **Every `@Name`/`@AutomationId` below marked `⚠ASSUMED` must be verified on the actual box** (query the live tree with `get_session_state(sections=["ui_tree"])` once and confirm) — Windows control names drift by build and locale.

| App | Launch (process) | Kind | Window locator | Key control locators | Flags |
|-----|------------------|------|----------------|----------------------|-------|
| Calculator | `calc.exe` (spawns `CalculatorApp`) | UWP under **ApplicationFrameHost** | `/app:*[@Name='ApplicationFrameHost']/control:Window[@Name='Calculator']` | `//control:Button[@AutomationId='num7Button']`, result `//control:Text[@AutomationId='CalculatorResults']` | `num7Button`/`CalculatorResults` are real but **locale-tied** (`CalculatorResults.Name` = "Display is 7") ⚠verify |
| Notepad | `notepad.exe` | Classic-hosted, own top-level window (modern Notepad is packaged but **not** under ApplicationFrameHost) | `/app:*[@Name='Notepad']//control:Window` | edit area `//control:Document` or `//control:Edit` | Modern Notepad edit AutomationId (`RichEditBox`/`15`) ⚠ASSUMED — verify Document vs Edit on the box |
| Paint | `mspaint.exe` | Packaged, own window | `/app:*[@Name='Paint']//control:Window` (or `@Name='mspaint'`) ⚠ASSUMED | canvas is often a bare `control:Pane`/`native:` — flag if unqueryable | `@Name` of window ⚠ASSUMED |
| Settings | `start ms-settings:` (spawns `SystemSettings`) | UWP under **ApplicationFrameHost** | `/app:*[@Name='ApplicationFrameHost']/control:Window[@Name='Settings']` | nav list `//control:List`, items `//control:ListItem[@Name='Bluetooth & devices']` ⚠locale | search box AutomationId ⚠ASSUMED |
| File Explorer | `explorer.exe` | Shell, own window | `/app:*[@Name='...']//control:Window` (process attribution to Explorer is fiddly) ⚠verify | address bar, items grid `//control:DataItem` | window @Name is the current folder — **not stable**; scope by AutomationId |
| WordPad | — | **REMOVED in Windows 11 24H2** | — | — | Do **not** schedule WordPad on 24H2; substitute another classic Win32 app and note it |

- **Confirm every scheduled scenario's app actually exists on the build under test** before starting; a missing app produces a fake "autonomy failure".

#### 1.5 Transcript & timing capture (the measurement source of truth)
All metrics derive from a **timestamped tool-call transcript**. Capture the agent client's structured stream and prepend a wall-clock timestamp to every line so request→response deltas are recoverable:
```powershell
# Example for a stream-json capable agent client; adapt the launch command to your agent.
& claude -p "<AGENT TASK PROMPT ONLY>" `
    --output-format stream-json --verbose `
    --mcp-config .mcp.json `
  | ForEach-Object { "{0:o}`t{1}" -f (Get-Date).ToUniversalTime(), $_ } `
  | Tee-Object -FilePath ".\runs\<SCENARIO_ID>.tsv"
```
- Each line is `<ISO-8601-UTC>\t<json-event>`. A **tool call's duration** = `timestamp(tool_result)` − `timestamp(tool_use)` for the matching `id`. This yields per-call duration, the slowest call, the fail-fast number, and the >5s hang flags **directly**, with no server instrumentation.
- **Cross-check** against server-side timing: many rf-mcp responses echo status/duration in the payload — record it when present, but the timestamped client stream is authoritative for wall-clock (it includes MCP transport + the agent's own turn latency you care about).
- Keep the **full JSON of every `tool_result`** — the `success`/`status`/`error` fields and any refusal message ("unscoped locator refused", "element not found", timeout) are the source for the failure taxonomy (§2.4).
- Save the generated `.robot` file and the `run_test_suite` (dry + full) output alongside the transcript for the Quality metrics.

#### 1.6 Agent-input discipline (measures true autonomy)
- The agent-under-test is given **only the scenario's verbatim `Agent task` prompt** — no locators, no keyword names, no hints, no the desktop_guidance echoed by hand. All desktop guidance must come to the agent through rf-mcp itself (the `manage_session(action="init")` response's `desktop_guidance`).
- **Do not answer clarifying questions.** If the agent asks one, log it (it counts against autonomy) and reply with a fixed neutral nudge (`"Proceed with your best judgment."`) — never new information. Repeated questions each count.
- One human intervention of any substance (supplying a locator, correcting a keyword, restarting the app) ends the "unaided" status for that scenario; log it and continue only if you want a `PARTIAL` observation.

---

### 2. Per-Scenario Metrics (precise definitions)

A **tool call** = one MCP tool invocation by the agent (`tool_use` block). A call is **FAILED** if any of: the tool errored/threw at the MCP layer; the `tool_result` payload has `success=false` / `status="FAIL"` / a non-empty `error`; an `execute_step`/`execute_batch` keyword ran with RF status FAIL; a `Query`/`Wait Until Exists` returned no element when one was required; or rf-mcp **refused** the call (e.g. unscoped `//` locator). Assertion mismatches from `Get Attribute`/`Should Be *` are FAILED calls. `analyze_scenario`/`manage_session` init that returns normally is a SUCCESS even if the agent later misuses it.

#### 2.1 Autonomy metrics
| Metric | Definition | Extract from transcript | Unit |
|--------|------------|-------------------------|------|
| Total tool calls | Count of all `tool_use` events for the scenario | count `tool_use` | int |
| Failed tool calls (count) | Calls meeting the FAILED definition above | count FAILED | int |
| Failed tool calls (%) | Failed ÷ total × 100 | derived | % |
| Distinct error types | Number of **unique** categories from the §2.4 taxonomy that appear | classify each FAILED result, dedupe | int |
| Self-recovery retries | Count of times the agent, after a FAILED call, issues a **corrected** call targeting the same sub-goal **without human input** and it succeeds or advances | walk the sequence per sub-goal | int |
| Repeated-loop failures | Max number of consecutive FAILED calls with the **same** error category on the same sub-goal (a stuck-loop signal) | longest same-category run | int |
| Clarifying questions / user inputs | Count of assistant turns that ask the operator anything or stall awaiting input | count question turns + any human intervention | int |
| Completed unaided | Task reached success with **zero** substantive human inputs | yes/no | bool |
| Turns-to-completion | Number of agent (assistant) turns from first action to verified completion | count assistant turns | int |

#### 2.2 Latency metrics
| Metric | Definition | Extract from transcript | Unit |
|--------|------------|-------------------------|------|
| Total wall-clock | `timestamp(last relevant tool_result)` − `timestamp(first tool_use)` | span | s |
| End-to-end session time | First prompt receipt → verified completion (includes agent thinking) | outer span | s |
| Per-call duration | Per matched `tool_use`→`tool_result` delta (record the vector) | per-call deltas | s |
| Slowest call | Max per-call duration + which tool + keyword/locator | argmax | s + label |
| Time on failed calls | Sum of durations of all FAILED calls | sum over FAILED | s |
| **Fail-fast number** | For each FAILED call whose cause is a **wrong/unscoped/not-found locator**, its duration. Report **min / median / max**, and split: **(a) pre-dispatch refusal** (unscoped `//` refused by rf-mcp — should be ~0, <100ms) vs **(b) query-timeout** (bad-but-scoped locator waited to timeout) | filter FAILED by category ∈ {unscoped-refused, element-not-found, wait-timeout} | s |
| Calls > 5s (hangs) | Count of any single call exceeding 5s | count | int |
| True hangs > 15s | Count exceeding 15s (near the old 180s pathology) | count | int |

#### 2.3 Quality metrics
| Metric | Definition | Evidence | Unit |
|--------|------------|----------|------|
| Task accomplished in-app | The app's real state matches the goal, verified by a **read-back** (`Get Attribute` on the scoped node) **or** a `Take Screenshot` inspected by the operator | screenshot file + read-back value | yes/no |
| Read-back matches expected | The verifying `Get Attribute`/assertion equals the expected value | payload value vs expected | yes/no |
| Suite built | `build_test_suite` produced a `.robot` at `output_path` | file exists | yes/no |
| Dry-run passed | `run_test_suite(mode="dry")` returned success (and did **not** hang — the 180s regression fix) | run output + duration | yes/no + s |
| Full-run passed | `run_test_suite(mode="full")` PASS | run output | yes/no |
| `.robot` path integrity | Windows drive-letter paths appear as forward-slash (`C:/...`), not corrupted `C:\...`; no broken escapes | grep the file | yes/no |
| `.robot` correctness | All locators are **app-scoped** (`/app:*[@Name=...]//`), no leading `//`, keyword names valid, Set Root used | inspect file | yes/no |

#### 2.4 Failure taxonomy (distinct-error-types buckets)
Classify every FAILED `tool_result` into exactly one:
- `unscoped-locator-refused` — leading `//` or over-broad locator refused pre-dispatch (expect ~0s; good fail-fast).
- `element-not-found` — scoped `Query`/`Wait Until Exists` returned nothing.
- `stale-tree` — app launched **after** the first PlatynUI keyword; window invisible until re-query/cache-clear.
- `wait-timeout` — `Wait Until Exists`/`Wait Until Query` hit its timeout (candidate hang).
- `wrong-window-control` — used `control:Frame` (Linux) instead of `control:Window` (Windows), or missed ApplicationFrameHost hosting for a UWP app.
- `keyword-not-found` / `wrong-keyword` — keyword name invalid or non-desktop keyword chosen.
- `argument-shape` — bad args (e.g. `execute_batch` steps malformed, dict-keys-instead-of-values).
- `session/library` — session not init, missing `PlatynUI.BareMetal`/`Process`.
- `assertion-mismatch` — `Get Attribute`/`Should Be *` failed the comparison.
- `locale-mismatch` — `@Name` didn't match because UI language ≠ English.
- `other` — record verbatim.

---

### 3. Per-Scenario Results Template (copy one block per run)

```yaml
# ─── RUN HEADER ───
scenario_id:            # e.g. WIN-CALC-01
scenario_title:
date_utc:
operator:
agent_model_client:     # e.g. Claude-Code + Opus / MiniMax-M3
os_build:               # Win11 23H2 / 24H2, VM or bare-metal
ui_language:            # en-US ?
display_scaling:        # 100% / 150%
rf_mcp_version:         # must be >= 0.34.0.dev3
platynui_version:       # 0.13.0.dev2
platynui_native_cli:
python_version:
transcript_file:        # runs/<id>.tsv
robot_file:             # path to generated .robot

# ─── AUTONOMY ───
total_tool_calls:
failed_tool_calls:
failed_pct:
distinct_error_types:        # count
error_categories_seen:       # list from §2.4
self_recovery_retries:
repeated_loop_max:           # longest same-error run
clarifying_questions:
human_interventions:
completed_unaided:           # yes/no
turns_to_completion:

# ─── LATENCY (seconds) ───
total_wall_clock:
end_to_end_session:
slowest_call:                # e.g. execute_step Query 4.2s @ /app:*[@Name='Calculator']//...
slowest_call_seconds:
time_on_failed_calls:
fail_fast_min:
fail_fast_median:
fail_fast_max:
fail_fast_unscoped_refusal:  # pre-dispatch refusal latency, expect <0.1
fail_fast_query_timeout:     # scoped-but-wrong locator latency
calls_over_5s:               # count  (each is a HANG — list them)
true_hangs_over_15s:         # count
hang_details:                # list: tool/keyword/locator/seconds

# ─── QUALITY ───
task_accomplished_in_app:    # yes/no
verification_evidence:       # screenshot path and/or read-back value
read_back_matches_expected:  # yes/no
suite_built:                 # yes/no
dry_run_passed:              # yes/no  (seconds: __, must not hang)
full_run_passed:             # yes/no
robot_path_integrity:        # yes/no  (C:/ not C:\)
robot_correctness:           # yes/no  (scoped, no //, Set Root, valid kw)

# ─── SCORES (see §4 rubric) ───
autonomy_score:              # PASS / PARTIAL / FAIL
latency_score:               # PASS / PARTIAL / FAIL
quality_score:               # PASS / PARTIAL / FAIL
scenario_verdict:            # weakest of the three

# ─── NOTES ───
suspected_rf_mcp_issue:      # tie to the scenario's "Suspected issue to watch"
assumptions_verified:        # which ⚠ASSUMED @Name/@AutomationId you confirmed on the box
freeform:
```

Optional at-a-glance table to paste beside the block:

| ID | Unaided | Tool calls | Failed % | Wall-clock | Slowest | Hangs>5s | Fail-fast (unscoped/timeout) | Quality | Verdict |
|----|---------|-----------|----------|-----------|---------|----------|------------------------------|---------|---------|
|    |         |           |          |           |         |          |                              |         |         |

---

### 4. What to send back + scoring rubric

**What to send back per scenario (attach, don't summarize):**
1. The timestamped transcript `.tsv` (full — includes every `tool_use`/`tool_result` JSON).
2. The generated `.robot` file + the `run_test_suite` dry and full outputs.
3. The verification screenshot(s) and/or the read-back value proving in-app state.
4. The filled §3 template block.
5. A one-line note per `⚠ASSUMED` locator: confirmed / corrected-to `<actual>`.
6. Any transcript excerpt where the agent looped, asked a question, or a call exceeded 5s.

**Scoring rubric (each axis PASS / PARTIAL / FAIL):**

- **Autonomy**
  - `PASS` — completed unaided **and** failed calls ≤ 1 **and** 0 clarifying questions **and** no same-error loop ≥ 3.
  - `PARTIAL` — completed unaided but 2–4 failed calls, **or** exactly 1 clarifying question, **or** one 2-deep self-recovery loop.
  - `FAIL` — not completed unaided, **or** > 4 failed calls, **or** any same-error loop ≥ 3, **or** ≥ 2 human interventions/questions.

- **Latency**
  - `PASS` — no single call > 5s **and** every wrong/unscoped-locator failure < 2s (unscoped refusal effectively instant, <0.1s) **and** total wall-clock within the scenario's stated budget.
  - `PARTIAL` — exactly one call in 5–15s, **or** a wrong-locator failure in 2–5s.
  - `FAIL` — any call > 15s (true hang), **or** wrong-locator failure ≥ 5s, **or** > 1 call > 5s.

- **Quality**
  - `PASS` — task verified in-app (screenshot/read-back) **and** build → dry → full all green **and** `.robot` scoped + path-clean.
  - `PARTIAL` — task done in-app but a suite step failed or the `.robot` needed a manual fix (record which).
  - `FAIL` — task not verifiably accomplished in the app.

**Scenario verdict** = the **weakest** of the three axes (an autonomy `FAIL` sinks the scenario even if it was fast). Record all three plus the verdict.

---

### 5. Aggregate roll-up (per-app and overall)

Fill after all scenarios; compute per-app and again across the whole suite. Means over completed scenarios; rates over all attempted.

```yaml
scope:                        # <app-name> | OVERALL
scenarios_attempted:
scenarios_with_target_app_present:   # exclude missing apps (e.g. WordPad on 24H2)

# Autonomy roll-up
unaided_completion_rate_pct:
autonomy_pass_rate_pct:
mean_total_tool_calls:
mean_failed_call_pct:
mean_self_recovery_retries:
mean_clarifying_questions:
scenarios_with_any_loop_ge3:

# Latency roll-up
mean_wall_clock_s:
median_slowest_call_s:
count_hangs_over_5s:          # total across scenarios
count_true_hangs_over_15s:
median_fail_fast_unscoped_s:  # should trend ~0 (pre-dispatch refusal working)
median_fail_fast_query_timeout_s:

# Quality roll-up
quality_pass_rate_pct:
suite_build_dry_full_pass_rate_pct:
robot_path_corruption_incidents:     # expect 0 given the Windows fix
dry_run_hang_incidents:              # expect 0 given the 180s fix

# Headline verdicts
overall_scenario_pass_rate_pct:
top_autonomy_failure_categories:     # ranked from §2.4 tallies
top_latency_offenders:               # tool/keyword/locator with worst durations
```

**Regression watch (call out explicitly in the overall roll-up):** `dry_run_hang_incidents` and `robot_path_corruption_incidents` must both be **0** — non-zero means one of the two shipped Windows fixes (dry-run stdin isolation; `C:\`→`/` rewrite) regressed on this build. `median_fail_fast_unscoped_s` near 0 confirms the unscoped-`//` refusal guard is firing (the primary anti-hang defense); a rising `median_fail_fast_query_timeout_s` or any `true_hangs_over_15s` points at a scoped-but-wrong-locator waiting to the PlatynUI query timeout, and is the single most important latency defect to file.

---

# 2. Standard Always-Installed Windows 11 Apps

These scenarios exercise the apps present on a stock Windows 11 image, chosen to isolate one PlatynUI capability at a time and to surface the two failure classes the evaluation cares about: **autonomy** (failed/looping tool calls, clarifying questions) and **latency** (a wrong keyword/locator that hangs instead of failing fast). Every scenario assumes the shared setup below and the app-scope discipline from the primitives brief.

**Shared setup (run once per scenario, do not repeat in the flow steps):**
1. `analyze_scenario(scenario="<task>", context="desktop")` → `session_id`.
2. `manage_session(action="init", session_id, libraries=["PlatynUI.BareMetal","Process","BuiltIn"])` → read `desktop_guidance` from the response; do **not** call `find_keywords`.
3. Launch with `Process.Start Process` **first**, then `Query` the app window **before** any other PlatynUI keyword (first-keyword desktop-tree snapshot rule), then `Set Root` once.

**Cross-cutting Windows trap (applies to almost every launch below):** `calc.exe`, `notepad.exe`, `snippingtool.exe`, `mspaint.exe` and the `ms-settings:` protocol are **app-execution-alias / launcher stubs**. The process you `Start Process` exits almost immediately and its PID is **not** the app; the real window appears under a *different* process (often `ApplicationFrameHost` or the packaged process). Agents that block on the started PID, or reuse its handle, will stall — always re-`Query` for the window by `@Name`.

---

### STD-01 — Calculator: launch and find window (UWP host resolution)
- App: Calculator (`calc.exe` → `CalculatorApp.exe`, always-installed UWP) | Kind: UWP under `ApplicationFrameHost`
- Capability probed: launch + window-find, host-container resolution
- Agent task (verbatim): "Open the Windows Calculator and confirm its window is on screen."
- Expected rf-mcp/PlatynUI flow:
  1. `execute_step(keyword="Start Process", arguments=["calc.exe"])`
  2. `execute_step(keyword="Wait Until Exists", arguments=["/app:*[@Name='ApplicationFrameHost']/control:Window[@Name='Calculator']"])` — the stub PID has already exited; the window is under the frame host.
  3. `execute_step(keyword="Get Attribute", arguments=["/app:*[@Name='ApplicationFrameHost']/control:Window[@Name='Calculator']", "Name", "==", "Calculator"])`
  4. `execute_step(keyword="Set Root", arguments=["/app:*[@Name='ApplicationFrameHost']/control:Window[@Name='Calculator']", "session"])`
- Success criteria: window exists within a bounded wait; `Get Attribute Name == Calculator`; ≤4 tool calls; no clarifying question.
- Autonomy risks: agent waits on the `calc.exe` PID (dead stub) and loops; agent searches under a `Calculator.exe`/`CalculatorApp.exe` top-level window and never finds it because it is frame-hosted; agent falls back to `find_keywords`.
- Latency risks: querying for the window *before* it renders returns empty and the agent retries with an unscoped `//control:Window` (hang) instead of `Wait Until Exists` on the scoped path (fast).
- Evidence to capture: wall-clock from `Start Process` to first successful `Get Attribute`; number of retry queries; whether the agent discovered the `ApplicationFrameHost` container unaided.
- Suspected rf-mcp/PlatynUI issue to watch: ⚠ **ASSUMED** — recent WinUI-3 Calculator builds may run as a **top-level** `control:Window[@Name='Calculator']` under `CalculatorApp.exe` rather than under `ApplicationFrameHost`. Verify the real container on the box; both paths must be tried.

---

### STD-02 — Calculator: 7 + 8 = 15 with read-back (flagship multi-step + verify)
- App: Calculator (always-installed UWP) | Kind: UWP under `ApplicationFrameHost`
- Capability probed: multi-step pointer-click sequence collapsed into one `execute_batch`, then value read-back via `Get Attribute`
- Agent task (verbatim): "In the Windows Calculator, compute 7 plus 8 and tell me the result shown on the display."
- Expected rf-mcp/PlatynUI flow (after STD-01 launch + `Set Root` to the Calculator window):
  1. `execute_batch(session_id, steps=[`
     `{keyword="Pointer Click", arguments=["control:Button[@AutomationId='num7Button']"]},`
     `{keyword="Pointer Click", arguments=["control:Button[@AutomationId='plusButton']"]},`
     `{keyword="Pointer Click", arguments=["control:Button[@AutomationId='num8Button']"]},`
     `{keyword="Pointer Click", arguments=["control:Button[@AutomationId='equalButton']"]}])` — a *known* sequence, one call.
  2. `execute_step(keyword="Get Attribute", arguments=["control:Text[@AutomationId='CalculatorResults']", "Name"])` → returns e.g. `"Display is 15"`.
  3. `execute_step(keyword="Should Contain", arguments=["<returned value>", "15"])`
- Success criteria: display reads 15; the four clicks land in **one** `execute_batch`; read-back asserts `15`; zero clarifying questions.
- Autonomy risks: agent issues four separate `execute_step` calls instead of one batch (turn-economy miss); agent tries to read the result off a button instead of the results `Text` node; agent asserts equality against the whole locale string instead of `Should Contain "15"`.
- Latency risks: after `Set Root`, relative `control:Button[@AutomationId='...']` locators resolve fast; an agent that re-scopes each click to the full `/app:*[@Name='ApplicationFrameHost']/...` path multiplies query cost.
- Evidence to capture: batch-vs-individual call count; exact string returned by `Get Attribute Name` (to confirm the `"Display is N"` locale format); whether the agent used AutomationIds (locale-proof) vs button `@Name` text.
- Suspected rf-mcp/PlatynUI issue to watch: the results element `Name` is **locale-dependent** (`"Display is 15"` only under English UI); `@AutomationId='CalculatorResults'` is stable but the exposed *value* still needs `Should Contain`. ⚠ Confirm `CalculatorResults` is a `control:Text` (not `control:Group`) on this build.

---

### STD-03 — Notepad: keyboard text entry and read-back (packaged-vs-classic ambiguity)
- App: Notepad (`notepad.exe`; **Win11 Notepad is now the packaged `Microsoft.WindowsNotepad` app**) | Kind: packaged app with its **own** top-level window (not frame-hosted)
- Capability probed: keyboard text entry (`Keyboard Type`) then read the text back
- Agent task (verbatim): "Open Notepad, type the line `hello platynui`, and read back the exact text in the document."
- Expected rf-mcp/PlatynUI flow:
  1. `Start Process  notepad.exe` → `Wait Until Exists  /app:*[@Name='Notepad']//control:Window[@Name='Untitled - Notepad']`
  2. `Set Root  /app:*[@Name='Notepad']//control:Window[@Name='Untitled - Notepad']  session`
  3. `Keyboard Type  control:Document  hello platynui` (packaged Notepad edit surface) — see ambiguity note.
  4. `Get Attribute  control:Document  Value` → `"hello platynui"`; then `Should Be Equal`.
- Success criteria: typed text equals read-back text; ≤5 calls; no clarifying question.
- Autonomy risks: **the crux** — the edit surface differs by build. Classic Notepad exposed a `control:Edit` (Win32 `ClassName='Edit'`, `@AutomationId='15'`); the WinUI/RichEdit packaged Notepad exposes a `control:Document`/`control:Edit` (often `@AutomationId='RichEditBox'`). An agent that hard-codes `control:Edit` on a packaged build (or vice-versa) fails and may loop. It should fall back to a `Query` of the window's editable descendants.
- Latency risks: typing into the wrong node (e.g. the tab strip or the window itself) fails fast; but a stale window `@Name` after typing (title flips to `*Untitled - Notepad`) breaks a re-query — see STD-12.
- Evidence to capture: which control type + `@AutomationId` the editable surface actually exposes on this build; whether `Get Attribute Value` (ValuePattern) or a Text-pattern read is required; call count before the agent finds the right node.
- Suspected rf-mcp/PlatynUI issue to watch: ⚠ **ASSUMED / must verify** — process name (`Notepad` vs `Notepad.exe`), window `@Name` (`Untitled - Notepad`), edit-surface control type and `@AutomationId`, and whether the RichEditBox even **supports ValuePattern** for read-back (it may require a Text-pattern path). Flag all four for on-box verification.

---

### STD-04 — Notepad: window operations (maximize / move / restore / minimize)
- App: Notepad (packaged) | Kind: packaged, own top-level window
- Capability probed: windowing keywords (`Maximize Window`, `Move Window`, `Restore Window`, `Minimize Window`) with state read-back
- Agent task (verbatim): "Open Notepad, maximize its window, then move it back to the top-left of the screen and restore it to normal size."
- Expected rf-mcp/PlatynUI flow:
  1. Launch + find window (as STD-03).
  2. `Maximize Window  /app:*[@Name='Notepad']//control:Window[@Name='Untitled - Notepad']`
  3. `Get Attribute  <window>  "Window.WindowVisualState"` (or bounding-rect) to confirm maximized — ⚠ attribute name assumed.
  4. `Restore Window  <window>`
  5. `Move Window  <window>  0  0`
  6. `Get Attribute  <window>  "BoundingRectangle"` → assert x≈0, y≈0.
  7. `Minimize Window  <window>` then `Restore Window  <window>`.
- Success criteria: window reaches maximized state, then restored + moved to (0,0); each state change is verifiable via a read-back; ≤8 calls.
- Autonomy risks: agent cannot find an objective way to confirm "maximized" and asks the user; agent conflates `Restore Window` with `Minimize Window`; agent minimizes and then can't act on the window because a minimized window's descendants may not be queryable.
- Latency risks: acting on descendants while minimized can hang on some UIA providers — restore before further queries. A window-op keyword pointed at a non-window node (`control:Document`) should fail fast, not hang.
- Evidence to capture: which attribute exposes window state (`WindowVisualState` / `BoundingRectangle`); whether `Move Window` coordinates are screen-absolute; behaviour of queries against a minimized window.
- Suspected rf-mcp/PlatynUI issue to watch: ⚠ verify the state attribute name and that `Move Window`/`Resize Window` accept integer pixel args in this build; confirm minimized windows remain queryable or must be restored first.

---

### STD-05 — Settings: toggle a switch (checkbox/toggle read-back)
- App: Settings (`SystemSettings.exe`, package `windows.immersivecontrolpanel`, always-installed UWP) | Kind: UWP, ⚠ likely under `ApplicationFrameHost`
- Capability probed: toggle/checkbox interaction with on/off state read-back
- Agent task (verbatim): "Open Windows Settings to Accessibility → Visual effects, and turn the Transparency effects toggle off; tell me its final state."
- Expected rf-mcp/PlatynUI flow:
  1. `Start Process  explorer.exe  ms-settings:easeofaccess-visualeffects` (deep-link; explorer forwards to `SystemSettings.exe` and returns — do not track the explorer PID).
  2. `Wait Until Exists  /app:*[@Name='ApplicationFrameHost']/control:Window[@Name='Settings']`
  3. `Set Root  <settings window>  session`
  4. `Get Attribute  control:Button[@Name='Transparency effects']  "Toggle.ToggleState"` → read current state — ⚠ attribute name assumed.
  5. If On: `Pointer Click  control:Button[@Name='Transparency effects']`
  6. `Get Attribute  control:Button[@Name='Transparency effects']  "Toggle.ToggleState"  ==  Off`
- Success criteria: toggle ends Off; final state read back and asserted; ≤6 calls; no clarifying question.
- Autonomy risks: the toggle is a UIA **ToggleButton** exposed as `control:Button` (not `control:CheckBox`) — an agent that filters on `control:CheckBox` finds nothing and loops; deep-link lands on the wrong Settings page and the agent must renavigate via the left rail; agent clicks without first reading state and can't tell whether it turned it on or off.
- Latency risks: Settings pages virtualize content — a control scrolled out of view may not exist yet; a scoped `Wait Until Exists` is fast, an unscoped fallback hangs.
- Evidence to capture: the toggle's real control type and the exact attribute exposing on/off; whether the deep-link reliably lands on Visual effects; virtualization behaviour (does the toggle exist before scrolling?).
- Suspected rf-mcp/PlatynUI issue to watch: ⚠ **ASSUMED** — control name `Transparency effects`, its control type, and the toggle-state attribute (`Toggle.ToggleState` vs `IsToggled` vs `Name` suffix). Pick a hardware-independent toggle (Visual effects) so the scenario runs on any machine; verify all three on box.

---

### STD-06 — Paint: toolbar interaction and Take Screenshot
- App: Paint (`mspaint.exe`; Win11 packaged `Microsoft.Paint` WinUI app) | Kind: packaged, own top-level window
- Capability probed: toolbar/tool-button selection + `Take Screenshot` of a scoped node
- Agent task (verbatim): "Open Paint, select the Pencil tool, then capture a screenshot of the Paint window to `paint_pencil.png`."
- Expected rf-mcp/PlatynUI flow:
  1. `Start Process  mspaint.exe` → `Wait Until Exists  /app:*[@Name='Paint']//control:Window[@Name='Untitled - Paint']`
  2. `Set Root  <paint window>  session`
  3. `Pointer Click  control:Button[@Name='Pencil']` — ⚠ tool button `@Name` assumed English.
  4. `Get Attribute  control:Button[@Name='Pencil']  "Name"  ==  Pencil` (or a pressed/selected attribute) to confirm the click registered.
  5. `Take Screenshot  /app:*[@Name='Paint']//control:Window[@Name='Untitled - Paint']  paint_pencil.png` — pass the **descriptor** as arg 1 and the **filename** as arg 2 (never a bare path in the descriptor slot).
- Success criteria: Pencil selected (verifiable), `paint_pencil.png` written non-empty; ≤5 calls.
- Autonomy risks: Paint's toolbar buttons are icon-only with tooltips — the accessible `@Name` may differ from the tooltip text; agent hard-codes a wrong `@Name` and loops; agent tries to introspect the drawing **canvas** (a custom control with little UIA structure) and stalls.
- Latency risks: **`Take Screenshot` filename-in-descriptor-slot** is a known hang trap — passing the path where the descriptor belongs must fail fast (recent fail-fast guard). Confirm the guard triggers rather than a 30 s hang.
- Evidence to capture: real `@Name`/`@AutomationId` of the Pencil tool; that `Take Screenshot(descriptor, filename)` arg order is honoured; screenshot file size > 0.
- Suspected rf-mcp/PlatynUI issue to watch: ⚠ packaged Paint tool-button names; whether the window-scoped `Take Screenshot` captures client area vs full window; canvas is effectively opaque to UIA (do not expect to verify drawn pixels via the tree — that is a vision-only check).

---

### STD-07 — File Explorer: navigate and read a list item (list/DataItem read-back)
- App: File Explorer (`explorer.exe`, always-running shell) | Kind: classic shell window (own window, shared `explorer.exe` process)
- Capability probed: `control:List` / `control:ListItem` (or `control:DataItem` in Details view) enumeration + name read-back
- Agent task (verbatim): "Open File Explorer to `C:\Windows`, and tell me whether a file named `explorer.exe` is listed there."
- Expected rf-mcp/PlatynUI flow:
  1. `Start Process  explorer.exe  C:\Windows` (⚠ drive-letter path: the generated `.robot` rewrites `C:\...` → forward slashes — recent fix; verify no corruption).
  2. `Wait Until Exists  /app:*[@Name='explorer']//control:Window[@Name='Windows']` (folder name = window `@Name`) — ⚠ may be `File Explorer` depending on view.
  3. `Set Root  <explorer window>  session`
  4. `Wait Until Exists  control:List//control:ListItem[@Name='explorer.exe']` (Details view exposes rows as `control:DataItem`, icon views as `control:ListItem`).
  5. `Get Attribute  control:List//control:ListItem[@Name='explorer.exe']  "Name"  ==  explorer.exe`
- Success criteria: the item is found and its `Name` asserted; ≤6 calls; the drive path survives suite generation intact.
- Autonomy risks: `explorer.exe` process hosts **many** windows — the app-root `@Name='explorer'` may match the taskbar/desktop too; agent must disambiguate by window `@Name`. Details-vs-Tiles view changes `ListItem` ↔ `DataItem`, causing a locator miss + loop.
- Latency risks: the items view virtualizes — a target scrolled out of view does not exist until scrolled; a scoped `Wait Until Exists` is fast, an unscoped fallback walks the shell and hangs.
- Evidence to capture: the real window `@Name` for a folder; whether rows are `ListItem` or `DataItem` in the default view; confirmation the generated `.robot` did not corrupt `C:\Windows`.
- Suspected rf-mcp/PlatynUI issue to watch: ⚠ multiple `explorer.exe` windows sharing one app-root; view-mode-dependent control type; **regression check on the Windows drive-letter path rewrite** in `build_test_suite`.

---

### STD-08 — Task Manager: read a value from the process grid (grid/cell)
- App: Task Manager (`Taskmgr.exe`, always-installed WinUI app) | Kind: packaged, own top-level window
- Capability probed: grid navigation, `control:DataItem` cell read-back
- Agent task (verbatim): "Open Task Manager, go to the Processes view, and read back the name of the first process row."
- Expected rf-mcp/PlatynUI flow:
  1. `Start Process  taskmgr.exe` → `Wait Until Exists  /app:*[@Name='Taskmgr']//control:Window[@Name='Task Manager']`
  2. `Set Root  <taskmgr window>  session`
  3. `Pointer Click  control:ListItem[@Name='Processes']` (left NavigationView rail) — ⚠ nav-item `@Name` assumed English.
  4. `Query  control:DataItem  only_first=True` → first grid row.
  5. `Get Attribute  control:DataItem[1]  "Name"` → read the row/cell name and `Log` it.
- Success criteria: a non-empty process name is read from the first row and logged; ≤5 calls; no clarifying question.
- Autonomy risks: the WinUI grid may expose rows as `control:DataItem`, `control:ListItem`, or a `control:Table` row — an agent that guesses one and misses loops; the left rail item might be `control:ListItem` (NavigationViewItem) not `control:MenuItem`.
- Latency risks: the grid is virtualized and updates continuously; `Query` with `only_first=True` bounds cost. An unbounded `Query  control:DataItem` over hundreds of live rows is slow — measure it.
- Evidence to capture: real control type of a Task Manager row; whether the nav rail exposes items as `ListItem`; query time for first-row vs all-rows.
- Suspected rf-mcp/PlatynUI issue to watch: ⚠ **run without elevation** (default) — some rows/details are hidden without admin, and the tree may differ; UAC-elevated Task Manager may be in a **different integrity level** that PlatynUI (non-elevated) cannot introspect. Flag the elevation dependency explicitly.

---

### STD-09 — Snipping Tool: single click "New" (transient-window probe)
- App: Snipping Tool (`SnippingTool.exe`, package `Microsoft.ScreenSketch`, always-installed) | Kind: packaged, own top-level window
- Capability probed: single `Pointer Click` on a named button; handling a transient capture-overlay window
- Agent task (verbatim): "Open the Snipping Tool and start a new snip (click New); then cancel it."
- Expected rf-mcp/PlatynUI flow:
  1. `Start Process  snippingtool.exe` → `Wait Until Exists  /app:*[@Name='SnippingTool']//control:Window[@Name='Snipping Tool']`
  2. `Set Root  <snip window>  session`
  3. `Pointer Click  control:Button[@Name='New']`
  4. `Wait Until Exists  /app:*[@Name='SnippingTool']//control:Window` (the capture toolbar/overlay is a **new, separate** top-level window; the main window minimizes) — must re-`Query`, the cached tree is stale.
  5. `Keyboard Press  control:Window  ESCAPE` to cancel the snip. (⚠ target the overlay window, or send to the focused element.)
- Success criteria: New click registers; capture overlay appears; Esc cancels back to the main window; ≤6 calls; no clarifying question.
- Autonomy risks: after clicking New the **main window minimizes** and the agent keeps querying the (now-hidden) main window — it must recognize the overlay is a new window and re-query (classic "app launched later is invisible until re-queried" trap, applied to a transient window).
- Latency risks: querying the stale minimized main window for the overlay's controls hangs until timeout; the correct move (re-`Query` at desktop scope for the new `SnippingTool` window) is fast. This is a prime **cache-staleness** latency probe.
- Evidence to capture: whether the overlay is a distinct top-level window under the same app root; time lost querying the stale tree before the agent re-queries; whether Esc reliably cancels.
- Suspected rf-mcp/PlatynUI issue to watch: ⚠ modern Snipping Tool's capture overlay is a full-screen transient window that dims the desktop — verify it is enumerable and that the first-keyword tree snapshot is refreshed for it; confirm `@Name='New'` (vs "New snip") on the box.

---

### STD-10 — WordPad: deprecated-app graceful failure (autonomy under absence)
- App: WordPad (`wordpad.exe` / `write.exe`; **deprecated, removed in Windows 11 24H2**) | Kind: classic Win32, own window (when present)
- Capability probed: launch + graceful failure detection when the app is absent; if present, ribbon/document read-back
- Agent task (verbatim): "Open WordPad, type `deprecated but here`, and read the text back. If WordPad is not available on this machine, say so."
- Expected rf-mcp/PlatynUI flow:
  1. `Start Process  wordpad.exe` — on 24H2 this **fails / process exits immediately** (no such executable).
  2. `Wait Until Exists  /app:*[@Name='wordpad']//control:Window` with a **short** bound → times out fast if absent.
  3. On timeout: `Log  WordPad not installed (removed in Win11 24H2)` and stop — report to user, do **not** loop.
  4. If present: `Set Root  <window>`; `Keyboard Type  control:Document  deprecated but here`; `Get Attribute  control:Document  Value` → assert.
- Success criteria: on an absent machine, the agent reports "not available" within a small bound and asks **no** clarifying question; on a present machine, text round-trips.
- Autonomy risks: the crux — an agent that assumes every task is completable will **retry `Start Process` repeatedly**, try alternate paths, or ask the user how to install WordPad. Correct behaviour is a single bounded probe + a clear "not present" report.
- Latency risks: a failed launch must not leave a long `Wait Until Exists` on a full-desktop `//control:Window`; keep the wait scoped and short so absence is detected in seconds, not at a 180 s query timeout.
- Evidence to capture: the OS build (23H2 has WordPad, 24H2 does not); how many retries the agent made before concluding absence; total wall-clock to the "not available" report.
- Suspected rf-mcp/PlatynUI issue to watch: `write.exe`/`wordpad.exe` may be missing entirely; `Start Process` error surfacing — confirm rf-mcp returns a clean launch-failure the agent can act on rather than a swallowed error that leaves the agent guessing.

---

### STD-11 — FAILURE PROBE A: unscoped `//control:Button` (hang-vs-fail-fast latency)
- App: Calculator (reuse the STD-01/02 running instance) | Kind: UWP under `ApplicationFrameHost`
- Capability probed: rf-mcp's **refusal of unscoped locators** and the latency of an over-broad-but-scoped query
- Agent task (verbatim): "Click the plus button in the Calculator." *(Deliberately worded to tempt a bare `//control:Button[@Name='Plus']` locator.)*
- Expected rf-mcp/PlatynUI flow (what the human measures):
  1. **Unscoped, must be refused fast:** `Query(expression="//control:Button")` and `Pointer Click("//control:Button[@Name='Plus']")` → rf-mcp **refuses** the leading-`//` desktop-wide walk and returns an error in milliseconds.
  2. **Scoped-but-broad, must be budget-bounded:** `Query("/app:*[@Name='ApplicationFrameHost']//control:Button")` matches many buttons — measure that it returns under the query budget, not an open-ended walk.
  3. **Correct:** `Pointer Click("control:Button[@AutomationId='plusButton']")` after `Set Root` — fast, unambiguous.
- Success criteria: the bare-`//` locator is refused (fast error), **not** executed; the scoped-broad query returns bounded; the agent recovers to the scoped AutomationId locator without user input.
- Autonomy risks: after the refusal, the agent doesn't understand *why* and retries the same `//` locator (loop) instead of re-scoping to `/app:*[@Name='...']//...`; agent asks the user which button.
- Latency risks: **the core measurement** — confirm the leading-`//` path fail-fasts (ms), not a multi-second desktop walk; record the refusal latency and the scoped-broad-query latency separately.
- Evidence to capture: exact refusal message + latency for `//control:Button`; latency + match count for the scoped-broad query; number of retries before the agent self-corrects to a scoped locator.
- Suspected rf-mcp/PlatynUI issue to watch: does rf-mcp refuse **all** leading-`//` forms (including inside `Query`, `Pointer Click`, `Wait Until Exists`), or only some entry points? Any keyword that lets an unscoped locator slip through to the runtime is a latent multi-second hang.

---

### STD-12 — FAILURE PROBE B: wrong control type + stale locator (recovery)
- App: Notepad (packaged; reuse STD-03 instance) | Kind: packaged, own top-level window
- Capability probed: recovery from (a) a wrong control type and (b) a stale window `@Name` after the title changed
- Agent task (verbatim): "In Notepad, type `x` at the start of the document, then verify the document now contains `x`."
- Expected rf-mcp/PlatynUI flow (the human injects the two faults and measures recovery):
  1. **Wrong control type:** `Keyboard Type("control:Window", "x")` — typing into the *window* node instead of the editable `control:Document`/`control:Edit`. Must **fail fast** (window has no text-input pattern), and the agent should recover to the editable descendant.
  2. **Stale locator after title change:** typing flips the title `Untitled - Notepad` → `*Untitled - Notepad`. A locator hard-pinned to `control:Window[@Name='Untitled - Notepad']` now matches nothing. `Wait Until Exists("/app:*[@Name='Notepad']//control:Window[@Name='Untitled - Notepad']")` must time out on a **short** bound, and the agent should recover by re-querying on a **stable** predicate (`@AutomationId`, or `control:Window` under the app root without the volatile `@Name`, or matching `*Untitled`).
  3. **Correct recovery:** re-`Set Root` on the app-root window by AutomationId/class, then `Get Attribute  control:Document  Value` → `Should Contain "x"`.
- Success criteria: the wrong-type `Keyboard Type` errors fast (no hang); the stale-`@Name` wait fails fast; the agent recovers **without** user input and completes the read-back; total added wall-clock small.
- Autonomy risks: agent re-pins to the same stale `@Name` and loops; agent doesn't realize the `*` dirty-marker changed the title and blames the wrong thing; agent asks the user for the new window title.
- Latency risks: **the core measurement** — a wrong-control-type `Keyboard Type` and a stale exact-`@Name` `Wait Until Exists` must both fail fast (short bound), not sit at the query timeout; record both latencies.
- Evidence to capture: fail-fast latency for typing into a non-editable `control:Window`; timeout latency for the stale `@Name`; whether the agent switched to a stable predicate on its own; retry count.
- Suspected rf-mcp/PlatynUI issue to watch: the dirty-marker title flip (`Untitled` → `*Untitled`) is a **real, reproducible** stale-locator trigger — confirm rf-mcp guidance nudges agents toward `@AutomationId`/stable predicates over volatile `@Name`; confirm no wrong-control-type keyword path hangs at the full query timeout instead of returning a fast "control does not support this pattern" error.

---

# 3. Microsoft Word

Desktop Word is the hardest classic-Win32 target in this suite: a huge auto-generated ribbon tree, an editing surface that exposes UIAutomation `TextPattern` but rarely a clean `ValuePattern`, a Start screen that stands between launch and a usable document, and heavy localization of every ribbon `@Name`. That makes it the best probe for both autonomy (does the agent reach a blank doc and read its own typing back without looping?) and latency (does a wrong ribbon `@Name` fail fast or hang on a desktop-wide query?).

**Shared Word facts & assumptions (verify the ASSUMED ones on the box before scoring):**
- Process: **`winword.exe`** (real). Launch: `Process.Start Process    winword.exe`. A Store/Click-to-Run install still runs `winword.exe` and is a classic top-level window under its own process — **but** if Word was installed as a *Store* package it can be reparented under `ApplicationFrameHost`; the launch scenario treats that as an autonomy branch.
- Main window: **`control:Window[@ClassName='OpusApp']`** (real, long-standing Word window class). Its `@Name` is the document title, e.g. **`Document1 - Word`** (locale/edition-dependent; may read `Document1 - Microsoft Word` or `- Word - <user>`).
- **App root `@Name` is ASSUMED.** I use `/app:*[@Name='Word']` throughout; the real value may be `WINWORD`, `WINWORD.EXE`, or the process display name. First real run must confirm this — everything downstream is scoped to it, so a wrong app `@Name` breaks *every* Word scenario identically.
- **On Windows the window control type is `control:Window`** (Linux would be `control:Frame`). Never start a locator with `//`.
- Editing surface: a **`control:Document`** node (Word's canvas). Reading its full text is the known-hard part — see W2/W7.
- Ribbon: tabs are `control:Tab` containing `control:TabItem` (`@Name='Home'`, `'Insert'`, `'Layout'`, `'References'`, `'Review'`, `'View'` — English UI); commands are `control:Button` with a localized `@Name` and an `@AutomationId` that usually equals the Office `idMso` (e.g. Bold → `@AutomationId='Bold'`, ASSUMED but far more stable than `@Name`).
- Shortcuts that avoid the ribbon entirely (all real): **Ctrl+B** Bold, **Ctrl+F** Navigation-pane Find, **Ctrl+S** Save, **F12** classic Save As dialog, **Ctrl+A** select-all, **Enter/Esc** dismiss Start screen.

---

### W1 — Launch Word and reach a blank document (defeat the Start screen)
- App: Microsoft Word (always-installed on Office/M365 Win11 fleets; the canonical desktop office app) | Kind: classic Win32 (`winword.exe`, own top-level window) — with a UWP/`ApplicationFrameHost` branch if Store-installed.
- Capability probed: launch + window-find + first-keyword tree snapshot timing + Start-screen dismissal.
- Agent task (verbatim): "Open Microsoft Word and get to a new, blank document."
- Expected rf-mcp/PlatynUI flow:
  1. `analyze_scenario(scenario="Open Word to a blank document", context="desktop")` → `session_id`.
  2. `manage_session(action="init", session_id, libraries=["PlatynUI.BareMetal","Process","BuiltIn"])` → read `desktop_guidance`; do **not** call `find_keywords`.
  3. `execute_step("Start Process", ["winword.exe"], session_id)`.
  4. `execute_step("Wait Until Exists", ["/app:*[@Name='Word']//control:Window[@ClassName='OpusApp']"], session_id)` — this forces a fresh desktop-tree snapshot *after* launch, defeating the "app launched after first keyword is invisible" trap.
  5. Dismiss the Start screen: `execute_step("Pointer Click", ["/app:*[@Name='Word']//control:ListItem[@Name='Blank document']"], session_id)` (the template gallery tile; `@Name='Blank document'` ASSUMED English). Fallback the agent should prefer if the tile isn't found: `execute_step("Keyboard Press", ["/app:*[@Name='Word']//control:Window[@ClassName='OpusApp']", "escape"], session_id)` or `"enter"` — either opens a blank doc.
  6. Confirm the canvas: `execute_step("Wait Until Exists", ["/app:*[@Name='Word']//control:Document"], session_id)`.
  7. `execute_step("Set Root", ["/app:*[@Name='Word']//control:Window[@ClassName='OpusApp']"], session_id)` so W2+ use short relative locators.
- Success criteria: `control:Document` exists and is enabled; window `@Name` matches `*- Word`; ≤ 6 tool calls; zero clarifying questions.
- Autonomy risks: agent types into the document *before* the Start screen is dismissed (keystrokes vanish into the gallery); agent burns turns on `find_keywords` instead of using `desktop_guidance`; agent assumes Word opens directly to a blank doc and never handles the Start screen; Store-install reparenting under `ApplicationFrameHost` breaks the `@ClassName='OpusApp'` window locator and the agent must re-scope to `/app:*[@Name='ApplicationFrameHost']/control:Window[@Name='… - Word']`.
- Latency risks: querying the window **before** `Start Process` returns → the tree snapshot predates Word and the agent waits out `Wait Until Exists` timeout; a `//control:Window[@Name='Document1 - Word']` (unscoped, exact-title) locator walks the desktop and hangs. Expected fail-fast: scoped `Wait Until Exists` should time out on a bounded, small wait, not multi-second hang.
- Evidence to capture: wall-clock from `Start Process` to `control:Document` visible; whether tile-click vs Esc/Enter path was taken; the *actual* window `@Name` and app-root `@Name` observed (feeds every later scenario).
- Suspected rf-mcp/PlatynUI issue to watch: first-keyword desktop snapshot missing a just-launched `winword.exe`; app-root `@Name` mismatch making the whole scope wrong; splash/Start-screen delaying `control:Document` beyond a short `Wait Until Exists` window.

---

### W2 — Type a paragraph and read it back
- App: Microsoft Word (popular; core text-entry surface) | Kind: classic Win32.
- Capability probed: keyboard entry into `control:Document` + the crux **read-back** off a UIA Document.
- Agent task (verbatim): "In the open Word document, type the sentence: The quick brown fox jumps over the lazy dog. Then confirm the document contains that exact sentence."
- Expected rf-mcp/PlatynUI flow:
  1. (Root already set to the window from W1.) `execute_step("Focus", ["control:Document"], session_id)`.
  2. `execute_step("Keyboard Type", ["control:Document", "The quick brown fox jumps over the lazy dog."], session_id)`.
  3. Read-back attempt A (fast, may fail): `execute_step("Get Attribute", ["control:Document", "Value", "==", "The quick brown fox jumps over the lazy dog."], session_id)`.
  4. If A returns empty/absent (Document exposes `TextPattern`, not `ValuePattern`), read-back attempt B: `Query("control:Document//control:Text", root=window, only_first=false)` then `Get Attribute(descriptor="…the matched Text run…", attribute_name="Name")` — Word often surfaces line/run text as `control:Text` whose `@Name` equals the text (ASSUMED; verify).
  5. If both fail, read-back attempt C (the reliable fallback): Ctrl+F round-trip — see W4 — using the sentence as the query and asserting ≥ 1 match.
  6. `Should Contain` / `Should Be Equal` on whatever read-back returned.
- Success criteria: the sentence is visibly in the document (screenshot) **and** at least one programmatic read-back path returns it; the agent settles on one working read-back and does not thrash between A/B/C more than once each.
- Autonomy risks: infinite loop retrying `Get Attribute(Value)` when the Document simply has no Value pattern; agent never discovers the Ctrl+F fallback; agent types before focusing and the text lands in the ribbon search box or nowhere.
- Latency risks: `Get Attribute` against a missing pattern should return **fast** (empty/null), not hang — this is a key fail-fast measurement. A `Query("control:Document//control:Text")` that materializes thousands of runs on a large doc could be slow; here the doc is one sentence, so it must be quick.
- Evidence to capture: which read-back path (A/B/C) actually worked; the raw value `Get Attribute(Value)` returned (empty string vs error vs text); time for the failed `Get Attribute` to return (fail-fast proof).
- Suspected rf-mcp/PlatynUI issue to watch: no clean full-text read attribute on `control:Document`; `Get Attribute` on an unsupported UIA property erroring slowly or ambiguously instead of returning empty fast; `Keyboard Type` dropping characters (autocorrect/autocapitalize altering "The quick…").

---

### W3 — Apply Bold via the Home-ribbon button (with keyboard fallback)
- App: Microsoft Word | Kind: classic Win32.
- Capability probed: ribbon `TabItem` activation + `control:Button` click by `@Name`/`@AutomationId`; toggle-state read-back.
- Agent task (verbatim): "Select all the text in the document and make it bold using the Home ribbon."
- Expected rf-mcp/PlatynUI flow:
  1. `execute_step("Focus", ["control:Document"], session_id)`; `execute_step("Keyboard Press", ["control:Document", "ctrl+a"], session_id)`.
  2. Activate Home tab: `execute_step("Pointer Click", ["control:TabItem[@Name='Home']"], session_id)` (relative to rooted window; English `@Name` ASSUMED).
  3. Click Bold, preferring the stable id: `execute_step("Pointer Click", ["control:Button[@AutomationId='Bold']"], session_id)` (`@AutomationId='Bold'` ASSUMED = Office `idMso`); `@Name='Bold'` fallback.
  4. Read-back toggle state: `execute_step("Get Attribute", ["control:Button[@AutomationId='Bold']", "Toggle.ToggleState", "==", "On"], session_id)` (UIA `TogglePattern` state; exact attribute name ASSUMED — may be `ToggleState` or `IsPressed`).
  5. Keyboard fallback if the button can't be resolved: `execute_step("Keyboard Press", ["control:Document", "ctrl+b"], session_id)`.
- Success criteria: after the action, the Bold control reports toggled-on **or** re-reading via Ctrl+F/selection shows bold formatting; the agent does not need the human to disambiguate the button.
- Autonomy risks: agent clicks Bold on the wrong (collapsed) ribbon because the window is too narrow and the Home group is behind an overflow chevron; agent can't read the toggle state and loops toggling on/off; agent forgets Ctrl+A and bolds nothing.
- Latency risks: `control:Button[@Name='Bold']` without the `control:TabItem[@Name='Home']` context may match a Bold-in-a-different-context control or force a broad subtree scan; a mistyped `@Name` (`'bold'`, localized) should fail fast on the scoped query, not hang.
- Evidence to capture: whether ribbon-button path or Ctrl+B fallback succeeded; the toggle-state attribute name that actually worked; ribbon-collapse behavior at the test window size.
- Suspected rf-mcp/PlatynUI issue to watch: toggle/press state not exposed as a readable `Get Attribute`, forcing screenshot-only verification; ribbon overflow hiding the Bold button; `@AutomationId='Bold'` assumption wrong for this Office build.

---

### W4 — Find text with the Navigation pane (Ctrl+F) and assert a match
- App: Microsoft Word | Kind: classic Win32.
- Capability probed: Ctrl+F → Navigation-pane search Edit entry + result read-back (doubles as W2's reliable read-back path).
- Agent task (verbatim): "Search the document for the word 'fox' and confirm it is found."
- Expected rf-mcp/PlatynUI flow:
  1. `execute_step("Keyboard Press", ["control:Document", "ctrl+f"], session_id)` → Navigation pane opens.
  2. `execute_step("Wait Until Exists", ["control:Edit[@Name='Search document']"], session_id)` (search box `@Name`/`@AutomationId` ASSUMED — may be `'Search'` or an `@AutomationId`; verify).
  3. `execute_step("Keyboard Type", ["control:Edit[@Name='Search document']", "fox"], session_id)`.
  4. Assert results: `execute_step("Wait Until Query", ["control:Text[contains(@Name,'Result')]", "contains", "1 of"], session_id)` — the Navigation pane shows a "1 of N" / "Result 1 of N" status (exact text ASSUMED). Alternatively assert the in-document highlight exists.
- Success criteria: search box receives "fox"; a match count / highlighted result is observable; ≤ 4 calls; no clarifying question.
- Autonomy risks: agent types into the document instead of the search box (focus not moved to the pane); agent can't find the result-count element and declares failure despite a visible highlight; agent confuses Ctrl+F (Navigation pane) with Ctrl+H (Replace) and lands in the wrong UI.
- Latency risks: the search Edit `@Name` is a strong guess — a wrong scoped locator should fail fast; an unscoped `//control:Edit` to "find the search box" would walk the desktop and hang.
- Evidence to capture: the real search-box locator (`@Name` vs `@AutomationId`); the exact result-status string used for the assertion; whether this path is more reliable than W2's `Get Attribute` read-back.
- Suspected rf-mcp/PlatynUI issue to watch: Navigation pane result count not exposed as a readable node (only a visual badge), forcing screenshot verification; search box focus not auto-set after Ctrl+F.

---

### W5 — Save the document to a path (exercise fixed Windows drive-letter handling)
- App: Microsoft Word | Kind: classic Win32 (Save As is a Win32 common file dialog).
- Capability probed: F12 classic Save As dialog + path Edit + `build_test_suite` writing a `.robot` that contains a `C:\...` path (the fixed drive-letter rewrite).
- Agent task (verbatim): "Save the current document as C:\\Temp\\eval_word.docx, then generate and dry-run a Robot Framework suite for these steps."
- Expected rf-mcp/PlatynUI flow:
  1. `execute_step("Keyboard Press", ["control:Window[@ClassName='OpusApp']", "f12"], session_id)` → classic **Save As** common dialog (F12 bypasses the modern Backstage — deterministic).
  2. `execute_step("Wait Until Exists", ["/app:*[@Name='Word']//control:Window[@Name='Save As']"], session_id)` (dialog is a child window; `@Name='Save As'` real for the common dialog).
  3. `execute_step("Keyboard Type", ["//control:Edit[@Name='File name:']", "C:\\Temp\\eval_word.docx"], session_id)` — scoped under the Save As window (real `@Name='File name:'` for the Win32 file dialog). **Note:** ensure `C:\Temp` exists or accept the default folder + bare filename.
  4. `execute_step("Pointer Click", ["control:Button[@Name='Save']"], session_id)` (real `@Name='Save'`).
  5. `build_test_suite(session_id, test_name="word_save", output_path="C:\\Temp\\word_save.robot")` — verify the emitted `.robot` keeps the path usable (drive-letter rewritten to forward slashes so it isn't parsed as an RF escape).
  6. `run_test_suite(mode="dry")` then, if green, `run_test_suite(mode="full")`.
- Success criteria: file exists at the target path; the generated `.robot` contains a non-corrupted path (`C:/Temp/eval_word.docx` or a correctly-escaped form) and **dry-run does not hang** (the 180s dry-run hang fix); dry-run passes.
- Autonomy risks: agent fights the modern Backstage ("Save this file" mini-dialog / OneDrive default) instead of using F12; agent can't type a full path into a folder-only field and loops; agent picks a path it lacks permission to write; agent overwrites-prompt ("already exists, replace?") not handled.
- Latency risks: dry-run subprocess must not hang (regression check); the Save As child-window locator, if written unscoped, could walk the desktop.
- Evidence to capture: raw path string as it appears in the generated `.robot` (before/after the drive-letter rewrite); dry-run wall-clock (assert < a few seconds, not 180s); whether the modern Backstage intercepted F12.
- Suspected rf-mcp/PlatynUI issue to watch: `C:\Temp\...` in the generated suite still parsed as RF escapes (`\T`, `\e`) if the rewrite missed a code path; overwrite/OneDrive dialogs blocking the save; dry-run stdin-isolation regression.

---

### W6 — Insert a 2×2 table and fill cells via keyboard navigation
- App: Microsoft Word | Kind: classic Win32.
- Capability probed: Insert-tab command → dialog spinners → per-cell entry via Tab; weak UIA table introspection.
- Agent task (verbatim): "Insert a 2-by-2 table and put the values a1, b1, a2, b2 into its four cells."
- Expected rf-mcp/PlatynUI flow:
  1. `execute_step("Pointer Click", ["control:TabItem[@Name='Insert']"], session_id)`.
  2. `execute_step("Pointer Click", ["control:Button[@Name='Table']"], session_id)` → gallery. **Avoid** the hover-grid picker (pointer-fragile); instead click the menu entry: `execute_step("Pointer Click", ["control:MenuItem[@Name='Insert Table…']"], session_id)` (`@Name='Insert Table…'` ASSUMED; ellipsis may be `...`).
  3. In the **Insert Table** dialog, set columns/rows: `execute_step("Keyboard Type", ["control:Edit[@Name='Number of columns:']", "2"], session_id)`, likewise `'Number of rows:'` = `2` (`@Name`s ASSUMED). Then `execute_step("Pointer Click", ["control:Button[@Name='OK']"], session_id)`.
  4. Fill cells by keyboard (robust, since Word table cells expose poorly in UIA): `Focus` the first cell (click near table top-left or rely on caret-in-first-cell after insert), then `Keyboard Type("control:Document","a1")`, `Keyboard Press(..., "tab")`, `"b1"`, `tab`, `"a2"`, `tab`, `"b2"` — collapse into one `execute_batch`.
  5. Optional read-back: `Query("control:Document//control:DataItem", only_first=false)` and `Get Attribute(@Name)` per cell (ASSUMED Word maps cells to `DataItem`/`Text`; likely weak).
- Success criteria: a 2×2 table with the four values appears (screenshot); ideally at least one cell value is read back programmatically; the fill sequence is one batched call.
- Autonomy risks: agent tries the hover-grid picker and mis-picks dimensions, then loops undoing; agent Tabs past the last cell and creates a 3rd row (Tab in the last cell adds a row); cell read-back returns nothing and the agent declares failure despite correct visual state.
- Latency risks: hover-grid interaction adds pointer-move round-trips and flakiness (not a hang, but turn-cost); a broad `//control:DataItem` query to find cells would walk the desktop and hang — must be scoped under the rooted Document.
- Evidence to capture: whether cells are readable as `DataItem`/`Text`/nothing in UIA (documents Word's table introspection quality for the whole eval); number of tool calls; screenshot of the filled table.
- Suspected rf-mcp/PlatynUI issue to watch: Word table cells not individually addressable via UIA (only via TextPattern ranges) → no per-cell read-back; the Table gallery's grid picker being the only obvious path and being pointer-fragile.

---

### W7 — Round-trip read-back verification of typed content
- App: Microsoft Word | Kind: classic Win32.
- Capability probed: end-to-end "type → independently verify" as a single measured autonomy loop; establishes the canonical Word read-back recipe.
- Agent task (verbatim): "Type 'Evaluation build 0.34' on a new line, then prove — without relying on the screenshot — that the document actually contains that exact string."
- Expected rf-mcp/PlatynUI flow:
  1. `Focus("control:Document")`; `Keyboard Press(..., "ctrl+end")`; `Keyboard Press(..., "enter")`; `Keyboard Type("control:Document","Evaluation build 0.34")`.
  2. Read-back ladder (agent should try in order, stop at first success): (A) `Get Attribute("control:Document","Value","==","…")`; (B) `Query("control:Document//control:Text[@Name='Evaluation build 0.34']", only_first=true)` then `Wait Until Exists` on it; (C) Ctrl+F the exact string and assert a match (W4 mechanism).
  3. `Should Be Equal` / `Should Contain` on the winning path's value.
- Success criteria: at least one non-screenshot path confirms the exact string; the agent records *which* path worked (so the eval can bless one recipe); ≤ ~7 calls including the ladder.
- Autonomy risks: agent exhausts the ladder and gives up; agent treats an empty `Get Attribute(Value)` as "text missing" and re-types (duplicating content); agent inserts at the caret's old position instead of end-of-document (missing Ctrl+End).
- Latency risks: each failed read-back rung must return fast; the `control:Text[@Name='…']` query is exact-match and scoped, so it should resolve or fail quickly; only an unscoped variant would hang.
- Evidence to capture: the definitive Word read-back path (A/B/C) and its latency; whether `Get Attribute(Value)` ever returns real text on this build; any content duplication caused by misread-then-retype.
- Suspected rf-mcp/PlatynUI issue to watch: no reliable programmatic full-text read on `control:Document`, making Ctrl+F the de-facto verification and inflating turn count; `Keyboard Type` autocorrect mutating "0.34".

---

### W8 — Failure-mode probe: ambiguous "Bold" ribbon control (multi-match)
- App: Microsoft Word | Kind: classic Win32.
- Capability probed: the agent's behavior when a locator matches **multiple** controls (Bold appears on the Home tab **and** on the floating Mini-Toolbar and inside the Font dialog), and when a bad `@Name` matches nothing — measures disambiguation + fail-fast.
- Agent task (verbatim): "Make the selected text bold." (Deliberately under-specified — no mention of *where* to find Bold.)
- Expected rf-mcp/PlatynUI flow (ideal, robust):
  1. Prefer the shortcut and sidestep ambiguity entirely: `Keyboard Press("control:Document","ctrl+b")`.
  2. If the agent insists on the ribbon, it must **scope + first-match**: `Query("control:TabItem[@Name='Home']//control:Button[@AutomationId='Bold']", only_first=true)` then `Pointer Click` the returned descriptor — not a bare `control:Button[@Name='Bold']` that can match the Mini-Toolbar's Bold, the ribbon Bold, and a Font-dialog Bold simultaneously.
  3. Read-back toggle state (W3 step 4) to confirm exactly one action took effect.
- Success criteria: bold is applied exactly once; the agent either uses Ctrl+B or resolves the ambiguity with `only_first=true`/tab-scoping **without asking the human** which Bold to click.
- Autonomy risks (the point of the probe): a bare `control:Button[@Name='Bold']` returns >1 node → agent either errors and loops, clicks the wrong one (Mini-Toolbar Bold that vanishes on mouse-move → dead click), or stops to ask the user "which Bold?"; agent toggles twice (double-match click) leaving text un-bold. **Any clarifying question here is an autonomy failure.**
- Latency risks: a non-existent `@Name` (localized/misspelled, e.g. `'Bold '` with a trailing space) must fail fast on the scoped query; the multi-match case should return promptly with N matches, not hang enumerating the ribbon; a fallback to unscoped `//control:Button[@Name='Bold']` would walk the desktop and hang.
- Evidence to capture: how many nodes `control:Button[@Name='Bold']` matches on this build; the agent's disambiguation strategy (shortcut / `only_first` / tab-scope / ask-user); time-to-return for the multi-match and the no-match cases (fail-fast proof); final toggle state (applied once vs zero vs twice).
- Suspected rf-mcp/PlatynUI issue to watch: whether rf-mcp/PlatynUI **surfaces the ambiguity** (returns match-count / refuses) or silently clicks the first hit; whether the Mini-Toolbar Bold is even in the tree at rest (a phantom match that disappears on hover); no guidance nudging the agent toward `only_first`/shortcuts when `@Name` is non-unique.

---

# 4. Microsoft Excel

Excel is the hardest of the Office desktop targets for accessibility-tree automation: the worksheet grid is **virtualized** (only on-screen cells exist in the UIA tree), cell identity in UIA is ambiguous (the `DataItem.Name` is frequently the cell's *displayed value*, not its A1 address), and the "computed result vs. underlying formula" split is exactly the thing a naive agent conflates. Excel 365 desktop is a **classic Win32 app** (process `EXCEL.EXE`, its own top-level window — *not* under `ApplicationFrameHost`), so it does not have the UWP hosting indirection Calculator does. These scenarios are ordered to build a workbook end-to-end, then two failure-mode probes.

**Shared setup / canonical anchors (every scenario reuses these — VERIFY the flagged ones on the box with Accessibility Insights before running).**

| Element | Recommended locator | Confidence |
|---|---|---|
| Main window | `//control:Window[@ClassName='XLMAIN']` | **High** — `XLMAIN` is Excel's long-stable, locale-independent top-level window class; prefer it over the title. |
| App root | `/app:*[@Name='Excel']` | **ASSUMED** — PlatynUI's app-node `@Name` may be `EXCEL`, `Excel`, or the product string. Prefer anchoring on the window `@ClassName` above. |
| Window title | `@Name='Book1 - Excel'` | **ASSUMED** — 365 form; older builds `'Book1 - Microsoft Excel'`; **changes after save** to `<file> - Excel`. Do not depend on it. |
| Name Box | `//control:Edit[@Name='Name Box']` | **ASSUMED (English)** — may surface as `control:ComboBox`; `AutomationId` unknown. |
| Formula bar | `//control:Edit[@Name='Formula Bar']` | **ASSUMED (English)** — returns the *formula text* of the active cell. |
| Worksheet grid | `//control:Table` (named `'Sheet1'`/`'Book1'`) | **ASSUMED** — may be `control:Pane` with a Grid pattern. |
| A cell | `//control:DataItem[@Name='A1']` | **CONTENTIOUS** — `DataItem.Name` is often the cell's **value**, not its address. If so, address-based cell locators return nothing and you must select via the Name Box and read the *selected* cell. **This is the single most important thing to verify first.** |
| "Blank workbook" tile | `//control:*[@Name='Blank workbook']` (Button or ListItem) | **ASSUMED (English)**. |
| Save As dialog | `//control:Window[@Name='Save As']` (`@ClassName='#32770'`) | **ASSUMED (English)** — File-name edit `@Name='File name:'` (ControlId `1001`), Save button `@Name='Save'` (ControlId `1`). |

Global write strategy that sidesteps the cell-locator trap: **navigate with the Name Box + type into the active cell** (`Keyboard Type value + \n`) — writing never needs a cell locator. Only *reading a specific cell back* forces you to resolve the `DataItem.Name` ambiguity, so that is where the probes concentrate.

---

### XL-01 — Launch Excel and reach a blank workbook
- App: Microsoft Excel 365 desktop (always-installed on Office/enterprise Win11 images; the canonical spreadsheet) | Kind: classic Win32, process `EXCEL.EXE`
- Capability probed: launch + window-find, plus handling the **Start screen** ("Blank workbook" template picker) that blocks the grid
- Agent task (verbatim): "Open Microsoft Excel and get me to a new blank workbook."
- Expected rf-mcp/PlatynUI flow:
  1. `analyze_scenario(scenario="Open Excel to a blank workbook", context="desktop")` → `session_id`.
  2. `manage_session(action="init", libraries=["PlatynUI.BareMetal","Process","BuiltIn"])`; use the returned `desktop_guidance`, not `find_keywords`.
  3. `execute_step("Start Process", ["cmd.exe","/c","start","",  "excel"])` — launch via the shell so the App-Paths registry entry resolves `excel` (a bare `Start Process excel.exe` fails if EXCEL.EXE is not on PATH; the direct path is `C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE`, `Office16` for 2016–365, possibly `Program Files (x86)`).
  4. `execute_step("Wait Until Exists", ["//control:Window[@ClassName='XLMAIN']"])` — **query the window before any other PlatynUI keyword** so the runtime snapshots the tree with Excel present.
  5. Dismiss the Start screen: `execute_step("Pointer Click", ["//control:Window[@ClassName='XLMAIN']//control:*[@Name='Blank workbook']"])`, or (often works) `execute_step("Keyboard Press", ["//control:Window[@ClassName='XLMAIN']","{Esc}"])` which opens a blank workbook.
  6. `execute_step("Set Root", ["//control:Window[@ClassName='XLMAIN']"])`; confirm the grid: `Wait Until Exists("//control:Table")`.
- Success criteria: `XLMAIN` window exists; the Start screen is gone; a worksheet grid (`control:Table`/Pane) is queryable; ≤ 2 failed tool calls; **zero clarifying questions**.
- Autonomy risks: agent does not know Excel 365 opens to a Start screen and tries to type into a grid that is behind the template picker; agent guesses wrong tile name; agent uses `Start Process excel.exe` (not on PATH) and loops on a launch failure.
- Latency risks: querying the window **before** `Start Process` completes returns nothing and must be retried; if the agent skips the `Wait Until Exists` and fires a scoped grid query immediately, first-snapshot-misses-Excel forces a re-query. A wrong tile locator (`//` unscoped) should refuse/fail-fast, not hang.
- Evidence to capture: wall-clock launch→grid-ready; whether "Blank workbook" tile or `{Esc}` was needed; the actual app-node `@Name` and window title observed (feeds the anchor table).
- Suspected issue to watch: `desktop_guidance` may not mention the Start-screen gate; `Start Process` shell-launch ergonomics (the child `cmd` exits immediately, so process-handle tracking may look "terminated" while Excel keeps running).

---

### XL-02 — Select a cell via the Name Box (type 'B2' + Enter)
- App: Microsoft Excel 365 | Kind: classic Win32
- Capability probed: keyboard entry into a named control (Name Box) as the *reliable* cell-navigation primitive
- Agent task (verbatim): "In the open workbook, move the selection to cell B2."
- Expected rf-mcp/PlatynUI flow:
  1. `execute_step("Pointer Click", ["//control:Edit[@Name='Name Box']"])` (focus the Name Box).
  2. `execute_step("Keyboard Type", ["//control:Edit[@Name='Name Box']","B2\n"])` — typing an address + Enter jumps the active cell.
  3. Verify: `execute_step("Get Attribute", ["//control:Edit[@Name='Name Box']","Value","==","B2"])` — after the jump the Name Box shows the active cell's address.
- Success criteria: Name Box reads `B2`; single-pass, no retries; no clarifying question.
- Autonomy risks: the Name Box `@Name` is not literally "Name Box" in this locale/build → agent can't find it and falls back to blind clicking somewhere in the grid; agent forgets the trailing `\n` and the address is typed but never committed.
- Latency risks: a wrong Name Box locator scoped to the app should fail-fast on `Get Attribute`/`Keyboard Type`; the danger is an **unscoped** `//control:Edit[@Name='Name Box']` walking the whole desktop — must stay under `Set Root`/`XLMAIN`.
- Evidence to capture: the real `@Name`/`@AutomationId`/control-type of the Name Box; whether Name-Box read-back is a robust selection oracle.
- Suspected issue to watch: Name Box may expose its text via a different attribute (`Value` vs `Name`) depending on whether it's an Edit or ComboBox.

---

### XL-03 — Select a cell by clicking it (grid `DataItem`) vs. the Name Box
- App: Microsoft Excel 365 | Kind: classic Win32
- Capability probed: pointer click on a grid cell, contrasted with XL-02 — directly stresses the virtualization + `DataItem.Name` ambiguity
- Agent task (verbatim): "Click on cell C3 in the spreadsheet."
- Expected rf-mcp/PlatynUI flow:
  1. Try the direct locator: `execute_step("Query", ["//control:DataItem[@Name='C3']", "<root>", "true"])` (scoped under `Set Root` to `XLMAIN`).
  2. **If it resolves** → `execute_step("Pointer Click", ["//control:DataItem[@Name='C3']"])`.
  3. **If `DataItem.Name` is the value (empty cell → empty/blank Name), the address locator returns nothing** → correct fallback is the Name-Box path (XL-02: type `C3\n`), which is why XL-02 is the recommended primitive.
  4. Verify selection via Name Box read-back (`Get Attribute Value == "C3"`).
- Success criteria: C3 becomes the active cell (Name Box shows `C3`); the agent recognizes when the address-based `DataItem` locator is unusable and pivots to the Name Box rather than looping.
- Autonomy risks: the classic trap — agent assumes `@Name='C3'` addresses the cell, the query returns 0 matches on an empty grid, and the agent **retries the same locator repeatedly** instead of switching strategy; agent clicks by pixel coordinates (`Pointer Click x,y`) and lands on the wrong cell.
- Latency risks: a scoped `DataItem` query that misses should fail-fast at the query timeout; the failure to watch is an agent adding `Sleep`s and re-querying (accumulated wall-clock).
- Evidence to capture: **the ground-truth meaning of `DataItem.Name` on this build** (address vs. value); number of retries before the agent pivots to the Name Box.
- Suspected issue to watch: for an *empty* cell there may be no `DataItem` at all (grid only materializes non-empty/on-screen cells), so click-to-select by locator is unreliable for empty cells — the eval should confirm the Name-Box path is the sanctioned answer and `desktop_guidance` says so.

---

### XL-04 — Enter data into cells A1 and A2
- App: Microsoft Excel 365 | Kind: classic Win32
- Capability probed: keyboard data entry into the active cell; **turn economy via `execute_batch`**
- Agent task (verbatim): "Put the number 10 in cell A1 and 25 in cell A2."
- Expected rf-mcp/PlatynUI flow (collapse the known sequence into ONE `execute_batch`):
  1. `execute_batch(session_id, steps=[`
     `{"keyword":"Keyboard Press","arguments":["//control:Window[@ClassName='XLMAIN']","^{Home}"]},`  (Ctrl+Home → A1)
     `{"keyword":"Keyboard Type","arguments":["//control:Window[@ClassName='XLMAIN']","10\n"]},`  (Enter commits A1, moves to A2)
     `{"keyword":"Keyboard Type","arguments":["//control:Window[@ClassName='XLMAIN']","25\n"]}])`  (commits A2)
  2. Read-back A1: select via Name Box (`A1\n`) then `Get Attribute` on the formula bar (`Name`/`Value == "10"`), or read the `DataItem` if address-locators work here (its `Name` should now be `"10"`).
- Success criteria: A1=10, A2=25 committed (visible in formula bar when each is selected); the whole entry is **one batch call**, not three round-trips; zero questions.
- Autonomy risks: agent types into the Name Box instead of the grid; agent omits `\n` so values sit in edit-mode uncommitted; agent uses `Set Query Settings`/individual `execute_step` calls instead of batching (turn-economy miss, not a correctness fail).
- Latency risks: focus target for `Keyboard Type` — if the descriptor points at a non-focusable node the keystrokes go nowhere and the agent burns turns verifying empty cells. Typing to the `XLMAIN` window (which routes to the active cell) is the safe descriptor.
- Evidence to capture: whether `execute_batch` keystroke steps to the window descriptor land in the grid; number of tool calls for the full two-cell entry.
- Suspected issue to watch: `Keyboard Type` with an embedded `\n` — confirm the runtime maps `\n` to the Enter key (commit) and not a literal newline within the cell (which is Alt+Enter in Excel).

---

### XL-05 — Enter a formula `=A1+A2` and read the COMPUTED result back
- App: Microsoft Excel 365 | Kind: classic Win32
- Capability probed: formula entry + the crux **displayed-value vs. underlying-formula** read-back via `Get Attribute`
- Agent task (verbatim): "In cell A3, enter a formula that adds A1 and A2, then tell me the result."
- Expected rf-mcp/PlatynUI flow:
  1. Select A3: Name Box → `Keyboard Type("A3\n")`.
  2. Enter formula into the active cell: `Keyboard Type("//control:Window[@ClassName='XLMAIN']","=A1+A2\n")` (Enter commits and recalculates instantly).
  3. Re-select A3 (Name Box → `A3\n`) so it is the active cell again.
  4. **Read the FORMULA:** `Get Attribute("//control:Edit[@Name='Formula Bar']","Value")` → expect `"=A1+A2"`.
  5. **Read the COMPUTED RESULT:** `Get Attribute("//control:DataItem[@Name='...A3...']","Name")` → expect `"35"`; if the address-locator doesn't resolve, read the *selected* cell's `Name`/displayed text, or use `Wait Until Query("<A3 cell>","==","35")`.
  6. Report `35` (result), distinct from `=A1+A2` (formula).
- Success criteria: agent returns **35** (the computed value) AND correctly distinguishes it from the formula string; formula bar shows `=A1+A2`; the cell shows `35`.
- Autonomy risks: agent reads the **formula bar** and reports `=A1+A2` as "the result" (conflates formula with value) — a primary autonomy failure this scenario is designed to expose; agent cannot locate A3's `DataItem` (Name=value ambiguity) and gives up on the computed value.
- Latency risks: if the agent tries `//control:DataItem[@Name='=A1+A2']` (looking up by formula text) it will never match → must fail-fast, not hang; recalc is synchronous so no long polling should be needed.
- Evidence to capture: **which attribute yields the displayed value** (`DataItem.Name` vs. a `Value`/`LegacyIAccessible.Value` property); whether the agent reports value vs. formula correctly; the exact `Get Attribute attribute_name` that works.
- Suspected issue to watch: the "read displayed value" path may require a UIA property PlatynUI's `Get Attribute` doesn't surface for `DataItem` (cells often expose `GridItem`/`TableItem` patterns, and text via `Name`/`LegacyIAccessible`, but not `ValuePattern`). This is the highest-value finding of the whole Excel section.

---

### XL-06 — Navigate (Ctrl+Home / arrows) and select a range, then read a cell back
- App: Microsoft Excel 365 | Kind: classic Win32
- Capability probed: keyboard navigation + range selection + targeted read-back
- Agent task (verbatim): "Go to the top-left of the sheet, select the block A1 through A3, then tell me what's in A2."
- Expected rf-mcp/PlatynUI flow:
  1. `Keyboard Press("//control:Window[@ClassName='XLMAIN']","^{Home}")` → active cell A1.
  2. Select the range — two options: (a) `Keyboard Press(win,"+{Down}+{Down}")` (Shift+Down×2 → A1:A3), or (b) Name Box → `Keyboard Type("A1:A3\n")` (selects the range directly, most robust).
  3. Confirm selection via Name Box read-back (shows `A1` as the anchor of a multi-cell selection) — note the Name Box shows the active-cell anchor, not the whole range, so also consider `Get Attribute` on a "selection" reporting element if available.
  4. Read A2: Name Box → `A2\n`, then read the cell's displayed value (`Get Attribute`, per XL-05) → expect `"25"`.
- Success criteria: A1:A3 is selected (visually/highlight-confirmable); A2 read-back returns `25`; navigation done via keyboard, no pixel-clicking; zero questions.
- Autonomy risks: agent conflates "select range" with "select a cell" and only lands on A3; agent uses `Shift+Arrow` counts off-by-one; agent expects the Name Box to report the full range `A1:A3` and loops when it only shows `A1`.
- Latency risks: none inherent to navigation (keystrokes are instant); the risk is a follow-up read using an unresolvable cell locator → must fail-fast.
- Evidence to capture: does the Name Box / any element report the selected **range** vs. only the anchor cell? which selection primitive (Shift-arrows vs. `A1:A3` in Name Box) the agent chooses.
- Suspected issue to watch: no clean UIA affordance to read back a multi-cell *selection extent*, so "verify the range is selected" may only be checkable via `Highlight`/screenshot, not an attribute — flag whether the agent can objectively confirm range selection at all.

---

### XL-07 — Save the workbook to a path (F12 → Save As dialog)
- App: Microsoft Excel 365 | Kind: classic Win32
- Capability probed: window/dialog handling, file-dialog text entry, **drive-letter path** correctness in the generated suite
- Agent task (verbatim): "Save this workbook as C:\\Users\\Public\\rfmcp_demo.xlsx."
- Expected rf-mcp/PlatynUI flow:
  1. `Keyboard Press("//control:Window[@ClassName='XLMAIN']","{F12}")` — **F12 opens the classic Save As dialog directly**; prefer it over Ctrl+S, which in 365 opens the modern backstage/"Save this file" flow (OneDrive-first, much harder to drive).
  2. `Wait Until Exists("//control:Window[@Name='Save As']")` (a `#32770` common dialog).
  3. `Keyboard Type("//control:Window[@Name='Save As']//control:Edit[@Name='File name:']","C:\\Users\\Public\\rfmcp_demo.xlsx")` — set the full path in the file-name edit.
  4. `Pointer Click("//control:Window[@Name='Save As']//control:Button[@Name='Save']")`.
  5. Handle a possible "replace existing?" confirmation (`{Enter}` / Yes button) if the file exists.
  6. Verify: `XLMAIN` window title changed to `rfmcp_demo - Excel` (`Get Attribute @Name`), and/or `Process`/file existence check.
  7. `build_test_suite(..., output_path=...)` → `run_test_suite(mode="dry")` then `"full"`.
- Success criteria: file exists at the path; window title reflects the saved name; the **generated .robot** contains the `C:\...` path rewritten to forward slashes (per the recent Windows fix) and the dry-run does **not** hang (per the stdin-isolation fix); ≤ 2 failed calls.
- Autonomy risks: agent uses Ctrl+S, lands in the 365 backstage, and cannot find a "File name" edit (there isn't one in the same shape) → loops; agent forgets the `.xlsx` extension and gets a format prompt; agent doesn't anticipate the overwrite confirmation.
- Latency risks: waiting for the Save As dialog before it renders → re-query; a wrong `File name:` `@Name` (locale) should fail-fast on `Keyboard Type`. The historical 180s dry-run hang must be gone — measure dry-run wall-clock as a regression check.
- Evidence to capture: dry-run duration (regression on the 180s fix); exact bytes of the path argument in the generated `.robot` (regression on the drive-letter fix — confirm `C:\...` became `C:/...` and no `\U`/`\r` corruption); the real `@Name`/ControlId of the file-name edit and Save button.
- Suspected issue to watch: Ctrl+S-vs-F12 divergence in 365 (the agent's instinct is Ctrl+S); the "Save this file" mini-dialog may intercept before the classic dialog and needs its own handling path.

---

### XL-08 — FAILURE PROBE: locate an off-screen / virtualized cell (A500)
- App: Microsoft Excel 365 | Kind: classic Win32
- Capability probed: **grid virtualization** — hang-vs-fail-fast when a far off-screen cell is queried before it is scrolled into view
- Agent task (verbatim): "Read the value in cell A500."
- Expected rf-mcp/PlatynUI flow:
  1. Naive path (the probe): `Query("//control:DataItem[@Name='A500']", "<XLMAIN root>", "true")` while A500 is off-screen → **expected 0 matches**; if the agent wraps it in `Wait Until Exists`, it should time out at the configured query timeout (a few seconds, tunable via `Set Query Settings`), **not** hang 30–180s.
  2. Correct path: Name Box → `Keyboard Type("A500\n")` scrolls A500 into view; **then** the `DataItem` may materialize and be readable (per XL-05 read-back).
- Success criteria: the off-screen query **fails fast** (bounded by the query timeout, single-digit seconds), the agent recognizes virtualization and pivots to Name-Box-scroll-then-read, and ultimately reports A500's value (or "empty"); total wall-clock stays low.
- Autonomy risks: agent loops re-querying `A500` verbatim expecting the tree to change; agent scrolls with `Pointer Scroll` an unknown number of ticks and can't tell when A500 is visible; agent never discovers the Name-Box scroll trick.
- Latency risks: **the core measurement** — how long does a scoped-but-non-matching `DataItem` query take to return empty? Compare against `Set Query Settings` timeout. Any multi-second-per-attempt cost multiplied by retries is the failure signature.
- Evidence to capture: measured time for the missing-cell query to resolve empty; whether `Set Query Settings` timeout is honored; whether Name-Box navigation materializes the cell; retry count before pivot.
- Suspected issue to watch: does PlatynUI attempt a full-subtree walk of the (huge) grid before concluding "no match," inflating latency? Whether virtualized cells ever appear via the `only_first`/query path at all, or strictly require prior scroll.

---

### XL-09 — FAILURE PROBE: wrong / non-existent cell `@Name`
- App: Microsoft Excel 365 | Kind: classic Win32
- Capability probed: fail-fast on a locator that can never match (typo'd address / relying on `@Name=address` when Name is the value)
- Agent task (verbatim): "Read the value in cell ZZ9999." (a valid-but-empty, deliberately-obscure address; also stands in for a typo)
- Expected rf-mcp/PlatynUI flow:
  1. `Query("//control:DataItem[@Name='ZZ9999']", "<XLMAIN root>", "true")` → **0 matches** (empty + off-screen + possibly Name≠address) → fail-fast.
  2. Contrast: an **unscoped** `//control:DataItem[@Name='ZZ9999']` (no app scope) — rf-mcp should **refuse** it (starts with `//`, walks the whole desktop), returning a scope-error immediately rather than hanging.
  3. Recovery: Name Box → `ZZ9999\n` (scrolls there), then read the selected cell → likely reports empty.
- Success criteria: the scoped miss returns fast; the **unscoped** variant is refused/errored immediately (validates the anti-hang scope guard); the agent does not loop and reports "empty/no value" or navigates via the Name Box; total wall-clock low.
- Autonomy risks: agent keeps issuing the same failing locator; agent "fixes" it by removing the app scope (making it unscoped, which is worse) rather than switching to the Name Box; agent invents a `Value`-attribute assertion that silently never matches.
- Latency risks: **the comparison of interest** — scoped-miss latency vs. unscoped-refusal latency. The unscoped query is the classic whole-desktop-walk hang; confirm rf-mcp refuses it in ~0s. Multiple failing retries are the accumulation risk.
- Evidence to capture: exact error text/latency for scoped-miss vs. unscoped-refusal; whether the agent's self-correction moves toward scoping (good) or away from it (bad); does `desktop_guidance` steer it to the Name Box on a miss?
- Suspected issue to watch: the scope-guard must fire on the leading `//` even when a `Set Root` is active (relative-after-root locators are fine, but a fresh `//`-prefixed query without root should still be caught); confirm the guard doesn't get bypassed once `Set Root` has been called.

---

# 5. Capability Matrix, Windows Risk Register & Fail-Fast Probes

This section is app-agnostic. It measures the *substrate* — PlatynUI.BareMetal over Windows UIAutomation, driven through the rf-mcp desktop tool loop — against the two evaluation axes that matter: **autonomy** (task done with few failed/looping tool calls and zero clarifying questions) and **latency** (a wrong keyword/locator fails *fast*, never hangs). Every probe below is written to *provoke* a specific failure and time it, not to exercise a feature.

**Locators/AutomationIds flagged `ASSUMED` must be verified on the box before the run** — many are locale- and build-dependent (Windows 11 23H2 vs 24H2, packaged vs classic).

Timeout note used throughout: PlatynUI exposes `Set Query Settings` / `Wait Until Query`; the effective per-query timeout is the fail-fast lever. Where I write `T_query` I mean the configured query timeout (**ASSUMED default ~10s — verify via `Set Query Settings` and record the real number first; the whole latency story hinges on it**).

---

### 1. Capability Matrix

Legend: **Probe** = smallest tool sequence that exercises the row and exposes its risk. "Fail-fast target" = error/return in **< ~1–2s**; "hang" = anything approaching or exceeding `T_query`, or a multi-timeout pile-up.

| Capability | Expected Windows behaviour (BareMetal + UIAutomation) | Suspected Windows risk | How to probe (concrete) |
|---|---|---|---|
| **Launch + window-find** | `Process.Start Process` launches; `Query("/app:*[@Name='<app>']//control:Window", only_first=True)` (or `Wait Until Exists`) returns the top-level window once shell registers it. UWP window sits under `ApplicationFrameHost`; Win32 under its own process. | (a) Tree snapshotted on first keyword → app launched *after* first PlatynUI call is invisible until re-query/cache-clear. (b) UWP window not yet hosted by ApplicationFrameHost → title empty/"App" for 200–800ms. (c) `control:Frame` (Linux) matches nothing → looks like launch failure. | Launch, then immediately `Wait Until Exists` for the scoped window; measure time-to-first-match. Repeat with app launched *after* an unrelated keyword already ran (staleness probe, §2 R3). |
| **Pointer click accuracy** | `Pointer Click(descriptor)` resolves the node, computes its bounding rect center in physical pixels, moves + clicks. Optional `x,y` are offsets within the element. | High-DPI (scale ≠ 100%) or multi-monitor negative-coordinate origin → click lands off-target if PlatynUI mixes logical/physical px or per-monitor DPI. Off-screen/minimized element → click at stale rect. | Click a Calculator `num7Button` at 100%, 150%, 200% scale and on a secondary monitor left of primary (negative X). Read back result element (§ read-back). Compare hit/miss across DPI. |
| **Keyboard entry** | `Keyboard Type(descriptor, text)` focuses element then injects Unicode; `Keyboard Press`/`Release` for chords (e.g. `CTRL+A`). | Focus not actually landing on the target (focus race, §2 R6) → keystrokes go to wrong control or dropped. IME/dead-key/Unicode-beyond-BMP dropped. Modifier-chord syntax token names (`CTRL`, `ENTER`, `LWIN`) unverified. | `Keyboard Type` a mixed string (ASCII + `ü` + emoji) into Notepad edit; read it back with `Get Attribute`. Then `Keyboard Press` `CTRL+A` then `DELETE`; confirm cleared. Record any dropped chars. |
| **Read-back / Get Attribute** | `Get Attribute(descriptor, "Name")` / a Value property returns live text. Optional `assertion_operator`+`assertion_expected` does the compare in-keyword. | **Provider-dependent**: edit contents live in UIA `ValuePattern.Value` or `LegacyIAccessible.Value`, *not* always `Name`. Reading typed text via `Name` returns the label/placeholder, not the value → silent wrong-pass or wasted retries. | On the same Notepad/Edit, try `Get Attribute` with `attribute_name` = `Name`, then `Value`, then `LegacyIAccessible.Value` (verify which names BareMetal exposes). Record which one returns the typed text. |
| **Grid / cell** | `control:DataItem` / `control:ListItem` addressable by `@Name`/`@AutomationId`; cell text via `Get Attribute`. | Virtualized grids (Explorer details, Settings lists) → off-screen rows absent from the tree until scrolled → `Query` empty → timeout. Cell `@Name` locale/columns-dependent. | In File Explorer details view, `Query` a `control:DataItem` for a filename far down a long folder; observe empty-then-timeout. Then `Pointer Scroll` and re-query. |
| **Menu / ribbon / dropdown** | Open menu via `Pointer Click` on `control:MenuItem` / ribbon `control:Button`; submenu items appear as new tree nodes after open. | Menu items are *ephemeral* — not present until the menu is opened; agent queries before opening → empty → timeout. Popup menus render under a *different* host window / `control:Menu` popup, not under the app window → scoped locator misses. | Open Notepad/WordPad "File" menu; `Query` the "Save As" `control:MenuItem` *before* vs *after* the click that opens it. Check whether the popup is under `/app:*[@Name='<app>']` or a separate popup root. |
| **Window ops** | `Activate/Maximize/Minimize/Restore/Move/Resize/Move And Resize/Bring To Front/Close Window` act on the resolved window node. | `auto_activate` + OS foreground-lock steal races → op targets a window that lost foreground. `Move Window` with off-screen coords orphans the window. Minimized window's children drop from tree → subsequent reads fail. | Maximize, then Minimize, then Restore a Calculator window; between each, `Get Attribute` the window `@Name`/bounding rect and a child button. Confirm children re-appear after Restore. Measure activation-race retries. |
| **Screenshot** | `Take Screenshot(descriptor, filename=..., rect=...)`: **first positional is a DESCRIPTOR**, filename is a named arg. Scoped-node screenshot crops to that element. | **Descriptor-vs-filename trap**: passing a path string as the first positional treats `"C:\shots\a.png"` as a locator → resolves nothing → timeout/hang (the exact Linux fail-fast fix must be confirmed on Windows). Backslash path corruption in generated `.robot` (fixed in ≥0.34.0.dev3 — re-verify). | Deliberately call `Take Screenshot("C:\out\x.png")` (path in descriptor slot) and confirm **fail-fast guard** (< 1s error), not a `T_query` hang. Then the correct form `Take Screenshot("/app:*[@Name='Calculator']//control:Window", filename="C:/out/x.png")`. |
| **Wait / synchronization** | `Wait Until Exists`/`Wait Until Gone`/`Wait Until Query(expr, op, expected)` poll until condition or timeout; `Set Query Settings` tunes timeout/retry. | Default `T_query` too long → every *legitimate* miss (menu not open, row virtualized, elevated window) becomes a multi-second stall. No per-call short-timeout → agent can't cheaply "probe and move on". | Record default `T_query`. Set it to 1–2s via `Set Query Settings`; re-run the R1/R2/R3 probes; confirm bad locators now fail in ≤ the set bound. |
| **Suite build + dry-run + run** | `build_test_suite(output_path=...)` writes byte-exact `.robot` (drive letters rewritten `C:\` → `C:/`); `run_test_suite(mode="dry")` parses without hanging; `mode="full"` executes. | Regression of the two recent Windows fixes: (a) dry-run 180s stdin-isolation hang; (b) drive-letter path corruption in generated file. Session-state → generated locators drift from what actually worked interactively. | After any successful interactive scenario: `build_test_suite(output_path="C:/out/t.robot")`, inspect the file for `C:\`/`\n` corruption, `run_test_suite(mode="dry")` (time it — must be seconds, not 180s), then `mode="full"`. |

---

### 2. Windows Risk Catalog + Fail-Fast Probes

Each risk: **symptom → why it hurts autonomy/latency → probe (full template) → fail-fast target vs bad(hang) signature.**

---

#### IC-R1 — Unscoped `//` locator walks the whole desktop
- **App:** Calculator (always-installed UWP; used only as a bystander) | **Kind:** UWP under ApplicationFrameHost
- **Capability probed:** launch+window-find / query-scope guard
- **Symptom / why it hurts:** A locator starting with `//` (no `/app:*` anchor) forces UIAutomation to `FindAll` from the desktop root across every process — cross-process COM marshaling makes this the #1 Windows hang source, mirroring the Linux ~47s AT-SPI walk. An agent that "just tries `//control:Button[@Name='7']`" burns `T_query` (or far more) per attempt and loops.
- **Agent task (verbatim):** "Open the Windows Calculator and click the 7 key."
- **Expected rf-mcp/PlatynUI flow:** 1) `analyze_scenario` → session. 2) `manage_session(init, ["PlatynUI.BareMetal","Process","BuiltIn"])`. 3) `Start Process CalculatorApp.exe` (or `calc.exe` shim). 4) `Wait Until Exists "/app:*[@Name='ApplicationFrameHost']/control:Window[@Name='Calculator']"`. 5) `Pointer Click "/app:*[@Name='ApplicationFrameHost']/control:Window[@Name='Calculator']//control:Button[@AutomationId='num7Button']"`.
- **Success criteria:** Correct display read-back = "7"; **zero** `//`-prefixed queries issued; any accidental `//` query returns a guard error in **< 1s**.
- **Autonomy risks:** Agent copies the loose `//...[@Name='7']` idiom from generic XPath habit; loops on repeated broad queries; may ask the user "why is it slow?".
- **Latency risks:** Bad = each `//` query ≈ `T_query`+ (10s+), compounding over retries into minutes. Fail-fast = rf-mcp **refuses** the `//` prefix pre-dispatch.
- **Evidence to capture:** For a *forced* `//control:Button[@Name='7']` call: wall-clock to error, and the exact refusal message text (must name the scoping rule). Count of `//` attempts before the agent self-corrects.
- **Suspected issue to watch:** Whether the `//`-refusal guard is present *and identical* on Windows (Linux has it); if absent, the UIAutomation walk hangs far longer than the AT-SPI one.

---

#### IC-R2 — `control:Frame` (Linux idiom) vs `control:Window` (Windows)
- **App:** Notepad (always-installed) | **Kind:** classic Win32 (own process)
- **Capability probed:** window element-type correctness / fail-fast on no-match
- **Symptom / why it hurts:** On Windows the top-level window is `control:Window`; `control:Frame` is the Linux GTK trap. A wrong type **matches nothing** and *should* fail fast — but if the runtime waits `T_query` for a node that can never appear, every step built on `Set Root` to that phantom hangs, and the agent has no signal distinguishing "wrong type" from "app not up yet".
- **Agent task (verbatim):** "Open Notepad, type Hello, and tell me what the window title bar says."
- **Expected flow:** 1–3) session + init + `Start Process notepad.exe`. 4) `Wait Until Exists "/app:*[@Name='Notepad']//control:Window"` → `Set Root` to it. 5) `Keyboard Type` into the scoped edit (`control:Document`/`control:Edit`). 6) `Get Attribute` window `@Name`.
- **Success criteria:** Title read back (e.g. `"*Untitled - Notepad"` / `"Untitled - Notepad"` — **ASSUMED, locale/build-dependent**); if the agent tries `control:Frame` it gets a fast no-match, not a wait.
- **Autonomy risks:** An agent primed on Linux PlatynUI docs uses `control:Frame`; without a distinguishing error it retries the same wrong type or asks the user.
- **Latency risks:** Bad = `Wait Until Exists control:Frame` blocks full `T_query` then times out with a generic "not found". Fail-fast = **immediate** empty result from `Query` (`only_first=True`) so the agent can pivot in one turn.
- **Evidence to capture:** Time and message for `Query(".../control:Frame", only_first=True)` on Windows; does BareMetal's `desktop_guidance` (from init) explicitly say "Windows = control:Window"? Note whether it steers the agent away up front.
- **Suspected issue to watch:** Whether `desktop_guidance` is platform-aware (emits `control:Window` on Windows) or ships the Linux `control:Frame` crib — a doc bug here directly causes looped no-matches.

---

#### IC-R3 — Desktop tree snapshotted on first keyword → later-launched apps invisible
- **App:** Notepad launched *after* an initial BareMetal keyword (any bystander already open) | **Kind:** classic Win32
- **Capability probed:** cache freshness / re-query discipline
- **Symptom / why it hurts:** The runtime snapshots the desktop tree on the first PlatynUI keyword. An app started *after* that is absent from the cached tree; `Query` for it returns empty and waits `T_query`, even though the app is clearly on screen. This is a pure **autonomy trap**: the agent sees a running app but the tool "can't find it".
- **Agent task (verbatim):** "Take a screenshot of the current desktop, then open Notepad and type 'ready'."
- **Expected flow:** The screenshot keyword warms/snapshots the tree first. Correct agent then either (a) re-queries with a cache-clearing `Query`/`Wait Until Exists` (BareMetal should re-snapshot on a fresh top-level query), or (b) knows to `Set Root`/re-query after `Start Process`. Ideal: `Start Process notepad.exe` → `Wait Until Exists "/app:*[@Name='Notepad']//control:Window"` (this call must force a re-scan) → `Keyboard Type`.
- **Success criteria:** Notepad found and typed into on the **first** post-launch `Wait Until Exists`; no `T_query` stall attributable to staleness.
- **Autonomy risks:** Agent concludes "launch failed", re-launches Notepad (now two instances), then two windows → ambiguous locator (see IC-R9). Or asks the user "did Notepad open?".
- **Latency risks:** Bad = one full `T_query` stall per stale query, ×N retries. Fail-fast = re-query returns the new window within the normal launch settle time (< ~2s).
- **Evidence to capture:** Does `Wait Until Exists` for a newly-launched top-level window force a re-snapshot, or does it poll a stale cache until timeout? Time-to-find for the app launched *after* first keyword vs an app launched *before* it (delta = the staleness cost). Whether any explicit cache-clear affordance exists.
- **Suspected issue to watch:** No agent-visible "refresh tree" primitive → the agent cannot recover except by luck; and whether `Wait Until Exists` polling actually re-reads the live tree each iteration.

---

#### IC-R4 — Get Attribute read-back is UIAutomation-pattern-dependent
- **App:** Notepad | **Kind:** classic Win32 (edit exposes UIA patterns)
- **Capability probed:** read-back / verify typed text
- **Symptom / why it hurts:** Typed text lives in `ValuePattern.Value` (or `LegacyIAccessible.Value` / `TextPattern`), **not** `Name`. `Get Attribute(..., "Name")` on an Edit often returns the empty string or the control label — the agent "verifies" and gets a false negative (retries typing) or false positive (moves on with wrong data). Provider variance across classic Edit vs modern RichEditBox makes this non-uniform.
- **Agent task (verbatim):** "Type 'invoice-42' into Notepad and confirm the document contains exactly that text."
- **Expected flow:** After `Keyboard Type`, read the *scoped edit* with the value-bearing attribute: `Get Attribute("/app:*[@Name='Notepad']//control:Document", attribute_name="Value", assertion_operator="==", assertion_expected="invoice-42")` (attribute name **ASSUMED** — the probe's job is to find which name BareMetal maps to `ValuePattern.Value`).
- **Success criteria:** Exactly one attribute name returns `"invoice-42"`; the assertion form passes; agent uses that name, not `Name`.
- **Autonomy risks:** Agent guesses `Name`, gets `""`, loops re-typing; or tries several attribute names blindly (each an extra call) inflating turn count.
- **Latency risks:** Low per-call, but wrong-attribute → verify-fail → re-type loops multiply total wall-clock and turns.
- **Evidence to capture:** A small table: `{Name, Value, LegacyIAccessible.Value, Text}` → actual returned string, for classic Notepad Edit **and** Windows 11 modern Notepad RichEditBox (they likely differ). Whether `desktop_guidance` documents the value attribute name.
- **Suspected issue to watch:** No canonical "get the editable value" affordance → every agent rediscovers the pattern name by trial; and modern Notepad's RichEditBox may expose text only via `TextPattern` (no `ValuePattern`), breaking `Get Attribute` value read-back entirely.

---

#### IC-R5 — Coordinate/click targeting under high-DPI and multi-monitor
- **App:** Calculator | **Kind:** UWP under ApplicationFrameHost
- **Capability probed:** pointer click accuracy across DPI / monitor origin
- **Symptom / why it hurts:** If PlatynUI mixes logical and physical pixels, or uses system-DPI instead of per-monitor-DPI-aware rects, clicks miss at scale ≠ 100% or on a monitor with a negative-X origin. A geometric miss produces *no error* — the click lands on the wrong control or empty space, and the agent only discovers it via a wrong read-back, then loops.
- **Agent task (verbatim):** "In Calculator, compute 7 plus 8 and read the result."
- **Expected flow:** Scoped clicks `num7Button` → `plusButton` → `num8Button` → `equalButton`, then `Get Attribute` on results element (`AutomationId='CalculatorResults'`, `Name` ≈ `"Display is 15"` — **ASSUMED English**). Element-descriptor clicks (not raw `x,y`) should be DPI-robust because PlatynUI computes the rect; the probe verifies that claim.
- **Success criteria:** Result reads 15 at **100%, 150%, 200%** scale and when Calculator is on a secondary monitor positioned left of primary (origin negative X). Zero coordinate-based retries.
- **Autonomy risks:** On a miss the agent can't tell "wrong element" from "wrong pixel"; it re-clicks/re-queries and may give up asking the user to "check the screen".
- **Latency risks:** Misses don't hang, but silent-miss → verify-fail → retry loops inflate turns; a click on empty space that opens nothing wastes a full verify cycle each time.
- **Evidence to capture:** Per DPI/monitor combo: hit/miss, final result value, number of retries. If `x,y` offsets are used anywhere, whether they're logical or physical px.
- **Suspected issue to watch:** Per-monitor-DPI-v2 awareness of the PlatynUI Rust core process (manifest/`SetProcessDpiAwarenessContext`); negative-origin monitor rect handling. Flag if only element-descriptor clicks are safe and raw-coordinate keywords are DPI-broken.

---

#### IC-R6 — Focus / activation races and dialog stacking (auto_activate)
- **App:** Notepad + its modal "Save changes?" dialog | **Kind:** classic Win32 with child modal
- **Capability probed:** focus/activation ordering, multi-window/dialog stacking
- **Symptom / why it hurts:** `auto_activate` and the OS foreground-lock can race: a keyword activates window A while a modal dialog B is actually foreground; keystrokes go to the wrong surface, or the click targets an occluded control. Modal dialogs are separate top-level windows (often a distinct `control:Window` under the same process) that *block* the parent — querying the parent's controls while the modal is up returns disabled/occluded nodes.
- **Agent task (verbatim):** "Open Notepad, type 'draft', then close it without saving."
- **Expected flow:** Type → `Close Window` → a "Don't Save" dialog appears as a **new** `control:Window` → `Wait Until Exists "/app:*[@Name='Notepad']//control:Window//control:Button[@Name='Don't Save']"` (Name **ASSUMED English**) → `Pointer Click` it.
- **Success criteria:** Dialog detected as a distinct window; "Don't Save" clicked; process exits. No keystrokes/clicks land on the parent while the modal is up.
- **Autonomy risks:** Agent keeps targeting the parent edit after the modal appears (parent is disabled) → clicks do nothing → loop. Or it doesn't anticipate the dialog and reports "closed" while the modal is still on screen (user must intervene).
- **Latency risks:** Bad = activation race makes `Get Attribute`/click retry against the wrong foreground window until `T_query`. Fail-fast = modal appears in the tree within settle time and the scoped `Wait Until Exists` catches it.
- **Evidence to capture:** Does the modal appear under the app process or a separate host? Time from `Close Window` to modal-visible. Any evidence of keystrokes reaching the parent while modal up (parent text mutated). Whether `Focus` before `Keyboard Type` is *required* for reliability (record success rate with/without an explicit `Focus`).
- **Suspected issue to watch:** No serialization between `auto_activate` and foreground-lock → intermittent wrong-target actions (flaky, the worst kind for autonomy). Also: whether occluded/disabled controls are reported as clickable.

---

#### IC-R7 — Locale-dependent control `@Name`
- **App:** Notepad or Calculator on a **non-English** Windows | **Kind:** either
- **Capability probed:** locator stability across UI language
- **Symptom / why it hurts:** `@Name` reflects the localized label ("Save" → "Speichern", "Don't Save" → "Nicht speichern"). A `@Name`-based locator that worked on English silently matches nothing on a German/French box → empty → `T_query` stall → agent loops. `@AutomationId` is language-invariant and is the fix.
- **Agent task (verbatim):** "Save the current Notepad document to the file C:\\temp\\note.txt."
- **Expected flow:** Prefer `@AutomationId` throughout (e.g. dialog buttons by AutomationId, not `@Name`). Where an AutomationId is unknown, the agent should read a candidate node's `@Name` via `Get Attribute` and adapt — not hard-code English.
- **Success criteria:** Flow completes on a non-English UI with **no** English `@Name` literal in the successful path; any `@Name='Save'` attempt fails fast (empty match), not a hang.
- **Autonomy risks:** Agent hard-codes English `@Name`, gets empty matches, retries the same literal, then asks the user to translate.
- **Latency risks:** Each localized-miss = one `T_query`; compounding across a multi-button dialog flow.
- **Evidence to capture:** For each control used: is a stable `@AutomationId` available, and does BareMetal expose it? List every place the ideal flow *must* fall back to `@Name` (i.e. no AutomationId exists) — those are the genuine locale-fragility points. Note which `@Name`s the doc/`desktop_guidance` hard-codes in English.
- **Suspected issue to watch:** `desktop_guidance` / cheat-sheet shipping English `@Name` examples that agents copy verbatim; and controls (e.g. some menu items) that expose *only* a localized `@Name` with no AutomationId.

---

#### IC-R8 — Take Screenshot descriptor-vs-filename trap
- **App:** Calculator (bystander) | **Kind:** UWP
- **Capability probed:** screenshot argument-shape fail-fast
- **Symptom / why it hurts:** First positional of `Take Screenshot` is a **descriptor**, not a path. `Take Screenshot("C:\shots\calc.png")` treats the path as a locator → resolves nothing → without a guard, waits `T_query`. This is the exact Linux failure the fail-fast guard fixed; the probe confirms parity on Windows (where the backslash path is also a corruption risk in generated `.robot`).
- **Agent task (verbatim):** "Take a screenshot of the Calculator window and save it as C:\\temp\\calc.png."
- **Expected flow:** `Take Screenshot("/app:*[@Name='ApplicationFrameHost']/control:Window[@Name='Calculator']", filename="C:/temp/calc.png")` — descriptor first, `filename=` named, forward slashes.
- **Success criteria:** File written to `C:\temp\calc.png`; the wrong form (path in first positional) returns a **guard error < 1s** naming the descriptor/filename distinction.
- **Autonomy risks:** Agent passes the path positionally (natural mistake), and without a guard interprets the timeout as "screenshot unsupported" → abandons evidence capture (hurts every downstream visual check).
- **Latency risks:** Bad = `T_query` hang on the path-as-descriptor. Fail-fast = pre-dispatch guard (mirror of the Linux `_ALLOW_PATH_DESCRIPTOR` guard) rejects a filesystem-looking first positional immediately.
- **Evidence to capture:** Time+message for the wrong form; whether the guard exists on Windows; whether the written path in a *generated* suite keeps `C:/temp/...` (drive-letter fix intact) rather than a mangled `C:\temp` / `C:	emp` (`\t`).
- **Suspected issue to watch:** Windows-specific: a bare-path descriptor that also contains a drive letter may bypass a Linux-tuned "looks like a path" heuristic (`/`-based) → guard misses `C:\...`. Verify the guard recognizes `X:\` and UNC `\\` forms.

---

#### IC-R9 — Locator resolves to MULTIPLE nodes
- **App:** File Explorer (two panes / duplicate labels) or two Notepad instances | **Kind:** classic Win32
- **Capability probed:** ambiguity handling on action keywords
- **Symptom / why it hurts:** When a descriptor matches several nodes, an action keyword (`Pointer Click`, `Keyboard Type`) must either deterministically pick (first / raise error) or it acts on an arbitrary/wrong node. `Query` has `only_first`; action keywords don't — so the behaviour is implicit. Ambiguity → wrong-target action → silent failure → loop. This is common on Windows: multiple "Close" buttons, duplicate toolbar items, two app instances after an accidental double-launch (see IC-R3).
- **Agent task (verbatim):** "Two Notepad windows are open. Type 'first' into the Notepad whose title is 'Untitled - Notepad'."
- **Expected flow:** Disambiguate by title in the scope: `Set Root "/app:*[@Name='Notepad']/control:Window[@Name='Untitled - Notepad']"` (**ASSUMED** exact title), then act on the *relative* edit under that root. `Query(..., only_first=True)` where a single match is expected; error if the intended-unique node is non-unique.
- **Success criteria:** Text lands in the correct window only; if the descriptor is ambiguous, the action keyword **errors clearly** ("N matches") rather than silently picking one.
- **Autonomy risks:** Agent uses a non-unique descriptor, acts on the wrong window, verifies the wrong one, and loops; or can't tell why the visible target didn't change.
- **Latency risks:** Not a hang per se, but ambiguous-action → wrong-verify → retry loops inflate turns and wall-clock.
- **Evidence to capture:** Does `Pointer Click`/`Keyboard Type` on a multi-match descriptor (a) act on first, (b) raise, or (c) act on all? Record the exact behaviour and message. Is there any agent-facing signal that a descriptor is non-unique *before* acting?
- **Suspected issue to watch:** Silent "first match wins" on action keywords (no error) — the most autonomy-hostile default, because the agent gets no feedback that its locator was ambiguous.

---

#### IC-R10 — Batch-vs-step latency trade-off (and cumulative-timeout amplification)
- **App:** Calculator | **Kind:** UWP
- **Capability probed:** `execute_batch` vs `execute_step` economy and failure isolation
- **Symptom / why it hurts:** `execute_batch` collapses a known sequence into one MCP call (fewer turns, less handshake overhead — good for autonomy). But if a middle step has a wrong locator, a naive batch runner waits `T_query` on that step and, worse, may attempt subsequent steps that then also miss → **cumulative multi-timeout pile-up** in a single opaque call, with coarse recovery. `execute_step` gives per-step feedback (better latency isolation, more turns). The eval must quantify where the crossover is.
- **Agent task (verbatim):** "In Calculator compute 12 × 6 and read the result." (Human injects one wrong AutomationId into the middle of the ideal batch to measure isolation.)
- **Expected flow (batch):** `execute_batch(steps=[Click num1, Click num2, Click multiplyButton, Click num6, Click equalButton, Get Attribute results])`. Human-tampered variant: replace `num6Button` with a bogus `numSixButton`. Compare against the same sequence via six `execute_step` calls.
- **Success criteria:** (a) Clean batch computes 72 in one call, meaningfully fewer turns than 6 steps. (b) Tampered batch **fails fast at the bad step** — stops, reports the failing step index + locator, does **not** run remaining steps into further timeouts — ideally within one `T_query`, not six.
- **Autonomy risks:** Agent over-batches an *unknown* sequence (locators unverified) → a single bad locator sinks the whole batch and the agent can't localize the failure → re-runs the whole batch or falls back to slow one-by-one.
- **Latency risks:** Bad = tampered batch burns `N × T_query` (runs every remaining step against a now-wrong state). Fail-fast = batch aborts at first failing step, returns the index, and (per desktop recovery tiers) suggests step-mode retry.
- **Evidence to capture:** Wall-clock and turn-count: clean-batch vs 6-steps vs tampered-batch. Does the batch runner **short-circuit** on first failure or run to completion? Does the error name the failing step + locator (recoverable) or return a generic `'keyword'`/dict-key error (the known cryptic-batch-error footgun)? Per-step timeout cap present?
- **Suspected issue to watch:** Batch not short-circuiting on a hard locator miss (cumulative hang); and cryptic batch error payloads that give the agent nothing to recover on — pushing it back to slow step mode and defeating the turn-economy purpose.

---

#### IC-R11 — Integrity level / UAC boundary (elevated windows & secure desktop) — *Windows-only*
- **App:** any app running **elevated** (e.g. Task Manager, or an app "Run as administrator"), and the UAC consent prompt itself | **Kind:** elevated Win32 / secure desktop
- **Capability probed:** cross-integrity-level visibility, hang-vs-clean-error on invisible targets
- **Symptom / why it hurts:** A medium-integrity automation host **cannot** read a higher-integrity (elevated) window's UIAutomation tree (UIPI), and the UAC consent prompt lives on the **secure desktop** — entirely invisible to any automation. `Query` returns empty and waits `T_query`, indistinguishable from "app not up yet". Pure autonomy trap with no in-band recovery: the agent will loop forever or ask the user to click UAC (which it cannot see or do).
- **Agent task (verbatim):** "Open Task Manager and read the number of running processes." (If Task Manager launches elevated / requires consent on this box.)
- **Expected flow:** Launch → `Wait Until Exists` the Task Manager `control:Window`. If elevated and host is not, the query returns empty; the *ideal* behaviour is a **fast, explicit** "target not accessible / possible elevation boundary" signal so the agent can report the blocker to the user in one turn instead of hanging.
- **Success criteria:** If accessible (host elevated too), task completes. If not, the tool surfaces an **integrity/elevation-boundary error fast** (≤ a short bound), and the agent **reports the blocker** rather than looping or silently timing out.
- **Autonomy risks:** Agent retries the empty query indefinitely; or worse, tries to "click" a UAC prompt it cannot see; or asks the user repeatedly. No amount of re-query helps across the UIPI boundary.
- **Latency risks:** Bad = every query against the elevated window burns `T_query`, ×retries → minutes of dead time. Fail-fast = distinguish "empty because inaccessible" from "empty because not-yet-present" and short-circuit.
- **Evidence to capture:** Integrity level of the PlatynUI host process; does `Query` against an elevated window return empty-then-timeout, or a distinct access error? Behaviour when a UAC prompt is on the secure desktop (host sees nothing — confirm and time it). Whether any guidance tells the agent "elevation boundary → stop and report".
- **Suspected issue to watch:** No integrity-boundary detection → indefinite loops on elevated targets; and the secure-desktop UAC prompt as an unrecoverable, invisible blocker with no fast, honest error.

---

#### IC-R12 — ApplicationFrameHost hosting indirection & UWP launch race — *Windows-only*
- **App:** Settings (`SystemSettings.exe`) or Calculator | **Kind:** UWP under ApplicationFrameHost
- **Capability probed:** correct host-anchored locator + launch-settle timing for packaged apps
- **Symptom / why it hurts:** A UWP window is **not** under its own process node — it's hosted by `ApplicationFrameHost.exe`, and during launch the frame briefly shows an empty/"App" title before the real title resolves. An agent that scopes to `/app:*[@Name='Settings']//control:Window` (the app's own process name) finds nothing, because the window lives at `/app:*[@Name='ApplicationFrameHost']/control:Window[@Name='Settings']`. Empty → `T_query` stall → wrong conclusion "launch failed".
- **Agent task (verbatim):** "Open Windows Settings and tell me which page it opens on."
- **Expected flow:** `Start Process` the Settings URI (`ms-settings:`) or `SystemSettings.exe` → `Wait Until Exists "/app:*[@Name='ApplicationFrameHost']/control:Window[@Name='Settings']"` (retry until title resolves past the transient "App"/empty) → `Get Attribute` the selected nav item / header `@Name`.
- **Success criteria:** Window found under **ApplicationFrameHost** within launch-settle time; agent does not scope to the app's own process name for the window; transient empty title handled by `Wait Until Exists` polling.
- **Autonomy risks:** Agent scopes to `/app:*[@Name='Settings']//control:Window`, gets empty, re-launches or gives up. Or reads the transient "App" title and reports it as the page name.
- **Latency risks:** Bad = `T_query` stall against the wrong host anchor, ×retries. Fail-fast = the correct ApplicationFrameHost-anchored `Wait Until Exists` resolves within a couple seconds.
- **Evidence to capture:** Does `desktop_guidance` document the ApplicationFrameHost pattern for UWP (vs own-process for Win32)? Duration of the transient empty/"App" title window. Whether the app's own process node ever exposes the visible window at all.
- **Suspected issue to watch:** Guidance not distinguishing UWP (ApplicationFrameHost-hosted) from classic Win32 (own process) → agents systematically mis-scope every packaged app (Calculator, Settings, Store, some Office builds), the single highest-frequency Windows locator error.

---

### 3. How to Score This Section

Aggregate across all probes into two headline numbers per axis:

- **Autonomy score:** for each probe, `1 − (failed_tool_calls + clarifying_questions) / total_tool_calls`, plus a hard **fail flag** if the agent asked the user anything it should have resolved itself (R3 re-query, R7 AutomationId fallback, R12 host anchor). Loop detection: any locator issued ≥3× unchanged = autonomy failure.
- **Latency score:** per probe, record wall-clock for the *wrong* path and assert it meets the fail-fast target (< ~1–2s for guarded cases R1/R2/R8; ≤ configured `T_query` for the honest-miss cases R3/R11/R12). Any single tool call ≥ 30s = latency failure; any cumulative-timeout pile-up (R10) = latency failure.

**Run `Set Query Settings` to a short `T_query` (1–2s) for the whole failure-mode suite** — it both bounds the blast radius of a genuine hang and sharpens the distinction between "fails fast by design" (guarded) and "hangs until timeout" (unguarded). Record the *default* `T_query` first, since several risk severities (R1, R3, R11, R12) scale directly with it.

**Cross-check the two recent Windows fixes as regression gates on every suite build:** (a) `run_test_suite(mode="dry")` must return in seconds (not 180s — stdin-isolation regression); (b) every generated `.robot` must contain forward-slash paths (`C:/...`), never `C:\` or a `\t`/`\n`-mangled path.

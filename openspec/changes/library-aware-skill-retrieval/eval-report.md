# Evaluation: library-aware skill retrieval (task 8.2)

**Date:** 2026-11-12.
**Harness:** `experiments/library-aware-skills-eval/` (see its README).
**Official skills:** MarketSquare/robotframework-plugins at `37562367fc8c330db778ecadec287cef24054fc5`, served through `--skills-dir`.
**Models (OpenRouter):** `deepseek/deepseek-v4.1-flash-1017`, `openai/gpt-6-luna`, `minimax/minimax-m3.1-flash`.
**Size:** 2 tasks (Browser: an iframe and an iframe inside a shadow root; RequestsLibrary: bearer token, nested payload, money arithmetic, negative test) × 3 arms × 5 repetitions = **90 runs**. A run may take up to 60 turns. The total cost was **$1.23**.

| arm | skills | pointers |
| --- | --- | --- |
| A | rf-mcp built-in only | on |
| B | official, then built-in | on |
| C | official, then built-in | off |

**Pass:** `rf_analyze` is clean, `rf_run` passes, and the suite contains every required test name. **Bug caught:** the passing suite also fails against a broken copy of the app (the card shows three digits, the coupon is "rejected", or the VAT is 16%).

## Results

| arm | n | pass | pass + bug caught | median turns | median tokens | cost $ | pointers | skill reads | reads of a pointed skill |
|---|---|---|---|---|---|---|---|---|---|
| A | 30 | 1.00 | 1.00 | 22 | 378,792 | 0.323 | 72 | 42 | 19 |
| B | 30 | 0.97 | 0.97 | 23 | 444,226 | 0.454 | 73 | 73 | 16 |
| C | 30 | 0.97 | 0.97 | 24 | 478,286 | 0.454 | 0 | 76 | 0 |

| task / arm | n | pass | median turns | median tokens |
|---|---|---|---|---|
| browser A | 15 | 1.00 | 27 | 437,764 |
| browser B | 15 | 1.00 | 23 | 389,256 |
| browser C | 15 | 0.93 | 31 | 657,367 |
| requests A | 15 | 1.00 | 17 | 297,846 |
| requests B | 15 | 0.93 | 23 | 470,612 |
| requests C | 15 | 1.00 | 21 | 367,737 |

There were two failures, each a single event:
- minimax, requests, B: the model ended after 23 turns without writing the suite.
- luna, browser, C: the model reached the turn limit with a suite that still failed.

Every passing suite also caught the bug in the broken copy.

## Gate (D8)

- **Pass rate:** B 0.97 against A 1.00. Not met, by one run in 30.
- **Median tokens:** B 444,226 against a limit of 1.1 × A = 416,671. Not met: B used 17% more than A.

**The gate fails.** The follow-up change that would retire `rf-locators` and `rf-requests` should not go ahead on this evidence.

## What the numbers say

1. **The tasks are at the ceiling.** 88 of 90 runs passed, so pass rates can't separate the arms, which is the same saturation the `bundle-agent-skills` evaluation hit. The difference in passes is one run. Only the cost dimension carries signal.
2. **The extra tokens with official skills come from extra reading, not from the pointers.** Arms B and C read about 75 skill files, A read 42:
   - **`rf-language` was read 16 times in both B and C**, and nothing pointed to it. The official rf-core skill's description ("Use first, before reading .robot/.resource files…") pulls agents in for every task.
   - **Both locator skills were read.** The instructions' routing row still sends UI work to `skill://rf-locators/SKILL.md` while `rf-browser` is served, so B read `rf-locators` 11 times next to `rf-browser` 13 times.
3. **The pointers help when the official skills are present, but not consistently.** Overall, B used 7% fewer tokens than C. For Browser that was 41% fewer (389k against 657k), with C's one failure. For RequestsLibrary, B used 28% more than C. Follow-through was moderate: 16 of 73 skill reads in B were of a pointed skill. In A it was 19 of 42, because the built-in catalogue points to the same `rf-locators` and `rf-requests` that the instructions already name.
4. **The pointers are correct now.** Before the skill-name rule (design D1), a Browser project with RequestsLibrary installed got `rf-requests` and `rf-restinstance` ahead of `rf-browser`. In the full run, every B pointer named the project's own libraries' skills.

## Findings to act on

**In rf-mcp, as follow-ups outside this change:**
- When the official skills are served, the instructions' routing row for UI work should point to the matching official skill, or to `project_skills`, instead of `rf-locators`. Then agents read one locator skill, not two. The next evaluation should check this first, because it explains most of the token gap.
- **Bug found:** `rf_repl` with Browser fails in a fresh project. `New Browser` fails on `results/playwright-log.txt` while the output folder doesn't exist yet, and the next `New Context` hangs until the send times out. Every Browser dry run hit this, in every arm. The full run created `results/` in each project to keep this noise out of the comparison. The fix is to create the output folder when a REPL or debug session starts.

**Upstream, for MarketSquare/robotframework-plugins:**
- `rf-core/rf-language`'s "Use first" description makes agents read it on every task. When rf-mcp serves it, it costs tokens without changing the outcome on these tasks.
- `catalog.yaml` has no per-library skill list (task 1.2). rf-mcp's skill-name rule works around it for the current plugins.

**For the next evaluation:** tasks with headroom, where the built-in-only arm passes well below 1.0, so the pass rates can tell the arms apart.

## Setup notes

- **Turn limit.** The dry runs raised it from 40 to 60 turns. With 40, minimax ran out of turns on the Browser task while still exploring.
- **File tools.** The agent writes files with the harness's own `write_file` / `read_file` tools, as a coding agent would.
- **Isolation.** Each run uses a fresh project whose `.robotframework/rf-mcp.toml` points at a separate evaluation interpreter (robotframework 7.5.1, Browser 20.6.0, RequestsLibrary 0.9.7, robotcode 2.9.0-dev.115). rf-mcp therefore sees the project's real libraries, and the inventory names Browser 20.6.0 as supported by the official `>=20,<21`.
- **Results.** The raw results (`results.jsonl`, transcripts) live in the session's scratch folder and aren't committed.

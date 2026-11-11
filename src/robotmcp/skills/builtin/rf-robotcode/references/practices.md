# Upstream robotcode practices P1-P46 in rf-mcp terms

Source: `robotframework-agent-plugins@next`, `plugins/robotcode/skills/robotcode` (SKILL.md and references). Each practice names where it is covered here, or why it does not apply to the `rf_*` tools.

| # | Practice | Where |
|---|---|---|
| P1 | robotcode instead of robot/rebot/libdoc, from the project environment | config-profiles.md (Environment) |
| P2 | assume installed; check only on "not found" / `No such command` | config-profiles.md (`rf-mcp doctor`) |
| P3 | ask about install scope and extras in one round trip | config-profiles.md (Environment) |
| P4 | read robotcode.io/llms-full.txt instead of guessing | config-profiles.md (Environment) |
| P5 | pick the mode first | SKILL.md (Pick the mode first) |
| P6 | real test involved: debugger, never a REPL copy | SKILL.md rule 3, debugging.md |
| P7 | "why did X fail?": recorded run first | SKILL.md rule 2, triage.md (Workflow B) |
| P8 | REPL instead of a throwaway file | repl-authoring.md |
| P9 | select by longname, never by a single file | SKILL.md rule 1, large-projects.md; enforced by the `rf_run` / `rf_debug` selection warning |
| P10 | longnames from discover; top suite named after `paths` | large-projects.md (Longnames) |
| P11 | other selectors (tags, `-btm`), options before paths | config-profiles.md (Selectors) |
| P12 | global options before the subcommand | config-profiles.md (`root` / `profile` parameters) |
| P13 | no default paths or output options | config-profiles.md |
| P14 | long runs: wait for the exit code, never watch output.xml | triage.md (Workflow A), large-projects.md |
| P15 | exit codes; 252 = run did not start | triage.md (Workflow A), debugging.md |
| P16 | `--output` relative to the output dir, other paths to the root | config-profiles.md, results.md |
| P17 | never read output.xml; use results | results.md |
| P18 | results subcommands and filters | results.md |
| P19 | another run with `-o` after the subcommand | results.md (`output_path`) |
| P20 | `results log --search` for "did X occur?" | results.md |
| P21 | report counts, reasons, report paths | results.md (Reporting), triage.md |
| P22 | debugger start options and break flags | debugging.md (Start) |
| P23 | breakpoint plus run scope | debugging.md (Start) |
| P24 | debug run must not replace the last output | debugging.md; `rf_debug` writes no output files by default |
| P25 | debugger commands step by step | debugging.md (At the stop) |
| P26 | prompt fixes are ephemeral; write them back | debugging.md |
| P27 | sessions when the shell cannot write to a process | not applicable as written: `rf_repl` / `rf_debug` manage `robotcode session` and quote inputs; client mode returns ready-quoted commands |
| P28 | one whole input per send | superseded: a send may hold several lines that run in order and stop at the first failure (contract test `test_contract_multi_line_send_runs_in_order_and_stops_at_first_failure`); blocks still go into one send - repl-authoring.md |
| P29 | send exit codes, status/result/interrupt/restart | debugging.md (Sessions); `status`, `run_exit_code` fields |
| P30 | reuse sessions, idle timeout, stop or announce | SKILL.md rule 6, debugging.md (Sessions) |
| P31 | per-command sandbox: no escapes, no unseen claims | debugging.md (Sessions, client mode) |
| P32 | REPL input is not a .robot file; imports are keywords; `${_}` | repl-authoring.md (Input rules) |
| P33 | REPL dot commands and start flags | repl-authoring.md (Dot commands) |
| P34 | watching: non-headless browser, keep the session | repl-authoring.md (Exploring) |
| P35 | get state, pick a stable id, act, re-check | repl-authoring.md (Exploring), rf-locators skill |
| P36 | discover for inventory, never grep | large-projects.md |
| P37 | doc lookup order | SKILL.md rule 5, repl-authoring.md (Authoring loop) |
| P38 | analyze before running; bitmask; filters; suppression | analyze.md |
| P39 | effective config from profiles show | config-profiles.md (Profiles) |
| P40 | RPA | config-profiles.md (RPA) |
| P41 | Workflow A: run and report | triage.md |
| P42 | Workflow B: one failing test | triage.md |
| P43 | Workflow C: lint changed files | analyze.md (Lint only changed files) |
| P44 | Workflow E: pin, triage, rerun, merge | triage.md |
| P45 | authoring: reuse, prototype, write, analyze, run new tests | repl-authoring.md (Authoring loop) |
| P46 | large projects: filter at the source, aggregate | large-projects.md; `--format json | jq` is not applicable (responses are already projected and size-limited) |

# Tasks

## 1. Response size

- [x] 1.1 Drop `command` from passed server-mode `CommandResult` and `SessionResult` responses; verify `test_server_mode_responses_have_no_duplicates` and `test_discover_tests_response_is_compact` pass
- [x] 1.2 Cap REPL and debug output with a `.doc` tip; verify `test_session_output_is_capped_with_a_doc_tip` passes
- [x] 1.3 Cap merged batch output as a whole with deduplicated notes; verify `test_batched_session_output_is_capped_as_a_whole` passes
- [x] 1.4 Cut a dominant string before shortening lists in `truncate_data`; verify `test_truncate_data_cuts_one_dominant_string_before_lists` passes
- [x] 1.5 Compact `rf_results log` output; verify `test_results_log_is_compact` passes

## 2. Loop guidance and tolerance

- [x] 2.1 Split REPL sends on dot-lines and stop at the first failed part; verify `test_split_repl_batch_isolates_dot_commands_only` and the real-REPL `test_contract_repl_batch_with_dot_commands` pass
- [x] 2.2 Add the batching nudge after 3 one-line sends; verify `test_repl_nudges_batching_after_three_one_line_sends` passes
- [x] 2.3 Add the rf_debug nudge after 3 failing runs of one test; verify `test_failure_streak_points_to_rf_debug_after_three_failing_runs` passes
- [x] 2.4 Add `ArgumentAliases` middleware for `rf_results` result file aliases; verify `test_results_accepts_path_as_output_path` passes and the published schema has no alias
- [x] 2.5 Treat "" for `target` / `shell` / `stop_scope` as unset in the argument middleware (generalized in 2.8); verify `test_empty_choice_arguments_mean_unset` passes
- [x] 2.6 Expand `rf_analyze severity` to that severity and worse (`info` -> `error,warning,info`); verify `test_run_output_collect_unused_and_selection_guard` passes and `robotcode analyze code --severity error,warning,info` reports the s3 errors
- [x] 2.7 Lead `rf_run` responses with a PASS/FAIL `outcome` and add a note to clean `rf_analyze` responses; verify `test_run_outcome_leads_with_the_verdict` and `test_server_mode_responses_have_no_duplicates` pass
- [x] 2.8 Treat empty values (`""`, `"[]"`, `"{}"`, `"null"`, `[]`, `{}`) as unset for every argument and parse Python list text in list arguments; verify `test_python_lists_and_empty_literals_are_tolerated` and `test_empty_choice_arguments_mean_unset` pass

## 3. Codex onboarding

- [x] 3.1 Append the Codex notes (skill sync, commented approval line) to the config from `rf-mcp init --agent codex`; verify `test_codex_config_says_how_to_route_and_approve` passes

## 4. Eval harness Codex driver

- [x] 4.1 Parse `cx-` model names and build the `codex exec` command with sandbox, approval and disabled features; verify `test_codex_model_names_and_config` passes
- [x] 4.2 Add the `mcp+sync` setup and project `.codex/config.toml` with pre-approved tools; verify with `--dry-run` that the seeded project has `.agents/skills` and `.codex/config.toml`
- [x] 4.3 Map Codex `--json` events to graded calls and report input tokens; verify `test_codex_transcript_maps_to_claude_style_calls` passes
- [x] 4.4 Document Codex runs and the 2026-11-07 results in `tests/e2e/agent_eval/README.md`; verify the documented commands match the CLI flags
- [x] 4.5 Run local Ollama models with `model_provider="ollama"` and a window from `OLLAMA_CONTEXT_LENGTH`; verify `test_codex_model_names_and_config` passes and a `cx-oss-qwen3_8b` run reaches rf-mcp
- [x] 4.6 Drop `WindowsApps` from the Codex agent's PATH so the sandbox can start a shell; verify a Codex run executes `powershell.exe` with exit code 0
- [x] 4.7 Accept a t3 defect report by test id named as a defect or bug; verify `test_t3_defect_report_by_test_id_counts` passes and regrade the stored t3 results
- [x] 4.8 Give local models a Codex model catalog entry (direct tools, skills usage instructions, real window, no freeform `apply_patch`); verify `test_codex_oss_catalog_entry` passes and a dry run passes `model_catalog_json`
- [x] 4.9 Run OpenRouter models through Claude Code (`or-`) and Codex (`cx-or-`, `apply_patch` kept), key from env or `.env` for the agent only; verify `test_openrouter_model_names` passes and a smoke run creates a file through each harness
- [x] 4.10 Pass Codex project trust as one inline table so paths with dots stay trusted; verify a `cx-or-openai.gpt-oss-20b` run has no "unrecognized configuration settings" event
- [x] 4.11 Flag `unverified` when a written suite has no passing run after its last write; verify `test_unverified_finish` passes and the 2026-11-09 Qwen authoring runs are flagged

## 5. Integration

- [x] 5.1 Run the full pytest suite and the cookbook tests with the lab env; verify all pass (215 passed, 8 skipped; cookbook 8/8)
- [x] 5.2 Finish the interrupted demoshop sweep (t3 @32k, t6 both windows) and add the results to the README table
- [ ] 5.3 Run the baseline suite with local Ollama models through Codex (`cx-oss-*`) and record which models complete the scenarios
- [x] 5.4 Run the baseline suite with small OpenRouter models through Claude Code and Codex and record pass rates, rf-mcp use and cost
- [x] 5.5 Repeat gpt-6-luna and qwen3-coder (both harnesses, 2 reps) on the `outcome` code with the s5 prompt that asserts only created data; record pass spread and `unverified` counts

## Workflow follow-up

- Archive `restructure-robotcode-mcp` before archiving this change; `mcp-skills-provider` is introduced there.

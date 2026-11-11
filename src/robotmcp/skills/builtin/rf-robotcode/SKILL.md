---
name: rf-robotcode
description: Author, run, analyze, debug and triage Robot Framework tests and tasks with the rf-mcp tools (robotcode >= 2.9). Read first for any Robot Framework work; references/ hold the details per workflow.
---

# Robot Framework with rf-mcp

The `rf_*` tools run `robotcode` in the project's environment. Use them instead of shell commands, file greps or reading `output.xml`.

## Pick the mode first

| Intent | Tool | Details |
|---|---|---|
| What tests, tags, suites, files, profiles exist? | `rf_discover` | [large-projects](references/large-projects.md) |
| Why did X fail? Fix this test. | `rf_results` first, then `rf_debug` | [triage](references/triage.md) |
| A real test is involved and must be inspected live | `rf_debug` - never a REPL copy | [debugging](references/debugging.md) |
| Explore a library, page or API; write a new test | `rf_repl` | [repl-authoring](references/repl-authoring.md) |
| Check files without running | `rf_analyze` | [analyze](references/analyze.md) |
| Keyword names, arguments, docs | `rf_doc`, `.kw`, `.doc` | [repl-authoring](references/repl-authoring.md) |
| Run tests and report | `rf_run`, then `rf_results` | [results](references/results.md) |
| Profiles, robot.toml, environment | `rf_discover action=profile_show` | [config-profiles](references/config-profiles.md) |

Several modes chain: explore, then author, then run, then inspect.

## Rules that save the most turns

1. **Longnames, not files.** Select with `by_longname` taken from `rf_discover`. A single `.robot` file in `paths` becomes the top suite: parent `__init__.robot` setups are skipped and longnames lose their prefix.
2. **Recorded run first.** For "why did X fail?" call `rf_results` (`action="show", failed_only=true`, then `action="log", by_longname=[...]`) before reading files or debugging. Never read or grep `output.xml`.
3. **Debugger for real tests.** If an existing test is involved, `rf_debug` it; a REPL copy runs a misleading imitation.
4. **Analyze before running.** `rf_analyze` after every edit; fix `KeywordNotFound` and argument errors before `rf_run`.
5. **Look keywords up.** `rf_doc action=keywords` (several name patterns at once) or `.kw <text>` in the REPL; never guess names or argument order.
6. **One session, reused.** Start one `rf_repl` session, send several known steps per send, and stop it at the end (or say it is still open).
7. **Every test stands alone.** Each test creates its own data and passes when run alone by longname.
8. **Done means verified.** A task is done only when `rf_analyze` is clean and `rf_run` passes.

## References

- [triage.md](references/triage.md) - failing runs, workflows A, B and E (pinned run, rerun, merge)
- [debugging.md](references/debugging.md) - `rf_debug` commands, breakpoints, output files
- [repl-authoring.md](references/repl-authoring.md) - `rf_repl` input rules, dot commands, authoring loop
- [results.md](references/results.md) - `rf_results` actions and filters
- [analyze.md](references/analyze.md) - `rf_analyze`, exit bitmask, suppressions
- [config-profiles.md](references/config-profiles.md) - environment, global options, robot.toml, RPA
- [large-projects.md](references/large-projects.md) - inventory and aggregation without enumerating
- [practices.md](references/practices.md) - upstream robotcode practices P1-P46 and where each lives here

# Triage: failing tests and runs

## Workflow B - one failing test

1. `rf_results(action="show", failed_only=true)` - which tests failed and why (short messages).
2. `rf_results(action="log", by_longname=["<longname>"], max_depth=2, level="WARN")` - the keyword path to the failure.
3. Optional: `rf_doc(action="keyword", library="<Lib>", pattern="<Keyword>")` to check the failing keyword's arguments.
4. Optional: `rf_debug(action="start", by_longname=["<longname>"])` when the recorded run does not explain it (see debugging.md).
5. Fix the `.robot` / `.resource` / `.py` file, `rf_analyze`, then confirm with `rf_run(by_longname=["<longname>"])`.

Read the recorded run first; do not start with external tools, a REPL copy, or reading project files.

## Workflow A - run and report

1. `rf_run(include_tags=["smoke"], profile="ci")` and wait for it to return (it returns the process exit code; never poll `output.xml`).
2. `rf_results(action="summary", failed_only=true)`.
3. Report counts first, then one line per failure reason, then the `log.html` / `report.html` paths. Do not paste JSON or XML.

Exit codes of `rf_run`: the number of failed tests (capped at 250); `252` means the run did not start (bad options, no tests matched, invalid `robot.toml`) - read `stdout`/`stderr`, because `rf_results` would show the previous run.

## Workflow E - fix a whole failing run

1. Pin the full run: `rf_run(output="full.xml")`. `output` is relative to the output directory.
2. Triage against the pinned file: `rf_results(action="show", failed_only=true, output_path="results/full.xml")`; cluster failures by cause (same message, same keyword).
3. Debug one representative test per cause with `rf_debug` (it writes no output files, so `full.xml` stays).
4. Fix, then `rf_analyze`.
5. `rf_run(rerunfailed="results/full.xml", output="rerun1.xml")`, then `rf_results(action="merge", merge_inputs=["results/full.xml", "results/rerun1.xml"])` and read the merged file with `rf_results(output_path=...)`.

Anti-patterns: debugging test by test without clustering; stepping through the whole run in one paused session; rerunning the whole suite after each fix. Exception - order-dependent failures: `rf_debug(action="start", break_on_failed_test=true, no_break_on_exception=true)` over the whole suite.

## Test defect or product defect?

A test that still fails after its locators and data are right may have found a real defect. Keep the assertion, report the defect with the evidence from `rf_results` (expected vs actual), and do not weaken the check to make it pass.

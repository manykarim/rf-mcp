# Reading results with rf_results

Never read, grep or parse `output.xml` / `output.json` directly. `rf_results` returns structured, filtered data.

| Need | Call |
|---|---|
| Headline counts | `rf_results(action="summary")`; `failed_only=true` adds the failed tests |
| Which tests failed, slowest first | `rf_results(action="show", failed_only=true, sort="elapsed", top=10)` |
| Why one test failed | `rf_results(action="log", by_longname=["<longname>"], max_depth=2, level="WARN")` |
| Did a value occur in the run? | `rf_results(action="log", search="order 4711")` (or `search_regex`) |
| Screenshots of failures | `rf_results(action="log", failed_only=true, extract_dir="artifacts")` |
| Totals by tag or suite | `rf_results(action="stats", by=["tag", "suite"])` |
| What changed since a baseline | `rf_results(action="diff", baseline="baseline/output.xml", only=["new-failures"])` |
| Combine a pinned run with reruns | `rf_results(action="merge", merge_inputs=["results/full.xml", "results/rerun1.xml"])` |

Shared filters: `include_tags` / `exclude_tags`, `by_longname`, `by_test_metadata`, `search` / `search_regex`.

## Another result file

Pass it only as `output_path` (sent as `-o <file>` after the subcommand), relative to the project root: `rf_results(action="show", output_path="results/full.xml")`. A result file in any other parameter is rejected. `rf_run(output="full.xml")` writes relative to the output directory, so the file is read back as `results/full.xml` when `output-dir = "results"`.

## Reporting

Counts first, then one short reason per failure (or per cluster of failures with the same cause), then the `log.html` / `report.html` paths. Do not dump JSON or XML into the answer.

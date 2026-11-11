# Large projects: inventory without enumerating

## Discover, never grep

| Need | Call |
|---|---|
| Environment (Python, Robot Framework, robotcode versions) | `rf_discover(action="info")` |
| Tests / tasks / suites | `rf_discover(action="tests")` (tests only), `action="tasks"`, `action="suites"` |
| Everything at once | `rf_discover(action="all")` |
| Tag -> tests | `rf_discover(action="tags")` |
| Test metadata | `rf_discover(action="metadata", include_values=true)` |
| Files robotcode uses | `rf_discover(action="files")` |
| Find by text | add `search="Open Browser"` (or `search_regex`) |
| A folder outside the configured `paths` | `paths=["tasks/"]` |

Each item comes back as `longname`, `source` (`relative/path.robot:line`) and `tags`; `verbose=true` returns robotcode's full items.

## Longnames

The top suite is named after `paths` (`paths = ["tests"]` gives `Tests.`), not after the project folder. A longname from `rf_results` only holds if that run used the same paths; otherwise a run says `contains no tests after model modifiers` - take the longname from `rf_discover`. A subfolder is selected with `by_longname=["Tests.Sub"]`.

## Filter at the source

- Narrow with `include_tags`, `by_longname`, `search` instead of listing everything.
- Aggregate: `action="tags"` or `rf_results(action="stats", by=["tag"])` instead of reading every test.
- Large responses are cut at a stated limit with a note naming the shortened list; narrow the query rather than paging through it.
- Long runs: give `rf_run` a large `timeout_seconds`; a timeout stops the run with all its processes and says so.

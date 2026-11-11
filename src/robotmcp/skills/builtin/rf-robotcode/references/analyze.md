# Static analysis with rf_analyze

Run `rf_analyze` before running anything new and after every edit. It finds unknown keywords, wrong argument counts, missing imports and undefined variables in seconds, without a run.

## Reading the result

- `data.diagnostics` maps each file to its diagnostics (`range` with 0-based `line` / `character`, `severity` 1 error ... 4 hint, `code`, `message`); `data.summary` counts them.
- `has_errors`, `has_warnings`, `has_infos`, `has_hints` decode robotcode's exit-code bitmask (1 error, 2 warning, 4 info, 8 hint). A non-zero exit code is a finding, not a tool failure.
- Filter with `severity="error"` or `code="KeywordNotFound"` instead of scanning the output.
- `paths=[...]` limits the check to the files you changed; `filter_glob="tests/**/*.robot"` selects by pattern.
- `collect_unused=true` (whole project only) reports unused keywords and variables.

## Fixing findings

| Code | Usual fix |
|---|---|
| `KeywordNotFound` | look the name up with `rf_doc(action="keywords", ...)` or import the library/resource |
| `VariableNotFound` | define it, pass it as an argument, or import the resource that defines it. Variables set at run time (`Set Suite Variable`, `-v`) are invisible to static analysis - confirm with a run before changing code |
| argument errors | `rf_doc(action="keyword", library=..., pattern=...)` for the signature |

## Suppressing a finding on purpose

- In the file: `# robotcode: ignore[CODE]` at the end of a line, before a block, or in column 0 for the rest of the file.
- In `robot.toml`: `[tool.robotcode-analyze.modifiers]` (or the `-mi CODE` option of the CLI).

## Lint only changed files

Pass the changed `.robot` / `.resource` files as `paths` (for example from `git diff --name-only --diff-filter=ACMR HEAD`).

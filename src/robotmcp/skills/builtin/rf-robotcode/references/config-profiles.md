# Environment, configuration and profiles

## Environment

- rf-mcp runs `robotcode` (never bare `robot`, `rebot` or `libdoc`) from the project's Python environment, so project libraries resolve. `rf-mcp init` writes an MCP config that starts the server from that environment; `uvx` / `pipx run` cannot see the project's libraries.
- Assume robotcode is installed and just call the tools. If a tool reports that `robotcode` was not found, or `No such command 'session'`, run `rf-mcp doctor`: it lists the missing subcommands and the install command.
- Before installing anything, ask the user once about scope (dev dependency, e.g. `uv add --dev`, or venv only, e.g. `uv pip install`) and extras. The minimum is `robotcode[runner,analyze,repl]`; `repl` provides repl, session, doc and robot-debug. Never pick silently.
- For CLI details you are unsure of, read https://robotcode.io/llms-full.txt instead of guessing.

## Global options are tool parameters

Every tool takes `root` (`-r`) and `profile` (`-p`), placed before the subcommand for you. Do not add `paths` or output options by default: `robot.toml` usually sets `paths` and `output-dir`. If it sets no `paths`, pass the test folder, and the same folder in every call.

## Selectors

`by_longname` (`-bl`), `include_tags` / `exclude_tags` (`-i` / `-e`, tag patterns such as `smokeANDcritical`), `by_test_metadata` / `exclude_by_test_metadata` (`-btm "Issue:4409"`, RF 7.5+). Robot's own options always come before a path.

## Paths of output files

- `rf_run(output=...)` is relative to the output directory.
- `rf_results(output_path=...)`, `rf_run(rerunfailed=...)` and `merge_inputs` are relative to the project root.

## Profiles

- `rf_discover(action="profiles")` lists profiles; `rf_discover(action="profile_show", profile="ci")` shows the effective settings of one. `action="config"` shows the raw configuration with no profile applied - use `profile_show` to know what a run will do.
- Preview a profile's effect on selection: `rf_discover(action="tests", profile="ci")`.
- To change configuration: look the key up (https://robotcode.io), edit `robot.toml`, then verify with `profile_show`.

## RPA

Read "test" as "task": `rf_discover(action="tasks")`, `rf_run(rpa=true)`. `--rpa` belongs to runs only, never to discovery. Do not mix test files and task files in one run.

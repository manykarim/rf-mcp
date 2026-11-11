# Exploring and authoring with rf_repl

Use `rf_repl` when no existing test is involved: trying a library, mapping a page or an API, prototyping steps for a new test. Never write a throwaway `.robot` file for a "try it" task.

## Input rules

- REPL input is test-body input, not a `.robot` file: no `*** Settings ***` headers, no indentation for plain steps.
- Imports are keyword calls: `Import Library    Browser`, `Import Resource    resources/common.resource`, `Import Variables    vars.py`.
- Cells are separated by at least two spaces (four is the convention). `${_}` holds the last return value.
- One `send` may hold several lines: they run in order and stop at the first failure (pinned by a contract test). Send what you already know works as one batch; send uncertain steps alone.
- A `FOR` / `IF` / `TRY` / `WHILE` / `GROUP` block goes into one send including its `END`.
- `Run Keywords` cannot assign variables (`${r}=` inside it fails) - send the lines instead.
- After a failure, change one thing and retry once; never repeat the same failing input.

## Dot commands

| Command | Use |
|---|---|
| `.kw [name or text]` | search keywords of the imported libraries |
| `.doc <library or resource>` | documentation of a library |
| `.vars [--user]` | variables in scope (`--user` hides Robot internals) |
| `.imports` | loaded libraries and resources |
| `.cwd` | working directory |
| `.save [-t "Name"] [-a] [--keep-failed] file.robot` | write the session as a test (`rf_repl action="save"`) |
| `.debug on`, `.break <Keyword>` | debug a keyword inside the REPL: then run it and use `.where` |

Start options (`action="start"`): `variables` (`-v`), `variable_files` (`-V`), `source` (load a `.robot` / `.resource` file's imports and keywords). `action="restart"` reloads edited Python libraries and resources in a fresh process under the same session id.

## Exploring a page or API

Get state, pick a stable handle, act, re-check:

1. State: `Get Aria Snapshot    css=body` (Browser), `Get Source` (Selenium/Appium), or an API response logged with `expected_status=any`.
2. Pick a stable id, role + name, or field name (see the rf-locators and rf-requests skills).
3. Act with that handle.
4. Re-check the state before the next step.

If the user wants to watch, open the browser with `headless=False` and keep the session open.

## Authoring loop

1. Reuse before writing: `rf_doc(action="keywords", library="resources/common.resource")`, `rf_discover(action="tests", search="...")`, `rf_discover(action="tags")`; match the project's conventions (`rf_discover(action="profile_show")`).
2. Prototype the steps in one `rf_repl` session; look keywords up instead of guessing.
3. Write the test into the suite (or `action="save"` for a first draft; `keep_failed=true` keeps failed steps commented).
4. `rf_analyze` the files you wrote until it reports no errors.
5. Optional: `rf_run(dryrun=true, by_longname=[...])`.
6. Run only the new tests with `rf_run(by_longname=[...])` until they pass. Use `rf_debug` for live fixes.

### Every test creates its own data

A test must pass when run alone (by longname, or as a rerun of failed tests). Never hand ids or records from one test to the next with `Set Suite Variable`. Put creation in a keyword and call it in each test that needs the record; use Suite Setup only for what every test reads (an authenticated session, base data).

The task is done only when `rf_analyze` is clean and `rf_run` passes. Stop the sessions you started, or say which are still open.

# Spec Delta

## Purpose

Keeps rf-mcp tool responses within the context window of small models and steers agents out of loops that
waste calls and tokens, such as one-line REPL sends and edit-and-rerun cycles.

## ADDED Requirements

### Requirement: Passed server-mode calls do not echo the command
In server mode, a call that succeeds (exit code 0, or a session send with status passed) SHALL NOT include the
robotcode argv (`command`) in its response. A failing call SHALL keep it so the agent can see what ran. Client
mode keeps `shell_command`, which is the payload there.

#### Scenario: Passed discover call
- **WHEN** `rf_discover action='tests'` succeeds in server mode
- **THEN** the response has no `command` field and still lists the tests

#### Scenario: Failed run keeps the command
- **WHEN** `rf_run` with an unknown longname fails in server mode
- **THEN** the response includes `command` starting with `robotcode`

### Requirement: Session output is capped
REPL and debug output returned to the agent SHALL be capped (20,000 characters by default) with a note saying
how much was cut. When the cut statement is a `.doc` command, the note SHALL point to `.kw <name>` or
`rf_doc action='keywords'` with name patterns.

#### Scenario: Large .doc output
- **WHEN** `rf_repl send` runs `.doc XML`, whose output exceeds the cap
- **THEN** the returned output is within the cap and the instruction mentions `.kw <name>` and `rf_doc`

### Requirement: Merged batch output is capped as a whole
When several parts of one send are executed separately and their outputs merged, the merged output SHALL be
capped as a whole, and the instruction SHALL carry each part's notes once.

#### Scenario: Batch of large outputs
- **WHEN** a send of three `.doc` lines each produces output near the cap
- **THEN** the merged output is within the cap and the cut note appears once

### Requirement: A dominant string is cut before lists
When structured tool data exceeds the response limit, a single string longer than half the limit SHALL be cut
first; lists SHALL be shortened only if the data is still over the limit. The note SHALL name the cut fields.

#### Scenario: Library doc with one huge markdown field
- **WHEN** a doc response has an 83,000-character `markdown` field and a short `keywords` list, and the limit is 20,000
- **THEN** the response fits the limit, `keywords` is complete, and the note says `cut text: markdown`

### Requirement: Results logs omit noise
`rf_results action='log'` SHALL omit start times, timestamps, `isHtml` when false, NOT RUN branches, and the
absolute `source` when a relative source is present. Each test SHALL keep its own elapsed time and every message.

#### Scenario: Failed test with an untaken IF branch
- **WHEN** the log of a failed test with an `IF $False` branch is requested
- **THEN** the failure message and the taken branch appear, and the untaken branch, timestamps and absolute paths do not

### Requirement: REPL sends with dot-commands are split
`rf_repl send` with several lines SHALL send each line that starts with `.` as its own part and keep
consecutive keyword lines together. Parts SHALL run in order and stop at the first failed part, and the response
SHALL say how many parts were not sent.

#### Scenario: Keyword lines mixed with .kw
- **WHEN** a send contains `Log  a`, `.kw Log` and `Log  b`
- **THEN** three parts run in order and all outputs are returned in one response

### Requirement: Repeated one-line sends get a batching nudge
After 3 passed one-line REPL sends in a row, and then on every 10th, the response instruction SHALL ask the
agent to send the next known steps together, one keyword per line. Dot-commands and multi-line sends SHALL NOT
count, and any multi-line send SHALL reset the count.

#### Scenario: Third one-line send
- **WHEN** an agent sends three passing single keywords in a row
- **THEN** the third response's instruction says to send the next steps together

### Requirement: Repeated failing runs point to rf_debug
In server mode, when the same test fails in 3 `rf_run` calls in a row, and then on every 3rd, the response
instruction SHALL name the test and point to `rf_debug action='start'` to inspect the failure. A passing run or
a different failure SHALL reset that test's count.

#### Scenario: Third failing rerun
- **WHEN** `Order Total` fails in three consecutive `rf_run` calls
- **THEN** the third response's instruction says `Order Total failed in 3 runs in a row` and mentions `rf_debug`

### Requirement: Result file aliases
`rf_results` SHALL accept `path`, `result_file` and `output_file` as aliases of `output_path` when
`output_path` is not given. The published schema SHALL NOT list the aliases.

#### Scenario: Weak model passes path
- **WHEN** `rf_results action='summary'` is called with `path='results/saucedemo.xml'`
- **THEN** the call reads that file as if `output_path` had been given

### Requirement: Empty arguments mean unset
An argument sent as `""`, `"[]"`, `"{}"`, `"null"`, `[]` or `{}` SHALL be treated as if it were not given, for every
tool, instead of failing validation or reaching robotcode as text.

#### Scenario: Model sends every optional argument
- **WHEN** `rf_analyze` is called with `target=""` and `shell=""`
- **THEN** the call runs in the configured mode instead of failing with "Input should be 'powershell' or 'bash'"

#### Scenario: Empty literals on rf_run
- **WHEN** `rf_run` is called with `rerunfailed="[]"`, `profile="[]"`, `files=""` and `output=[]`
- **THEN** the command has no `--rerunfailed`, no `-p` and no `[]`, and the call does not fail validation

### Requirement: List arguments accept Python list text
A list argument sent as Python list text, such as `"['Tests.A', 'Tests.B']"`, SHALL be parsed into its items, as a
JSON array string already is.

#### Scenario: Stringified by_longname
- **WHEN** `rf_run` is called with `by_longname="['A.A', 'A.B']"`
- **THEN** the command has `-bl A.A` and `-bl A.B`

### Requirement: Analyze severity includes worse severities
`rf_analyze` with `severity` SHALL report diagnostics of that severity and every worse one, as its description
says; robotcode's `--severity` reports only the listed severities, so the tool SHALL pass the expanded list.

#### Scenario: Severity info on a file with errors
- **WHEN** `rf_analyze severity='info'` runs on a project with a `KeywordNotFound` error
- **THEN** robotcode is called with `--severity error,warning,info` and the error is reported

### Requirement: Run results lead with a verdict
In server mode, a finished `rf_run` response SHALL start with an `outcome` line: `PASS: <counts>` when robot exited
with 0 and reported no failures, `PASS (dry run: ...)` for a dry run, and otherwise `FAIL: <counts>` with an
instruction to fix and rerun before reporting the task as done, or a FAIL line saying no test ran. Timeouts and
client mode SHALL have no `outcome`.

#### Scenario: Failing run
- **WHEN** `rf_run` runs a suite where 1 of 2 tests fails
- **THEN** the first key of the response is `outcome` with `FAIL: 2 tests, 1 passed, 1 failed. The tests do not pass yet: ...`

### Requirement: Clean analysis says it ran nothing
A server-mode `rf_analyze` response without errors SHALL add a `note` that analysis runs no test and that written
or changed tests are verified with `rf_run`.

#### Scenario: Clean analysis after writing a suite
- **WHEN** `rf_analyze` reports no errors
- **THEN** the response has a `note` that names `rf_run`

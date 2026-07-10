# Proposal: build-suite-safe-persist

## Why

The 2026-07-10 Docker capability experiment (MiniMax-M3 → opencode → robotmcp)
produced a **corrupted generated suite** on disk in the file-processing scenario:
`experiments/file-proc/fileproc/suite.robot` had raw line breaks inside a
`Create File` argument and rendered the *resolved* file content as an assignment
target (`line1⏎line2⏎line3⏎line4 =  Get File …`) instead of `${file_content}`.
It was initially filed as an rf-mcp `build_test_suite` **rendering** defect (F1).

Direct reproduction through the real execute path **refutes that diagnosis**:
`build_test_suite` renders correctly — it escapes newlines (`line1\nline2\nline3\n`),
emits `${file_content} =` as the assignment, and preserves `${file_content}` in
arguments. The corruption came from the **persistence step**: the agent saved the
generated `rf_text` to disk with the Robot Framework `Create File` keyword, and
RF **resolved `${file_content}` to its runtime value and interpreted the `\n`
escapes into real newlines *inside the argument***. That is expected RF behaviour
(confirmed: `Create File  out  "x ${v}\ny"` writes `x <value>` + a real newline).

So the real gap is not rendering — it is that `build_test_suite` **only returns
text and offers no safe way to write it to disk**, so agents reach for
`Create File` (and my own scenario prompt told them to), which mangles any suite
containing `${…}` or escape sequences.

## What Changes

- `build_test_suite` / `TestBuilder.build_suite` gain an optional `output_path`
  parameter. When set, the generated `.robot` text is written to that path via
  **plain UTF-8 file I/O** (parent dirs created), preserving it byte-for-byte
  with no RF variable resolution or escape expansion. The response reports
  `output_path` + `output_bytes` (or `output_error` on a write failure — the
  build still succeeds).
- The tool docstring instructs agents to persist via `output_path` and **never**
  to write `rf_text` through the `Create File` keyword, explaining the corruption.

## Capabilities

### New Capabilities

- `build-suite-safe-persist`: `build_test_suite` can persist the generated suite
  to disk itself, byte-for-byte, without Robot Framework resolving `${variables}`
  or expanding escapes in the content.

### Modified Capabilities

(none — additive optional parameter; existing text-returning behaviour unchanged)

## Impact

- `src/robotmcp/components/test_builder.py` — `build_suite(output_path=...)`
  writes `rf_text` via `open(...).write()` after generation; adds
  `output_path`/`output_bytes`/`output_error` to the result.
- `src/robotmcp/server.py` — `build_test_suite(output_path="")` passes it through;
  docstring documents the safe-persistence contract and the Create-File trap.
- Tests: `tests/unit/test_build_suite_persistence.py` — (1) `rf_text` is correct
  (escaped newlines + preserved `${var}`), refuting the F1 rendering framing;
  (2) `output_path` persists byte-for-byte AND the file parses via
  `robot.api.TestSuiteBuilder`; (3) the `Create File` round-trip corrupts,
  documenting the root cause.

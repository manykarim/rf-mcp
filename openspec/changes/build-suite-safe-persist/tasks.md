# Tasks: build-suite-safe-persist

## 1. Diagnosis
- [x] 1.1 Reproduce the file-proc pattern through the real execute path; confirm `build_test_suite` `rf_text` is correct (escaped `\n`, `${file_content}` preserved) — F1 rendering framing refuted
- [x] 1.2 Confirm the corruption source: RF `Create File  out  "…${v}…\n…"` resolves the var and expands the escape into the written file

## 2. Safe persistence
- [x] 2.1 `TestBuilder.build_suite(output_path="")`: after generating `rf_text`, write it to `output_path` via plain UTF-8 I/O (create parent dirs); add `output_path`/`output_bytes`; soft-fail to `output_error` (build still succeeds)
- [x] 2.2 `server.build_test_suite(output_path="")`: pass through; docstring directs agents to persist via `output_path` and NEVER via `Create File`, explaining the corruption

## 3. Tests
- [x] 3.1 `test_rf_text_is_correct_not_corrupted`: generated text escapes newlines and preserves `${file_content}` in both the assignment and the arg
- [x] 3.2 `test_output_path_persists_byte_for_byte_and_parses`: file == `rf_text`, `${var}` unresolved, escapes intact, and `robot.api.TestSuiteBuilder` parses it
- [x] 3.3 `test_create_file_roundtrip_corrupts_documents_root_cause`: `Create File` round-trip resolves the var + expands the escape (the documented anti-pattern)

## 4. Validation
- [x] 4.1 Persistence tests green (3 passed)
- [x] 4.2 Full unit suite green (no regressions) — 6853 passed + 1 skipped

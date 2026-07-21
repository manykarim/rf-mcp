## 1. Windows drive-letter path normalization (the confirmed-bug fix)

- [x] 1.1 In `TestBuilder._escape_robot_argument`, after the namespace fix and before the control-char/dash-guard escaping, rewrite a value matching `^[A-Za-z]:[\\/]` from `\` to `/`. Forward slashes are valid on Windows (Start Process/subprocess + OperatingSystem keywords) and POSIX and are RF-escape-safe, so the path survives parsing and runs on the current OS.
- [x] 1.2 Value-shape guard only — flags (`/w`), URLs, variable references (`${…}`), inline evals (`${{…}}`), regexes and UNC/relative paths do not match and are untouched.

## 2. Blanket backslash escaping — evaluated and DEFERRED

- [x] 2.1 Prototyped doubling every literal backslash to also fix regexes/relative/UNC; REVERTED because it is not idempotent with the function's existing contract (re-corrupts the dash-guard `\=` and intentional `\n`, breaking `test_already_escaped_dash_arg_untouched` and `test_already_escaped_backslash_n_not_doubled`). Deferred to an escape-aware follow-up (design.md "Deferred").

## 3. Tests

- [x] 3.1 Round-trip: `C:\WINDOWS\system32\calc.exe` and `C:\Users\name\report.txt` → forward-slash form that RF parses unchanged; lowercase drive letter; already-forward-slash path unchanged (`tests/unit/test_suite_path_escaping.py`).
- [x] 3.2 Narrowness: flags/URLs/variable/inline-eval refs are NOT separator-rewritten.
- [x] 3.3 Regression guards: dash-guard `\=` idempotency and real control-char escaping unchanged. Full unit suite green (7102 passed) — the two idempotency tests that a blanket rule would have broken still pass.

## 4. Validation

- [x] 4.1 `openspec validate fix-suite-path-escaping --strict` passes.

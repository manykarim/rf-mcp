## 1. Escape-aware backslash doubling

- [x] 1.1 Added `_escape_aware_backslashes` + call in `TestBuilder._escape_robot_argument` (after drive-letter normalization, before control-char escaping): double a `\` unless the next char is an RF-recognized/intended escape — PRESERVE `n r t x u U` and `= # $ { } @ & %` and space, keep an existing `\\` pair; double everything else (`\d \w \s \W \.` …) and a trailing lone `\`.
- [x] 1.2 Idempotent (already-escaped input unchanged: `\\d`→`\\d`, `\=`→`\=`, `\n` text→`\n`); `${…}`/`${{…}}` untouched (markers have no backslash).

## 2. Tests

- [x] 2.1 Round-trip (`tests/unit/test_suite_arg_escape_roundtrip.py`): `\d+`, `\w\s`, `\W`, `\D\S`, `\b`, `data\output.txt`, `logs\app.log`, `${dir}\file.txt` recover the original; `\${x}` stays a literal `${x}`.
- [x] 2.2 Regression/idempotency: `\=` (dash-guard) and `\n` text preserved; real control chars still escaped; drive-letter path still → forward slashes; URLs/flags/vars unchanged; the helper pass is idempotent + targeted.
- [x] 2.3 Full unit suite green (7111 passed).

## 3. Validation

- [x] 3.1 `openspec validate fix-suite-arg-escape-roundtrip --strict` passes.

## Residual (documented)

A relative path whose separator is immediately followed by `n`/`r`/`t` (`data\new`,
`data\report`) is still ambiguous — RF's `\n`/`\r`/`\t` are real escapes, so they are
preserved (→ control char), not doubled. Use forward slashes or an absolute drive-letter
path for those.

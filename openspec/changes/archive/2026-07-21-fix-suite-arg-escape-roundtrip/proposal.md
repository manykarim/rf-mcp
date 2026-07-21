## Why

The previous change (`fix-suite-path-escaping`) fixed Windows drive-letter paths in
generated suites by normalizing them to forward slashes, but explicitly deferred the
*broader* backslash round-trip. RF drops the backslash for every "unrecognized" escape
(only `\n`/`\r`/`\t` are recognized), so any other backslash literal written into a
generated `.robot` is still silently corrupted on parse:

- `Should Match Regexp    ...    \d+` → RF parses `d+` (regex broken)
- `\w`, `\s`, `\W`, `\.` etc. → the backslash is dropped
- relative Windows path `data\output` → `dataoutput`

A blanket "double every backslash" was rejected earlier because it re-corrupts the
dash-guard's `\=` and intentional `\n` (not idempotent with `_escape_robot_argument`'s
existing contract, which also runs on already-escaped values). The correct fix is
**escape-aware**: double a literal backslash only where RF would drop it, while preserving
RF's real escapes and the intentional syntax escapes — which is idempotent.

## What Changes

- **Escape-aware backslash doubling in `_escape_robot_argument`.** Double a `\` only when
  the following character is one RF would treat as an *unrecognized* escape (dropping the
  backslash and corrupting the value: `\d \w \s \W \. …`). PRESERVE, unchanged: RF's real
  control escapes (`\n \r \t`), hex/unicode (`\x \u \U`), an already-doubled `\\`, and the
  syntax escapes rf-mcp/agents rely on (`\=` dash-guard, `\#` comment-guard, `\$ \{ \} \@
  \& \%` variable-syntax, `\ ` space). This is idempotent (already-escaped input is left
  as-is) and preserves variable references (markers contain no backslash).
- Ordered after the existing drive-letter forward-slash normalization (a normalized path
  has no backslashes left) and before the control-char escaping.
- **Tests:** regex `\d+`/`\w`/`\s` and a relative `data\output.txt` round-trip; the existing
  idempotency guards (`\=`, `\n` text) and real control-char escaping stay green.

## Capabilities

### Modified Capabilities
- `suite-argument-escaping`: extends the escaping guarantees — generated-suite argument
  values with backslashes round-trip through RF parsing (escape-aware, idempotent),
  covering regexes and relative paths, not just drive-letter paths.

## Impact

- **Code:** `src/robotmcp/components/test_builder.py` (`_escape_robot_argument`, one
  escape-aware pass). No signature/tool changes.
- **Behaviour:** generated suites with regex arguments or relative Windows paths now run
  correctly; drive-letter paths (previous change), live `execute_step`, and the
  `output_path` byte-write are unchanged.
- **Residual (documented):** a relative path whose separator is immediately followed by
  `n`/`r`/`t` (`data\new`) is still ambiguous (RF's `\n` is a real escape) — rare; use
  forward slashes or an absolute drive-letter path.

## Why

`build_test_suite` corrupts literal backslashes in generated `.robot` files. Robot
Framework treats `\` as an escape character, but rf-mcp's `_escape_robot_argument`
escapes actual `\n`/`\r`/`\t` **characters** and never doubles literal backslashes. So a
recorded value with backslashes is written raw and RF re-parses it to the wrong thing —
observed in a real generated suite:

- `C:\WINDOWS\system32\calc.exe` → RF re-parse → `C:WINDOWSsystem32calc.exe` (separators eaten)
- `C:\Users\name\report.txt` → embeds a real newline (`\n`) + CR (`\r`)
- `Should Match Regexp    ...    \d+` → `d+` (regex broken too)

The fix (RF's own `escape()` semantics) is to double literal backslashes so the value
round-trips. Separately, the natural follow-on is a **platform-independent** path form so
generated suites run on the OS that executes them regardless of the separator the agent
supplied (Windows accepts forward slashes for `Start Process`/subprocess and the
OperatingSystem keywords). Live `execute_step` is unaffected (args go straight to the
keyword API — no RF parsing).

## What Changes

- **Normalize Windows drive-letter paths to forward slashes** in `_escape_robot_argument`:
  a value matching the drive-letter shape (`X:\…` or `X:/…`) is rewritten to forward
  slashes — the one representation that is both **RF-escape-safe** and **valid on the
  current OS** (Windows accepts `/` for `Start Process`/subprocess and the OperatingSystem
  keywords; POSIX too). This is value-shape based (no keyword map needed) and directly
  fixes the observed corruption without any change to the function's existing escape
  contract. Flags (`/w`), URLs (`https://…`), variable references (`${…}`), inline evals
  (`${{…}}`), regexes and UNC/relative paths do NOT match the shape and are left unchanged.
- **Round-trip tests** for the drive-letter case + regression guards that non-path values,
  the dash-guard `\=` idempotency, and real control-char escaping are all unchanged.
- **Deferred (design.md):** the *broader* backslash round-trip — regexes (`\d+`→`d+`),
  relative Windows paths, UNC — is NOT included. Blanket backslash-doubling is not
  idempotent with this function's existing contract (it re-corrupts the dash-guard's `\=`
  and intentional `\n`, breaking two existing replay tests). A correct fix needs
  escape-aware, not blanket, handling; scoped out to keep this change regression-free.

## Capabilities

### New Capabilities
- `suite-argument-escaping`: how generated `.robot` argument values are made RF-safe —
  starting with platform-independent forward-slash normalization of Windows drive-letter
  paths so they survive RF parsing and run on the current OS.

## Impact

- **Code:** `src/robotmcp/components/test_builder.py` (`_escape_robot_argument`, one
  value-shape guard). No signature/tool changes.
- **Behaviour:** generated suites with Windows drive-letter paths now run correctly (`C:\…`
  → `C:/…`). Everything else — live `execute_step`, the `output_path` byte-write, and every
  non-drive-letter argument — is unchanged (full unit suite 7102 passed).
- **Scope note:** the broader backslash round-trip for regex/relative/UNC values is a
  deliberate follow-up (see design.md), because the safe fix there is escape-aware and
  larger than this focused, confirmed-bug fix.

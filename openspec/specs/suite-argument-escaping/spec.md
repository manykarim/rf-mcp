# suite-argument-escaping Specification

## Purpose
TBD - created by archiving change fix-suite-path-escaping. Update Purpose after archive.
## Requirements
### Requirement: Windows drive-letter paths are written in a platform-independent form

A generated-suite argument that is recognizably a Windows drive-letter filesystem path
(`X:\…` or `X:/…`) SHALL be normalized to forward slashes, because Robot Framework treats
`\` as an escape character and a literal backslash path is otherwise corrupted on parse
(`C:\WINDOWS\system32` → `C:WINDOWSsystem32`; `C:\Users\name` → an embedded newline). Forward
slashes are valid for `Start Process`/subprocess and the OperatingSystem keywords on Windows
and POSIX and are RF-escape-safe, so the value survives parsing and runs on the OS that
executes the suite regardless of the separator the agent supplied.

Values that are not the drive-letter path shape — flags (`/w`), URLs (`https://…`), variable
references (`${…}`), inline evaluations (`${{…}}`), regexes, UNC paths (`\\srv`) and relative
paths — MUST NOT be separator-rewritten by this rule; their existing serialization is
unchanged.

#### Scenario: a Windows drive-letter path round-trips and runs on Windows

- **WHEN** a step argument `C:\WINDOWS\system32\calc.exe` is written to a generated suite
- **THEN** the suite carries `C:/WINDOWS/system32/calc.exe`, which RF parses unchanged and which `Start Process`/subprocess accepts on Windows (no dropped separators, no embedded control characters)

#### Scenario: non-path values are left unchanged

- **WHEN** an argument is a flag (`/w`), a URL (`https://example.com/a`), or a variable reference (`${dir}`)
- **THEN** it is written with its separators unchanged (the drive-letter normalization does not apply)

### Requirement: Backslash literals round-trip through RF parsing (escape-aware)

A generated-suite argument value that contains literal backslashes SHALL be escaped so
Robot Framework's parser recovers the recorded value, because RF drops the backslash for
every escape it does not recognize (`\d`→`d`, `\W`→`W`) and only `\n`/`\r`/`\t` (and
hex/unicode) are real escapes. A backslash MUST be doubled when the following character
makes it an unrecognized escape, and MUST be left single when the following character is an
RF-recognized escape (`\n \r \t`, `\x \u \U`) or an intended syntax escape the tool relies
on (`\=`, `\#`, `\$`, `\{`, `\}`, `\@`, `\&`, `\%`, `\ `), or another backslash (`\\`). The
transformation MUST be idempotent (an already-escaped value is unchanged) and MUST NOT alter
variable references (`${…}`, `@{…}`, `&{…}`, `%{…}`, `${{…}}`).

#### Scenario: a regex argument round-trips

- **WHEN** a step records `\d+` (or `\w`, `\s`, `\W`) as an argument and the suite is generated and parsed
- **THEN** RF passes the original regex to the keyword (`\d+`, not `d+`)

#### Scenario: a relative Windows path round-trips

- **WHEN** a step records `data\output.txt` as an argument
- **THEN** RF passes `data\output.txt` (the backslash preserved), not `dataoutput.txt`

#### Scenario: the transformation is idempotent and preserves intended escapes

- **WHEN** an already-escaped value is serialized again — a dash-guarded `-env:X\=Y`, an `\n` text sequence, or a `${dir}\file` reference
- **THEN** the `\=` and `\n` are left unchanged (no double-escaping), the `${dir}` reference is preserved, and applying the escaping twice yields the same text


## ADDED Requirements

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

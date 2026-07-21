## ADDED Requirements

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

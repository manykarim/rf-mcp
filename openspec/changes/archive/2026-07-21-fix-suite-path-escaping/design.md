## Context

`build_test_suite` serializes recorded steps to `.robot` text. Argument values pass through
`TestBuilder._escape_robot_argument`, which currently:
1. coerces non-strings to str,
2. fixes malformed evaluation-namespace expressions,
3. replaces real `\n`/`\r`/`\t` **characters** with their `\n`/`\r`/`\t` text form,
4. escapes a leading `-...=` (named-arg misparse guard).

It never doubles literal backslashes. RF's parser treats `\` as an escape char, so a raw
`C:\WINDOWS` in the text loses its separators (`\W`→`W`) and `C:\name` embeds a newline
(`\n`). Proven with `robot.utils.escaping.unescape` and observed in a real suite. Regexes
break the same way (`\d+`→`d+`). Live `execute_step` is unaffected — its args go to the
keyword API directly, never through RF suite parsing.

## Goals / Non-Goals

**Goals**
- Any recorded argument value round-trips through `build_test_suite` → RF parse unchanged
  (paths, regexes, arbitrary backslashes), while variable references still work.
- Known path arguments come out in a form valid on the OS that runs the suite, independent
  of the separator the agent supplied.

**Non-Goals**
- Converting a path's *meaning* across OSes (`C:\…` has no Linux equivalent) — only
  separators/escaping.
- Normalizing ambiguous slots (`Start Process` `arguments`/shell command lines) or
  non-filesystem strings (URLs).
- Any change to live `execute_step` or the `output_path` byte-for-byte write.

## Decisions

1. **Normalize Windows drive-letter paths to forward slashes (the shipped fix).** In
   `_escape_robot_argument`, a value matching `^[A-Za-z]:[\\/]` has its `\` replaced with
   `/`, right after the namespace fix and before the existing control-char/dash-guard
   escaping. *Rationale:* forward slashes are valid on Windows (subprocess + OperatingSystem
   keywords) and POSIX and are RF-escape-safe, so one form runs everywhere and — crucially —
   there are no backslashes left for RF to corrupt. Value-shape based, so it needs no
   keyword map: the drive-letter shape is unambiguous and excludes flags (`/w`), URLs
   (`https://…`), regexes (`\d+`), UNC (`\\srv`) and variable references. *Alternative
   rejected:* `os.path.normpath` to native current-OS format — it collapses `.`/`..`,
   mangles URLs (`https://` → `https:/`), and native backslashes would still need escaping.

2. **Blanket backslash-doubling was tried and REVERTED.** The first cut doubled every
   literal `\` (RF `escape()` semantics) to also fix regexes/relative/UNC paths, but it is
   **not idempotent with this function's existing contract**: `_escape_robot_argument` is
   called on already-escaped values in the replay path, and doubling re-corrupts the
   dash-guard's `\=` (→ `\\=`) and turns an intentional `\n` into a literal — breaking
   `test_already_escaped_dash_arg_untouched` and `test_already_escaped_backslash_n_not_doubled`.
   Since the drive-letter normalization already fixes the *confirmed* bug with zero
   regressions, the broader escaping is deferred rather than shipped with a
   contract-breaking blanket rule.

## Deferred (follow-up): escape-aware round-trip for non-drive-letter backslash values

Regexes (`\d+` → `d+`), relative Windows paths (`data\file`) and UNC (`\\srv`) still lose
their backslashes on RF parse. A correct fix must be **escape-aware, not blanket**: double a
literal `\` only where RF would drop it (an "unrecognized escape"), while preserving RF's
real escapes (`\n \r \t \x \u \U`) and the intentional syntax escapes rf-mcp/agents rely on
(`\=`, `\#`, `\ `). It must also stay idempotent (skip already-escaped input) and preserve
variable references. That is a larger, separately-testable change; out of scope here.

## Risks / Trade-offs

- **A non-path value that happens to be drive-letter-shaped** (e.g. a literal `C:\x` passed
  to `Log`) is swapped to `C:/x` — still an equivalent path string, harmless.
- **UNC / relative / regex values remain unfixed** — a known, documented limitation (see
  Deferred), not a regression: their pre-existing (still-imperfect) serialization is
  unchanged.

## Migration Plan

1. Add the drive-letter forward-slash guard to `_escape_robot_argument`; add round-trip +
   regression-preservation tests.
2. Run the full suite (existing `_escape_robot_argument` callers are the regression surface).
   Rollback = revert the one guard (purely additive).

## Open Questions

- The deferred escape-aware round-trip: implement now as a follow-up change, or wait for a
  reported regex/relative-path corruption? (The regex case is real — worth a follow-up.)
- Should a `Start Process` command that is a *bare exe name* (`calc.exe`, no separators) get
  any treatment? (No — nothing to normalize; left as-is.)

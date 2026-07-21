## Context

`TestBuilder._escape_robot_argument` serializes recorded argument values into `.robot` text.
RF's parser treats `\` as an escape char and, empirically, recognizes only `\n`/`\r`/`\t`
(control) and `\xHH`/`\uHHHH`/`\UHHHHHHHH` (hex/unicode); for every other `\c` it drops the
backslash (`\d`→`d`, `\W`→`W`, `\=`→`=`). So an un-escaped literal backslash is corrupted on
parse. The prior change fixed drive-letter paths (→ forward slashes); this change fixes the
rest, escape-aware.

The function is also called on already-escaped values in the replay path, so the fix must be
**idempotent** (it must not re-escape `\=`/`\n`), and it must preserve variable references.

## Goals / Non-Goals

**Goals**
- A recorded value with backslashes round-trips through `build_test_suite` → RF parse
  unchanged (regex `\d+`, relative `data\output.txt`, `${dir}\file`), idempotently.
- Do not regress the existing escape contract (`\=` dash-guard, `\n` text, real control
  chars, variable/inline-eval refs).

**Non-Goals**
- Fixing a relative path whose separator is followed by `n`/`r`/`t` (genuinely ambiguous
  with RF's real `\n`/`\r`/`\t`) — documented residual; use `/` or a drive letter.
- Any change to live `execute_step` or the `output_path` byte-write.

## Decisions

1. **Double a backslash only for RF-unrecognized escapes; preserve the rest.** Walk the
   string; at a `\`, look at the next char:
   - next is `\` → keep the `\\` pair verbatim (idempotent), advance 2.
   - next ∈ PRESERVE → keep a single `\` (do not double).
   - otherwise (unrecognized escape, or trailing lone `\`) → emit `\\`.
   PRESERVE = `n r t` (RF control) ∪ `x u U` (hex/unicode) ∪ `= # $ { } @ & %` and space
   (RF/rf-mcp syntax escapes). *Rationale:* RF drops the backslash for exactly the
   non-PRESERVE chars, so doubling those is what makes the literal survive; PRESERVE keeps
   real escapes and the dash-guard's `\=` intact → idempotent. Verified empirically that
   `\n \r \t` are the only recognized single-char escapes.
   *Alternative rejected:* blanket doubling (not idempotent — the reverted first cut of the
   prior change); `robot.utils.escaping.escape()` (also escapes `$`/`#`, breaking `${var}`
   and comments).

2. **No separate variable-reference splitting needed.** `${…}`/`@{…}`/`&{…}`/`%{…}` markers
   contain no backslash, so the walk leaves them untouched; an intentionally-escaped
   `\${x}` is preserved because `$` ∈ PRESERVE.

3. **Order:** namespace-fix → drive-letter forward-slash (prior change) → escape-aware
   doubling (this change) → control-char escape (real newline→`\n` text) → dash-guard →
   `#` guard. A drive-letter path has no backslashes left for this step; a real control char
   is not a backslash so the escape-aware step ignores it and the control-char step handles
   it.

## Risks / Trade-offs

- **A literal `\=`/`\#`/`\$` in a raw value is preserved (backslash later dropped by RF)**
  → e.g. `C:\=x` → `C:=x`. Rare; the syntax-escape preservation is required for idempotency
  with the dash-guard, and these characters after a backslash are almost always intentional
  escapes, not path/literal content.
- **Relative path with `\n`/`\r`/`\t`** (`data\new`) stays corrupted → documented residual;
  drive-letter and forward-slash paths are unaffected.
- **Hex/unicode preserved** (`\x41` stays a hex escape) → correct for intended escapes; a
  path containing `\x…` is vanishingly rare.

## Migration Plan

1. Add the escape-aware pass to `_escape_robot_argument`; add regex/relative round-trip
   tests + keep the existing idempotency/regression tests green.
2. Full unit suite (the existing `_escape_robot_argument` callers are the regression
   surface). Rollback = revert the one pass (purely additive within the function).

## Open Questions

- Extend PRESERVE handling so a valid `\xHH` is preserved but a bare `\x` (no hex digits) is
  doubled? (Not now — bare `\x` in test args is negligible.)

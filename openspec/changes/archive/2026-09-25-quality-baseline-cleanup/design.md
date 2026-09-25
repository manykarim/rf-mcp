# Design: quality-baseline-cleanup

## Why these three bugs survived 7,300 passing tests

All three sit on paths a green test run never takes:

```
  suite_execution_service.py:28   runs only if Robot Framework is UNIMPORTABLE
                                  - never true in a working environment

  aggregates.py:150               runs only on the snapshot COMPRESSION path
                                  - reached with large ARIA trees

  rf_native_context_manager:975   runs only on the BuiltIn FALLBACK
                                  - reached when the primary path already failed
```

That is the pattern worth noticing: **every one is on an error or degraded path**. Tests
overwhelmingly cover the happy path, so the code that runs when things go wrong is the
least exercised and the most likely to carry a latent `NameError`. Static analysis finds
these precisely because it does not need to execute them - which is the argument for the
`Quality` job existing at all.

It also means fixing them is not enough. Each needs a test that *executes* the path, or
the same class of defect returns the next time someone edits a handler.

## Fixing them without changing behaviour

| defect | fix | risk |
|---|---|---|
| `logger` used before definition | move the `logger = logging.getLogger(__name__)` assignment above the `try` block | none - module-level, no ordering dependency |
| `new_snapshot` self-reference | compute the token estimate into a local before constructing, or call the estimator on the instance afterwards | low - must preserve the value, not just silence the name |
| `_normalize_arg` out of scope | promote the nested helper to module scope, or re-define it at the call site | low - the nested copy at line 596 must keep working |

The second one deserves care: the naive fix is to drop the field, which would silence the
linter and change the output. The estimate has to be preserved.

## Annotation-only F821s are not the same problem

Nine `Any`, plus `FrontendConfig`, `ExecutionSession`, `TimeoutPolicy`, are names used in
annotations but never imported. With `from __future__ import annotations` these are never
evaluated, so they do not crash - they are a reporting problem, not a runtime one.

They are still worth clearing, for one reason: while they sit in the report, `F821` reads
as "mostly harmless annotation noise", and the three real crashes are camouflaged among
them. A rule that mixes crashes with cosmetics gets skimmed. Import them properly (under
`TYPE_CHECKING` where the import would be circular or costly) so any future `F821` means
what it says.

## Why the S rules are excluded

160 `try-except-pass` is the largest single finding group, and in this codebase most are
deliberate: rf-mcp swallows exceptions on best-effort paths - display probes, optional
library detection, cleanup - by design. Mechanically "fixing" them would either add noise
logging to hot paths or change error semantics.

They need triage, not a sweep, and triage is a different kind of work from this change.
Bundling them would put a hundred judgement calls into a diff that is otherwise
mechanical, and the three real bugs would be lost in it.

## Ordering

Fix the bugs and add their tests FIRST, before the auto-fix sweep. A `ruff --fix` pass
touches many files; landing it first would bury three genuinely important diffs in a
mechanical changeset and make review of the part that matters harder.

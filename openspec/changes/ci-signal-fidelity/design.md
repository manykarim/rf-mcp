## Context

See proposal.md - Why. The mechanics, measured on this repository on 2026-09-24:

```
quality.sarif:  295 results, every one level="error"
                 18 rules,   every one properties["problem.severity"]="error"
                 no rule carries defaultConfiguration
```

Ruff's SARIF therefore carries **no severity information at all** - the level is a
constant. GitHub code scanning reads that constant, raises every finding as an
error-level alert, and fails a check named `ruff` when a pull request touches a line
carrying one. That is how PR #93 got a red `ruff` check for two `S110`/`S112` findings
it did not introduce: it *relocated* a `try/except: pass` (promoting a nested helper to
module scope) and dropped an unused `except ... as e` binding. Project-wide S110+S112
stayed at 182 across both commits.

The repository already solves this exact problem once. The security digest cannot use
the SARIF level either, so it derives a tier from an explicit rule set, validated
against bandit:

```
HIGH = {"S102","S301","S302","S306","S307","S324","S602","S604","S605","S606"}
```

Applying that same set to `quality.sarif` yields **6 error / 289 warning** - and 6 is
exactly bandit's HIGH count for this source (all S324). The set is already justified;
it is simply not applied on the path that feeds code scanning.

## Goals / Non-Goals

**Goals:**
- One severity definition, used by both the digest and the SARIF upload.
- No failed check from a finding, for every check the analysis produces.
- A ratio assertion fails only on a measurement large enough to mean something.

**Non-Goals:**
- Reducing the 295 findings. Backlog size is a separate concern from signal fidelity.
- Suppressing individual findings with `noqa`. That hides the finding; this change
  changes only the severity it is *published* at. All 295 stay in the report, the
  artifact and the digest.
- Re-tuning benchmark thresholds to be more permissive. The limits stay; only the
  precondition for asserting them changes.

## Decisions

### D1. Map severity in a step between analysis and upload, not by configuring ruff

Ruff has no option to emit per-rule SARIF levels, so the mapping has to happen to the
file. A dedicated step rewrites `quality.sarif` before `upload-sarif`.

Both severity carriers are rewritten, because it is not documented which one code
scanning prefers and they currently agree:
- each result's `level`
- each rule's `properties["problem.severity"]`

*Alternative rejected - change the repository's code-scanning "check failure"
threshold to None.* One click, no code. Rejected as the primary mechanism because it is
invisible repository state that no one reviewing the workflow can see, it applies to
every tool rather than this one, and it would equally silence a genuinely high-severity
finding. Kept as the documented fallback under R1.

*Alternative rejected - stop uploading SARIF.* This would satisfy "no red check" by
discarding the capability `ci-quality-scanning` explicitly requires ("findings are
published in a machine-readable form", "annotations in code scanning").

### D2. Extract the HIGH set to one committed file

The set currently lives inline in a heredoc in the security-digest step. A second inline
copy in the mapping step would drift. It moves to one committed file that both steps
read, which also satisfies the new spec requirement that the declared set be recorded in
the repository.

*Alternative rejected - derive severity from ruff's rule prefix* (`S` = security =
high). Measured wrong: of 222 `S` findings only 6 are what bandit rates HIGH; 160 are
`S110 try-except-pass`. Prefix is category, not severity.

### D3. Non-high findings publish as `warning`, not `note`

`warning` keeps the finding visible in the code-scanning UI and in PR annotations while
not failing the check under GitHub's default threshold. `note` is quieter than the
capability wants - these findings are meant to be *visible* and not blocking.

### D4. Ratio assertions compare best-of-N samples, not single timings

**Corrected during implementation.** The original decision here was "require an absolute
floor on the slower side". Implementing it revealed that
`test_routing_overhead_comparison` *already has* exactly that - a 100µs
`legacy_floor_seconds` with a `pytest.skip` below it, added by an earlier flake fix -
and it still failed at **65.76x**. So the floor alone is not the fix.

The reason is that a floor guards only the DENOMINATOR. It establishes that the baseline
measurement is large enough to be resolvable, but says nothing about a one-off spike in
the NUMERATOR. A GC pause or scheduler slice landing inside the measured path inflates
the ratio no matter how well-resolved the baseline is. That is precisely what 65.76x is:
the docstring's own note observes "50-80x even though both paths are absolutely fast".

The standard technique for micro-benchmarks applies instead: take the **minimum of N
repeats** for each side. The minimum is the sample least contaminated by interference -
work can only ever be added by noise, never removed - so it estimates the true cost far
more stably than a single sample or a mean. The existing absolute floor is KEPT; the two
are complementary (floor = "is this resolvable at all", best-of-N = "is this sample
clean").

`test_bench_optional_vs_required_overhead` gets the same treatment, plus a floor
expressed in terms of the target latency it already declares (`target_ms=0.005`): when
the measured per-call cost sits far below the declared target, a 2.5x ratio on a
0.6µs difference is not a regression anyone can act on.

*Alternative rejected - raise the ratio limits.* Keeps a noise-dominated comparison and
merely moves the flake threshold; 65.76x would still clear a 10x limit eventually.

*Alternative rejected - delete the assertions.* They do catch real regressions once the
sample is clean. Best-of-N makes them do that instead of removing them.

*Alternative rejected - a blanket 1ms floor* (the original D4). For the ADR-009 test the
per-call cost is ~0.4µs and will never reach 1ms, so that floor would silence the
assertion permanently - deletion wearing a floor's clothing.

### D5. Classify browser-startup failure at the point it is detected

`SessionNotCreatedException: DevToolsActivePort file doesn't exist` means Chrome did not
start. The test currently surfaces it as `AssertionError: Expected HTML, got: ` or as a
raw tool error, both of which read like product defects. It is reported as an
environment failure instead.

Whether such a failure should *skip* rather than fail is deliberately left alone here -
silently skipping browser tests risks them never running. This change makes the cause
legible; it does not change the outcome.

## Risks / Trade-offs

- **[The repo's check-failure threshold also counts warnings, so the check stays red]**
  → Verify on the first pull request carrying this change. If it does, the same mapping
  step emits `note` for non-high instead - a one-line change to the same step. The
  repository setting (rejected in D1) remains the last resort.
- **[Downgrading severity hides something that mattered]** → The 6 findings bandit rates
  HIGH keep error severity. Nothing is removed: all 295 remain in the digest, the
  uploaded SARIF and the downloadable artifact. Only the level they publish at changes.
- **[Best-of-N hides a regression that only shows up under load]** → The measurement is still
  recorded and printed on every run, so a drift is observable; it simply does not fail
  the build from noise. A path that genuinely regresses past 1ms starts asserting again.
- **[The HIGH set drifts from bandit as ruff adds rules]** → The spec now requires the
  comparison be re-run and recorded whenever the set changes; the extracted file gives it
  one place to live.

## Migration Plan

No migration. The change is additive to CI and to two test files; nothing in `src/`
changes and no published interface moves. Rollback is reverting the commit - the prior
behaviour (everything at error severity) returns with it.

Existing open code-scanning alerts keep their current severity until the next upload
re-publishes them; the first run on `main` after merge re-levels them.

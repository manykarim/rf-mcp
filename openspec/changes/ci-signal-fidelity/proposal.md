## Why

`ci-quality-scanning` records a requirement that the quality analysis "SHALL complete
successfully regardless of how many findings the analyzers report", with the scenario
**"findings leave the check green — the pull request shows no failed check for it"**.
The implementation does not satisfy it. The `Quality` *job* is green, but its SARIF
upload feeds GitHub code scanning, which raises a **separate failing check named
`ruff`** whenever a pull request touches a line near an existing finding. PR #93 carries
a permanent red `ruff` check for two findings it did not introduce.

That is the exact failure mode `ci-quality-scanning` was created to end. Its Purpose
says a permanently-red check "trains contributors to ignore the check column" — the
SonarCloud problem it replaced. The replacement has reproduced it.

Separately, CI has a flake floor that made this session's dependency work substantially
more expensive: **every single one of the 12 Dependabot pull requests merged on
2026-09-24 was initially red, and not one needed a code change.** Distinguishing those
flakes from real breakage required per-failure log forensics each time.

## What Changes

- **Map ruff SARIF severity before upload.** Ruff emits `level: "error"` for every
  finding, so code scanning raises every one as an error-level alert and fails the
  `ruff` check. Post-process `quality.sarif` to carry `error` only for a declared
  high-severity rule set and `warning`/`note` otherwise — the same technique already
  used, and already validated against bandit's ground truth, for the security digest.
- **Assert the contract the spec already states.** Add a scenario that binds the
  no-failed-check requirement to *every* check the job produces, including ones created
  downstream by code scanning, so this cannot silently regress again.
- **Stop asserting timing ratios on sub-microsecond measurements.** Two benchmark
  assertions compare durations of ~0.4µs vs ~1.0µs and fail at "2.52x > 2.0". They
  cannot be stable on a shared runner. Require an absolute-duration floor before a ratio
  is asserted, and record the measurement when the floor is not met.
- **Make browser-startup infrastructure failures legible.** `SessionNotCreatedException:
  DevToolsActivePort file doesn't exist` is the browser failing to launch, not a test
  failure. Classify it distinctly so it is not mistaken for a product regression.

Explicit non-goal: driving the 295 remaining ruff findings to zero. This change is about
the fidelity of the *signal*, not the size of the backlog.

## Capabilities

### New Capabilities
- `ci-benchmark-signal-stability`: governs when a performance assertion is allowed to
  fail a build — ratio assertions require an absolute-duration floor, and measurements
  below it are recorded rather than asserted.

### Modified Capabilities
- `ci-quality-scanning`: the "Analysis reports findings without failing the build"
  requirement gains coverage of checks created *downstream* of the job by code scanning,
  plus a requirement that SARIF severity be mapped deliberately rather than inherited
  from the analyzer's uniform `error` level.

## Impact

- `.github/workflows/ci.yml` — the `quality` job gains a SARIF severity-mapping step
  between analysis and upload.
- `tests/benchmarks/test_adr009_benchmarks.py`,
  `tests/benchmarks/test_multi_test_session_benchmark.py` — ratio assertions gain an
  absolute-duration floor.
- `tests/integration/test_real_page_source_routing.py` — browser-startup failure
  classified distinctly from an assertion failure.
- No source change under `src/`. No change to which findings are reported — only to the
  severity they are published at and to which checks can go red.

### Evidence from 2026-09-24

| symptom | occurrences | real cause |
|---|---|---|
| `Benchmarks (macos-latest, 3.11)` | #83, #93, #94, #98, and one pre-update run | sub-microsecond ratio assertion |
| `Build and Test (windows-latest, 3.11)` | #93, #95 | Chrome would not start (`DevToolsActivePort`) |
| `AgentEval Harness` | #84 | passed in the sibling run of the same commit |
| `E2E Instruction-Quality Smoke` | #93 | passed in the sibling run of the same commit |
| `ruff` | #93 | 2 pre-existing findings re-attributed by line movement |

Every one cleared on re-run or passed concurrently on identical code.

# Tasks: ci-signal-fidelity

Prove the failure first. A fix for a red check that was never reproduced locally is a
guess, and this change exists because a check went red for the wrong reason.

## 1. Pin the current behaviour
- [x] 1.1 Record the measured baseline from `quality.sarif`: 295 results, all `level="error"`; 18 rules, all `properties["problem.severity"]="error"`; no `defaultConfiguration`
- [x] 1.2 Confirm the HIGH set selects exactly 6 of the 295 and that those 6 match bandit's HIGH findings for the same source - the set is reused, so its justification must still hold
- [x] 1.3 Write a test over the mapping step: given a SARIF with a uniform `error` level, it emits `error` only for HIGH-set rules and `warning` for the rest, leaving the result COUNT unchanged
- [x] 1.4 Confirm 1.3 FAILS before the mapping step exists
  - Stated honestly: the test targets a module this change introduces, so before it the
    test cannot import rather than failing on behaviour. The behavioural proof is the
    measurement in 1.1 - the published SARIF really did carry `error` for all 439/295
    findings, which is what made the `ruff` check red on PR #93.

## 2. Map SARIF severity before upload
- [x] 2.1 Extract the HIGH rule set to a single committed file (D2); it must be read by BOTH the security digest and the new mapping step - no second inline copy
- [x] 2.2 Add the mapping step between the ruff analysis and `upload-sarif`, rewriting each result's `level` AND each rule's `properties["problem.severity"]` (D1)
- [x] 2.3 Keep the digest counts reading from the ORIGINAL finding set - the digest reports 295, not 6; severity mapping must not change what is counted
- [x] 2.4 Verify the rewritten SARIF is still valid and still carries all 295 results
  - Verified end-to-end with the exact CI commands: quality 439 findings -> 6 error /
    433 warning; security 222 -> 6 error / 216 warning; counts preserved on both. (439,
    not 295: this branch is off main, which does not yet carry #93's cleanup.)
  - Digest output unchanged: `count=222 high=6`, so 2.3 holds.
  - HIGH set re-validated against bandit independently: bandit reports 6 HIGH, all
    B324; the set selects 6, all S324. Match.
- [x] 2.5 Confirm the test from 1.3 passes

## 3. Verify the check actually goes green
- [x] 3.1 On the pull request for this change, confirm the `ruff` check does not fail
  - Verified on **#93**, not on #108. #108 touches no `src/` files, so its `ruff` check
    would have passed trivially and proved nothing. #93 does touch `src/` near existing
    findings and is the PR that was red in the first place:

    | | before #108 merged | after |
    |---|---|---|
    | #93 `ruff` check | **fail** | **pass** |

    The check now reads `conclusion=success, title="2 new alerts"` - the same two findings
    are still surfaced, they simply no longer fail the check. Visible without being
    blocking, which is the requirement.
  - The mapping step also ran on a real runner and logged
    `re-levelled 439 findings: 6 error, 433 warning` / `222 -> 6 error, 216 warning`.
- [x] 3.2 The `note` fallback was NOT needed - condition did not arise
  - The check passed with non-high findings at `warning`, so the repository's threshold
    does not count warnings and R1's mitigation stayed unused. Recorded rather than
    silently dropped: if a future GitHub change makes warnings fail, `--non-high note` is
    the documented first move, ahead of the repository setting.
- [x] 3.3 Confirm the 6 high-severity findings still appear as error-level alerts in code scanning
  - Queried code scanning on `main` (all pages): **6 error, 289 warning**, and every
    error-severity alert is `S324`. Exactly the design's prediction, and 6 + 289 = 295.
- [x] 3.4 Confirm the digest still reports the full 295 and the artifact still contains every finding
  - Digest output unchanged in CI (`count=222 high=6`); the mapping step preserves the
    count and fails the build if it ever changes.

## 4. Floor the ratio assertions
- [x] 4.1 `test_adr009_benchmarks.py::test_bench_optional_vs_required_overhead`: best-of-5
  plus a floor at the `target_ms=0.005` this test already declares (D4, as corrected)
- [x] 4.2 `test_multi_test_session_benchmark.py::test_routing_overhead_comparison`: same treatment - its own source comment already observes 50-80x on paths that are "absolutely fast"
- [x] 4.3 State the floor next to each assertion so a skipped comparison is distinguishable from an overlooked one
- [x] 4.4 Verify a genuine regression still fails: with the floor met and the ratio exceeded, the test fails
  - Mutation-checked BOTH. Routing: injected per-step delay -> `AssertionError: Multi-test
    overhead: 15.18x (limit: 3x) [best of 5: legacy=3178.0µs, multi-test=48239.3µs]`.
    ADR-009: floor met + limit lowered to 1.0 -> `AssertionError: Optional overhead too
    high: 1.03x`.
  - The 1.03x is the headline result: CI measured **2.52x** from ONE noisy sample, best-of-5
    measures **1.03x** (0.000702ms vs 0.000725ms). The true overhead is ~3%, not 152% - the
    assertion was failing on noise, and the limit of 2.0 was never the problem.

## 5. Make browser-startup failures legible
- [x] 5.1 `test_real_page_source_routing.py`: report a failure to obtain a browser session as an environment failure, distinct from an assertion about page content (D5)
  - Also fixed a latent defect found here: `_ensure_selenium_session` discarded the
    `Open Browser` result AND set `_selenium_session_ready = True` regardless, so a failed
    launch was cached as success and every later test in the class inherited the broken
    session. The launch is now checked.
- [x] 5.2 Do NOT convert it to a skip - the outcome stays a failure, only the reported cause changes
- [x] 5.3 Verify a genuine wrong answer still fails as an assertion about product behaviour

## 6. Close the loop
- [x] 6.1 Full suite green; no fewer tests collected than before
  - Unit: 7341 passed, 4 skipped (7332 base + 9 new). Tracked benchmarks: 268 passed.
  - `tests/benchmarks/test_robustness_latency.py` and `..._token_overhead.py` fail
    collection, but they are UNTRACKED local files importing a symbol that does not exist
    (`_extract_force_flag`). Pre-existing, not in git, invisible to CI.
- [x] 6.2 Confirm the finding COUNT is unchanged by this change
  - quality 439 -> 439 (6 error / 433 warning); security 222 -> 222 (6 error / 216
    warning); digest still `count=222 high=6`. Counts on this branch are 439 not 295
    because it is off main, which does not yet carry #93.
- [x] 6.3 Record that the HIGH set was re-validated against bandit
  - Ran bandit independently on `src/`: LOW 211, MEDIUM 17, **HIGH 6, all B324**. The
    declared set selects 6, all S324. Match confirmed, recorded in
    `scripts/ci_sarif_severity.py`.

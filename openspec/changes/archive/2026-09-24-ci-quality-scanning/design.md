# Design: ci-quality-scanning

## The constraint that drives the shape

"Show the status, but do not break the build" has no clean primitive in GitHub Actions.
Both obvious mechanisms are wrong:

```
  step-level continue-on-error   job is GREEN; the failure is invisible unless
                                 someone expands the step log              -> no status

  job-level continue-on-error    the workflow run does not fail, BUT the job
                                 shows a red X and the PR reads
                                 "Some checks were not successful"         -> looks broken

  analyzer always exits 0,       check is GREEN, counts are in the log and the
  findings published as SARIF    run summary, findings are annotated on the
                                 PR and listed in the Security tab         -> what we want
```

GitHub has never shipped Travis-style `allow_failure`, and the long-running community
request for it is still open. So the job must *succeed on purpose* and carry its signal
out-of-band: digest to the log and run summary, SARIF to code scanning, reports to an
artifact.

The important subtlety: "succeed on purpose" must not swallow a *broken* analyzer. An
analyzer that fails to install produces zero findings, which is indistinguishable from a
clean result unless the two cases are separated. Hence the spec requires findings to be
non-fatal while infrastructure failure stays fatal.

## Tool selection

| dimension | chosen | licence | why |
|---|---|---|---|
| quality + security | **ruff** | MIT | single binary, SARIF native, milliseconds on 90K LOC |
| dependencies | **osv-scanner** | Apache-2.0 | reads `uv.lock` directly, SARIF with severity |
| maintainability | **radon** | MIT | the dimension linters do not cover |

### Rejected, with the measurement behind each

**bandit** - redundant. Ruff's `S` rule set *is* flake8-bandit. Measured on `src/`:

| | bandit | `ruff --select S` |
|---|---|---|
| total | 234 | 222 |
| HIGH | 6 x `B324` weak MD5 | 6 x `S324` - the same six |
| SARIF | needs a third-party formatter plugin | native |

Two tools, one job, and the one without native SARIF loses.

**Semgrep CE / OpenGrep** - built for multi-language SAST with custom rule authoring.
This is a single-language Python repository. Ruff covers it faster, and OpenGrep exists
precisely because Semgrep's rule-registry licensing became contested - a question this
change does not need to have.

**SonarQube Community Build** - the only genuinely open-source SonarQube, but it needs a
server and a database. Run ephemerally in CI it produces no trends and no gate, which is
the entire reason to run a server. All of the cost, none of the benefit.

**CodeQL / Qodana Community** - free for this repository, but neither is open source.
That was a stated constraint, and "free" is not the same property.

### Why `pip-audit` is not the dependency tool

`pip-audit` is what produced the 30-advisory measurement in the proposal, and it works
well locally. But its output formats are JSON, CycloneDX and markdown - no SARIF - so it
cannot satisfy the code-scanning half of the requirement without a converter.
`osv-scanner` reads the same lockfile, emits SARIF natively, and draws on the same OSV
database. `pip-audit` remains the better local/ad-hoc tool.

## The rule-set decision matters more than the tool

Measured on identical code, `src/`, 90,210 LOC:

| ruff rule set | findings |
|---|---|
| `E4,E7,E9,F` (classic minimal) | 217 |
| `E4,E7,E9,F,S` (**chosen**) | **439** (128 auto-fixable) |
| ruff 0.16.8 bare default | 5,763 |
| `--select ALL` | 11,263 auto-fixable alone |

A 26x swing from configuration alone. 5,763 findings would meet exactly the fate of
today's permanently-red SonarCloud: ignored. 439 is a set someone can actually drive to
zero, and it already contains the findings worth having - weak hashing, hardcoded SQL,
unguarded `subprocess`, hardcoded passwords.

This is also why the rule set must be **declared in the repository, not inherited from
the analyzer's defaults**. Ruff's bare default produced 5,763 here while its documented
classic default produces 217; pinning the selection in config means an upstream default
change cannot silently multiply the finding volume by 26.

Ratcheting the rule set wider later is cheap. Starting wide and retreating is not - the
credibility of the signal is spent by then.

## What this gives up

SonarQube's **"new code" quality gate** - a stored baseline with failure only on newly
introduced issues. No artifact-based open-source stack provides it without building it.

Deliberately out of scope. If the 439 is driven down and starts creeping back up, the
cheap version is a committed baseline count compared per run; the expensive version is a
persistent SonarQube Community Build. Neither is worth doing before there is evidence the
number moves in the wrong direction.

## Sequencing note

**Enable Dependabot first, and separately.** It is free on a public repository, costs no
CI minutes, and - unlike any scanner in this job - it opens fix pull requests rather than
only reporting. Against 30 live advisories that is the difference between visibility and
resolution. The job's dependency scan then serves a different purpose: a per-run count in
the digest, and lockfile-exact SARIF on the pull request.

Retiring SonarCloud should come **last**, after the new job has produced output on at
least one pull request. Removing the old signal before the new one is proven would leave
a window with no analysis at all.

## Risks

- **Stale pins inflate the dependency baseline.** The 71-advisory figure reflects
  dependencies that have simply not been upgraded, not a set of unfixable problems.
  `pyjwt` is the worked example: 6 of its 7 advisories are fixed in 2.12.x/2.13.0 and
  `uv lock --upgrade-package pyjwt` resolves cleanly to 2.14.0. Expect the baseline to
  fall substantially once dependencies are refreshed - do not read it as 71 outstanding
  defects.

  (An earlier draft of this design asserted `pyjwt` had "no published fix". That was
  wrong: the aggregation used to produce it printed the FIRST advisory's `fix_versions`,
  which happened to be the single empty one, and generalised it to all seven.)

- **Disputed advisories need a written decision, not silence.** The policy is: upgrade
  when a fix exists, suppress only with a recorded reason, and re-review suppressions
  when the dependency moves. Suppressing without a reason is how a dependency line
  becomes noise.

  Worth noting how little the escape hatch was needed in practice. `PYSEC-2025-183`
  (`CVE-2025-45768`) looked like the case for it - disputed by the maintainer, no fix
  version listed. But it is bounded `last_affected: 2.10.1`, so upgrading past it clears
  it like the rest: after the bump to 2.14.0, pip-audit reports NO remaining pyjwt
  advisories. "No fix version" is not the same as "cannot be resolved", and checking the
  affected RANGE is the difference.
- **439 findings are reported but not fixed by this change.** The job is honest about a
  number nobody has acted on yet. That is intended - visibility first - but it means the
  first weeks show a non-zero count that must not be mistaken for a regression.
- **Code scanning requires the repository to stay public** for this to remain free. A
  change to private visibility would silently remove the publication half of the design,
  leaving only the artifact and the digest.

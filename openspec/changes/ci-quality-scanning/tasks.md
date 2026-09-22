# Tasks: ci-quality-scanning

Ordered deliberately: Dependabot first (free, immediate, opens fix PRs), the job second,
SonarCloud retirement last (only once the replacement has proven itself on a real PR).

## 1. Dependency alerting - do this first, it needs no CI
- [x] 1.1 Enable Dependabot security updates on the repository (currently `disabled`; `GET /vulnerability-alerts` returns 404)
- [x] 1.2 Add `.github/dependabot.yml` for the `uv` / pip ecosystem, scoped to security updates initially
- [x] 1.3 Confirm alerts appear for the known-affected packages (`cryptography`, `pyjwt`, `urllib3`, `requests`, `lxml`, `idna`, `click`, `anyio`, `soupsieve`, `python-dotenv`)
- [x] 1.4 Record the advisory policy (DECIDED): upgrade when a fix exists; suppress only with a written reason; re-review on dependency movement
- [x] 1.5 Applied to `pyjwt`: `uv lock --upgrade-package pyjwt` 2.10.1 -> 2.14.0 - clears ALL 7 (pip-audit: none remaining)
- [x] 1.6 NOT NEEDED - `PYSEC-2025-183` is bounded `last_affected: 2.10.1`, so the upgrade clears it. No suppression file created; the policy in design.md stands for future cases.

## 2. Pin the rule set in the repository
- [x] 2.1 Add `[tool.ruff]` to `pyproject.toml` with `select = ["E4","E7","E9","F","S"]` - the measured 439-finding set
- [x] 2.2 Pin the ruff version used by CI so an upstream default change cannot alter the finding volume (bare default measured at 5,763 vs 217 for the classic default)
- [x] 2.3 Set `target-version` to match `requires-python` (>=3.10)
- [x] 2.4 Confirm the count locally before wiring CI: `ruff check src/ --statistics` should report 439

## 3. The `quality` job - analysis
- [x] 3.1 New job in `.github/workflows/ci.yml`, single OS, single Python version (analysis is platform-independent; the matrix would triple cost for identical results)
- [x] 3.2 `ruff check src/ --output-format=sarif` -> `reports/quality.sarif`
- [x] 3.3 `ruff check src/ --select S --output-format=sarif` -> `reports/security.sarif` (separate so security findings stay identifiable per the spec)
- [x] 3.4 `osv-scanner` against `uv.lock`, SARIF output -> `reports/deps.sarif`
- [x] 3.5 `radon cc` + `radon mi` JSON -> `reports/complexity.json`
- [x] 3.6 Capture each analyzer's exit status separately, distinguishing "found issues" from "failed to run"

## 4. The `quality` job - reporting
- [x] 4.1 Build the digest: quality count, security count + highest-severity count, dependency advisories + affected package count, files at the maintainability floor
- [x] 4.2 Print the digest to the job log AND append it to `$GITHUB_STEP_SUMMARY`
- [x] 4.3 Digest names the artifact and states that findings are published to code scanning
- [x] 4.4 Zero findings render as an explicit `0`, never an omitted line
- [x] 4.5 `actions/upload-artifact` with the whole `reports/` directory
- [x] 4.6 `github/codeql-action/upload-sarif` per SARIF file, each with a DISTINCT `category:` - without this they overwrite each other in code scanning
- [x] 4.7 Job exits 0 whenever the analyzers ran, and non-zero only when one could not run

## 5. Verify the behaviour the spec actually requires
- [x] 5.1 On a PR with findings: the check is green and the PR shows no failed check
- [x] 5.2 Findings appear as code scanning results, and all three SARIF categories remain separately visible
- [x] 5.3 The artifact downloads and contains all four reports
- [x] 5.4 Simulate a broken analyzer (e.g. an unresolvable pin) and confirm the job FAILS - the absence of analysis must not read as a clean result
- [x] 5.5 Confirm `continue-on-error` is used nowhere in the job
- [x] 5.6 Record the baseline counts in the PR description so later movement is interpretable

## 6. Retire SonarCloud - only after section 5 passes
- [x] 6.1 Confirm the new job has produced output on at least one real pull request
- [x] 6.2 SonarCloud integration removed from the repository (done by the maintainer; it is a GitHub App, so there is nothing to delete in-tree)
- [x] 6.3 Verified nothing left in-tree: no sonar-project.properties, no workflow reference, no repository secret
- [x] 6.4 Verified: no SonarCloud row on PR #86 - absent as both a check run and a commit status, on the new head AND the previous one (uninstalling the app also removed its previously-posted check runs)

## 7. Documentation
- [x] 7.1 Document the quality job in the contributor docs: what runs, where findings appear, and that it never blocks a merge
- [x] 7.2 State the rule-set policy - what is selected today and how to widen it - so the next person does not "fix" the 439 by switching to the bare default
- [x] 7.3 Note the deliberately-omitted "new code" gate and the conditions under which it would be revisited

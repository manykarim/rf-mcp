# Proposal: ci-quality-scanning

## Why

**rf-mcp has no working static analysis, and no dependency vulnerability coverage at all.**

Measured on 2026-09-22 against `main` + this branch:

| tool | declared | actually runs in CI |
|---|---|---|
| black | dev dependency | never |
| mypy | dev dependency | never |
| pre-commit | dev dependency | no config file exists |
| coverage | yes | yes - HTML + XML uploaded as artifacts |
| **SonarCloud** | - | yes, and **red on `main` for 4+ consecutive commits** |

SonarCloud is the only analysis running. It has been failing since before the current
branch existed, nobody acts on it, and it is not open source - so it cannot be
self-hosted or reasoned about. A permanently-red signal that no one reads is worse than
no signal: it trains everyone to ignore the check column.

### Dependencies are unscanned, and there are live advisories

```
security_and_analysis.dependabot_security_updates : disabled
GET /repos/manykarim/rf-mcp/vulnerability-alerts   : 404
.github/dependabot.yml                             : absent
```

Nothing scans dependencies. `pip-audit` against the current `uv.lock` finds
**30 unique advisories across 13 of 79 packages**, including `cryptography` (7),
`pyjwt` (7), `urllib3` (3), plus `requests`, `lxml`, `idna`,
`click`, `anyio`, `soupsieve` and the direct dependency `python-dotenv`.

### Code-quality signal exists and is small enough to act on

Measured with `ruff --select E4,E7,E9,F,S` on `src/` (90,210 LOC, 219 files):

```
Found 439 errors.   (128 auto-fixable)
```

Not noise - the top findings are real: 6x weak MD5 (`S324`), 14x hardcoded SQL
expression (`S608`), 10x `subprocess` without a shell guard (`S603`), 2x hardcoded
password (`S105`/`S107`), 3x bare `except`.

Complexity is healthy on average - **A (4.58)** across 3,244 blocks - but **8 files rank
`C` on the maintainability index**, six of them at exactly `MI = 0.00`: `server.py` (8,799 lines),
`keyword_executor.py`, `test_builder.py`, `rf_native_context_manager.py`,
`dynamic_keyword_orchestrator.py`, `rf_native_type_converter.py`.

## What Changes

- **A new `quality` CI job that never fails the build.** It runs the analyzers, prints a
  short digest to the log, writes the same digest to the run summary, uploads the
  machine-readable reports as an artifact, and publishes findings to code scanning.
  The check itself stays green; the findings carry the signal.
- **Ruff for code quality and code security**, starting at `E4,E7,E9,F,S` - the measured
  439-finding ruleset - with SARIF output.
- **OSV-Scanner for dependencies**, reading `uv.lock` directly and emitting SARIF.
- **Radon for the maintainability-floor list**, reported in the digest.
- **Dependabot enabled** for security updates - free on this public repository, needs no
  CI minutes, and opens fix PRs rather than only reporting.
- **SonarCloud retired** so the repository has one analysis signal rather than a working
  one beside a permanently-red one.

## Non-Goals

- **A "new code" quality gate.** SonarQube's baseline-and-ratchet gate is the one
  capability this loses. No artifact-based open-source stack provides it for free.
  Revisit only if the 439 findings are driven down and the count starts creeping back up.
- **Fixing the 439 findings, the 30 advisories, or the six MI-floor files.** This change
  makes them visible and keeps them visible. Acting on them is separate work, and
  bundling it would make this change unreviewable.
- **Replacing `black`/`mypy`** or wiring up `pre-commit`. Worth doing, but a different
  change: this one is about the analysis surface, not about formatting policy.
- **Multi-language SAST.** rf-mcp is a single-language Python project; Semgrep CE and
  OpenGrep are built for breadth this repository does not need.

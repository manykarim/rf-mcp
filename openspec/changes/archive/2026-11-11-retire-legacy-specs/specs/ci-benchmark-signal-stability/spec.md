## REMOVED Requirements

### Requirement: A performance assertion only fails on a measurement it can resolve
**Reason**: The test infrastructure this requirement governed (the agenteval harness, hard-gated e2e scenarios, performance benchmarks, and regression tests for defects in removed code) was removed with the 0.x implementation.

**Migration**: Agents are measured by the baseline suite in `tests/e2e/agent_eval` (run weekly by `.github/workflows/e2e-weekly.yml`); code quality is governed by `ci-quality-scanning`.

### Requirement: Environment failures are distinguishable from product failures
**Reason**: The test infrastructure this requirement governed (the agenteval harness, hard-gated e2e scenarios, performance benchmarks, and regression tests for defects in removed code) was removed with the 0.x implementation.

**Migration**: Agents are measured by the baseline suite in `tests/e2e/agent_eval` (run weekly by `.github/workflows/e2e-weekly.yml`); code quality is governed by `ci-quality-scanning`.

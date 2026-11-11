## REMOVED Requirements

### Requirement: Error handling paths must not raise on their own
**Reason**: The test infrastructure this requirement governed (the agenteval harness, hard-gated e2e scenarios, performance benchmarks, and regression tests for defects in removed code) was removed with the 0.x implementation.

**Migration**: Agents are measured by the baseline suite in `tests/e2e/agent_eval` (run weekly by `.github/workflows/e2e-weekly.yml`); code quality is governed by `ci-quality-scanning`.

### Requirement: No name is used before it is bound
**Reason**: The test infrastructure this requirement governed (the agenteval harness, hard-gated e2e scenarios, performance benchmarks, and regression tests for defects in removed code) was removed with the 0.x implementation.

**Migration**: Agents are measured by the baseline suite in `tests/e2e/agent_eval` (run weekly by `.github/workflows/e2e-weekly.yml`); code quality is governed by `ci-quality-scanning`.

### Requirement: Fixed defects are covered by tests that execute the defective path
**Reason**: The test infrastructure this requirement governed (the agenteval harness, hard-gated e2e scenarios, performance benchmarks, and regression tests for defects in removed code) was removed with the 0.x implementation.

**Migration**: Agents are measured by the baseline suite in `tests/e2e/agent_eval` (run weekly by `.github/workflows/e2e-weekly.yml`); code quality is governed by `ci-quality-scanning`.

### Requirement: The reported baseline reflects decisions, not debris
**Reason**: The test infrastructure this requirement governed (the agenteval harness, hard-gated e2e scenarios, performance benchmarks, and regression tests for defects in removed code) was removed with the 0.x implementation.

**Migration**: Agents are measured by the baseline suite in `tests/e2e/agent_eval` (run weekly by `.github/workflows/e2e-weekly.yml`); code quality is governed by `ci-quality-scanning`.

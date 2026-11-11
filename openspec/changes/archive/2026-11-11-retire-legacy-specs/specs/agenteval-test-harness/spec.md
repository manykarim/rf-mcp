## REMOVED Requirements

### Requirement: MCP-server-surface tests run through agenteval against a spawned server
**Reason**: The test infrastructure this requirement governed (the agenteval harness, hard-gated e2e scenarios, performance benchmarks, and regression tests for defects in removed code) was removed with the 0.x implementation.

**Migration**: Agents are measured by the baseline suite in `tests/e2e/agent_eval` (run weekly by `.github/workflows/e2e-weekly.yml`); code quality is governed by `ci-quality-scanning`.

### Requirement: Agentic e2e scenarios run through agenteval's agent adapters
**Reason**: The test infrastructure this requirement governed (the agenteval harness, hard-gated e2e scenarios, performance benchmarks, and regression tests for defects in removed code) was removed with the 0.x implementation.

**Migration**: Agents are measured by the baseline suite in `tests/e2e/agent_eval` (run weekly by `.github/workflows/e2e-weekly.yml`); code quality is governed by `ci-quality-scanning`.

### Requirement: Harness isolation keeps agenteval's dependencies out of rf-mcp's environment
**Reason**: The test infrastructure this requirement governed (the agenteval harness, hard-gated e2e scenarios, performance benchmarks, and regression tests for defects in removed code) was removed with the 0.x implementation.

**Migration**: Agents are measured by the baseline suite in `tests/e2e/agent_eval` (run weekly by `.github/workflows/e2e-weekly.yml`); code quality is governed by `ci-quality-scanning`.

### Requirement: The harness does not replace unit, internal-state, or benchmark tests
**Reason**: The test infrastructure this requirement governed (the agenteval harness, hard-gated e2e scenarios, performance benchmarks, and regression tests for defects in removed code) was removed with the 0.x implementation.

**Migration**: Agents are measured by the baseline suite in `tests/e2e/agent_eval` (run weekly by `.github/workflows/e2e-weekly.yml`); code quality is governed by `ci-quality-scanning`.

### Requirement: The agenteval dependency is version-pinned and upgraded deliberately
**Reason**: The test infrastructure this requirement governed (the agenteval harness, hard-gated e2e scenarios, performance benchmarks, and regression tests for defects in removed code) was removed with the 0.x implementation.

**Migration**: Agents are measured by the baseline suite in `tests/e2e/agent_eval` (run weekly by `.github/workflows/e2e-weekly.yml`); code quality is governed by `ci-quality-scanning`.

### Requirement: CI runs the harness against a spawned server
**Reason**: The test infrastructure this requirement governed (the agenteval harness, hard-gated e2e scenarios, performance benchmarks, and regression tests for defects in removed code) was removed with the 0.x implementation.

**Migration**: Agents are measured by the baseline suite in `tests/e2e/agent_eval` (run weekly by `.github/workflows/e2e-weekly.yml`); code quality is governed by `ci-quality-scanning`.

### Requirement: Migrating a black-box integration test preserves coverage and retires the pytest origin
**Reason**: The test infrastructure this requirement governed (the agenteval harness, hard-gated e2e scenarios, performance benchmarks, and regression tests for defects in removed code) was removed with the 0.x implementation.

**Migration**: Agents are measured by the baseline suite in `tests/e2e/agent_eval` (run weekly by `.github/workflows/e2e-weekly.yml`); code quality is governed by `ci-quality-scanning`.

### Requirement: Desktop scenarios are gated and not run on the stock CI runner
**Reason**: The test infrastructure this requirement governed (the agenteval harness, hard-gated e2e scenarios, performance benchmarks, and regression tests for defects in removed code) was removed with the 0.x implementation.

**Migration**: Agents are measured by the baseline suite in `tests/e2e/agent_eval` (run weekly by `.github/workflows/e2e-weekly.yml`); code quality is governed by `ci-quality-scanning`.

### Requirement: The browser-driving scenario is runnable headless and CI-gated
**Reason**: The test infrastructure this requirement governed (the agenteval harness, hard-gated e2e scenarios, performance benchmarks, and regression tests for defects in removed code) was removed with the 0.x implementation.

**Migration**: Agents are measured by the baseline suite in `tests/e2e/agent_eval` (run weekly by `.github/workflows/e2e-weekly.yml`); code quality is governed by `ci-quality-scanning`.

### Requirement: A partial integration file is split, not migrated wholesale
**Reason**: The test infrastructure this requirement governed (the agenteval harness, hard-gated e2e scenarios, performance benchmarks, and regression tests for defects in removed code) was removed with the 0.x implementation.

**Migration**: Agents are measured by the baseline suite in `tests/e2e/agent_eval` (run weekly by `.github/workflows/e2e-weekly.yml`); code quality is governed by `ci-quality-scanning`.

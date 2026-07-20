# agentic-e2e-instruction-quality Specification

## Purpose
TBD - created by archiving change autonomous-e2e-coverage. Update Purpose after archive.
## Requirements
### Requirement: Gate scenarios are validated before they hard-gate

A scenario SHALL hard-gate rf-mcp instruction quality only after it passes a validation
canary proving it is both calibrated and sensitive; otherwise it is demoted to
inform/diagnostic and cannot fail the build.

Calibrated means the reference model completes the scenario on good rf-mcp (baseline
`task_completion` at or above a configured threshold, so there is headroom to drop).
Sensitive means a targeted degradation of the instruction surface the scenario exercises
lowers the aggregate gate metric monotonically (degraded < good).

#### Scenario: an uncalibrated scenario is refused as a hard gate
- **WHEN** a candidate scenario's reference-model baseline shows the task is not completed on good rf-mcp (e.g. completion below threshold)
- **THEN** the scenario is recorded as inform/diagnostic only and never contributes a hard failure

#### Scenario: a non-monotonic scenario is refused as a hard gate
- **WHEN** degrading the instruction surface a candidate scenario exercises does not lower its aggregate metric versus the good-rf-mcp baseline
- **THEN** the scenario is flagged as an invalid probe and excluded from the hard gate

#### Scenario: a validated scenario hard-gates
- **WHEN** a scenario is both calibrated (reference completes on good rf-mcp) and sensitive (degradation lowers the metric)
- **THEN** it is eligible to hard-gate and a regression on it fails the build

### Requirement: Gate metrics are robust to agent floundering

The gate SHALL treat `task_completion` and first-try tool-selection correctness as the
primary quality signals and MUST NOT hard-gate on raw `tool_hit_rate` alone, because a
floundering agent inflates hit-rate by brute-forcing extra tool calls.

The gate additionally records `unexpected_tool_rate`, the discovery-to-execute call
ratio, and `artifact_executes` (a built suite that passes `run_test_suite` dry/full), and
retains infra-fault detection as a model-independent hard signal.

#### Scenario: raw hit-rate does not by itself fail the build
- **WHEN** a run's `tool_hit_rate` is higher than baseline but `task_completion` dropped and successful-call rate fell
- **THEN** the gate evaluates completion/success (not the inflated hit-rate) and reports the hit-rate as non-authoritative

#### Scenario: completion regression fails the build
- **WHEN** the reference model's aggregate `task_completion` for a validated scenario drops below its baseline beyond tolerance
- **THEN** the gate hard-fails with a regression attributed to that scenario

### Requirement: Per-model baseline regression with a capability-tiered roster

The gate SHALL compare each model's N-run aggregate against that model's own committed
baseline and MUST classify every model into a tier — reference, hard_gate, inform, or
excluded — with tier membership stored in the baseline file.

Reference and hard_gate models fail the build on a regression beyond tolerance; inform
models only warn on regression; excluded models (demonstrated broken tool-calling) are
not run in the gate. Infra faults hard-fail any non-excluded tier on any run.

#### Scenario: a hard-gate model regression fails
- **WHEN** a hard_gate model's aggregate metric drops below its baseline minus tolerance on a validated scenario
- **THEN** the gate hard-fails for that model

#### Scenario: an inform model regression only warns
- **WHEN** an inform-tier model regresses below its baseline
- **THEN** the gate records a warning and stays green (the model is too noisy to gate)

#### Scenario: an excluded model is not gated
- **WHEN** a model is listed as excluded (broken tool-calling)
- **THEN** the gate does not run it and it cannot fail or falsely pass the build

### Requirement: The reference model is a pinnable open-weight model

The reference model SHALL be an open-weight model pinnable to exact weights (recorded by
model identifier plus a pin descriptor such as revision and template/quant hashes), so
that a baseline change is attributable to an rf-mcp change rather than a silent vendor
model update.

A proprietary model MAY be retained only as a secondary, non-authoritative cross-check.
The baseline file records the reference model's pin metadata.

#### Scenario: baseline records the reference pin
- **WHEN** a baseline is captured for the reference model
- **THEN** the stored entry includes the reference model identifier and its pin descriptor

#### Scenario: a changed reference pin invalidates the baseline
- **WHEN** the gate runs with a reference model pin that differs from the pin recorded in the baseline
- **THEN** the gate flags the baseline as not matching the current reference and does not treat a difference as an rf-mcp regression

### Requirement: Gate scenarios use outcome-focused prompts and cover key surfaces

A gate scenario prompt SHALL describe the desired outcome without prescribing which MCP
tools to call, because prescribing the tool calls defeats the discoverability signal the
gate exists to measure.

The validated gate scenario set MUST cover the primary rf-mcp instruction surfaces that
calibrate cleanly — including API, XML, suite build-and-execute, discovery, desktop
discovery (display-free), data-driven, and locator/intent ergonomics.

#### Scenario: a prescriptive prompt is rejected from the gate set
- **WHEN** a scenario prompt enumerates the exact tool calls or keyword sequence to run
- **THEN** the scenario is not admitted to the hard-gate set until rewritten to be outcome-focused

#### Scenario: surface coverage is enforced
- **WHEN** the validated gate set is missing a primary surface that has a calibrated scenario available
- **THEN** the coverage gap is reported so the scenario can be added

### Requirement: Baselines carry provenance and staleness signals

Each baseline entry SHALL record its capture provenance — at minimum the capture
timestamp, the reference/model pin, and the rf-mcp git revision — and the gate MUST warn
when a baseline is stale or was captured against a different model pin.

Baselines are only lowered through a human-reviewed change (the no-decrease ratchet); the
gate never rewrites its own baselines during a normal (non-capture) run.

#### Scenario: stale baseline warning
- **WHEN** a baseline's recorded model pin or capture provenance no longer matches the current run configuration
- **THEN** the gate emits a staleness warning identifying the affected baseline

#### Scenario: a normal run does not self-heal
- **WHEN** the gate runs without the explicit capture flag and detects a regression
- **THEN** it fails or warns per tier and does not overwrite the committed baseline

### Requirement: Tiered CI execution

CI SHALL run a fast per-commit gate (the reference model over a small validated scenario
subset) and a broader scheduled gate (the full roster over the full validated set with a
higher run count), so per-commit feedback stays fast while breadth is covered on a
schedule.

Gate jobs are gated on the availability of their model API keys and skip (not fail) when
a required key is absent.

#### Scenario: per-commit smoke runs the reference subset
- **WHEN** a commit triggers CI and the reference model key is present
- **THEN** the reference model is evaluated over the fast validated subset and a regression fails the job

#### Scenario: missing key skips rather than fails
- **WHEN** a gate job's required model API key is absent
- **THEN** the job is skipped with a warning instead of failing the build


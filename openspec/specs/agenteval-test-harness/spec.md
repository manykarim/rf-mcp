# agenteval-test-harness Specification

## Purpose
TBD - created by archiving change adopt-agenteval-harness. Update Purpose after archive.
## Requirements
### Requirement: MCP-server-surface tests run through agenteval against a spawned server

The harness SHALL author deterministic MCP-server-surface tests as `robotframework-agenteval` `.robot`
suites that spawn the rf-mcp server as a subprocess over the real MCP protocol and assert on tool
results and recorded tool-call traces, rather than reaching into rf-mcp's internal Python objects. This
tests rf-mcp exactly as an agent consumes it (black box), and uses agenteval's `MCPLibrary` keywords
(`MCP.Start Server`, `MCP.Connect To Server`, `MCP.List Tools`, `MCP.Call Tool`, and the tool-call
readers) as the vocabulary.

#### Scenario: a deterministic suite drives the real server
- **WHEN** a MCP-surface test runs
- **THEN** it spawns rf-mcp as a subprocess, completes the MCP handshake, and asserts on the advertised tool set and on a tool call's returned result — with no model and no API key

#### Scenario: tool-call behavior is asserted from the recorded trace
- **WHEN** a suite exercises one or more rf-mcp tools
- **THEN** assertions read the recorded tool-call trace (names, counts, success, was-called) rather than rf-mcp's in-process state

### Requirement: Agentic e2e scenarios run through agenteval's agent adapters

The harness SHALL drive agentic e2e scenarios with agenteval's agent adapter (in-process pydantic-ai, or
a coding-agent CLI adapter) and project outcomes with agenteval's deterministic readers plus its
Metrics and Stat keywords, replacing rf-mcp's bespoke agent-integration, metrics, quality-gate, and
model-comparison machinery.

#### Scenario: an agent drives rf-mcp and the run is measured deterministically
- **WHEN** an agentic e2e scenario runs against a model credential
- **THEN** a real agent drives the spawned rf-mcp over MCP, and the harness asserts on the recorded tool-call trace and on token/cost/latency metrics read from the run result, never from model self-report

#### Scenario: stochastic outcomes are reduced with statistics
- **WHEN** an agentic assertion is stochastic
- **THEN** the harness expresses it as a pass@k / confidence-band assertion over repeated trials rather than a single brittle run

### Requirement: Harness isolation keeps agenteval's dependencies out of rf-mcp's environment

The harness SHALL run agenteval in an environment separate from rf-mcp's, invoking rf-mcp as a
subprocess; agenteval's pinned dependencies (its Robot Framework, MCP SDK, and pydantic-ai versions)
SHALL NOT be added to rf-mcp's runtime dependencies or its main development dependency set.

#### Scenario: adding the harness does not perturb rf-mcp's own dependency resolution
- **WHEN** the harness is installed and run
- **THEN** rf-mcp's runtime and main dev lockfile are unchanged, and rf-mcp executes from its own environment while agenteval executes from an isolated one

### Requirement: The harness does not replace unit, internal-state, or benchmark tests

The harness SHALL NOT replace rf-mcp's unit tests, its integration tests that assert on internal Python
state, or its performance benchmarks; those remain authored and executed with pytest. agenteval is a
black-box MCP/agentic surface library and is adopted only for the MCP-server-surface and agentic e2e
layers.

#### Scenario: an internal-state test stays in pytest
- **WHEN** a test needs to assert on rf-mcp's internal objects (session models, converters, discovery, or component wiring)
- **THEN** it remains a pytest test and is not migrated to the agenteval harness

### Requirement: The agenteval dependency is version-pinned and upgraded deliberately

The harness SHALL pin an exact `robotframework-agenteval` version and treat upgrades as reviewed changes
gated on its CHANGELOG, because its keyword libraries are labeled `provisional` and may break across
minor versions.

#### Scenario: the pin is exact and upgrades are intentional
- **WHEN** the harness dependency is declared
- **THEN** it names an exact agenteval version, and moving to a new version is a deliberate change that accounts for any documented breaking changes

### Requirement: CI runs the harness against a spawned server

CI SHALL execute the deterministic harness tier on every change and gate the agentic tier behind a model
credential so the keyless tier stays a fast, always-on gate.

#### Scenario: deterministic tier is always-on, agentic tier is credential-gated
- **WHEN** CI runs
- **THEN** the deterministic agenteval suites run without any model key, and the agentic suites run only when a model credential is present and skip cleanly otherwise

### Requirement: Migrating a black-box integration test preserves coverage and retires the pytest origin

When a black-box MCP-surface test is moved from pytest to the agenteval harness, the migration SHALL
preserve the original's observable coverage — the agenteval suite asserts the same tool results and
tool-call-trace facts the pytest test asserted — and the pytest original SHALL be removed only after its
agenteval port passes, so there is never a window of missing or long-lived duplicated coverage. A
candidate whose assertions cannot be expressed over the MCP surface SHALL be left in pytest, not
force-fit with weakened assertions.

#### Scenario: a ported test preserves its assertions
- **WHEN** a black-box integration test is ported to an agenteval `.robot` suite
- **THEN** the suite drives rf-mcp over the MCP protocol and asserts the same observable facts (tool result payloads and/or the recorded tool-call trace) that the pytest original asserted

#### Scenario: the pytest original is retired only when green
- **WHEN** an agenteval port of a pytest integration test is added
- **THEN** the pytest original is removed only after the port passes, and never before — so coverage is neither dropped nor left permanently duplicated

#### Scenario: a non-translatable candidate stays in pytest
- **WHEN** a candidate turns out to assert on internal Python state with no MCP-surface equivalent
- **THEN** it remains a pytest test and is reported as out of scope, rather than migrated with weakened assertions

### Requirement: Desktop scenarios are gated and not run on the stock CI runner

Harness scenarios that drive a real desktop (GTK apps via PlatynUI/AT-SPI) SHALL be gated behind an
explicit opt-in (`AGENTEVAL_DESKTOP`) and SHALL NOT run on the standard headless CI runner — they SHALL
skip cleanly there. Desktop coverage in CI SHALL require a dedicated desktop environment (a Docker image
providing Xvfb + a WM + AT-SPI + the GTK apps), which is out of scope for the standard runner because
hosted runners provide no `systemd --user` session the current suites depend on.

#### Scenario: desktop scenario skips on the stock runner
- **WHEN** the harness runs on a standard headless CI runner without the desktop opt-in
- **THEN** each desktop scenario skips cleanly (no failure), because no display/desktop environment is present

#### Scenario: desktop coverage requires a dedicated environment
- **WHEN** desktop scenarios are to actually run
- **THEN** they run only in a dedicated desktop environment (display + WM + AT-SPI + the apps) with `AGENTEVAL_DESKTOP` set, not on the stock runner

### Requirement: The browser-driving scenario is runnable headless and CI-gated

The harness SHALL allow its browser-driving agentic scenario to run with a **headless** browser (needing
no display) via an environment override, so it can execute on a headless CI runner, and CI SHALL run it
only in a **gated** context (a scheduled or manually-dispatched job), not on every push, because it is a
live model-driven browser run. The scenario definition SHALL remain unchanged (a visible browser stays
the local default); the headless behavior applies only when the override is opted in.

#### Scenario: the scenario runs headless when opted in
- **WHEN** the harness loads the browser-driving scenario with the headless override set
- **THEN** the agent is instructed to launch a headless browser, so the run completes on a runner with no display

#### Scenario: the scenario definition is unchanged by default
- **WHEN** the headless override is not set
- **THEN** the loaded scenario keeps its authored browser mode (a visible browser for local observability)

#### Scenario: CI runs the web scenario gated, not per-push
- **WHEN** CI runs
- **THEN** the web scenario runs only in a scheduled or dispatched job with a model credential and a browser installed, and never on the always-on per-push tier

### Requirement: A partial integration file is split, not migrated wholesale

When an integration file mixes MCP-observable tests with internal-state tests, the migration SHALL port
ONLY the MCP-observable tests to the agenteval harness and SHALL leave the internal-state tests in a
trimmed pytest file — never deleting the whole file and never porting an internal-state test with
weakened assertions. A file whose MCP-observable subset is too small to justify a split SHALL be left
whole in pytest and recorded as such, rather than fragmented for little gain.

#### Scenario: only the MCP-observable tests move
- **WHEN** a partial file is split
- **THEN** its MCP-observable tests become an agenteval suite and are removed from the pytest file, while its internal-state tests remain in the now-trimmed pytest file

#### Scenario: the split preserves coverage on both sides
- **WHEN** a partial file is split
- **THEN** every ported test asserts the same observable facts it did in pytest, and every retained pytest test is unchanged — no assertion is dropped or weakened on either side

#### Scenario: a low-value partial is left whole
- **WHEN** a partial file's MCP-observable subset is too small to justify a split
- **THEN** the whole file stays in pytest and the decision is recorded

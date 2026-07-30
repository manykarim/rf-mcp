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

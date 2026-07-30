## ADDED Requirements

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

# desktop-docker-agent-harness Specification

## Purpose
TBD - created by archiving change desktop-docker-agent-harness. Update Purpose after archive.
## Requirements
### Requirement: Reproducible isolated-X11 desktop image
The harness SHALL provide a Docker image that runs an isolated X11 desktop (Xvfb `:99`), an EWMH window manager, the AT-SPI accessibility stack, PlatynUI installed from pinned prebuilt wheels, robotmcp, and at least one desktop application under test — built from a dedicated Dockerfile without modifying the existing browser image.

#### Scenario: Image builds and the desktop comes up
- **WHEN** the desktop image is built and run
- **THEN** Xvfb `:99`, a window manager, the AT-SPI registry, and robotmcp are running, and the display is marked isolated (`ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY=:99`)

### Requirement: AT-SPI accessibility provider is active and verifiable
The container SHALL start the AT-SPI registry so PlatynUI's native runtime reports an active AT-SPI2 provider, and the harness SHALL fail loudly (not hang) when it is not.

#### Scenario: Provider is up
- **WHEN** the smoke harness queries the PlatynUI providers
- **THEN** an AT-SPI2 provider is listed as active

#### Scenario: Provider down fails fast
- **WHEN** the AT-SPI registry is not running
- **THEN** the harness reports a clear accessibility-bus diagnostic and exits non-zero instead of stalling

### Requirement: PlatynUI installed from pinned prebuilt wheels
`platynui-native` and `robotframework-PlatynUI` SHALL be installed from pinned PyPI prebuilt wheels at the same dev release (no in-image Rust build), and their native pattern API SHALL import cleanly.

#### Scenario: Native symbols import
- **WHEN** the container resolves a window node and checks `supported_patterns()` / `has_pattern`
- **THEN** the call succeeds with no `ImportError` for native symbols (e.g. `WindowSurface`)

### Requirement: Deterministic steering + read-back smoke (no LLM)
The harness SHALL include a deterministic smoke that launches the desktop AUT, drives it via PlatynUI keywords, reads a resulting control's value back via a keyword, and captures a screenshot — proving steering actions work and are confirmed, without any language model.

#### Scenario: Calculator compute + read-back
- **WHEN** the smoke launches gnome-calculator, drives `7 × 6 =`, and reads the result display via `Get Attribute`/`Query`
- **THEN** the read-back value is `42`, a non-trivial screenshot PNG is written to the artifacts directory, and the smoke exits zero

#### Scenario: Machine-checkable finding record
- **WHEN** the smoke completes
- **THEN** it writes a JSON record of provider status, resolved AUT identity, the read-back value, and the screenshot path(s)

### Requirement: Agent wiring for a MiniMax-class model
The harness SHALL wire a coding agent (opencode) to robotmcp inside the container, using a MiniMax provider configured from an API key supplied via environment/secret, with an example prompt to automate the AUT. The agent path SHALL be runnable when a key is present and SHALL NOT be required for the deterministic smoke.

#### Scenario: Agent drives the AUT with a key
- **WHEN** a MiniMax API key is provided and the agent run script is invoked with the example prompt
- **THEN** the agent connects to robotmcp, automates the calculator, and the run is confirmed by a screenshot and a keyword read-back of the result

#### Scenario: No key still gates on the deterministic smoke
- **WHEN** no API key is provided
- **THEN** the deterministic smoke still runs and gates the harness; only the agent rungs are skipped

### Requirement: Observation and artifact channels
Screenshots and run records SHALL be written to a mountable artifacts directory honoring the screenshot-path policy, and live observation SHALL be available via noVNC.

#### Scenario: Artifacts survive the container
- **WHEN** the artifacts directory is mounted and `ROBOTMCP_SCREENSHOT_DIR` points at it
- **THEN** screenshots and the JSON finding record are present on the host after the run

#### Scenario: Live view
- **WHEN** an operator opens the noVNC port
- **THEN** the in-container desktop and the AUT are visible live

### Requirement: The deterministic desktop smoke runs in CI as a gated, keyless job

CI SHALL run the deterministic desktop smoke by building `docker/Dockerfile.desktop` and running
its default smoke command, gating on the smoke's exit status, and this job SHALL be keyless (no
model credential) and gated to a scheduled/dispatched cadence rather than the per-push critical
path — because the image is large to build and its PlatynUI dependency is a pinned pre-release, so a
scheduled run surfaces a break on a cadence instead of red-ing every push. The job SHALL upload the
smoke's screenshot and JSON finding-record as artifacts when it fails, so a headless failure is
diagnosable without a live display.

#### Scenario: the gated job builds the image and passes on a healthy stack
- **WHEN** the scheduled or manually-dispatched desktop-smoke job runs
- **THEN** it builds the desktop image and runs the deterministic smoke, and the job passes only when the smoke exits zero (AT-SPI provider up, read-back value `42`, screenshot written) — with no model credential

#### Scenario: the job is gated, not per-push
- **WHEN** a normal push or pull request triggers CI
- **THEN** the desktop-smoke job does not run on that per-push path; it runs only on its scheduled cadence or on explicit manual dispatch

#### Scenario: a failure is diagnosable from uploaded artifacts
- **WHEN** the desktop-smoke job fails
- **THEN** it uploads the smoke's screenshot PNG(s) and the machine-checkable JSON finding-record (provider status, resolved AUT identity, read-back value, screenshot path) as build artifacts

### Requirement: The desktop image build context is trimmed by a dockerignore

The repository SHALL provide a `.dockerignore` that excludes large build-irrelevant paths (at least
the local virtualenv, the git directory, and generated artifacts) from the Docker build context, so
building `docker/Dockerfile.desktop` sends only the files the image actually needs rather than the
whole working tree.

#### Scenario: the build context excludes the virtualenv and git directory
- **WHEN** `docker/Dockerfile.desktop` is built from the repository root
- **THEN** the `.dockerignore` keeps the local virtualenv, `.git`, and generated artifact directories out of the build context, so the context is a small fraction of the full working-tree size


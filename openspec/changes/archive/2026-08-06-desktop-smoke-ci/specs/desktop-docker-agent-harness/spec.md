## ADDED Requirements

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

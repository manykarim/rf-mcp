# install-extra-resolvability Specification

## Purpose

Governs whether rf-mcp can actually be INSTALLED from PyPI by the commands its own
documentation advertises. rf-mcp 0.34.0/0.35.0 shipped with `desktop` folded into the
`[all]` extra, and because uv grants its pre-release allowance only to FIRST-PARTY
requirements, the pinned PlatynUI pre-release was refused for every consumer -
`uv tool install "rf-mcp[all]"` failed on every OS for every uv-backed installer,
including pipx. This capability defines what must hold for the published package to
resolve: extras that install with default resolver settings, a documented route for the
pre-release desktop extra, and a CI guard that installs rf-mcp the way a user does rather
than as a local path.

## Requirements

### Requirement: The `all` extra resolves with default resolver settings on every supported platform
`rf-mcp[all]` SHALL resolve and install with **no extra resolver flags** (no
`--prerelease`, no `--pre`) under uv, pip, pipx, poetry and pdm, on every platform rf-mcp
supports: Linux x86_64 and aarch64, macOS x86_64 and arm64, and Windows x86_64, for every
Python version in rf-mcp's `requires-python` range. The `all` extra SHALL NOT depend,
directly or transitively, on any distribution that is only published as a pre-release, and
SHALL NOT depend on any distribution whose available wheels exclude a supported platform
when no sdist is published for it.

#### Scenario: uv tool install of [all] succeeds without flags
- **WHEN** a user runs `uv tool install "rf-mcp[all]"` against the published package on a supported platform
- **THEN** resolution succeeds and the `robotmcp` executable is installed, with no `--prerelease` flag required

#### Scenario: pipx install of [all] succeeds without flags
- **WHEN** a user runs `pipx install "rf-mcp[all]"` with a pipx whose backend is uv
- **THEN** resolution succeeds, rather than failing with "No solution found when resolving dependencies"

#### Scenario: [all] is installable on a platform with no PlatynUI wheel
- **WHEN** a user on macOS x86_64, Alpine/musl, or a glibc < 2.34 distribution installs `rf-mcp[all]`
- **THEN** installation succeeds and provides web/API/mobile/database support, because `all` does not pull a desktop-only dependency that has no wheel for that platform

### Requirement: The `desktop` extra states its resolver and platform requirements
The `desktop` extra SHALL be documented with the exact command that installs it under each
resolver, including the `--prerelease=allow` requirement for uv-backed installers
(`uv`, `uvx`, `pipx`), and SHALL list the platforms for which PlatynUI wheels exist
(Linux x86_64/aarch64 with glibc >= 2.34, macOS arm64, Windows x86_64/arm64) together with
the platforms that are not supported (macOS x86_64, musl/Alpine, glibc < 2.34). No rf-mcp
documentation or source comment SHALL claim that the desktop pre-release pin resolves
without a pre-release flag under all resolvers.

#### Scenario: documented desktop install command works under uv
- **WHEN** a user follows the documented desktop install command for uv
- **THEN** the command includes `--prerelease=allow` (or sets `UV_PRERELEASE=allow`) and resolution succeeds

#### Scenario: unsupported platform is stated before the user attempts an install
- **WHEN** a user on macOS Intel or Alpine consults the extras documentation for desktop support
- **THEN** the documentation states that PlatynUI publishes no wheel for that platform, rather than leaving them to discover it as a resolution failure

### Requirement: Installability is verified the way a user installs, not as a local path
CI SHALL verify installability from a **built distribution resolved as a third-party
dependency** — not via `pip install -e .`, `uv pip install -e .`, `uv sync`, or the
committed lock file, all of which make rf-mcp's own pins first-party and therefore mask
pre-release and platform defects. The guard SHALL assert that `rf-mcp[all]` resolves with
default resolver settings, and SHALL assert the documented `desktop` install command
resolves on a platform where PlatynUI wheels exist.

#### Scenario: a pre-release dependency added to [all] fails CI
- **WHEN** a dependency that is only available as a pre-release is added to the `all` extra
- **THEN** the installability guard fails, naming the offending distribution, instead of the defect reaching PyPI

#### Scenario: the guard does not use an editable or lock-based install
- **WHEN** the installability guard runs
- **THEN** it installs from a built wheel/sdist through a resolver that treats rf-mcp's dependencies as transitive, so a first-party "explicit" pre-release allowance cannot hide the defect

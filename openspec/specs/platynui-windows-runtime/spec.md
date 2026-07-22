# platynui-windows-runtime Specification

## Purpose
TBD - created by archiving change fix-platynui-windows-runtime. Update Purpose after archive.
## Requirements
### Requirement: Desktop automation never leaves keyboard modifiers held
The system SHALL release all keyboard modifier keys (Ctrl, Alt, AltGr, Shift, Win — left and
right) whenever a PlatynUI desktop session ends, a keyboard keyword fails or times out, or the
server process exits, so that an interrupted or killed run can never leave the user's keyboard
physically wedged.

The release SHALL be best-effort and non-raising, SHALL be a no-op for keys that are not held, and
SHALL NOT start the native runtime solely to perform a release (it acts only when the runtime is
already open). Guidance SHALL steer agents toward the atomic `Keyboard Type <chord>` over a bare
`Keyboard Press`, which sends key-down only and must be paired with `Keyboard Release`.

#### Scenario: Session end releases held modifiers
- **WHEN** a PlatynUI desktop session is closed after any run
- **THEN** the plugin's session-end teardown dispatches a release of all modifier keys against the
  open runtime, and the call is a no-op when nothing is held

#### Scenario: Keyword failure or timeout releases held modifiers
- **WHEN** a desktop keyboard keyword (`Keyboard Press`/`Keyboard Type`/`Keyboard Release`) raises
  or times out mid-execution
- **THEN** a `finally` path releases all modifier keys before the error is returned, so a chord
  killed mid-flight does not leave a modifier down

#### Scenario: Process exit releases held modifiers
- **WHEN** the server process receives SIGTERM or exits (e.g. an `-p` run is killed) after desktop
  use has begun
- **THEN** an atexit/signal handler registered once on first desktop use releases all modifier keys

#### Scenario: Release never resurrects the runtime
- **WHEN** the release-all path runs but the native runtime is not open
- **THEN** it returns without starting the runtime and without raising

### Requirement: Desktop keyword queries fail fast and honor timeout_ms
The system SHALL bound PlatynUI desktop keyword queries with a short default timeout (rather than
PlatynUI's ~30s / ~60s library default) and SHALL honor a caller-supplied `timeout_ms` by mapping
it onto the PlatynUI query override, so that a wrong or honest-miss locator fails in ~1–2 seconds
and cannot stack into multi-minute stalls.

The short default SHALL remain overridable per session so a deliberately long wait (e.g. for an
application window to appear) can request a larger `timeout_ms`.

#### Scenario: Honest-miss locator fails fast
- **WHEN** a desktop keyword targets a locator that does not resolve
- **THEN** the query fails within the short default (~1.5s) instead of waiting PlatynUI's ~30s/60s
  default

#### Scenario: Caller timeout is honored
- **WHEN** a desktop keyword is executed with an explicit `timeout_ms`
- **THEN** that value is mapped onto the PlatynUI query override for the call and governs the wait

### Requirement: Desktop keyword execution never blocks the server event loop
The system SHALL execute PlatynUI desktop/native keywords off the server event loop (via a worker
thread) with a bounded wait, so that a slow desktop query cannot wedge the server or starve
unrelated calls such as metadata lookups.

#### Scenario: Metadata calls stay responsive during a slow desktop query
- **WHEN** a desktop `Focus`/`Query` on a large tree is in progress
- **THEN** the event loop remains free and concurrent metadata calls (e.g. `get_keyword_info`)
  return promptly instead of hanging behind the query

#### Scenario: Slow desktop call returns a bounded failure
- **WHEN** an off-thread desktop keyword exceeds its bounded wait
- **THEN** the caller receives a timeout failure without the loop having been blocked

### Requirement: The desktop safety guard is platform-aware on Windows
The system SHALL classify a Windows host as a distinct `windows` desktop state and, by default,
SHALL allow PlatynUI interaction keywords on Windows with a one-time active-desktop warning and a
Windows-accurate note, rather than refusing every keyword with a Linux-only Xephyr/Xvfb remediation
that cannot be followed on Windows.

The Windows branch SHALL leave the Linux isolated/active/unknown classification and its
refuse-by-default behaviour unchanged, and SHALL keep a strict opt-in for operators who want
isolation enforced on Windows.

#### Scenario: Windows host is allowed by default
- **WHEN** a PlatynUI interaction keyword runs on a Windows host with no explicit override
- **THEN** the guard classifies the display as `windows`, allows the keyword (not bypassed), and
  emits a one-time warning that it will drive the active desktop

#### Scenario: No Linux remediation is shown on Windows
- **WHEN** the safety guard evaluates a Windows host
- **THEN** the returned reason/remediation contains no Xephyr/Xvfb recipe and instead references a
  dedicated/RDP session for isolation

#### Scenario: Linux behaviour is unchanged
- **WHEN** the guard evaluates a Linux host with an active EWMH desktop and no isolation marker
- **THEN** it still refuses by default exactly as before


# Spec: desktop-isolation-marker-hardening

## ADDED Requirements

### Requirement: The isolation marker grants ISOLATED only when ownership is corroborated
The safety guard SHALL classify a bound display as ISOLATED from the marker only
when the marker is corroborated by a companion `ROBOTMCP_PLATYNUI_ISOLATED_XPID`
that resolves to a live X server for the claimed display; any marker without such
positive ownership proof — a stale/invalid XPID, no XPID at all, or ownership
that cannot be determined — SHALL fail closed to `unknown`, so synthetic input
is refused rather than allowed on the marker's assertion alone. Deployments that
own an isolated display record the XPID automatically (entrypoint / bootstrap).

#### Scenario: a corroborated marker on a WM-owning display is isolated
- **WHEN** the marker names the bound display, the recorded XPID resolves to a live X server for it, and an EWMH window manager (e.g. fluxbox on an isolated Xvfb) owns the display
- **THEN** classification is `isolated` and input is allowed — the legitimate isolated case is preserved despite a WM being present

#### Scenario: an invalid recorded XPID fails closed
- **WHEN** the marker names the bound display and `ROBOTMCP_PLATYNUI_ISOLATED_XPID` is set but does not resolve to a live X server for that display
- **THEN** classification is `unknown`, not `isolated`, and the guard refuses synthetic input (absent an explicit active-desktop opt-in)

#### Scenario: a marker with no ownership proof fails closed
- **WHEN** the marker names the bound display but no `ROBOTMCP_PLATYNUI_ISOLATED_XPID` is recorded (or ownership cannot be determined)
- **THEN** classification is `unknown` and the guard refuses synthetic input

### Requirement: Marker-vs-probe conflict is recorded distinctly
When the marker claims the bound display but the live EWMH probe reports an
active-desktop-shaped window manager that is not the expected isolated WM, the
guard SHALL record a distinct `marker_over_active_wm` provenance so the conflict
is observable to the operator and the agent, rather than silently classifying
ISOLATED from the marker.

#### Scenario: marker conflicts with an active-desktop probe
- **WHEN** the marker names the bound display, ownership is not positively corroborated, and the EWMH probe indicates an active desktop
- **THEN** `isolation_source` is `marker_over_active_wm` and classification is `unknown` (input refused unless the active-desktop opt-in is set)

#### Scenario: no marker leaves the probe path unchanged
- **WHEN** no isolation marker is set
- **THEN** classification follows the existing EWMH-probe logic (active when a WM is present, unknown when absent or the probe is inconclusive)

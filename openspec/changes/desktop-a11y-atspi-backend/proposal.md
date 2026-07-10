# Proposal: desktop-a11y-atspi-backend

## Why

An in-container desktop experiment (isolated Xvfb `:99` + fluxbox + AT-SPI, a
freshly launched GTK app under test — run outside any live GNOME session)
established that **`GTK_A11Y=1` is wrong on modern GTK**: GTK rejects the value
`1` with `Unrecognized accessibility backend '1'. Try GTK_A11Y=help`, and the
application then exposes **NO AT-SPI tree at all** — so every name-based
PlatynUI locator fails to resolve. The value must be the backend **name**,
`GTK_A11Y=atspi`. `GTK_A11Y=1` only ever appeared to work on hosts where the
GNOME session had already enabled accessibility for other reasons (it was
being ignored, not honored).

rf-mcp shipped two effects of the old, incorrect guidance:

1. Two agent-facing **remediation strings** told operators/agents to fix an
   empty AT-SPI tree by launching with `GTK_A11Y=1` — the exact value that
   guarantees an empty tree on a fresh app. Following that advice makes the
   condition permanent.
2. The **desktop suite replay preamble** (`Prepare Desktop Display
   Environment`) pinned `XDG_SESSION_TYPE`, `GDK_BACKEND`, `QT_QPA_PLATFORM`
   and removed `WAYLAND_DISPLAY`, but set **no** accessibility-bridge variable.
   A generated desktop suite replayed on a bare/headless environment therefore
   launches its GTK AUT with no forced AT-SPI bridge, and name-based locators
   can silently fail to resolve.

This change corrects both so rf-mcp's advice and its generated suites reflect
what actually exposes an AT-SPI tree.

## What Changes

- The `accessibility_not_exposed` remediation (`ui_tree_service`) now
  recommends **`GTK_A11Y=atspi`** and explicitly flags `GTK_A11Y=1` as rejected
  by modern GTK (empty-tree cause), instead of recommending `1`.
- The PlatynUI accessibility-exposure guidance rules
  (`rf_native_type_converter`) name **`GTK_A11Y=atspi`** as the bridge fix, not
  `GTK_A11Y=1`.
- The desktop replay preamble (`test_builder`) additionally emits
  `Set Environment Variable    GTK_A11Y    atspi`, so a replayed desktop suite
  forces the AT-SPI bridge for its GTK AUT. Non-desktop suites are unaffected.

Out of scope: the experiment harness itself (Docker/X11 clean room + agent
rung) is scratch tooling for gathering this evidence and is deliberately kept
out of the delivered repo — this change ships only the code corrections the
evidence justifies. Hand-written validation fixtures that still set
`GTK_A11Y 1` (e.g. `tests/lo_validation/libreoffice_writer.robot`) are noted
but not load-bearing for delivery.

## Capabilities

### Modified Capabilities

- `desktop-accessibility-exposure-diagnostic`: the bridge remediation recommends
  the correct `GTK_A11Y=atspi` backend name and warns against the rejected `1`.
- `desktop-suite-replay-environment`: the replay preamble also pins
  `GTK_A11Y=atspi` so a replayed desktop GTK AUT exposes its AT-SPI tree.

## Impact

- `src/robotmcp/components/execution/ui_tree_service.py` — remediation string.
- `src/robotmcp/utils/rf_native_type_converter.py` — guidance rules string.
- `src/robotmcp/components/test_builder.py` — one added preamble step.
- Tests: `tests/unit/test_suite_replay_environment.py` (asserts the `atspi`
  pin, forbids `1`), `tests/unit/test_desktop_native_platynui_alignment.py`
  (remediation + guidance assert `atspi`, reject `1` as an anti-pattern).

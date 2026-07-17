## Why

When a coding agent drives PlatynUI desktop automation through rf-mcp, two failure modes silently corrupt results: (1) the application under test (AUT) is automated against its accessibility tree while **no window is actually shown / focused**, so pointer and keyboard operations land on empty space or on whatever window happens to be active; and (2) clicks and key sequences are delivered to a **different, currently-focused window** instead of the AUT. Both produce green-looking steps that did not act on the intended app — the worst kind of false positive for generated test suites. This was reproduced repeatedly during ADR-025 validation across Claude CLI and Codex CLI (the "stacking trap": launching apps up-front leaves a later one on top, and the first scenario's clicks hit the wrong window). rf-mcp must make PlatynUI execution **window-targeted and visibly-focused by default**, not by accident.

## What Changes

- **Focus-before-act default for desktop sessions**: every pointer/keyboard operation in a DESKTOP_TESTING session resolves the AUT's top-level window and ensures it is raised + focused *before* the operation is dispatched, so input cannot leak to another window. Opt-out via an explicit escape hatch.
- **New focus/window intent + keywords surfaced through rf-mcp**: a first-class "ensure application focused" capability (wrapping PlatynUI window-surface activation, with a portable fallback for WM-less / WM-present X11 and Wayland) that agents are guided to call at the start of each test and after any context switch between apps.
- **Visibility precondition + guidance**: desktop session init and `get_session_state` surface whether the AUT window is mapped/visible/on-screen; locator guidance and MCP instructions steer agents to launch apps visibly and verify visibility before interacting. **BREAKING (behavioral)**: desktop steps that target an unmapped/occluded AUT window now emit a warning (and optionally fail-fast) instead of silently "passing".
- **Window-scoped operation safety**: pointer operations verify the resolved target element belongs to the AUT's window subtree (not just that *some* node matched the descriptor), catching cross-window coordinate collisions.
- **Cross-agent research + experiment matrix**: a documented, reproducible experiment harness running the same stepwise PlatynUI scenarios (Calculator + Text Editor) across **Codex CLI, OpenCode CLI, Kilo CLI, and Claude CLI**, collecting every focus/visibility/targeting issue each agent hits, to validate the defaults above and capture agent-specific environment quirks (e.g. stripped MCP-server env, sandbox MCP auto-deny).
- **Stepwise-by-default execution posture**: confirm and document that rf-mcp executes desktop steps individually with per-step verification, and that the focus guarantees hold across the stepwise → `build_test_suite` flow.

## Capabilities

### New Capabilities
- `platynui-window-focus`: Resolving the AUT's top-level window from any descriptor and ensuring it is raised + focused before pointer/keyboard dispatch; the focus intent/keyword surface; portable focus strategy across WM-present X11, WM-less X11 (e.g. Xvfb), and Wayland/XWayland.
- `platynui-visibility-guarantees`: Detecting and reporting AUT window visibility/mapped/on-screen state; init-time and execution-time visibility preconditions; warning/fail-fast policy for operations against non-visible windows.
- `platynui-window-scoped-actions`: Verifying a resolved interaction target belongs to the AUT window subtree before dispatch, preventing cross-window coordinate collisions; window-scoped descriptor resolution defaults.
- `platynui-agent-experiment-harness`: Reproducible multi-CLI (Codex/OpenCode/Kilo/Claude) experiment matrix and issue-collection methodology for PlatynUI stepwise automation, including environment-normalization requirements and the canonical Calculator/Text-Editor scenarios.

### Modified Capabilities
<!-- No pre-existing openspec/specs/ capabilities; the new-core PlatynUI behavior shipped under ADR-025 is not yet captured as an openspec capability, so all of the above are introduced as new specs. -->

## Impact

- **Code**:
  - `src/robotmcp/plugins/builtin/platynui_plugin.py` (focus-before-act hook, visibility hints, escape hatch env/arg)
  - `src/robotmcp/components/execution/keyword_executor.py` (desktop pre-dispatch focus + window-subtree verification)
  - `src/robotmcp/domains/intent/` (focus/ensure-window-active intent verb + PlatynUI mapping)
  - `src/robotmcp/components/execution/ui_tree_service.py` + `server.py` `get_session_state` (visibility/focus state in `ui_tree`)
  - `src/robotmcp/utils/rf_native_type_converter.py` (locator guidance: focus + visibility rules)
  - MCP instructions / `get_locator_guidance` `platynui` topic
- **Behavior**: desktop sessions gain a focus-before-act default and visibility precondition; opt-out preserves current behavior. No change to web/mobile/API sessions.
- **Tests**: new unit tests for focus resolution / visibility detection / window-scope verification; new live e2e assertions that operations only affect the AUT window; the multi-CLI experiment harness + its findings report.
- **Dependencies / environment**: relies on PlatynUI window-surface activation where a WM is present; documents the WM-less and Wayland fallbacks. Requires the `resolve_extents` upstream coordinate fix (ADR-025) for correct targeting on WM-less X servers. No new Python runtime dependency.
- **Docs**: new ADR (proposed) capturing the focus/visibility model and the cross-agent findings; updates to ADR-025 implementation notes.

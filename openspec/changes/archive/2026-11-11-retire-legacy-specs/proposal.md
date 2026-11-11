## Why

`restructure-robotcode-mcp` replaced the 0.x implementation with the robotcode-based rf-mcp 2.0, but it only
added the five specs for the new package. The 77 main specs written for 0.x are still in `openspec/specs/` and
describe tools, a desktop stack, a dashboard, an installer and test infrastructure that no longer exist. Anyone
reading the specs to learn what rf-mcp promises gets 277 requirements that are false.

## What Changes

- Remove every requirement of the 77 legacy capabilities and delete their main specs (`retire_capabilities`).
  Each removal names the reason and the spec or document that replaces it.
- Keep the six capabilities that describe rf-mcp 2.0: `robotcode-mcp-tools`, `robotcode-error-hinting`,
  `mcp-skills-provider`, `opentelemetry-observability`, `hybrid-execution-modes` and `ci-quality-scanning`
  (the `Quality` job in `ci.yml` still implements the last one).
- No code changes. The retired specs stay readable in git history and at the `v0.36.0` tag.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

All requirements are REMOVED, and each capability is retired:

- **0.x tool surface (14):** `agent-ergonomics-fixes`, `api-cookbook`, `build-suite-safe-persist`,
  `failed-step-error-signaling`, `fast-mcp-handshake-lazy-init`, `launch-env-fidelity`, `lazy-offline-embeddings`,
  `mcp-diagnostics-hygiene`, `mcp-instruction-set`, `mcp-stdio-log-safety`, `project-keyword-discovery`,
  `suite-argument-escaping`, `tool-profile-fastmcp3`, `visual-inspection-guidance`
- **Desktop, PlatynUI, stepwise recording and batch execution (55):** `batch-resume-argument-fidelity`,
  `batch-step-argument-compat`, every `desktop-*` capability (44), `native-platynui-desktop-alignment`, every
  `platynui-*` capability (5), `stepwise-suite-hygiene`, `stepwise-suite-isolation`, `suite-recording-fidelity`
- **Dashboard (1):** `frontend-dashboard`
- **0.x installer and extras (3):** `install-extra-resolvability`, `installer-cli-safety`, `tool-install-onboarding`
- **0.x test infrastructure (4):** `agenteval-test-harness`, `agentic-e2e-instruction-quality`,
  `ci-benchmark-signal-stability`, `quality-baseline-cleanup`

## Impact

- `openspec/specs/` shrinks from 83 to 6 capabilities.
- Archived changes under `openspec/changes/archive/` are untouched; they remain the record of how the 0.x
  behaviour was decided.
- Gaps this leaves open: nothing specifies the agent-eval baseline harness or the CI `installability` job yet.
  Both exist and are tested; specifying them is separate work.

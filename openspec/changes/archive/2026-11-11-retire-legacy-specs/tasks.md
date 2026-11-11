# Tasks

## 1. Classify the main specs

- [x] 1.1 Check every main spec against the rf-mcp 2.0 code, CLI and workflows, and keep only those it still implements: `robotcode-mcp-tools`, `robotcode-error-hinting`, `mcp-skills-provider`, `opentelemetry-observability`, `hybrid-execution-modes`, `ci-quality-scanning` (verified: `rf-mcp --help`, `rf-mcp init --help`, the `Quality` job in `.github/workflows/ci.yml`, `src/robotmcp/`)

## 2. Retire the legacy capabilities

- [x] 2.1 Write a REMOVED delta with a reason and a migration for every requirement of the 77 legacy capabilities, and set `retire_capabilities: true` in `.openspec.yaml` (verified: `openspec validate retire-legacy-specs --strict`)

## Workflow follow-up

- Archive the change; `openspec/specs/` then holds the six kept capabilities (`openspec list --specs`).

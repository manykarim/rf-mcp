# opentelemetry-observability Specification

## Purpose
Defines the OpenTelemetry tracing and metrics instrumentation for `rf-mcp` tool invocations, `robotcode` CLI/REPL/debug operations, skill reads, and error hint generation.

## Requirements
### Requirement: OpenTelemetry trace spans for robotcode operations and hinting
The system SHALL emit OpenTelemetry trace spans under the `rf-mcp` tracer nested beneath FastMCP's `fastmcp` server spans, recording attributes for `rf_mcp.operation`, `rf_mcp.execution_mode` (`server` or `client`), `rf_mcp.profile`, `rf_mcp.exit_code`, and `rf_mcp.hints_count`.

#### Scenario: Child span recorded for rf_analyze execution
- **WHEN** an agent invokes `rf_analyze` while an OpenTelemetry `TracerProvider` is active
- **THEN** the system records a parent FastMCP `tools/call rf_analyze` span and a child `rf_mcp.analyze` span with `rf_mcp.execution_mode` and `rf_mcp.exit_code` attributes

### Requirement: OpenTelemetry metrics for commands, durations, hints, and skills
The system SHALL record OpenTelemetry metrics via `opentelemetry.metrics` including counters for total `robotcode` operations (`rf_mcp.commands.total`), error hints generated (`rf_mcp.hints.generated`), and skills read (`rf_mcp.skills.read`), plus a duration histogram (`rf_mcp.command.duration_ms`).

#### Scenario: Skill reads are counted
- **WHEN** an agent reads `skill://rf-robotcode/SKILL.md` via MCP resource read or the `read_resource` tool while a `MeterProvider` is active
- **THEN** `rf_mcp.skills.read` is incremented by 1 with attribute `skill="rf-robotcode"`

#### Scenario: Zero-overhead no-op when OTel SDK is not configured
- **WHEN** `rf-mcp` starts without `OTEL_EXPORTER_OTLP_ENDPOINT` or `--otel-endpoint` configured
- **THEN** the server uses `opentelemetry-api`'s default no-op tracer and meter without requiring `opentelemetry-sdk` to be installed

#### Scenario: OTLP export enabled via CLI or environment variable
- **WHEN** `rf-mcp` starts with `OTEL_EXPORTER_OTLP_ENDPOINT` or `--otel-endpoint` and the `[otel]` extra installed
- **THEN** the server configures `TracerProvider` and `MeterProvider` with OTLP exporters to ship traces and metrics to the target collector

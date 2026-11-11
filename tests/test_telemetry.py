"""Unit tests for OpenTelemetry spans and metrics in src/robotmcp/telemetry.py."""

from __future__ import annotations

import builtins
import pytest
from fastmcp import FastMCP
from fastmcp.client import Client
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pathlib import Path

from robotmcp import telemetry
from robotmcp.config import ServerConfig
from robotmcp.server import create_server
from robotmcp.telemetry import (
    configure_otel,
    record_operation,
    record_skill_read,
)


def test_noop_when_endpoint_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert configure_otel(endpoint=None) is False


def test_noop_when_sdk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    orig_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name.startswith("opentelemetry.sdk") or name.startswith(
            "opentelemetry.exporter"
        ):
            raise ImportError("Simulated missing opentelemetry-sdk")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert configure_otel(endpoint="http://localhost:4318") is False


async def test_nested_fastmcp_and_rf_mcp_spans_and_metrics() -> None:
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    mcp = FastMCP("otel-test-server")

    @mcp.tool
    def rf_analyze(path: str = "tests/") -> dict[str, object]:
        with record_operation(
            "rf_mcp.analyze",
            operation="analyze",
            execution_mode="server",
            profile="ci",
            attributes={"rf_mcp.path": path},
        ) as tracker:
            tracker.set_result(exit_code=2, hints_count=1)
            return {"ok": False, "path": path}

    async with Client(mcp) as client:
        res = await client.call_tool("rf_analyze", {"path": "tests/login.robot"})
        assert res.data is not None

    record_skill_read("rf-robotcode")

    spans = span_exporter.get_finished_spans()
    span_names = [s.name for s in spans]
    assert any("rf_analyze" in name for name in span_names)
    assert "rf_mcp.analyze" in span_names
    assert "rf_mcp.skill.read" in span_names

    analyze_span = next(s for s in spans if s.name == "rf_mcp.analyze")
    parent_span = next(
        s for s in spans if "rf_analyze" in s.name and s.name != "rf_mcp.analyze"
    )
    assert analyze_span.parent is not None
    assert analyze_span.parent.span_id == parent_span.context.span_id
    assert analyze_span.attributes is not None
    assert analyze_span.attributes.get("rf_mcp.operation") == "analyze"
    assert analyze_span.attributes.get("rf_mcp.execution_mode") == "server"
    assert analyze_span.attributes.get("rf_mcp.profile") == "ci"
    assert analyze_span.attributes.get("rf_mcp.exit_code") == 2
    assert analyze_span.attributes.get("rf_mcp.hints_count") == 1

    metrics_data = metric_reader.get_metrics_data()
    assert metrics_data is not None
    metric_names: set[str] = set()
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                metric_names.add(m.name)

    assert "rf_mcp.commands.total" in metric_names
    assert "rf_mcp.command.duration_ms" in metric_names
    assert "rf_mcp.hints.generated" in metric_names
    assert "rf_mcp.skills.read" in metric_names


# --- 8.6 skill reads are counted on both read paths ----------------------------------------


async def test_every_skill_read_is_counted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    monkeypatch.setattr(telemetry.metrics, "get_meter_provider", lambda: provider)
    monkeypatch.setattr(telemetry.metrics, "get_meter", lambda name, *a, **k: provider.get_meter(name))

    async with Client(create_server(ServerConfig(execution_mode="client", workspace_root=tmp_path))) as client:
        await client.read_resource("skill://rf-robotcode/SKILL.md")
        await client.call_tool("read_resource", {"uri": "skill://rf-robotcode/SKILL.md"})

    total = 0
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "rf_mcp.skills.read":
                    for point in metric.data.data_points:
                        assert point.attributes["skill"] == "rf-robotcode"
                        total += point.value
    assert total == 2

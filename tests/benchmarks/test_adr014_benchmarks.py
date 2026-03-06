"""ADR-014 Benchmarks: Memory system performance measurements.

Measures embedding generation, vector search, and end-to-end recall times.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from typing import Any, Dict, List

import pytest

try:
    import sqlite_vec  # noqa: F401
    import model2vec  # noqa: F401

    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not MEMORY_AVAILABLE, reason="rf-mcp[memory] not installed"),
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.benchmark,
]


@pytest.fixture(scope="module")
def memory_services():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "bench_memory.db")

    from robotmcp.domains.memory.services import create_memory_services
    from robotmcp.domains.memory.value_objects import StorageConfig

    config = StorageConfig(db_path=db_path, enabled=True)
    services = create_memory_services(config)
    assert services is not None
    yield services

    if os.path.exists(db_path):
        os.unlink(db_path)
    for ext in ("-wal", "-shm"):
        p = db_path + ext
        if os.path.exists(p):
            os.unlink(p)


class TestEmbeddingBenchmarks:
    """Benchmark embedding generation speed."""

    async def test_single_embed_latency(self, memory_services):
        svc = memory_services["embedding_service"]
        # Warm up (first call loads model)
        await svc.embed("warmup text")

        times = []
        for _ in range(10):
            t0 = time.monotonic()
            await svc.embed("Click Element  id=login-btn")
            times.append((time.monotonic() - t0) * 1000)

        avg_ms = sum(times) / len(times)
        print(f"\nSingle embed: avg={avg_ms:.2f}ms, min={min(times):.2f}ms, max={max(times):.2f}ms")
        assert avg_ms < 50, f"Single embed too slow: {avg_ms:.2f}ms"

    async def test_batch_embed_throughput(self, memory_services):
        svc = memory_services["embedding_service"]
        texts = [f"Step {i}: Click Element  id=button-{i}" for i in range(100)]

        t0 = time.monotonic()
        vecs = await svc.embed_batch(texts)
        elapsed = (time.monotonic() - t0) * 1000

        per_text = elapsed / len(texts)
        print(f"\nBatch embed (100 texts): total={elapsed:.1f}ms, per_text={per_text:.2f}ms")
        assert len(vecs) == 100
        assert per_text < 10, f"Batch embed too slow: {per_text:.2f}ms/text"


class TestStorageBenchmarks:
    """Benchmark storage throughput."""

    async def test_store_throughput(self, memory_services):
        hook = memory_services["hook_service"]

        t0 = time.monotonic()
        for i in range(50):
            await hook.on_step_success(
                keyword=f"Click Element",
                arguments=[f"id=button-{i}"],
                result={"success": True, "output": f"Clicked {i}", "library": "SeleniumLibrary"},
                session_id="bench-sess",
            )
        elapsed = (time.monotonic() - t0) * 1000

        per_store = elapsed / 50
        print(f"\nStore throughput (50 records): total={elapsed:.1f}ms, per_record={per_store:.2f}ms")

    async def test_search_latency(self, memory_services):
        query_svc = memory_services["query_service"]

        from robotmcp.domains.memory.value_objects import MemoryQuery, MemoryType

        # Warm up search
        await query_svc.recall(MemoryQuery(
            query_text="Click Element button",
            memory_type=MemoryType.working_steps(),
            top_k=5,
            min_similarity=0.0,
        ))

        times = []
        for _ in range(10):
            t0 = time.monotonic()
            results = await query_svc.recall(MemoryQuery(
                query_text="Click Element login button",
                memory_type=MemoryType.working_steps(),
                top_k=5,
                min_similarity=0.0,
            ))
            times.append((time.monotonic() - t0) * 1000)

        avg_ms = sum(times) / len(times)
        print(f"\nSearch latency (10 queries): avg={avg_ms:.2f}ms, min={min(times):.2f}ms, max={max(times):.2f}ms")


class TestEndToEndBenchmarks:
    """Benchmark full store→recall cycle."""

    async def test_full_cycle(self, memory_services):
        hook = memory_services["hook_service"]
        query_svc = memory_services["query_service"]

        # Store
        t0 = time.monotonic()
        await hook.on_error_recovered(
            error_text="TimeoutError: Element id=slow-element not visible after 30s",
            fix_steps=[
                {"keyword": "Set Selenium Speed", "arguments": ["0.5s"]},
                {"keyword": "Wait Until Element Is Visible", "arguments": ["id=slow-element", "60s"]},
            ],
            session_id="bench-cycle",
        )
        store_ms = (time.monotonic() - t0) * 1000

        # Recall
        t0 = time.monotonic()
        results = await query_svc.recall_for_error("TimeoutError: Element not visible")
        recall_ms = (time.monotonic() - t0) * 1000

        print(f"\nFull cycle: store={store_ms:.1f}ms, recall={recall_ms:.1f}ms, results={len(results)}")

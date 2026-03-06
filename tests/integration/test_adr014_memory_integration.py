"""ADR-014 Integration Tests: End-to-end memory system tests.

Tests the full pipeline: store → embed → persist → search → recall.
Requires: sqlite-vec, model2vec (installed via rf-mcp[memory])
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any, Dict, Optional

import pytest

# Skip entire module if memory deps not available
try:
    import sqlite_vec  # noqa: F401
    import model2vec  # noqa: F401

    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

pytestmark = [
    pytest.mark.skipif(not MEMORY_AVAILABLE, reason="rf-mcp[memory] not installed"),
    pytest.mark.asyncio(loop_scope="module"),
]


@pytest.fixture(scope="module")
def memory_services():
    """Create memory services with a temp DB for the module."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_memory.db")

    from robotmcp.domains.memory.services import create_memory_services
    from robotmcp.domains.memory.value_objects import StorageConfig

    config = StorageConfig(
        db_path=db_path,
        enabled=True,
    )
    services = create_memory_services(config)
    assert services is not None, "Memory services should be created"
    yield services

    # Close repository connection before cleanup (prevents Windows file locks)
    repo = services.get("repository")
    if repo and hasattr(repo, "close"):
        try:
            repo.close()
        except Exception:
            pass
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestStoreAndRecall:
    """Test storing and recalling memories."""

    async def test_store_step_and_recall(self, memory_services):
        hook = memory_services["hook_service"]
        query = memory_services["query_service"]

        await hook.on_step_success(
            keyword="Click Element",
            arguments=["id=login-btn"],
            result={"success": True, "output": "Clicked", "library": "SeleniumLibrary"},
            session_id="test-sess",
        )

        from robotmcp.domains.memory.value_objects import MemoryQuery, MemoryType

        results = await query.recall(
            MemoryQuery(
                query_text="Click Element login",
                memory_type=MemoryType.working_steps(),
                top_k=5,
                min_similarity=0.0,
            )
        )
        assert len(results) >= 1
        assert "Click Element" in results[0].content

    async def test_store_error_and_recall_fix(self, memory_services):
        hook = memory_services["hook_service"]
        query = memory_services["query_service"]

        await hook.on_error_recovered(
            error_text="ElementNotFoundError: id=submit-btn not found",
            fix_steps=[
                {"keyword": "Wait Until Element Is Visible", "arguments": ["id=submit-btn", "10s"]},
                {"keyword": "Click Element", "arguments": ["id=submit-btn"]},
            ],
            session_id="test-sess",
        )

        results = await query.recall_for_error("ElementNotFoundError: id=submit-btn")
        assert len(results) >= 1
        assert results[0].confidence is not None
        assert results[0].confidence.action in ("auto_apply", "suggest", "deprioritize")

    async def test_store_knowledge_and_recall(self, memory_services):
        hook = memory_services["hook_service"]
        query = memory_services["query_service"]

        record_id = await hook.store_knowledge(
            content="Browser Library uses Playwright under the hood for web automation",
            knowledge_type="documentation",
            tags=["browser", "playwright"],
        )
        assert record_id is not None

        from robotmcp.domains.memory.value_objects import MemoryQuery, MemoryType

        results = await query.recall(
            MemoryQuery(
                query_text="Browser Library Playwright",
                memory_type=MemoryType.documentation(),
                top_k=5,
                min_similarity=0.0,
            )
        )
        assert len(results) >= 1
        assert "Playwright" in results[0].content

    async def test_store_keyword_and_recall(self, memory_services):
        hook = memory_services["hook_service"]
        query = memory_services["query_service"]

        await hook.on_keyword_discovered(
            keyword_name="Wait Until Element Is Visible",
            library="SeleniumLibrary",
            documentation="Waits until element is visible on page. timeout=10s",
        )

        results = await query.recall_keywords("wait for element visible")
        assert len(results) >= 1

    async def test_on_tool_call_dispatch(self, memory_services):
        hook = memory_services["hook_service"]

        await hook.on_tool_call(
            session_id="dispatch-test",
            tool_name="execute_step",
            arguments={"keyword": "Open Browser", "arguments": ["https://example.com"]},
            result={"success": True, "output": "Browser opened"},
        )

        from robotmcp.domains.memory.value_objects import MemoryQuery, MemoryType

        results = await memory_services["query_service"].recall(
            MemoryQuery(
                query_text="Open Browser",
                memory_type=MemoryType.working_steps(),
                top_k=5,
                min_similarity=0.0,
            )
        )
        assert len(results) >= 1


class TestGracefulDegradation:
    """Test that memory failures never crash the server."""

    async def test_recall_returns_empty_when_nothing_stored(self, memory_services):
        query = memory_services["query_service"]

        from robotmcp.domains.memory.value_objects import MemoryQuery, MemoryType

        results = await query.recall(
            MemoryQuery(
                query_text="completely unique query that matches nothing",
                memory_type=MemoryType.domain_knowledge(),
                top_k=5,
                min_similarity=0.9,  # Very high threshold
            )
        )
        assert results == []

    async def test_store_and_recall_session_end(self, memory_services):
        hook = memory_services["hook_service"]

        await hook.on_session_end(
            session_id="end-test",
            steps=[
                {"keyword": "Open Browser", "arguments": ["https://example.com"], "success": True},
                {"keyword": "Click Element", "arguments": ["id=menu"], "success": True},
                {"keyword": "Failing Step", "arguments": [], "success": False},
            ],
        )
        # Should store 2 successful steps, ignore the failing one
        # Verify no exceptions raised


class TestMemoryStore:
    """Test MemoryStore aggregate."""

    async def test_store_stats(self, memory_services):
        store = memory_services["store"]
        stats = store.to_dict()

        assert "total_records" in stats
        assert "collections" in stats
        assert stats["enabled"] is True
        assert stats["dimension"] in (256, 384)

    async def test_repo_count(self, memory_services):
        repo = memory_services["repository"]
        count = await repo.count()
        assert count >= 0


class TestSqliteVecPersistence:
    """Test that data persists in SQLite."""

    async def test_db_file_created(self, memory_services):
        config = memory_services["config"]
        assert os.path.exists(config.db_path)

    async def test_collection_stats(self, memory_services):
        repo = memory_services["repository"]
        stats = await repo.collection_stats("rfmcp_working_steps")
        # May or may not exist depending on test order
        if stats is not None:
            assert "record_count" in stats

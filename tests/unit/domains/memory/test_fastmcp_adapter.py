"""Tests for FastMCPMemoryAdapter.

Covers: tool registration, recall_step, recall_fix, recall_locator,
store_knowledge, get_memory_status, graceful degradation when services
are None or raise, and error dict format on failure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotmcp.domains.memory.aggregates import MemoryStore
from robotmcp.domains.memory.services import (
    EmbeddingService,
    MemoryHookService,
    MemoryQueryService,
)
from robotmcp.domains.memory.value_objects import (
    ConfidenceScore,
    LocatorRecallResult,
    LocatorStrategy,
    MemoryType,
    RecallResult,
    SimilarityScore,
    StorageConfig,
)

_async_mark = pytest.mark.asyncio(loop_scope="module")

# =========================================================================
# Helpers
# =========================================================================


def _make_recall_result(
    content: str = "test content",
    similarity: float = 0.8,
    memory_type: str = "working_steps",
    confidence: float | None = None,
    rank: int = 1,
) -> RecallResult:
    """Build a RecallResult for test assertions."""
    conf = ConfidenceScore(value=confidence) if confidence is not None else None
    return RecallResult(
        record_id="rec-001",
        content=content,
        memory_type=MemoryType(memory_type),
        similarity=SimilarityScore.cosine(similarity),
        adjusted_similarity=SimilarityScore.cosine(similarity),
        age_days=1.0,
        metadata={},
        confidence=conf,
        rank=rank,
    )


class FakeMCP:
    """Minimal stand-in for a FastMCP server that captures registered tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, Any] = {}

    def tool(self, **kwargs):
        """Decorator that captures the wrapped function by name."""

        def decorator(fn):
            self._tools[fn.__name__] = fn
            return fn

        return decorator

    def get_tool(self, name: str):
        return self._tools.get(name)


def _build_adapter(
    query_service: MemoryQueryService | None = None,
    hook_service: MemoryHookService | None = None,
    store: MemoryStore | None = None,
    embedding_service: EmbeddingService | None = None,
):
    """Build a FastMCPMemoryAdapter with mocked or default services."""
    from robotmcp.domains.memory.adapters.fastmcp_adapter import FastMCPMemoryAdapter

    if store is None:
        store = MemoryStore.create(StorageConfig.default())
    if embedding_service is None:
        embedding_service = MagicMock(spec=EmbeddingService)
        embedding_service.model_info = {
            "backend_name": "test",
            "model_name": "test-model",
            "dimension": 256,
            "is_available": True,
        }
    if query_service is None:
        query_service = AsyncMock(spec=MemoryQueryService)
    if hook_service is None:
        hook_service = AsyncMock(spec=MemoryHookService)

    adapter = FastMCPMemoryAdapter(
        query_service=query_service,
        hook_service=hook_service,
        store=store,
        embedding_service=embedding_service,
    )
    return adapter, query_service, hook_service, store, embedding_service


# =========================================================================
# Tool registration
# =========================================================================


class TestToolRegistration:
    def test_registers_all_five_tools(self):
        adapter, *_ = _build_adapter()
        mcp = FakeMCP()
        adapter.register_tools(mcp)

        expected_tools = {
            "recall_step",
            "recall_fix",
            "recall_locator",
            "store_knowledge",
            "get_memory_status",
        }
        assert set(mcp._tools.keys()) == expected_tools

    def test_tools_are_callable(self):
        adapter, *_ = _build_adapter()
        mcp = FakeMCP()
        adapter.register_tools(mcp)

        for name in mcp._tools:
            assert callable(mcp.get_tool(name))


# =========================================================================
# recall_step
# =========================================================================


@_async_mark
class TestRecallStep:
    async def test_returns_formatted_results(self):
        adapter, query_service, *_ = _build_adapter()
        results = [
            _make_recall_result(content="step 1", similarity=0.9, rank=1),
            _make_recall_result(content="step 2", similarity=0.7, rank=2),
        ]
        query_service.recall_steps = AsyncMock(return_value=results)

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        recall_step = mcp.get_tool("recall_step")

        response = await recall_step(scenario="login flow", top_k=5)
        assert response["count"] == 2
        assert len(response["results"]) == 2
        assert response["results"][0]["content"] == "step 1"
        assert response["results"][1]["content"] == "step 2"

    async def test_returns_empty_on_no_results(self):
        adapter, query_service, *_ = _build_adapter()
        query_service.recall_steps = AsyncMock(return_value=[])

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        recall_step = mcp.get_tool("recall_step")

        response = await recall_step(scenario="unknown scenario")
        assert response["count"] == 0
        assert response["results"] == []

    async def test_returns_unavailable_on_exception(self):
        adapter, query_service, *_ = _build_adapter()
        query_service.recall_steps = AsyncMock(
            side_effect=RuntimeError("connection lost")
        )

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        recall_step = mcp.get_tool("recall_step")

        response = await recall_step(scenario="anything")
        assert response["count"] == 0
        assert response["results"] == []
        assert "warning" in response


# =========================================================================
# recall_fix
# =========================================================================


@_async_mark
class TestRecallFix:
    async def test_returns_formatted_results_with_confidence(self):
        adapter, query_service, *_ = _build_adapter()
        results = [
            _make_recall_result(
                content="Fix: add retry",
                similarity=0.85,
                memory_type="common_errors",
                confidence=0.85,
                rank=1,
            ),
        ]
        query_service.recall_for_error = AsyncMock(return_value=results)

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        recall_fix = mcp.get_tool("recall_fix")

        response = await recall_fix(error_text="TimeoutError")
        assert response["count"] == 1
        result_dict = response["results"][0]
        assert result_dict["content"] == "Fix: add retry"
        assert result_dict["confidence"] == 0.85
        assert result_dict["action"] == "suggest"

    async def test_returns_unavailable_on_exception(self):
        adapter, query_service, *_ = _build_adapter()
        query_service.recall_for_error = AsyncMock(side_effect=Exception("boom"))

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        recall_fix = mcp.get_tool("recall_fix")

        response = await recall_fix(error_text="any error")
        assert response["count"] == 0
        assert "warning" in response


# =========================================================================
# recall_locator
# =========================================================================


@_async_mark
class TestRecallLocator:
    async def test_returns_results(self):
        adapter, query_service, *_ = _build_adapter()
        results = [
            LocatorRecallResult(
                locator="id=loginButton",
                strategy=LocatorStrategy.ID,
                keyword="Click",
                library="Browser",
                outcome="success",
                page_url="https://example.com",
                description="loginButton (id)",
                similarity=0.75,
            ),
        ]
        query_service.recall_locators = AsyncMock(return_value=results)

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        recall_locator = mcp.get_tool("recall_locator")

        response = await recall_locator(element_description="login button")
        assert response["count"] == 1
        assert response["results"][0]["locator"] == "id=loginButton"
        assert response["results"][0]["strategy"] == "id"
        assert response["results"][0]["keyword"] == "Click"

    async def test_returns_unavailable_on_exception(self):
        adapter, query_service, *_ = _build_adapter()
        query_service.recall_locators = AsyncMock(
            side_effect=Exception("fail")
        )

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        recall_locator = mcp.get_tool("recall_locator")

        response = await recall_locator(element_description="anything")
        assert response["count"] == 0
        assert "warning" in response

    async def test_passes_correct_element_description(self):
        adapter, query_service, *_ = _build_adapter()
        query_service.recall_locators = AsyncMock(return_value=[])

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        recall_locator = mcp.get_tool("recall_locator")

        await recall_locator(element_description="submit button")
        query_service.recall_locators.assert_called_once_with(
            "submit button"
        )


# =========================================================================
# store_knowledge
# =========================================================================


@_async_mark
class TestStoreKnowledge:
    async def test_returns_record_id_on_success(self):
        adapter, _, hook_service, *__ = _build_adapter()
        hook_service.store_knowledge = AsyncMock(return_value="rec-xyz-123")

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        store_knowledge = mcp.get_tool("store_knowledge")

        response = await store_knowledge(
            content="Browser Library requires headless mode",
            knowledge_type="domain_knowledge",
            tags=["browser"],
        )
        assert response["stored"] is True
        assert response["record_id"] == "rec-xyz-123"
        assert response["memory_type"] == "domain_knowledge"

    async def test_documentation_type_accepted(self):
        adapter, _, hook_service, *__ = _build_adapter()
        hook_service.store_knowledge = AsyncMock(return_value="doc-001")

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        store_knowledge = mcp.get_tool("store_knowledge")

        response = await store_knowledge(
            content="API documentation excerpt",
            knowledge_type="documentation",
        )
        assert response["stored"] is True
        assert response["memory_type"] == "documentation"

    async def test_rejects_invalid_knowledge_type(self):
        adapter, _, hook_service, *__ = _build_adapter()

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        store_knowledge = mcp.get_tool("store_knowledge")

        response = await store_knowledge(
            content="something",
            knowledge_type="working_steps",
        )
        assert response["stored"] is False
        assert "reason" in response
        assert "Invalid knowledge_type" in response["reason"]

    async def test_returns_failure_on_exception(self):
        adapter, _, hook_service, *__ = _build_adapter()
        hook_service.store_knowledge = AsyncMock(
            side_effect=RuntimeError("storage error")
        )

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        store_knowledge = mcp.get_tool("store_knowledge")

        response = await store_knowledge(
            content="some content",
            knowledge_type="domain_knowledge",
        )
        assert response["stored"] is False
        assert "reason" in response

    async def test_returns_stored_false_when_hook_returns_none(self):
        adapter, _, hook_service, *__ = _build_adapter()
        hook_service.store_knowledge = AsyncMock(return_value=None)

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        store_knowledge = mcp.get_tool("store_knowledge")

        response = await store_knowledge(
            content="content",
            knowledge_type="domain_knowledge",
        )
        assert response["stored"] is False
        assert response["record_id"] is None

    async def test_default_tags_empty_list(self):
        adapter, _, hook_service, *__ = _build_adapter()
        hook_service.store_knowledge = AsyncMock(return_value="rec-001")

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        store_knowledge = mcp.get_tool("store_knowledge")

        await store_knowledge(content="content")
        # Verify hook was called with default empty tags
        call_kwargs = hook_service.store_knowledge.call_args
        assert call_kwargs.kwargs.get("tags", []) == [] or call_kwargs[1].get(
            "tags", []
        ) == []


# =========================================================================
# get_memory_status
# =========================================================================


@_async_mark
class TestGetMemoryStatus:
    async def test_returns_stats_dict(self):
        adapter, _, __, store, embedding_service = _build_adapter()

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        get_memory_status = mcp.get_tool("get_memory_status")

        response = await get_memory_status()
        assert "store_id" in response
        assert "enabled" in response
        assert "backend" in response
        assert response["backend"]["backend_name"] == "test"

    async def test_returns_collection_info(self):
        adapter, _, __, store, embedding_service = _build_adapter()

        # Add a collection to the store
        store.ensure_collection(MemoryType.keywords())

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        get_memory_status = mcp.get_tool("get_memory_status")

        response = await get_memory_status()
        assert "collections" in response
        assert "rfmcp_keywords" in response["collections"]

    async def test_returns_error_dict_on_exception(self):
        adapter, _, __, store, embedding_service = _build_adapter()
        # Make store.to_dict() raise
        store.to_dict = MagicMock(side_effect=RuntimeError("internal error"))

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        get_memory_status = mcp.get_tool("get_memory_status")

        response = await get_memory_status()
        assert response["enabled"] is False
        assert "error" in response


# =========================================================================
# Unavailable recall sentinel
# =========================================================================


class TestUnavailableRecallSentinelStructure:
    def test_sentinel_structure(self):
        from robotmcp.domains.memory.adapters.fastmcp_adapter import (
            _UNAVAILABLE_RECALL,
        )

        assert _UNAVAILABLE_RECALL["results"] == []
        assert _UNAVAILABLE_RECALL["count"] == 0
        assert "warning" in _UNAVAILABLE_RECALL


@_async_mark
class TestUnavailableRecallSentinelCopy:
    async def test_sentinel_is_copied_not_shared(self):
        """Each failure response should be an independent copy."""
        adapter, query_service, *_ = _build_adapter()
        query_service.recall_steps = AsyncMock(side_effect=Exception("fail"))

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        recall_step = mcp.get_tool("recall_step")

        r1 = await recall_step(scenario="a")
        r2 = await recall_step(scenario="b")
        assert r1 is not r2
        # Mutating one should not affect the other
        r1["extra"] = "mutated"
        assert "extra" not in r2


# =========================================================================
# DISABLED_TOOL_KWARGS integration
# =========================================================================


class TestDisabledToolKwargs:
    def test_adapter_imports_disabled_tool_kwargs(self):
        """The adapter module should use DISABLED_TOOL_KWARGS from compat."""
        from robotmcp.domains.memory.adapters import fastmcp_adapter

        assert hasattr(fastmcp_adapter, "DISABLED_TOOL_KWARGS")


# =========================================================================
# RecallResult.to_dict integration
# =========================================================================


class TestRecallResultToDict:
    def test_to_dict_without_confidence(self):
        result = _make_recall_result(content="step 1", similarity=0.8, rank=1)
        d = result.to_dict()
        assert d["record_id"] == "rec-001"
        assert d["content"] == "step 1"
        assert d["similarity"] == 0.8
        assert d["rank"] == 1
        assert "confidence" not in d

    def test_to_dict_with_confidence(self):
        result = _make_recall_result(
            content="fix", similarity=0.9, confidence=0.95, rank=1
        )
        d = result.to_dict()
        assert d["confidence"] == 0.95
        assert d["action"] == "auto_apply"

    def test_to_dict_metadata_included_when_present(self):
        result = RecallResult(
            record_id="r-1",
            content="test",
            memory_type=MemoryType.keywords(),
            similarity=SimilarityScore.cosine(0.5),
            adjusted_similarity=SimilarityScore.cosine(0.5),
            age_days=2.0,
            metadata={"keyword": "Click Button"},
            rank=1,
        )
        d = result.to_dict()
        assert d["metadata"] == {"keyword": "Click Button"}


# =========================================================================
# _recall_suggestion() helper (ADR-014.2 Phase 3)
# =========================================================================


class TestRecallSuggestion:
    """Tests for the _recall_suggestion() module-level helper that generates
    actionable suggestion strings from recall results."""

    def test_empty_results(self):
        from robotmcp.domains.memory.adapters.fastmcp_adapter import (
            _recall_suggestion,
        )

        result = _recall_suggestion([])
        assert result == "No relevant memories found."

    def test_high_similarity_result(self):
        from robotmcp.domains.memory.adapters.fastmcp_adapter import (
            _recall_suggestion,
        )

        results = [
            _make_recall_result(content="step 1", similarity=0.8, rank=1),
        ]
        result = _recall_suggestion(results)
        assert "High-confidence" in result
        assert "directly" in result

    def test_low_similarity_result(self):
        from robotmcp.domains.memory.adapters.fastmcp_adapter import (
            _recall_suggestion,
        )

        results = [
            _make_recall_result(content="step 1", similarity=0.3, rank=1),
        ]
        result = _recall_suggestion(results)
        assert "Low-confidence" in result
        assert "starting points" in result

    def test_with_recall_result_adjusted_similarity(self):
        """RecallResult objects use adjusted_similarity property."""
        from robotmcp.domains.memory.adapters.fastmcp_adapter import (
            _recall_suggestion,
        )

        result_obj = RecallResult(
            record_id="r-1",
            content="fix content",
            memory_type=MemoryType.common_errors(),
            similarity=SimilarityScore.cosine(0.9),
            adjusted_similarity=SimilarityScore.cosine(0.9),
            age_days=0.5,
            metadata={},
            rank=1,
        )
        suggestion = _recall_suggestion([result_obj])
        assert "High-confidence" in suggestion

    def test_with_locator_recall_result(self):
        """LocatorRecallResult objects have a simple float similarity."""
        from robotmcp.domains.memory.adapters.fastmcp_adapter import (
            _recall_suggestion,
        )

        locator_result = LocatorRecallResult(
            locator="id=loginBtn",
            strategy=LocatorStrategy.ID,
            keyword="Click",
            library="Browser",
            outcome="success",
            page_url="https://example.com",
            description="loginBtn (id)",
            similarity=0.7,
        )
        suggestion = _recall_suggestion([locator_result])
        # LocatorRecallResult has no adjusted_similarity, falls back to
        # .similarity attribute which is a float > 0.5
        assert "High-confidence" in suggestion

    def test_boundary_similarity_exactly_half(self):
        """Similarity exactly at 0.5 should be low-confidence (not > 0.5)."""
        from robotmcp.domains.memory.adapters.fastmcp_adapter import (
            _recall_suggestion,
        )

        results = [
            _make_recall_result(content="step", similarity=0.5, rank=1),
        ]
        result = _recall_suggestion(results)
        assert "Low-confidence" in result


# =========================================================================
# Tool description directives (ADR-014.2 Phase 3)
# =========================================================================


class TestToolDescriptionDirectives:
    """Verify that MCP tool descriptions contain the expected action
    directives so LLMs know when to call each tool."""

    def test_recall_step_description_contains_before(self):
        adapter, *_ = _build_adapter()
        mcp = FakeMCP()
        adapter.register_tools(mcp)

        # Get the recall_step function and inspect its registration
        # The FakeMCP captures the function, and the tool decorator
        # receives the description kwarg. We verify via the adapter
        # source description constants.
        from robotmcp.domains.memory.adapters import fastmcp_adapter
        import inspect

        source = inspect.getsource(fastmcp_adapter.FastMCPMemoryAdapter.register_tools)
        # The recall_step description must contain "BEFORE"
        assert "BEFORE" in source

    def test_recall_fix_description_contains_immediately(self):
        from robotmcp.domains.memory.adapters import fastmcp_adapter
        import inspect

        source = inspect.getsource(fastmcp_adapter.FastMCPMemoryAdapter.register_tools)
        assert "IMMEDIATELY" in source

    def test_recall_locator_description_contains_before(self):
        from robotmcp.domains.memory.adapters import fastmcp_adapter
        import inspect

        source = inspect.getsource(fastmcp_adapter.FastMCPMemoryAdapter.register_tools)
        # Count occurrences of "BEFORE" - should be at least 2
        # (one for recall_step, one for recall_locator)
        count = source.count("BEFORE")
        assert count >= 2


# =========================================================================
# recall tool suggestion field (ADR-014.2 Phase 3)
# =========================================================================


@_async_mark
class TestRecallToolSuggestionField:
    """Verify that recall tools include a 'suggestion' field in response."""

    async def test_recall_step_includes_suggestion(self):
        adapter, query_service, *_ = _build_adapter()
        results = [
            _make_recall_result(content="step 1", similarity=0.9, rank=1),
        ]
        query_service.recall_steps = AsyncMock(return_value=results)

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        recall_step = mcp.get_tool("recall_step")

        response = await recall_step(scenario="login flow")
        assert "suggestion" in response
        assert isinstance(response["suggestion"], str)

    async def test_recall_fix_includes_suggestion(self):
        adapter, query_service, *_ = _build_adapter()
        results = [
            _make_recall_result(
                content="Fix: add retry",
                similarity=0.85,
                memory_type="common_errors",
                confidence=0.85,
                rank=1,
            ),
        ]
        query_service.recall_for_error = AsyncMock(return_value=results)

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        recall_fix = mcp.get_tool("recall_fix")

        response = await recall_fix(error_text="TimeoutError")
        assert "suggestion" in response

    async def test_recall_locator_includes_suggestion(self):
        adapter, query_service, *_ = _build_adapter()
        results = [
            LocatorRecallResult(
                locator="id=loginButton",
                strategy=LocatorStrategy.ID,
                keyword="Click",
                library="Browser",
                outcome="success",
                page_url="https://example.com",
                description="loginButton (id)",
                similarity=0.75,
            ),
        ]
        query_service.recall_locators = AsyncMock(return_value=results)

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        recall_locator = mcp.get_tool("recall_locator")

        response = await recall_locator(element_description="login button")
        assert "suggestion" in response

    async def test_recall_step_empty_results_suggestion(self):
        adapter, query_service, *_ = _build_adapter()
        query_service.recall_steps = AsyncMock(return_value=[])

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        recall_step = mcp.get_tool("recall_step")

        response = await recall_step(scenario="unknown")
        assert response["suggestion"] == "No relevant memories found."

    async def test_recall_locator_empty_results_suggestion(self):
        adapter, query_service, *_ = _build_adapter()
        query_service.recall_locators = AsyncMock(return_value=[])

        mcp = FakeMCP()
        adapter.register_tools(mcp)
        recall_locator = mcp.get_tool("recall_locator")

        response = await recall_locator(element_description="unknown element")
        assert "suggestion" in response
        assert "No relevant locator" in response["suggestion"]

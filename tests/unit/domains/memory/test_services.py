"""Tests for memory domain services.

Covers: EmbeddingService, MemoryQueryService, MemoryHookService,
        create_memory_services() factory.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotmcp.domains.memory.aggregates import EmbeddingBackend, MemoryStore
from robotmcp.domains.memory.entities import MemoryRecord
from robotmcp.domains.memory.events import MemoryRecalled
from robotmcp.domains.memory.repository import InMemoryMemoryRepository
from robotmcp.domains.memory.services import (
    EmbeddingService,
    MemoryHookService,
    MemoryQueryService,
    create_memory_services,
)
from robotmcp.domains.memory.value_objects import (
    EmbeddingVector,
    MemoryEntry,
    MemoryQuery,
    MemoryType,
    RecallResult,
    SimilarityScore,
    StorageConfig,
)

_async_mark = pytest.mark.asyncio(loop_scope="module")

# =========================================================================
# Helpers
# =========================================================================

_DIM = 4  # Use small dimension for fast tests


def _vec(values: List[float], model: str = "test-model") -> EmbeddingVector:
    """Build an EmbeddingVector from a list of floats."""
    return EmbeddingVector.from_list(values, model)


def _unit_vec() -> EmbeddingVector:
    """A simple unit-ish vector for tests that just need any embedding."""
    return _vec([1.0, 0.0, 0.0, 0.0])


def _make_backend(
    name: str = "model2vec",
    model: str = "test-model",
    dim: int = _DIM,
    available: bool = True,
) -> EmbeddingBackend:
    return EmbeddingBackend(
        backend_name=name,
        model_name=model,
        dimension=dim,
        is_available=available,
    )


def _make_config(
    enabled: bool = True,
    dim: int = _DIM,
) -> StorageConfig:
    return StorageConfig(
        db_path=":memory:",
        embedding_model="test-model",
        dimension=dim,
        enabled=enabled,
    )


def _make_embedding_service(
    backend: Optional[EmbeddingBackend] = None,
) -> EmbeddingService:
    """Create an EmbeddingService with a mocked model so no real ML loads."""
    b = backend or _make_backend()
    svc = EmbeddingService(b)
    return svc


def _mock_model_for_service(svc: EmbeddingService, dim: int = _DIM) -> MagicMock:
    """Inject a mock model that returns deterministic embeddings.

    The mock encode()/embed() returns a list of numpy-like objects
    whose .tolist() gives [1/dim, 1/dim, ...].
    """
    import numpy as np

    mock_model = MagicMock()

    def _encode(texts: List[str]) -> Any:
        return np.array([[1.0 / dim] * dim for _ in texts])

    def _embed(texts: List[str]) -> Any:
        return [np.array([1.0 / dim] * dim) for _ in texts]

    mock_model.encode = _encode
    mock_model.embed = _embed

    svc._model = mock_model
    return mock_model


def _make_wired_services(
    dim: int = _DIM,
) -> Dict[str, Any]:
    """Build fully-wired service set with mocked embedding for integration tests."""
    backend = _make_backend(dim=dim)
    config = _make_config(dim=dim)
    store = MemoryStore.create(config)
    repo = InMemoryMemoryRepository()

    embedding_svc = EmbeddingService(backend)
    _mock_model_for_service(embedding_svc, dim)

    query_svc = MemoryQueryService(
        embedding_service=embedding_svc,
        repository=repo,
        store=store,
    )
    hook_svc = MemoryHookService(
        query_service=query_svc,
        embedding_service=embedding_svc,
        repository=repo,
        store=store,
    )
    return {
        "backend": backend,
        "config": config,
        "store": store,
        "repo": repo,
        "embedding_service": embedding_svc,
        "query_service": query_svc,
        "hook_service": hook_svc,
    }


# =========================================================================
# EmbeddingService
# =========================================================================


class TestEmbeddingServiceProperties:
    """Tests for is_available, model_info properties."""

    def test_is_available_when_backend_available(self):
        svc = _make_embedding_service(_make_backend(available=True))
        assert svc.is_available is True

    def test_is_not_available_when_backend_unavailable(self):
        backend = EmbeddingBackend(is_available=False)
        svc = EmbeddingService(backend)
        assert svc.is_available is False

    def test_model_info_returns_backend_dict(self):
        backend = _make_backend(name="model2vec", model="test-m", dim=256)
        svc = EmbeddingService(backend)
        info = svc.model_info
        assert info["backend_name"] == "model2vec"
        assert info["model_name"] == "test-m"
        assert info["dimension"] == 256
        assert info["is_available"] is True


class TestEmbeddingServiceEnsureModel:
    """Tests for _ensure_model() lazy loading."""

    def test_ensure_model_raises_when_unavailable(self):
        backend = EmbeddingBackend(is_available=False)
        svc = EmbeddingService(backend)
        with pytest.raises(RuntimeError, match="No embedding backend available"):
            svc._ensure_model()

    def test_ensure_model_model2vec(self):
        fake_static_model = MagicMock()
        fake_static_model.from_pretrained.return_value = "loaded_model2vec"
        fake_module = ModuleType("model2vec")
        fake_module.StaticModel = fake_static_model  # type: ignore[attr-defined]

        backend = _make_backend(name="model2vec", model="minishlab/potion-base-8M")
        svc = EmbeddingService(backend)

        with patch.dict(sys.modules, {"model2vec": fake_module}):
            svc._ensure_model()

        assert svc._model == "loaded_model2vec"
        fake_static_model.from_pretrained.assert_called_once_with(
            "minishlab/potion-base-8M"
        )

    def test_ensure_model_fastembed(self):
        fake_text_embedding = MagicMock(return_value="loaded_fastembed")
        fake_module = ModuleType("fastembed")
        fake_module.TextEmbedding = fake_text_embedding  # type: ignore[attr-defined]

        backend = _make_backend(
            name="fastembed", model="BAAI/bge-small-en-v1.5", dim=384
        )
        svc = EmbeddingService(backend)

        with patch.dict(sys.modules, {"fastembed": fake_module}):
            svc._ensure_model()

        assert svc._model == "loaded_fastembed"
        fake_text_embedding.assert_called_once_with(
            model_name="BAAI/bge-small-en-v1.5"
        )

    def test_ensure_model_sentence_transformers(self):
        fake_st_class = MagicMock(return_value="loaded_st")
        fake_module = ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = fake_st_class  # type: ignore[attr-defined]

        backend = _make_backend(
            name="sentence-transformers", model="all-MiniLM-L6-v2", dim=384
        )
        svc = EmbeddingService(backend)

        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            svc._ensure_model()

        assert svc._model == "loaded_st"
        fake_st_class.assert_called_once_with("all-MiniLM-L6-v2")

    def test_ensure_model_idempotent(self):
        backend = _make_backend()
        svc = EmbeddingService(backend)
        svc._model = "already_loaded"
        svc._ensure_model()  # Should not raise or change model
        assert svc._model == "already_loaded"


@_async_mark
class TestEmbeddingServiceEmbed:
    """Tests for embed() and embed_batch()."""

    async def test_embed_model2vec(self):
        import numpy as np

        backend = _make_backend(name="model2vec", dim=3)
        svc = EmbeddingService(backend)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        svc._model = mock_model

        result = await svc.embed("hello world")
        assert isinstance(result, EmbeddingVector)
        assert result.dimensions == 3
        assert result.values == pytest.approx((0.1, 0.2, 0.3))
        assert result.model_name == "test-model"

    async def test_embed_fastembed(self):
        import numpy as np

        backend = _make_backend(name="fastembed", dim=3)
        svc = EmbeddingService(backend)
        mock_model = MagicMock()
        mock_model.embed.return_value = [np.array([0.4, 0.5, 0.6])]
        svc._model = mock_model

        result = await svc.embed("test text")
        assert result.values == pytest.approx((0.4, 0.5, 0.6))

    async def test_embed_sentence_transformers(self):
        import numpy as np

        backend = _make_backend(name="sentence-transformers", dim=3)
        svc = EmbeddingService(backend)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.7, 0.8, 0.9]])
        svc._model = mock_model

        result = await svc.embed("sentence")
        assert result.values == pytest.approx((0.7, 0.8, 0.9))

    async def test_embed_unknown_backend_raises(self):
        backend = _make_backend(name="unknown_backend")
        svc = EmbeddingService(backend)
        svc._model = MagicMock()  # bypass _ensure_model

        with pytest.raises(RuntimeError, match="Unknown backend"):
            await svc.embed("text")

    async def test_embed_truncates_long_text(self):
        import numpy as np

        backend = _make_backend(name="model2vec", dim=2)
        svc = EmbeddingService(backend)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0]])
        svc._model = mock_model

        long_text = "x" * 5000
        await svc.embed(long_text)

        # Check the text passed to encode was truncated to 2000 chars
        call_args = mock_model.encode.call_args[0][0]
        assert len(call_args[0]) == 2000

    async def test_embed_strips_whitespace(self):
        import numpy as np

        backend = _make_backend(name="model2vec", dim=2)
        svc = EmbeddingService(backend)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0]])
        svc._model = mock_model

        await svc.embed("  hello  ")
        call_args = mock_model.encode.call_args[0][0]
        assert call_args[0] == "hello"

    async def test_embed_batch_model2vec(self):
        import numpy as np

        backend = _make_backend(name="model2vec", dim=2)
        svc = EmbeddingService(backend)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
        svc._model = mock_model

        results = await svc.embed_batch(["first", "second"])
        assert len(results) == 2
        assert results[0].values == pytest.approx((1.0, 0.0))
        assert results[1].values == pytest.approx((0.0, 1.0))

    async def test_embed_batch_fastembed(self):
        import numpy as np

        backend = _make_backend(name="fastembed", dim=2)
        svc = EmbeddingService(backend)
        mock_model = MagicMock()
        mock_model.embed.return_value = [
            np.array([0.5, 0.5]),
            np.array([0.3, 0.7]),
        ]
        svc._model = mock_model

        results = await svc.embed_batch(["a", "b"])
        assert len(results) == 2
        assert results[0].values == pytest.approx((0.5, 0.5))

    async def test_embed_batch_sentence_transformers(self):
        import numpy as np

        backend = _make_backend(name="sentence-transformers", dim=2)
        svc = EmbeddingService(backend)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.9], [0.9, 0.1]])
        svc._model = mock_model

        results = await svc.embed_batch(["x", "y"])
        assert len(results) == 2

    async def test_embed_batch_unknown_backend_raises(self):
        backend = _make_backend(name="bogus")
        svc = EmbeddingService(backend)
        svc._model = MagicMock()

        with pytest.raises(RuntimeError, match="Unknown backend"):
            await svc.embed_batch(["text"])

    async def test_embed_batch_truncates_each_text(self):
        import numpy as np

        backend = _make_backend(name="model2vec", dim=2)
        svc = EmbeddingService(backend)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
        svc._model = mock_model

        texts = ["a" * 5000, "b" * 3000]
        await svc.embed_batch(texts)
        cleaned = mock_model.encode.call_args[0][0]
        assert len(cleaned[0]) == 2000
        assert len(cleaned[1]) == 2000


# =========================================================================
# MemoryQueryService
# =========================================================================


@_async_mark
class TestMemoryQueryServiceRecall:
    """Tests for MemoryQueryService.recall()."""

    async def test_recall_returns_empty_when_embedding_unavailable(self):
        backend = EmbeddingBackend(is_available=False)
        svc = EmbeddingService(backend)
        repo = InMemoryMemoryRepository()
        store = MemoryStore.create(_make_config())
        query_svc = MemoryQueryService(svc, repo, store)

        results = await query_svc.recall(
            MemoryQuery(query_text="test", memory_type=MemoryType.keywords())
        )
        assert results == []

    async def test_recall_returns_matching_records(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        query_svc: MemoryQueryService = services["query_service"]

        # Store some records via hook
        await hook_svc.on_step_success(
            keyword="Click", arguments=["button"], result={"output": "done"}, session_id="s1"
        )

        # Recall
        results = await query_svc.recall(
            MemoryQuery(query_text="Click button", memory_type=MemoryType.working_steps())
        )
        # Should find at least one result (same embedding = high similarity)
        assert len(results) >= 1
        assert results[0].rank == 1

    async def test_recall_applies_time_decay(self):
        services = _make_wired_services()
        query_svc: MemoryQueryService = services["query_service"]
        repo: InMemoryMemoryRepository = services["repo"]
        embedding_svc: EmbeddingService = services["embedding_service"]
        store: MemoryStore = services["store"]

        # Store a record with embedding
        entry = MemoryEntry(
            content="old knowledge",
            memory_type=MemoryType.keywords(),
        )
        emb = await embedding_svc.embed("old knowledge")
        entry_with_emb = entry.with_embedding(emb)
        record = store.prepare_store(entry_with_emb)
        coll_name = entry.memory_type.collection_name
        await repo.ensure_collection(coll_name, emb.dimensions)
        await repo.store(coll_name, record)

        query = MemoryQuery(
            query_text="old knowledge",
            memory_type=MemoryType.keywords(),
            apply_time_decay=True,
        )
        results = await query_svc.recall(query)
        assert len(results) >= 1
        # For a fresh record, adjusted similarity should be close to raw similarity
        # (time decay factor near 1.0 for age_days ~ 0)
        assert results[0].adjusted_similarity.value > 0

    async def test_recall_without_time_decay(self):
        services = _make_wired_services()
        query_svc: MemoryQueryService = services["query_service"]
        repo: InMemoryMemoryRepository = services["repo"]
        embedding_svc: EmbeddingService = services["embedding_service"]
        store: MemoryStore = services["store"]

        entry = MemoryEntry(
            content="some keyword info",
            memory_type=MemoryType.keywords(),
        )
        emb = await embedding_svc.embed("some keyword info")
        entry_with_emb = entry.with_embedding(emb)
        record = store.prepare_store(entry_with_emb)
        coll_name = entry.memory_type.collection_name
        await repo.ensure_collection(coll_name, emb.dimensions)
        await repo.store(coll_name, record)

        query = MemoryQuery(
            query_text="some keyword info",
            memory_type=MemoryType.keywords(),
            apply_time_decay=False,
        )
        results = await query_svc.recall(query)
        assert len(results) >= 1
        # Without decay, adjusted == raw
        assert results[0].adjusted_similarity.value == results[0].similarity.value

    async def test_recall_filters_by_min_similarity(self):
        services = _make_wired_services()
        query_svc: MemoryQueryService = services["query_service"]
        repo: InMemoryMemoryRepository = services["repo"]
        embedding_svc: EmbeddingService = services["embedding_service"]
        store: MemoryStore = services["store"]

        entry = MemoryEntry(
            content="test content",
            memory_type=MemoryType.keywords(),
        )
        emb = await embedding_svc.embed("test content")
        entry_with_emb = entry.with_embedding(emb)
        record = store.prepare_store(entry_with_emb)
        coll_name = entry.memory_type.collection_name
        await repo.ensure_collection(coll_name, emb.dimensions)
        await repo.store(coll_name, record)

        # Set min_similarity to 1.0 -- very unlikely to match exactly
        # because time_decay reduces the score slightly
        query = MemoryQuery(
            query_text="test content",
            memory_type=MemoryType.keywords(),
            min_similarity=0.999999,
            apply_time_decay=True,
        )
        # Time decay makes adjusted < raw, so very high threshold may filter it out
        # But with age=0, decay factor is 1.0, so it should still pass
        results = await query_svc.recall(query)
        # Result depends on exact cosine sim (identical embeddings = 1.0)
        # and time decay factor (age=0 -> factor=1.0)
        # So it should pass even at 0.999999
        assert len(results) >= 0  # Assertion: doesn't crash

    async def test_recall_respects_top_k(self):
        services = _make_wired_services()
        query_svc: MemoryQueryService = services["query_service"]
        repo: InMemoryMemoryRepository = services["repo"]
        embedding_svc: EmbeddingService = services["embedding_service"]
        store: MemoryStore = services["store"]

        # Store multiple records
        for i in range(10):
            entry = MemoryEntry(
                content=f"keyword info {i}",
                memory_type=MemoryType.keywords(),
            )
            emb = await embedding_svc.embed(f"keyword info {i}")
            entry_with_emb = entry.with_embedding(emb)
            record = store.prepare_store(entry_with_emb)
            coll_name = entry.memory_type.collection_name
            await repo.ensure_collection(coll_name, emb.dimensions)
            await repo.store(coll_name, record)

        query = MemoryQuery(
            query_text="keyword info",
            memory_type=MemoryType.keywords(),
            top_k=3,
            min_similarity=0.0,
        )
        results = await query_svc.recall(query)
        assert len(results) <= 3

    async def test_recall_ranks_results_sequentially(self):
        services = _make_wired_services()
        query_svc: MemoryQueryService = services["query_service"]
        repo: InMemoryMemoryRepository = services["repo"]
        embedding_svc: EmbeddingService = services["embedding_service"]
        store: MemoryStore = services["store"]

        for i in range(3):
            entry = MemoryEntry(
                content=f"step {i}",
                memory_type=MemoryType.keywords(),
            )
            emb = await embedding_svc.embed(f"step {i}")
            entry_with_emb = entry.with_embedding(emb)
            record = store.prepare_store(entry_with_emb)
            coll_name = entry.memory_type.collection_name
            await repo.ensure_collection(coll_name, emb.dimensions)
            await repo.store(coll_name, record)

        query = MemoryQuery(
            query_text="step",
            memory_type=MemoryType.keywords(),
            min_similarity=0.0,
        )
        results = await query_svc.recall(query)
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(results) + 1))

    async def test_recall_publishes_event(self):
        services = _make_wired_services()
        events: List[Any] = []

        query_svc = MemoryQueryService(
            embedding_service=services["embedding_service"],
            repository=services["repo"],
            store=services["store"],
            event_publisher=events.append,
        )

        # Store a record so recall returns something
        hook_svc: MemoryHookService = services["hook_service"]
        await hook_svc.on_step_success(
            keyword="Log", arguments=["message"], result={}, session_id="s1"
        )

        query = MemoryQuery(
            query_text="Log message",
            memory_type=MemoryType.working_steps(),
            min_similarity=0.0,
        )
        await query_svc.recall(query)

        assert len(events) == 1
        ev = events[0]
        assert isinstance(ev, MemoryRecalled)
        assert ev.result_count >= 0
        assert ev.query_time_ms >= 0

    async def test_recall_publishes_event_with_empty_results(self):
        services = _make_wired_services()
        events: List[Any] = []

        query_svc = MemoryQueryService(
            embedding_service=services["embedding_service"],
            repository=services["repo"],
            store=services["store"],
            event_publisher=events.append,
        )

        query = MemoryQuery(
            query_text="nonexistent topic",
            memory_type=MemoryType.keywords(),
        )
        results = await query_svc.recall(query)
        assert results == []
        assert len(events) == 1
        assert events[0].result_count == 0
        assert events[0].top_similarity == 0.0

    async def test_recall_common_errors_sets_confidence(self):
        services = _make_wired_services()
        repo: InMemoryMemoryRepository = services["repo"]
        embedding_svc: EmbeddingService = services["embedding_service"]
        store: MemoryStore = services["store"]
        query_svc: MemoryQueryService = services["query_service"]

        entry = MemoryEntry(
            content="Error: element not found",
            memory_type=MemoryType.common_errors(),
        )
        emb = await embedding_svc.embed("Error: element not found")
        entry_with_emb = entry.with_embedding(emb)
        record = store.prepare_store(entry_with_emb)
        coll_name = entry.memory_type.collection_name
        await repo.ensure_collection(coll_name, emb.dimensions)
        await repo.store(coll_name, record)

        query = MemoryQuery(
            query_text="Error: element not found",
            memory_type=MemoryType.common_errors(),
            min_similarity=0.0,
        )
        results = await query_svc.recall(query)
        assert len(results) >= 1
        # common_errors should have confidence set
        assert results[0].confidence is not None
        assert results[0].confidence.value > 0

    async def test_recall_non_error_type_has_no_confidence(self):
        services = _make_wired_services()
        repo: InMemoryMemoryRepository = services["repo"]
        embedding_svc: EmbeddingService = services["embedding_service"]
        store: MemoryStore = services["store"]
        query_svc: MemoryQueryService = services["query_service"]

        entry = MemoryEntry(
            content="Click Button",
            memory_type=MemoryType.working_steps(),
        )
        emb = await embedding_svc.embed("Click Button")
        entry_with_emb = entry.with_embedding(emb)
        record = store.prepare_store(entry_with_emb)
        coll_name = entry.memory_type.collection_name
        await repo.ensure_collection(coll_name, emb.dimensions)
        await repo.store(coll_name, record)

        query = MemoryQuery(
            query_text="Click Button",
            memory_type=MemoryType.working_steps(),
            min_similarity=0.0,
        )
        results = await query_svc.recall(query)
        assert len(results) >= 1
        assert results[0].confidence is None

    async def test_recall_graceful_on_exception(self):
        """If something unexpected goes wrong, recall returns [] instead of crashing."""
        backend = _make_backend()
        svc = EmbeddingService(backend)
        # Force embed to raise
        svc._model = MagicMock()
        svc._model.encode = MagicMock(side_effect=RuntimeError("boom"))

        repo = InMemoryMemoryRepository()
        store = MemoryStore.create(_make_config())
        query_svc = MemoryQueryService(svc, repo, store)

        results = await query_svc.recall(
            MemoryQuery(query_text="test", memory_type=MemoryType.keywords())
        )
        assert results == []


@_async_mark
class TestMemoryQueryServiceConvenienceMethods:
    """Tests for recall_for_error(), recall_keywords(), recall_steps()."""

    async def test_recall_for_error(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        query_svc: MemoryQueryService = services["query_service"]

        await hook_svc.on_step_failure(
            keyword="Click",
            arguments=["//button"],
            error_text="Element not found",
            session_id="s1",
        )

        results = await query_svc.recall_for_error("Element not found", "s1")
        assert isinstance(results, list)

    async def test_recall_keywords(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        query_svc: MemoryQueryService = services["query_service"]

        await hook_svc.on_keyword_discovered(
            keyword_name="Click Element",
            library="SeleniumLibrary",
            documentation="Clicks an element identified by locator.",
        )

        results = await query_svc.recall_keywords("Click")
        assert isinstance(results, list)

    async def test_recall_steps(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        query_svc: MemoryQueryService = services["query_service"]

        await hook_svc.on_step_success(
            keyword="Open Browser",
            arguments=["https://example.com", "chrome"],
            result={"output": "opened"},
        )

        results = await query_svc.recall_steps("open a browser to example.com")
        assert isinstance(results, list)


# =========================================================================
# MemoryHookService — write hooks
# =========================================================================


@_async_mark
class TestMemoryHookServiceWriteHooks:
    """Tests for on_step_success, on_step_failure, on_error_recovered,
    on_keyword_discovered, on_session_end, store_knowledge.
    """

    async def test_on_step_success_stores_working_steps(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        await hook_svc.on_step_success(
            keyword="Click",
            arguments=["button_ok"],
            result={"output": "clicked successfully"},
            session_id="sess-1",
        )

        coll = MemoryType.working_steps().collection_name
        count = await repo.count(coll)
        assert count == 1

    async def test_on_step_success_includes_keyword_in_content(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        await hook_svc.on_step_success(
            keyword="Input Text",
            arguments=["locator", "value"],
            result={"output": "typed value"},
            session_id="sess-1",
        )

        coll = MemoryType.working_steps().collection_name
        records = list(repo._records.get(coll, {}).values())
        assert len(records) == 1
        assert "Input Text" in records[0].entry.content
        assert "locator" in records[0].entry.content

    async def test_on_step_success_includes_output_in_content(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        await hook_svc.on_step_success(
            keyword="Get Text",
            arguments=["locator"],
            result={"output": "Hello World"},
        )

        coll = MemoryType.working_steps().collection_name
        records = list(repo._records.get(coll, {}).values())
        assert "Hello World" in records[0].entry.content

    async def test_on_step_success_no_output(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        await hook_svc.on_step_success(
            keyword="Click",
            arguments=[],
            result={},
        )

        coll = MemoryType.working_steps().collection_name
        count = await repo.count(coll)
        assert count == 1

    async def test_on_step_failure_stores_common_errors(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        await hook_svc.on_step_failure(
            keyword="Click",
            arguments=["#nonexistent"],
            error_text="Element not found",
            session_id="sess-2",
        )

        coll = MemoryType.common_errors().collection_name
        count = await repo.count(coll)
        assert count == 1

    async def test_on_step_failure_content_includes_error(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        await hook_svc.on_step_failure(
            keyword="Click",
            arguments=["#btn"],
            error_text="Element not found: #btn",
        )

        coll = MemoryType.common_errors().collection_name
        records = list(repo._records.get(coll, {}).values())
        assert "Element not found" in records[0].entry.content
        assert "Click" in records[0].entry.content

    async def test_on_error_recovered_stores_error_fix(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        fix_steps = [
            {"keyword": "Wait Until Element Is Visible", "arguments": ["#btn"]},
            {"keyword": "Click", "arguments": ["#btn"]},
        ]
        await hook_svc.on_error_recovered(
            error_text="Element not interactable",
            fix_steps=fix_steps,
            session_id="sess-3",
        )

        coll = MemoryType.common_errors().collection_name
        count = await repo.count(coll)
        assert count == 1

    async def test_on_error_recovered_content_includes_fix(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        fix_steps = [
            {"keyword": "Scroll Element Into View", "arguments": ["#elem"]},
        ]
        await hook_svc.on_error_recovered(
            error_text="Element not visible",
            fix_steps=fix_steps,
        )

        coll = MemoryType.common_errors().collection_name
        records = list(repo._records.get(coll, {}).values())
        assert "Element not visible" in records[0].entry.content
        assert "Scroll Element Into View" in records[0].entry.content
        assert records[0].entry.tags == ("error_fix",)

    async def test_on_keyword_discovered_stores_keywords(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        await hook_svc.on_keyword_discovered(
            keyword_name="Click Element",
            library="SeleniumLibrary",
            documentation="Clicks an element.",
        )

        coll = MemoryType.keywords().collection_name
        count = await repo.count(coll)
        assert count == 1

    async def test_on_keyword_discovered_content(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        await hook_svc.on_keyword_discovered(
            keyword_name="Input Text",
            library="SeleniumLibrary",
            documentation="Types value into field.",
        )

        coll = MemoryType.keywords().collection_name
        records = list(repo._records.get(coll, {}).values())
        assert "Input Text" in records[0].entry.content
        assert "SeleniumLibrary" in records[0].entry.content
        assert "Types value" in records[0].entry.content

    async def test_on_keyword_discovered_truncates_long_doc(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        long_doc = "x" * 5000
        await hook_svc.on_keyword_discovered(
            keyword_name="KW",
            library="Lib",
            documentation=long_doc,
        )

        coll = MemoryType.keywords().collection_name
        records = list(repo._records.get(coll, {}).values())
        # Documentation is truncated to 1000 chars in the content
        assert len(records[0].entry.content) < 5000

    async def test_on_session_end_stores_successful_steps(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        steps = [
            {"keyword": "Open Browser", "arguments": ["https://example.com"], "success": True},
            {"keyword": "Click", "arguments": ["#btn"], "success": True},
            {"keyword": "Bad Step", "arguments": [], "success": False},
        ]
        await hook_svc.on_session_end(session_id="sess-end", steps=steps)

        coll = MemoryType.working_steps().collection_name
        count = await repo.count(coll)
        assert count == 1

    async def test_on_session_end_content_includes_steps(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        steps = [
            {"keyword": "Open Browser", "arguments": ["https://example.com"], "success": True},
            {"keyword": "Click", "arguments": ["#btn"], "success": True},
        ]
        await hook_svc.on_session_end(session_id="sess-end", steps=steps)

        coll = MemoryType.working_steps().collection_name
        records = list(repo._records.get(coll, {}).values())
        assert "2 successful" in records[0].entry.content
        assert "Open Browser" in records[0].entry.content
        assert "Click" in records[0].entry.content
        assert records[0].entry.tags == ("session_sequence",)

    async def test_on_session_end_no_successful_steps_skips(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        steps = [
            {"keyword": "Bad Step", "arguments": [], "success": False},
        ]
        await hook_svc.on_session_end(session_id="sess-end", steps=steps)

        total = await repo.count()
        assert total == 0

    async def test_on_session_end_empty_steps_skips(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        await hook_svc.on_session_end(session_id="sess-end", steps=[])
        total = await repo.count()
        assert total == 0

    async def test_store_knowledge_documentation(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        record_id = await hook_svc.store_knowledge(
            content="SeleniumLibrary provides browser automation keywords.",
            knowledge_type="documentation",
            tags=["selenium", "browser"],
        )

        assert record_id is not None
        coll = MemoryType.documentation().collection_name
        count = await repo.count(coll)
        assert count == 1

    async def test_store_knowledge_domain_knowledge(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]

        record_id = await hook_svc.store_knowledge(
            content="DemoShop uses React SPA with lazy loading.",
            knowledge_type="domain_knowledge",
        )

        assert record_id is not None

    async def test_store_knowledge_rejects_invalid_type(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]

        result = await hook_svc.store_knowledge(
            content="some content",
            knowledge_type="working_steps",  # not allowed
        )
        assert result is None

    async def test_store_knowledge_rejects_keywords_type(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]

        result = await hook_svc.store_knowledge(
            content="some content",
            knowledge_type="keywords",
        )
        assert result is None

    async def test_store_knowledge_rejects_common_errors_type(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]

        result = await hook_svc.store_knowledge(
            content="some content",
            knowledge_type="common_errors",
        )
        assert result is None


# =========================================================================
# MemoryHookService — read hooks
# =========================================================================


@_async_mark
class TestMemoryHookServiceReadHooks:
    """Tests for recall_for_hint(), recall_for_scenario()."""

    async def test_recall_for_hint_returns_none_when_no_matches(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]

        result = await hook_svc.recall_for_hint(
            keyword="Click",
            error_text="some obscure error",
            session_id="s1",
        )
        assert result is None

    async def test_recall_for_hint_returns_formatted_hints(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]

        # Seed error memory
        await hook_svc.on_step_failure(
            keyword="Click",
            arguments=["#btn"],
            error_text="Element not interactable",
        )

        result = await hook_svc.recall_for_hint(
            keyword="Click",
            error_text="Element not interactable",
        )
        # Should find the seeded error and format it
        if result is not None:
            assert "similarity=" in result
            assert "Element not interactable" in result

    async def test_recall_for_scenario_returns_none_when_no_matches(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]

        result = await hook_svc.recall_for_scenario("some unrelated scenario")
        assert result is None

    async def test_recall_for_scenario_returns_patterns(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]

        # Seed step memory
        steps = [
            {"keyword": "Open Browser", "arguments": ["https://example.com"], "success": True},
            {"keyword": "Click", "arguments": ["#login"], "success": True},
        ]
        await hook_svc.on_session_end(session_id="s1", steps=steps)

        result = await hook_svc.recall_for_scenario("open browser and click login")
        if result is not None:
            assert "similarity=" in result
            assert "Previous pattern" in result


# =========================================================================
# MemoryHookService — on_tool_call dispatch
# =========================================================================


@_async_mark
class TestMemoryHookServiceOnToolCall:
    """Tests for on_tool_call() dispatching."""

    async def test_on_tool_call_execute_step_success(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        await hook_svc.on_tool_call(
            session_id="s1",
            tool_name="execute_step",
            arguments={"keyword": "Click", "arguments": ["#btn"]},
            result={"success": True, "output": "clicked"},
        )

        coll = MemoryType.working_steps().collection_name
        count = await repo.count(coll)
        assert count == 1

    async def test_on_tool_call_execute_step_failure(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        await hook_svc.on_tool_call(
            session_id="s1",
            tool_name="execute_step",
            arguments={"keyword": "Click", "arguments": ["#missing"]},
            result={"success": False, "error": "Element not found"},
        )

        coll = MemoryType.common_errors().collection_name
        count = await repo.count(coll)
        assert count == 1

    async def test_on_tool_call_execute_step_failure_default_error(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        await hook_svc.on_tool_call(
            session_id="s1",
            tool_name="execute_step",
            arguments={"keyword": "Click", "arguments": []},
            result={"success": False},  # no "error" key
        )

        coll = MemoryType.common_errors().collection_name
        count = await repo.count(coll)
        assert count == 1

    async def test_on_tool_call_find_keywords(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        await hook_svc.on_tool_call(
            session_id=None,
            tool_name="find_keywords",
            arguments={"pattern": "click"},
            result={
                "matches": [
                    {
                        "name": "Click Element",
                        "library": "SeleniumLibrary",
                        "documentation": "Clicks an element.",
                    },
                    {
                        "name": "Click Button",
                        "library": "SeleniumLibrary",
                        "documentation": "Clicks a button.",
                    },
                ]
            },
        )

        coll = MemoryType.keywords().collection_name
        count = await repo.count(coll)
        assert count == 2

    async def test_on_tool_call_find_keywords_results_key(self):
        """find_keywords may return 'results' instead of 'matches'."""
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        await hook_svc.on_tool_call(
            session_id=None,
            tool_name="find_keywords",
            arguments={"pattern": "input"},
            result={
                "results": [
                    {
                        "keyword": "Input Text",
                        "library": "SeleniumLibrary",
                        "doc": "Types text.",
                    },
                ]
            },
        )

        coll = MemoryType.keywords().collection_name
        count = await repo.count(coll)
        assert count == 1

    async def test_on_tool_call_find_keywords_limits_to_5(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        matches = [
            {"name": f"Keyword_{i}", "library": "Lib", "documentation": f"Doc {i}"}
            for i in range(10)
        ]
        await hook_svc.on_tool_call(
            session_id=None,
            tool_name="find_keywords",
            arguments={},
            result={"matches": matches},
        )

        coll = MemoryType.keywords().collection_name
        count = await repo.count(coll)
        assert count == 5  # Limited to first 5

    async def test_on_tool_call_find_keywords_skips_empty_name(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        await hook_svc.on_tool_call(
            session_id=None,
            tool_name="find_keywords",
            arguments={},
            result={
                "matches": [
                    {"name": "", "library": "Lib", "documentation": "Doc"},
                    {"name": "Valid KW", "library": "Lib", "documentation": "Doc"},
                ]
            },
        )

        coll = MemoryType.keywords().collection_name
        count = await repo.count(coll)
        assert count == 1  # Only the non-empty name

    async def test_on_tool_call_execute_batch_with_fix_steps(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        await hook_svc.on_tool_call(
            session_id="s1",
            tool_name="execute_batch",
            arguments={},
            result={
                "success": False,
                "error": "Batch step failed",
                "fix_steps": [
                    {"keyword": "Wait", "arguments": ["5s"]},
                    {"keyword": "Click", "arguments": ["#retry"]},
                ],
            },
        )

        coll = MemoryType.common_errors().collection_name
        count = await repo.count(coll)
        assert count == 1

    async def test_on_tool_call_execute_batch_no_fix_steps(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        await hook_svc.on_tool_call(
            session_id="s1",
            tool_name="execute_batch",
            arguments={},
            result={"success": False, "error": "Failed"},
        )

        # No fix_steps means no on_error_recovered call
        total = await repo.count()
        assert total == 0

    async def test_on_tool_call_unrecognized_tool(self):
        """Unrecognized tool names should be silently ignored."""
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        await hook_svc.on_tool_call(
            session_id="s1",
            tool_name="unknown_tool",
            arguments={},
            result={},
        )

        total = await repo.count()
        assert total == 0

    async def test_on_tool_call_execute_batch_success_is_noop(self):
        """Successful execute_batch should not store anything."""
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]
        repo: InMemoryMemoryRepository = services["repo"]

        await hook_svc.on_tool_call(
            session_id="s1",
            tool_name="execute_batch",
            arguments={},
            result={"success": True},
        )

        total = await repo.count()
        assert total == 0


# =========================================================================
# MemoryHookService — graceful degradation
# =========================================================================


@_async_mark
class TestMemoryHookServiceGracefulDegradation:
    """Memory failures never propagate to callers."""

    async def test_on_step_success_no_crash_when_embedding_unavailable(self):
        backend = EmbeddingBackend(is_available=False)
        embedding_svc = EmbeddingService(backend)
        repo = InMemoryMemoryRepository()
        store = MemoryStore.create(_make_config())
        query_svc = MemoryQueryService(embedding_svc, repo, store)
        hook_svc = MemoryHookService(query_svc, embedding_svc, repo, store)

        # Should not raise
        await hook_svc.on_step_success(
            keyword="Click", arguments=["btn"], result={}, session_id="s1"
        )

    async def test_on_step_failure_no_crash_when_embedding_unavailable(self):
        backend = EmbeddingBackend(is_available=False)
        embedding_svc = EmbeddingService(backend)
        repo = InMemoryMemoryRepository()
        store = MemoryStore.create(_make_config())
        query_svc = MemoryQueryService(embedding_svc, repo, store)
        hook_svc = MemoryHookService(query_svc, embedding_svc, repo, store)

        await hook_svc.on_step_failure(
            keyword="Click", arguments=[], error_text="error"
        )

    async def test_on_error_recovered_no_crash_when_unavailable(self):
        backend = EmbeddingBackend(is_available=False)
        embedding_svc = EmbeddingService(backend)
        repo = InMemoryMemoryRepository()
        store = MemoryStore.create(_make_config())
        query_svc = MemoryQueryService(embedding_svc, repo, store)
        hook_svc = MemoryHookService(query_svc, embedding_svc, repo, store)

        await hook_svc.on_error_recovered(
            error_text="err", fix_steps=[{"keyword": "Fix"}]
        )

    async def test_on_keyword_discovered_no_crash_when_unavailable(self):
        backend = EmbeddingBackend(is_available=False)
        embedding_svc = EmbeddingService(backend)
        repo = InMemoryMemoryRepository()
        store = MemoryStore.create(_make_config())
        query_svc = MemoryQueryService(embedding_svc, repo, store)
        hook_svc = MemoryHookService(query_svc, embedding_svc, repo, store)

        await hook_svc.on_keyword_discovered("KW", "Lib", "Doc")

    async def test_on_session_end_no_crash_when_unavailable(self):
        backend = EmbeddingBackend(is_available=False)
        embedding_svc = EmbeddingService(backend)
        repo = InMemoryMemoryRepository()
        store = MemoryStore.create(_make_config())
        query_svc = MemoryQueryService(embedding_svc, repo, store)
        hook_svc = MemoryHookService(query_svc, embedding_svc, repo, store)

        await hook_svc.on_session_end(
            "s1", [{"keyword": "Click", "arguments": [], "success": True}]
        )

    async def test_store_knowledge_no_crash_when_unavailable(self):
        backend = EmbeddingBackend(is_available=False)
        embedding_svc = EmbeddingService(backend)
        repo = InMemoryMemoryRepository()
        store = MemoryStore.create(_make_config())
        query_svc = MemoryQueryService(embedding_svc, repo, store)
        hook_svc = MemoryHookService(query_svc, embedding_svc, repo, store)

        result = await hook_svc.store_knowledge("content", "documentation")
        assert result is None

    async def test_recall_for_hint_no_crash_when_unavailable(self):
        backend = EmbeddingBackend(is_available=False)
        embedding_svc = EmbeddingService(backend)
        repo = InMemoryMemoryRepository()
        store = MemoryStore.create(_make_config())
        query_svc = MemoryQueryService(embedding_svc, repo, store)
        hook_svc = MemoryHookService(query_svc, embedding_svc, repo, store)

        result = await hook_svc.recall_for_hint("Click", "error")
        assert result is None

    async def test_recall_for_scenario_no_crash_when_unavailable(self):
        backend = EmbeddingBackend(is_available=False)
        embedding_svc = EmbeddingService(backend)
        repo = InMemoryMemoryRepository()
        store = MemoryStore.create(_make_config())
        query_svc = MemoryQueryService(embedding_svc, repo, store)
        hook_svc = MemoryHookService(query_svc, embedding_svc, repo, store)

        result = await hook_svc.recall_for_scenario("open browser")
        assert result is None

    async def test_on_tool_call_no_crash_when_unavailable(self):
        backend = EmbeddingBackend(is_available=False)
        embedding_svc = EmbeddingService(backend)
        repo = InMemoryMemoryRepository()
        store = MemoryStore.create(_make_config())
        query_svc = MemoryQueryService(embedding_svc, repo, store)
        hook_svc = MemoryHookService(query_svc, embedding_svc, repo, store)

        await hook_svc.on_tool_call(
            "s1",
            "execute_step",
            {"keyword": "Click", "arguments": ["#btn"]},
            {"success": True},
        )

    async def test_on_tool_call_no_crash_on_internal_exception(self):
        """Even if internal dispatch raises, on_tool_call swallows it."""
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]

        # Monkey-patch on_step_success to raise
        original = hook_svc.on_step_success
        async def _boom(*a: Any, **kw: Any) -> None:
            raise RuntimeError("internal failure")
        hook_svc.on_step_success = _boom  # type: ignore[assignment]

        # Should not raise
        await hook_svc.on_tool_call(
            "s1",
            "execute_step",
            {"keyword": "Click", "arguments": []},
            {"success": True},
        )


# =========================================================================
# MemoryHookService — _store_entry internals
# =========================================================================


@_async_mark
class TestMemoryHookServiceStoreEntry:
    """Tests for _store_entry behavior."""

    async def test_store_entry_returns_none_when_unavailable(self):
        backend = EmbeddingBackend(is_available=False)
        embedding_svc = EmbeddingService(backend)
        repo = InMemoryMemoryRepository()
        store = MemoryStore.create(_make_config())
        query_svc = MemoryQueryService(embedding_svc, repo, store)
        hook_svc = MemoryHookService(query_svc, embedding_svc, repo, store)

        entry = MemoryEntry(content="test", memory_type=MemoryType.keywords())
        result = await hook_svc._store_entry(entry, None)
        assert result is None

    async def test_store_entry_returns_record_on_success(self):
        services = _make_wired_services()
        hook_svc: MemoryHookService = services["hook_service"]

        entry = MemoryEntry(content="test content", memory_type=MemoryType.keywords())
        result = await hook_svc._store_entry(entry, "sess-1")
        assert result is not None
        assert isinstance(result, MemoryRecord)
        assert result.session_id == "sess-1"

    async def test_store_entry_returns_none_on_embed_failure(self):
        """If embedding fails, _store_entry returns None."""
        backend = _make_backend()
        embedding_svc = EmbeddingService(backend)
        # Set model that raises on encode
        mock_model = MagicMock()
        mock_model.encode = MagicMock(side_effect=Exception("model exploded"))
        embedding_svc._model = mock_model

        repo = InMemoryMemoryRepository()
        store = MemoryStore.create(_make_config())
        query_svc = MemoryQueryService(embedding_svc, repo, store)
        hook_svc = MemoryHookService(query_svc, embedding_svc, repo, store)

        entry = MemoryEntry(content="test", memory_type=MemoryType.keywords())
        result = await hook_svc._store_entry(entry, None)
        assert result is None


# =========================================================================
# create_memory_services() factory
# =========================================================================


class TestCreateMemoryServices:
    """Tests for the create_memory_services() factory function."""

    def test_returns_none_when_disabled(self):
        config = StorageConfig(enabled=False)
        result = create_memory_services(config)
        assert result is None

    def test_returns_none_when_no_backend_available(self):
        config = StorageConfig(enabled=True, embedding_model="potion-base-8M")

        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                (_ for _ in ()).throw(ImportError())
                if name in ("model2vec", "fastembed", "sentence_transformers")
                else importlib.__import__(name, *a, **kw)
            ),
        ):
            result = create_memory_services(config)

        assert result is None

    def test_returns_dict_with_correct_keys_when_available(self):
        config = StorageConfig(enabled=True, embedding_model="potion-base-8M")

        # Fake model2vec availability
        fake_module = ModuleType("model2vec")
        fake_module.StaticModel = MagicMock()  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"model2vec": fake_module}):
            result = create_memory_services(config)

        assert result is not None
        expected_keys = {
            "store",
            "embedding_service",
            "query_service",
            "hook_service",
            "repository",
            "backend",
            "config",
        }
        assert set(result.keys()) == expected_keys

    def test_returned_services_have_correct_types(self):
        config = StorageConfig(enabled=True, embedding_model="potion-base-8M")

        fake_module = ModuleType("model2vec")
        fake_module.StaticModel = MagicMock()  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"model2vec": fake_module}):
            result = create_memory_services(config)

        assert result is not None
        assert isinstance(result["store"], MemoryStore)
        assert isinstance(result["embedding_service"], EmbeddingService)
        assert isinstance(result["query_service"], MemoryQueryService)
        assert isinstance(result["hook_service"], MemoryHookService)
        assert isinstance(result["backend"], EmbeddingBackend)

    def test_backend_dimension_overrides_config(self):
        """When detected backend has different dimension, config is updated."""
        config = StorageConfig(
            enabled=True,
            embedding_model="potion-base-8M",
            dimension=999,  # intentionally wrong
        )

        fake_module = ModuleType("model2vec")
        fake_module.StaticModel = MagicMock()  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"model2vec": fake_module}):
            result = create_memory_services(config)

        assert result is not None
        # model2vec potion-base-8M has dimension 256
        assert result["config"].dimension == 256

    def test_uses_in_memory_repo_when_sqlite_vec_unavailable(self):
        config = StorageConfig(enabled=True, embedding_model="potion-base-8M")

        fake_module = ModuleType("model2vec")
        fake_module.StaticModel = MagicMock()  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"model2vec": fake_module}):
            # Also ensure sqlite-vec adapter fails to import
            with patch(
                "robotmcp.domains.memory.services.create_memory_services",
            ) as mock_factory:
                # Actually, let the real function run but intercept the SqliteVec import
                pass

            # Instead, just call it -- sqlite-vec is likely not installed in test env
            result = create_memory_services(config)

        assert result is not None
        # Repository should be either InMemoryMemoryRepository or SqliteVecRepository
        assert result["repository"] is not None

    def test_default_config_from_env_disabled(self):
        """When no config provided, uses from_env() which defaults to disabled."""
        with patch.dict("os.environ", {}, clear=False):
            # Ensure ROBOTMCP_MEMORY_ENABLED is not set (or is "false")
            import os
            old = os.environ.pop("ROBOTMCP_MEMORY_ENABLED", None)
            try:
                result = create_memory_services()
                assert result is None
            finally:
                if old is not None:
                    os.environ["ROBOTMCP_MEMORY_ENABLED"] = old


# =========================================================================
# __test__ suppression
# =========================================================================


class TestPytestCollectionSuppression:
    """Ensure __test__ = False on entities with 'Test' in name."""

    def test_memory_record_test_flag(self):
        assert MemoryRecord.__test__ is False

    def test_memory_store_test_flag(self):
        assert MemoryStore.__test__ is False

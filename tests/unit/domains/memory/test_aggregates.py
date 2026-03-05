"""Tests for memory domain aggregates.

Covers: MemoryStore, EmbeddingBackend.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import patch

import pytest

from robotmcp.domains.memory.aggregates import EmbeddingBackend, MemoryStore
from robotmcp.domains.memory.entities import MemoryCollection
from robotmcp.domains.memory.events import (
    CollectionCreated,
    MemoryPruned,
    MemoryStored,
)
from robotmcp.domains.memory.value_objects import (
    MemoryEntry,
    MemoryQuery,
    MemoryType,
    StorageConfig,
)


# =========================================================================
# MemoryStore
# =========================================================================


class TestMemoryStore:
    # -- create() ------------------------------------------------------------

    def test_create_default_config(self):
        store = MemoryStore.create()
        assert store.store_id is not None
        assert len(store.store_id) == 36
        assert store.config.enabled is False
        assert store.collections == {}
        assert store.time_decay.half_life_days == 30.0

    def test_create_with_custom_config(self):
        config = StorageConfig(
            dimension=384,
            time_decay_half_life=60.0,
        )
        store = MemoryStore.create(config)
        assert store.config.dimension == 384
        assert store.time_decay.half_life_days == 60.0

    def test_create_unique_ids(self):
        s1 = MemoryStore.create()
        s2 = MemoryStore.create()
        assert s1.store_id != s2.store_id

    # -- ensure_collection ---------------------------------------------------

    def test_ensure_collection_creates_new(self):
        store = MemoryStore.create()
        mt = MemoryType.keywords()
        coll = store.ensure_collection(mt)

        assert isinstance(coll, MemoryCollection)
        assert coll.collection_id == "rfmcp_keywords"
        assert coll.dimension == 256

    def test_ensure_collection_idempotent(self):
        store = MemoryStore.create()
        mt = MemoryType.keywords()
        c1 = store.ensure_collection(mt)
        c2 = store.ensure_collection(mt)
        assert c1 is c2

    def test_ensure_collection_emits_event_once(self):
        store = MemoryStore.create()
        mt = MemoryType.keywords()
        store.ensure_collection(mt)
        store.ensure_collection(mt)
        events = store.collect_events()
        created_events = [e for e in events if isinstance(e, CollectionCreated)]
        assert len(created_events) == 1

    def test_ensure_collection_event_content(self):
        store = MemoryStore.create()
        mt = MemoryType.documentation()
        store.ensure_collection(mt)
        events = store.collect_events()
        ev = events[0]
        assert isinstance(ev, CollectionCreated)
        assert ev.collection_id == "rfmcp_documentation"
        assert ev.memory_type == "documentation"
        assert ev.dimension == 256

    def test_ensure_collection_uses_config_dimension(self):
        config = StorageConfig(dimension=384)
        store = MemoryStore.create(config)
        coll = store.ensure_collection(MemoryType.keywords())
        assert coll.dimension == 384

    # -- get_collection ------------------------------------------------------

    def test_get_collection_exists(self):
        store = MemoryStore.create()
        mt = MemoryType.keywords()
        store.ensure_collection(mt)
        coll = store.get_collection(mt)
        assert coll is not None
        assert coll.collection_id == "rfmcp_keywords"

    def test_get_collection_not_exists(self):
        store = MemoryStore.create()
        assert store.get_collection(MemoryType.keywords()) is None

    # -- has_collection ------------------------------------------------------

    def test_has_collection_true(self):
        store = MemoryStore.create()
        mt = MemoryType.keywords()
        store.ensure_collection(mt)
        assert store.has_collection(mt) is True

    def test_has_collection_false(self):
        store = MemoryStore.create()
        assert store.has_collection(MemoryType.keywords()) is False

    # -- prepare_store -------------------------------------------------------

    def test_prepare_store_creates_record(self):
        store = MemoryStore.create()
        entry = MemoryEntry(content="hello", memory_type=MemoryType.keywords())
        record = store.prepare_store(entry)

        assert record.record_id is not None
        assert record.entry is entry
        assert record.session_id is None

    def test_prepare_store_with_session_id(self):
        store = MemoryStore.create()
        entry = MemoryEntry(content="hello", memory_type=MemoryType.keywords())
        record = store.prepare_store(entry, session_id="s1")
        assert record.session_id == "s1"

    def test_prepare_store_ensures_collection(self):
        store = MemoryStore.create()
        entry = MemoryEntry(content="hello", memory_type=MemoryType.keywords())
        store.prepare_store(entry)
        assert store.has_collection(MemoryType.keywords()) is True

    def test_prepare_store_increments_count(self):
        store = MemoryStore.create()
        entry = MemoryEntry(content="hello", memory_type=MemoryType.keywords())
        store.prepare_store(entry)
        coll = store.get_collection(MemoryType.keywords())
        assert coll.record_count == 1

    def test_prepare_store_emits_memory_stored_event(self):
        store = MemoryStore.create()
        entry = MemoryEntry(content="test content here", memory_type=MemoryType.keywords())
        record = store.prepare_store(entry, session_id="sess")
        events = store.collect_events()

        stored_events = [e for e in events if isinstance(e, MemoryStored)]
        assert len(stored_events) == 1
        ev = stored_events[0]
        assert ev.record_id == record.record_id
        assert ev.memory_type == "keywords"
        assert ev.content_preview == "test content here"
        assert ev.collection_id == "rfmcp_keywords"
        assert ev.session_id == "sess"

    def test_prepare_store_multiple_records(self):
        store = MemoryStore.create()
        for i in range(5):
            entry = MemoryEntry(
                content=f"content {i}", memory_type=MemoryType.keywords()
            )
            store.prepare_store(entry)
        coll = store.get_collection(MemoryType.keywords())
        assert coll.record_count == 5

    def test_prepare_store_different_types(self):
        store = MemoryStore.create()
        for mt_val in ["keywords", "documentation", "common_errors"]:
            entry = MemoryEntry(content="c", memory_type=MemoryType(mt_val))
            store.prepare_store(entry)
        assert store.collection_count == 3
        assert store.total_record_count == 3

    # -- prepare_recall ------------------------------------------------------

    def test_prepare_recall_scoped(self):
        store = MemoryStore.create()
        query = MemoryQuery(query_text="test", memory_type=MemoryType.keywords())
        result = store.prepare_recall(query)
        assert result is query
        assert store.has_collection(MemoryType.keywords()) is True

    def test_prepare_recall_unscoped(self):
        store = MemoryStore.create()
        query = MemoryQuery(query_text="test")
        result = store.prepare_recall(query)
        assert result is query
        # No collection should be created for unscoped query
        assert store.collection_count == 0

    # -- mark_for_pruning ----------------------------------------------------

    def test_mark_for_pruning_emits_event(self):
        store = MemoryStore.create()
        mt = MemoryType.keywords()
        store.ensure_collection(mt)
        store.collect_events()  # Clear creation event

        store.mark_for_pruning(mt)
        events = store.collect_events()
        assert len(events) == 1
        assert isinstance(events[0], MemoryPruned)
        assert events[0].collection_id == "rfmcp_keywords"
        assert events[0].memory_type == "keywords"
        assert events[0].records_removed == 0
        assert events[0].max_age_days == 90.0  # Config default

    def test_mark_for_pruning_custom_age(self):
        store = MemoryStore.create()
        mt = MemoryType.keywords()
        store.ensure_collection(mt)
        store.collect_events()

        store.mark_for_pruning(mt, max_age_days=30.0)
        events = store.collect_events()
        assert events[0].max_age_days == 30.0

    def test_mark_for_pruning_no_collection(self):
        store = MemoryStore.create()
        mt = MemoryType.keywords()
        store.mark_for_pruning(mt)
        events = store.collect_events()
        # No event when collection does not exist
        assert len(events) == 0

    # -- collect_events ------------------------------------------------------

    def test_collect_events_empties_queue(self):
        store = MemoryStore.create()
        store.ensure_collection(MemoryType.keywords())
        events = store.collect_events()
        assert len(events) == 1

        events2 = store.collect_events()
        assert len(events2) == 0

    def test_collect_events_returns_list_copy(self):
        store = MemoryStore.create()
        store.ensure_collection(MemoryType.keywords())
        events = store.collect_events()
        assert isinstance(events, list)
        # Modifying returned list should not affect internal state
        events.append("bogus")
        assert len(store.collect_events()) == 0

    # -- Stats ---------------------------------------------------------------

    def test_total_record_count_empty(self):
        store = MemoryStore.create()
        assert store.total_record_count == 0

    def test_total_record_count_multiple(self):
        store = MemoryStore.create()
        for mt_val in ["keywords", "documentation"]:
            entry = MemoryEntry(content="c", memory_type=MemoryType(mt_val))
            store.prepare_store(entry)
            store.prepare_store(entry)
        assert store.total_record_count == 4

    def test_collection_count(self):
        store = MemoryStore.create()
        assert store.collection_count == 0
        store.ensure_collection(MemoryType.keywords())
        assert store.collection_count == 1
        store.ensure_collection(MemoryType.documentation())
        assert store.collection_count == 2

    # -- to_dict -------------------------------------------------------------

    def test_to_dict(self):
        config = StorageConfig(dimension=256, enabled=True, embedding_model="potion-base-8M")
        store = MemoryStore.create(config)
        entry = MemoryEntry(content="c", memory_type=MemoryType.keywords())
        store.prepare_store(entry)
        store.collect_events()  # Clear

        d = store.to_dict()
        assert d["store_id"] == store.store_id
        assert d["enabled"] is True
        assert d["embedding_model"] == "potion-base-8M"
        assert d["dimension"] == 256
        assert d["total_records"] == 1
        assert "rfmcp_keywords" in d["collections"]

    def test_to_dict_empty(self):
        store = MemoryStore.create()
        d = store.to_dict()
        assert d["total_records"] == 0
        assert d["collections"] == {}

    # -- validate ------------------------------------------------------------

    def test_validate_no_errors(self):
        store = MemoryStore.create()
        store.ensure_collection(MemoryType.keywords())
        errors = store.validate()
        assert errors == []

    def test_validate_dimension_mismatch(self):
        config = StorageConfig(dimension=256)
        store = MemoryStore.create(config)
        store.ensure_collection(MemoryType.keywords())
        # Manually tamper with collection dimension
        coll = store.get_collection(MemoryType.keywords())
        coll.dimension = 384
        errors = store.validate()
        assert len(errors) == 1
        assert "dimension" in errors[0]

    def test_validate_multiple_mismatches(self):
        store = MemoryStore.create()
        store.ensure_collection(MemoryType.keywords())
        store.ensure_collection(MemoryType.documentation())
        for coll in store.collections.values():
            coll.dimension = 999
        errors = store.validate()
        assert len(errors) == 2

    # -- __test__ suppression ------------------------------------------------

    def test_test_flag_suppression(self):
        assert MemoryStore.__test__ is False


# =========================================================================
# EmbeddingBackend
# =========================================================================


class TestEmbeddingBackend:
    # -- detect with model2vec available -------------------------------------

    def test_detect_model2vec_available(self):
        """When model2vec is importable, detect should return model2vec backend."""
        fake_module = ModuleType("model2vec")
        fake_module.StaticModel = object  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"model2vec": fake_module}):
            backend = EmbeddingBackend.detect("potion-base-8M")

        assert backend.backend_name == "model2vec"
        assert backend.model_name == "minishlab/potion-base-8M"
        assert backend.dimension == 256
        assert backend.is_available is True

    def test_detect_model2vec_unknown_model_tries_fastembed(self):
        """When model is not in model2vec map, fall through to fastembed."""
        fake_m2v = ModuleType("model2vec")
        fake_m2v.StaticModel = object  # type: ignore[attr-defined]
        fake_fe = ModuleType("fastembed")
        fake_fe.TextEmbedding = object  # type: ignore[attr-defined]

        with patch.dict(
            sys.modules,
            {"model2vec": fake_m2v, "fastembed": fake_fe},
        ):
            backend = EmbeddingBackend.detect("unknown-model-xyz")

        assert backend.backend_name == "fastembed"

    # -- detect with fastembed available -------------------------------------

    def test_detect_fastembed_available(self):
        """When only fastembed is importable, detect should use it."""
        fake_module = ModuleType("fastembed")
        fake_module.TextEmbedding = object  # type: ignore[attr-defined]

        with patch.dict(
            sys.modules,
            {"model2vec": None, "fastembed": fake_module},  # type: ignore[dict-item]
        ):
            # model2vec import will raise ImportError because value is None
            # Actually we need to make sure model2vec import fails
            pass

        # Use a more direct approach: remove model2vec, add fastembed
        import importlib

        with patch.dict(sys.modules, {"fastembed": fake_module}):
            # Force model2vec import to fail
            with patch(
                "builtins.__import__",
                side_effect=lambda name, *a, **kw: (
                    fake_module if name == "fastembed"
                    else (_ for _ in ()).throw(ImportError())
                    if name == "model2vec"
                    else importlib.__import__(name, *a, **kw)
                ),
            ):
                backend = EmbeddingBackend.detect()

        assert backend.backend_name == "fastembed"
        assert backend.model_name == "BAAI/bge-small-en-v1.5"
        assert backend.dimension == 384
        assert backend.is_available is True

    # -- detect with sentence-transformers available -------------------------

    def test_detect_sentence_transformers_available(self):
        """When only sentence-transformers is importable."""
        import importlib

        fake_st = ModuleType("sentence_transformers")
        fake_st.SentenceTransformer = object  # type: ignore[attr-defined]

        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                fake_st if name == "sentence_transformers"
                else (_ for _ in ()).throw(ImportError())
                if name in ("model2vec", "fastembed")
                else importlib.__import__(name, *a, **kw)
            ),
        ):
            backend = EmbeddingBackend.detect()

        assert backend.backend_name == "sentence-transformers"
        assert backend.model_name == "all-MiniLM-L6-v2"
        assert backend.dimension == 384
        assert backend.is_available is True

    # -- detect with nothing available ---------------------------------------

    def test_detect_nothing_available(self):
        """When no backends are importable, detect returns unavailable."""
        import importlib

        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                (_ for _ in ()).throw(ImportError())
                if name in ("model2vec", "fastembed", "sentence_transformers")
                else importlib.__import__(name, *a, **kw)
            ),
        ):
            backend = EmbeddingBackend.detect()

        assert backend.is_available is False
        assert backend.backend_name is None
        assert backend.model_name is None
        assert backend.dimension is None

    # -- collect_events ------------------------------------------------------

    def test_collect_events_empty(self):
        backend = EmbeddingBackend()
        events = backend.collect_events()
        assert events == []

    def test_collect_events_clears(self):
        backend = EmbeddingBackend()
        backend._pending_events.append("test_event")
        events = backend.collect_events()
        assert events == ["test_event"]
        assert backend.collect_events() == []

    # -- to_dict -------------------------------------------------------------

    def test_to_dict_available(self):
        backend = EmbeddingBackend(
            backend_name="model2vec",
            model_name="minishlab/potion-base-8M",
            dimension=256,
            is_available=True,
        )
        d = backend.to_dict()
        assert d == {
            "backend_name": "model2vec",
            "model_name": "minishlab/potion-base-8M",
            "dimension": 256,
            "is_available": True,
        }

    def test_to_dict_unavailable(self):
        backend = EmbeddingBackend()
        d = backend.to_dict()
        assert d == {
            "backend_name": None,
            "model_name": None,
            "dimension": None,
            "is_available": False,
        }

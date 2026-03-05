"""Memory Domain — Services.

Domain services coordinating memory operations, following ADR-001 conventions.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .aggregates import EmbeddingBackend, MemoryStore
from .entities import MemoryRecord
from .events import MemoryRecalled
from .repository import MemoryRepository
from .value_objects import (
    ConfidenceScore,
    EmbeddingVector,
    LocatorDescription,
    LocatorRecallResult,
    LocatorStrategy,
    MemoryEntry,
    MemoryQuery,
    MemoryType,
    RecallResult,
    SimilarityScore,
    StorageConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EmbeddingService
# ---------------------------------------------------------------------------

class EmbeddingService:
    """Generate embeddings from text, with backend auto-detection.

    Lazy-loads the embedding model on first use. Supports three
    backends in priority order: model2vec > fastembed > sentence-transformers.
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        event_publisher: Optional[Callable] = None,
    ) -> None:
        self._backend = backend
        self._model: Any = None
        self._event_publisher = event_publisher

    @property
    def is_available(self) -> bool:
        return self._backend.is_available

    @property
    def model_info(self) -> Dict[str, Any]:
        return self._backend.to_dict()

    def _ensure_model(self) -> None:
        """Lazy-load the embedding model."""
        if self._model is not None:
            return
        if not self._backend.is_available:
            raise RuntimeError("No embedding backend available")

        if self._backend.backend_name == "model2vec":
            from model2vec import StaticModel

            self._model = StaticModel.from_pretrained(self._backend.model_name)
            logger.info(
                "Loaded model2vec model: %s (dim=%d)",
                self._backend.model_name,
                self._backend.dimension,
            )
        elif self._backend.backend_name == "fastembed":
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._backend.model_name)
            logger.info(
                "Loaded fastembed model: %s (dim=%d)",
                self._backend.model_name,
                self._backend.dimension,
            )
        elif self._backend.backend_name == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._backend.model_name)
            logger.info(
                "Loaded sentence-transformers model: %s (dim=%d)",
                self._backend.model_name,
                self._backend.dimension,
            )

    async def embed(self, text: str) -> EmbeddingVector:
        """Generate embedding for a single text."""
        self._ensure_model()
        text = text.strip()[:2000]  # Truncate for safety

        if self._backend.backend_name == "model2vec":
            vectors = self._model.encode([text])
            values = vectors[0].tolist()
        elif self._backend.backend_name == "fastembed":
            embeddings = list(self._model.embed([text]))
            values = embeddings[0].tolist()
        elif self._backend.backend_name == "sentence-transformers":
            vectors = self._model.encode([text])
            values = vectors[0].tolist()
        else:
            raise RuntimeError(f"Unknown backend: {self._backend.backend_name}")

        return EmbeddingVector.from_list(
            values=values,
            model_name=self._backend.model_name or "",
        )

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingVector]:
        """Generate embeddings for multiple texts."""
        self._ensure_model()
        cleaned = [t.strip()[:2000] for t in texts]

        if self._backend.backend_name == "model2vec":
            vectors = self._model.encode(cleaned)
            return [
                EmbeddingVector.from_list(v.tolist(), self._backend.model_name or "")
                for v in vectors
            ]
        elif self._backend.backend_name == "fastembed":
            embeddings = list(self._model.embed(cleaned))
            return [
                EmbeddingVector.from_list(e.tolist(), self._backend.model_name or "")
                for e in embeddings
            ]
        elif self._backend.backend_name == "sentence-transformers":
            vectors = self._model.encode(cleaned)
            return [
                EmbeddingVector.from_list(v.tolist(), self._backend.model_name or "")
                for v in vectors
            ]
        else:
            raise RuntimeError(f"Unknown backend: {self._backend.backend_name}")


# ---------------------------------------------------------------------------
# MemoryQueryService
# ---------------------------------------------------------------------------

class MemoryQueryService:
    """Execute memory queries with time-decay scoring.

    Pipeline:
    1. Validate query via MemoryStore.prepare_recall()
    2. Embed query text via EmbeddingService.embed()
    3. Search via MemoryRepository.search() (over-fetch 2x for re-ranking)
    4. Apply time decay per TimeDecayFactor
    5. Filter by min_similarity, sort by adjusted score
    6. Truncate to top_k
    7. Publish MemoryRecalled event
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        repository: MemoryRepository,
        store: MemoryStore,
        event_publisher: Optional[Callable] = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._repository = repository
        self._store = store
        self._event_publisher = event_publisher

    async def recall(self, query: MemoryQuery) -> List[RecallResult]:
        """Execute a memory recall query."""
        if not self._embedding_service.is_available:
            return []

        start_time = time.monotonic()

        try:
            # 1. Validate
            query = self._store.prepare_recall(query)

            # 2. Embed query
            query_embedding = await self._embedding_service.embed(query.query_text)

            # 3. Search (over-fetch 2x for re-ranking)
            raw_results = await self._repository.search(
                collection_names=query.collection_names,
                query_embedding=query_embedding,
                top_k=query.top_k * 2,
                min_similarity=0.0,  # Filter after decay
            )

            # 4-5. Apply time decay and filter
            recall_results: List[RecallResult] = []
            for record, similarity in raw_results:
                if query.apply_time_decay:
                    adjusted = self._store.time_decay.compute(
                        similarity, record.age_days
                    )
                else:
                    adjusted = similarity

                if adjusted.value < query.min_similarity:
                    continue

                confidence = None
                if record.entry.memory_type.value == MemoryType.COMMON_ERRORS:
                    confidence = ConfidenceScore(value=adjusted.value)

                recall_results.append(
                    RecallResult(
                        record_id=record.record_id,
                        content=record.entry.content,
                        memory_type=record.entry.memory_type,
                        similarity=similarity,
                        adjusted_similarity=adjusted,
                        age_days=record.age_days,
                        metadata=dict(record.entry.metadata)
                        if record.entry.metadata
                        else {},
                        confidence=confidence,
                        rank=0,
                    )
                )

            # 6. Sort and truncate
            recall_results.sort(
                key=lambda r: r.adjusted_similarity.value, reverse=True
            )
            recall_results = recall_results[: query.top_k]

            # Assign ranks
            ranked = []
            for i, r in enumerate(recall_results):
                ranked.append(
                    RecallResult(
                        record_id=r.record_id,
                        content=r.content,
                        memory_type=r.memory_type,
                        similarity=r.similarity,
                        adjusted_similarity=r.adjusted_similarity,
                        age_days=r.age_days,
                        metadata=r.metadata,
                        confidence=r.confidence,
                        rank=i + 1,
                    )
                )

            # 7. Record access and publish event
            elapsed_ms = (time.monotonic() - start_time) * 1000
            top_sim = ranked[0].adjusted_similarity.value if ranked else 0.0

            if self._event_publisher:
                self._event_publisher(
                    MemoryRecalled(
                        query_text=query.query_text[:200],
                        memory_type=(
                            query.memory_type.value if query.memory_type else None
                        ),
                        result_count=len(ranked),
                        top_similarity=top_sim,
                        query_time_ms=elapsed_ms,
                    )
                )

            return ranked

        except Exception as e:
            logger.debug("Memory recall failed: %s", e)
            return []

    async def recall_for_error(
        self, error_text: str, session_id: Optional[str] = None
    ) -> List[RecallResult]:
        return await self.recall(
            MemoryQuery.for_error_fix(error_text, session_id)
        )

    async def recall_keywords(self, keyword_hint: str) -> List[RecallResult]:
        return await self.recall(MemoryQuery.for_keyword_recall(keyword_hint))

    async def recall_steps(self, scenario: str) -> List[RecallResult]:
        return await self.recall(MemoryQuery.for_step_recall(scenario))

    async def recall_locators(
        self, element_description: str
    ) -> List[LocatorRecallResult]:
        """Recall structured locator mappings for an element."""
        results = await self.recall(
            MemoryQuery.for_locator_recall(element_description)
        )
        locator_results: List[LocatorRecallResult] = []
        for r in results:
            meta = r.metadata or {}
            strategy_val = meta.get("strategy", "auto")
            try:
                strategy = LocatorStrategy(strategy_val)
            except ValueError:
                strategy = LocatorStrategy.AUTO
            locator_results.append(
                LocatorRecallResult(
                    locator=meta.get("locator", ""),
                    strategy=strategy,
                    keyword=meta.get("keyword", ""),
                    library=meta.get("library", ""),
                    outcome=meta.get("outcome", "success"),
                    page_url=meta.get("page_url", ""),
                    description=meta.get("description", ""),
                    similarity=r.adjusted_similarity.value,
                    error_text=meta.get("error_text", ""),
                )
            )
        return locator_results


# ---------------------------------------------------------------------------
# MemoryHookService
# ---------------------------------------------------------------------------

class MemoryHookService:
    """Integrate memory with rf-mcp execution lifecycle.

    Called by the execution layer (never the reverse). All methods
    catch exceptions internally — memory failures never propagate.
    """

    def __init__(
        self,
        query_service: MemoryQueryService,
        embedding_service: EmbeddingService,
        repository: MemoryRepository,
        store: MemoryStore,
    ) -> None:
        self._query_service = query_service
        self._embedding_service = embedding_service
        self._repository = repository
        self._store = store
        self._session_urls: Dict[str, str] = {}

    # -- Locator keywords ----------------------------------------------------

    _LOCATOR_KEYWORDS = frozenset({
        "Click", "Click Button", "Click Element", "Click Link",
        "Fill Text", "Fill Secret", "Type Text", "Type Secret",
        "Get Text", "Get Property", "Get Attribute",
        "Hover", "Focus", "Scroll To Element",
        "Select Options By", "Check Checkbox", "Uncheck Checkbox",
        "Wait For Elements State", "Get Element Count",
        "Input Text", "Input Password",
        "Click Image", "Click Element At Coordinates",
    })

    def _is_locator_keyword(self, keyword: str) -> bool:
        """Check if keyword takes a locator as first argument."""
        return keyword in self._LOCATOR_KEYWORDS

    def _track_navigation(
        self, keyword: str, arguments: List[str], session_id: Optional[str],
    ) -> None:
        """Track page URL from navigation keywords."""
        if not session_id:
            return
        nav_keywords = {"New Page", "Go To", "Open Browser"}
        if keyword in nav_keywords and arguments:
            self._session_urls[session_id] = arguments[0]

    async def on_locator_used(
        self,
        keyword: str,
        locator: str,
        success: bool,
        library: str,
        session_id: Optional[str] = None,
        error_text: str = "",
    ) -> None:
        """Store locator usage as LOCATORS memory."""
        try:
            desc = LocatorDescription.from_locator(locator)
            strategy = LocatorStrategy.detect(locator)
            page_url = self._session_urls.get(session_id or "", "")
            outcome = "success" if success else "failure"

            content = f"{desc.value} ({keyword}, {library})"
            metadata: Dict[str, Any] = {
                "locator": locator,
                "strategy": strategy.value,
                "keyword": keyword,
                "library": library,
                "outcome": outcome,
                "page_url": page_url,
                "description": desc.value,
            }
            if error_text:
                metadata["error_text"] = error_text[:500]

            tags = ("locator", outcome)
            entry = MemoryEntry(
                content=content,
                memory_type=MemoryType.locators(),
                metadata=metadata,
                tags=tags,
            )
            await self._store_entry(entry, session_id)
        except Exception as e:
            logger.debug("Memory store (locator used) failed: %s", e)

    # -- WRITE hooks ---------------------------------------------------------

    async def on_step_success(
        self,
        keyword: str,
        arguments: List[str],
        result: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> None:
        """Store successful step as WORKING_STEPS memory."""
        try:
            content = f"Keyword: {keyword}\nArguments: {', '.join(arguments)}"
            if result.get("output"):
                content += f"\nResult: {str(result['output'])[:500]}"

            entry = MemoryEntry(
                content=content,
                memory_type=MemoryType.working_steps(),
                metadata={
                    "keyword": keyword,
                    "arguments": arguments,
                    "library": result.get("library", ""),
                },
            )
            await self._store_entry(entry, session_id)
        except Exception as e:
            logger.debug("Memory store (step success) failed: %s", e)

    async def on_step_failure(
        self,
        keyword: str,
        arguments: List[str],
        error_text: str,
        session_id: Optional[str] = None,
    ) -> None:
        """Store failure as COMMON_ERRORS memory."""
        try:
            content = (
                f"Error: {error_text}\n"
                f"Keyword: {keyword}\n"
                f"Arguments: {', '.join(arguments)}"
            )
            entry = MemoryEntry(
                content=content,
                memory_type=MemoryType.common_errors(),
                metadata={
                    "keyword": keyword,
                    "arguments": arguments,
                    "error_text": error_text[:500],
                },
            )
            await self._store_entry(entry, session_id)
        except Exception as e:
            logger.debug("Memory store (step failure) failed: %s", e)

    async def on_error_recovered(
        self,
        error_text: str,
        fix_steps: List[Dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> None:
        """Store error+fix pair as COMMON_ERRORS."""
        try:
            fix_desc = "\n".join(
                f"  {s.get('keyword', '?')} {' '.join(s.get('arguments', []))}"
                for s in fix_steps
            )
            content = (
                f"Error: {error_text}\n"
                f"Fix:\n{fix_desc}"
            )
            entry = MemoryEntry(
                content=content,
                memory_type=MemoryType.common_errors(),
                metadata={
                    "error_text": error_text[:500],
                    "fix_steps": fix_steps,
                    "confidence": 0.6,
                },
                tags=("error_fix",),
            )
            await self._store_entry(entry, session_id)
        except Exception as e:
            logger.debug("Memory store (error recovered) failed: %s", e)

    async def on_keyword_discovered(
        self,
        keyword_name: str,
        library: str,
        documentation: str,
    ) -> None:
        """Store keyword info as KEYWORDS memory."""
        try:
            content = (
                f"Keyword: {keyword_name}\n"
                f"Library: {library}\n"
                f"Documentation: {documentation[:1000]}"
            )
            entry = MemoryEntry(
                content=content,
                memory_type=MemoryType.keywords(),
                metadata={
                    "keyword_name": keyword_name,
                    "library": library,
                },
            )
            await self._store_entry(entry, None)
        except Exception as e:
            logger.debug("Memory store (keyword discovered) failed: %s", e)

    async def on_session_end(
        self,
        session_id: str,
        steps: List[Dict[str, Any]],
    ) -> None:
        """Batch store session's successful step sequences."""
        try:
            successful = [s for s in steps if s.get("success", False)]
            if not successful:
                return

            # Store as a single working sequence
            step_lines = []
            for s in successful:
                kw = s.get("keyword", "?")
                args = s.get("arguments", [])
                step_lines.append(f"  {kw} {' '.join(str(a) for a in args)}")

            content = f"Session steps ({len(successful)} successful):\n" + "\n".join(
                step_lines
            )
            entry = MemoryEntry(
                content=content[:MemoryEntry.MAX_CONTENT_LENGTH],
                memory_type=MemoryType.working_steps(),
                metadata={
                    "step_count": len(successful),
                    "session_id": session_id,
                },
                tags=("session_sequence",),
            )
            await self._store_entry(entry, session_id)
        except Exception as e:
            logger.debug("Memory store (session end) failed: %s", e)

    async def store_knowledge(
        self,
        content: str,
        knowledge_type: str = "domain_knowledge",
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Manual storage of DOCUMENTATION or DOMAIN_KNOWLEDGE."""
        try:
            mt = MemoryType.from_string(knowledge_type)
            if mt.value not in (
                MemoryType.DOCUMENTATION,
                MemoryType.DOMAIN_KNOWLEDGE,
            ):
                raise ValueError(
                    f"knowledge_type must be 'documentation' or "
                    f"'domain_knowledge', got '{knowledge_type}'"
                )
            entry = MemoryEntry(
                content=content,
                memory_type=mt,
                tags=tuple(tags or []),
            )
            record = await self._store_entry(entry, None)
            return record.record_id if record else None
        except Exception as e:
            logger.debug("Memory store (knowledge) failed: %s", e)
            return None

    # -- READ hooks ----------------------------------------------------------

    async def recall_for_hint(
        self,
        keyword: str,
        error_text: str,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """Before execution or on error, recall relevant hints."""
        try:
            results = await self._query_service.recall_for_error(
                error_text, session_id
            )
            if not results:
                return None

            hints = []
            for r in results[:3]:
                action = r.confidence.action if r.confidence else "suggest"
                hints.append(
                    f"[{action}] (similarity={r.adjusted_similarity.value:.2f}): "
                    f"{r.content[:200]}"
                )
            return "\n".join(hints)
        except Exception as e:
            logger.debug("Memory recall (hint) failed: %s", e)
            return None

    async def recall_for_scenario(
        self, scenario_text: str
    ) -> Optional[str]:
        """Recall step patterns for a scenario."""
        try:
            results = await self._query_service.recall_steps(scenario_text)
            if not results:
                return None

            parts = []
            for r in results[:3]:
                parts.append(
                    f"Previous pattern (similarity={r.adjusted_similarity.value:.2f}):\n"
                    f"{r.content[:300]}"
                )
            return "\n---\n".join(parts)
        except Exception as e:
            logger.debug("Memory recall (scenario) failed: %s", e)
            return None

    # -- Central dispatch ----------------------------------------------------

    async def on_tool_call(
        self,
        session_id: Optional[str],
        tool_name: str,
        arguments: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        """Dispatch to appropriate write hook based on tool_name."""
        try:
            success = result.get("success", True)
            if tool_name == "execute_step":
                kw = arguments.get("keyword", "")
                args = arguments.get("arguments", [])
                # Track navigation for URL context
                self._track_navigation(kw, args, session_id)
                if success:
                    await self.on_step_success(kw, args, result, session_id)
                else:
                    error = result.get("error", "Unknown error")
                    await self.on_step_failure(kw, args, error, session_id)
                # Dual-write to LOCATORS for locator keywords
                if self._is_locator_keyword(kw) and args:
                    library = result.get("library", "")
                    err = result.get("error", "") if not success else ""
                    await self.on_locator_used(
                        keyword=kw,
                        locator=args[0],
                        success=success,
                        library=library,
                        session_id=session_id,
                        error_text=err,
                    )

            elif tool_name == "find_keywords":
                matches = result.get("matches") or result.get("results", [])
                for m in matches[:5]:
                    name = m.get("name", m.get("keyword", ""))
                    lib = m.get("library", "")
                    doc = m.get("documentation", m.get("doc", ""))
                    if name:
                        await self.on_keyword_discovered(name, lib, doc)

            elif tool_name == "execute_batch":
                if not success:
                    error = result.get("error", "Batch failed")
                    fix_steps = result.get("fix_steps", [])
                    if fix_steps:
                        await self.on_error_recovered(
                            error, fix_steps, session_id
                        )
        except Exception as e:
            logger.debug("Memory on_tool_call dispatch failed: %s", e)

    # -- Internal helpers ----------------------------------------------------

    async def _store_entry(
        self,
        entry: MemoryEntry,
        session_id: Optional[str],
    ) -> Optional[MemoryRecord]:
        """Embed and store an entry. Returns record or None on failure."""
        if not self._embedding_service.is_available:
            return None

        try:
            embedding = await self._embedding_service.embed(entry.content)
            entry_with_emb = entry.with_embedding(embedding)
            record = self._store.prepare_store(entry_with_emb, session_id)
            coll_name = entry.memory_type.collection_name
            await self._repository.ensure_collection(
                coll_name, embedding.dimensions
            )
            await self._repository.store(coll_name, record)
            return record
        except Exception as e:
            logger.debug("Memory _store_entry failed: %s", e)
            return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_memory_services(
    config: Optional[StorageConfig] = None,
) -> Optional[Dict[str, Any]]:
    """Create and wire all memory domain services.

    Returns None if memory is not enabled or no embedding backend is available.
    Returns dict with keys: store, embedding_service, query_service,
                            hook_service, repository, backend
    """
    if config is None:
        config = StorageConfig.from_env()

    if not config.enabled:
        return None

    backend = EmbeddingBackend.detect(config.embedding_model)
    if not backend.is_available:
        logger.warning(
            "Memory enabled but no embedding backend available. "
            "Install: uv pip install rf-mcp[memory]"
        )
        return None

    # Use the detected backend's dimension
    if backend.dimension and backend.dimension != config.dimension:
        import os

        config = StorageConfig(
            db_path=config.db_path,
            embedding_model=backend.model_name or config.embedding_model,
            dimension=backend.dimension,
            max_records_per_collection=config.max_records_per_collection,
            prune_age_days=config.prune_age_days,
            time_decay_half_life=config.time_decay_half_life,
            enabled=config.enabled,
            project_id=config.project_id,
        )

    store = MemoryStore.create(config)
    embedding_service = EmbeddingService(backend)

    # Try sqlite-vec, fall back to in-memory
    repository: MemoryRepository
    try:
        from .adapters.sqlite_vec_adapter import SqliteVecRepository

        repository = SqliteVecRepository(config.db_path)
        logger.info("Using sqlite-vec memory repository at %s", config.db_path)
    except ImportError:
        from .repository import InMemoryMemoryRepository

        repository = InMemoryMemoryRepository()
        logger.info("Using in-memory repository (sqlite-vec not available)")

    query_service = MemoryQueryService(
        embedding_service=embedding_service,
        repository=repository,
        store=store,
    )

    hook_service = MemoryHookService(
        query_service=query_service,
        embedding_service=embedding_service,
        repository=repository,
        store=store,
    )

    return {
        "store": store,
        "embedding_service": embedding_service,
        "query_service": query_service,
        "hook_service": hook_service,
        "repository": repository,
        "backend": backend,
        "config": config,
    }

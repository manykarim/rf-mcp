"""Tests for ADR-014.1 Structured Locator Memory.

Covers: LocatorStrategy, LocatorOutcome, LocatorDescription,
LocatorRecallResult, MemoryType.LOCATORS, MemoryQuery.for_locator_recall,
LocatorStored event, MemoryHookService locator methods,
MemoryQueryService.recall_locators.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotmcp.domains.memory.events import LocatorStored
from robotmcp.domains.memory.value_objects import (
    LocatorDescription,
    LocatorOutcome,
    LocatorRecallResult,
    LocatorStrategy,
    MemoryEntry,
    MemoryQuery,
    MemoryType,
    RecallResult,
    SimilarityScore,
)

_async_mark = pytest.mark.asyncio(loop_scope="module")


# =========================================================================
# LocatorStrategy
# =========================================================================


class TestLocatorStrategy:
    def test_css_prefix(self):
        assert LocatorStrategy.detect("css=.btn") == LocatorStrategy.CSS

    def test_css_colon_prefix(self):
        assert LocatorStrategy.detect("css:.btn") == LocatorStrategy.CSS

    def test_xpath_prefix(self):
        assert LocatorStrategy.detect("xpath=//div") == LocatorStrategy.XPATH

    def test_xpath_colon_prefix(self):
        assert LocatorStrategy.detect("xpath://div") == LocatorStrategy.XPATH

    def test_xpath_double_slash(self):
        assert LocatorStrategy.detect("//button[@id='ok']") == LocatorStrategy.XPATH

    def test_xpath_parenthesized(self):
        assert LocatorStrategy.detect("(//button)[1]") == LocatorStrategy.XPATH

    def test_id_prefix(self):
        assert LocatorStrategy.detect("id=submit-btn") == LocatorStrategy.ID

    def test_id_colon_prefix(self):
        assert LocatorStrategy.detect("id:submit-btn") == LocatorStrategy.ID

    def test_text_prefix(self):
        assert LocatorStrategy.detect("text=Submit") == LocatorStrategy.TEXT

    def test_text_colon_prefix(self):
        assert LocatorStrategy.detect("text:Submit") == LocatorStrategy.TEXT

    def test_name_prefix(self):
        assert LocatorStrategy.detect("name=email") == LocatorStrategy.NAME

    def test_name_colon_prefix(self):
        assert LocatorStrategy.detect("name:email") == LocatorStrategy.NAME

    def test_link_prefix(self):
        assert LocatorStrategy.detect("link=Home") == LocatorStrategy.LINK

    def test_link_colon_prefix(self):
        assert LocatorStrategy.detect("link:Home") == LocatorStrategy.LINK

    def test_auto_fallback(self):
        assert LocatorStrategy.detect("Submit") == LocatorStrategy.AUTO

    def test_auto_hash_no_css_prefix(self):
        assert LocatorStrategy.detect("#main") == LocatorStrategy.AUTO

    def test_strips_whitespace(self):
        assert LocatorStrategy.detect("  css=.btn  ") == LocatorStrategy.CSS

    def test_enum_values(self):
        assert LocatorStrategy.CSS.value == "css"
        assert LocatorStrategy.XPATH.value == "xpath"
        assert LocatorStrategy.ID.value == "id"
        assert LocatorStrategy.TEXT.value == "text"
        assert LocatorStrategy.NAME.value == "name"
        assert LocatorStrategy.LINK.value == "link"
        assert LocatorStrategy.AUTO.value == "auto"

    def test_is_string_enum(self):
        assert isinstance(LocatorStrategy.CSS, str)
        assert LocatorStrategy.CSS == "css"


# =========================================================================
# LocatorOutcome
# =========================================================================


class TestLocatorOutcome:
    def test_success(self):
        lo = LocatorOutcome(
            success=True,
            keyword="Click",
            library="Browser",
            locator="id=ok",
        )
        assert lo.success is True
        assert lo.keyword == "Click"
        assert lo.library == "Browser"
        assert lo.locator == "id=ok"

    def test_failure_with_error(self):
        lo = LocatorOutcome(
            success=False,
            keyword="Click",
            library="Browser",
            locator="id=missing",
            error_text="Element not found",
        )
        assert lo.success is False
        assert lo.error_text == "Element not found"

    def test_with_page_url(self):
        lo = LocatorOutcome(
            success=True,
            keyword="Click",
            library="Browser",
            locator="id=ok",
            page_url="https://example.com",
        )
        assert lo.page_url == "https://example.com"

    def test_frozen(self):
        lo = LocatorOutcome(
            success=True, keyword="Click", library="Browser", locator="id=ok"
        )
        with pytest.raises(AttributeError):
            lo.success = False  # type: ignore[misc]

    def test_defaults(self):
        lo = LocatorOutcome(
            success=True, keyword="Click", library="Browser", locator="id=ok"
        )
        assert lo.page_url == ""
        assert lo.error_text == ""


# =========================================================================
# LocatorDescription + _extract_description
# =========================================================================


class TestLocatorDescription:
    def test_has_text_pattern(self):
        desc = LocatorDescription.from_locator("button >> has-text('Submit')")
        assert desc.value == "Submit"
        assert desc.strategy == LocatorStrategy.AUTO

    def test_text_prefix(self):
        desc = LocatorDescription.from_locator("text=Add to cart")
        assert desc.value == "Add to cart"
        assert desc.strategy == LocatorStrategy.TEXT

    def test_id_prefix(self):
        desc = LocatorDescription.from_locator("id=submit-btn")
        assert desc.value == "submit btn (id)"
        assert desc.strategy == LocatorStrategy.ID

    def test_name_prefix(self):
        desc = LocatorDescription.from_locator("name=email_field")
        assert desc.value == "email field (name)"
        assert desc.strategy == LocatorStrategy.NAME

    def test_link_prefix(self):
        desc = LocatorDescription.from_locator("link=Home Page")
        assert desc.value == "Home Page"
        assert desc.strategy == LocatorStrategy.LINK

    def test_css_id_selector(self):
        desc = LocatorDescription.from_locator("css=#main-content")
        assert desc.value == "main content element"
        assert desc.strategy == LocatorStrategy.CSS

    def test_css_class_selector(self):
        desc = LocatorDescription.from_locator("css=.btn-primary")
        assert desc.value == "btn primary element"
        assert desc.strategy == LocatorStrategy.CSS

    def test_css_general(self):
        desc = LocatorDescription.from_locator("css=div.container > span")
        assert "div" in desc.value
        assert desc.strategy == LocatorStrategy.CSS

    def test_xpath_attribute(self):
        desc = LocatorDescription.from_locator("//button[@id='submit']")
        assert desc.value == "submit button (id)"
        assert desc.strategy == LocatorStrategy.XPATH

    def test_xpath_simple_tag(self):
        desc = LocatorDescription.from_locator("//div")
        assert desc.value == "div element"
        assert desc.strategy == LocatorStrategy.XPATH

    def test_auto_fallback(self):
        desc = LocatorDescription.from_locator("Submit Button")
        assert desc.value == "Submit Button"
        assert desc.strategy == LocatorStrategy.AUTO

    def test_preserves_raw_locator(self):
        loc = "css=.badge"
        desc = LocatorDescription.from_locator(loc)
        assert desc.raw_locator == loc

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            LocatorDescription(value="", raw_locator="x", strategy=LocatorStrategy.AUTO)

    def test_frozen(self):
        desc = LocatorDescription.from_locator("id=foo")
        with pytest.raises(AttributeError):
            desc.value = "bar"  # type: ignore[misc]

    def test_xpath_with_prefix(self):
        desc = LocatorDescription.from_locator("xpath=//input[@name='email']")
        assert desc.value == "email input (name)"
        assert desc.strategy == LocatorStrategy.XPATH


# =========================================================================
# LocatorRecallResult
# =========================================================================


class TestLocatorRecallResult:
    def test_to_dict_success(self):
        lr = LocatorRecallResult(
            locator="id=submit-btn",
            strategy=LocatorStrategy.ID,
            keyword="Click",
            library="Browser",
            outcome="success",
            page_url="https://example.com",
            description="submit btn (id)",
            similarity=0.85,
        )
        d = lr.to_dict()
        assert d["locator"] == "id=submit-btn"
        assert d["strategy"] == "id"
        assert d["keyword"] == "Click"
        assert d["library"] == "Browser"
        assert d["outcome"] == "success"
        assert d["page_url"] == "https://example.com"
        assert d["description"] == "submit btn (id)"
        assert d["similarity"] == 0.85
        assert "error_text" not in d

    def test_to_dict_failure_includes_error(self):
        lr = LocatorRecallResult(
            locator="css=.missing",
            strategy=LocatorStrategy.CSS,
            keyword="Click",
            library="Browser",
            outcome="failure",
            error_text="Element not found",
            similarity=0.6,
        )
        d = lr.to_dict()
        assert d["outcome"] == "failure"
        assert d["error_text"] == "Element not found"

    def test_similarity_rounded(self):
        lr = LocatorRecallResult(
            locator="id=x",
            strategy=LocatorStrategy.ID,
            keyword="Click",
            library="Browser",
            outcome="success",
            similarity=0.123456789,
        )
        assert lr.to_dict()["similarity"] == 0.1235

    def test_frozen(self):
        lr = LocatorRecallResult(
            locator="id=x",
            strategy=LocatorStrategy.ID,
            keyword="Click",
            library="Browser",
            outcome="success",
        )
        with pytest.raises(AttributeError):
            lr.locator = "id=y"  # type: ignore[misc]

    def test_defaults(self):
        lr = LocatorRecallResult(
            locator="id=x",
            strategy=LocatorStrategy.ID,
            keyword="Click",
            library="Browser",
            outcome="success",
        )
        assert lr.page_url == ""
        assert lr.description == ""
        assert lr.similarity == 0.0
        assert lr.error_text == ""


# =========================================================================
# MemoryType.LOCATORS
# =========================================================================


class TestMemoryTypeLocators:
    def test_locators_factory(self):
        mt = MemoryType.locators()
        assert mt.value == "locators"

    def test_collection_name(self):
        mt = MemoryType.locators()
        assert mt.collection_name == "rfmcp_locators"

    def test_from_string(self):
        mt = MemoryType.from_string("locators")
        assert mt.value == "locators"

    def test_in_all_types(self):
        all_values = [mt.value for mt in MemoryType.all_types()]
        assert "locators" in all_values


# =========================================================================
# MemoryQuery.for_locator_recall
# =========================================================================


class TestMemoryQueryForLocatorRecall:
    def test_factory(self):
        mq = MemoryQuery.for_locator_recall("login button")
        assert mq.query_text == "login button"
        assert mq.memory_type is not None
        assert mq.memory_type.value == "locators"
        assert mq.top_k == 5
        assert mq.min_similarity == 0.2

    def test_truncates_long_text(self):
        long_text = "x" * 3000
        mq = MemoryQuery.for_locator_recall(long_text)
        assert len(mq.query_text) == MemoryQuery.MAX_QUERY_LENGTH

    def test_collection_names(self):
        mq = MemoryQuery.for_locator_recall("button")
        assert mq.collection_names == ["rfmcp_locators"]


# =========================================================================
# LocatorStored event
# =========================================================================


class TestLocatorStoredEvent:
    def test_basic(self):
        ev = LocatorStored(
            record_id="rec-1",
            locator="id=submit",
            keyword="Click",
            library="Browser",
            outcome="success",
        )
        assert ev.record_id == "rec-1"
        assert ev.locator == "id=submit"
        assert ev.outcome == "success"

    def test_to_dict(self):
        ev = LocatorStored(
            record_id="rec-1",
            locator="id=submit",
            keyword="Click",
            library="Browser",
            outcome="success",
            page_url="https://example.com",
            session_id="sess-1",
        )
        d = ev.to_dict()
        assert d["event_type"] == "LocatorStored"
        assert d["locator"] == "id=submit"
        assert d["keyword"] == "Click"
        assert d["page_url"] == "https://example.com"
        assert d["session_id"] == "sess-1"
        assert "timestamp" in d

    def test_defaults(self):
        ev = LocatorStored(
            record_id="rec-1",
            locator="id=x",
            keyword="Click",
            library="Browser",
            outcome="success",
        )
        assert ev.page_url == ""
        assert ev.session_id is None

    def test_frozen(self):
        ev = LocatorStored(
            record_id="rec-1",
            locator="id=x",
            keyword="Click",
            library="Browser",
            outcome="success",
        )
        with pytest.raises(AttributeError):
            ev.outcome = "failure"  # type: ignore[misc]


# =========================================================================
# MemoryHookService — locator methods
# =========================================================================


def _build_hook_service():
    """Build a MemoryHookService with mocked dependencies."""
    from robotmcp.domains.memory.services import MemoryHookService

    query_service = MagicMock()
    embedding_service = MagicMock()
    embedding_service.is_available = True
    embedding_service.embed = AsyncMock(return_value=MagicMock(dimensions=256))
    repository = MagicMock()
    repository.ensure_collection = AsyncMock()
    repository.store = AsyncMock()
    store = MagicMock()
    store.prepare_store = MagicMock(
        side_effect=lambda entry, sid: MagicMock(record_id="rec-1")
    )

    svc = MemoryHookService(
        query_service=query_service,
        embedding_service=embedding_service,
        repository=repository,
        store=store,
    )
    return svc, query_service, embedding_service, repository, store


@_async_mark
class TestIsLocatorKeyword:
    async def test_click_is_locator(self):
        svc, *_ = _build_hook_service()
        assert svc._is_locator_keyword("Click") is True

    async def test_fill_text_is_locator(self):
        svc, *_ = _build_hook_service()
        assert svc._is_locator_keyword("Fill Text") is True

    async def test_go_to_is_not_locator(self):
        svc, *_ = _build_hook_service()
        assert svc._is_locator_keyword("Go To") is False

    async def test_new_page_is_not_locator(self):
        svc, *_ = _build_hook_service()
        assert svc._is_locator_keyword("New Page") is False


@_async_mark
class TestTrackNavigation:
    async def test_tracks_go_to(self):
        svc, *_ = _build_hook_service()
        svc._track_navigation("Go To", ["https://example.com"], "sess-1")
        assert svc._session_urls["sess-1"] == "https://example.com"

    async def test_tracks_new_page(self):
        svc, *_ = _build_hook_service()
        svc._track_navigation("New Page", ["https://example.com"], "sess-1")
        assert svc._session_urls["sess-1"] == "https://example.com"

    async def test_ignores_without_session(self):
        svc, *_ = _build_hook_service()
        svc._track_navigation("Go To", ["https://example.com"], None)
        assert len(svc._session_urls) == 0

    async def test_ignores_non_nav_keywords(self):
        svc, *_ = _build_hook_service()
        svc._track_navigation("Click", ["id=ok"], "sess-1")
        assert len(svc._session_urls) == 0


@_async_mark
class TestOnLocatorUsed:
    async def test_stores_success(self):
        svc, _, emb_svc, repo, store = _build_hook_service()
        await svc.on_locator_used(
            keyword="Click",
            locator="id=submit-btn",
            success=True,
            library="Browser",
            session_id="sess-1",
        )
        # Verify store was called
        assert store.prepare_store.called
        call_args = store.prepare_store.call_args[0]
        entry = call_args[0]
        assert entry.memory_type.value == "locators"
        assert entry.metadata["locator"] == "id=submit-btn"
        assert entry.metadata["strategy"] == "id"
        assert entry.metadata["outcome"] == "success"
        assert entry.metadata["keyword"] == "Click"
        assert entry.metadata["library"] == "Browser"
        assert "locator" in entry.tags
        assert "success" in entry.tags

    async def test_stores_failure_with_error(self):
        svc, _, emb_svc, repo, store = _build_hook_service()
        await svc.on_locator_used(
            keyword="Click",
            locator="css=.missing",
            success=False,
            library="Browser",
            error_text="Element not found",
        )
        call_args = store.prepare_store.call_args[0]
        entry = call_args[0]
        assert entry.metadata["outcome"] == "failure"
        assert entry.metadata["error_text"] == "Element not found"
        assert "failure" in entry.tags

    async def test_includes_page_url_from_tracking(self):
        svc, *_ = _build_hook_service()
        svc._session_urls["sess-1"] = "https://example.com"
        await svc.on_locator_used(
            keyword="Click",
            locator="id=ok",
            success=True,
            library="Browser",
            session_id="sess-1",
        )
        call_args = svc._store.prepare_store.call_args[0]
        entry = call_args[0]
        assert entry.metadata["page_url"] == "https://example.com"

    async def test_description_in_content(self):
        svc, *_ = _build_hook_service()
        await svc.on_locator_used(
            keyword="Click",
            locator="id=submit-btn",
            success=True,
            library="Browser",
        )
        call_args = svc._store.prepare_store.call_args[0]
        entry = call_args[0]
        assert "submit btn (id)" in entry.content
        assert "Click" in entry.content
        assert "Browser" in entry.content

    async def test_swallows_exceptions(self):
        svc, _, emb_svc, *_ = _build_hook_service()
        emb_svc.embed = AsyncMock(side_effect=Exception("embed fail"))
        # Should not raise
        await svc.on_locator_used(
            keyword="Click",
            locator="id=x",
            success=True,
            library="Browser",
        )

    async def test_xpath_strategy_detected(self):
        svc, *_ = _build_hook_service()
        await svc.on_locator_used(
            keyword="Click",
            locator="//button[@id='ok']",
            success=True,
            library="Browser",
        )
        call_args = svc._store.prepare_store.call_args[0]
        entry = call_args[0]
        assert entry.metadata["strategy"] == "xpath"

    async def test_css_strategy_detected(self):
        svc, *_ = _build_hook_service()
        await svc.on_locator_used(
            keyword="Click",
            locator="css=.badge",
            success=True,
            library="Browser",
        )
        call_args = svc._store.prepare_store.call_args[0]
        entry = call_args[0]
        assert entry.metadata["strategy"] == "css"


@_async_mark
class TestOnToolCallLocatorDispatch:
    async def test_dispatches_locator_keyword(self):
        svc, *_ = _build_hook_service()
        svc.on_locator_used = AsyncMock()
        svc.on_step_success = AsyncMock()
        await svc.on_tool_call(
            session_id="sess-1",
            tool_name="execute_step",
            arguments={"keyword": "Click", "arguments": ["id=submit"]},
            result={"success": True, "library": "Browser"},
        )
        svc.on_locator_used.assert_called_once_with(
            keyword="Click",
            locator="id=submit",
            success=True,
            library="Browser",
            session_id="sess-1",
            error_text="",
        )
        svc.on_step_success.assert_called_once()

    async def test_no_dispatch_for_non_locator(self):
        svc, *_ = _build_hook_service()
        svc.on_locator_used = AsyncMock()
        svc.on_step_success = AsyncMock()
        await svc.on_tool_call(
            session_id="sess-1",
            tool_name="execute_step",
            arguments={"keyword": "Log", "arguments": ["hello"]},
            result={"success": True},
        )
        svc.on_locator_used.assert_not_called()
        svc.on_step_success.assert_called_once()

    async def test_dispatch_failure_passes_error(self):
        svc, *_ = _build_hook_service()
        svc.on_locator_used = AsyncMock()
        svc.on_step_failure = AsyncMock()
        await svc.on_tool_call(
            session_id="sess-1",
            tool_name="execute_step",
            arguments={"keyword": "Click", "arguments": ["css=.missing"]},
            result={"success": False, "error": "Not found", "library": "Browser"},
        )
        svc.on_locator_used.assert_called_once()
        call_kwargs = svc.on_locator_used.call_args[1]
        assert call_kwargs["success"] is False
        assert call_kwargs["error_text"] == "Not found"

    async def test_no_dispatch_without_args(self):
        svc, *_ = _build_hook_service()
        svc.on_locator_used = AsyncMock()
        svc.on_step_success = AsyncMock()
        await svc.on_tool_call(
            session_id="sess-1",
            tool_name="execute_step",
            arguments={"keyword": "Click", "arguments": []},
            result={"success": True},
        )
        svc.on_locator_used.assert_not_called()

    async def test_tracks_navigation_before_dispatch(self):
        svc, *_ = _build_hook_service()
        svc.on_step_success = AsyncMock()
        await svc.on_tool_call(
            session_id="sess-1",
            tool_name="execute_step",
            arguments={"keyword": "Go To", "arguments": ["https://example.com"]},
            result={"success": True},
        )
        assert svc._session_urls.get("sess-1") == "https://example.com"

    async def test_find_keywords_still_dispatches(self):
        svc, *_ = _build_hook_service()
        svc.on_keyword_discovered = AsyncMock()
        await svc.on_tool_call(
            session_id="sess-1",
            tool_name="find_keywords",
            arguments={},
            result={
                "success": True,
                "results": [
                    {"name": "Click", "library": "Browser", "doc": "Click elem"},
                ],
            },
        )
        svc.on_keyword_discovered.assert_called_once()


# =========================================================================
# MemoryQueryService.recall_locators
# =========================================================================


def _build_query_service():
    """Build a MemoryQueryService with mocked dependencies."""
    from robotmcp.domains.memory.services import MemoryQueryService

    embedding_service = MagicMock()
    embedding_service.is_available = True
    repository = MagicMock()
    store = MagicMock()
    store.prepare_recall = MagicMock(side_effect=lambda q: q)
    store.time_decay = MagicMock()

    svc = MemoryQueryService(
        embedding_service=embedding_service,
        repository=repository,
        store=store,
    )
    return svc, embedding_service, repository, store


@_async_mark
class TestRecallLocators:
    async def test_returns_structured_results(self):
        svc, emb_svc, repo, store = _build_query_service()
        # Mock the recall pipeline
        emb_svc.embed = AsyncMock(return_value=MagicMock(dimensions=256))
        raw_record = MagicMock()
        raw_record.record_id = "rec-1"
        raw_record.entry.content = "submit btn (id) (Click, Browser)"
        raw_record.entry.memory_type = MemoryType.locators()
        raw_record.entry.metadata = {
            "locator": "id=submit-btn",
            "strategy": "id",
            "keyword": "Click",
            "library": "Browser",
            "outcome": "success",
            "page_url": "https://example.com",
            "description": "submit btn (id)",
        }
        raw_record.age_days = 1.0
        sim = SimilarityScore.cosine(0.85)
        repo.search = AsyncMock(return_value=[(raw_record, sim)])
        store.time_decay.compute = MagicMock(return_value=sim)

        results = await svc.recall_locators("submit button")
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, LocatorRecallResult)
        assert r.locator == "id=submit-btn"
        assert r.strategy == LocatorStrategy.ID
        assert r.keyword == "Click"
        assert r.library == "Browser"
        assert r.outcome == "success"
        assert r.page_url == "https://example.com"
        assert r.description == "submit btn (id)"
        assert r.similarity == 0.85

    async def test_handles_invalid_strategy(self):
        svc, emb_svc, repo, store = _build_query_service()
        emb_svc.embed = AsyncMock(return_value=MagicMock(dimensions=256))
        raw_record = MagicMock()
        raw_record.record_id = "rec-1"
        raw_record.entry.content = "test"
        raw_record.entry.memory_type = MemoryType.locators()
        raw_record.entry.metadata = {
            "locator": "whatever",
            "strategy": "invalid_strategy",
            "keyword": "Click",
            "library": "Browser",
            "outcome": "success",
        }
        raw_record.age_days = 1.0
        sim = SimilarityScore.cosine(0.7)
        repo.search = AsyncMock(return_value=[(raw_record, sim)])
        store.time_decay.compute = MagicMock(return_value=sim)

        results = await svc.recall_locators("test")
        assert len(results) == 1
        assert results[0].strategy == LocatorStrategy.AUTO

    async def test_empty_results(self):
        svc, emb_svc, repo, store = _build_query_service()
        emb_svc.embed = AsyncMock(return_value=MagicMock(dimensions=256))
        repo.search = AsyncMock(return_value=[])

        results = await svc.recall_locators("nonexistent")
        assert results == []

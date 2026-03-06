"""Tests for Artifact Output Domain Services (ADR-015)."""
from __future__ import annotations

__test__ = True

from datetime import datetime, timedelta

import pytest

from robotmcp.domains.artifact_output.aggregates import ArtifactStore
from robotmcp.domains.artifact_output.services import (
    ArtifactExternalizationService,
    ArtifactRetrievalService,
)
from unittest.mock import patch

from robotmcp.domains.artifact_output.value_objects import (
    ArtifactPolicy,
    ExternalizationResult,
    ExternalizationRule,
    FETCH_ARTIFACT_SUMMARY_TEMPLATE,
    FILE_PATH_SUMMARY_TEMPLATE,
    OutputMode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(**policy_kwargs) -> ArtifactStore:
    return ArtifactStore.create(
        ArtifactPolicy(**policy_kwargs) if policy_kwargs else None
    )


def _make_ext_service(
    store: ArtifactStore | None = None,
    mode: OutputMode | None = None,
    rules: list | None = None,
) -> ArtifactExternalizationService:
    return ArtifactExternalizationService(
        store=store or _make_store(),
        output_mode=mode,
        rules=rules,
    )


# ===========================================================================
# ArtifactExternalizationService
# ===========================================================================


# ---------------------------------------------------------------------------
# INLINE mode returns response unchanged
# ---------------------------------------------------------------------------


class TestInlineModeUnchanged:
    """INLINE mode returns response unchanged."""

    def test_response_dict_identical(self):
        svc = _make_ext_service(
            mode=OutputMode.INLINE,
            rules=[ExternalizationRule(tool_name="my_tool", field_path="output")],
        )
        response = {"output": "x" * 10_000}
        result, exts = svc.externalize("my_tool", response, "s1")
        assert result["output"] == "x" * 10_000
        assert exts == []

    def test_no_artifacts_created(self):
        store = _make_store()
        svc = _make_ext_service(
            store=store,
            mode=OutputMode.INLINE,
            rules=[ExternalizationRule(tool_name="t", field_path="output")],
        )
        svc.externalize("t", {"output": "x" * 10_000}, "s1")
        assert store.list_artifacts() == []


# ---------------------------------------------------------------------------
# AUTO mode skips small fields
# ---------------------------------------------------------------------------


class TestAutoModeSkipsSmall:
    """AUTO mode skips small fields."""

    def test_small_content_stays_inline(self):
        store = _make_store(max_inline_tokens=100)
        svc = _make_ext_service(
            store=store,
            mode=OutputMode.AUTO,
            rules=[ExternalizationRule(tool_name="t", field_path="output")],
        )
        # 5 chars / 4 = 1.25 tokens, well under 100
        response = {"output": "small"}
        result, exts = svc.externalize("t", response, "s1")
        assert result["output"] == "small"
        assert exts == []

    def test_no_artifact_stored_for_small(self):
        store = _make_store(max_inline_tokens=100)
        svc = _make_ext_service(
            store=store,
            mode=OutputMode.AUTO,
            rules=[ExternalizationRule(tool_name="t", field_path="output")],
        )
        svc.externalize("t", {"output": "tiny"}, "s1")
        assert store.list_artifacts() == []


# ---------------------------------------------------------------------------
# AUTO mode externalizes large fields
# ---------------------------------------------------------------------------


class TestAutoModeExternalizesLarge:
    """AUTO mode externalizes large fields."""

    def test_large_content_replaced(self):
        store = _make_store(max_inline_tokens=10)
        svc = _make_ext_service(
            store=store,
            mode=OutputMode.AUTO,
            rules=[ExternalizationRule(tool_name="t", field_path="output")],
        )
        # 10 tokens * 4 chars = 40 chars threshold; use > 40
        big = "x" * 50
        response = {"output": big}
        result, exts = svc.externalize("t", response, "s1")
        assert len(exts) == 1
        assert result["output"] != big

    def test_artifact_stored_for_large(self):
        store = _make_store(max_inline_tokens=10)
        svc = _make_ext_service(
            store=store,
            mode=OutputMode.AUTO,
            rules=[ExternalizationRule(tool_name="t", field_path="output")],
        )
        svc.externalize("t", {"output": "x" * 50}, "s1")
        assert len(store.list_artifacts("s1")) == 1


# ---------------------------------------------------------------------------
# FILE mode always externalizes matching rules
# ---------------------------------------------------------------------------


class TestFileModeAlwaysExternalizes:
    """FILE mode always externalizes matching rules."""

    def test_small_content_externalized(self):
        store = _make_store()
        svc = _make_ext_service(
            store=store,
            mode=OutputMode.FILE,
            rules=[ExternalizationRule(tool_name="t", field_path="output")],
        )
        response = {"output": "tiny"}
        result, exts = svc.externalize("t", response, "s1")
        assert len(exts) == 1
        assert "externalized" in result["output"].lower() or "artifact" in result["output"].lower()

    def test_unmatched_tool_not_externalized(self):
        store = _make_store()
        svc = _make_ext_service(
            store=store,
            mode=OutputMode.FILE,
            rules=[ExternalizationRule(tool_name="other_tool", field_path="output")],
        )
        response = {"output": "data"}
        result, exts = svc.externalize("my_tool", response, "s1")
        assert result["output"] == "data"
        assert exts == []

    def test_missing_field_skipped(self):
        store = _make_store()
        svc = _make_ext_service(
            store=store,
            mode=OutputMode.FILE,
            rules=[ExternalizationRule(tool_name="t", field_path="nonexistent")],
        )
        response = {"other": "data"}
        result, exts = svc.externalize("t", response, "s1")
        assert result == {"other": "data"}
        assert exts == []


# ---------------------------------------------------------------------------
# ExternalizationResult with correct token savings
# ---------------------------------------------------------------------------


class TestExternalizationResult:
    """Returns ExternalizationResult with correct token savings."""

    def test_result_type(self):
        store = _make_store()
        svc = _make_ext_service(
            store=store,
            mode=OutputMode.FILE,
            rules=[ExternalizationRule(tool_name="t", field_path="output")],
        )
        _, exts = svc.externalize("t", {"output": "x" * 4000}, "s1")
        assert len(exts) == 1
        assert isinstance(exts[0], ExternalizationResult)

    def test_original_token_estimate(self):
        store = _make_store()
        svc = _make_ext_service(
            store=store,
            mode=OutputMode.FILE,
            rules=[ExternalizationRule(tool_name="t", field_path="output")],
        )
        content = "x" * 400  # 400 / 4 = 100 tokens
        _, exts = svc.externalize("t", {"output": content}, "s1")
        assert exts[0].original_token_estimate == 100

    def test_saved_tokens_positive_for_large_content(self):
        store = _make_store()
        svc = _make_ext_service(
            store=store,
            mode=OutputMode.FILE,
            rules=[ExternalizationRule(tool_name="t", field_path="output")],
        )
        _, exts = svc.externalize("t", {"output": "x" * 4000}, "s1")
        assert exts[0].saved_tokens > 0

    def test_saved_tokens_never_negative(self):
        store = _make_store()
        svc = _make_ext_service(
            store=store,
            mode=OutputMode.FILE,
            rules=[ExternalizationRule(tool_name="t", field_path="output")],
        )
        # Very short content -- summary may be longer
        _, exts = svc.externalize("t", {"output": "x"}, "s1")
        assert exts[0].saved_tokens >= 0

    def test_artifact_ref_present(self):
        store = _make_store()
        svc = _make_ext_service(
            store=store,
            mode=OutputMode.FILE,
            rules=[ExternalizationRule(tool_name="t", field_path="output")],
        )
        _, exts = svc.externalize("t", {"output": "data"}, "s1")
        assert exts[0].artifact_ref is not None
        assert exts[0].artifact_ref.artifact_id.startswith("art_")

    def test_summary_contains_artifact_id(self):
        store = _make_store()
        svc = _make_ext_service(
            store=store,
            mode=OutputMode.FILE,
            rules=[ExternalizationRule(tool_name="t", field_path="output")],
        )
        _, exts = svc.externalize("t", {"output": "data"}, "s1")
        assert exts[0].artifact_ref.artifact_id in exts[0].summary


# ---------------------------------------------------------------------------
# Nested field path access
# ---------------------------------------------------------------------------


class TestNestedFieldPath:
    """Nested field path access."""

    def test_single_level_nested(self):
        store = _make_store()
        svc = _make_ext_service(
            store=store,
            mode=OutputMode.FILE,
            rules=[ExternalizationRule(tool_name="t", field_path="data.content")],
        )
        response = {"data": {"content": "nested value"}}
        result, exts = svc.externalize("t", response, "s1")
        assert len(exts) == 1
        # The nested field should be replaced with the summary
        assert isinstance(result["data"]["content"], str)
        assert result["data"]["content"] != "nested value"

    def test_deeply_nested_path(self):
        store = _make_store()
        svc = _make_ext_service(
            store=store,
            mode=OutputMode.FILE,
            rules=[ExternalizationRule(tool_name="t", field_path="a.b.c")],
        )
        response = {"a": {"b": {"c": "deep"}}}
        result, exts = svc.externalize("t", response, "s1")
        assert len(exts) == 1

    def test_non_dict_intermediate_skipped(self):
        store = _make_store()
        svc = _make_ext_service(
            store=store,
            mode=OutputMode.FILE,
            rules=[ExternalizationRule(tool_name="t", field_path="a.b")],
        )
        response = {"a": "not a dict"}
        result, exts = svc.externalize("t", response, "s1")
        assert exts == []


# ===========================================================================
# ArtifactRetrievalService
# ===========================================================================


# ---------------------------------------------------------------------------
# fetch returns content with pagination
# ---------------------------------------------------------------------------


class TestRetrievalFetch:
    """fetch returns content with pagination."""

    def _setup(self):
        store = _make_store()
        art = store.create_artifact("hello world content", "tool", "field", "s1")
        svc = ArtifactRetrievalService(store)
        return store, art, svc

    def test_fetch_success(self):
        _, art, svc = self._setup()
        result = svc.fetch(str(art.id))
        assert result["success"] is True
        assert result["content"] == "hello world content"

    def test_fetch_total_size(self):
        _, art, svc = self._setup()
        result = svc.fetch(str(art.id))
        assert result["total_size"] == len("hello world content")

    def test_fetch_with_offset(self):
        _, art, svc = self._setup()
        result = svc.fetch(str(art.id), offset=6, limit=5)
        assert result["success"] is True
        assert result["content"] == "world"

    def test_fetch_with_limit(self):
        _, art, svc = self._setup()
        result = svc.fetch(str(art.id), offset=0, limit=5)
        assert result["success"] is True
        assert result["content"] == "hello"
        assert result["has_more"] is True

    def test_fetch_full_no_more(self):
        _, art, svc = self._setup()
        result = svc.fetch(str(art.id), offset=0, limit=10_000)
        assert result["success"] is True
        assert result["has_more"] is False


# ---------------------------------------------------------------------------
# fetch returns error for missing artifact
# ---------------------------------------------------------------------------


class TestRetrievalFetchMissing:
    """fetch returns error for missing artifact."""

    def test_nonexistent_id(self):
        store = _make_store()
        svc = ArtifactRetrievalService(store)
        result = svc.fetch("nonexistent_id")
        assert result["success"] is False
        assert "error" in result
        assert "not found" in result["error"].lower() or "expired" in result["error"].lower()

    def test_hint_present(self):
        store = _make_store()
        svc = ArtifactRetrievalService(store)
        result = svc.fetch("nonexistent_id")
        assert "hint" in result

    def test_expired_artifact(self):
        store = _make_store(retention_ttl_seconds=0)
        art = store.create_artifact("data", "t", "f", "s")
        art.created_at = datetime.now() - timedelta(seconds=10)
        svc = ArtifactRetrievalService(store)
        result = svc.fetch(str(art.id))
        assert result["success"] is False


# ---------------------------------------------------------------------------
# list_artifacts returns dicts
# ---------------------------------------------------------------------------


class TestRetrievalListArtifacts:
    """list_artifacts returns dicts."""

    def test_returns_list_of_dicts(self):
        store = _make_store()
        store.create_artifact("a", "tool1", "field1", "s1")
        svc = ArtifactRetrievalService(store)
        arts = svc.list_artifacts()
        assert len(arts) == 1
        assert isinstance(arts[0], dict)

    def test_dict_has_expected_keys(self):
        store = _make_store()
        store.create_artifact("a", "tool1", "field1", "s1")
        svc = ArtifactRetrievalService(store)
        arts = svc.list_artifacts()
        assert "id" in arts[0]
        assert "tool" in arts[0]
        assert "field" in arts[0]
        assert "session" in arts[0]

    def test_list_filters_by_session(self):
        store = _make_store()
        store.create_artifact("a", "t", "f", "s1")
        store.create_artifact("b", "t", "f", "s1")
        store.create_artifact("c", "t", "f", "s2")
        svc = ArtifactRetrievalService(store)
        arts = svc.list_artifacts("s1")
        assert len(arts) == 2

    def test_list_empty_returns_empty(self):
        store = _make_store()
        svc = ArtifactRetrievalService(store)
        assert svc.list_artifacts() == []

    def test_list_all_no_filter(self):
        store = _make_store()
        store.create_artifact("a", "t1", "f1", "s1")
        store.create_artifact("b", "t2", "f2", "s2")
        svc = ArtifactRetrievalService(store)
        arts = svc.list_artifacts()
        assert len(arts) == 2


# ===========================================================================
# Template selection based on ROBOTMCP_FETCH_ARTIFACT env var
# ===========================================================================


class TestTemplateSelection:
    """Tests that service selects correct summary template based on env var."""

    def test_default_uses_file_path_template(self):
        """Without ROBOTMCP_FETCH_ARTIFACT, summaries should NOT mention fetch_artifact."""
        import os
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROBOTMCP_FETCH_ARTIFACT", None)
            store = _make_store()
            svc = ArtifactExternalizationService(
                store=store, output_mode=OutputMode.FILE,
            )
            response = {"output": "x" * 500}
            rules = [ExternalizationRule(tool_name="execute_step", field_path="output")]
            svc = ArtifactExternalizationService(
                store=store, output_mode=OutputMode.FILE, rules=rules,
            )
            result, results = svc.externalize("execute_step", response, "s1")
            assert len(results) == 1
            assert "Content saved to" in result["output"]
            assert "fetch_artifact" not in result["output"]

    def test_fetch_artifact_enabled_uses_fetch_template(self):
        """With ROBOTMCP_FETCH_ARTIFACT=true, summaries SHOULD mention fetch_artifact."""
        import os
        with patch.dict(os.environ, {"ROBOTMCP_FETCH_ARTIFACT": "true"}):
            store = _make_store()
            rules = [ExternalizationRule(tool_name="execute_step", field_path="output")]
            svc = ArtifactExternalizationService(
                store=store, output_mode=OutputMode.FILE, rules=rules,
            )
            response = {"output": "x" * 500}
            result, results = svc.externalize("execute_step", response, "s1")
            assert len(results) == 1
            assert "fetch_artifact" in result["output"]

    def test_summary_contains_file_path(self):
        """File path summary should contain actual artifact file path."""
        import os
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ROBOTMCP_FETCH_ARTIFACT", None)
            store = _make_store()
            rules = [ExternalizationRule(tool_name="execute_step", field_path="output")]
            svc = ArtifactExternalizationService(
                store=store, output_mode=OutputMode.FILE, rules=rules,
            )
            response = {"output": "x" * 500}
            result, results = svc.externalize("execute_step", response, "s1")
            assert ".robotmcp_artifacts" in result["output"] or "/tmp" in result["output"]
            assert ".txt" in result["output"]

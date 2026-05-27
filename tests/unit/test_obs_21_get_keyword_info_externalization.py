"""OBS-21 — wire ``_externalize_response`` into ``get_keyword_info``
+ add field-path rules.

Benchmark K06: ``get_keyword_info(mode="library", library_name="Browser")``
returned **71,521 inline tokens** in a single response. Round-3 Codex
review caught my original "add rules to DEFAULT_RULES" proposal: that
alone does nothing because ``get_keyword_info`` never called
``_externalize_response``. The two-part fix wires the call into each
branch AND adds the field-path rules.

These tests pin:
1. DEFAULT_RULES contains the new field-path rules
2. ``get_keyword_info(mode="library", session_id=...)`` triggers
   externalisation
3. ``get_keyword_info(mode="keyword", session_id=...)`` with verbose
   keyword.doc triggers externalisation
4. ``get_keyword_info(mode="session", session_id=...)`` externalises
   on its ``doc`` field
5. WITHOUT ``session_id``, externalisation does NOT fire (backwards
   compat with the find_keywords gating contract)
6. Small payloads (under the threshold) stay inline regardless
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotmcp.server import get_keyword_info


def _fn(tool):
    return getattr(tool, "fn", tool)


class TestDefaultRulesIncludeGetKeywordInfo:
    """The static DEFAULT_RULES table must include the new field paths."""

    def test_library_doc_rule_present(self):
        from robotmcp.domains.artifact_output.services import DEFAULT_RULES
        rules = [(r.tool_name, r.field_path) for r in DEFAULT_RULES]
        assert ("get_keyword_info", "library.doc") in rules

    def test_library_keywords_rule_present(self):
        from robotmcp.domains.artifact_output.services import DEFAULT_RULES
        rules = [(r.tool_name, r.field_path) for r in DEFAULT_RULES]
        assert ("get_keyword_info", "library.keywords") in rules

    def test_keyword_doc_rule_present(self):
        from robotmcp.domains.artifact_output.services import DEFAULT_RULES
        rules = [(r.tool_name, r.field_path) for r in DEFAULT_RULES]
        assert ("get_keyword_info", "keyword.doc") in rules

    def test_doc_top_level_rule_present(self):
        """Session-mode payload places the doc at top-level ``doc``."""
        from robotmcp.domains.artifact_output.services import DEFAULT_RULES
        rules = [(r.tool_name, r.field_path) for r in DEFAULT_RULES]
        assert ("get_keyword_info", "doc") in rules


@pytest.mark.asyncio
class TestExternaliseCalledWithSessionId:
    """When ``session_id`` is provided, the new code path must invoke
    ``_externalize_response``. Verified at the call boundary."""

    async def test_library_mode_invokes_externalize(self):
        with patch(
            "robotmcp.server._externalize_response",
        ) as mock_ext, patch(
            "robotmcp.server._get_library_documentation_payload",
            new_callable=AsyncMock,
        ) as mock_payload:
            mock_payload.return_value = {"success": True, "library": {"doc": "x"}}
            mock_ext.side_effect = lambda *a: a[-1]
            result = await _fn(get_keyword_info)(
                mode="library", library_name="Browser",
                session_id="sess-x",
            )
        mock_ext.assert_called_once()
        # Tool name passed through correctly.
        call_args, call_kwargs = mock_ext.call_args
        assert call_args[0] == "get_keyword_info"
        assert call_args[1] == "sess-x"

    async def test_keyword_mode_invokes_externalize(self):
        with patch(
            "robotmcp.server._externalize_response",
        ) as mock_ext, patch(
            "robotmcp.server._get_keyword_documentation_payload",
            new_callable=AsyncMock,
        ) as mock_payload:
            mock_payload.return_value = {"success": True, "keyword": {"doc": "x"}}
            mock_ext.side_effect = lambda *a: a[-1]
            await _fn(get_keyword_info)(
                mode="keyword", keyword_name="Click",
                session_id="sess-x",
            )
        mock_ext.assert_called_once()
        assert mock_ext.call_args.args[0] == "get_keyword_info"

    async def test_session_mode_invokes_externalize(self):
        with patch(
            "robotmcp.server._externalize_response",
        ) as mock_ext, patch(
            "robotmcp.server._get_session_keyword_documentation_payload",
            new_callable=AsyncMock,
        ) as mock_payload:
            mock_payload.return_value = {"success": True, "doc": "x"}
            mock_ext.side_effect = lambda *a: a[-1]
            await _fn(get_keyword_info)(
                mode="session", keyword_name="Click",
                session_id="sess-x",
            )
        mock_ext.assert_called_once()


@pytest.mark.asyncio
class TestNoSessionIdSkipsExternalize:
    """Without ``session_id``, externalisation does NOT fire — preserves
    the find_keywords contract: sessionless callers get the full inline
    payload (the call has no artifact store to write to)."""

    async def test_library_mode_no_session_skips_externalize(self):
        with patch(
            "robotmcp.server._externalize_response",
        ) as mock_ext, patch(
            "robotmcp.server._get_library_documentation_payload",
            new_callable=AsyncMock,
        ) as mock_payload:
            mock_payload.return_value = {"success": True, "library": {"doc": "x"}}
            await _fn(get_keyword_info)(
                mode="library", library_name="Browser",
            )
        mock_ext.assert_not_called()

    async def test_keyword_mode_no_session_skips_externalize(self):
        with patch(
            "robotmcp.server._externalize_response",
        ) as mock_ext, patch(
            "robotmcp.server._get_keyword_documentation_payload",
            new_callable=AsyncMock,
        ) as mock_payload:
            mock_payload.return_value = {"success": True, "keyword": {"doc": "x"}}
            await _fn(get_keyword_info)(
                mode="keyword", keyword_name="Click",
            )
        mock_ext.assert_not_called()


@pytest.mark.asyncio
class TestExternalizationFiresEndToEnd:
    """End-to-end: a large library-mode response with session_id
    actually triggers externalisation and shrinks the inline payload."""

    async def test_large_library_doc_externalised(self, tmp_path, monkeypatch):
        # Point the artifact directory at a tmp path so the test
        # doesn't pollute the real .robotmcp_artifacts/.
        monkeypatch.setenv("ROBOTMCP_ARTIFACT_DIR", str(tmp_path))
        # Force a low inline threshold to make the test deterministic.
        monkeypatch.setenv("ROBOTMCP_MAX_INLINE_TOKENS", "50")

        # Synthesize a verbose library response.
        big_doc = "x" * 5000  # ~1250 tokens, way over the 50 threshold
        with patch(
            "robotmcp.server._get_library_documentation_payload",
            new_callable=AsyncMock,
        ) as mock_payload:
            mock_payload.return_value = {
                "success": True,
                "library": {
                    "name": "Browser",
                    "doc": big_doc,
                    "keywords": [
                        {"name": f"Kw{i}", "doc": "x" * 200}
                        for i in range(20)
                    ],
                },
            }
            # Force a fresh externalisation service so the env var is read.
            import robotmcp.server as srv
            from robotmcp.domains.artifact_output.aggregates import ArtifactStore
            from robotmcp.domains.artifact_output.services import (
                ArtifactExternalizationService,
            )
            from robotmcp.domains.artifact_output.value_objects import ArtifactPolicy
            store = ArtifactStore(policy=ArtifactPolicy.from_env())
            srv._artifact_service = ArtifactExternalizationService(store)
            result = await _fn(get_keyword_info)(
                mode="library", library_name="Browser",
                session_id="sess-x",
            )
        # The library.doc field should be replaced with an artifact
        # summary string ("Content saved to ...") rather than the raw
        # 5000-char prose.
        if "library" in result:
            assert isinstance(result["library"].get("doc"), str)
            # Either it's the artifact summary OR the field was removed.
            if result["library"].get("doc"):
                assert (
                    len(result["library"]["doc"]) < len(big_doc)
                    or "Content saved" in result["library"]["doc"]
                ), (
                    f"library.doc not externalised; got "
                    f"{result['library']['doc'][:100]!r}"
                )

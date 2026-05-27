"""OBS-30 — honest semantic-strategy docstring + optional extra +
embedding state reported at INFO level.

The ``find_keywords`` tool advertised ``strategy="semantic"`` as
"Natural language search (best for exploring)". In default deployments
without ``sentence-transformers`` installed (and it's NOT in
``pyproject.toml`` dependencies), the matcher uses pattern + tag +
``difflib.SequenceMatcher`` ranking — no embedding similarity. The
docstring over-promised.

Fix:
1. Docstring updated to honestly describe the hybrid + optional-extra
   nature of the semantic strategy.
2. ``pyproject.toml`` declares ``[semantic]`` optional dependency
   group with ``sentence-transformers`` + ``scipy``.
3. ``KeywordMatcher.__init__`` logs at INFO whether embedding mode
   is active or fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest


class TestDocstringHonesty:
    """The strategy docstring must mention embedding similarity is
    optional + how to enable it."""

    def test_docstring_mentions_optional_embeddings(self):
        from robotmcp.server import find_keywords
        fn = getattr(find_keywords, "fn", find_keywords)
        doc = (fn.__doc__ or "").lower()
        # The old narrow phrasing was "natural language search (best for exploring)"
        # — that's now part of a fuller description that includes the
        # optional extra.
        assert "sentence-transformers" in doc, (
            f"docstring should mention sentence-transformers optional dep; "
            f"got: {doc[:600]}"
        )
        assert "robotmcp[semantic]" in doc, (
            "docstring should mention the [semantic] extra name"
        )

    def test_docstring_describes_fallback_behaviour(self):
        from robotmcp.server import find_keywords
        fn = getattr(find_keywords, "fn", find_keywords)
        doc = (fn.__doc__ or "").lower()
        # The fallback must be described — not just "Natural language search".
        assert (
            "difflib" in doc
            or "sequencematcher" in doc
            or "pattern + tag" in doc
        ), (
            f"docstring should describe the fallback ranking; got: {doc[:600]}"
        )


class TestPyprojectExtra:
    """The ``[semantic]`` optional dependency group must exist in
    pyproject.toml so users can install ``sentence-transformers``
    via ``uv add robotmcp[semantic]``."""

    def test_semantic_extra_declared(self):
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
        assert "semantic = [" in pyproject or "semantic=[" in pyproject, (
            "pyproject.toml should declare a [semantic] optional-dependency group"
        )
        # The group must include sentence-transformers.
        semantic_block = pyproject.split("semantic = [", 1)[1].split("]", 1)[0]
        assert "sentence-transformers" in semantic_block, (
            f"[semantic] extra should include sentence-transformers; "
            f"block: {semantic_block!r}"
        )


class TestMatcherLogsEmbeddingMode:
    """Matcher __init__ must log the embedding mode at INFO level so
    operators can tell whether semantic ranking is using embeddings
    or the fallback."""

    def test_logs_fallback_when_embeddings_unavailable(self, caplog):
        """In an environment without sentence-transformers (the
        current default), __init__ logs the DISABLED message."""
        caplog.set_level(logging.INFO, logger="robotmcp")
        from robotmcp.components.keyword_matcher import KeywordMatcher
        KeywordMatcher()
        log_text = "\n".join(r.getMessage() for r in caplog.records).lower()
        # Either "ACTIVE" (embeddings installed) or "DISABLED" (fallback)
        # must appear — the exact one depends on the test environment,
        # but the operator-visible signal must surface either way.
        assert (
            "embedding similarity active" in log_text
            or "embedding similarity disabled" in log_text
        ), f"expected embedding-mode log line; got: {log_text}"

    def test_fallback_log_mentions_install_path(self, caplog):
        """When the fallback path triggers, the log line must tell
        operators how to enable embeddings."""
        from robotmcp.components.keyword_matcher import (
            EMBEDDINGS_AVAILABLE, KeywordMatcher,
        )
        if EMBEDDINGS_AVAILABLE:
            pytest.skip(
                "sentence-transformers IS installed in this env; "
                "fallback-log test does not apply"
            )
        caplog.set_level(logging.INFO, logger="robotmcp")
        KeywordMatcher()
        log_text = "\n".join(r.getMessage() for r in caplog.records).lower()
        assert "robotmcp[semantic]" in log_text, (
            f"fallback log should mention the install command; got: {log_text}"
        )

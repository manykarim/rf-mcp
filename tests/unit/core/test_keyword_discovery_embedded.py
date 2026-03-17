"""Tests for embedded argument matching in KeywordDiscovery (ADR-019 Phase 2)."""
import pytest

from robotmcp.core.keyword_discovery import KeywordDiscovery
from robotmcp.models.library_models import KeywordInfo, LibraryInfo


def _make_lib_info(name: str, keywords: dict[str, KeywordInfo]) -> LibraryInfo:
    """Create a minimal LibraryInfo for testing."""
    lib = LibraryInfo(name=name, instance=None)
    lib.keywords = keywords
    return lib


def _make_kw(name: str, library: str) -> KeywordInfo:
    return KeywordInfo(
        name=name,
        library=library,
        method_name=name.lower().replace(" ", "_").replace("${", "").replace("}", ""),
    )


class TestKeywordDiscoveryEmbedded:
    """Tests for embedded keyword detection and matching in KeywordDiscovery."""

    def test_add_embedded_keyword_to_cache(self):
        kd = KeywordDiscovery()
        kw = _make_kw("Click ${element}", "TestLib")
        lib = _make_lib_info("TestLib", {"Click ${element}": kw})
        kd.add_keywords_to_cache(lib)

        assert len(kd.embedded_keywords) == 1
        pattern, info = kd.embedded_keywords[0]
        assert pattern.template_name == "Click ${element}"
        assert info.library == "TestLib"

    def test_find_keyword_embedded_match(self):
        kd = KeywordDiscovery()
        kw = _make_kw("Click ${element}", "TestLib")
        lib = _make_lib_info("TestLib", {"Click ${element}": kw})
        kd.add_keywords_to_cache(lib)

        result = kd.find_keyword("Click login button")
        assert result is not None
        assert result.name == "Click ${element}"
        assert result.library == "TestLib"

    def test_find_keyword_exact_before_embedded(self):
        """Exact match should take priority over embedded match."""
        kd = KeywordDiscovery()
        exact_kw = _make_kw("Click login button", "ExactLib")
        embedded_kw = _make_kw("Click ${element}", "EmbeddedLib")
        lib1 = _make_lib_info("ExactLib", {"Click login button": exact_kw})
        lib2 = _make_lib_info("EmbeddedLib", {"Click ${element}": embedded_kw})
        kd.add_keywords_to_cache(lib1)
        kd.add_keywords_to_cache(lib2)

        result = kd.find_keyword("Click login button")
        assert result is not None
        # Exact match from ExactLib should win
        assert result.library == "ExactLib"

    def test_remove_keywords_clears_embedded(self):
        kd = KeywordDiscovery()
        kw = _make_kw("Click ${element}", "TestLib")
        lib = _make_lib_info("TestLib", {"Click ${element}": kw})
        kd.add_keywords_to_cache(lib)
        assert len(kd.embedded_keywords) == 1

        kd.remove_keywords_from_cache(lib)
        assert len(kd.embedded_keywords) == 0

    def test_non_embedded_keyword_not_added(self):
        kd = KeywordDiscovery()
        kw = _make_kw("Click Button", "TestLib")
        lib = _make_lib_info("TestLib", {"Click Button": kw})
        kd.add_keywords_to_cache(lib)
        assert len(kd.embedded_keywords) == 0

    def test_embedded_no_match_returns_none(self):
        kd = KeywordDiscovery()
        kw = _make_kw("Set ${var} to ${val}", "TestLib")
        lib = _make_lib_info("TestLib", {"Set ${var} to ${val}": kw})
        kd.add_keywords_to_cache(lib)

        result = kd.find_keyword("totally unrelated keyword name xyz")
        assert result is None

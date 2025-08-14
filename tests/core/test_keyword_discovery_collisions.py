import os
import sys

import pytest

# Ensure src path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from robotmcp.core.keyword_discovery import KeywordDiscovery
from robotmcp.models.library_models import LibraryInfo, KeywordInfo


def _make_library(name: str) -> LibraryInfo:
    lib = LibraryInfo(name=name, instance=object())
    kw = KeywordInfo(name="Shared Keyword", library=name, method_name="kw")
    lib.keywords["Shared Keyword"] = kw
    return lib


def test_cross_library_keyword_collisions():
    discovery = KeywordDiscovery()
    discovery.add_keywords_to_cache(_make_library("Lib1"))
    discovery.add_keywords_to_cache(_make_library("Lib2"))

    # Expecting ambiguity error when keyword name is shared, but first library wins
    with pytest.raises(ValueError):
        discovery.find_keyword("Shared Keyword")

    # Fully qualified name should still resolve
    kw = discovery.find_keyword("Lib1.Shared Keyword")
    assert kw.library == "Lib1"

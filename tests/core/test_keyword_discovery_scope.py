import os
import sys

# Ensure src path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

import pytest

from robotmcp.core.dynamic_keyword_orchestrator import DynamicKeywordDiscovery
from robotmcp.core.keyword_discovery import KeywordDiscovery
from robotmcp.core.library_manager import LibraryManager
from robotmcp.components.execution.session_manager import SessionManager
from robotmcp.models.library_models import LibraryInfo, KeywordInfo


def _make_library(name: str) -> LibraryInfo:
    lib = LibraryInfo(name=name, instance=object())
    kw = KeywordInfo(name="Common", library=name, method_name="kw")
    lib.keywords["Common"] = kw
    return lib


def test_global_vs_session_keyword_discovery():
    orchestrator = DynamicKeywordDiscovery()
    # Replace library manager and keyword discovery with fresh ones to avoid interference
    orchestrator.library_manager = LibraryManager()
    orchestrator.keyword_discovery = KeywordDiscovery()

    lib1 = _make_library("LibA")
    lib2 = _make_library("LibB")

    orchestrator.library_manager.libraries["LibA"] = lib1
    orchestrator.library_manager.libraries["LibB"] = lib2
    orchestrator.keyword_discovery.add_keywords_to_cache(lib1)
    orchestrator.keyword_discovery.add_keywords_to_cache(lib2)

    # Global discovery should raise ambiguity
    with pytest.raises(ValueError):
        orchestrator.keyword_discovery.find_keyword("Common")

    # Session-specific discovery should respect search order
    session_manager = SessionManager()
    orchestrator.session_manager = session_manager
    session = session_manager.create_session("s1")
    session.search_order = ["LibA"]

    kw = orchestrator._find_keyword_with_session("Common", session_id="s1")
    assert kw.library == "LibA"

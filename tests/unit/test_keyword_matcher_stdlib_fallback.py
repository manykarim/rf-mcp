"""Regression: the fallback keyword loader must not crash on RF standard
libraries. `robot.libraries.STDLIBS` is a frozenset of names, not a
name->module map, so the old `STDLIBS[library_name]` subscript raised
`'frozenset' object is not subscriptable` for any stdlib routed through the
fallback (surfaced as a stderr warning for `Dialogs`). Found via stderr
forensics during the docker uv-tool-install experiments.
"""
import asyncio

import pytest

from robotmcp.components.keyword_matcher import STDLIBS, KeywordMatcher


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_stdlibs_is_a_frozenset_of_names():
    # Guards the assumption the fix relies on.
    assert "Dialogs" in STDLIBS
    with pytest.raises(TypeError):
        STDLIBS["Dialogs"]  # subscripting a frozenset is the original bug


@pytest.mark.parametrize("lib", ["Dialogs", "String", "Collections"])
def test_stdlib_fallback_does_not_crash(lib, caplog):
    km = KeywordMatcher()
    _run(km._load_library_keywords_fallback(lib))
    # registry gets an entry (never a subscript TypeError), and no frozenset warning
    assert lib in km.keyword_registry
    assert "not subscriptable" not in caplog.text

import pytest

from robotmcp.plugins.builtin.selenium_plugin import SeleniumLibraryPlugin
from robotmcp.plugins.manager import LibraryPluginManager


def test_selenium_plugin_keyword_map_contains_input_password():
    mgr = LibraryPluginManager()
    mgr.register_plugin(SeleniumLibraryPlugin(), source="test")

    assert mgr.get_library_for_keyword("Input Password") == "SeleniumLibrary"
    assert mgr.get_keyword_override("SeleniumLibrary", "Input Password") is not None


def test_selenium_plugin_keyword_map_case_insensitive():
    mgr = LibraryPluginManager()
    mgr.register_plugin(SeleniumLibraryPlugin(), source="test")

    assert mgr.get_library_for_keyword("input password") == "SeleniumLibrary"
    assert mgr.get_library_for_keyword("INPUT PASSWORD") == "SeleniumLibrary"

import pytest

from robotmcp.plugins.manager import LibraryPluginManager
from robotmcp.plugins.builtin.browser_plugin import BrowserLibraryPlugin
from robotmcp.plugins.builtin.selenium_plugin import SeleniumLibraryPlugin


@pytest.fixture
def plugin_manager():
    mgr = LibraryPluginManager()
    mgr.register_plugin(BrowserLibraryPlugin(), source="test")
    mgr.register_plugin(SeleniumLibraryPlugin(), source="test")
    return mgr


def test_browser_locator_normalizer_passthrough(plugin_manager):
    norm = plugin_manager.get_locator_normalizer("Browser")
    assert norm is not None
    locator = "css=.login >> text=Docs"
    assert norm(locator) == locator


def test_selenium_locator_normalizer_passthrough(plugin_manager):
    norm = plugin_manager.get_locator_normalizer("SeleniumLibrary")
    assert norm is not None
    locator = "xpath=//div[@id='main']"
    assert norm(locator) == locator


def test_basic_validator_non_empty(plugin_manager):
    validate = plugin_manager.get_locator_validator("Browser")
    result = validate("css=#id") if validate else {}
    assert result.get("valid") is True
    assert result.get("warnings") == []


def test_basic_validator_empty_locator(plugin_manager):
    validate = plugin_manager.get_locator_validator("SeleniumLibrary")
    result = validate("") if validate else {}
    assert result.get("valid") is False
    assert "Empty locator" in result.get("warnings", [])
